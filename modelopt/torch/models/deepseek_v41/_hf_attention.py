# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Architecture reference: https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/1bae0e1088993b9b82de08f59de5be5a34e42170/inference/model.py
# Copyright (c) 2023 DeepSeek. MIT License.
# SPDX-License-Identifier: Apache-2.0 AND MIT

"""CSA2 sliding windows, compressed latents, and shared sparse index ownership."""

import torch
from torch import nn

from ._hf_math import (
    V41GroupedProjection,
    V41Linear,
    V41RMSNorm,
    fp4_inputs,
    rotary,
    select_candidates,
    sparse_attention,
)
from .moe import _fp8_inputs

__all__ = []


class V41Compressor(nn.Module):
    def __init__(self, config, ratio):
        super().__init__()
        self.ratio = ratio
        self.wkv = V41Linear(config.hidden_size, config.head_dim, fp32=ratio > 1)
        self.wgate = (
            V41Linear(config.hidden_size, config.head_dim, fp32=True) if ratio > 1 else None
        )
        self.norm = V41RMSNorm(config.head_dim, config.rms_norm_eps)

    def forward(self, hidden_states):
        if self.ratio == 1:
            return self.norm(self.wkv(hidden_states)).to(hidden_states.dtype)
        cutoff = hidden_states.shape[1] // self.ratio * self.ratio
        value = hidden_states[:, :cutoff].float()
        latent = self.wkv(value).float().unflatten(1, (-1, self.ratio))
        assert self.wgate is not None
        gates = self.wgate(value).float().unflatten(1, (-1, self.ratio))
        pooled = (latent * gates.softmax(2)).sum(2)
        return self.norm(pooled.to(hidden_states.dtype))


class V41Indexer(nn.Module):
    def __init__(self, config, layer_id, owns_keys):
        super().__init__()
        self.config, self.layer_id, self.owns_keys = config, layer_id, owns_keys
        self.wq_b = V41Linear(config.q_lora_rank, config.index_n_heads * config.index_head_dim)
        self.weights_proj = V41Linear(config.hidden_size, config.index_n_heads)
        if owns_keys:
            self.wk = V41Linear(config.head_dim, config.index_head_dim)
            self.k_norm = V41RMSNorm(config.index_head_dim, config.rms_norm_eps)

    def forward(
        self, hidden_states, query_latent, latent, positions, ratio, offset, state, token_mask
    ):
        count = hidden_states.shape[1] // ratio
        if self.owns_keys:
            key = self.k_norm(self.wk(latent)).to(hidden_states.dtype)
            key = rotary(key, positions[:, : count * ratio : ratio], self.config, True)
            state["index_key"] = fp4_inputs(key) if self.config.attention_quantization else key
        if not count:
            return torch.empty(
                *hidden_states.shape[:2], 0, dtype=torch.long, device=hidden_states.device
            )
        query = (
            self.wq_b(query_latent)
            .to(hidden_states.dtype)
            .unflatten(-1, (self.config.index_n_heads, self.config.index_head_dim))
        )
        query = rotary(query, positions, self.config, True)
        if self.config.attention_quantization:
            query = fp4_inputs(query)
        weights = (
            self.weights_proj(hidden_states).float()
            / (self.config.index_head_dim * self.config.index_n_heads) ** 0.5
        )
        with torch.autocast(query.device.type, enabled=False):
            scores = torch.einsum("bshd,btd->bsht", query.float(), state["index_key"].float())
        scores = (scores.relu() * weights.unsqueeze(-1)).sum(2)
        lengths = (
            torch.arange(1, hidden_states.shape[1] + 1, device=hidden_states.device) // ratio
        ).unsqueeze(-1)
        scores = scores.masked_fill(
            torch.arange(count, device=scores.device) >= lengths, -torch.inf
        )
        if token_mask is not None:
            compressed_valid = token_mask[:, : count * ratio].unflatten(-1, (-1, ratio)).all(-1)
            scores = scores.masked_fill(~compressed_valid[:, None, :], -torch.inf)
        if self.layer_id == self.config.candidate_source_layer_id:
            state["candidates"] = select_candidates(
                scores, lengths, self.config.candidate_topk_blocks, self.config.candidate_block_size
            )
        elif 0 <= self.config.candidate_source_layer_id < self.layer_id:
            if "candidates" not in state:
                raise ValueError("Candidate source must run before a candidate-filtered indexer.")
            scores = scores.masked_fill(~state["candidates"], -torch.inf)
        indices = (
            scores.topk(min(self.config.index_topk, count), -1, sorted=False)
            .indices.sort(-1)
            .values
        )
        return torch.where(indices < lengths, indices + offset, -1)


class V41Attention(nn.Module):
    def __init__(self, config, layer_id):
        super().__init__()
        self.config, self.layer_id = config, layer_id
        self.compress_ratio = config.compress_ratios[layer_id]
        self.is_kv_source = layer_id in config.kv_source_layer_ids
        self.is_index_source = layer_id in config.index_source_layer_ids
        self.attn_sink = nn.Parameter(torch.zeros(config.num_attention_heads, dtype=torch.float32))
        self.wq_a = V41Linear(config.hidden_size, config.q_lora_rank)
        self.q_norm = V41RMSNorm(config.q_lora_rank, config.rms_norm_eps)
        self.wq_b = V41Linear(config.q_lora_rank, config.num_attention_heads * config.head_dim)
        self.wkv = V41Linear(config.hidden_size, config.head_dim)
        self.kv_norm = V41RMSNorm(config.head_dim, config.rms_norm_eps)
        self.wo_a = V41GroupedProjection(
            config.num_attention_heads * config.head_dim // config.o_groups,
            config.o_groups * config.o_lora_rank,
            config.o_groups,
        )
        self.wo_b = V41Linear(config.o_groups * config.o_lora_rank, config.hidden_size)
        self.compressor = V41Compressor(config, self.compress_ratio) if self.is_kv_source else None
        self.indexer = (
            V41Indexer(config, layer_id, self.is_kv_source) if self.is_index_source else None
        )

    def forward(self, hidden_states, positions, state, token_mask=None):
        batch, sequence, _ = hidden_states.shape
        qr = self.q_norm(self.wq_a(hidden_states)).to(hidden_states.dtype)
        query = (
            self.wq_b(qr)
            .to(hidden_states.dtype)
            .unflatten(-1, (self.config.num_attention_heads, self.config.head_dim))
        )
        query = rotary(query, positions, self.config, bool(self.compress_ratio))
        kv = self.kv_norm(self.wkv(hidden_states)).to(hidden_states.dtype)
        kv = rotary(kv, positions, self.config, bool(self.compress_ratio))
        if self.config.attention_quantization:
            kv = _fp8_inputs(kv)
        end = torch.arange(sequence, device=hidden_states.device).unsqueeze(-1)
        indices = (end - self.config.sliding_window + 1).clamp_min(0) + torch.arange(
            min(sequence, self.config.sliding_window), device=hidden_states.device
        )
        indices = torch.where(indices <= end, indices, -1).unsqueeze(0).expand(batch, -1, -1)
        if token_mask is not None:
            valid = token_mask.gather(1, indices.clamp_min(0).flatten(1)).reshape_as(indices)
            indices = indices.masked_fill(~valid, -1)
        if self.compress_ratio:
            latent = None
            if self.is_kv_source:
                assert self.compressor is not None
                latent = self.compressor(hidden_states)
            if self.is_index_source:
                assert self.indexer is not None
                state["topk"] = self.indexer(
                    hidden_states,
                    qr,
                    latent,
                    positions,
                    self.compress_ratio,
                    sequence,
                    state,
                    token_mask,
                )
            if self.is_kv_source:
                assert latent is not None
                count = latent.shape[1]
                compressed = rotary(
                    latent,
                    positions[:, : count * self.compress_ratio : self.compress_ratio],
                    self.config,
                    True,
                )
                state["compressed_kv"] = (
                    fp4_inputs(compressed, 16, True)
                    if self.config.attention_quantization
                    else compressed
                )
            kv = torch.cat((kv, state["compressed_kv"]), 1)
            indices = torch.cat((indices, state["topk"]), -1)
            if token_mask is not None and state["topk"].shape[-1]:
                count = state["compressed_kv"].shape[1]
                valid = (
                    token_mask[:, : count * self.compress_ratio]
                    .unflatten(-1, (-1, self.compress_ratio))
                    .all(-1)
                )
                all_valid = torch.cat((token_mask, valid), -1)
                selected_valid = all_valid.gather(1, indices.clamp_min(0).flatten(1)).reshape_as(
                    indices
                )
                indices = indices.masked_fill(~selected_valid, -1)
        output = sparse_attention(query, kv, self.attn_sink, indices)
        output = rotary(output, positions, self.config, bool(self.compress_ratio), inverse=True)
        output = output.reshape(batch, sequence, self.config.o_groups, -1)
        if self.config.attention_quantization:
            output = _fp8_inputs(output)
        output = self.wo_a(output)
        return self.wo_b(output.flatten(2)).to(hidden_states.dtype)
