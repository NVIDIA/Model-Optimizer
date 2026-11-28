# Architecture reference: https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/1bae0e1088993b9b82de08f59de5be5a34e42170/inference/model.py
# Copyright (c) 2023 DeepSeek. MIT License.

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 AND MIT
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

_INDEX_SCORE_ELEMENTS = 8 * 1024 * 1024


def _compression_layout(positions, token_mask, ratio):
    """Group chronological valid tokens, independent of physical padding offsets."""
    count = positions.shape[1] // ratio
    order = torch.argsort((~token_mask).int(), dim=-1, stable=True)
    grouped = order[:, : count * ratio].unflatten(-1, (count, ratio))
    starts = positions.gather(1, grouped[:, :, 0])
    lengths = token_mask.long().cumsum(-1).unsqueeze(-1) // ratio
    valid = torch.arange(count, device=positions.device) < (
        token_mask.sum(-1, keepdim=True) // ratio
    )
    return grouped.flatten(1), starts, lengths, valid


class V41Compressor(nn.Module):
    def __init__(self, config, ratio):
        super().__init__()
        self.ratio = ratio
        self.wkv = V41Linear(config.hidden_size, config.head_dim, fp32=ratio > 1)
        self.wgate = (
            V41Linear(config.hidden_size, config.head_dim, fp32=True) if ratio > 1 else None
        )
        self.norm = V41RMSNorm(config.head_dim, config.rms_norm_eps)

    def forward(self, hidden_states, order):
        hidden_states = hidden_states.gather(
            1, order.unsqueeze(-1).expand(-1, -1, hidden_states.shape[-1])
        )
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

    def forward(self, hidden_states, query_latent, latent, positions, ratio, offset, state, layout):
        count = hidden_states.shape[1] // ratio
        _, compressed_positions, lengths, compressed_valid = layout
        if self.owns_keys:
            key = self.k_norm(self.wk(latent)).to(hidden_states.dtype)
            key = rotary(key, compressed_positions, self.config, True)
            state["index_key"] = (fp4_inputs(key) if self.config.attention_quantization else key,)
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
        source = self.layer_id == self.config.candidate_source_layer_id
        consuming = 0 <= self.config.candidate_source_layer_id < self.layer_id
        if consuming and "candidates" not in state:
            raise ValueError("Candidate source must run before a candidate-filtered indexer.")
        candidates = state["candidates"][0] if consuming else None
        if source:
            state["candidates"] = (
                torch.empty(
                    hidden_states.shape[0],
                    hidden_states.shape[1],
                    count,
                    dtype=torch.bool,
                    device=hidden_states.device,
                ),
            )
        key_count = (
            min(count, self.config.candidate_topk_blocks * self.config.candidate_block_size)
            if consuming
            else count
        )
        # Bound both the per-head product and the global-coordinate score tile.
        chunk = max(
            1,
            _INDEX_SCORE_ELEMENTS
            // (hidden_states.shape[0] * max(count, self.config.index_n_heads * key_count)),
        )
        result = torch.empty(
            *hidden_states.shape[:2],
            min(self.config.index_topk, count),
            device=hidden_states.device,
            dtype=torch.long,
        )
        # Only discrete top-k indices escape this block; no score gradient is
        # consumed. Keeping a per-tile autograd graph would retain the full S²H
        # activation set despite bounded forward allocations.
        with torch.no_grad(), torch.autocast(query.device.type, enabled=False):
            key = state["index_key"][0].float()
            for start in range(0, hidden_states.shape[1], chunk):
                stop = min(start + chunk, hidden_states.shape[1])
                if candidates is not None:
                    candidate_mask = candidates[:, start:stop]
                    selected = (
                        candidate_mask.to(torch.uint8).topk(key_count, -1, sorted=False).indices
                    )
                    selected_keys = key.gather(
                        1, selected.flatten(1).unsqueeze(-1).expand(-1, -1, key.shape[-1])
                    )
                    selected_keys = selected_keys.unflatten(1, (stop - start, key_count))
                    products = torch.einsum(
                        "bshd,bstd->bsht", query[:, start:stop].float(), selected_keys
                    )
                else:
                    products = torch.einsum("bshd,btd->bsht", query[:, start:stop].float(), key)
                scores = (products.relu() * weights[:, start:stop].unsqueeze(-1)).sum(2)
                if candidates is not None:
                    scores = scores.masked_fill(~candidate_mask.gather(-1, selected), -torch.inf)
                    # Preserve global top-k coordinates, including tie behavior,
                    # without constructing per-head products for excluded keys.
                    scores = scores.new_full((*scores.shape[:2], count), -torch.inf).scatter_(
                        -1, selected, scores
                    )
                scores.masked_fill_(
                    torch.arange(count, device=scores.device) >= lengths[:, start:stop], -torch.inf
                )
                scores.masked_fill_(~compressed_valid[:, None, :], -torch.inf)
                if source:
                    state["candidates"][0][:, start:stop] = select_candidates(
                        scores,
                        lengths[:, start:stop],
                        self.config.candidate_topk_blocks,
                        self.config.candidate_block_size,
                    )
                indices = scores.topk(result.shape[-1], -1, sorted=False).indices.sort(-1).values
                result[:, start:stop] = torch.where(
                    (indices < lengths[:, start:stop])
                    & ~torch.isneginf(scores.gather(-1, indices)),
                    indices + offset,
                    -1,
                )
        return result


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
            mask = (
                torch.ones_like(positions, dtype=torch.bool) if token_mask is None else token_mask
            )
            layout = _compression_layout(positions, mask, self.compress_ratio)
            order, compressed_positions, _, compressed_valid = layout
            latent = None
            if self.is_kv_source:
                assert self.compressor is not None
                latent = self.compressor(hidden_states, order)
            if self.is_index_source:
                assert self.indexer is not None
                state["topk"] = (
                    self.indexer(
                        hidden_states,
                        qr,
                        latent,
                        positions,
                        self.compress_ratio,
                        sequence,
                        state,
                        layout,
                    ),
                )
            if self.is_kv_source:
                assert latent is not None
                compressed = rotary(
                    latent,
                    compressed_positions,
                    self.config,
                    True,
                )
                state["compressed_kv"] = (
                    fp4_inputs(compressed, 16, True)
                    if self.config.attention_quantization
                    else compressed,
                )
            kv = torch.cat((kv, state["compressed_kv"][0]), 1)
            indices = torch.cat((indices, state["topk"][0]), -1)
            if token_mask is not None and state["topk"][0].shape[-1]:
                all_valid = torch.cat((token_mask, compressed_valid), -1)
                selected_valid = all_valid.gather(1, indices.clamp_min(0).flatten(1)).reshape_as(
                    indices
                )
                indices = indices.masked_fill(~selected_valid, -1)
        output = sparse_attention(query, kv, self.attn_sink, indices)
        output = rotary(output, positions, self.config, bool(self.compress_ratio), inverse=True)
        output = output.reshape(batch, sequence, self.config.o_groups, -1)
        if self.config.attention_quantization or getattr(self.wo_a, "checkpoint_quantized", False):
            output = _fp8_inputs(output)
        output = self.wo_a(output)
        return self.wo_b(output.flatten(2)).to(hidden_states.dtype)
