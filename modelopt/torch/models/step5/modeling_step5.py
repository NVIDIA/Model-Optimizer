# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Differentiable Step-5 text implementation with bounded CSA attention tiles.

Architecture reference: stepfun-ai/Step-5-Preview revision
364b3b03fed2477aa0953fe855a2c4c8c48ee3cc, inference/model.py and kernel.py.
The discrete indexer preserves the published BF16/FP8 rounding; main-network
projections and attention use ordinary trainable Torch operations.
"""

import math
import re
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PreTrainedModel
from transformers.cache_utils import DynamicCache
from transformers.generation import GenerationMixin
from transformers.integrations import use_experts_implementation
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

from .configuration_step5 import Step5Config

__all__ = ["Step5ForCausalLM", "Step5Model", "Step5PreTrainedModel"]


class _FP32Parameters:
    def _apply(self, fn, recurse=True):
        result = super()._apply(fn, recurse=recurse)
        for parameter in self.parameters(recurse=False):
            if parameter.is_floating_point():
                parameter.data = parameter.data.float()
        for name, buffer in self.named_buffers(recurse=False):
            if buffer.is_floating_point():
                setattr(self, name, buffer.float())
        return result


class Step5RMSNorm(_FP32Parameters, nn.Module):
    """Zero-centered RMSNorm, with FP32 reduction and additive FP32 scale."""

    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(hidden_size, dtype=torch.float32))
        self.variance_epsilon = eps

    def forward(self, hidden, *, output_dtype=None):
        """Normalize the unrounded residual and cast only the final result."""
        raw = hidden.float()
        normed = raw * torch.rsqrt(raw.square().mean(-1, keepdim=True) + self.variance_epsilon)
        return (normed * (1 + self.weight.float())).to(output_dtype or hidden.dtype)


class _IndexerRMSNorm(Step5RMSNorm):
    def forward(self, hidden):
        raw = hidden.float()
        scale = self.weight.to(hidden.dtype).float() + 1
        return (
            raw * torch.rsqrt(raw.square().mean(-1, keepdim=True) + self.variance_epsilon) * scale
        ).to(hidden.dtype)


class _IndexerLayerNorm(_FP32Parameters, nn.LayerNorm):
    def forward(self, hidden):
        raw = hidden.float()
        mean = raw.mean(-1, keepdim=True)
        variance = raw.square().mean(-1, keepdim=True) - mean.square()
        result = (raw - mean) * torch.rsqrt(variance + self.eps)
        return (
            result * self.weight.to(hidden.dtype).float() + self.bias.to(hidden.dtype).float()
        ).to(hidden.dtype)


class _IndexerWeightProjection(_FP32Parameters, nn.Linear):
    def forward(self, hidden):
        return F.linear(hidden, self.weight.to(hidden.dtype)).float()


class _RouterLinear(_FP32Parameters, nn.Linear):
    def forward(self, hidden):
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            return F.linear(hidden.float(), self.weight.float())


class Step5RotaryEmbedding(nn.Module):
    """Half-split partial RoPE with the published BF16 table rounding boundary."""

    def __init__(self, config, layer_idx, span):
        super().__init__()
        if span < 0 or span % 2:
            raise ValueError("Step-5 rotary span must be nonnegative and even")
        self.span = span
        self.theta = (
            config.rope_theta[layer_idx]
            if isinstance(config.rope_theta, list)
            else config.rope_theta
        )
        kind = config.layer_types[layer_idx]
        self.scaling = config.rope_parameters[kind] if kind in config.yarn_only_types else None

    def forward(self, hidden, positions):
        """Rotate the leading subspace using FP32 arithmetic and rounded tables."""
        if self.span == 0:
            return hidden
        inv = 1.0 / self.theta ** (
            torch.arange(0, self.span, 2, device=hidden.device, dtype=torch.float32) / self.span
        )
        if self.scaling and self.scaling.get("rope_type", "default") != "default":
            cfg = self.scaling
            wavelength = 2 * math.pi / inv
            original, low, high = (
                cfg["original_max_position_embeddings"],
                cfg["low_freq_factor"],
                cfg["high_freq_factor"],
            )
            smooth = (original / wavelength - low) / (high - low)
            scaled = torch.where(wavelength > original / low, inv / cfg["factor"], inv)
            inv = torch.where(
                (wavelength >= original / high) & (wavelength <= original / low),
                (1 - smooth) * inv / cfg["factor"] + smooth * inv,
                scaled,
            )
        angles = positions.float().unsqueeze(-1) * inv
        dtype = torch.bfloat16 if hidden.dtype == torch.bfloat16 else torch.float32
        cos, sin = (
            angles.cos().to(dtype).float().unsqueeze(-2),
            angles.sin().to(dtype).float().unsqueeze(-2),
        )
        half = self.span // 2
        a, b = hidden[..., :half].float(), hidden[..., half : self.span].float()
        return torch.cat(
            (
                (a * cos - b * sin).to(hidden.dtype),
                (a * sin + b * cos).to(hidden.dtype),
                hidden[..., self.span :],
            ),
            -1,
        )


def clamped_swiglu(gate, up, limit):
    """Apply SiLU and asymmetric clamps in FP32, then round the final product."""
    gate, up_float = F.silu(gate.float()), up.float()
    if limit > 0:
        gate, up_float = gate.clamp(max=limit), up_float.clamp(-limit, limit)
    return (gate * up_float).to(up.dtype)


class Step5MLP(nn.Module):
    """Dense or shared expert with the published clamped SwiGLU operation."""

    def __init__(self, config, width, limit):
        super().__init__()
        self.limit = limit
        self.gate_proj = nn.Linear(config.hidden_size, width, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, width, bias=False)
        self.down_proj = nn.Linear(width, config.hidden_size, bias=False)

    def forward(self, hidden):
        """Evaluate one feed-forward branch."""
        return self.down_proj(
            clamped_swiglu(self.gate_proj(hidden), self.up_proj(hidden), self.limit)
        )


@use_experts_implementation
class Step5Experts(nn.Module):
    """Fused expert weights compatible with native HF EP and ModelOpt PTQ."""

    _fp8_experts_clamp_after_activation = True

    def __init__(self, config, limit):
        super().__init__()
        self.num_experts = config.moe_num_experts
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.moe_intermediate_size
        self.limit = limit
        self.act_fn = F.silu
        self.gate_up_proj = nn.Parameter(
            torch.empty(self.num_experts, 2 * self.intermediate_dim, self.hidden_dim)
        )
        self.down_proj = nn.Parameter(
            torch.empty(self.num_experts, self.hidden_dim, self.intermediate_dim)
        )

    def _apply_gate(self, gate_up):
        gate, up = gate_up.chunk(2, -1)
        return clamped_swiglu(gate, up, self.limit)

    def forward(self, hidden_states, top_k_index, top_k_weights):
        """Accumulate BF16 expert values in FP32 in top-k slot order."""
        slots = hidden_states.new_zeros(
            hidden_states.shape[0], top_k_index.shape[-1], self.hidden_dim
        )
        for expert in range(self.num_experts):
            rows, positions = torch.where(top_k_index == expert)
            values = F.linear(hidden_states[rows], self.gate_up_proj[expert])
            values = F.linear(self._apply_gate(values), self.down_proj[expert])
            slots[rows, positions] = values
        result = torch.zeros_like(hidden_states, dtype=torch.float32)
        for slot in range(top_k_index.shape[-1]):
            result = result + slots[:, slot].float() * top_k_weights[:, slot, None]
        return result.to(hidden_states.dtype)


class Step5MoE(_FP32Parameters, nn.Module):
    """FP32 sigmoid router with a selection-only correction bias."""

    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.gate = _RouterLinear(
            config.hidden_size, config.moe_num_experts, bias=False, dtype=torch.float32
        )
        self.register_buffer(
            "router_bias", torch.zeros(config.moe_num_experts, dtype=torch.float32)
        )
        self.experts = Step5Experts(config, config.swiglu_limits[layer_idx])

    def forward(self, hidden):
        """Return the routed contribution; the decoder adds its sibling shared expert."""
        shape = hidden.shape
        hidden = hidden.reshape(-1, shape[-1])
        scores = self.gate(hidden).sigmoid()
        indices = (scores + self.router_bias.float()).topk(self.config.moe_top_k, -1).indices
        weights = scores.gather(-1, indices)
        if self.config.norm_expert_weight:
            weights = weights / (weights.sum(-1, keepdim=True) + 1e-20)
        weights = weights * self.config.moe_router_scaling_factor
        return self.experts(hidden, indices, weights).reshape(shape)


@dataclass
class _RegionLayout:
    positions: torch.Tensor
    valid: torch.Tensor
    documents: torch.Tensor
    token_regions: torch.Tensor
    sources: torch.Tensor
    region_documents: torch.Tensor


def _region_layout(positions, valid, block):
    starts = (positions.remainder(block) == 0) & valid
    token_regions = starts.long().cumsum(-1) - 1
    # The dynamic layout is built once per forward, shared by every CSA layer.
    regions = max(1, int(starts.sum(-1).max()))
    batch, length = positions.shape
    physical = torch.arange(length, device=positions.device).expand(batch, -1)
    slots = token_regions.clamp_min(0) * block + positions.remainder(block)
    sources = positions.new_full((batch, regions * block), -1)
    sources.scatter_reduce_(
        1, slots, physical.masked_fill(~valid, -1), reduce="amax", include_self=True
    )
    sources = sources.view(batch, regions, block)
    documents = ((positions == 0) & valid).long().cumsum(-1)
    region_documents = documents.gather(1, sources[:, :, 0].clamp_min(0))
    return _RegionLayout(positions, valid, documents, token_regions, sources, region_documents)


class Step5SparseIndexer(_FP32Parameters, nn.Module):
    """Discrete CSA region selector, independent of the differentiable K/V path."""

    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config.sparse_config
        dim, heads = self.config.proxy_dim, self.config.sparse_indexer_num_heads
        self.q_proj = nn.Linear(config.hidden_size, heads * dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, dim, bias=False)
        self.z_proj = nn.Linear(config.hidden_size, dim, bias=False)
        self.w_proj = _IndexerWeightProjection(
            config.hidden_size, heads, bias=False, dtype=torch.float32
        )
        self.q_norm = _IndexerRMSNorm(dim, config.rms_norm_eps)
        self.k_norm = _IndexerLayerNorm(dim, eps=config.rms_norm_eps, dtype=torch.float32)
        self.register_buffer(
            "ssmax_s", torch.zeros(config.num_attention_heads, dtype=torch.float32)
        )
        span = self.config.sparse_indexer_rope_dim if self.config.sparse_indexer_use_rope else 0
        self.rotary_emb = Step5RotaryEmbedding(config, layer_idx, span)
        self.requires_grad_(False)

    @torch.no_grad()
    def prepare(self, hidden, positions, layout, cached=None):
        """Compress each position-aligned region with per-dimension softmax weights."""
        keys = self.rotary_emb(self.k_norm(self.k_proj(hidden)).unsqueeze(-2), positions).squeeze(
            -2
        )
        logits = self.z_proj(hidden)
        if cached is not None:
            keys, logits = torch.cat((cached[0], keys), 1), torch.cat((cached[1], logits), 1)
        source = layout.sources.clamp_min(0)
        rows = torch.arange(hidden.shape[0], device=hidden.device)[:, None, None]
        packed_keys = keys[rows, source].float()
        packed_z = (
            logits[rows, source].float().masked_fill((layout.sources < 0).unsqueeze(-1), -torch.inf)
        )
        shift = packed_z.amax((-1, -2), keepdim=True)
        weights = (packed_z - shift).exp().nan_to_num(0.0)
        summaries = (weights * packed_keys).sum(2) / weights.sum(2).clamp_min(1e-20)
        summaries = summaries.to(torch.bfloat16).to(torch.float8_e4m3fn).to(torch.bfloat16)
        return (hidden.detach(), positions, layout, summaries), (keys, logits)

    @torch.no_grad()
    def select(self, state, start, stop):
        """Select one bounded query tile, including the current causal region."""
        hidden, positions, layout, summaries = state
        cfg = self.config
        groups, dim = cfg.num_provider_groups, cfg.proxy_dim
        heads = cfg.sparse_indexer_num_heads // groups
        local = hidden[:, start:stop]
        query = self.q_norm(self.q_proj(local).unflatten(-1, (groups * heads, dim)))
        query = self.rotary_emb(query, positions[:, start:stop])
        query = query.to(torch.float8_e4m3fn).to(torch.bfloat16).unflatten(-2, (groups, heads))
        weights = self.w_proj(local).unflatten(-1, (groups, heads)) * heads**-0.5
        scores = torch.zeros(
            hidden.shape[0], groups, stop - start, summaries.shape[1], device=hidden.device
        )
        for head in range(heads):
            dot = torch.einsum("bqgd,brd->bgqr", query[:, :, :, head].float(), summaries.float())
            scores = scores + dot.relu() * weights[:, :, :, head].transpose(1, 2).unsqueeze(-1)
        offset = layout.positions.shape[1] - hidden.shape[1]
        current = layout.token_regions[:, offset + start : offset + stop]
        docs = layout.documents[:, offset + start : offset + stop]
        region_range = torch.arange(summaries.shape[1], device=hidden.device)
        candidates = (region_range < current.unsqueeze(-1)) & (
            layout.region_documents[:, None] == docs.unsqueeze(-1)
        )
        scores.masked_fill_(~candidates[:, None], -torch.inf)
        budget = min(cfg.topk, summaries.shape[1])
        # Stable order implements the published ascending-region tie break.
        selected = scores.argsort(dim=-1, descending=True, stable=True)[..., :budget]
        selected = selected.masked_fill(
            ~candidates[:, None].expand_as(scores).gather(-1, selected), summaries.shape[1]
        )
        selected = selected.sort(dim=-1).values
        selected.masked_fill_(selected == summaries.shape[1], -1)
        selected = torch.cat((selected, current[:, None, :, None].expand(-1, groups, -1, -1)), -1)
        rows = torch.arange(hidden.shape[0], device=hidden.device)[:, None, None, None]
        tokens = layout.sources[rows, selected.clamp_min(0)]
        tokens.masked_fill_(selected.unsqueeze(-1) < 0, -1)
        return tokens.flatten(-2)

    def forward(self, hidden, positions, valid):
        """Return selected physical token indices for small standalone indexer calls."""
        layout = _region_layout(positions, valid, self.config.region_block_size)
        state, _ = self.prepare(hidden, positions, layout)
        return self.select(state, 0, hidden.shape[1])


class Step5Cache(DynamicCache):
    """HF dynamic K/V cache plus the small CSA index projections and token metadata."""

    def __init__(self):
        super().__init__()
        self.positions = None
        self.valid = None
        self.indexer_states = {}

    def reorder_cache(self, beam_idx):
        """Reorder K/V, indexer projections, and layout metadata together."""
        super().reorder_cache(beam_idx)
        self._map_metadata(lambda value: value.index_select(0, beam_idx.to(value.device)))

    def crop(self, tokens_to_remove):
        """Crop CSA projections and layout metadata to the resulting K/V length."""
        super().crop(tokens_to_remove)
        length = self.get_seq_length()
        if length == 0:
            self._clear_metadata()
        else:
            self._map_metadata(lambda value: value[:, :length])

    def batch_repeat_interleave(self, repeats):
        """Repeat K/V and all CSA side state along the batch dimension."""
        super().batch_repeat_interleave(repeats)
        self._map_metadata(lambda value: value.repeat_interleave(repeats, dim=0))

    def batch_select_indices(self, indices):
        """Select matching batch rows from K/V and every CSA side tensor."""
        super().batch_select_indices(indices)
        self._map_metadata(lambda value: value.index_select(0, indices.to(value.device)))

    def reset(self):
        """Reset K/V and discard CSA projections and token metadata."""
        super().reset()
        self._clear_metadata()

    def _clear_metadata(self):
        self.positions = self.valid = None
        self.indexer_states = {}

    def _map_metadata(self, transform):
        if self.positions is not None:
            self.positions, self.valid = transform(self.positions), transform(self.valid)
        self.indexer_states = {
            layer: (transform(key), transform(z)) for layer, (key, z) in self.indexer_states.items()
        }


class Step5Attention(nn.Module):
    """Gated GQA using SWA or CSA with bounded differentiable attention tiles."""

    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_heads, self.groups, self.head_dim = (
            config.num_attention_heads,
            config.num_attention_groups,
            config.head_dim,
        )
        self.heads_per_group = self.num_heads // self.groups
        self.window = (
            config.sliding_window if config.layer_types[layer_idx] == "sliding_attention" else None
        )
        self.qkv_proj = nn.Linear(
            config.hidden_size, (self.num_heads + 2 * self.groups) * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)
        self.g_proj = (
            nn.Linear(config.hidden_size, self.num_heads, bias=False)
            if config.use_head_wise_attn_gate
            else None
        )
        self.q_norm = Step5RMSNorm(self.head_dim, config.rms_norm_eps)
        self.k_norm = Step5RMSNorm(self.head_dim, config.rms_norm_eps)
        span = round(self.head_dim * config.partial_rotary_factors[layer_idx])
        self.rotary_emb = Step5RotaryEmbedding(config, layer_idx, span - span % 2)
        sparse = (
            config.sparse_config.enabled
            and config.layer_types[layer_idx] in config.sparse_config.apply_to_layer_types
        )
        self.indexer = Step5SparseIndexer(config, layer_idx) if sparse else None

    def forward(self, hidden, positions, layout, past_key_values=None):
        """Attend selected keys with gradients through Q, K, V and both projections."""
        batch, sequence = hidden.shape[:2]
        dim, groups, heads = self.head_dim, self.groups, self.heads_per_group
        projected = self.qkv_proj(hidden)
        q, k, v = projected.split((self.num_heads * dim, groups * dim, groups * dim), -1)
        query = self.rotary_emb(self.q_norm(q.unflatten(-1, (self.num_heads, dim))), positions)
        key = self.rotary_emb(self.k_norm(k.unflatten(-1, (groups, dim))), positions).transpose(
            1, 2
        )
        value = v.unflatten(-1, (groups, dim)).transpose(1, 2)
        if past_key_values is not None:
            key, value = past_key_values.update(key, value, self.layer_idx)
        total = key.shape[-2]
        offset = total - sequence
        state = None
        if self.indexer is not None:
            cached = (
                None
                if past_key_values is None
                else past_key_values.indexer_states.get(self.layer_idx)
            )
            state, updated = self.indexer.prepare(hidden, positions, layout, cached)
            if past_key_values is not None:
                past_key_values.indexer_states[self.layer_idx] = updated
        outputs = []
        rows = torch.arange(batch, device=hidden.device)[:, None, None, None]
        group_ids = torch.arange(groups, device=hidden.device)[None, :, None, None]
        for start in range(0, sequence, 16):
            stop = min(start + 16, sequence)
            count = stop - start
            if state is not None:
                indices = self.indexer.select(state, start, stop)
            elif self.window is not None:
                width = min(self.window, total)
                physical = torch.arange(offset + start, offset + stop, device=hidden.device)
                indices = physical[:, None] + torch.arange(1 - width, 1, device=hidden.device)
                indices = indices.expand(batch, groups, -1, -1)
            else:
                indices = torch.arange(total, device=hidden.device).expand(batch, groups, count, -1)
            safe = indices.clamp(0, total - 1)
            selected_k, selected_v = key[rows, group_ids, safe], value[rows, group_ids, safe]
            key_positions = layout.positions[rows, safe]
            key_docs = layout.documents[rows, safe]
            distance = positions[:, None, start:stop, None] - key_positions
            query_docs = layout.documents[:, None, offset + start : offset + stop, None]
            allowed = (
                (indices >= 0)
                & (distance >= 0)
                & layout.valid[rows, safe]
                & (key_docs == query_docs)
            )
            allowed = allowed & layout.valid[:, None, offset + start : offset + stop, None]
            if self.window is not None:
                allowed = allowed & (distance < self.window)
            local_q = query[:, start:stop].unflatten(-2, (groups, heads)).permute(0, 2, 1, 3, 4)
            local_q = local_q.reshape(batch * groups * count, heads, 1, dim)
            selected_k = selected_k.reshape(batch * groups * count, 1, -1, dim)
            selected_v = selected_v.reshape(batch * groups * count, 1, -1, dim)
            attended = F.scaled_dot_product_attention(
                local_q,
                selected_k,
                selected_v,
                attn_mask=allowed.reshape(batch * groups * count, 1, 1, -1),
                enable_gqa=True,
            )
            outputs.append(
                attended.view(batch, groups, count, heads, dim).permute(0, 2, 1, 3, 4).flatten(2, 3)
            )
        attended = torch.cat(outputs, 1)
        if self.g_proj is not None:
            attended = attended * self.g_proj(hidden).sigmoid().unsqueeze(-1)
        return self.o_proj(attended.flatten(-2))


class Step5DecoderLayer(GradientCheckpointingLayer):
    """Pre-norm decoder with FP32 residual additions and a routed/shared FFN."""

    def __init__(self, config, layer_idx):
        super().__init__()
        self.self_attn = Step5Attention(config, layer_idx)
        self.input_layernorm = Step5RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = Step5RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.moe = (
            Step5MoE(config, layer_idx)
            if config.use_moe and layer_idx in config.moe_layer_list
            else None
        )
        self.share_expert = (
            Step5MLP(config, config.share_expert_dim, config.swiglu_limits_shared[layer_idx])
            if self.moe is not None
            else None
        )
        self.mlp = (
            Step5MLP(config, config.intermediate_size, config.swiglu_limits[layer_idx])
            if self.moe is None
            else None
        )

    def forward(self, hidden_states, positions, layout, compute_dtype, past_key_values=None):
        """Keep residual precision across both branches of the decoder."""
        normed = self.input_layernorm(hidden_states, output_dtype=compute_dtype)
        hidden_states = (
            hidden_states + self.self_attn(normed, positions, layout, past_key_values).float()
        )
        normed = self.post_attention_layernorm(hidden_states, output_dtype=compute_dtype)
        if self.moe is None:
            return hidden_states + self.mlp(normed).float()
        return hidden_states + (self.moe(normed).float() + self.share_expert(normed).float())


class Step5PreTrainedModel(PreTrainedModel):
    """HF model contract for the Step-5 text architecture."""

    config_class = Step5Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Step5DecoderLayer"]
    _supports_sdpa = True
    _supports_attention_backend = False
    _keep_in_fp32_modules = [
        "input_layernorm",
        "post_attention_layernorm",
        "norm",
        "q_norm",
        "k_norm",
        "w_proj",
        "moe.gate",
    ]
    _keep_in_fp32_modules_strict = [*_keep_in_fp32_modules, "router_bias", "ssmax_s"]

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=self.config.initializer_range)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, Step5Experts):
            nn.init.normal_(module.gate_up_proj, std=self.config.initializer_range)
            nn.init.normal_(module.down_proj, std=self.config.initializer_range)
        elif isinstance(module, Step5RMSNorm):
            nn.init.zeros_(module.weight)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)


class Step5Model(Step5PreTrainedModel):
    """Complete Step-5 text decoder; image/video and MTP modules are excluded."""

    def __init__(self, config):
        """Build every text decoder layer using ordinary trainable Torch weights."""
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            Step5DecoderLayer(config, i) for i in range(config.num_hidden_layers)
        )
        self.norm = Step5RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_hidden_states=None,
        return_dict=True,
        **kwargs,
    ):
        """Return normalized text states, optionally recording a differentiable K/V cache."""
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Supply exactly one of input_ids or inputs_embeds")
        if any(
            kwargs.get(name) is not None
            for name in ("pixel_values", "images", "videos", "image_mask")
        ):
            raise ValueError("Step5Model supports text inputs only")
        if attention_mask is not None and attention_mask.ndim != 2:
            raise ValueError("Step-5 expects a 2D token-validity attention_mask")
        use_cache = self.config.use_cache if use_cache is None else use_cache
        checkpointing = getattr(self, "is_activation_checkpointing", self.is_gradient_checkpointing)
        if self.training and checkpointing and use_cache:
            raise ValueError("Step-5 caching cannot be combined with gradient checkpointing")
        if past_key_values is not None and not isinstance(past_key_values, Step5Cache):
            raise TypeError("Step-5 requires its own cache for CSA indexer metadata")
        cache = (
            past_key_values
            if past_key_values is not None
            else (Step5Cache() if use_cache else None)
        )
        embedded = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        batch, length = embedded.shape[:2]
        valid = (
            torch.ones((batch, length), dtype=torch.bool, device=embedded.device)
            if attention_mask is None
            else attention_mask[:, -length:].bool()
        )
        if position_ids is None:
            offset = 0 if cache is None or cache.positions is None else cache.positions[:, -1:] + 1
            position_ids = (offset + valid.long().cumsum(-1) - 1).clamp_min(0)
        else:
            if position_ids.ndim != 2 or position_ids.shape[-1] != length:
                raise ValueError("Step-5 position_ids must have shape [batch, sequence]")
            position_ids = position_ids.expand(batch, -1)
        all_positions, all_valid = position_ids, valid
        if cache is not None and cache.positions is not None:
            all_positions = torch.cat((cache.positions, position_ids), 1)
            all_valid = torch.cat((cache.valid, valid), 1)
        if cache is not None:
            cache.positions, cache.valid = all_positions, all_valid
        layout = _region_layout(
            all_positions, all_valid, self.config.sparse_config.region_block_size
        )
        hidden = embedded.float()
        history = () if output_hidden_states else None
        for layer in self.layers:
            if output_hidden_states:
                history += (hidden,)
            hidden = layer(hidden, position_ids, layout, embedded.dtype, past_key_values=cache)
        hidden = self.norm(hidden, output_dtype=embedded.dtype)
        if output_hidden_states:
            history += (hidden,)
        output = BaseModelOutputWithPast(
            last_hidden_state=hidden, past_key_values=cache, hidden_states=history
        )
        return output if return_dict else output.to_tuple()


class Step5ForCausalLM(Step5PreTrainedModel, GenerationMixin):
    """Differentiable HF causal language model for the Step-5 Preview text backbone."""

    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config):
        """Build a complete text decoder and its untied vocabulary projection."""
        super().__init__(config)
        self.model = Step5Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    @classmethod
    def _supports_default_dynamic_cache(cls):
        return False

    def load_published_state_dict(self, state_dict, *, strict=True, assign=False):
        """Load supplied floating text tensors in the original published naming/layout.

        FP8 weights and scale tensors require a checkpoint quantization loader;
        this entry point deliberately accepts the published BF16 variant only.
        No files are downloaded, and no tensors are moved implicitly.
        """
        converted = {}
        index_names = {part: part + "_proj" for part in ("q", "k", "z", "w")}
        index_names.update({"q_norm": "q_norm", "k_norm": "k_norm"})
        for name, tensor in state_dict.items():
            if name.endswith("_scale_inv") or tensor.dtype not in (
                torch.float16,
                torch.bfloat16,
                torch.float32,
                torch.float64,
            ):
                raise ValueError("Step-5 published tensor loading requires the BF16 checkpoint")
            for source, target in index_names.items():
                name = re.sub(
                    r"\.self_attn\.sparse_indexer_" + source + r"\.",
                    ".self_attn.indexer." + target + ".",
                    name,
                )
            name = name.replace(".self_attn.ssmax_s", ".self_attn.indexer.ssmax_s")
            name = name.replace(".moe.down_proj.weight", ".moe.experts.down_proj")
            if name in converted:
                raise ValueError(f"Duplicate Step-5 checkpoint mapping for {name!r}.")
            converted[name] = tensor
        for name in list(converted):
            if name.endswith(".self_attn.q_proj.weight"):
                prefix = name.removesuffix("q_proj.weight")
                if prefix + "qkv_proj.weight" in converted:
                    raise ValueError("Step-5 checkpoint mixes split and fused attention weights.")
                if any(prefix + part + "_proj.weight" not in converted for part in ("q", "k", "v")):
                    raise ValueError("Step-5 checkpoint requires all Q/K/V projections.")
                converted[prefix + "qkv_proj.weight"] = torch.cat(
                    [converted.pop(prefix + part + "_proj.weight") for part in ("q", "k", "v")], 0
                )
            elif name.endswith(".moe.gate_proj.weight"):
                prefix = name.removesuffix("gate_proj.weight")
                if prefix + "experts.gate_up_proj" in converted:
                    raise ValueError("Step-5 checkpoint mixes split and fused expert weights.")
                if prefix + "up_proj.weight" not in converted:
                    raise ValueError("Step-5 checkpoint requires both gate and up projections.")
                converted[prefix + "experts.gate_up_proj"] = torch.cat(
                    [converted.pop(prefix + part + "_proj.weight") for part in ("gate", "up")], 1
                )
        expected = self.state_dict()
        if strict and expected.keys() != converted.keys():
            missing, extra = expected.keys() - converted.keys(), converted.keys() - expected.keys()
            raise ValueError(
                f"Step-5 checkpoint keys differ: missing={sorted(missing)}, extra={sorted(extra)}"
            )
        for name, tensor in converted.items():
            if name in expected and tensor.shape != expected[name].shape:
                raise ValueError(f"Step-5 checkpoint shape mismatch for {name}")
        return self.load_state_dict(converted, strict=strict, assign=assign)

    def get_input_embeddings(self):
        """Return the token embedding module."""
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        """Replace token embeddings using the HF contract."""
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        """Return the language projection module."""
        return self.lm_head

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        logits_to_keep=0,
        output_hidden_states=None,
        return_dict=True,
        **kwargs,
    ):
        """Evaluate text logits and document-aware shifted causal loss."""
        if labels is not None and not (isinstance(logits_to_keep, int) and logits_to_keep == 0):
            raise ValueError("Step-5 labels require logits_to_keep=0")
        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            **kwargs,
        )
        indices = (
            slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        )
        logits = self.lm_head(output.last_hidden_state[:, indices])
        loss = None
        if labels is not None:
            shifted_labels = labels[:, 1:].clone()
            if attention_mask is not None:
                shifted_labels.masked_fill_(
                    ~(attention_mask[:, 1:].bool() & attention_mask[:, :-1].bool()), -100
                )
            if position_ids is not None:
                shifted_labels.masked_fill_(position_ids[:, 1:] == 0, -100)
            loss = F.cross_entropy(
                logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                shifted_labels.reshape(-1),
                reduction="sum",
            ) / (shifted_labels != -100).sum().clamp_min(1)
        result = CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=output.past_key_values,
            hidden_states=output.hidden_states,
        )
        return result if return_dict else result.to_tuple()
