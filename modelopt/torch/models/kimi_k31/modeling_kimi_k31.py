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


"""Ordinary Torch K3.1 text math, without serving kernels or NeMo dependencies.

Architecture follows moonshotai/Kimi-K3's modeling_kimi_linear.py and SGLang's
kimi_k3.py, configs/kimi_linear.py and layers/attention/dsa/dsa_indexer.py.
This reference implementation prioritizes differentiability over long-context speed.
"""

import re
from functools import lru_cache

import torch
import torch.nn.functional as F
from torch import nn
from transformers.cache_utils import DynamicCache
from transformers.generation import GenerationMixin
from transformers.integrations.moe import use_experts_implementation
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.utils import logging

from .configuration_kimi_k31 import KimiK31Config

__all__ = [
    "KimiK31ForCausalLM",
    "KimiK31Model",
    "KimiK31PreTrainedModel",
    "convert_kimi_k31_state_dict",
]

logger = logging.get_logger(__name__)


@lru_cache(maxsize=1)
def _fla_kernels():
    # Defer optional CUDA imports so CPU HF execution needs neither FLA nor Triton.
    try:
        from fla.modules.conv.causal_conv1d import causal_conv1d
        from fla.ops.kda import chunk_kda
    except ModuleNotFoundError as error:
        if error.name == "fla":
            return None
        raise
    return chunk_kda, causal_conv1d


class _ContiguousConvGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        return value

    @staticmethod
    def backward(ctx, grad):
        return grad.contiguous()


def _situ(gate, up, beta, linear_beta):
    dtype = gate.dtype
    gate, up = gate.float(), up.float()
    activated = beta * torch.tanh(gate / beta) * gate.sigmoid()
    if linear_beta is not None:
        up = linear_beta * torch.tanh(up / linear_beta)
    return (activated * up).to(dtype)


class KimiK31RMSNorm(nn.Module):
    def __init__(self, size, eps=1e-5, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size, dtype=dtype))
        self.variance_epsilon = eps

    def forward(self, x):
        value = x.float()
        value = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + self.variance_epsilon)
        return self.weight * value.to(x.dtype)


class KimiK31GatedRMSNorm(KimiK31RMSNorm):
    def __init__(self, size, eps):
        super().__init__(size, eps, dtype=torch.float32)

    def forward(self, x, gate):
        value = x.float()
        value = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + self.variance_epsilon)
        return (value * self.weight.float() * gate.float().sigmoid()).to(x.dtype)


class KimiK31IndexerLayerNorm(nn.LayerNorm):
    def forward(self, x):
        return F.layer_norm(
            x.float(), self.normalized_shape, self.weight.float(), self.bias.float(), self.eps
        )


class KimiK31ShortConvolution(nn.Module):
    def __init__(self, channels, kernel_size):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(channels, 1, kernel_size, dtype=torch.float32))
        self.groups = channels
        self.kernel_size = kernel_size

    def forward(
        self, x, *, kernel=None, cache_params=None, layer_idx=None, state_idx=0, cu_seqlens=None
    ):
        # Own every weight access inside the module's FSDP unshard boundary.
        if kernel is not None and cache_params is None:
            value, _ = kernel(
                x=x,
                weight=self.weight.squeeze(1),
                bias=None,
                initial_state=None,
                output_final_state=False,
                activation="silu",
                backend="triton",
                cu_seqlens=cu_seqlens,
            )
            return _ContiguousConvGradient.apply(value)
        length, dtype = x.shape[1], x.dtype
        x = x.transpose(1, 2)
        if cache_params is not None:
            x = cache_params.update_conv_state(
                x,
                layer_idx,
                state_idx=state_idx,
                conv_kernel_size=self.kernel_size,
            )
        with torch.autocast(device_type=x.device.type, enabled=False):
            value = F.conv1d(
                x.float(),
                self.weight.float(),
                groups=self.groups,
                padding=self.kernel_size - 1,
            )
            value = F.silu(value[..., : x.shape[-1]][..., -length:])
        return value.transpose(1, 2).to(dtype)


class KimiK31DecayGate(nn.Module):
    def __init__(self, num_heads, head_dim, lower_bound):
        super().__init__()
        self.num_heads, self.head_dim = num_heads, head_dim
        self.lower_bound = lower_bound
        self.A_log = nn.Parameter(torch.empty(num_heads, dtype=torch.float32))
        self.dt_bias = nn.Parameter(torch.zeros(num_heads * head_dim, dtype=torch.float32))

    def forward(self, raw):
        gate = raw.float().unflatten(-1, (self.num_heads, self.head_dim))
        gate = gate + self.dt_bias.float().view(self.num_heads, self.head_dim)
        decay = self.A_log.float().exp().view(self.num_heads, 1)
        if self.lower_bound is not None:
            return self.lower_bound * torch.sigmoid(decay * gate)
        return -decay * F.softplus(gate)


def _mix_residual(prefix, blocks, norm, projection):
    values = torch.stack([*blocks, prefix], dim=-2).float()
    keys = values * torch.rsqrt(values.square().mean(-1, keepdim=True) + norm.variance_epsilon)
    score_weight = norm.weight.float() * projection.weight[0].float()
    weights = (keys * score_weight).sum(-1).softmax(-1)
    return (values * weights.unsqueeze(-1)).sum(-2).to(prefix.dtype)


class KimiK31MLP(nn.Module):
    def __init__(self, config, intermediate_size=None):
        super().__init__()
        intermediate = config.intermediate_size if intermediate_size is None else intermediate_size
        self.gate_proj = nn.Linear(config.hidden_size, intermediate, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, config.hidden_size, bias=False)
        self.beta = config.activation_situ_beta
        self.linear_beta = config.activation_situ_linear_beta

    def forward(self, x):
        return self.down_proj(
            _situ(self.gate_proj(x), self.up_proj(x), self.beta, self.linear_beta)
        )


@use_experts_implementation
class KimiK31Experts(nn.Module):
    """Fused expert tensors supporting native HF expert dispatch and ModelOpt PTQ."""

    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.hidden_dim = config.routed_expert_hidden_size
        self.intermediate_dim = config.moe_intermediate_size
        self.gate_up_proj = nn.Parameter(
            torch.empty(self.num_experts, 2 * self.intermediate_dim, self.hidden_dim)
        )
        self.down_proj = nn.Parameter(
            torch.empty(self.num_experts, self.hidden_dim, self.intermediate_dim)
        )
        self.beta = config.activation_situ_beta
        self.linear_beta = config.activation_situ_linear_beta

    def _apply_gate(self, gate_up):
        gate, up = gate_up.chunk(2, dim=-1)
        return _situ(gate, up, self.beta, self.linear_beta)

    def forward(self, hidden_states, top_k_index, top_k_weights):
        output = torch.zeros_like(hidden_states)
        for expert in range(self.num_experts):
            token, slot = torch.where(top_k_index == expert)
            value = F.linear(hidden_states[token], self.gate_up_proj[expert])
            value = F.linear(self._apply_gate(value), self.down_proj[expert])
            value = value * top_k_weights[token, slot, None]
            output.index_add_(0, token, value.to(output.dtype))
        return output


class KimiK31Router(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.weight = nn.Parameter(
            torch.empty(config.num_experts, config.hidden_size, dtype=torch.float32)
        )
        self.e_score_correction_bias = nn.Parameter(
            torch.zeros(config.num_experts, dtype=torch.float32), requires_grad=False
        )

    def forward(self, x):
        config = self.config
        with torch.autocast(device_type=x.device.type, enabled=False):
            logits = F.linear(x.float(), self.weight.float())
        scores = (
            logits.sigmoid()
            if config.moe_router_activation_func == "sigmoid"
            else logits.softmax(-1)
        )
        choice = scores + self.e_score_correction_bias.float()
        if config.num_expert_group > config.topk_group:
            grouped = choice.unflatten(-1, (config.num_expert_group, -1))
            groups = grouped.topk(2, -1).values.sum(-1).topk(config.topk_group, -1).indices
            group_mask = torch.zeros_like(grouped[..., 0], dtype=torch.bool).scatter_(
                -1, groups, True
            )
            choice = choice.masked_fill(
                ~group_mask.unsqueeze(-1).expand_as(grouped).flatten(-2), -torch.inf
            )
        indices = choice.topk(config.num_experts_per_token, dim=-1, sorted=False).indices
        weights = scores.gather(-1, indices)
        if config.moe_renormalize and config.num_experts_per_token > 1:
            weights = weights / (weights.sum(-1, keepdim=True) + 1e-20)
        return indices, weights * config.routed_scaling_factor


class KimiK31MoE(nn.Module):
    """Latent routed SiTU experts plus shared experts in the original hidden space."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.gate = KimiK31Router(config)
        self.experts = KimiK31Experts(config)
        self.routed_expert_down_proj = nn.Linear(
            config.hidden_size, config.routed_expert_hidden_size, bias=False
        )
        self.routed_expert_up_proj = nn.Linear(
            config.routed_expert_hidden_size, config.hidden_size, bias=False
        )
        self.routed_expert_norm = (
            KimiK31RMSNorm(config.routed_expert_hidden_size, config.rms_norm_eps)
            if config.latent_moe_use_norm
            else nn.Identity()
        )
        self.shared_experts = (
            KimiK31MLP(config, config.moe_intermediate_size * config.num_shared_experts)
            if config.num_shared_experts
            else None
        )

    @property
    def router(self):
        return self.gate

    def forward(self, hidden_states):
        x = hidden_states.reshape(-1, hidden_states.shape[-1])
        indices, weights = self.gate(x)
        routed = self.experts(self.routed_expert_down_proj(x), indices, weights)
        output = self.routed_expert_up_proj(self.routed_expert_norm(routed))
        if self.shared_experts is not None:
            output = output + self.shared_experts(x)
        return output.reshape_as(hidden_states)


def _kda_recurrence(q, k, v, g, beta, initial_state=None):
    dtype = q.dtype
    q, k, v = q.float(), k.float(), v.float()
    q = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6) * (q.shape[-1] ** -0.5)
    k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)
    state = (
        q.new_zeros(q.shape[0], q.shape[2], q.shape[3], v.shape[-1])
        if initial_state is None
        else initial_state.float()
    )
    outputs = []
    for i in range(q.shape[1]):
        state = state * g[:, i].float().exp().unsqueeze(-1)
        memory = (state * k[:, i].unsqueeze(-1)).sum(-2)
        delta = (v[:, i] - memory) * beta[:, i].float().unsqueeze(-1)
        state = state + k[:, i].unsqueeze(-1) * delta.unsqueeze(-2)
        outputs.append((state * q[:, i].unsqueeze(-1)).sum(-2))
    return torch.stack(outputs, dim=1).to(dtype), state


class KimiK31DeltaAttention(nn.Module):
    """K3 bounded-decay KDA, including short convolutions and the full-rank output gate."""

    def __init__(self, config, layer_idx):
        super().__init__()
        linear = config.linear_attn_config
        self.layer_idx = layer_idx
        self.implementation = config.kda_implementation
        self.num_heads, self.head_dim = linear["num_heads"], linear["head_dim"]
        self.conv_size = linear["short_conv_kernel_size"]
        self.gate_lower_bound = linear.get("gate_lower_bound")
        self.use_full_rank_gate = linear.get("use_full_rank_gate", False)
        projection = self.num_heads * self.head_dim
        for name in ("q", "k", "v"):
            setattr(self, name + "_proj", nn.Linear(config.hidden_size, projection, bias=False))
            setattr(
                self,
                name + "_conv1d",
                KimiK31ShortConvolution(projection, self.conv_size),
            )
        self.decay_gate = KimiK31DecayGate(self.num_heads, self.head_dim, self.gate_lower_bound)
        self.f_a_proj = nn.Linear(config.hidden_size, self.head_dim, bias=False)
        self.f_b_proj = nn.Linear(self.head_dim, projection, bias=False)
        self.b_proj = nn.Linear(config.hidden_size, self.num_heads, bias=False)
        if self.use_full_rank_gate:
            self.g_proj = nn.Linear(config.hidden_size, projection, bias=False)
        else:
            self.g_a_proj = nn.Linear(config.hidden_size, self.head_dim, bias=False)
            self.g_b_proj = nn.Linear(self.head_dim, projection, bias=False)
        self.o_norm = KimiK31GatedRMSNorm(self.head_dim, config.rms_norm_eps)
        self.o_proj = nn.Linear(projection, config.hidden_size, bias=False)

    @property
    def A_log(self):  # noqa: N802 - published checkpoint attribute
        return self.decay_gate.A_log

    @property
    def dt_bias(self):
        return self.decay_gate.dt_bias

    def _core(self, hidden_states, cache_params=None, cu_seqlens=None):
        shape = (*hidden_states.shape[:2], self.num_heads, self.head_dim)
        kernels = (
            _fla_kernels() if hidden_states.is_cuda and self.implementation != "torch" else None
        )
        if self.implementation == "fla" and kernels is None:
            raise RuntimeError("K3.1 kda_implementation='fla' requires CUDA and FLA.")
        if hidden_states.is_cuda and self.implementation == "auto" and kernels is None:
            logger.warning_once(
                "FLA is unavailable; K3.1 KDA is using the slower Torch recurrence."
            )
        projected = []
        for slot, name in enumerate(("q", "k", "v")):
            x = getattr(self, name + "_proj")(hidden_states)
            conv = getattr(self, name + "_conv1d")
            value = conv(
                x,
                kernel=kernels[1] if kernels is not None else None,
                cache_params=cache_params,
                layer_idx=self.layer_idx,
                state_idx=slot,
                cu_seqlens=cu_seqlens,
            )
            projected.append(value.reshape(shape))
        gate = self.decay_gate(self.f_b_proj(self.f_a_proj(hidden_states)))
        beta = self.b_proj(hidden_states).float().sigmoid()
        initial = (
            cache_params.get_recurrent_state(self.layer_idx) if cache_params is not None else None
        )
        if kernels is not None:
            output, state = kernels[0](
                q=projected[0],
                k=projected[1],
                v=projected[2],
                g=gate.contiguous(),
                beta=beta.contiguous(),
                initial_state=initial,
                output_final_state=cache_params is not None,
                cu_seqlens=cu_seqlens,
                use_qk_l2norm_in_kernel=True,
                safe_gate=self.gate_lower_bound is not None,
                transpose_state_layout=True,
            )
        else:
            output, state = _kda_recurrence(
                projected[0], projected[1], projected[2], gate, beta, initial
            )
        if cache_params is not None:
            cache_params.update_recurrent_state(state, self.layer_idx)
        gate = (
            self.g_proj(hidden_states)
            if self.use_full_rank_gate
            else self.g_b_proj(self.g_a_proj(hidden_states))
        )
        return self.o_proj(
            self.o_norm(output, gate.reshape(shape)).to(hidden_states.dtype).flatten(-2)
        )

    def forward(self, hidden_states, attention_mask=None, cache_params=None, **kwargs):
        if attention_mask is None:
            return self._core(hidden_states, cache_params)
        valid = attention_mask[:, -hidden_states.shape[1] :].bool()
        if cache_params is not None:
            torch._assert_async(valid.all(), "Cached K3.1 KDA requires unpadded input chunks.")
            return self._core(hidden_states, cache_params)
        if hidden_states.is_cuda and self.implementation != "torch" and _fla_kernels() is not None:
            indices = valid.flatten().nonzero().flatten()
            lengths = valid.sum(-1, dtype=torch.int32)
            cu_seqlens = F.pad(lengths.cumsum(0, dtype=torch.int32), (1, 0))
            unpadded = hidden_states.flatten(0, 1)[indices].unsqueeze(0)
            output = self._core(unpadded, cu_seqlens=cu_seqlens)[0]
            return (
                torch.zeros_like(hidden_states)
                .flatten(0, 1)
                .index_copy(0, indices, output)
                .reshape_as(hidden_states)
            )
        rows = []
        for x, keep in zip(hidden_states, valid):
            indices = keep.nonzero().flatten()
            output = torch.zeros_like(x)
            if indices.numel():
                output = output.index_copy(0, indices, self._core(x[indices].unsqueeze(0))[0])
            rows.append(output)
        return torch.stack(rows)


class KimiK31Indexer(nn.Module):
    """K3.1's NoPE, LayerNorm, non-Hadamard DSA token selector."""

    def __init__(self, config, layer_idx):
        super().__init__()
        dsa = config.dsa_config
        self.layer_idx = layer_idx
        self.n_heads, self.head_dim = dsa["index_num_heads"], dsa["index_head_dim"]
        self.index_topk = dsa["index_topk"]
        self.wq_b = nn.Linear(config.q_lora_rank, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(config.hidden_size, self.head_dim, bias=False)
        self.k_norm = KimiK31IndexerLayerNorm(
            self.head_dim, eps=config.rms_norm_eps, dtype=torch.float32
        )
        self.weights_proj = nn.Linear(config.hidden_size, self.n_heads, bias=False)

    @torch.no_grad()
    def forward(self, hidden_states, q_resid, attention_mask, cache_params=None):
        query = self.wq_b(q_resid).unflatten(-1, (self.n_heads, self.head_dim))
        key = self.k_norm(self.wk(hidden_states)).to(hidden_states.dtype)
        if cache_params is not None:
            key = cache_params.update_indexer(key, self.layer_idx)
        # This intermediate rounding matches K3.1's bf16 head-gate path.
        weights = (self.weights_proj(hidden_states) * (self.n_heads**-0.5)).float()
        scores = torch.einsum("bshd,btd->bsht", query.float(), key.float()).relu()
        scores = (scores * weights.unsqueeze(-1)).sum(-2) * (self.head_dim**-0.5)
        scores = scores.masked_fill(~attention_mask[:, 0], -torch.inf)
        return scores.topk(min(self.index_topk, key.shape[1]), dim=-1).indices


def eager_attention_forward(
    module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs
):
    scores = (query @ key.transpose(-1, -2)) * scaling
    scores = scores.masked_fill(~attention_mask, -torch.inf)
    weights = scores.float().softmax(-1).to(query.dtype).nan_to_num()
    weights = F.dropout(weights, p=dropout, training=module.training)
    return (weights @ value).transpose(1, 2), weights


class KimiK31DSAAttention(nn.Module):
    """K3's NoPE MLA with K3.1 sparse indexing and the trained output gate."""

    def __init__(self, config, layer_idx):
        super().__init__()
        self.config, self.layer_idx = config, layer_idx
        self.num_heads = config.num_attention_heads
        self.num_key_value_groups = 1
        self.q_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
        self.scaling = self.q_head_dim**-0.5
        self.is_causal = True
        self.q_a_proj = nn.Linear(config.hidden_size, config.q_lora_rank, bias=False)
        self.q_a_layernorm = KimiK31RMSNorm(config.q_lora_rank, eps=1e-6)
        self.q_b_proj = nn.Linear(config.q_lora_rank, self.num_heads * self.q_head_dim, bias=False)
        self.kv_a_proj_with_mqa = nn.Linear(
            config.hidden_size, config.kv_lora_rank + config.qk_rope_head_dim, bias=False
        )
        self.kv_a_layernorm = KimiK31RMSNorm(config.kv_lora_rank, eps=1e-6)
        self.kv_b_proj = nn.Linear(
            config.kv_lora_rank,
            self.num_heads * (config.qk_nope_head_dim + config.v_head_dim),
            bias=False,
        )
        self.g_proj = nn.Linear(config.hidden_size, self.num_heads * config.v_head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * config.v_head_dim, config.hidden_size, bias=False)
        self.indexer = KimiK31Indexer(config, layer_idx)

    def forward(self, hidden_states, attention_mask, cache_params=None, **kwargs):
        config = self.config
        batch, seq = hidden_states.shape[:2]
        q_resid = self.q_a_layernorm(self.q_a_proj(hidden_states))
        query = (
            self.q_b_proj(q_resid).view(batch, seq, self.num_heads, self.q_head_dim).transpose(1, 2)
        )
        kv, key_rot = self.kv_a_proj_with_mqa(hidden_states).split(
            [config.kv_lora_rank, config.qk_rope_head_dim], -1
        )
        kv = (
            self.kv_b_proj(self.kv_a_layernorm(kv))
            .view(batch, seq, self.num_heads, -1)
            .transpose(1, 2)
        )
        key_nope, value = kv.split([config.qk_nope_head_dim, config.v_head_dim], -1)
        key = torch.cat([key_nope, key_rot[:, None].expand(batch, self.num_heads, seq, -1)], -1)
        if cache_params is not None:
            key, value = cache_params.update(key, value, self.layer_idx)
        indices = self.indexer(hidden_states, q_resid, attention_mask, cache_params)
        selected = torch.zeros_like(attention_mask[:, 0]).scatter_(-1, indices, True)
        mask = selected[:, None] & attention_mask
        interface = ALL_ATTENTION_FUNCTIONS.get_interface(
            config._attn_implementation, eager_attention_forward
        )
        output, _ = interface(
            self,
            query,
            key,
            value,
            attention_mask=mask,
            scaling=self.scaling,
            dropout=config.attention_dropout if self.training else 0.0,
        )
        output = output.reshape(batch, seq, -1) * self.g_proj(hidden_states).sigmoid()
        return self.o_proj(output)


class KimiK31DecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.is_kda = config.layer_types[layer_idx] == "linear_attention"
        self.self_attn = (
            KimiK31DeltaAttention(config, layer_idx)
            if self.is_kda
            else KimiK31DSAAttention(config, layer_idx)
        )
        self.mlp = (
            KimiK31MoE(config)
            if layer_idx >= config.first_k_dense_replace and layer_idx % config.moe_layer_freq == 0
            else KimiK31MLP(config)
        )
        self.input_layernorm = KimiK31RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = KimiK31RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.attn_res_block_size = config.attn_res_block_size
        if self.attn_res_block_size is not None:
            self.self_attention_res_norm = KimiK31RMSNorm(config.hidden_size, config.rms_norm_eps)
            self.mlp_res_norm = KimiK31RMSNorm(config.hidden_size, config.rms_norm_eps)
            self.self_attention_res_proj = nn.Linear(config.hidden_size, 1, bias=False)
            self.mlp_res_proj = nn.Linear(config.hidden_size, 1, bias=False)

    def forward(
        self, hidden_states, blocks=(), attention_mask=None, padding_mask=None, cache_params=None
    ):
        mask = padding_mask if self.is_kda else attention_mask
        if self.attn_res_block_size is None:
            hidden_states = hidden_states + self.self_attn(
                self.input_layernorm(hidden_states),
                attention_mask=mask,
                cache_params=cache_params,
            )
            return hidden_states + self.mlp(self.post_attention_layernorm(hidden_states)), blocks
        prefix = hidden_states
        if blocks:
            hidden_states = _mix_residual(
                prefix, blocks, self.self_attention_res_norm, self.self_attention_res_proj
            )
        if self.layer_idx % self.attn_res_block_size == 0:
            blocks = (*blocks, prefix)
            prefix = None
        output = self.self_attn(
            self.input_layernorm(hidden_states), attention_mask=mask, cache_params=cache_params
        )
        prefix = output if prefix is None else prefix + output
        mlp_input = _mix_residual(prefix, blocks, self.mlp_res_norm, self.mlp_res_proj)
        return prefix + self.mlp(self.post_attention_layernorm(mlp_input)), blocks


class KimiK31PreTrainedModel(PreTrainedModel):
    """HF lifecycle and distributed-plan contract for the K3.1 text port."""

    config_class = KimiK31Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["KimiK31DecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_sdpa = True
    _supports_attention_backend = True
    supports_context_parallel = False
    context_parallel_fallback = "replicated_full_context"
    _keep_in_fp32_modules_strict = [
        r"\.decay_gate\.",
        r"\.q_conv1d\.",
        r"\.k_conv1d\.",
        r"\.v_conv1d\.",
        r"\.o_norm\.",
        r"\.k_norm\.",
        r"\.mlp\.gate\.",
    ]

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, KimiK31Experts):
            nn.init.normal_(module.gate_up_proj, std=self.config.initializer_range)
            nn.init.normal_(module.down_proj, std=self.config.initializer_range)
        elif isinstance(module, KimiK31Router):
            nn.init.normal_(module.weight, std=self.config.initializer_range)
            nn.init.zeros_(module.e_score_correction_bias)
        elif isinstance(module, KimiK31DecayGate):
            nn.init.uniform_(module.A_log, 1, 16)
            module.A_log.log_()
            nn.init.zeros_(module.dt_bias)
        elif isinstance(module, KimiK31ShortConvolution):
            nn.init.normal_(module.weight, std=self.config.initializer_range)
        elif isinstance(module, KimiK31RMSNorm):
            nn.init.ones_(module.weight)


class KimiK31Model(KimiK31PreTrainedModel):
    """Complete K3.1 text decoder, including learned attention residual blocks."""

    def __init__(self, config):
        """Build the configured mixture of KDA/DSA and dense/latent-MoE layers."""
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList(
            [KimiK31DecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = KimiK31RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.gradient_checkpointing = False
        if config.attn_res_block_size is not None:
            self.output_attn_res_norm = KimiK31RMSNorm(config.hidden_size, config.rms_norm_eps)
            self.output_attn_res_proj = nn.Linear(config.hidden_size, 1, bias=False)
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_hidden_states=False,
        return_dict=True,
        **kwargs,
    ):
        """Run causal text tokens; pixel/media inputs and packed documents are unsupported."""
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Supply exactly one of input_ids and inputs_embeds.")
        if any(
            kwargs.get(key) is not None
            for key in ("pixel_values", "images", "videos", "cu_seqlens", "packed_token_indices")
        ):
            raise ValueError("This K3.1 port accepts ordinary causal text batches only.")
        use_cache = self.config.use_cache if use_cache is None else use_cache
        if (
            self.gradient_checkpointing
            and self.training
            and (use_cache or past_key_values is not None)
        ):
            raise ValueError("K3.1 gradient checkpointing requires use_cache=False.")
        if past_key_values is not None and not use_cache:
            raise ValueError("A supplied K3.1 cache requires use_cache=True.")
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)
        hidden = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        batch, seq = hidden.shape[:2]
        past = past_key_values.get_seq_length() if past_key_values is not None else 0
        query_positions = torch.arange(past, past + seq, device=hidden.device)
        key_positions = torch.arange(past + seq, device=hidden.device)
        causal = key_positions[None, :] <= query_positions[:, None]
        causal = causal[None, None].expand(batch, 1, seq, past + seq)
        if attention_mask is not None:
            if attention_mask.ndim != 2 or attention_mask.shape != (batch, past + seq):
                raise ValueError(
                    "K3.1 attention_mask must cover the full [batch, past + sequence] input."
                )
            causal = causal & attention_mask[:, None, None, :].bool()
        all_hidden = () if output_hidden_states else None
        blocks = ()
        for layer in self.layers:
            if output_hidden_states:
                all_hidden += (hidden,)
            hidden, blocks = layer(
                hidden,
                blocks,
                attention_mask=causal,
                padding_mask=attention_mask,
                cache_params=past_key_values,
            )
        if blocks:
            hidden = _mix_residual(
                hidden, blocks, self.output_attn_res_norm, self.output_attn_res_proj
            )
        hidden = self.norm(hidden)
        if output_hidden_states:
            all_hidden += (hidden,)
        result = BaseModelOutputWithPast(
            last_hidden_state=hidden, past_key_values=past_key_values, hidden_states=all_hidden
        )
        return result if return_dict else result.to_tuple()


class KimiK31ForCausalLM(KimiK31PreTrainedModel, GenerationMixin):
    """Differentiable K3.1 text causal language model; no vision tower is constructed."""

    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    _tp_plan = {"lm_head": "colwise_gather_output"}
    _fsdp_plan = {"lm_head": "keep_full_weight"}

    def __init__(self, config):
        """Build the complete text decoder with its causal language modeling head."""
        super().__init__(config)
        self.model = KimiK31Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.vocab_size = config.vocab_size
        self.post_init()

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
        return_dict=True,
        **kwargs,
    ):
        """Return standard HF logits/loss, preserving gradients through the full text decoder."""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            return_dict=True,
            **kwargs,
        )
        indices = (
            slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        )
        logits = self.lm_head(outputs.last_hidden_state[:, indices])
        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits, labels=labels, vocab_size=self.vocab_size, **kwargs
            )
        result = CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
        )
        return result if return_dict else result.to_tuple()


def convert_kimi_k31_state_dict(state_dict, config):
    """Map unquantized published text weights onto fused experts and normalized names.

    Packed checkpoint tensors must first be dequantized. Vision weights are rejected
    so callers must explicitly select the language_model subtree.
    """
    output, experts = {}, {}
    for name, value in state_dict.items():
        if name.startswith(("vision_tower.", "mm_projector.")):
            raise ValueError("Select text checkpoint weights before converting K3.1.")
        name = name.removeprefix("language_model.").replace(".block_sparse_moe.", ".mlp.")
        if name.endswith(("weight_packed", "weight_scale", "weight_scale_inv")):
            raise ValueError("Dequantize the published packed K3.1 checkpoint before conversion.")
        if value.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            raise ValueError("K3.1 conversion requires reconstructed floating checkpoint tensors.")
        match = re.fullmatch(r"(.*\.mlp\.experts)\.(\d+)\.(w1|w2|w3)\.weight", name)
        if match:
            base, expert, projection = match.groups()
            weights = experts.setdefault(base, {})
            if (int(expert), projection) in weights:
                raise ValueError(f"Duplicate K3.1 checkpoint mapping for {name!r}.")
            weights[int(expert), projection] = value
            continue
        if name.endswith((".self_attn.A_log", ".self_attn.dt_bias")):
            name = name.replace(".self_attn.", ".self_attn.decay_gate.")
        if name.endswith(".decay_gate.A_log"):
            heads = config.linear_attn_config["num_heads"]
            if value.ndim != 1 or value.numel() < heads:
                raise ValueError("K3.1 A_log must contain at least one value per KDA head.")
            value = value[:heads]
        if name in output:
            raise ValueError(f"Duplicate K3.1 checkpoint mapping for {name!r}.")
        output[name] = value
    for base, weights in experts.items():
        expected = {(i, part) for i in range(config.num_experts) for part in ("w1", "w2", "w3")}
        if weights.keys() != expected:
            raise ValueError(f"Incomplete or unexpected K3.1 expert tensors for {base!r}.")
        for (_, part), value in weights.items():
            shape = (config.moe_intermediate_size, config.routed_expert_hidden_size)
            if part == "w2":
                shape = shape[::-1]
            if tuple(value.shape) != shape:
                raise ValueError(f"K3.1 expert {part} must have shape {shape}.")
        if any(base + suffix in output for suffix in (".gate_up_proj", ".down_proj")):
            raise ValueError(f"Duplicate K3.1 fused and split expert tensors for {base!r}.")
        output[base + ".gate_up_proj"] = torch.stack(
            [
                torch.cat([weights[i, "w1"], weights[i, "w3"]], dim=0)
                for i in range(config.num_experts)
            ]
        )
        output[base + ".down_proj"] = torch.stack(
            [weights[i, "w2"] for i in range(config.num_experts)]
        )
    return output
