# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fused V4.1 experts with HF dispatch and the published pre-down routing weights."""

import torch
import torch.nn.functional as F
from torch import nn
from transformers.integrations import moe as hf_moe
from transformers.integrations.moe import ExpertsInterface, use_experts_implementation

from ._hf_parallel import V41EpDispatchExpertsParallel
from .moe import DeepseekV41MoEConfig, _fp8_inputs, _V41Expert, _V41Router

__all__ = []


def _grouped_v41_forward(module, hidden_states, top_k_index, top_k_weights):
    module._validate_ep_dispatch()
    count, topk = top_k_index.shape
    expert_ids, permutation = top_k_index.flatten().sort()
    token_ids = permutation // topk
    values = hidden_states[token_ids]
    sentinel = expert_ids >= module.num_experts
    values = values.masked_fill(sentinel.unsqueeze(-1), 0)
    offsets = (
        torch.bincount(expert_ids.clamp_max(module.num_experts), minlength=module.num_experts + 1)[
            :-1
        ]
        .cumsum(0)
        .int()
    )
    if module.quantized_gate_up:
        mask = torch.zeros(module.num_experts + 1, dtype=torch.bool, device=values.device)
        mask[module._local_quantized_ids("quantized_gate_up")] = True
        values = torch.where(mask[expert_ids, None], _fp8_inputs(values), values)
    projected = hf_moe._grouped_linear(values, module.gate_up_proj, offsets)
    projected = projected.masked_fill(sentinel.unsqueeze(-1), 0)
    activated = module._apply_gate(projected).float() * top_k_weights.flatten()[permutation, None]
    activated = activated.to(hidden_states.dtype)
    if module.quantized_down:
        mask = torch.zeros(module.num_experts + 1, dtype=torch.bool, device=values.device)
        mask[module._local_quantized_ids("quantized_down")] = True
        activated = torch.where(mask[expert_ids, None], _fp8_inputs(activated), activated)
    output = hf_moe._grouped_linear(activated, module.down_proj, offsets)
    output = output.masked_fill(sentinel.unsqueeze(-1), 0)
    inverse = permutation.argsort()
    return output[inverse].float().reshape(count, topk, -1).sum(1)


class _V41ExpertsInterface(ExpertsInterface):
    # Generic HF backends multiply routing weights after the down projection;
    # V4.1 rounds the weighted activation before that projection instead.
    _global_mapping = {"grouped_mm": _grouped_v41_forward}


@use_experts_implementation(experts_interface=_V41ExpertsInterface())
class V41FusedExperts(nn.Module):
    _modelopt_ep_dispatch = V41EpDispatchExpertsParallel()

    def __init__(self, config):
        super().__init__()
        self.num_experts = config.n_routed_experts
        self.hidden_dim, self.intermediate_dim = config.hidden_size, config.moe_intermediate_size
        self.gate_up_proj = nn.Parameter(
            torch.empty(self.num_experts, 2 * self.intermediate_dim, self.hidden_dim)
        )
        self.down_proj = nn.Parameter(
            torch.empty(self.num_experts, self.hidden_dim, self.intermediate_dim)
        )
        self.swiglu_limit = config.swiglu_limit
        self.quantized_gate_up = frozenset()
        self.quantized_down = frozenset()
        self.act_fn = F.silu

    def _validate_ep_dispatch(self):
        if self._is_expert_parallel and not getattr(self, "_modelopt_custom_ep_dispatch", False):
            raise RuntimeError(
                "V4.1 expert parallelism requires its model-owned pre-down dispatcher."
            )

    def _local_quantized_ids(self, attribute):
        selected = getattr(self, attribute)
        if self._is_expert_parallel and self.num_experts != self.config.n_routed_experts:
            if selected and len(selected) != self.config.n_routed_experts:
                raise NotImplementedError(
                    "Expert parallelism requires uniform source activation precision within each projection."
                )
        return [index for index in range(self.num_experts) if index in selected]

    def _apply_gate(self, projected):
        gate, up = projected.float().chunk(2, -1)
        if self.swiglu_limit > 0:
            gate, up = (
                gate.clamp(max=self.swiglu_limit),
                up.clamp(-self.swiglu_limit, self.swiglu_limit),
            )
        return F.silu(gate) * up

    def forward(self, hidden_states, top_k_index, top_k_weights):
        self._validate_ep_dispatch()
        self._local_quantized_ids("quantized_gate_up")
        self._local_quantized_ids("quantized_down")
        result = torch.zeros_like(hidden_states, dtype=torch.float32)
        for expert in range(self.num_experts):
            tokens, slots = torch.where(top_k_index == expert)
            if not tokens.numel():
                continue
            inputs = hidden_states[tokens]
            if expert in self.quantized_gate_up:
                inputs = _fp8_inputs(inputs)
            projected = F.linear(inputs, self.gate_up_proj[expert])
            activated = self._apply_gate(projected) * top_k_weights[tokens, slots, None]
            activated = activated.to(hidden_states.dtype)
            if expert in self.quantized_down:
                activated = _fp8_inputs(activated)
            output = F.linear(activated, self.down_proj[expert])
            result = result.index_add(0, tokens, output.float())
        return result


class V41MoE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        component = DeepseekV41MoEConfig.from_hf_config(config)
        self.gate = _V41Router(component, None)
        self.experts = V41FusedExperts(config)
        self.shared_experts = _V41Expert(component, None, None)

    def forward(self, hidden_states, image_mask=None):
        flat = hidden_states.reshape(-1, self.config.hidden_size)
        weights, indices = self.gate(flat, None if image_mask is None else image_mask.reshape(-1))
        routed = self.experts(flat, indices, weights)
        return (
            (routed.float() + self.shared_experts(flat).float())
            .to(hidden_states.dtype)
            .reshape_as(hidden_states)
        )
