# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Copyright 2024 Nvidia Corporation, Google Inc, HuggingFace Inc, EleutherAI. All rights reserved.
#
# Pared-down DeciLM building blocks for Model-Optimizer puzzletron / AnyModel flows.
# The full HF DeciLM decoder stack (decoder layers, attention, rope, etc.) is not vendored here;
# AnyModel loads real models via transformers. This module keeps shared helpers: RMSNorm,
# gated/vanilla MLP (used by MoE accounting), MoE, and LMHead for replacement / validation code.
# mypy: ignore-errors

import torch
import torch.nn.functional as F
from torch import nn
from transformers.utils import logging

from .block_config import FFNConfig, MoEConfig
from .configuration_decilm import DeciLMConfig
from .transformers_4_44_2__activations import ACT2FN
from .transformers_4_44_2__pytorch_utils import ALL_LAYERNORM_LAYERS

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "DeciLMConfig"


class DeciLMRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        DeciLMRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


ALL_LAYERNORM_LAYERS.append(DeciLMRMSNorm)


def sparsity_backward_hook(*args, **kwargs):
    raise NotImplementedError(
        "No support for sparsity when training HF DeciLM (inference is ok though)"
    )


class DeciLMGatedMLP(nn.Module):
    def __init__(
        self,
        config: DeciLMConfig,
        ffn_config: FFNConfig,
    ):
        super().__init__()
        self.config = config
        self.ffn_config = ffn_config
        self.hidden_size = config.hidden_size
        self.intermediate_size = ffn_config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        self.act_fn = ACT2FN[getattr(ffn_config, "hidden_act", "silu")]

        if ffn_config.sparsify is not None:
            self.register_full_backward_hook(sparsity_backward_hook)

    def forward(self, x):
        if self.config.pretraining_tp > 1:
            slice = self.intermediate_size // self.config.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice, dim=0)
            down_proj_slices = self.down_proj.weight.split(slice, dim=1)

            gate_proj = torch.cat(
                [F.linear(x, gate_proj_slices[i]) for i in range(self.config.pretraining_tp)],
                dim=-1,
            )
            up_proj = torch.cat(
                [F.linear(x, up_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1
            )

            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
            down_proj = [
                F.linear(intermediate_states[i], down_proj_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            down_proj = sum(down_proj)
        else:
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj


class DeciLMVanillaMLP(nn.Module):
    def __init__(
        self,
        config: DeciLMConfig,
        ffn_config: FFNConfig,
    ):
        super().__init__()
        self.config = config
        self.ffn_config = ffn_config
        self.hidden_size = config.hidden_size
        self.intermediate_size = ffn_config.intermediate_size
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        self.act_fn = ACT2FN[getattr(ffn_config, "hidden_act", "silu")]

        if ffn_config.sparsify is not None:
            self.register_full_backward_hook(sparsity_backward_hook)

        assert self.config.pretraining_tp == 1, (
            "Unsupported pretraining_tp != 1 for DeciLMVanillaMLP"
        )

    def forward(self, x):
        return self.down_proj(self.act_fn(self.up_proj(x)))


class DeciLMMoe(nn.Module):
    """
    Implementation of Mixture of Experts module for DeciLM.
    Equivalent to Llama4 MoE but implemented more frugally.
    """

    def __init__(self, config: DeciLMConfig, ffn_config: FFNConfig):
        super().__init__()
        self.config = config
        self.ffn_config = ffn_config

        # MoE parameters
        assert ffn_config.moe is not None, "MoE configuration must be provided to use DeciLMMoe"
        self.moe_config: MoEConfig = ffn_config.moe
        self.hidden_dim = config.hidden_size
        self.num_experts_per_tok = self.moe_config.num_experts_per_tok
        self.num_local_experts = self.moe_config.num_local_experts
        self.expert_intermediate_dim = self.moe_config.expert_intermediate_dim
        self.shared_expert_intermediate_dim = self.moe_config.shared_expert_intermediate_dim

        # Initialize experts and router
        routed_expert_ffn_config = FFNConfig(
            intermediate_size=self.expert_intermediate_dim,
        )

        self.experts = nn.ModuleList(
            [
                DeciLMGatedMLP(config, routed_expert_ffn_config)
                for _ in range(self.num_local_experts)
            ]
        )

        self.router = nn.Linear(config.hidden_size, self.num_local_experts, bias=False)

        # Initialize shared expert as a standard MLP
        shared_expert_ffn_config = FFNConfig(
            intermediate_size=self.moe_config.shared_expert_intermediate_dim
        )
        self.shared_expert = DeciLMGatedMLP(config, shared_expert_ffn_config)

        if ffn_config.sparsify is not None:
            self.register_full_backward_hook(sparsity_backward_hook)

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the MoE layer.

        Args:
            hidden_states (torch.Tensor): Input tensor of shape (batch, seq_len, hidden_dim)

        Returns:
            tuple:
                - torch.Tensor: Output tensor of shape (batch, seq_len, hidden_dim)
                - torch.Tensor: Router scores for loss computation
        """
        router_logits = self.router(hidden_states)

        routed_out = self.forward_routed_experts(hidden_states, router_logits)

        shared_out = self.shared_expert(hidden_states)

        moe_out = routed_out + shared_out

        return moe_out, router_logits

    def forward_routed_experts(
        self, hidden_states: torch.Tensor, router_logits: torch.Tensor
    ) -> torch.Tensor:
        """
        For each expert:
        1. Build the input to the expert based on the router mask
        2. Run the expert
        3. Add the result of the expert into the total MoE result using +=
        """
        router_top_values, router_indices = torch.topk(
            router_logits, self.num_experts_per_tok, dim=-1
        )
        router_scores = torch.sigmoid(router_top_values.float()).to(hidden_states.dtype)

        routed_out = torch.zeros_like(hidden_states)
        for i_expert in range(self.num_local_experts):
            expert_mask = router_indices == i_expert
            if expert_mask.any():
                is_token_routed_to_this_expert = expert_mask.any(dim=-1)
                relevant_hidden_states = hidden_states[is_token_routed_to_this_expert, :]
                relevant_scores = router_scores[expert_mask]
                expert_in = relevant_hidden_states * relevant_scores.unsqueeze(-1)

                expert_out = self.experts[i_expert](expert_in).to(hidden_states.device)

                routed_out[is_token_routed_to_this_expert, :] += expert_out

        return routed_out

    def extra_repr(self) -> str:
        return (
            f"(MoE): num_local_experts={self.num_local_experts}, "
            f"expert_intermediate_dim={self.expert_intermediate_dim},"
        )


class LMHead(nn.Linear):
    """
    Special class to allow FSDP wrapping without affecting other Linear layers in the model.
    """
