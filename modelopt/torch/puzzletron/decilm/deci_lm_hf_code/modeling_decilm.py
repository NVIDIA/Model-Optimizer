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
# gated MLP, and LMHead for replacement / validation code.
# mypy: ignore-errors

import torch
import torch.nn.functional as F
from torch import nn

from .block_config import FFNConfig
from .configuration_decilm import DeciLMConfig
from .transformers_4_44_2__activations import ACT2FN


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


class LMHead(nn.Linear):
    """
    Special class to allow FSDP wrapping without affecting other Linear layers in the model.
    """
