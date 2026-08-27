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

import torch
import torch.nn as nn

import modelopt.torch.quantization as mtq
from modelopt.recipe import load_recipe
from modelopt.torch.quantization.plugins.huggingface import register_fused_experts_on_the_fly


class _MiniMaxFusedExperts(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_experts = 2
        self.intermediate_dim = 32
        self.gate_up_proj = nn.Parameter(torch.randn(2, 64, 32))
        self.down_proj = nn.Parameter(torch.randn(2, 32, 32))

    def forward(self, hidden_states):
        return hidden_states


class _MiniMaxMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.experts = _MiniMaxFusedExperts()
        self.shared_experts = nn.Linear(32, 32, bias=False)
        self.gate = nn.Linear(32, 2, bias=False)


class _MiniMaxLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(32, 32, bias=False)
        self.mlp = _MiniMaxMLP()


class _MiniMaxModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.language_model = nn.Module()
        self.model.language_model.layers = nn.ModuleList([_MiniMaxLayer()])
        self.model.vision_tower = nn.ModuleList([nn.Linear(32, 32, bias=False)])
        self.lm_head = nn.Linear(32, 32, bias=False)


def test_mxfp8_nvfp4_experts_recipe_quantizer_precedence():
    model = _MiniMaxModel()
    register_fused_experts_on_the_fly(model)
    recipe = load_recipe("huggingface/minimax_m3_vl/ptq/mxfp8_nvfp4_experts")
    config = recipe.quantize.model_dump()
    assert config["algorithm"]["layerwise"]["enable"] is True
    config["algorithm"] = None
    mtq.quantize(model, config)

    layer = model.model.language_model.layers[0]
    assert layer.q_proj.weight_quantizer.block_sizes == {
        -1: 32,
        "type": "dynamic",
        "scale_bits": (8, 0),
    }
    assert layer.mlp.shared_experts.weight_quantizer.block_sizes[-1] == 32

    experts = layer.mlp.experts
    weight_quantizer = experts.gate_up_proj_weight_quantizers[0]
    assert weight_quantizer.num_bits == (2, 1)
    assert weight_quantizer.block_sizes[-1] == 16
    assert weight_quantizer.block_sizes["type"] == "static"

    input_quantizer = experts.gate_up_proj_input_quantizer
    assert input_quantizer.num_bits == (2, 1)
    assert input_quantizer.amax.item() == 2688.0

    assert layer.mlp.gate.weight_quantizer.is_enabled is False
    assert model.model.vision_tower[0].weight_quantizer.is_enabled is False
    assert model.lm_head.weight_quantizer.is_enabled is False
