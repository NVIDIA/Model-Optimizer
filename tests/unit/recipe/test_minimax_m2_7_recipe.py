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

import pytest
import torch
import torch.nn as nn

import modelopt.torch.quantization as mtq
from modelopt.recipe import load_recipe
from modelopt.torch.quantization.nn import TensorQuantizer
from modelopt.torch.quantization.plugins.huggingface import register_fused_experts_on_the_fly


class _FusedExperts(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_experts = 2
        self.intermediate_dim = 32
        self.gate_up_proj = nn.Parameter(torch.randn(2, 64, 32))
        self.down_proj = nn.Parameter(torch.randn(2, 32, 32))

    def forward(self, hidden_states):
        return hidden_states


@pytest.mark.parametrize("layout", ["block_sparse_moe", "mlp"])
@pytest.mark.parametrize("fused", [False, True])
def test_minimax_m2_7_expert_selection(layout, fused):
    model = nn.Module()
    model.model = nn.Module()
    layer = nn.Module()
    model.model.layers = nn.ModuleList([layer])
    moe = nn.Module()
    layer.add_module(layout, moe)
    moe.experts = (
        _FusedExperts()
        if fused
        else nn.ModuleList(
            [nn.ModuleDict({p: nn.Linear(32, 32) for p in ("gate_proj", "up_proj", "down_proj")})]
        )
    )
    moe.gate = nn.Linear(32, 2)
    moe.shared_experts = nn.Linear(32, 32)
    layer.self_attn = nn.ModuleDict(
        {p: nn.Linear(32, 32) for p in ("q_proj", "k_proj", "v_proj", "o_proj")}
    )
    model.lm_head = nn.Linear(32, 32)
    preserved = {name: p.detach().clone() for name, p in model.named_parameters()}
    register_fused_experts_on_the_fly(model)

    config = load_recipe(
        "models/MiniMaxAI/MiniMax-M2.7/ptq/nvfp4_experts_only-kv_fp8_cast"
    ).quantize.model_dump()
    assert config["algorithm"]["method"] == "max"
    config["algorithm"] = None
    mtq.quantize(model, config)

    enabled = {
        name: q
        for name, q in model.named_modules()
        if isinstance(q, TensorQuantizer) and q.is_enabled
    }
    assert len(enabled) == 6  # Three W4A4 projections, or four fused weights and two inputs.
    for name, q in enabled.items():
        assert f".{layout}.experts." in name
        assert q.num_bits == (2, 1)
        assert q.block_sizes[-1] == 16
        assert q.block_sizes["type"] == "dynamic"
    for name, p in model.named_parameters():
        torch.testing.assert_close(p, preserved[name], rtol=0, atol=0)
