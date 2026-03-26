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

import torch
import torch.nn as nn

from modelopt.torch.export.hf_config_map import HF_CONFIG_MAP
from modelopt.torch.export.layer_utils import get_expert_linear_names, get_experts_list, is_moe
from modelopt.torch.export.quant_utils import PQS_FUSE_MODULE_MAPPING


class _FakeDeepseekExpert(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(8, 16, bias=False)
        self.down_proj = nn.Linear(16, 8, bias=False)
        self.up_proj = nn.Linear(8, 16, bias=False)


class _FakeDeepseekGate(nn.Module):
    def __init__(self, num_experts=2):
        super().__init__()
        self.top_k = 1
        self.n_routed_experts = num_experts
        self.gating_dim = 8
        self.weight = nn.Parameter(torch.empty(num_experts, 8))
        nn.init.normal_(self.weight)


class DeepseekV3MoE(nn.Module):
    def __init__(self, num_experts=2):
        super().__init__()
        self.gate = _FakeDeepseekGate(num_experts)
        self.experts = nn.ModuleList([_FakeDeepseekExpert() for _ in range(num_experts)])
        self.shared_experts = _FakeDeepseekExpert()


def test_is_moe_detects_deepseek_v3_moe():
    assert is_moe(DeepseekV3MoE())


def test_get_expert_linear_names_for_deepseek_v3():
    assert get_expert_linear_names(DeepseekV3MoE()) == ["gate_proj", "down_proj", "up_proj"]


def test_get_experts_list_for_deepseek_model_type():
    module = DeepseekV3MoE(num_experts=3)

    experts_list = get_experts_list(module, "deepseekv3forcausallm")

    assert len(experts_list) == 3
    assert all(len(expert_group) == 3 for expert_group in experts_list)
    assert experts_list[0][0] is module.experts[0].gate_proj
    assert experts_list[1][1] is module.experts[1].down_proj
    assert experts_list[2][2] is module.experts[2].up_proj


def test_hf_config_map_supports_deepseek_num_experts():
    assert any(
        output_name == "moe_num_experts" and "n_routed_experts" in input_names
        for input_names, output_name in HF_CONFIG_MAP
    )


def test_prequant_fuse_mapping_covers_deepseek_v3():
    assert any(
        "DeepseekV3Attention" in module_names and linear_pair == ("v_proj", "o_proj")
        for module_names, linear_pair in PQS_FUSE_MODULE_MAPPING
    )
    assert any(
        "DeepseekV3MLP" in module_names and linear_pair == ("up_proj", "down_proj")
        for module_names, linear_pair in PQS_FUSE_MODULE_MAPPING
    )
