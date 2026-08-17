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

"""Tests for the Kimi-K3 checkpoint-mirror PTQ recipe."""

from modelopt.recipe import load_recipe

RECIPE = "huggingface/models/moonshotai/Kimi-K3/ptq/nvfp4_experts-fp8_pb_attention"


def test_kimi_k3_recipe_matches_published_quantization_map():
    config = load_recipe(RECIPE).quantize.model_dump()
    quant_cfg = config["quant_cfg"]

    by_name = {
        entry["quantizer_name"]: entry
        for entry in quant_cfg
        if isinstance(entry, dict) and "quantizer_name" in entry
    }

    expert_input = by_name["*block_sparse_moe.experts.*input_quantizer"]["cfg"]
    assert expert_input["constant_amax"] == 2688.0

    attention_projections = {
        "q_proj",
        "k_proj",
        "v_proj",
        "b_proj",
        "f_a_proj",
        "f_b_proj",
        "q_a_proj",
        "q_b_proj",
        "kv_a_proj_with_mqa",
        "kv_b_proj",
        "o_proj",
        "g_proj",
    }
    for projection in attention_projections:
        weight_cfg = by_name[f"*self_attn.{projection}*weight_quantizer"]["cfg"]
        assert weight_cfg["num_bits"] == (4, 3)
        assert weight_cfg["block_sizes"] == {-1: 128, -2: 128}

    assert not any("kv_cache" in name for name in by_name)
