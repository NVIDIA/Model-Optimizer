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

"""Tests for the Kimi-K3 PTQ recipes: the checkpoint mirror and the vLLM-loadable one."""

import torch.nn as nn

import modelopt.torch.quantization as mtq
from modelopt.recipe import load_recipe

RECIPE = "models/moonshotai/Kimi-K3/ptq/nvfp4_experts-fp8_pb_attention"
VLLM_RECIPE = "model_type/kimi_k3/ptq/nvfp4_experts_only-kv_fp8_cast"
# The layers vLLM's Kimi-K3 builds unquantized, as named in the tiny model below.
VLLM_UNQUANTIZED = {
    "layers.0.block_sparse_moe.routed_expert_down_proj",
    "layers.0.block_sparse_moe.routed_expert_up_proj",
    "layers.0.self_attention_res_proj",
    "layers.0.mlp_res_proj",
    "output_attn_res_proj",
    "layers.0.self_attn.q_conv1d",
    "layers.0.self_attn.k_conv1d",
    "layers.0.self_attn.v_conv1d",
}


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


def _tiny_kimi_k3():
    """Kimi-K3's module paths as its checkpoint's code names them: one KDA layer, MoE included."""

    def linear():
        return nn.Linear(16, 16, bias=False)

    layer = nn.Module()
    layer.self_attn = nn.Module()
    for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
        setattr(layer.self_attn, name, linear())
    for name in ("q_conv1d", "k_conv1d", "v_conv1d"):
        setattr(layer.self_attn, name, nn.Conv1d(16, 16, 4, groups=16, bias=False))
    moe = layer.block_sparse_moe = nn.Module()
    moe.experts = nn.ModuleList([nn.Module()])
    for name in ("w1", "w2", "w3"):
        setattr(moe.experts[0], name, linear())
    moe.shared_experts = nn.Module()
    for name in ("gate_proj", "up_proj", "down_proj"):
        setattr(moe.shared_experts, name, linear())
    moe.routed_expert_down_proj, moe.routed_expert_up_proj = linear(), linear()
    layer.self_attention_res_proj, layer.mlp_res_proj = linear(), linear()
    model = nn.Module()
    model.layers = nn.ModuleList([layer])
    model.output_attn_res_proj = linear()
    return model


def _weight_quantizers_off(config):
    """Names of the tiny model's modules whose weight quantizer ``config`` leaves disabled."""
    model = _tiny_kimi_k3()
    mtq.quantize(model, {**config, "algorithm": None})
    return {
        name.removesuffix(".weight_quantizer")
        for name, module in model.named_modules()
        if name.endswith(".weight_quantizer") and not module.is_enabled
    }


def test_kimi_k3_vllm_recipe_quantizes_the_experts_but_not_what_vllm_builds_unquantized():
    off = _weight_quantizers_off(load_recipe(VLLM_RECIPE).quantize.model_dump())
    assert off >= VLLM_UNQUANTIZED
    # The routed and, as in the general recipe, the shared experts are quantized.
    assert not off & {
        "layers.0.block_sparse_moe.experts.0.w1",
        "layers.0.block_sparse_moe.shared_experts.up_proj",
    }


def test_kimi_k3_vllm_recipe_exclusions_name_exactly_what_vllm_builds_unquantized():
    """With every weight quantized first, the recipe's exclusions alone decide what stays BF16."""
    config = load_recipe(VLLM_RECIPE).quantize.model_dump()
    every_weight = {"quantizer_name": "*weight_quantizer", "cfg": {"num_bits": 8, "axis": 0}}
    config["quant_cfg"].insert(1, every_weight)  # after base_disable_all
    assert _weight_quantizers_off(config) == VLLM_UNQUANTIZED
