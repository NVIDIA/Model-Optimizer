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

"""Step-3.7 PTQ recipes: what the `moe` / `share_expert` naming does and does not match."""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import modelopt.torch.quantization as mtq
from modelopt.recipe import load_recipe
from modelopt.torch.quantization.nn import QuantModuleRegistry

HIDDEN_SIZE = 32
MOE_INTERMEDIATE_SIZE = 16
NUM_EXPERTS = 2


class _MoELinear(nn.Module):
    """Step's expert-indexed projection: one 3-D weight, ``forward(x, expert_id)``."""

    def __init__(self, num_experts, in_features, out_features):
        super().__init__()
        self.num_experts = num_experts
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(num_experts, out_features, in_features) * 0.02)

    def forward(self, x, expert_id):
        return F.linear(x.float(), self.weight[expert_id].float())


class _StepMoEMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_experts = NUM_EXPERTS
        self.gate = nn.Linear(HIDDEN_SIZE, NUM_EXPERTS, bias=False)  # router
        self.up_proj = _MoELinear(NUM_EXPERTS, HIDDEN_SIZE, MOE_INTERMEDIATE_SIZE)
        self.gate_proj = _MoELinear(NUM_EXPERTS, HIDDEN_SIZE, MOE_INTERMEDIATE_SIZE)
        self.down_proj = _MoELinear(NUM_EXPERTS, MOE_INTERMEDIATE_SIZE, HIDDEN_SIZE)


class _StepMLP(nn.Module):
    """Dense FFN — used both as the non-MoE layers' ``mlp`` and as ``share_expert``."""

    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(HIDDEN_SIZE, MOE_INTERMEDIATE_SIZE, bias=False)
        self.up_proj = nn.Linear(HIDDEN_SIZE, MOE_INTERMEDIATE_SIZE, bias=False)
        self.down_proj = nn.Linear(MOE_INTERMEDIATE_SIZE, HIDDEN_SIZE, bias=False)


class _StepMoELayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = nn.Module()
        self.self_attn.q_proj = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE, bias=False)
        self.moe = _StepMoEMLP()
        self.share_expert = _StepMLP()


class _StepDenseLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = nn.Module()
        self.self_attn.q_proj = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE, bias=False)
        self.mlp = _StepMLP()


class _StepModel(nn.Module):
    """Mirrors Step3p7ForConditionalGeneration's module paths (one MoE, one dense layer)."""

    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.language_model = nn.Module()
        self.model.language_model.layers = nn.ModuleList([_StepMoELayer(), _StepDenseLayer()])
        self.lm_head = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE, bias=False)


@pytest.fixture(autouse=True)
def _unregister_moe_linear():
    yield
    if QuantModuleRegistry.get(_MoELinear) is not None:
        QuantModuleRegistry.unregister(_MoELinear)


def _quantize_with_recipe(name):
    """Convert (no calibration) with a built-in recipe and return the model."""
    model = _StepModel()
    config = load_recipe(name).quantize.model_dump()
    config["algorithm"] = None
    # `mtq.quantize` runs the custom-model plugins itself, which is what registers Step's
    # `MoELinear` — no explicit registration here, so this also covers that hook firing.
    mtq.quantize(model, config)
    return model


def _enabled(module, quantizer="weight_quantizer"):
    return getattr(module, quantizer).is_enabled


@pytest.mark.parametrize(
    "recipe_name",
    [
        "huggingface/step3p7/ptq/nvfp4_experts_only-kv_fp8_cast",
        "huggingface/step3p7/ptq/nvfp4_mlp_only-kv_fp8",
    ],
)
def test_routed_experts_are_quantized(recipe_name):
    """The routed experts — the bulk of the model — must end up quantized, per expert."""
    model = _quantize_with_recipe(recipe_name)
    moe = model.model.language_model.layers[0].moe

    for proj in ("up_proj", "gate_proj", "down_proj"):
        experts = getattr(moe, proj).experts
        assert len(experts) == NUM_EXPERTS
        for expert in experts:
            assert _enabled(expert)
            assert _enabled(expert, "input_quantizer")
            # Dynamic NVFP4: 16-element blocks along the input dim.
            assert expert.weight_quantizer.block_sizes[-1] == 16
            assert expert.weight_quantizer.num_bits == (2, 1)


@pytest.mark.parametrize(
    "recipe_name",
    [
        "huggingface/step3p7/ptq/nvfp4_experts_only-kv_fp8_cast",
        "huggingface/step3p7/ptq/nvfp4_mlp_only-kv_fp8",
    ],
)
def test_router_shared_expert_and_head_stay_bf16(recipe_name):
    """`*moe*` also matches the router; the shared expert and lm_head stay unquantized too."""
    model = _quantize_with_recipe(recipe_name)
    moe_layer = model.model.language_model.layers[0]

    assert not _enabled(moe_layer.moe.gate)
    for proj in ("gate_proj", "up_proj", "down_proj"):
        assert not _enabled(getattr(moe_layer.share_expert, proj))
    assert not _enabled(model.lm_head)


def test_experts_only_leaves_dense_mlp_bf16():
    model = _quantize_with_recipe("huggingface/step3p7/ptq/nvfp4_experts_only-kv_fp8_cast")
    dense_mlp = model.model.language_model.layers[1].mlp

    for proj in ("gate_proj", "up_proj", "down_proj"):
        assert not _enabled(getattr(dense_mlp, proj))


def test_mlp_only_also_quantizes_dense_mlp():
    model = _quantize_with_recipe("huggingface/step3p7/ptq/nvfp4_mlp_only-kv_fp8")
    dense_mlp = model.model.language_model.layers[1].mlp

    for proj in ("gate_proj", "up_proj", "down_proj"):
        assert _enabled(getattr(dense_mlp, proj))
    # Attention projections are out of scope for both recipes.
    assert not _enabled(model.model.language_model.layers[1].self_attn.q_proj)
