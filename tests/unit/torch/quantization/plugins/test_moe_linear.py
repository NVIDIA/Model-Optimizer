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

"""Tests for _QuantMoELinear: expert-indexed MoE weights (Step-3.5 / Step-3.7 remote code)."""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

pytest.importorskip("transformers")

import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.nn import QuantModuleRegistry
from modelopt.torch.quantization.plugins.huggingface import (
    _is_expert_indexed_moe_linear,
    _QuantMoELinear,
    _reconstruct_fused_moe_linear,
    register_moe_linear_on_the_fly,
)

NUM_EXPERTS = 4
HIDDEN_SIZE = 32
MOE_INTERMEDIATE_SIZE = 16
TOP_K = 2


class _SyntheticMoELinear(nn.Module):
    """Mimics Step-3.5 / Step-3.7 ``MoELinear`` (verbatim layout from their remote code)."""

    def __init__(self, num_experts, in_features, out_features):
        super().__init__()
        self.num_experts = num_experts
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(num_experts, out_features, in_features) * 0.02)

    def forward(self, x, expert_id):
        return F.linear(x.float(), self.weight[expert_id].float())


class _SyntheticStepMoEMLP(nn.Module):
    """Mimics ``Step3p7MoEMLP``: a router plus three expert-indexed projections."""

    def __init__(self):
        super().__init__()
        self.num_experts = NUM_EXPERTS
        self.top_k = TOP_K
        self.gate = nn.Linear(HIDDEN_SIZE, NUM_EXPERTS, bias=False)
        self.act_fn = nn.SiLU()
        self.up_proj = _SyntheticMoELinear(NUM_EXPERTS, HIDDEN_SIZE, MOE_INTERMEDIATE_SIZE)
        self.gate_proj = _SyntheticMoELinear(NUM_EXPERTS, HIDDEN_SIZE, MOE_INTERMEDIATE_SIZE)
        self.down_proj = _SyntheticMoELinear(NUM_EXPERTS, MOE_INTERMEDIATE_SIZE, HIDDEN_SIZE)

    def forward(self, hidden_states):
        tokens = hidden_states.view(-1, HIDDEN_SIZE)
        routing = F.softmax(self.gate(tokens).float(), dim=-1)
        weights, indices = torch.topk(routing, self.top_k, dim=-1)
        out = torch.zeros_like(tokens)
        for expert_id in range(self.num_experts):
            pos, token_idx = torch.where(indices == expert_id)
            if token_idx.numel() == 0:
                continue
            current = tokens[pos]
            gate = self.act_fn(self.gate_proj(current, expert_id))
            up = self.up_proj(current, expert_id)
            expert_out = self.down_proj(gate * up, expert_id)
            out.index_add_(0, pos, (expert_out * weights[pos, token_idx, None]).to(out.dtype))
        return out.view_as(hidden_states)


class _TinyStepModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.moe = _SyntheticStepMoEMLP()

    def forward(self, x):
        return self.moe(x)


@pytest.fixture(autouse=True)
def _unregister_synthetic_moe_linear():
    """Keep the on-the-fly registration from leaking into other tests."""
    yield
    if QuantModuleRegistry.get(_SyntheticMoELinear) is not None:
        QuantModuleRegistry.unregister(_SyntheticMoELinear)


def _moe_quant_cfg():
    """Per-tensor INT8 on the expert projections only — CPU-friendly, no kernels needed."""
    return {
        "quant_cfg": [
            {"quantizer_name": "*", "enable": False},
            {"quantizer_name": "*moe*weight_quantizer", "cfg": {"num_bits": 8, "axis": 0}},
            {"quantizer_name": "*moe*input_quantizer", "cfg": {"num_bits": 8, "axis": None}},
            {"quantizer_name": "*moe.gate.*", "enable": False},
        ],
        "algorithm": "max",
    }


def test_expert_indexed_moe_linear_is_detected():
    assert _is_expert_indexed_moe_linear(
        _SyntheticMoELinear(NUM_EXPERTS, HIDDEN_SIZE, MOE_INTERMEDIATE_SIZE)
    )


@pytest.mark.parametrize(
    "module",
    [
        pytest.param(nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE), id="plain_linear_2d_weight"),
        pytest.param(nn.LayerNorm(HIDDEN_SIZE), id="norm_1d_weight"),
    ],
)
def test_unrelated_modules_are_not_claimed(module):
    assert not _is_expert_indexed_moe_linear(module)


def test_module_with_3d_weight_but_other_forward_is_not_claimed():
    """A 3-D weight alone is not enough — the forward must take ``(x, expert_id)``."""

    class _NotExpertIndexed(_SyntheticMoELinear):
        def forward(self, x, top_k_index, top_k_weights):
            return x

    assert not _is_expert_indexed_moe_linear(
        _NotExpertIndexed(NUM_EXPERTS, HIDDEN_SIZE, MOE_INTERMEDIATE_SIZE)
    )


def test_registration_is_not_gated_on_model_class_name():
    """Detection is structural, so a Step-3.7-style model registers as readily as Step-3.5."""
    model = _TinyStepModel()
    assert QuantModuleRegistry.get(_SyntheticMoELinear) is None

    register_moe_linear_on_the_fly(model)

    assert issubclass(QuantModuleRegistry.get(_SyntheticMoELinear), _QuantMoELinear)


def test_expert_indexed_moe_is_quantized_and_reconstructed():
    """Each expert gets its own quantizers, and export folds them back to the 3-D layout."""
    torch.manual_seed(0)
    model = _TinyStepModel()
    reference_weight = model.moe.up_proj.weight.detach().clone()

    def forward_loop(m):
        m(torch.randn(2, 8, HIDDEN_SIZE))

    mtq.quantize(model, _moe_quant_cfg(), forward_loop=forward_loop)

    # Every expert of every projection carries its own calibrated quantizer pair.
    for proj in ("up_proj", "gate_proj", "down_proj"):
        experts = getattr(model.moe, proj).experts
        assert len(experts) == NUM_EXPERTS
        for expert in experts:
            assert expert.weight_quantizer.is_enabled
            assert expert.weight_quantizer.amax is not None
    # The router stays untouched.
    assert not model.moe.gate.weight_quantizer.is_enabled

    _reconstruct_fused_moe_linear(model)

    # Back to the original ``[num_experts, out_features, in_features]`` parameter, so the
    # exported keys match the hub checkpoint instead of per-expert names.
    up_proj = model.moe.up_proj
    assert not hasattr(up_proj, "experts")
    assert up_proj.weight.shape == reference_weight.shape
    assert torch.equal(up_proj.weight, reference_weight)
