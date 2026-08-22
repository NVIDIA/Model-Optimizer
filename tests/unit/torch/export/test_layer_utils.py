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

"""Unit tests for modelopt.torch.export.layer_utils — MoE detection and expert naming."""

import pytest
import torch
import torch.nn as nn

from modelopt.torch.export.layer_utils import get_expert_linear_names, is_moe

# ---------------------------------------------------------------------------
# is_moe tests
# ---------------------------------------------------------------------------


class _FakeSparseMoeBlock(nn.Module):
    """Name ends with 'sparsemoeblock' — detected by naming convention."""


class _FakeMoeLayer(nn.Module):
    """Name contains 'moelayer' — detected by naming convention."""


class _FakeArcticMoe(nn.Module):
    """Name contains 'arcticmoe' — detected by explicit match."""


class _StructuralMoeModule(nn.Module):
    """Has router + experts attributes — detected by structural check."""

    def __init__(self):
        super().__init__()
        self.router = nn.Linear(8, 4)
        self.experts = nn.ModuleList([nn.Linear(8, 8) for _ in range(4)])


class _NotMoeModule(nn.Module):
    """Plain module — should NOT be classified as MoE."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(8, 8)


class _PartialStructuralModule(nn.Module):
    """Has router but no experts — should NOT be classified as MoE."""

    def __init__(self):
        super().__init__()
        self.router = nn.Linear(8, 4)


@pytest.mark.parametrize(
    "module_cls",
    [_FakeSparseMoeBlock, _FakeMoeLayer, _FakeArcticMoe],
)
def test_is_moe_name_based(module_cls):
    assert is_moe(module_cls())


def test_is_moe_structural():
    assert is_moe(_StructuralMoeModule())


def test_is_moe_negative():
    assert not is_moe(_NotMoeModule())


def test_is_moe_partial_structural():
    assert not is_moe(_PartialStructuralModule())


# ---------------------------------------------------------------------------
# get_expert_linear_names tests
# ---------------------------------------------------------------------------


class _FakeGemma4TextDecoderLayer(nn.Module):
    pass


class _FakeMixtralSparseMoeBlock(nn.Module):
    pass


class _FakeNemotronHMOE(nn.Module):
    pass


def test_get_expert_linear_names_gemma4():
    assert get_expert_linear_names(_FakeGemma4TextDecoderLayer()) == [
        "gate_proj",
        "down_proj",
        "up_proj",
    ]


def test_get_expert_linear_names_mixtral():
    assert get_expert_linear_names(_FakeMixtralSparseMoeBlock()) == ["w1", "w2", "w3"]


def test_get_expert_linear_names_nemotron():
    assert get_expert_linear_names(_FakeNemotronHMOE()) == ["up_proj", "down_proj"]


class _SeparateGateUpExperts(nn.Module):
    """A sequential MoE: each expert its own module with separate gate/up linears.

    transformers 5.x stacks the experts on every built-in MoE architecture, so this is the
    only shape in reach that ``sync_moe_gate_up_amax`` can act on -- and it is the shape
    ``trust_remote_code`` MoEs (DeepSeek-V3, Kimi-K3) still use.
    """

    def __init__(self, n_experts: int = 3, gate_name: str = "gate_proj", up_name: str = "up_proj"):
        super().__init__()
        self.router = nn.Linear(8, n_experts)
        experts = []
        for i in range(n_experts):
            expert = nn.Module()
            for offset, name in enumerate((gate_name, up_name)):
                linear = nn.Linear(8, 8, bias=False)
                quantizer = nn.Module()
                # Distinct per side and per expert, so a partial sync is visible.
                quantizer.amax = torch.tensor(float(i + 1 + offset))
                linear.weight_quantizer = quantizer
                setattr(expert, name, linear)
            experts.append(expert)
        self.experts = nn.ModuleList(experts)


class _StackedExperts(nn.Module):
    """Experts fused into one tensor -- the transformers 5.x layout. Nothing to sync."""

    def __init__(self):
        super().__init__()
        self.router = nn.Linear(8, 4)
        self.experts = nn.Module()
        self.experts.gate_up_proj = nn.Parameter(torch.zeros(4, 8, 16))


def test_sync_moe_gate_up_amax_unifies_each_expert_pair():
    """Serving engines fuse gate_up_proj and need one weight_scale_2 per expert."""
    from modelopt.torch.export.layer_utils import sync_moe_gate_up_amax

    model = _SeparateGateUpExperts()
    synced = sync_moe_gate_up_amax(model)

    assert synced == 3, "every expert's pair started mismatched"
    for i, expert in enumerate(model.experts):
        gate = expert.gate_proj.weight_quantizer.amax
        up = expert.up_proj.weight_quantizer.amax
        assert torch.equal(gate, up), f"expert {i} left unsynced"
        # Element-wise max, not either side's own value.
        assert gate.item() == float(i + 2)


def test_sync_moe_gate_up_amax_handles_w1_w3_naming():
    """Mixtral-style experts name the pair w1/w3."""
    from modelopt.torch.export.layer_utils import sync_moe_gate_up_amax

    model = _SeparateGateUpExperts(n_experts=2, gate_name="w1", up_name="w3")

    assert sync_moe_gate_up_amax(model) == 2
    assert torch.equal(
        model.experts[0].w1.weight_quantizer.amax, model.experts[0].w3.weight_quantizer.amax
    )


def test_sync_moe_gate_up_amax_skips_stacked_experts():
    """Stacked experts share one tensor, so there is no pair to reconcile."""
    from modelopt.torch.export.layer_utils import sync_moe_gate_up_amax

    assert sync_moe_gate_up_amax(_StackedExperts()) == 0


def test_sync_moe_gate_up_amax_is_idempotent():
    """Export calls it after calibration may already have unified the group."""
    from modelopt.torch.export.layer_utils import sync_moe_gate_up_amax

    model = _SeparateGateUpExperts()
    sync_moe_gate_up_amax(model)
    assert sync_moe_gate_up_amax(model) == 0, "a second pass should find nothing to do"
