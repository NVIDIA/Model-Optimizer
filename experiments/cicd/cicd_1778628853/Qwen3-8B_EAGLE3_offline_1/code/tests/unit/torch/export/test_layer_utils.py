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
