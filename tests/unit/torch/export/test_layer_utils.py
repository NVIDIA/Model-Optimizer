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


# ---------------------------------------------------------------------------
# _get_hidden_act tests (activation_func=None and strict validation)
# ---------------------------------------------------------------------------

import functools

import pytest

from modelopt.torch.export.layer_utils import _get_hidden_act


class TestGetHiddenActStrictValidation:
    def test_get_hidden_act_none_raises_value_error(self):
        with pytest.raises(ValueError, match="Activation function evaluated to None"):
            _get_hidden_act(None)

    def test_get_hidden_act_functools_partial_unwrapped(self):
        # We need a mock function that has a recognized __name__
        def silu():
            pass

        partial_act = functools.partial(silu, inplace=True)
        # Should unwrap and see "silu" which maps to "silu"
        result = _get_hidden_act(partial_act)
        assert result == "silu"

    def test_get_hidden_act_lambda_raises_value_error(self):
        lambda_act = lambda x: x  # noqa: E731
        with pytest.raises(ValueError, match="lambda without a concrete name"):
            _get_hidden_act(lambda_act)

    def test_get_hidden_act_unmapped_raises_value_error(self):
        def custom_unmapped_func():
            pass

        with pytest.raises(
            ValueError, match="missing from our Hugging Face translation dictionary"
        ):
            _get_hidden_act(custom_unmapped_func)
