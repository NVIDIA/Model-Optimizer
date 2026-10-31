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

"""Tests for ``modelopt.torch.utils.fsdp2_utils``: which parameters stay out of FSDP2's shards, and
the distribution plan a wrap leaves to its caller."""

import pytest
import torch
import torch.nn as nn

import modelopt.torch.utils.distributed as dist
from modelopt.torch.quantization.utils.layerwise_calib import LayerActivationCollector
from modelopt.torch.utils import fsdp2_utils
from modelopt.torch.utils.distributed import (
    DistributionPlan,
    distribution_plan,
    set_distribution_plan,
)
from modelopt.torch.utils.fsdp2_utils import _off_dtype_params


def _model(*sizes_and_dtypes) -> nn.Module:
    model = nn.Module()
    for i, (numel, dtype) in enumerate(sizes_and_dtypes):
        model.register_parameter(f"p{i}", nn.Parameter(torch.zeros(numel, dtype=dtype)))
    return model


def test_off_dtype_params_uniform_is_empty(recwarn):
    model = _model((8, torch.bfloat16), (4, torch.bfloat16))
    assert _off_dtype_params(model) == set()
    assert not recwarn.list


def test_off_dtype_params_returns_only_the_minority():
    # An fp32 MoE router gate next to bf16 weights, as Nemotron-3-Nano ships it.
    model = _model((100, torch.bfloat16), (5, torch.float32))
    with pytest.warns(UserWarning, match="mixed parameter dtypes"):
        off = _off_dtype_params(model)
    assert off == {model.p1}


@pytest.mark.parametrize(
    ("params", "expected"),
    [
        # Dominant dtype is by element count, not parameter count: three small fp32 params
        # lose to one large bf16 param, and vice versa.
        ([(100, torch.bfloat16), (5, torch.float32), (5, torch.float32), (5, torch.float32)], "p0"),
        (
            [(100, torch.float32), (5, torch.bfloat16), (5, torch.bfloat16), (5, torch.bfloat16)],
            "p0",
        ),
    ],
)
def test_off_dtype_params_dominant_is_by_numel(params, expected):
    model = _model(*params)
    with pytest.warns(UserWarning, match="mixed parameter dtypes"):
        off = _off_dtype_params(model)
    kept = {n for n, p in model.named_parameters() if p not in off}
    assert kept == {expected}


def test_fsdp2_wrap_leaves_the_plan_to_the_load(monkeypatch):
    """load_pretrained records the plan once, at the end of the load; the wrap it calls records none,
    so a caller that wraps a model itself must record one before anything reads it."""
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 2))
    monkeypatch.setattr(
        LayerActivationCollector, "get_decoder_layers", staticmethod(lambda model: list(model))
    )
    monkeypatch.setattr(fsdp2_utils, "fully_shard", lambda module, **kwargs: None)
    monkeypatch.setattr(dist, "size", lambda group=None: 4)

    fsdp2_utils.fsdp2_wrap(model)

    with pytest.raises(RuntimeError, match="no distribution plan recorded"):
        distribution_plan(model)


@pytest.mark.parametrize(
    ("plan", "recorded_on"),
    [
        (DistributionPlan(), "model"),
        (DistributionPlan(4, dense="fsdp"), "model"),
        (DistributionPlan(), "submodule"),
    ],
    ids=["replicated", "sharded", "in-a-submodule"],
)
def test_fsdp2_wrap_refuses_a_model_with_a_plan_recorded(monkeypatch, plan, recorded_on):
    """A plan is recorded once, so a model that records one, as load_pretrained records one on every
    model it loads, is refused rather than sharded behind it: a replicated model would read as
    replicated, which the export would believe, and a sharded one would be wrapped twice."""
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 2))
    set_distribution_plan(model if recorded_on == "model" else model[0], plan)
    monkeypatch.setattr(
        LayerActivationCollector, "get_decoder_layers", staticmethod(lambda model: list(model))
    )
    monkeypatch.setattr(fsdp2_utils, "fully_shard", lambda *a, **k: pytest.fail("sharded"))

    # A ValueError, as set_distribution_plan raises for the same record: one type to catch.
    with pytest.raises(ValueError, match="already recorded as distributed"):
        fsdp2_utils.fsdp2_wrap(model)
