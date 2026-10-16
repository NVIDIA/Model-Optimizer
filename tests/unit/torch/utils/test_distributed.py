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

"""Pure-function tests for ``modelopt.torch.utils.distributed``."""

import os
import pickle

import pytest
import torch
import torch.nn as nn

from modelopt.torch.nas.hparams.container import DepthHparam
from modelopt.torch.utils.distributed import _deserialize, _off_dtype_params, _serialize


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


class _Gadget:
    def __reduce__(self):
        return (os.system, ("exit 1",))


@pytest.mark.parametrize("force_no_weights_only", [False, True])
def test_deserialize_rejects_pickle_gadget(monkeypatch, force_no_weights_only):
    # The override that trusts local checkpoints must not also trust peer payloads.
    if force_no_weights_only:
        monkeypatch.setenv("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    with pytest.raises(pickle.UnpicklingError):
        _deserialize(_serialize(_Gadget()))


def test_deserialize_traced_hp_subclass():
    # The allowlist matches exact classes, so each TracedHp subclass must be registered too.
    hp = _deserialize(_serialize(DepthHparam([1, 2])))
    assert type(hp) is DepthHparam
    assert hp.choices == [1, 2]
