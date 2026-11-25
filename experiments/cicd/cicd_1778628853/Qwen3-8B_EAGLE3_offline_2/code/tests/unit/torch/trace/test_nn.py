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

import pytest
import torch.nn as nn

from modelopt.torch.trace import Symbol
from modelopt.torch.trace.modules.nn import (
    _get_ndim,
    get_conv_sym_info,
    get_layer_norm_sym_info,
    get_linear_sym_info,
    get_norm_sym_info,
    get_sequential_sym_info,
)


@pytest.mark.parametrize(
    ("test_str", "raises_error", "expected_dim"),
    [
        ("Conv2d", False, 2),
        ("FakeConv7DPlus", False, 7),
        ("CombinedConv1dConv5D", True, None),
        ("MultiDConv", False, None),
    ],
)
def test_get_ndim(test_str: str, raises_error: bool, expected_dim: int):
    """Test _get_ndim."""
    # create class with name test_str
    test_cls = type(test_str, (nn.Module,), {})
    if raises_error:
        with pytest.raises(AssertionError):
            ndim = _get_ndim(test_cls())
    else:
        ndim = _get_ndim(test_cls())
        assert ndim == expected_dim, f"Expected ndim={expected_dim}, got {ndim}"


def test_get_linear_sym_info():
    lin = nn.Linear(10, 20)
    sym_info = get_linear_sym_info(lin)
    assert sym_info.is_shape_preserving is False
    assert sym_info.symbols.keys() == {"in_features", "out_features"}
    for sym in sym_info.symbols.values():
        assert sym.is_cross_layer, "Expected cross-layer symbol"
        assert sym.elastic_dims == {-1}, "Expected elastic dims to be {-1}"


@pytest.mark.parametrize(
    ("norm", "edims"),
    [
        (nn.BatchNorm1d(10), {1}),
        (nn.BatchNorm2d(10), {1, -3}),
        (nn.BatchNorm3d(10), {1, -4}),
        (nn.SyncBatchNorm(10), {1}),
        (nn.LayerNorm(10), {-1}),
        (nn.InstanceNorm1d(10), {1}),
        (nn.InstanceNorm2d(10), {1, -3}),
        (nn.InstanceNorm3d(10), {1, -4}),
    ],
)
def test_get_norm_sym_info(norm, edims):
    sym_info = (
        get_layer_norm_sym_info(norm) if isinstance(norm, nn.LayerNorm) else get_norm_sym_info(norm)
    )
    assert sym_info.is_shape_preserving is True
    assert sym_info.symbols.keys() == {"num_features"}
    for sym in sym_info.symbols.values():
        assert sym.cl_type == sym.CLType.INCOMING, "Expected cross-layer symbol"
        assert sym.elastic_dims == edims, f"Expected elastic dims to be {edims}"


def test_sequential_sym_info():
    model = nn.Sequential(*[nn.Linear(10, 10) for _ in range(5)])
    sym_info = get_sequential_sym_info(model)
    assert sym_info.is_shape_preserving is False
    assert sym_info.symbols.keys() == {"depth"}
    dsym = sym_info.symbols["depth"]
    assert dsym.is_searchable
    assert dsym.max_depth == len(model)
    assert dsym.min_depth == len(model)
    assert dsym.skippable_idxs == []

    with pytest.raises(RuntimeError):
        dsym.link_to(Symbol())


@pytest.mark.parametrize(
    ("conv", "edims", "has_channels"),
    [
        # regular convs
        (nn.Conv1d(10, 10, 3), {1, -2}, True),
        (nn.Conv2d(10, 10, 3), {1, -3}, True),
        (nn.Conv3d(10, 10, 3), {1, -4}, True),
        (nn.ConvTranspose1d(10, 10, 3), {1, -2}, True),
        (nn.ConvTranspose2d(10, 10, 3), {1, -3}, True),
        (nn.ConvTranspose3d(10, 10, 3), {1, -4}, True),
        # grouped tests
        (nn.Conv2d(10, 10, 3, groups=2), {1, -3}, True),
        (nn.Conv2d(10, 10, 3, groups=10), {1, -3}, True),
        (nn.Conv2d(10, 20, 3, groups=10), {1, -3}, False),
    ],
)
def test_conv_sym_info(conv, edims, has_channels):
    sym_info = get_conv_sym_info(conv)
    assert sym_info.is_shape_preserving is False
    sym_keys = {"kernel_size", "in_channels", "out_channels"}

    assert sym_info.symbols.keys() == sym_keys
    for k in ["in_channels", "out_channels"]:
        sym = sym_info.symbols[k]
        assert sym.is_cross_layer, "Expected cross-layer symbol"
        assert sym.elastic_dims == edims, f"Expected elastic dims to be {edims}"
    assert sym_info.symbols["kernel_size"].cl_type not in [
        Symbol.CLType.INCOMING,
        Symbol.CLType.OUTGOING,
    ]

    oc = sym_info.symbols["out_channels"]
    ic = sym_info.symbols["in_channels"]
    if has_channels:
        assert ic.is_free
        if conv.groups > 1:
            assert oc.is_dynamic
            assert oc.parent == sym_info.symbols["in_channels"]
        else:
            assert oc.is_searchable
    else:
        assert ic.is_constant
        assert oc.is_constant
