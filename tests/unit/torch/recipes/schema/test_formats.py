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

"""Tests for format registry (schema/formats.py)."""

import pytest

from modelopt.torch.recipes.schema.formats import (
    FORMAT_REGISTRY,
    KV_FORMAT_REGISTRY,
    get_format,
    get_kv_format,
)


def test_unknown_format_raises():
    with pytest.raises(KeyError, match="Unknown format"):
        get_format("nonexistent_format")


def test_unknown_kv_format_raises():
    with pytest.raises(KeyError, match="Unknown KV cache format"):
        get_kv_format("nonexistent_kv_format")


def test_format_registry_has_core_formats():
    """FORMAT_REGISTRY includes all core format names."""
    expected = {"fp8", "nvfp4", "int8", "int4", "mxfp8", "mxfp6", "mxfp4"}
    assert expected.issubset(set(FORMAT_REGISTRY.keys()))


def test_kv_format_registry_has_core_formats():
    """KV_FORMAT_REGISTRY includes core KV format names."""
    expected = {"fp8", "nvfp4"}
    assert expected.issubset(set(KV_FORMAT_REGISTRY.keys()))


def test_get_format_returns_weight_and_activation():
    """get_format returns dict with 'weight' and 'activation' keys."""
    fmt = get_format("fp8")
    assert "weight" in fmt
    assert "activation" in fmt
    assert "num_bits" in fmt["weight"]


def test_get_kv_format_returns_quantizer_config():
    """get_kv_format returns dict with quantizer pattern keys."""
    kv = get_kv_format("fp8")
    assert isinstance(kv, dict)
    assert len(kv) > 0
