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

"""Tests for recipes/utils.py."""

from modelopt.torch.recipes.utils import deep_merge, make_serializable


def test_make_serializable_tuples_to_lists():
    """Tuples and nested structures are converted to lists."""
    result = make_serializable({"a": (1, 2), "b": {"c": (3,)}, "d": [4, 5]})
    assert result == {"a": [1, 2], "b": {"c": [3]}, "d": [4, 5]}


def test_make_serializable_int_keys_to_strings():
    """Dict keys (including ints) are stringified."""
    result = make_serializable({-1: 16, "type": "dynamic"})
    assert result == {"-1": 16, "type": "dynamic"}


def test_make_serializable_primitives_passthrough():
    """Primitive types pass through unchanged."""
    assert make_serializable(42) == 42
    assert make_serializable("hello") == "hello"
    assert make_serializable(True) is True
    assert make_serializable(None) is None


def test_make_serializable_non_json_types():
    """Non-JSON types are converted to strings."""
    result = make_serializable({"key": {1, 2, 3}})
    assert isinstance(result["key"], str)


def test_deep_merge_simple():
    result = deep_merge({"a": 1, "b": 2}, {"b": 3, "c": 4})
    assert result == {"a": 1, "b": 3, "c": 4}


def test_deep_merge_nested():
    base = {"quant_cfg": {"default": {"enable": False}, "*weight*": {"num_bits": 8}}}
    override = {"quant_cfg": {"*weight*": {"axis": 0}}}
    result = deep_merge(base, override)
    assert result["quant_cfg"]["default"] == {"enable": False}
    assert result["quant_cfg"]["*weight*"] == {"num_bits": 8, "axis": 0}


def test_deep_merge_replaces_non_dict():
    result = deep_merge({"algorithm": "max"}, {"algorithm": "awq_lite"})
    assert result["algorithm"] == "awq_lite"


def test_deep_merge_no_mutation():
    base = {"a": {"b": 1}}
    override = {"a": {"c": 2}}
    result = deep_merge(base, override)
    assert "c" not in base["a"]
    assert result["a"] == {"b": 1, "c": 2}
