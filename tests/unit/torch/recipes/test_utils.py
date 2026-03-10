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

from modelopt.torch.recipes.utils import make_serializable


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
