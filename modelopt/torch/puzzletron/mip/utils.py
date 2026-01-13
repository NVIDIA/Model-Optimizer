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

"""Utility functions for MIP optimization."""

from typing import Any


class InfeasibleError(Exception):
    """Exception raised when optimization problem is infeasible."""

    pass


def sort_replacements(layer_replacements: list[dict]) -> list[dict]:
    """Sort layer replacements by parent layer indices.

    Args:
        layer_replacements: List of replacement dictionaries containing "parent_layer_indices"

    Returns:
        Sorted list of replacements
    """
    return sorted(layer_replacements, key=lambda replacement: replacement["parent_layer_indices"])


def get_nested_key(dictionary: dict[str, Any], nested_key: str) -> Any:
    """Access nested dictionary values using dot notation.

    If nested_key is "a.b.c" returns dictionary["a"]["b"]["c"]

    Args:
        dictionary: Dictionary to access
        nested_key: Dot-separated key path (e.g., "a.b.c")

    Returns:
        Value at the nested key location
    """
    value = dictionary
    for key in nested_key.split("."):
        value = value[key]
    return value


def consecutive_ngrams(l: int, n: int) -> list[list[int]]:
    """Generate all consecutive n-grams from range(l).

    Splits range(l) into all consecutive n-grams.

    Args:
        l: Length of the range
        n: Size of each n-gram

    Returns:
        List of consecutive n-grams

    Example:
        consecutive_ngrams(7, 2) = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6]]
    """
    ngrams = []
    for i in range(l - n + 1):
        ngrams.append(list(range(i, i + n)))
    return ngrams
