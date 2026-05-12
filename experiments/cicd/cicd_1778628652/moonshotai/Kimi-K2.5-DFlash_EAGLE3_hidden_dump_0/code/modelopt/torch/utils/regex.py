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

"""Utilities for pattern matching on module and layer names."""

import fnmatch
from collections.abc import Callable, Iterable

__all__ = [
    "matches_pattern",
]


def matches_pattern(
    name: str,
    patterns: str | Callable[[str], bool] | Iterable[str | Callable[[str], bool]] | None,
    *,
    allow_callable: bool = True,
) -> bool:
    """Check if a name matches any of the given patterns.

    This utility function checks if a given name (e.g., module name, layer name)
    matches any pattern in a collection of patterns. Patterns can be:
    - String wildcards (using fnmatch syntax, e.g., "*.attention.*")
    - Callable predicates that take a name and return bool
    - None (matches everything)

    Args:
        name: The name to check (e.g., "model.layer1.attention.weight")
        patterns: A single pattern, iterable of patterns, or None.
                 If None, returns True (matches everything).
        allow_callable: If False, raises TypeError when encountering callable patterns.
                       Useful for contexts where only string patterns are allowed.

    Returns:
        True if the name matches any pattern, False otherwise.

    Raises:
        TypeError: If pattern type is unsupported or if callable patterns are
                  provided when allow_callable=False.

    Examples:
        >>> matches_pattern("model.attention.query", "*.attention.*")
        True
        >>> matches_pattern("model.mlp.linear", ["*.attention.*", "*.mlp.*"])
        True
        >>> matches_pattern("model.layer1", lambda x: "layer" in x)
        True
        >>> matches_pattern("anything", None)
        True
    """
    if patterns is None:
        return True

    if isinstance(patterns, (str, bytes)):
        patterns_iter: Iterable[str | Callable[[str], bool]] = (patterns,)
    elif callable(patterns):
        if not allow_callable:
            raise TypeError("Callable patterns are not supported in this context.")
        patterns_iter = (patterns,)
    elif isinstance(patterns, Iterable):
        patterns_iter = tuple(patterns)
    else:
        raise TypeError(f"Unsupported pattern type: {type(patterns)}")

    for pattern in patterns_iter:
        if isinstance(pattern, (str, bytes)):
            if fnmatch.fnmatch(name, pattern):
                return True
        elif callable(pattern):
            if not allow_callable:
                raise TypeError("Callable patterns are not supported in this context.")
            if pattern(name):
                return True
        else:
            raise TypeError(f"Unsupported pattern type: {type(pattern)}")

    return False
