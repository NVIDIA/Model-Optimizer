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

"""Utils for operating on lists."""

from typing import Any

import numpy as np

__all__ = ["list_closest_to_median", "stats", "val2list", "val2tuple"]


def list_closest_to_median(x: list) -> Any:
    """Return element from list that's closest to list mean."""
    median = np.median(x)
    diff = [abs(elem - median) for elem in x]
    return x[diff.index(min(diff))]


def val2list(val: list | tuple | Any, repeat_time: int = 1) -> list:
    """Repeat `val` for `repeat_time` times and return the list or val if list/tuple."""
    if isinstance(val, (list, tuple)):
        return list(val)
    return [val for _ in range(repeat_time)]


def val2tuple(val: list | tuple | Any, min_len: int = 1, idx_repeat: int = -1) -> tuple:
    """Return tuple with min_len by repeating element at idx_repeat."""
    # convert to list first
    val = val2list(val)

    # repeat elements if necessary
    if len(val) > 0:
        val[idx_repeat:idx_repeat] = [val[idx_repeat] for _ in range(min_len - len(val))]

    return tuple(val)


def stats(vals: list[float]) -> dict[str, float]:
    """Compute min, max, avg, std of vals."""
    stats = {"min": np.min, "max": np.max, "avg": np.mean, "std": np.std}
    return {name: fn(vals) for name, fn in stats.items()} if vals else {}
