# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import math

import pytest

from modelopt.torch.utils.list import list_closest_to_median, stats, val2list, val2tuple


def test_list_closest_to_median_odd_length():
    assert list_closest_to_median([1, 2, 3, 4, 5]) == 3
    assert list_closest_to_median([10, 1, 7]) == 7  # median of [1, 7, 10] is 7


def test_list_closest_to_median_even_length_returns_first_closest():
    # median of [1, 2, 3, 4] is 2.5; 2 and 3 are equally close, first index wins
    assert list_closest_to_median([1, 2, 3, 4]) == 2


def test_list_closest_to_median_singleton_and_floats():
    assert list_closest_to_median([5]) == 5
    assert list_closest_to_median([0.1, 0.9, 0.5]) == 0.5


def test_val2list_scalar_repeated():
    assert val2list(3) == [3]
    assert val2list(3, repeat_time=4) == [3, 3, 3, 3]
    assert val2list(None, repeat_time=2) == [None, None]


def test_val2list_sequences_passed_through():
    # repeat_time is ignored for list/tuple inputs
    assert val2list([1, 2], repeat_time=5) == [1, 2]
    assert val2list((1, 2)) == [1, 2]


def test_val2tuple_scalar_expansion():
    assert val2tuple(3) == (3,)
    assert val2tuple(3, min_len=3) == (3, 3, 3)


def test_val2tuple_repeats_element_at_idx_repeat():
    # default idx_repeat=-1 repeats the last element (inserted before it)
    assert val2tuple([1, 2], min_len=4) == (1, 2, 2, 2)
    # idx_repeat=0 repeats the first element
    assert val2tuple([1, 2], min_len=4, idx_repeat=0) == (1, 1, 1, 2)


def test_val2tuple_no_padding_needed():
    assert val2tuple([1, 2, 3], min_len=2) == (1, 2, 3)
    assert val2tuple([], min_len=3) == ()  # empty input stays empty


def test_stats_exact_values():
    result = stats([1.0, 2.0, 3.0, 4.0])
    assert set(result.keys()) == {"min", "max", "avg", "std"}
    assert result["min"] == 1.0
    assert result["max"] == 4.0
    assert result["avg"] == 2.5
    assert result["std"] == pytest.approx(math.sqrt(1.25))


def test_stats_single_element():
    result = stats([2.0])
    assert result["min"] == result["max"] == result["avg"] == 2.0
    assert result["std"] == 0.0


def test_stats_empty_returns_empty_dict():
    assert stats([]) == {}
