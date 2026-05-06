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

"""Unit tests for get_distributed_modules_ownership in bypass_utils.py."""

import pytest

from modelopt.torch.puzzletron.bypass_distillation.bypass_utils import (
    get_distributed_modules_ownership,
)


def test_single_gpu_all_to_rank_0():
    """With world_size=1, all 4 modules should be assigned to rank 0."""
    ownership = get_distributed_modules_ownership(module_count=4, world_size=1)
    assert ownership == [0, 0, 0, 0]


def test_even_distribution():
    """With world_size=2 and 4 modules, each rank should own exactly 2 modules."""
    ownership = get_distributed_modules_ownership(module_count=4, world_size=2)
    assert ownership.count(0) == 2
    assert ownership.count(1) == 2
    assert len(ownership) == 4


def test_uneven_distribution():
    """With world_size=2 and 3 modules, rank 0 should own 2 and rank 1 should own 1."""
    ownership = get_distributed_modules_ownership(module_count=3, world_size=2)
    assert ownership.count(0) == 2
    assert ownership.count(1) == 1
    assert len(ownership) == 3


@pytest.mark.parametrize(
    ("module_count", "world_size"),
    [
        (1, 1),
        (4, 1),
        (4, 2),
        (4, 4),
        (7, 3),
        (10, 4),
        (1, 2),
    ],
)
def test_total_equals_module_count(module_count, world_size):
    """The length of the ownership list must always equal module_count."""
    ownership = get_distributed_modules_ownership(
        module_count=module_count, world_size=world_size
    )
    assert len(ownership) == module_count


def test_consecutive_ownership():
    """Each rank should own a contiguous block of indices (no interleaving)."""
    ownership = get_distributed_modules_ownership(module_count=7, world_size=3)
    # Verify that once we see a new rank, we never see the previous rank again.
    seen_ranks = set()
    prev_rank = ownership[0]
    seen_ranks.add(prev_rank)
    for rank in ownership[1:]:
        if rank != prev_rank:
            assert rank not in seen_ranks, (
                f"Rank {rank} appears non-consecutively in ownership list: {ownership}"
            )
            seen_ranks.add(rank)
            prev_rank = rank


def test_single_module():
    """With world_size=2 and only 1 module, rank 0 should be the sole owner."""
    ownership = get_distributed_modules_ownership(module_count=1, world_size=2)
    assert ownership == [0]
    assert len(ownership) == 1
