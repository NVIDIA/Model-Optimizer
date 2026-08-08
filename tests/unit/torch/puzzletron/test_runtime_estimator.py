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

import pytest

from modelopt.torch.puzzletron.block_config import AttentionConfig, BlockConfig, FFNConfig
from modelopt.torch.puzzletron.subblock_stats.calc_runtime_stats import (
    _assigned_runtime_shard_indices,
)
from modelopt.torch.puzzletron.subblock_stats.runtime_estimator import (
    candidate_slope,
    effective_repeat_count,
    fixed_intercept,
    homogeneous_layout,
    median_measurement,
    scaffolded_layout,
)
from modelopt.torch.puzzletron.subblock_stats.runtime_vllm import RuntimeMeasurement


def test_effective_repeat_rounds_up_to_pp_multiple():
    assert effective_repeat_count(3, 2) == 4
    assert effective_repeat_count(8, 4) == 8


def test_effective_repeat_rejects_non_positive_values():
    with pytest.raises(ValueError, match="positive"):
        effective_repeat_count(0, 2)
    with pytest.raises(ValueError, match="positive"):
        effective_repeat_count(4, 0)


def test_homogeneous_layout_contains_only_candidate():
    candidate = BlockConfig(subblock_configs=(FFNConfig(intermediate_size=16),))

    assert homogeneous_layout(candidate, repeat_count=4) == (candidate,) * 4


def test_scaffolded_layout_places_one_scaffold_in_each_pp_chunk():
    scaffold = BlockConfig(subblock_configs=(AttentionConfig(num_query_heads=8, num_kv_heads=2),))
    candidate = BlockConfig(subblock_configs=(FFNConfig(intermediate_size=16),))

    assert scaffolded_layout(candidate, scaffold, repeat_count=4, pp_size=2) == (
        scaffold,
        candidate,
        candidate,
        scaffold,
        candidate,
        candidate,
    )


def test_scaffolded_layout_requires_balanced_pp_chunks():
    scaffold = BlockConfig(subblock_configs=(AttentionConfig(),))
    candidate = BlockConfig(subblock_configs=(FFNConfig(),))

    with pytest.raises(ValueError, match="divisible"):
        scaffolded_layout(candidate, scaffold, repeat_count=3, pp_size=2)


def test_slope_and_intercept_are_component_wise():
    short = RuntimeMeasurement(total_ms=14.0, prefill_ms=6.0)
    long = RuntimeMeasurement(total_ms=22.0, prefill_ms=10.0)

    assert candidate_slope(short, long, 4) == RuntimeMeasurement(2.0, 1.0)
    assert fixed_intercept(short, long) == RuntimeMeasurement(6.0, 2.0)


def test_median_measurement_is_component_wise():
    result = median_measurement(
        [
            RuntimeMeasurement(total_ms=5.0, prefill_ms=3.0),
            RuntimeMeasurement(total_ms=1.0, prefill_ms=1.0),
            RuntimeMeasurement(total_ms=3.0, prefill_ms=2.0),
        ]
    )

    assert result == RuntimeMeasurement(total_ms=3.0, prefill_ms=2.0)


def test_runtime_sharding_keeps_paired_measurements_together():
    keys = [("spec", index) for index in range(6)]
    ordered_items = [(key, None) for key in keys]
    pairs = [(keys[0], keys[1]), (keys[2], keys[3]), (keys[4], keys[5])]

    assert _assigned_runtime_shard_indices(
        ordered_items, shard_count=2, shard_index=0, measurement_pairs=pairs
    ) == [0, 1, 4, 5]
    assert _assigned_runtime_shard_indices(
        ordered_items, shard_count=2, shard_index=1, measurement_pairs=pairs
    ) == [2, 3]
