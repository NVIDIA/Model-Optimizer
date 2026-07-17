# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from modelopt.torch.puzzletron.block_config import AttentionConfig, BlockConfig, FFNConfig
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
    scaffold = BlockConfig(
        subblock_configs=(AttentionConfig(num_query_heads=8, num_kv_heads=2),)
    )
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
