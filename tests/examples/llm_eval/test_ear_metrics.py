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

"""Offline scalar and frozen-position checks for the probability-overlap metric."""

import importlib
import math
from pathlib import Path

import pytest
import torch


@pytest.fixture
def ear_metrics(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / "examples" / "llm_eval"))
    return importlib.import_module("ear_metrics")


@pytest.mark.parametrize("chunk_size", [1, 2, 32])
def test_overlap_matches_scalar_full_normalization_and_shared_ids(ear_metrics, chunk_size):
    p = [0.5, 0.3, 0.2]
    q = [0.1, 0.2, 0.7]
    reference = torch.tensor([p, q, p], dtype=torch.float64).log()
    candidate = torch.tensor([q, p, p], dtype=torch.float64).log()
    actual = ear_metrics.per_position_ear(reference, candidate, (1, 2, 3), chunk_size)
    expected_full = sum(min(a, b) for a, b in zip(p, q))
    assert actual["full"].tolist() == pytest.approx([expected_full, expected_full, 1], abs=1e-15)
    assert actual["full"][0] == pytest.approx(1 - sum(abs(a - b) for a, b in zip(p, q)) / 2)
    assert actual["2"].tolist() == pytest.approx([0.3, 0.4, 0.8], abs=1e-15)
    assert actual["1"].tolist() == pytest.approx([0.1, 0.2, 0.5], abs=1e-15)
    assert actual["3"].tolist() == pytest.approx(actual["full"].tolist(), abs=1e-15)
    assert actual["reference_mass_2"].tolist() == pytest.approx([0.8, 0.9, 0.8])
    assert actual["2"][2] == actual["reference_mass_2"][2]
    assert all(
        value.dtype == torch.float64 and value.device.type == "cpu" for value in actual.values()
    )


def test_fp64_normalization_retains_small_tail_and_extreme_logit_stability(ear_metrics):
    reference = torch.tensor([[0, -25], [10000, -10000]], dtype=torch.float64)
    candidate = torch.tensor([[-25, 0], [10000, -10000]], dtype=torch.float64)
    actual = ear_metrics.per_position_ear(reference, candidate, (1,))
    tail = math.exp(-25) / (1 + math.exp(-25))
    assert actual["full"].tolist() == pytest.approx([2 * tail, 1], rel=1e-15)
    assert actual["1"][0] == pytest.approx(tail, rel=1e-15)
    assert actual["reference_mass_1"][0] < 1


def test_logical_endpoints_and_both_aggregation_weightings(ear_metrics):
    first = ear_metrics.logical_continuation([3, 9, 4, 9, 0], {9, 10})
    second = ear_metrics.logical_continuation([10, 4, 0], {9, 10})
    assert first == [3, 9] and second == [10]
    assert ear_metrics.logical_continuation([3, 4, 5], set(), max_tokens=2) == [3, 4]
    rows = [
        {"id": "a", "values": {"full": [0.2, 0.6][: len(first)]}},
        {"id": "b", "values": {"full": [1.0][: len(second)]}},
    ]
    actual = ear_metrics.aggregate_ear(rows)
    assert actual["equal_prompt"]["full"] == pytest.approx(0.7)
    assert actual["native_flat"]["full"] == pytest.approx(0.6)
    assert actual["selected_positions"] == 3
    prefix = ear_metrics.aggregate_ear(rows, prompt_count=1, max_tokens=1)
    assert prefix["equal_prompt"]["full"] == 0.2
    assert prefix["prompt_count"] == 1 and prefix["max_tokens"] == 1


def test_spaced_counts_are_endpoint_inclusive_and_recomputed_after_length_cap(ear_metrics):
    rows = [{"id": 0, "values": {"full": [i / 10 for i in range(10)]}}]
    assert ear_metrics.selection_indices(10, 1, "spaced") == [0]
    assert ear_metrics.selection_indices(10, 4, "spaced") == [0, 3, 6, 9]
    assert ear_metrics.selection_indices(10, 5, "spaced") == [0, 2, 4, 6, 9]
    assert ear_metrics.selection_indices(3, 4, "spaced") == [0, 1, 2]
    actual = ear_metrics.aggregate_ear(rows, max_tokens=7, position_count=3, selection="spaced")
    assert actual["equal_prompt"]["full"] == pytest.approx(0.3)
    assert actual["selected_positions"] == 3


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_logits_and_scores_are_not_dropped(ear_metrics, bad):
    with pytest.raises(ValueError):
        ear_metrics.per_position_ear(torch.zeros(2, 3), torch.tensor([[0, bad, 0], [0, 0, 0]]))
    with pytest.raises(ValueError):
        ear_metrics.aggregate_ear([{"id": 0, "values": {"full": [0.5, bad]}}], max_tokens=1)


def test_invalid_configuration_and_incomplete_rows_fail(ear_metrics):
    logits = torch.zeros(1, 4)
    for ks in ((0,), (5,), (2, 2), (True,)):
        with pytest.raises(ValueError):
            ear_metrics.per_position_ear(logits, logits, ks)
    rows = [{"id": 0, "values": {"full": [0.5]}}]
    with pytest.raises(ValueError):
        ear_metrics.aggregate_ear(rows, prompt_count=2)
    with pytest.raises(ValueError):
        ear_metrics.aggregate_ear(rows + rows)
    with pytest.raises(ValueError):
        ear_metrics.aggregate_ear([{"id": 0, "values": {"full": [], "2": [0.5]}}])
    with pytest.raises(ValueError):
        ear_metrics.selection_indices(10, 0, "spaced")
