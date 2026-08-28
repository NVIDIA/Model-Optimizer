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

"""Unit tests for :mod:`modelopt.onnx.quantization.sensitivity.picker`."""

import logging

import pytest

from modelopt.onnx.quantization.sensitivity.picker import suggest_exclusion, summarize_exclusion


class TestCoverageMode:
    """Tests the ``at most X%`` semantic: cumulative KL never exceeds target."""

    def test_stops_before_crossing_target(self):
        scores = {"a": 4.0, "b": 3.0, "c": 2.0, "d": 1.0}
        assert suggest_exclusion(scores, coverage=0.5) == ["a"]

    def test_includes_second_when_it_fits(self):
        scores = {"a": 4.0, "b": 3.0, "c": 2.0, "d": 1.0}
        assert suggest_exclusion(scores, coverage=0.8) == ["a", "b"]

    @pytest.mark.parametrize(
        ("scores", "expected"),
        [
            pytest.param({"a": 1.0, "b": 2.0, "c": 3.0}, ["c", "b", "a"], id="ascending_input"),
            pytest.param(
                {"low": 0.1, "high": 0.9, "mid": 0.5}, ["high", "mid", "low"], id="unsorted_input"
            ),
        ],
    )
    def test_full_coverage_returns_all_sorted_by_score_desc(self, scores, expected):
        # coverage=1.0 -> everything fits and result is sorted by KL desc.
        assert suggest_exclusion(scores, coverage=1.0) == expected

    def test_top_node_alone_exceeds_target(self):
        scores = {"a": 5.0, "b": 3.0, "c": 2.0}
        assert suggest_exclusion(scores, coverage=0.2) == []

    @pytest.mark.parametrize(
        ("scores", "coverage"),
        [
            pytest.param({"a": 5.0, "b": 3.0}, 0.0, id="zero_target"),
            pytest.param({"a": 0.0, "b": 0.0}, 0.9, id="zero_total"),
            pytest.param({}, 0.9, id="empty_scores"),
        ],
    )
    def test_returns_empty_for_boundary_cases(self, scores, coverage):
        assert suggest_exclusion(scores, coverage=coverage) == []

    def test_max_nodes_caps_exclusion_set(self):
        scores = {chr(ord("a") + i): 10.0 - i for i in range(10)}
        assert suggest_exclusion(scores, coverage=1.0, max_nodes=3) == ["a", "b", "c"]

    def test_min_score_floor_stops_before_low_nodes(self):
        scores = {"hi_1": 5.0, "hi_2": 4.0, "trivial_1": 0.001, "trivial_2": 0.0001}
        assert suggest_exclusion(scores, coverage=1.0, min_score_floor=0.01) == ["hi_1", "hi_2"]

    def test_vit_like_distribution_undershoots_cleanly(self):
        # Mimics ViT-tiny's distribution: 15 nodes at KL ~3.8-6.7, sharp drop to ~3.05
        # at ranks 16-17, then a long tail. Regression witness for the "at most X%" semantic.
        big = {f"top_{i}": 6.7 - i * 0.2 for i in range(15)}
        borderline = {"rank_16": 3.057, "rank_17": 3.055}
        tail = {f"tail_{i}": 0.5 - i * 0.02 for i in range(30)}
        scores = {**big, **borderline, **tail}
        total = sum(scores.values())
        result = suggest_exclusion(scores, coverage=0.90)
        assert sum(scores[n] for n in result) <= 0.90 * total
        assert len(result) < len(scores)


class TestThresholdMode:
    """Tests the absolute-KL cutoff semantic: exclude all nodes above threshold."""

    def test_picks_all_above_absolute_threshold(self):
        scores = {"a": 5.0, "b": 3.0, "c": 1.0, "d": 0.5, "e": 0.05}
        assert suggest_exclusion(scores, threshold=1.0) == ["a", "b"]

    def test_returns_sorted_by_kl_desc(self):
        scores = {"low_hit": 0.6, "high_hit": 0.9, "mid_hit": 0.75, "miss": 0.1}
        assert suggest_exclusion(scores, threshold=0.5) == ["high_hit", "mid_hit", "low_hit"]

    def test_boundary_score_is_excluded_from_set(self):
        # A score exactly at the threshold does NOT get excluded (strict >).
        scores = {"above": 0.11, "at": 0.10, "below": 0.09}
        assert suggest_exclusion(scores, threshold=0.10) == ["above"]

    def test_no_nodes_above_threshold_returns_empty(self):
        assert suggest_exclusion({"a": 0.01, "b": 0.005}, threshold=1.0) == []

    def test_threshold_overrides_coverage(self):
        scores = {"a": 5.0, "b": 3.0, "c": 2.98, "d": 2.0}
        assert suggest_exclusion(scores, coverage=0.99, threshold=2.5) == ["a", "b", "c"]

    def test_max_nodes_still_caps_threshold_mode(self):
        scores = {chr(ord("a") + i): 10.0 - i * 0.1 for i in range(10)}
        assert suggest_exclusion(scores, threshold=5.0, max_nodes=3) == ["a", "b", "c"]

    def test_min_score_floor_composes_with_threshold(self):
        scores = {"a": 5.0, "b": 0.5, "c": 0.3}
        assert suggest_exclusion(scores, threshold=0.1, min_score_floor=1.0) == ["a"]


class TestBlocks:
    """Tests the ``blocks=`` / ``block_agg=`` block-aware picker."""

    def test_first_match_wins_across_blocks_iteration_order(self):
        # 3 nodes; both group "a" and group "b" would match node "shared" via regex,
        # but "a" is listed first -> "shared" joins "a".
        scores = {"n_a": 5.0, "shared": 4.0, "n_b": 3.0}
        blocks = {
            "a": [r"^n_a$", r"^shared$"],
            "b": [r"^shared$", r"^n_b$"],
        }
        result = suggest_exclusion(scores, coverage=1.0, blocks=blocks, block_agg="sum")
        # Group "a" carries {n_a, shared} = 9.0; group "b" carries {n_b} = 3.0.
        # Both groups included at coverage=1.0.
        assert set(result) == {"n_a", "shared", "n_b"}

    def test_unmatched_nodes_become_singleton_groups(self):
        # "orphan" matches no group -> becomes its own singleton, ranked on its own score.
        scores = {"n_a1": 5.0, "n_a2": 4.0, "orphan": 3.0}
        blocks = {"a": [r"^n_a"]}
        # Sum aggregation: group "a" = 9.0, "orphan" = 3.0. Coverage=0.75 -> target 9.0.
        # Only group "a" fits (9.0 <= 9.0); orphan singleton (3.0) would push cumulative to 12.
        excluded = suggest_exclusion(scores, coverage=0.75, blocks=blocks, block_agg="sum")
        assert set(excluded) == {"n_a1", "n_a2"}

    def test_sum_aggregation(self):
        scores = {"a1": 1.0, "a2": 2.0, "b1": 4.0}
        blocks = {"a": [r"^a"], "b": [r"^b"]}
        # Sum: group a = 3.0, group b = 4.0. threshold=3.5 -> only b (>3.5) excluded.
        assert set(suggest_exclusion(scores, threshold=3.5, blocks=blocks, block_agg="sum")) == {
            "b1"
        }

    def test_max_aggregation(self):
        scores = {"a1": 1.0, "a2": 2.0, "b1": 4.0}
        blocks = {"a": [r"^a"], "b": [r"^b"]}
        # Max: group a = 2.0, group b = 4.0. threshold=3.5 -> only b excluded.
        assert set(suggest_exclusion(scores, threshold=3.5, blocks=blocks, block_agg="max")) == {
            "b1"
        }

    def test_mean_aggregation(self):
        scores = {"a1": 1.0, "a2": 5.0, "b1": 4.0}
        blocks = {"a": [r"^a"], "b": [r"^b"]}
        # Mean: group a = 3.0, group b = 4.0. threshold=3.5 -> only b excluded.
        assert set(suggest_exclusion(scores, threshold=3.5, blocks=blocks, block_agg="mean")) == {
            "b1"
        }

    def test_invalid_block_agg_raises(self):
        with pytest.raises(ValueError, match="block_agg"):
            suggest_exclusion(
                {"a": 1.0},
                coverage=1.0,
                blocks={"g": [r"^a$"]},
                block_agg="invalid",  # type: ignore[arg-type]
            )

    def test_return_value_is_union_of_member_nodes(self):
        # blocks= returns member nodes across selected groups, not group names.
        scores = {"blk0_n1": 5.0, "blk0_n2": 4.0, "blk1_n1": 1.0}
        blocks = {"blk0": [r"^blk0_"], "blk1": [r"^blk1_"]}
        excluded = suggest_exclusion(scores, threshold=0.5, blocks=blocks, block_agg="max")
        # Both groups have max > 0.5 -> both included -> all 3 member nodes in exclusion.
        assert set(excluded) == {"blk0_n1", "blk0_n2", "blk1_n1"}

    def test_threshold_max_and_coverage_sum_pick_equivalent_set(self):
        # Docs claim: threshold=0.1, block_agg="max" and coverage=1.0, max_nodes=<K>,
        # block_agg="sum" pick the same top-K groups on ViT-tiny-shaped data.
        # Synthetic: 6 blocks with max KL values decreasing; group max > 0.1 for the first 6.
        scores = {}
        for i in range(6):
            for j in range(3):
                scores[f"blk{i}_n{j}"] = (6 - i) * (1.0 if j == 0 else 0.1)
        # Add 4 low blocks below threshold
        for i in range(6, 10):
            for j in range(3):
                scores[f"blk{i}_n{j}"] = 0.01 * (10 - i)
        blocks = {f"blk{i}": [rf"^blk{i}_"] for i in range(10)}

        via_max = suggest_exclusion(scores, threshold=0.1, blocks=blocks, block_agg="max")
        via_sum = suggest_exclusion(
            scores, coverage=1.0, max_nodes=6, blocks=blocks, block_agg="sum"
        )
        assert set(via_max) == set(via_sum)


class TestNearTieWarning:
    """Warning fires when the cut-off between included and excluded is a near-tie."""

    def test_warning_fires_on_near_tied_cutoff(self, caplog):
        # b and c are near-tied (3.05/3.06 = 99.7%); coverage=0.75 cuts between them.
        scores = {"a": 6.0, "b": 3.06, "c": 3.05, "d": 0.1}
        with caplog.at_level(logging.WARNING, logger="modelopt.onnx"):
            suggest_exclusion(scores, coverage=0.75)
        assert "near-tie at the exclusion cut-off" in caplog.text

    def test_no_warning_when_cut_is_not_a_near_tie(self, caplog):
        scores = {"a": 6.0, "b": 3.0, "c": 0.1, "d": 0.05}
        with caplog.at_level(logging.WARNING, logger="modelopt.onnx"):
            suggest_exclusion(scores, coverage=0.75)
        assert "near-tie" not in caplog.text

    def test_warning_disabled_by_none(self, caplog):
        scores = {"a": 5.0, "b": 4.99, "c": 0.1}
        with caplog.at_level(logging.WARNING, logger="modelopt.onnx"):
            suggest_exclusion(scores, coverage=0.5, near_tie_ratio=None)
        assert "near-tie" not in caplog.text

    def test_threshold_mode_also_warns_on_near_tie(self, caplog):
        scores = {"a": 6.0, "b": 3.06, "c": 3.05, "d": 0.1}
        with caplog.at_level(logging.WARNING, logger="modelopt.onnx"):
            suggest_exclusion(scores, threshold=3.056)
        assert "near-tie" in caplog.text and "mode=threshold" in caplog.text


class TestSummarizeExclusion:
    def test_reports_coverage_pct_and_counts(self):
        scores = {"a": 4.0, "b": 3.0, "c": 2.0, "d": 1.0}
        summary = summarize_exclusion(scores, ["a", "b"])
        assert summary["num_excluded"] == 2
        assert summary["num_previously_quantized"] == 4
        assert summary["num_remaining_quantized"] == 2
        assert summary["coverage_pct"] == pytest.approx(70.0)
        assert summary["excluded_mass"] == pytest.approx(7.0)
        assert summary["total_mass"] == pytest.approx(10.0)

    def test_empty_scores_zero_coverage(self):
        summary = summarize_exclusion({}, [])
        assert summary["coverage_pct"] == 0.0
        assert summary["num_excluded"] == 0

    def test_missing_node_names_default_zero(self):
        scores = {"a": 5.0, "b": 5.0}
        summary = summarize_exclusion(scores, ["a", "unknown"])
        assert summary["excluded_mass"] == pytest.approx(5.0)
        assert summary["coverage_pct"] == pytest.approx(50.0)
        assert summary["num_excluded"] == 2
