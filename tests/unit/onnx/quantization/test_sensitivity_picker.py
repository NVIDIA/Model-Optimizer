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

import pytest

from modelopt.onnx.quantization.sensitivity.picker import suggest_exclusion, summarize_exclusion


class TestCoverageMode:
    """Tests the ``at most X%`` semantic: cumulative KL never exceeds target."""

    def test_stops_before_crossing_target(self):
        # Total = 10. coverage=0.5 -> target 5. top-1 is 4 (fits), top-2 would
        # be 7 (crosses 5) -> stop at 1 node.
        scores = {"a": 4.0, "b": 3.0, "c": 2.0, "d": 1.0}
        assert suggest_exclusion(scores, coverage=0.5) == ["a"]

    def test_includes_second_when_it_fits(self):
        # Total = 10. coverage=0.8 -> target 8. top-1 (4) + top-2 (7) both fit,
        # top-3 would be 9 (crosses 8) -> stop at 2 nodes.
        scores = {"a": 4.0, "b": 3.0, "c": 2.0, "d": 1.0}
        assert suggest_exclusion(scores, coverage=0.8) == ["a", "b"]

    def test_full_coverage_returns_all_nodes(self):
        # coverage=1.0 -> target = total, everything fits exactly.
        scores = {"a": 1.0, "b": 2.0, "c": 3.0}
        result = suggest_exclusion(scores, coverage=1.0)
        assert set(result) == {"a", "b", "c"}

    def test_returns_sorted_by_kl_desc(self):
        scores = {"low": 0.1, "high": 0.9, "mid": 0.5}
        # coverage=1.0 -> everything fits, and result is sorted by KL desc.
        assert suggest_exclusion(scores, coverage=1.0) == ["high", "mid", "low"]

    def test_top_node_alone_exceeds_target(self):
        # Total = 10, coverage=0.2 -> target 2. Top node (5) alone exceeds
        # target, so nothing is included.
        scores = {"a": 5.0, "b": 3.0, "c": 2.0}
        assert suggest_exclusion(scores, coverage=0.2) == []

    def test_zero_target_returns_empty(self):
        scores = {"a": 5.0, "b": 3.0}
        assert suggest_exclusion(scores, coverage=0.0) == []

    def test_zero_total_returns_empty(self):
        assert suggest_exclusion({"a": 0.0, "b": 0.0}, coverage=0.9) == []

    def test_empty_scores_returns_empty(self):
        assert suggest_exclusion({}, coverage=0.9) == []

    def test_max_nodes_caps_exclusion_set(self):
        # 10 nodes at KL 10..1. Total = 55. coverage=1.0 would include all,
        # but max_nodes=3 caps at 3.
        scores = {chr(ord("a") + i): 10.0 - i for i in range(10)}
        assert suggest_exclusion(scores, coverage=1.0, max_nodes=3) == ["a", "b", "c"]

    def test_min_score_floor_stops_before_low_nodes(self):
        # Even at coverage=1.0, nodes below the floor are excluded from the set.
        scores = {"hi_1": 5.0, "hi_2": 4.0, "trivial_1": 0.001, "trivial_2": 0.0001}
        result = suggest_exclusion(scores, coverage=1.0, min_score_floor=0.01)
        assert result == ["hi_1", "hi_2"]

    def test_vit_like_distribution_undershoots_cleanly(self):
        # Mimics ViT-tiny's distribution: 15 nodes at KL ~3.8-6.7, then a
        # sharp drop to ~3.05 for ranks 16-17, then a long tail.
        big = {f"top_{i}": 6.7 - i * 0.2 for i in range(15)}  # ranks 1-15, KL ~6.7 down to ~3.9
        borderline = {"rank_16": 3.057, "rank_17": 3.055}
        tail = {f"tail_{i}": 0.5 - i * 0.02 for i in range(30)}
        scores = {**big, **borderline, **tail}
        total = sum(scores.values())
        result = suggest_exclusion(scores, coverage=0.90)
        excluded_mass = sum(scores[n] for n in result)
        # Actual coverage never exceeds requested.
        assert excluded_mass <= 0.90 * total
        # But should still capture most of the mass with fewer than the total.
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
        # coverage=0.99 would try to include most; threshold overrides.
        assert suggest_exclusion(scores, coverage=0.99, threshold=2.5) == ["a", "b", "c"]

    def test_max_nodes_still_caps_threshold_mode(self):
        # All 10 nodes have score > 5.0 but max_nodes=3 caps at 3.
        scores = {chr(ord("a") + i): 10.0 - i * 0.1 for i in range(10)}
        assert suggest_exclusion(scores, threshold=5.0, max_nodes=3) == ["a", "b", "c"]

    def test_min_score_floor_composes_with_threshold(self):
        # threshold=0.1 would normally include all three, but min_score_floor=1.0
        # short-circuits after "a" (b=0.5 is below the floor).
        scores = {"a": 5.0, "b": 0.5, "c": 0.3}
        assert suggest_exclusion(scores, threshold=0.1, min_score_floor=1.0) == ["a"]


class TestNearTieWarning:
    """Warning fires when the cut-off between included and excluded is a near-tie."""

    def test_warning_fires_on_near_tied_cutoff(self, caplog):
        # Ranks 16 and 17 are near-tied at KL 3.06 vs 3.05 (99.7% ratio); coverage=0.94
        # cuts between them.
        scores = {
            f"node_{i:02d}": kl
            for i, kl in enumerate(
                [
                    6.7,
                    5.7,
                    4.6,
                    4.3,
                    4.1,
                    4.1,
                    4.0,
                    4.0,
                    3.8,
                    3.8,
                    3.8,
                    3.8,
                    3.7,
                    3.7,
                    3.7,
                    3.06,
                    3.05,
                    0.8,
                    0.5,
                    0.1,
                ],
                1,
            )
        }
        import logging

        with caplog.at_level(logging.WARNING, logger="modelopt.onnx"):
            suggest_exclusion(scores, coverage=0.94)
        messages = [r.message for r in caplog.records]
        assert any("near-tie at the exclusion cut-off" in m for m in messages)

    def test_no_warning_when_cut_is_not_a_near_tie(self, caplog):
        # ViT-like distribution where coverage=0.75 cuts between very different KL values.
        scores = {
            f"node_{i:02d}": kl for i, kl in enumerate([6.7, 5.7, 4.6, 4.3, 4.1, 0.5, 0.2, 0.1], 1)
        }
        import logging

        with caplog.at_level(logging.WARNING, logger="modelopt.onnx"):
            suggest_exclusion(scores, coverage=0.75)
        messages = [r.message for r in caplog.records]
        assert not any("near-tie" in m for m in messages)

    def test_warning_disabled_by_none(self, caplog):
        # Setting near_tie_ratio=None disables the warning entirely.
        scores = {"a": 5.0, "b": 4.99, "c": 0.1}
        import logging

        with caplog.at_level(logging.WARNING, logger="modelopt.onnx"):
            suggest_exclusion(scores, coverage=0.5, near_tie_ratio=None)
        messages = [r.message for r in caplog.records]
        assert not any("near-tie" in m for m in messages)

    def test_threshold_mode_also_warns_on_near_tie(self, caplog):
        # threshold=3.056 cuts between KL 3.06 (above threshold) and 3.05 (below) -- near-tie.
        scores = {"a": 6.7, "b": 5.7, "c": 3.06, "d": 3.05, "e": 0.1}
        import logging

        with caplog.at_level(logging.WARNING, logger="modelopt.onnx"):
            suggest_exclusion(scores, threshold=3.056)
        messages = [r.message for r in caplog.records]
        assert any("near-tie" in m and "mode=threshold" in m for m in messages)


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
