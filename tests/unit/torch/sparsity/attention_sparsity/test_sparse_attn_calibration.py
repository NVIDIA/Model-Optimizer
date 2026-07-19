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

"""Unit tests for vLLM-free skip-softmax calibration helpers (no vLLM needed)."""

import math
from types import SimpleNamespace

import pytest

from modelopt.torch.sparsity.attention_sparsity.calibration.calibrator import (
    DynamicThresholdCalibrator,
)
from modelopt.torch.sparsity.attention_sparsity.plugins.sparse_attn_calibration import (
    DEFAULT_THRESHOLD_TRIALS,
    build_sparse_attention_config,
    fit_from_counts,
    merge_count_records,
    merge_phase_counts,
    split_records_by_phase,
    stats_from_counts,
)
from modelopt.torch.sparsity.attention_sparsity.plugins.sparse_attn_config import (
    load_from_checkpoint_metadata,
)


def _record(phase, length, totals, skipped):
    return {
        "phase": phase,
        "sample_length": length,
        "total_tiles": list(totals),
        "skipped_tiles": list(skipped),
    }


class TestCountMerging:
    def test_split_records_by_phase_preserves_order(self):
        records = [
            _record("prefill", 100, [4], [1]),
            _record("decode", 101, [2], [0]),
            _record("prefill", 200, [8], [3]),
        ]
        split = split_records_by_phase(records)
        assert [r["sample_length"] for r in split["prefill"]] == [100, 200]
        assert [r["sample_length"] for r in split["decode"]] == [101]

    def test_merge_sums_counts_elementwise(self):
        layer_a = [_record("prefill", 128, [10, 10], [2, 4])]
        layer_b = [_record("prefill", 128, [10, 10], [1, 3])]
        merged = merge_count_records([layer_a, layer_b])
        assert merged == [{"sample_length": 128, "total_tiles": [20, 20], "skipped_tiles": [3, 7]}]

    def test_merge_rejects_ragged_sources(self):
        long = [_record("prefill", 1, [1], [0]), _record("prefill", 2, [1], [1])]
        short = [_record("prefill", 1, [1], [1])]
        with pytest.raises(ValueError, match="disagree on sample count"):
            merge_count_records([long, short])

    def test_merge_rejects_misaligned_sample_lengths(self):
        with pytest.raises(ValueError, match="Misaligned calibration records"):
            merge_count_records(
                [[_record("prefill", 100, [1], [0])], [_record("prefill", 200, [1], [0])]]
            )

    def test_merge_rejects_threshold_width_mismatch(self):
        with pytest.raises(ValueError, match="threshold-vector widths"):
            merge_count_records(
                [[_record("prefill", 100, [1, 2], [0, 1])], [_record("prefill", 100, [1], [0])]]
            )

    def test_merge_phase_counts_rejects_rank_phase_mismatch(self):
        rank0 = {"prefill": [_record("prefill", 64, [5], [1])]}
        rank1 = {"prefill": []}
        with pytest.raises(ValueError, match="recorded no 'prefill' samples"):
            merge_phase_counts([rank0, rank1])

    def test_merge_phase_counts_across_ranks(self):
        rank0 = {"prefill": [_record("prefill", 64, [5], [1])], "decode": []}
        rank1 = {"prefill": [_record("prefill", 64, [5], [2])]}
        merged = merge_phase_counts([rank0, rank1])
        assert merged["prefill"][0]["total_tiles"] == [10]
        assert merged["prefill"][0]["skipped_tiles"] == [3]
        assert merged["decode"] == []

    def test_stats_from_counts_forms_ratios_after_merge(self):
        stats = stats_from_counts(
            [{"sample_length": 64, "total_tiles": [8, 0], "skipped_tiles": [2, 0]}]
        )
        assert stats == [{"sample_length": 64, "sparsity": [0.25, 0.0]}]


class TestFitFromCounts:
    def test_fit_recovers_synthetic_exponential(self):
        a_true, b_true = 5.0, 8.0
        trials = DEFAULT_THRESHOLD_TRIALS

        def counts(length, total):
            sparsity = [
                min(0.95, max(0.0, math.log(max(t * length, 1e-9) / a_true) / b_true))
                for t in trials
            ]
            return {
                "sample_length": length,
                "total_tiles": [total] * len(trials),
                "skipped_tiles": [int(s * total) for s in sparsity],
            }

        per_phase = {"prefill": [counts(length, 4000) for length in (2048, 4096, 8192, 16384)]}
        params = fit_from_counts(per_phase, trials)
        assert abs(params["prefill"]["a"] - a_true) / a_true < 0.3
        assert abs(params["prefill"]["b"] - b_true) / b_true < 0.15
        assert 0.0 <= params["prefill"]["min_observed_sparsity"] <= 1.0

    def test_empty_phase_produces_no_fit(self):
        assert fit_from_counts({"decode": []}, DEFAULT_THRESHOLD_TRIALS) == {}


class TestCalibrateFromStats:
    def _stats(self, trials):
        a_true, b_true = 3.0, 9.0
        stats = []
        for length in (1024, 2048, 4096, 8192):
            sparsity = [
                min(0.95, max(0.0, math.log(max(t * length, 1e-9) / a_true) / b_true))
                for t in trials
            ]
            stats.append({"sample_length": length, "sparsity": sparsity})
        return stats

    def test_linear_fit_reports_fit_logspace_false(self):
        calibrator = DynamicThresholdCalibrator(threshold_trials=list(DEFAULT_THRESHOLD_TRIALS))
        result = calibrator.calibrate_from_stats(self._stats(DEFAULT_THRESHOLD_TRIALS), "prefill")
        assert result["fit_logspace"] is False
        assert "log_a" not in result
        assert len(result["per_sample_sparsity"]) == 4

    def test_logspace_fit_preserves_log_a(self):
        calibrator = DynamicThresholdCalibrator(
            threshold_trials=list(DEFAULT_THRESHOLD_TRIALS), fit_logspace=True
        )
        result = calibrator.calibrate_from_stats(self._stats(DEFAULT_THRESHOLD_TRIALS), "prefill")
        assert result["fit_logspace"] is True
        assert math.isclose(math.exp(result["log_a"]), result["a"], rel_tol=1e-9)


class TestBuildSparseAttentionConfig:
    _PARAMS = {"prefill": {"a": 7.9, "b": 8.6}, "decode": {"a": 0.12, "b": 9.8}}

    def test_canonical_schema(self):
        config = build_sparse_attention_config(self._PARAMS, 0.4)
        group = config["config_groups"]["group_0"]
        assert group["algorithm"] == "skip_softmax"
        assert group["threshold_scale_factor"]["prefill"] == {"a": 7.9, "b": 8.6}
        assert group["threshold_scale_factor"]["formula"] == "a * exp(b * target_sparsity)"
        assert group["target_sparsity"] == {"prefill": 0.4, "decode": 0.4}
        assert config["producer"]["name"] == "modelopt"

    def test_preserves_nm_groups_and_replaces_old_skip_group(self):
        existing = {
            "config_groups": {
                "group_0": {"algorithm": "sparse_softmax", "sparsity_n": 2, "sparsity_m": 4},
                "group_1": {
                    "algorithm": "skip_softmax",
                    "threshold_scale_factor": {"prefill": {"a": 1.0, "b": 1.0}},
                },
            }
        }
        config = build_sparse_attention_config(self._PARAMS, 0.5, existing_config=existing)
        groups = config["config_groups"]
        assert len(groups) == 2
        assert groups["group_0"]["algorithm"] == "skip_softmax"
        assert groups["group_0"]["threshold_scale_factor"]["prefill"]["a"] == 7.9
        assert groups["group_1"]["algorithm"] == "sparse_softmax"
        assert groups["group_1"]["sparsity_n"] == 2

    def test_round_trips_through_serving_loader(self):
        config = build_sparse_attention_config(self._PARAMS, {"prefill": 0.5, "decode": 0.3})
        hf_config = SimpleNamespace(sparse_attention_config=config)
        loaded = load_from_checkpoint_metadata(hf_config)
        assert loaded is not None
        sparse_cfg, preset = loaded
        assert preset == "CHECKPOINT_CALIBRATED_SOFTMAX_SKIP"
        layer_cfg = sparse_cfg["sparse_cfg"]["*attn*"]
        assert layer_cfg["method"] == "triton_skip_softmax"
        assert layer_cfg["threshold_scale_factor"]["decode"] == {"a": 0.12, "b": 9.8}
        assert layer_cfg["target_sparse_ratio"] == {"prefill": 0.5, "decode": 0.3}

    def test_preserves_legacy_toplevel_sparse_softmax(self):
        existing = {
            "config_groups": {
                "group_0": {"algorithm": "sparse_softmax", "sparsity_n": 2, "sparsity_m": 4}
            },
            "sparse_softmax": {"sparsity_n": 1, "sparsity_m": 4, "dense_recent_tokens": 128},
        }
        config = build_sparse_attention_config(self._PARAMS, 0.5, existing_config=existing)
        # The serving loader reads the legacy top-level key ahead of group params.
        assert config["sparse_softmax"] == existing["sparse_softmax"]
        loaded = load_from_checkpoint_metadata(SimpleNamespace(sparse_attention_config=config))
        assert loaded is not None
        layer_cfg = loaded[0]["sparse_cfg"]["*attn*"]
        assert layer_cfg["sparsity_n"] == 1
        assert layer_cfg["dense_recent_tokens"] == 128

    def test_round_trip_with_preserved_nm_group_activates_both(self):
        existing = {
            "config_groups": {
                "group_0": {"algorithm": "sparse_softmax", "sparsity_n": 2, "sparsity_m": 4}
            }
        }
        config = build_sparse_attention_config(self._PARAMS, 0.5, existing_config=existing)
        loaded = load_from_checkpoint_metadata(SimpleNamespace(sparse_attention_config=config))
        assert loaded is not None
        sparse_cfg, preset = loaded
        assert preset == "CHECKPOINT_CALIBRATED_SOFTMAX_SKIP_SPARSE_SOFTMAX"
        layer_cfg = sparse_cfg["sparse_cfg"]["*attn*"]
        assert layer_cfg["sparsity_n"] == 2
        assert "threshold_scale_factor" in layer_cfg
