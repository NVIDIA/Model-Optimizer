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

"""Unit tests for TritonSkipSoftmaxMethod (no GPU required)."""

import math
import warnings

import pytest
import torch

from modelopt.torch.sparsity.attention_sparsity.methods.triton_skip_softmax import (
    TritonSkipSoftmaxMethod,
)


class TestInit:
    def test_default_config(self):
        m = TritonSkipSoftmaxMethod()
        assert m.skip_softmax_threshold == 0.1
        assert m.skip_softmax_raw_threshold is None
        assert m._threshold_trials is None
        assert m._measure_sparsity is False

    def test_custom_config(self):
        m = TritonSkipSoftmaxMethod(
            {"skip_softmax_threshold": 0.05, "skip_softmax_raw_threshold": -3.0}
        )
        assert m.skip_softmax_threshold == 0.05
        assert m.skip_softmax_raw_threshold == -3.0

    def test_name(self):
        assert TritonSkipSoftmaxMethod().name == "triton_skip_softmax"


class TestCalculateSparsity:
    def test_returns_all_ones_mask(self):
        m = TritonSkipSoftmaxMethod()
        scores = torch.randn(2, 4, 8, 8)
        mask, stats = m.calculate_sparsity(scores)
        assert mask.shape == scores.shape
        assert mask.all()
        assert stats == {}


class TestApplySparsity:
    def test_raises_not_implemented(self):
        m = TritonSkipSoftmaxMethod()
        with pytest.raises(NotImplementedError, match="Triton kernel"):
            m.apply_sparsity(torch.randn(2, 2))


class TestGetScaleFactor:
    def test_uncalibrated_returns_none(self):
        m = TritonSkipSoftmaxMethod()
        assert m._get_scale_factor() is None

    def test_no_target_returns_none(self):
        m = TritonSkipSoftmaxMethod()
        m.calibration_params = {"prefill": {"a": 1.0, "b": 5.0}}
        m.target_sparse_ratio = None
        assert m._get_scale_factor() is None

    def test_calibrated_computation(self):
        m = TritonSkipSoftmaxMethod()
        m.calibration_params = {"prefill": {"a": 2.0, "b": 3.0}}
        m.target_sparse_ratio = {"prefill": 0.5}
        expected = 2.0 * math.exp(3.0 * 0.5)
        assert m._get_scale_factor() == pytest.approx(expected)

    def test_zero_a_returns_none(self):
        m = TritonSkipSoftmaxMethod()
        m.calibration_params = {"prefill": {"a": 0, "b": 5.0}}
        m.target_sparse_ratio = {"prefill": 0.5}
        assert m._get_scale_factor() is None

    def test_zero_b_returns_none(self):
        m = TritonSkipSoftmaxMethod()
        m.calibration_params = {"prefill": {"a": 1.0, "b": 0}}
        m.target_sparse_ratio = {"prefill": 0.5}
        assert m._get_scale_factor() is None

    def test_warns_below_min_observed(self):
        m = TritonSkipSoftmaxMethod()
        m.calibration_params = {
            "prefill": {
                "a": 1.0,
                "b": 5.0,
                "min_observed_sparsity": 0.3,
                "max_observed_sparsity": 0.8,
            }
        }
        m.target_sparse_ratio = {"prefill": 0.1}
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = m._get_scale_factor()
            assert result is not None
            assert len(w) == 1
            assert "below the minimum" in str(w[0].message)

    def test_warns_above_max_observed(self):
        m = TritonSkipSoftmaxMethod()
        m.calibration_params = {
            "prefill": {
                "a": 1.0,
                "b": 5.0,
                "min_observed_sparsity": 0.3,
                "max_observed_sparsity": 0.8,
            }
        }
        m.target_sparse_ratio = {"prefill": 0.95}
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = m._get_scale_factor()
            assert result is not None
            assert len(w) == 1
            assert "above the maximum" in str(w[0].message)


class TestGetThresholdInfo:
    def test_static_threshold(self):
        m = TritonSkipSoftmaxMethod({"skip_softmax_threshold": 0.05})
        info = m.get_threshold_info()
        assert info["type"] == "static"
        assert info["value"] == 0.05

    def test_calibrated_threshold(self):
        m = TritonSkipSoftmaxMethod()
        m.calibration_params = {"prefill": {"a": 2.0, "b": 3.0}}
        m.target_sparse_ratio = {"prefill": 0.5}
        info = m.get_threshold_info()
        assert info["type"] == "dynamic_calibrated"
        assert "scale_factor" in info


class TestSparsityMeasurement:
    def test_enable_disable(self):
        m = TritonSkipSoftmaxMethod()
        assert m._measure_sparsity is False
        m.enable_measure_sparsity(True)
        assert m._measure_sparsity is True
        m.enable_measure_sparsity(False)
        assert m._measure_sparsity is False

    def test_reset_counters(self):
        m = TritonSkipSoftmaxMethod()
        m._sparsity_total = 100
        m._sparsity_skipped = 50
        m.reset_sparsity_counters()
        assert m._sparsity_total == 0
        assert m._sparsity_skipped == 0

    def test_get_counters(self):
        m = TritonSkipSoftmaxMethod()
        m._sparsity_total = 200
        m._sparsity_skipped = 80
        total, skipped = m.get_sparsity_counters()
        assert total == 200
        assert skipped == 80


class TestGetSparseContext:
    def test_inference_mode_selected(self):
        m = TritonSkipSoftmaxMethod()
        m._calibration_mode = False
        module = type("M", (), {"_apply_skip_softmax": False})()
        ctx = m.get_sparse_context(module)
        # Should return the inference context (a generator-based context manager)
        assert hasattr(ctx, "__enter__")

    def test_calibration_mode_selected(self):
        m = TritonSkipSoftmaxMethod()
        m._calibration_mode = True
        m._threshold_trials = [0.01, 0.1]
        module = type("M", (), {"_apply_skip_softmax": False, "_last_stats": None})()
        ctx = m.get_sparse_context(module)
        assert hasattr(ctx, "__enter__")

    def test_calibration_mode_without_trials_falls_back_to_inference(self):
        m = TritonSkipSoftmaxMethod()
        m._calibration_mode = True
        m._threshold_trials = None  # No trials = falls back to inference
        module = type("M", (), {"_apply_skip_softmax": False})()
        ctx = m.get_sparse_context(module)
        assert hasattr(ctx, "__enter__")


class TestCollectCalibrationStats:
    """Defensive null-guards in _collect_calibration_stats (no GPU required).

    The happy-path (calibration context populating ``module._last_stats`` from
    real counters) is exercised by GPU tests in
    ``tests/gpu/torch/sparsity/attention_sparsity/``.
    """

    def test_no_counters_is_noop(self):
        """Skips writing stats when neither backend has counters."""
        m = TritonSkipSoftmaxMethod()
        m._threshold_trials = [0.01]
        module = type("M", (), {"_last_stats": None})()
        m._collect_calibration_stats(module)
        assert module._last_stats is None

    def test_no_threshold_trials_is_noop(self):
        """Skips writing stats when threshold_trials was never set."""
        m = TritonSkipSoftmaxMethod()
        m._threshold_trials = None
        module = type("M", (), {"_last_stats": None})()
        m._collect_calibration_stats(module)
        assert module._last_stats is None
