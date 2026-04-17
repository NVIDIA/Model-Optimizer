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
from unittest.mock import MagicMock, patch

import pytest
import torch

from modelopt.torch.sparsity.attention_sparsity.methods.triton_skip_softmax import (
    TritonSkipSoftmaxMethod,
)

# Module paths used by _set_triton_backends / _clear_triton_backends.
_DIFF_MOD_PATH = "modelopt.torch.sparsity.attention_sparsity.kernels.diffusers_triton_attention"
_LTX_MOD_PATH = "modelopt.torch.sparsity.attention_sparsity.kernels.ltx_triton_attention"


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


class TestSetClearTritonBackends:
    """Test _set_triton_backends / _clear_triton_backends import-fallback paths."""

    def test_set_backends_calls_both(self):
        """When both diffusers and LTX backends exist, both are configured."""
        m = TritonSkipSoftmaxMethod()
        with (
            patch(f"{_DIFF_MOD_PATH}.set_triton_skip_softmax_config", MagicMock()) as set_diff,
            patch(f"{_LTX_MOD_PATH}.set_ltx_triton_context", MagicMock()) as set_ltx,
        ):
            m._set_triton_backends(threshold=0.1)
            set_diff.assert_called_once_with(threshold=0.1)
            set_ltx.assert_called_once()

    def test_clear_backends_calls_both(self):
        """When both diffusers and LTX backends exist, both are cleared."""
        m = TritonSkipSoftmaxMethod()
        with (
            patch(f"{_DIFF_MOD_PATH}.clear_triton_skip_softmax_config", MagicMock()) as clear_diff,
            patch(f"{_LTX_MOD_PATH}.clear_ltx_triton_context", MagicMock()) as clear_ltx,
        ):
            m._clear_triton_backends()
            clear_diff.assert_called_once()
            clear_ltx.assert_called_once()


class TestInferenceContextManager:
    """Exercise _triton_inference_context's raw/scale/static branches."""

    def _get_module(self):
        return type("M", (), {"_apply_skip_softmax": False})()

    def _mock_backends(self):
        """Patch both backends so context managers don't touch real kernels."""
        return (
            patch(f"{_DIFF_MOD_PATH}.set_triton_skip_softmax_config", MagicMock()),
            patch(f"{_DIFF_MOD_PATH}.clear_triton_skip_softmax_config", MagicMock()),
            patch(f"{_LTX_MOD_PATH}.set_ltx_triton_context", MagicMock()),
            patch(f"{_LTX_MOD_PATH}.clear_ltx_triton_context", MagicMock()),
            patch(
                f"{_DIFF_MOD_PATH}.get_triton_attention_backend",
                MagicMock(side_effect=RuntimeError("no diffusers backend")),
            ),
        )

    def test_raw_threshold_branch(self):
        """Raw threshold takes precedence over scale_factor."""
        m = TritonSkipSoftmaxMethod({"skip_softmax_raw_threshold": -4.0})
        module = self._get_module()
        p0, p1, p2, p3, p4 = self._mock_backends()
        with p0, p1, p2, p3, p4:
            with m._triton_inference_context(module):
                assert module._apply_skip_softmax is True
            assert module._apply_skip_softmax is False

    def test_scale_factor_branch(self):
        """Calibrated scale_factor used when raw is None."""
        m = TritonSkipSoftmaxMethod()
        m.calibration_params = {"prefill": {"a": 2.0, "b": 3.0}}
        m.target_sparse_ratio = {"prefill": 0.5}
        module = self._get_module()
        p0, p1, p2, p3, p4 = self._mock_backends()
        with p0, p1, p2, p3, p4:
            with m._triton_inference_context(module):
                assert module._apply_skip_softmax is True
            assert module._apply_skip_softmax is False

    def test_static_threshold_branch(self):
        """Static threshold used when no raw/scale present."""
        m = TritonSkipSoftmaxMethod({"skip_softmax_threshold": 0.05})
        module = self._get_module()
        p0, p1, p2, p3, p4 = self._mock_backends()
        with p0, p1, p2, p3, p4, m._triton_inference_context(module):
            pass

    def test_measure_sparsity_path(self):
        """measure_sparsity=True triggers _collect_sparsity_counters on exit."""
        m = TritonSkipSoftmaxMethod({"skip_softmax_threshold": 0.05})
        m.enable_measure_sparsity(True)
        module = self._get_module()
        p0, p1, p2, p3, p4 = self._mock_backends()
        with (
            p0,
            p1,
            p2,
            p3,
            p4,
            patch(
                f"{_DIFF_MOD_PATH}.get_sparsity_counters",
                MagicMock(return_value=(100, 30)),
            ),
            m._triton_inference_context(module),
        ):
            pass
        # After the context, counters should be accumulated
        assert m._sparsity_total == 100
        assert m._sparsity_skipped == 30


class TestCalibrationContextManager:
    """Exercise _triton_calibration_context and _collect_calibration_stats."""

    def test_calibration_ctx_with_diffusers_counters(self):
        """Calibration context populates module._last_stats from diffusers counters."""
        m = TritonSkipSoftmaxMethod()
        m._calibration_mode = True
        m._threshold_trials = [0.01, 0.1, 0.5]
        module = type("M", (), {"_apply_skip_softmax": False, "_last_stats": None})()

        # counters: 3 thresholds, [total, skipped]
        fake_counters = torch.tensor([[100, 10], [100, 50], [100, 90]], dtype=torch.int64)

        with (
            patch(f"{_DIFF_MOD_PATH}.set_triton_skip_softmax_config", MagicMock()),
            patch(f"{_DIFF_MOD_PATH}.clear_triton_skip_softmax_config", MagicMock()),
            patch(f"{_LTX_MOD_PATH}.set_ltx_triton_context", MagicMock()),
            patch(f"{_LTX_MOD_PATH}.clear_ltx_triton_context", MagicMock()),
            patch(
                f"{_DIFF_MOD_PATH}.get_calibration_counters",
                MagicMock(return_value=fake_counters),
            ),
            patch(f"{_DIFF_MOD_PATH}.get_calibration_seq_k", MagicMock(return_value=4096)),
            patch(
                f"{_DIFF_MOD_PATH}.get_triton_attention_backend",
                MagicMock(side_effect=RuntimeError("no backend")),
            ),
            m._triton_calibration_context(module),
        ):
            pass

        assert module._last_stats is not None
        assert module._last_stats["phase"] == "prefill"
        assert module._last_stats["sample_length"] == 4096
        assert len(module._last_stats["sparsity"]) == 3
        # sparsity = skipped/total for each threshold
        assert module._last_stats["sparsity"][0] == pytest.approx(0.1)
        assert module._last_stats["sparsity"][1] == pytest.approx(0.5)
        assert module._last_stats["sparsity"][2] == pytest.approx(0.9)

    def test_collect_stats_no_counters(self):
        """_collect_calibration_stats is a no-op when no counters present."""
        m = TritonSkipSoftmaxMethod()
        m._threshold_trials = [0.01]
        module = type("M", (), {"_last_stats": None})()

        with (
            patch(f"{_DIFF_MOD_PATH}.get_calibration_counters", MagicMock(return_value=None)),
            patch(f"{_LTX_MOD_PATH}.get_calibration_counters", MagicMock(return_value=None)),
        ):
            m._collect_calibration_stats(module)
            # Should not have set _last_stats because no counters available
            assert module._last_stats is None

    def test_collect_stats_no_threshold_trials(self):
        """_collect_calibration_stats short-circuits when threshold_trials is None."""
        m = TritonSkipSoftmaxMethod()
        m._threshold_trials = None  # Not set
        module = type("M", (), {"_last_stats": None})()

        with patch(
            f"{_DIFF_MOD_PATH}.get_calibration_counters",
            MagicMock(return_value=torch.tensor([[100, 50]], dtype=torch.int64)),
        ):
            m._collect_calibration_stats(module)
            assert module._last_stats is None


class TestDiffusersBackendContext:
    """Test _get_diffusers_backend_context import-fallback."""

    def test_context_fallback_when_unavailable(self):
        """Context should yield even if diffusers attention backend unavailable."""
        with (
            patch(
                f"{_DIFF_MOD_PATH}.get_triton_attention_backend",
                MagicMock(side_effect=RuntimeError("not registered")),
            ),
            TritonSkipSoftmaxMethod._get_diffusers_backend_context(),
        ):
            pass  # Should not raise
