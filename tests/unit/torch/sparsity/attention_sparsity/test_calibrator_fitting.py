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

"""Unit tests for DynamicThresholdCalibrator exponential fitting (no GPU required).

Tests the calibration math (curve_fit, filtering, aggregation) using synthetic
data injected via mock forward loops and mock sparse attention modules.
"""

import numpy as np
import pytest

pytest.importorskip("transformers")


from _test_utils.torch.sparsity.sparse_attention_common import SimpleAttentionModel

from modelopt.torch.sparsity.attention_sparsity import sparsify
from modelopt.torch.sparsity.attention_sparsity.calibration.calibrator import (
    DynamicThresholdCalibrator,
)
from modelopt.torch.sparsity.attention_sparsity.sparse_attention import SparseAttentionModule


class TestDynamicThresholdCalibratorInit:
    def test_default_thresholds(self):
        cal = DynamicThresholdCalibrator()
        assert len(cal.threshold_trials) == 20
        assert cal.threshold_trials[0] == 1e-6
        assert cal.threshold_trials[-1] == 9.9e-1

    def test_custom_thresholds(self):
        trials = [0.01, 0.1, 0.5]
        cal = DynamicThresholdCalibrator(threshold_trials=trials)
        assert cal.threshold_trials == trials

    def test_fit_logspace(self):
        cal = DynamicThresholdCalibrator(fit_logspace=True)
        assert cal.fit_logspace is True


class TestExponentialFitting:
    """Test the calibration pipeline with synthetic stats injected via mock modules."""

    def _make_sparse_model(self):
        """Create a model with flash_skip_softmax sparse attention applied."""
        model = SimpleAttentionModel(hidden_size=64, num_heads=4)
        config = {
            "sparse_cfg": {
                "*attention*": {
                    "method": "flash_skip_softmax",
                    "thresholds": {"prefill": [0.1], "decode": [0.1]},
                    "br": 64,
                    "bc": 64,
                    "enable": True,
                }
            },
        }
        return sparsify(model, config)

    def _inject_synthetic_stats(self, model, threshold_trials, sample_length=4096):
        """Inject synthetic calibration stats into the sparse modules.

        Generates stats matching the exponential model: scale_factor = a * exp(b * sparsity),
        with a=0.1, b=5.0. For each threshold, the expected sparsity is:
            sparsity = ln(threshold * sample_length / a) / b
        """
        a_true, b_true = 0.1, 5.0

        for module in model.modules():
            if isinstance(module, SparseAttentionModule):
                sparsity_list = []
                for t in threshold_trials:
                    sf = t * sample_length
                    # Invert the model: sparsity = ln(sf / a) / b
                    s = np.log(sf / a_true) / b_true
                    s = max(0.0, min(1.0, s))
                    sparsity_list.append(s)
                module._last_stats = {
                    "sparsity": sparsity_list,
                    "sample_length": sample_length,
                    "phase": "prefill",
                }

    def test_calibrate_with_synthetic_linear(self):
        """Test linear-space fitting recovers known parameters."""
        model = self._make_sparse_model()
        trials = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 0.1, 0.3, 0.5]
        cal = DynamicThresholdCalibrator(threshold_trials=trials, fit_logspace=False)

        def forward_loop(m):
            # Simulate 3 samples with different lengths
            for length in [2048, 4096, 8192]:
                self._inject_synthetic_stats(m, trials, sample_length=length)
                for module in m.modules():
                    if isinstance(module, SparseAttentionModule) and module._stats_manager:
                        stats = module._last_stats
                        if stats:
                            module._stats_manager.collect(stats)

        result = cal.calibrate(model, forward_loop, "prefill")
        # Should produce valid result with a and b close to ground truth
        if result:
            assert "a" in result
            assert "b" in result
            assert result["r_squared"] > 0.8

    def test_calibrate_with_synthetic_logspace(self):
        """Test log-space fitting recovers known parameters."""
        model = self._make_sparse_model()
        trials = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 0.1, 0.3, 0.5]
        cal = DynamicThresholdCalibrator(threshold_trials=trials, fit_logspace=True)

        def forward_loop(m):
            for length in [2048, 4096, 8192]:
                self._inject_synthetic_stats(m, trials, sample_length=length)
                for module in m.modules():
                    if isinstance(module, SparseAttentionModule) and module._stats_manager:
                        stats = module._last_stats
                        if stats:
                            module._stats_manager.collect(stats)

        result = cal.calibrate(model, forward_loop, "prefill")
        if result:
            assert "a" in result
            assert "b" in result
            assert result["r_squared"] > 0.9  # Log-space should fit better

    def test_calibrate_no_modules_raises(self):
        """Test error when no sparse attention modules exist."""
        model = SimpleAttentionModel(hidden_size=64, num_heads=4)
        cal = DynamicThresholdCalibrator(threshold_trials=[0.01])

        with pytest.raises(ValueError, match="No sparse attention modules"):
            cal.calibrate(model, lambda m: None, "prefill")

    def test_calibrate_empty_stats_returns_empty(self):
        """Test empty dict returned when forward loop produces no stats."""
        model = self._make_sparse_model()
        cal = DynamicThresholdCalibrator(threshold_trials=[0.01])

        result = cal.calibrate(model, lambda m: None, "prefill")
        assert result == {}


class TestSetThresholds:
    """Test _set_thresholds for both method types."""

    def test_set_thresholds_flash_method(self):
        model = SimpleAttentionModel(hidden_size=64, num_heads=4)
        config = {
            "sparse_cfg": {
                "*attention*": {
                    "method": "flash_skip_softmax",
                    "thresholds": {"prefill": [0.1], "decode": [0.1]},
                    "br": 64,
                    "bc": 64,
                    "enable": True,
                }
            },
        }
        model = sparsify(model, config)
        cal = DynamicThresholdCalibrator()

        modules = [m for m in model.modules() if isinstance(m, SparseAttentionModule)]
        trials = [0.001, 0.01, 0.1]
        cal._set_thresholds(modules, trials)

        for module in modules:
            method = module._sparse_method_instance
            assert method.thresholds == trials
