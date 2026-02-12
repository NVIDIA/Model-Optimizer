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

"""Unit tests for sparse attention calibration."""

import pytest

pytest.importorskip("transformers")

import numpy as np
from _test_utils.torch.sparsity.sparse_attention_common import SimpleAttentionModel
from pydantic import ValidationError

from modelopt.torch.sparsity.attention_sparsity import sparsify
from modelopt.torch.sparsity.attention_sparsity.calibration import (
    DynamicThresholdCalibrator,
    RulerDatasetBuilder,
)
from modelopt.torch.sparsity.attention_sparsity.calibration.calibrate import (
    _extract_calibration_config,
    calibrate_sparse_attention,
)
from modelopt.torch.sparsity.attention_sparsity.calibration.ruler_dataset import (
    _generate_target_lengths,
)
from modelopt.torch.sparsity.attention_sparsity.config import CalibrationConfig
from modelopt.torch.sparsity.attention_sparsity.sparse_attention import SparseAttentionModule


class TestLengthGeneration:
    """Test automatic target length generation."""

    def test_generate_target_lengths_default(self):
        """Test default 4 bins generation."""
        lengths = _generate_target_lengths(32768, num_length_bins=4)
        assert lengths == [32768, 16384, 8192, 4096]

    def test_generate_target_lengths_stops_at_minimum(self):
        """Test generation stops at minimum threshold."""
        lengths = _generate_target_lengths(2048, num_length_bins=4)
        assert lengths == [2048, 1024]  # Stops at 1024


class TestRulerDatasetBuilder:
    """Test RULER dataset generation without requiring real tokenizers."""

    def test_builder_initialization(self):
        """Test that builder initializes correctly."""
        builder = RulerDatasetBuilder(
            samples=12,
            max_seqlen=2048,  # Generates: [2048, 1024]
            tokenizer_name_or_path="gpt2",
            seed=42,
        )

        assert builder.total_samples == 12
        assert builder.max_seqlen == 2048
        assert builder.target_lengths == [2048, 1024]
        assert builder.samples_per_length == [6, 6]  # Evenly distributed
        assert len(builder.subtasks) == 6  # All RULER_TASKS
        assert builder.seed == 42

    def test_builder_initialization_invalid_config(self):
        """Test that builder raises error for invalid inputs."""
        # Test invalid samples
        with pytest.raises(ValueError, match="samples must be positive"):
            RulerDatasetBuilder(
                samples=0,
                max_seqlen=2048,
                tokenizer_name_or_path="gpt2",
            )

        # Test max_seqlen below minimum
        with pytest.raises(ValueError, match="max_seqlen must be >= 1024"):
            RulerDatasetBuilder(
                samples=4,
                max_seqlen=512,  # Below minimum
                tokenizer_name_or_path="gpt2",
            )

    def test_dataset_generation_minimal(self):
        """Test generating small dataset."""
        builder = RulerDatasetBuilder(
            samples=12,  # 6 tasks x 2 lengths = need 12 for 1 per task per length
            max_seqlen=2048,  # Generates: [2048, 1024]
            tokenizer_name_or_path="gpt2",
            seed=42,
        )

        dataset = builder.build_calibration_dataset()

        # Should generate 12 samples (6 tasks x 1 sample per task x 2 lengths)
        assert len(dataset) == 12
        assert all(isinstance(sample, dict) for sample in dataset)

    def test_dataset_structure(self):
        """Test that dataset has correct structure."""
        builder = RulerDatasetBuilder(
            samples=6,  # Need at least 6 (1 per task)
            max_seqlen=1024,  # Generates: [1024]
            tokenizer_name_or_path="gpt2",
            seed=42,
        )

        dataset = builder.build_calibration_dataset()
        sample = dataset[0]

        # Check required fields
        assert "input" in sample
        assert "length" in sample
        assert "task" in sample
        assert "target_length" in sample

        # Check field types
        assert isinstance(sample["input"], str)
        assert isinstance(sample["length"], int)
        assert isinstance(sample["task"], str)
        assert sample["length"] > 0

    def test_uneven_sample_distribution(self):
        """Test that samples are distributed evenly (remainder dropped)."""
        builder = RulerDatasetBuilder(
            samples=50,  # 50 samples across 4 lengths
            max_seqlen=8192,  # Generates: [8192, 4096, 2048, 1024]
            tokenizer_name_or_path="gpt2",
            seed=42,
        )

        # Even distribution: 50//4 = 12 per length
        assert builder.total_samples == 50
        assert builder.target_lengths == [8192, 4096, 2048, 1024]
        assert builder.samples_per_length == [12, 12, 12, 12]
        assert sum(builder.samples_per_length) == 48  # 2 samples dropped (remainder)

        # Actual generated samples: 12//6=2 per task, 4 lengths, 6 tasks
        # Total: 2 x 6 x 4 = 48
        dataset = builder.build_calibration_dataset()
        assert len(dataset) == 48


class TestDynamicThresholdCalibrator:
    """Test calibration algorithm correctness (regression calculations)."""

    def test_regression_calculation_synthetic(self):
        """Test 'a' parameter calculation with synthetic data."""
        # Create synthetic optimal pairs
        # If threshold = a / length, then with perfect data:
        # length=1000, threshold=10 => a=10000
        # length=2000, threshold=5  => a=10000
        optimal_pairs = [
            {"length": 1000, "optimal_threshold": 10.0, "achieved_sparsity": 0.5},
            {"length": 2000, "optimal_threshold": 5.0, "achieved_sparsity": 0.5},
            {"length": 4000, "optimal_threshold": 2.5, "achieved_sparsity": 0.5},
        ]

        # Manual regression calculation
        lengths = np.array([p["length"] for p in optimal_pairs])
        thresholds = np.array([p["optimal_threshold"] for p in optimal_pairs])

        x = 1.0 / lengths
        y = thresholds

        # Calculate 'a' using least squares
        a_parameter = np.sum(x * y) / np.sum(x**2)

        # Should be close to 10000
        assert 9500 < a_parameter < 10500

        # Test individual 'a' values
        a_per_sample = y * lengths
        assert np.allclose(a_per_sample, 10000, rtol=0.05)


class TestCalibrationIntegration:
    """Test end-to-end calibration without GPU."""

    def test_calibration_disabled(self):
        """Test that no calibration occurs when disabled."""
        model = SimpleAttentionModel(hidden_size=64, num_heads=4)

        config = {
            "sparse_cfg": {
                "*attention*": {
                    "method": "flash_skip_softmax",
                    "threshold": {"prefill": 1e-3, "decode": 1e-4},
                    "br": 64,
                    "bc": 64,
                    "enable": True,
                }
            },
        }

        # No forward_loop needed when calibration disabled
        sparse_model = sparsify(model, config)

        # Check that sparse attention is applied but not calibrated
        has_sparse = any(isinstance(m, SparseAttentionModule) for m in sparse_model.modules())
        assert has_sparse

        # Check that no calibration is set
        for module in sparse_model.modules():
            if isinstance(module, SparseAttentionModule):
                method = module._sparse_method_instance
                assert not getattr(method, "calibration_params", None)

    def test_sparsify_with_calibration_requires_forward_loop(self):
        """Test that calibration requires forward_loop or proper model config."""
        model = SimpleAttentionModel(hidden_size=64, num_heads=4)

        config = {
            "sparse_cfg": {
                "calibration": {
                    "target_sparse_ratio": {"prefill": 0.5, "decode": 0.5},
                    "samples": 4,
                    "max_seqlen": 1024,
                },
                "*attention*": {
                    "method": "flash_skip_softmax",
                    "threshold": {"prefill": 1e-3, "decode": 1e-4},
                    "br": 64,
                    "bc": 64,
                    "enable": True,
                },
            },
        }

        # Without forward_loop and without model.config._name_or_path, should raise ValueError
        with pytest.raises(ValueError, match="Could not load tokenizer"):
            sparsify(model, config, forward_loop=None)

    def test_calibration_config_validation(self):
        """Test CalibrationConfig validation."""
        # Valid config
        config = CalibrationConfig(
            target_sparse_ratio={"prefill": 0.5, "decode": 0.5},
            samples=48,
            max_seqlen=32768,
        )
        assert config.target_sparse_ratio == {"prefill": 0.5, "decode": 0.5}
        assert config.samples == 48
        assert config.max_seqlen == 32768

        # Invalid target_sparse_ratio (> 1.0)
        with pytest.raises(ValueError, match="target_sparse_ratio.*must be between 0.0 and 1.0"):
            CalibrationConfig(
                target_sparse_ratio={"prefill": 1.5, "decode": 0.5}, samples=48, max_seqlen=32768
            )

        # Invalid target_sparse_ratio (< 0.0)
        with pytest.raises(ValueError, match="target_sparse_ratio.*must be between 0.0 and 1.0"):
            CalibrationConfig(
                target_sparse_ratio={"prefill": -0.1, "decode": 0.5}, samples=48, max_seqlen=32768
            )

        # Invalid samples
        with pytest.raises(ValueError, match="samples must be positive"):
            CalibrationConfig(
                target_sparse_ratio={"prefill": 0.5, "decode": 0.5}, samples=0, max_seqlen=32768
            )

        # Invalid max_seqlen
        with pytest.raises(ValueError, match="max_seqlen must be >= 1024"):
            CalibrationConfig(
                target_sparse_ratio={"prefill": 0.5, "decode": 0.5}, samples=48, max_seqlen=512
            )

    def test_threshold_trials_validation(self):
        """Test threshold_trials validation."""
        # Valid custom threshold_trials
        config = CalibrationConfig(
            target_sparse_ratio={"prefill": 0.5, "decode": 0.5},
            threshold_trials=[1e-5, 1e-4, 1e-3, 1e-2],
        )
        assert config.threshold_trials == [1e-5, 1e-4, 1e-3, 1e-2]

        # None (use defaults)
        config_default = CalibrationConfig(target_sparse_ratio={"prefill": 0.5, "decode": 0.5})
        assert config_default.threshold_trials is None

        # Invalid: empty list
        with pytest.raises(ValueError, match="threshold_trials must not be empty"):
            CalibrationConfig(threshold_trials=[])

        # Invalid: threshold out of range (>= 1.0)
        with pytest.raises(ValueError, match="must be in range"):
            CalibrationConfig(threshold_trials=[1e-4, 1.0])

        # Invalid: threshold out of range (<= 0)
        with pytest.raises(ValueError, match="must be in range"):
            CalibrationConfig(threshold_trials=[1e-4, 0])

        # Invalid: not a list (Pydantic raises ValidationError, not ValueError)
        with pytest.raises(ValidationError, match="Input should be a valid list"):
            CalibrationConfig(threshold_trials=1e-4)


class TestDynamicThresholdCalibratorMethods:
    """Test error paths and edge cases of DynamicThresholdCalibrator."""

    def test_calibrate_no_sparse_modules(self):
        """Test calibrate raises error when no sparse modules found."""
        model = SimpleAttentionModel(hidden_size=64, num_heads=4)

        # Don't apply sparse attention
        calibrator = DynamicThresholdCalibrator(
            threshold_trials=[0.001, 0.01],
        )

        def dummy_forward_loop(m):
            pass

        with pytest.raises(ValueError, match="No sparse attention modules found"):
            calibrator.calibrate(model, dummy_forward_loop, "prefill")

    def test_calibrate_empty_stats(self):
        """Test calibrate handles empty stats gracefully."""
        model = SimpleAttentionModel(hidden_size=64, num_heads=4)
        config = {
            "sparse_cfg": {
                "*attention*": {
                    "method": "flash_skip_softmax",
                    "threshold": {"prefill": 0.1, "decode": 0.1},
                    "br": 64,
                    "bc": 64,
                    "enable": True,
                }
            },
        }
        sparse_model = sparsify(model, config)

        calibrator = DynamicThresholdCalibrator(
            threshold_trials=[0.001],  # Only one threshold for speed
        )

        # Forward loop that doesn't generate any stats
        def empty_forward_loop(m):
            pass

        # Should return empty dict
        result = calibrator.calibrate(sparse_model, empty_forward_loop, "prefill")
        assert result == {}


class TestCalibrateFunction:
    """Test calibrate_sparse_attention function."""

    def test_calibrate_no_config(self):
        """Test calibration when config has no calibration section."""
        model = SimpleAttentionModel(hidden_size=64, num_heads=4)

        # Config without calibration
        config = {
            "sparse_cfg": {
                "*attention*": {
                    "method": "flash_skip_softmax",
                    "threshold": {"prefill": 0.1, "decode": 0.1},
                    "br": 64,
                    "bc": 64,
                    "enable": True,
                }
            },
        }

        # Should return empty dict when no calibration config
        result = calibrate_sparse_attention(model, config)

        assert result == {}

    def test_extract_calibration_config(self):
        """Test _extract_calibration_config function."""
        # Config with calibration
        config = {
            "sparse_cfg": {
                "calibration": {
                    "target_sparse_ratio": {"prefill": 0.3, "decode": 0.3},
                    "samples": 12,
                    "max_seqlen": 2048,
                },
                "*attn*": {
                    "method": "flash_skip_softmax",
                },
            },
        }

        calib_config = _extract_calibration_config(config)

        assert calib_config is not None
        assert calib_config.target_sparse_ratio == {"prefill": 0.3, "decode": 0.3}
        assert calib_config.samples == 12
        assert calib_config.max_seqlen == 2048

    def test_extract_calibration_config_none(self):
        """Test _extract_calibration_config when no calibration."""
        # Config without calibration
        config = {
            "sparse_cfg": {
                "*attn*": {
                    "method": "flash_skip_softmax",
                    "threshold": {"prefill": 0.1, "decode": 0.1},
                }
            },
        }

        calib_config = _extract_calibration_config(config)

        assert calib_config is None
