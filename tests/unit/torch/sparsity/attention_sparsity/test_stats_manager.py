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

"""Unit tests for SparseAttentionStatsManager."""

import pytest

pytest.importorskip("transformers")

from modelopt.torch.sparsity.attention_sparsity.stats_manager import SparseAttentionStatsManager


class TestStatsManagerInitialization:
    """Test stats manager initialization."""

    def test_initialization_defaults(self):
        """Test default initialization."""
        manager = SparseAttentionStatsManager(module_name="test_module")

        assert manager.module_name == "test_module"
        assert manager.enabled is True
        assert manager.calibration_mode is False
        assert manager.aggregated_stats["total_calls"] == 0
        assert manager.aggregated_stats["total_blocks"] == 0
        assert manager.aggregated_stats["sparse_blocks"] == 0
        assert manager.per_sample_stats == []

    def test_initialization_disabled(self):
        """Test initialization with disabled stats."""
        manager = SparseAttentionStatsManager(module_name="test_module", enabled=False)

        assert manager.enabled is False
        assert manager.calibration_mode is False

    def test_initialization_custom_name(self):
        """Test initialization with custom module name."""
        manager = SparseAttentionStatsManager(module_name="custom.attention.module")

        assert manager.module_name == "custom.attention.module"


class TestStatsCollection:
    """Test statistics collection functionality."""

    def test_collect_stats_enabled(self):
        """Test collecting stats when enabled."""
        manager = SparseAttentionStatsManager(module_name="test", enabled=True)

        stats = {
            "sparsity": 0.5,
            "phase": "prefill",
            "total_blocks": 100,
            "sparse_blocks": 50,
            "sample_length": 1024,
        }

        manager.collect(stats)

        assert manager.aggregated_stats["total_calls"] == 1
        assert manager.aggregated_stats["total_blocks"] == 100
        assert manager.aggregated_stats["sparse_blocks"] == 50
        assert manager.aggregated_stats["phase_counts"]["prefill"] == 1
        assert manager.aggregated_stats["phase_counts"]["decode"] == 0

    def test_collect_stats_disabled(self):
        """Test that collect() is no-op when disabled."""
        manager = SparseAttentionStatsManager(module_name="test", enabled=False)

        stats = {
            "sparsity": 0.5,
            "phase": "prefill",
            "total_blocks": 100,
            "sparse_blocks": 50,
        }

        manager.collect(stats)

        # Should remain at initial values
        assert manager.aggregated_stats["total_calls"] == 0
        assert manager.aggregated_stats["total_blocks"] == 0
        assert manager.aggregated_stats["sparse_blocks"] == 0

    def test_collect_multiple_calls(self):
        """Test accumulation over multiple collect calls."""
        manager = SparseAttentionStatsManager(module_name="test", enabled=True)

        # Collect multiple times
        for i in range(5):
            stats = {
                "sparsity": 0.5,
                "phase": "prefill",
                "total_blocks": 100,
                "sparse_blocks": 50,
            }
            manager.collect(stats)

        assert manager.aggregated_stats["total_calls"] == 5
        assert manager.aggregated_stats["total_blocks"] == 500
        assert manager.aggregated_stats["sparse_blocks"] == 250
        assert manager.aggregated_stats["phase_counts"]["prefill"] == 5

    def test_collect_different_phases(self):
        """Test phase counting."""
        manager = SparseAttentionStatsManager(module_name="test", enabled=True)

        # Collect prefill stats
        manager.collect({"phase": "prefill", "total_blocks": 100, "sparse_blocks": 50})
        manager.collect({"phase": "prefill", "total_blocks": 100, "sparse_blocks": 50})

        # Collect decode stats
        manager.collect({"phase": "decode", "total_blocks": 10, "sparse_blocks": 5})

        assert manager.aggregated_stats["phase_counts"]["prefill"] == 2
        assert manager.aggregated_stats["phase_counts"]["decode"] == 1
        assert manager.aggregated_stats["phase_counts"]["unknown"] == 0


class TestCalibrationMode:
    """Test calibration mode functionality."""

    def test_calibration_mode_per_sample_collection(self):
        """Test that calibration mode stores per-sample stats."""
        manager = SparseAttentionStatsManager(module_name="test", enabled=True)

        # Enable calibration mode
        manager.set_calibration_mode(enabled=True)

        stats = {
            "sparsity": 0.5,
            "phase": "prefill",
            "total_blocks": 100,
            "sparse_blocks": 50,
            "sample_length": 1024,
        }

        manager.collect(stats)

        # Should store in per_sample_stats
        assert len(manager.per_sample_stats) == 1
        assert manager.per_sample_stats[0]["module"] == "test"
        assert manager.per_sample_stats[0]["sparsity"] == 0.5
        assert manager.per_sample_stats[0]["sample_length"] == 1024
        assert manager.per_sample_stats[0]["phase"] == "prefill"

    def test_calibration_mode_off(self):
        """Test that per-sample stats are not collected when calibration mode is off."""
        manager = SparseAttentionStatsManager(module_name="test", enabled=True)
        # Calibration mode is off by default

        stats = {"sparsity": 0.5, "phase": "prefill", "total_blocks": 100, "sparse_blocks": 50}

        manager.collect(stats)

        # Should NOT store in per_sample_stats
        assert len(manager.per_sample_stats) == 0

        # But should still aggregate
        assert manager.aggregated_stats["total_calls"] == 1

    def test_set_calibration_mode_with_reset(self):
        """Test set_calibration_mode with reset_history=True."""
        manager = SparseAttentionStatsManager(module_name="test", enabled=True)

        # Collect some stats in calibration mode
        manager.set_calibration_mode(enabled=True)
        manager.collect(
            {
                "sparsity": 0.5,
                "phase": "prefill",
                "total_blocks": 100,
                "sparse_blocks": 50,
                "sample_length": 1024,
            }
        )
        assert len(manager.per_sample_stats) == 1

        # Re-enable with reset
        manager.set_calibration_mode(enabled=True, reset_history=True)
        assert len(manager.per_sample_stats) == 0  # Should be cleared

    def test_set_calibration_mode_without_reset(self):
        """Test set_calibration_mode with reset_history=False."""
        manager = SparseAttentionStatsManager(module_name="test", enabled=True)

        # Collect some stats
        manager.set_calibration_mode(enabled=True)
        manager.collect(
            {
                "sparsity": 0.5,
                "phase": "prefill",
                "total_blocks": 100,
                "sparse_blocks": 50,
                "sample_length": 1024,
            }
        )
        assert len(manager.per_sample_stats) == 1

        # Disable without reset
        manager.set_calibration_mode(enabled=False, reset_history=False)
        assert len(manager.per_sample_stats) == 1  # Should be preserved


class TestGetSummary:
    """Test get_summary() functionality."""

    def test_get_summary_with_data(self):
        """Test get_summary returns correct averages."""
        manager = SparseAttentionStatsManager(module_name="test_module", enabled=True)

        # Collect stats
        manager.collect({"phase": "prefill", "total_blocks": 100, "sparse_blocks": 30})
        manager.collect({"phase": "prefill", "total_blocks": 100, "sparse_blocks": 50})

        summary = manager.get_summary()

        assert summary["module"] == "test_module"
        assert summary["total_calls"] == 2
        # Average sparsity: (30+50) / (100+100) = 80/200 = 0.4
        assert summary["average_sparsity"] == 0.4
        assert summary["phase_distribution"]["prefill"] == 2

    def test_get_summary_no_data(self):
        """Test get_summary with no collected data."""
        manager = SparseAttentionStatsManager(module_name="test", enabled=True)

        summary = manager.get_summary()

        assert summary["module"] == "test"
        assert summary["total_calls"] == 0
        assert summary["average_sparsity"] == 0.0
        assert summary["phase_distribution"]["prefill"] == 0

    def test_get_summary_zero_blocks(self):
        """Test get_summary when total_blocks is zero."""
        manager = SparseAttentionStatsManager(module_name="test", enabled=True)

        # Collect stats with zero blocks
        manager.collect({"phase": "prefill", "total_blocks": 0, "sparse_blocks": 0})

        summary = manager.get_summary()

        assert summary["average_sparsity"] == 0.0  # Should handle division by zero


class TestGetCalibrationStats:
    """Test get_calibration_stats() functionality."""

    def test_get_calibration_stats(self):
        """Test retrieving per-sample calibration stats."""
        manager = SparseAttentionStatsManager(module_name="test", enabled=True)
        manager.set_calibration_mode(enabled=True)

        # Collect multiple samples
        for i in range(3):
            manager.collect(
                {
                    "sparsity": 0.3 + i * 0.1,
                    "phase": "prefill",
                    "total_blocks": 100,
                    "sparse_blocks": 30,
                    "sample_length": 1024 + i * 512,
                }
            )

        calib_stats = manager.get_calibration_stats()

        assert len(calib_stats) == 3
        assert calib_stats[0]["sparsity"] == 0.3
        assert calib_stats[1]["sparsity"] == 0.4
        assert calib_stats[2]["sparsity"] == 0.5

    def test_get_calibration_stats_empty(self):
        """Test get_calibration_stats when no calibration data."""
        manager = SparseAttentionStatsManager(module_name="test", enabled=True)

        calib_stats = manager.get_calibration_stats()

        assert calib_stats == []


class TestReset:
    """Test reset functionality."""

    def test_reset(self):
        """Test reset() clears all statistics."""
        manager = SparseAttentionStatsManager(module_name="test", enabled=True)
        manager.set_calibration_mode(enabled=True)

        # Collect some stats
        manager.collect(
            {
                "sparsity": 0.5,
                "phase": "prefill",
                "total_blocks": 100,
                "sparse_blocks": 50,
                "sample_length": 1024,
            }
        )
        manager.collect(
            {
                "sparsity": 0.3,
                "phase": "decode",
                "total_blocks": 10,
                "sparse_blocks": 3,
                "sample_length": 128,
            }
        )

        # Verify stats exist
        assert manager.aggregated_stats["total_calls"] == 2
        assert len(manager.per_sample_stats) == 2

        # Reset
        manager.reset()

        # All stats should be cleared
        assert manager.aggregated_stats["total_calls"] == 0
        assert manager.aggregated_stats["total_blocks"] == 0
        assert manager.aggregated_stats["sparse_blocks"] == 0
        assert manager.per_sample_stats == []
        assert manager.aggregated_stats["phase_counts"]["prefill"] == 0
        assert manager.aggregated_stats["phase_counts"]["decode"] == 0
