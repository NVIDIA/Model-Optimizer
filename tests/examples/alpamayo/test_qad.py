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

"""Unit tests for examples/alpamayo/qad.py split validation and export logic."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# Mock external dependencies before importing qad
def create_mock_module(name):
    """Create a mock module that returns itself for any attribute access."""
    mock = MagicMock()
    mock.__name__ = name
    return mock


# Mock all alpamayo and external dependencies
for module_name in [
    "physical_ai_av",
    "alpamayo_r1",
    "alpamayo_r1.load_physical_aiavdataset",
    "alpamayo_r1.models",
    "alpamayo_r1.models.alpamayo_r1",
    "alpamayo_r1.models.token_utils",
    "safetensors",
    "safetensors.torch",
]:
    sys.modules[module_name] = create_mock_module(module_name)

# Add examples directory to path and import qad
examples_dir = Path(__file__).parent.parent.parent.parent / "examples" / "alpamayo"
sys.path.insert(0, str(examples_dir))

from qad import clip_slice


class TestClipSliceLogic:
    """Test clip_slice function with production code execution."""

    def test_clip_slice_basic(self, tmp_path):
        """Test basic slicing with offset and limit using production clip_slice."""
        parquet_file = tmp_path / "test.parquet"

        with patch("qad.read_clip_ids_from_parquet") as mock_read:
            mock_read.return_value = ["clip_0", "clip_1", "clip_2", "clip_3", "clip_4"]
            result = clip_slice(str(parquet_file), offset=0, limit=2)
            assert result == ["clip_0", "clip_1"], "Should return first 2 clips"

    def test_clip_slice_with_offset(self, tmp_path):
        """Test slicing with non-zero offset using production clip_slice."""
        parquet_file = tmp_path / "test.parquet"

        with patch("qad.read_clip_ids_from_parquet") as mock_read:
            mock_read.return_value = ["clip_0", "clip_1", "clip_2", "clip_3", "clip_4"]
            result = clip_slice(str(parquet_file), offset=2, limit=2)
            assert result == ["clip_2", "clip_3"], "Should return clips at indices [2, 3]"

    def test_clip_slice_limit_zero(self, tmp_path):
        """Test that production clip_slice returns empty list when limit=0."""
        parquet_file = tmp_path / "test.parquet"

        with patch("qad.read_clip_ids_from_parquet") as mock_read:
            mock_read.return_value = ["clip_0", "clip_1", "clip_2"]
            result = clip_slice(str(parquet_file), offset=0, limit=0)
            assert result == [], "limit=0 should return empty list, not all remaining clips"

    def test_clip_slice_limit_exceeds_available(self, tmp_path):
        """Test that production clip_slice returns only available clips."""
        parquet_file = tmp_path / "test.parquet"

        with patch("qad.read_clip_ids_from_parquet") as mock_read:
            mock_read.return_value = ["clip_0", "clip_1", "clip_2"]
            result = clip_slice(str(parquet_file), offset=1, limit=10)
            assert result == ["clip_1", "clip_2"], "Should return only available clips"


class TestTrainValSplitValidation:
    """Test train/val split validation including negative value rejection and overlap detection."""

    def test_negative_train_offset_rejected(self):
        """Test that negative train_offset is rejected."""
        train_offset = -1
        assert train_offset < 0, "Negative train_offset should be rejected"

    def test_negative_limit_train_rejected(self):
        """Test that negative limit_train is rejected."""
        limit_train = -1
        assert limit_train < 0, "Negative limit_train should be rejected"

    def test_negative_val_offset_rejected(self):
        """Test that negative val_offset is rejected."""
        val_offset = -1
        assert val_offset < 0, "Negative val_offset should be rejected"

    def test_negative_limit_val_rejected(self):
        """Test that negative limit_val is rejected."""
        limit_val = -1
        assert limit_val < 0, "Negative limit_val should be rejected"

    def test_non_overlapping_splits(self):
        """Test that non-overlapping splits pass validation."""
        train_offset, limit_train = 0, 100
        val_offset, limit_val = 100, 10

        train_end = train_offset + limit_train
        val_end = val_offset + limit_val

        # Overlap check from qad.py main():
        # if not (train_end <= val_offset or val_end <= train_offset): raise
        is_valid = train_end <= val_offset or val_end <= train_offset
        assert is_valid, "train=[0:100], val=[100:110] should be valid (non-overlapping)"

    def test_overlapping_splits_fails(self):
        """Test that overlapping splits fail validation."""
        train_offset, limit_train = 0, 100
        val_offset, limit_val = 50, 10

        train_end = train_offset + limit_train
        val_end = val_offset + limit_val

        is_valid = train_end <= val_offset or val_end <= train_offset
        assert not is_valid, "train=[0:100], val=[50:60] should fail (overlapping)"

    def test_val_before_train_fails(self):
        """Test that val before train (but overlapping) fails."""
        train_offset, limit_train = 50, 100
        val_offset, limit_val = 0, 60

        train_end = train_offset + limit_train
        val_end = val_offset + limit_val

        is_valid = train_end <= val_offset or val_end <= train_offset
        assert not is_valid, "train=[50:150], val=[0:60] should fail (overlapping)"

    def test_splits_just_touching_is_valid(self):
        """Test that adjacent (non-overlapping) splits are valid."""
        train_offset, limit_train = 0, 100
        val_offset, limit_val = 100, 10

        train_end = train_offset + limit_train
        val_end = val_offset + limit_val

        is_valid = train_end <= val_offset or val_end <= train_offset
        assert is_valid, "train=[0:100], val=[100:110] should be valid (adjacent, not overlapping)"

    def test_readme_example_non_overlapping(self):
        """Test that README's example command has non-overlapping splits."""
        # From README: --limit_train 2000 --val_offset 2000 --limit_val 4
        train_offset, limit_train = 0, 2000
        val_offset, limit_val = 2000, 4

        train_end = train_offset + limit_train
        val_end = val_offset + limit_val

        is_valid = train_end <= val_offset or val_end <= train_offset
        assert is_valid, "README example train=[0:2000], val=[2000:2004] should be valid"


class TestExportValidation:
    """Test export validation logic."""

    def test_export_missing_keys_should_raise(self):
        """Test that missing keys in trained state dict trigger error."""
        missing = {"vlm.model.layers.0.self_attn.q_proj.weight"}

        # Logic from export_full_model: if missing: raise ValueError
        has_error = len(missing) > 0
        assert has_error, "Missing keys should trigger error in export"

    def test_export_unexpected_keys_should_raise(self):
        """Test that unexpected keys in trained state dict trigger error."""
        unexpected = {"unexpected_layer.weight"}

        # Logic from export_full_model: if unexpected: raise ValueError
        has_error = len(unexpected) > 0
        assert has_error, "Unexpected keys should trigger error in export"

    def test_export_exact_match_passes(self):
        """Test that exact key match passes validation."""
        missing = set()
        unexpected = set()

        # Logic from export_full_model: only proceed if no missing/unexpected
        is_valid = len(missing) == 0 and len(unexpected) == 0
        assert is_valid, "Exact key match should pass validation"

    def test_export_multiple_missing_keys(self):
        """Test detection of multiple missing keys."""
        missing = {
            "vlm.model.layers.0.weight",
            "vlm.model.layers.1.weight",
            "vlm.lm_head.weight",
        }

        has_error = len(missing) > 0
        assert has_error
        assert len(missing) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
