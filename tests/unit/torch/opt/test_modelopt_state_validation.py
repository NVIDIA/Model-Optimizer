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

import os
import tempfile

import pytest
import torch

import modelopt.torch.opt as mto


class TestModeloptStateValidation:
    """Test suite for modelopt state validation."""

    def test_validate_modelopt_state_valid(self):
        """Test validation of a valid modelopt state."""
        valid_state = {
            "modelopt_state_dict": [],
            "modelopt_version": "0.1.0",
        }
        # Should not raise any exception
        mto.ModeloptStateManager.validate_modelopt_state(valid_state)

    def test_validate_modelopt_state_not_dict(self):
        """Test validation fails when state is not a dictionary."""
        with pytest.raises(TypeError) as exc_info:
            mto.ModeloptStateManager.validate_modelopt_state([1, 2, 3])
        assert "Expected loaded modelopt state to be a dictionary" in str(exc_info.value)

    def test_validate_modelopt_state_missing_keys(self):
        """Test validation fails when required keys are missing."""
        with pytest.raises(ValueError) as exc_info:
            mto.ModeloptStateManager.validate_modelopt_state({"modelopt_state_dict": []})
        assert "missing required keys" in str(exc_info.value)
        assert "modelopt_version" in str(exc_info.value)

    def test_validate_modelopt_state_invalid_version_type(self):
        """Test validation fails when modelopt_version is not a string."""
        with pytest.raises(TypeError) as exc_info:
            mto.ModeloptStateManager.validate_modelopt_state(
                {
                    "modelopt_state_dict": [],
                    "modelopt_version": 123,
                }
            )
        assert "modelopt_version" in str(exc_info.value)
        assert "string" in str(exc_info.value)

    def test_validate_modelopt_state_invalid_state_dict_type(self):
        """Test validation fails when modelopt_state_dict is not a list."""
        with pytest.raises(TypeError) as exc_info:
            mto.ModeloptStateManager.validate_modelopt_state(
                {
                    "modelopt_state_dict": "not a list",
                    "modelopt_version": "0.1.0",
                }
            )
        assert "modelopt_state_dict" in str(exc_info.value)

    def test_validate_modelopt_state_invalid_entry_not_tuple(self):
        """Test validation fails when state_dict entry is not a tuple."""
        with pytest.raises(ValueError) as exc_info:
            mto.ModeloptStateManager.validate_modelopt_state(
                {
                    "modelopt_state_dict": [{"mode": "quantize"}],
                    "modelopt_version": "0.1.0",
                }
            )
        assert "tuple of length 2" in str(exc_info.value)

    def test_validate_modelopt_state_invalid_entry_wrong_length(self):
        """Test validation fails when tuple has wrong length."""
        with pytest.raises(ValueError) as exc_info:
            mto.ModeloptStateManager.validate_modelopt_state(
                {
                    "modelopt_state_dict": [("quantize",)],
                    "modelopt_version": "0.1.0",
                }
            )
        assert "tuple of length 2" in str(exc_info.value)

    def test_validate_modelopt_state_invalid_mode_name_type(self):
        """Test validation fails when mode name is not a string."""
        with pytest.raises(TypeError) as exc_info:
            mto.ModeloptStateManager.validate_modelopt_state(
                {
                    "modelopt_state_dict": [(123, {})],
                    "modelopt_version": "0.1.0",
                }
            )
        assert "mode name" in str(exc_info.value)
        assert "string" in str(exc_info.value)

    def test_validate_modelopt_state_invalid_mode_state_type(self):
        """Test validation fails when mode state is not a dictionary."""
        with pytest.raises(TypeError) as exc_info:
            mto.ModeloptStateManager.validate_modelopt_state(
                {
                    "modelopt_state_dict": [("quantize", "not a dict")],
                    "modelopt_version": "0.1.0",
                }
            )
        assert "mode state" in str(exc_info.value)
        assert "dictionary" in str(exc_info.value)

    def test_load_modelopt_state_valid_file(self):
        """Test loading a valid modelopt state from file."""
        valid_state = {
            "modelopt_state_dict": [],
            "modelopt_version": "0.1.0",
        }
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            temp_file = f.name
        try:
            torch.save(valid_state, temp_file)
            loaded_state = mto.load_modelopt_state(temp_file)
            assert loaded_state == valid_state
        finally:
            os.remove(temp_file)

    def test_load_modelopt_state_invalid_file(self):
        """Test loading an invalid modelopt state from file."""
        invalid_state = [1, 2, 3]
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            temp_file = f.name
        try:
            torch.save(invalid_state, temp_file)
            with pytest.raises(TypeError) as exc_info:
                mto.load_modelopt_state(temp_file)
            assert "Expected loaded modelopt state to be a dictionary" in str(exc_info.value)
        finally:
            os.remove(temp_file)

    def test_load_modelopt_state_with_valid_entries(self):
        """Test loading modelopt state with valid mode entries."""
        valid_state = {
            "modelopt_state_dict": [
                ("quantize", {"config": {}, "metadata": {}}),
            ],
            "modelopt_version": "0.1.0",
        }
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            temp_file = f.name
        try:
            torch.save(valid_state, temp_file)
            loaded_state = mto.load_modelopt_state(temp_file)
            assert loaded_state == valid_state
        finally:
            os.remove(temp_file)
