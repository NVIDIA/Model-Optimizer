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

"""Tests for EagleConfig model validators."""

import types
import warnings

import pytest
from pydantic import ValidationError

from modelopt.torch.speculative.config import EagleConfig

# --- rope scaling consistency validator tests ---


def test_rope_consistency_error_non_default_rope_type():
    """Error when eagle_export_rope_scaling is set but training rope_type is not 'default'."""
    cfg = {
        "eagle_export_rope_scaling": {"rope_type": "yarn", "factor": 32.0},
        "eagle_architecture_config": {"rope_scaling": {"rope_type": "llama3"}},
    }
    with pytest.raises(ValidationError, match="rope_type='llama3'"):
        EagleConfig.model_validate(cfg)


def test_rope_consistency_error_non_default_rope_type_alt_key():
    """Error when rope_scaling uses 'type' key instead of 'rope_type' (kimik2-style)."""
    cfg = {
        "eagle_export_rope_scaling": {"rope_type": "yarn", "factor": 32.0},
        "eagle_architecture_config": {"rope_scaling": {"type": "yarn"}},
    }
    with pytest.raises(ValidationError, match="rope_type='yarn'"):
        EagleConfig.model_validate(cfg)


def test_rope_consistency_ok_default_rope_type():
    """No error when training rope_type is 'default'."""
    cfg = {
        "eagle_export_rope_scaling": {"rope_type": "yarn", "factor": 32.0},
        "eagle_architecture_config": {"rope_scaling": {"rope_type": "default"}},
    }
    EagleConfig.model_validate(cfg)


def test_rope_consistency_ok_no_rope_scaling_in_arch():
    """No error when eagle_architecture_config has no rope_scaling (defaults to 'default')."""
    cfg = {
        "eagle_export_rope_scaling": {"rope_type": "yarn", "factor": 32.0},
        "eagle_architecture_config": {},
    }
    EagleConfig.model_validate(cfg)


def test_rope_consistency_ok_empty_export_rope():
    """No error when eagle_export_rope_scaling is empty (disabled)."""
    cfg = {
        "eagle_export_rope_scaling": {},
        "eagle_architecture_config": {"rope_scaling": {"rope_type": "llama3"}},
    }
    EagleConfig.model_validate(cfg)


# --- rope vs training_seq_len warning tests ---


def _make_training_args(training_seq_len: int):
    return types.SimpleNamespace(training_seq_len=training_seq_len)


def test_warn_rope_mismatch():
    """Warning should fire when original_max_position_embeddings != training_seq_len."""
    cfg = {
        "eagle_export_rope_scaling": {
            "rope_type": "yarn",
            "factor": 32.0,
            "original_max_position_embeddings": 2048,
        },
    }
    with pytest.warns(UserWarning, match="differs from training_seq_len"):
        EagleConfig.model_validate(cfg, context={"training_args": _make_training_args(4096)})


def test_no_warn_rope_match():
    """No warning when original_max_position_embeddings == training_seq_len."""
    cfg = {
        "eagle_export_rope_scaling": {
            "rope_type": "yarn",
            "factor": 32.0,
            "original_max_position_embeddings": 2048,
        },
    }
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        EagleConfig.model_validate(cfg, context={"training_args": _make_training_args(2048)})


def test_no_warn_without_context():
    """No warning when context is not provided (e.g. inside convert chain)."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        EagleConfig.model_validate({})


def test_no_warn_missing_orig_max_pos():
    """No warning when original_max_position_embeddings is absent from rope scaling config."""
    cfg = {"eagle_export_rope_scaling": {}}
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        EagleConfig.model_validate(cfg, context={"training_args": _make_training_args(4096)})


def test_no_warn_empty_context():
    """No warning when context dict has no training_args key."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        EagleConfig.model_validate({}, context={})
