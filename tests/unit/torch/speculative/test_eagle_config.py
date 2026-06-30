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

import warnings

import pytest
from pydantic import ValidationError

from modelopt.recipe.config import ModelOptEagleRecipe
from modelopt.torch.speculative.config import EagleConfig

# --- HSM mode validator tests ---


def test_hsm_mode_default_sparse_replace():
    """Default HSM mode preserves the existing sparse replacement behavior."""
    cfg = EagleConfig.model_validate({})

    assert cfg.eagle_hsm_mode == "sparse_replace"


def test_hsm_mode_uniform_layer_sample_ok_with_hsm_enabled():
    """Uniform layer sampling is valid when hidden-state mixing is enabled."""
    cfg = EagleConfig.model_validate(
        {"eagle_mix_hidden_states": True, "eagle_hsm_mode": "uniform_layer_sample"}
    )

    assert cfg.eagle_hsm_mode == "uniform_layer_sample"


def test_hsm_mode_unknown_error():
    """Unknown HSM modes fail at config validation time."""
    with pytest.raises(ValidationError, match="eagle_hsm_mode must be one of"):
        EagleConfig.model_validate({"eagle_hsm_mode": "random_weights"})


def test_hsm_mode_uniform_layer_sample_requires_hsm_enabled():
    """Uniform layer sampling is a hidden-state mixing mode, not a separate switch."""
    with pytest.raises(ValidationError, match="requires eagle_mix_hidden_states=True"):
        EagleConfig.model_validate({"eagle_hsm_mode": "uniform_layer_sample"})


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


# --- rope vs training_seq_len warning tests (on ModelOptEagleRecipe, where the validator lives) ---


_RopeMismatchMsg = "differs from training"


def _yarn_rope(orig_max_pos: int) -> dict:
    return {
        "rope_type": "yarn",
        "factor": 32.0,
        "original_max_position_embeddings": orig_max_pos,
    }


def test_warn_rope_mismatch():
    """Warning fires when original_max_position_embeddings != training.training_seq_len."""
    with pytest.warns(UserWarning, match=_RopeMismatchMsg):
        ModelOptEagleRecipe(
            eagle={"eagle_export_rope_scaling": _yarn_rope(2048)},
            training={"training_seq_len": 4096},
        )


def test_no_warn_rope_match():
    """No warning when original_max_position_embeddings == training.training_seq_len."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        ModelOptEagleRecipe(
            eagle={"eagle_export_rope_scaling": _yarn_rope(2048)},
            training={"training_seq_len": 2048},
        )


def test_no_warn_missing_orig_max_pos():
    """No warning when original_max_position_embeddings is absent from rope scaling config."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        ModelOptEagleRecipe(
            eagle={"eagle_export_rope_scaling": {}},
            training={"training_seq_len": 4096},
        )
