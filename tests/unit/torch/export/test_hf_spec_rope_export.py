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

"""Unit tests for EAGLE/DFlash export rope scaling logic in hf_spec_export.py."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from modelopt.torch.export.plugins.hf_spec_export import DFlashExporter, EagleExporter

DEFAULT_ROPE_SCALING = {
    "rope_type": "yarn",
    "factor": 32.0,
    "original_max_position_embeddings": 2048,
}


def _make_exporter(
    rope_type="default",
    rope_theta=10000,
    eagle_export_rope_scaling=None,
):
    if eagle_export_rope_scaling is None:
        eagle_export_rope_scaling = DEFAULT_ROPE_SCALING

    model = MagicMock()
    model.eagle_config.eagle_decoder_type = "llama"
    model.eagle_config.rope_scaling = {"rope_type": rope_type, "rope_theta": rope_theta}
    # rope_theta lives inside rope_scaling in transformers 5.x; clear the top-level attr
    # so the fallback path is exercised instead of MagicMock's auto-attr.
    model.eagle_config.rope_theta = None
    model.eagle_export_rope_scaling = eagle_export_rope_scaling
    model._draft_model_config = None
    model.config.rope_scaling = None
    model.config.rope_theta = None

    exporter = EagleExporter.__new__(EagleExporter)
    exporter.model = model
    exporter.eagle_decoder_type = "llama"
    exporter.num_hidden_layers = 1
    return exporter


def test_yarn_rope_injected_with_correct_config():
    """YaRN rope_scaling is injected as-is when training rope_type is 'default'."""
    config = _make_exporter(rope_type="default")._export_config()
    assert config["rope_scaling"] == DEFAULT_ROPE_SCALING


def test_rope_not_overridden_when_non_default_training_rope():
    """Export override is not applied when training rope_type is not 'default';
    rope_scaling falls through to the training config."""
    config = _make_exporter(rope_type="llama3")._export_config()
    assert config["rope_scaling"] == {"rope_type": "llama3", "rope_theta": 10000}


def test_rope_not_overridden_when_eagle_export_rope_scaling_is_empty():
    """Export override is not applied when eagle_export_rope_scaling is empty;
    rope_scaling falls through to the training config."""
    config = _make_exporter(eagle_export_rope_scaling={})._export_config()
    assert config["rope_scaling"] == {"rope_type": "default", "rope_theta": 10000}


def test_rope_theta_fallback_from_rope_scaling():
    """rope_theta is populated from rope_scaling when not available as top-level attr."""
    config = _make_exporter(rope_type="default", rope_theta=500000)._export_config()
    assert config["rope_theta"] == 500000


# ---------------------------------------------------------------------------
# DFlash export rope scaling (config-field convergence, mirrors the eagle style)
# ---------------------------------------------------------------------------

DFLASH_YARN = {
    "type": "yarn",
    "factor": 48.0,
    "original_max_position_embeddings": 4096,
    "beta_fast": 1.0,
    "beta_slow": 1.0,
    "mscale": 1.0,
    "mscale_all_dim": 1.0,
}


def _make_dflash_exporter(dflash_export_rope_scaling=None, base_rope_theta=5000000.0):
    base_config = SimpleNamespace(
        hidden_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=256,
        vocab_size=1000,
        max_position_embeddings=196608,
        initializer_range=0.02,
        num_hidden_layers=8,
        rope_theta=base_rope_theta,
        torch_dtype=torch.bfloat16,
    )
    draft_config = SimpleNamespace(num_hidden_layers=2)
    model = SimpleNamespace(
        config=base_config,
        dflash_config=draft_config,
        dflash_block_size=8,
        mask_token_id=999,
        target_layer_ids=[1, 3, 5, 7],
        dflash_export_rope_scaling=dflash_export_rope_scaling,
    )
    exporter = DFlashExporter.__new__(DFlashExporter)
    exporter.model = model
    return exporter


def test_dflash_yarn_rope_injected_from_config_field():
    """YaRN rope_scaling from dflash_export_rope_scaling is injected verbatim."""
    config = _make_dflash_exporter(dflash_export_rope_scaling=DFLASH_YARN)._export_config()
    assert config["rope_scaling"] == DFLASH_YARN


def test_dflash_rope_not_injected_when_field_empty():
    """Empty dict (default) disables rope scaling injection."""
    config = _make_dflash_exporter(dflash_export_rope_scaling={})._export_config()
    assert config["rope_scaling"] is None


def test_dflash_rope_theta_inherits_base():
    """rope_theta is inherited from the target/base config (draft drafts for the base)."""
    config = _make_dflash_exporter(base_rope_theta=5000000.0)._export_config()
    assert config["rope_theta"] == 5000000.0
