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

"""Unit tests for EAGLE export rope scaling logic in hf_spec_export.py."""

from unittest.mock import MagicMock

from modelopt.torch.export.plugins.hf_spec_export import EagleExporter

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
