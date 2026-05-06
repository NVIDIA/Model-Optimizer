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

"""Tests for TriAttention configuration."""

import pytest
from pydantic import ValidationError

from modelopt.torch.sparsity.kv_cache.config import TriAttentionConfig


def test_default_config():
    """Default config creates valid instance."""
    config = TriAttentionConfig()
    assert config.budget == 2048
    assert config.prune_interval == 128
    assert config.window_size == 128
    assert config.pruning_mode == "per_head"
    assert config.score_aggregation == "mean"
    assert config.offset_max_length == 65536
    assert config.disable_mlr is False
    assert config.disable_trig is False
    assert config.calib_size == 100000


def test_config_custom_values():
    """Config accepts custom values."""
    config = TriAttentionConfig(budget=4096, prune_interval=64, window_size=256)
    assert config.budget == 4096
    assert config.prune_interval == 64
    assert config.window_size == 256


def test_config_invalid_pruning_mode():
    """Invalid pruning mode raises validation error."""
    with pytest.raises(ValidationError):
        TriAttentionConfig(pruning_mode="invalid")


def test_config_invalid_aggregation():
    """Invalid score aggregation raises validation error."""
    with pytest.raises(ValidationError):
        TriAttentionConfig(score_aggregation="invalid")


def test_config_serialization_roundtrip():
    """Config can be serialized and deserialized."""
    config = TriAttentionConfig(budget=1024, prune_interval=64)
    data = config.model_dump()
    restored = TriAttentionConfig(**data)
    assert restored.budget == 1024
    assert restored.prune_interval == 64


def test_config_per_layer_per_head_mode():
    """per_layer_per_head is a valid pruning mode."""
    config = TriAttentionConfig(pruning_mode="per_layer_per_head")
    assert config.pruning_mode == "per_layer_per_head"


def test_config_max_aggregation():
    """max is a valid score aggregation."""
    config = TriAttentionConfig(score_aggregation="max")
    assert config.score_aggregation == "max"
