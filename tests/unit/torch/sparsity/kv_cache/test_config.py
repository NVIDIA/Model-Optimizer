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


def test_budget_only():
    """Setting only budget is valid."""
    config = TriAttentionConfig(budget=2048)
    assert config.budget == 2048


def test_target_sparsity_ratio_is_not_supported():
    """Ratio-based eviction is not part of the TriAttention API."""
    with pytest.raises(ValidationError):
        TriAttentionConfig(budget=2048, target_sparsity_ratio=0.7)


def test_missing_budget_raises():
    """TriAttention requires an explicit budget."""
    with pytest.raises(ValidationError, match="requires 'budget'"):
        TriAttentionConfig()


def test_non_positive_budget_raises():
    """Budget must be positive."""
    with pytest.raises(ValidationError, match="budget must be positive"):
        TriAttentionConfig(budget=0)


def test_config_custom_values():
    """Config accepts custom values alongside budget."""
    config = TriAttentionConfig(budget=4096, prune_interval=64, window_size=256)
    assert config.budget == 4096
    assert config.prune_interval == 64
    assert config.window_size == 256


def test_config_invalid_pruning_mode():
    """Invalid pruning mode raises validation error."""
    with pytest.raises(ValidationError):
        TriAttentionConfig(budget=2048, pruning_mode="invalid")


def test_config_invalid_aggregation():
    """Invalid score aggregation raises validation error."""
    with pytest.raises(ValidationError):
        TriAttentionConfig(budget=2048, score_aggregation="invalid")


def test_config_serialization_roundtrip_budget():
    """Config with budget survives serialization roundtrip."""
    config = TriAttentionConfig(budget=1024, prune_interval=64)
    data = config.model_dump()
    restored = TriAttentionConfig(**data)
    assert restored.budget == 1024
    assert restored.prune_interval == 64


def test_config_per_layer_per_head_mode():
    """per_layer_per_head is a valid pruning mode."""
    config = TriAttentionConfig(budget=2048, pruning_mode="per_layer_per_head")
    assert config.pruning_mode == "per_layer_per_head"


def test_config_max_aggregation():
    """max is a valid score aggregation."""
    config = TriAttentionConfig(budget=2048, score_aggregation="max")
    assert config.score_aggregation == "max"
