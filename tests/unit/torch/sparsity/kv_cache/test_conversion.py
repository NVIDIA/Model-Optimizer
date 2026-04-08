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

"""Tests for TriAttention mode registration and conversion."""

import torch
import torch.nn as nn

from modelopt.torch.sparsity.kv_cache.config import TriAttentionConfig
from modelopt.torch.sparsity.kv_cache.conversion import (
    convert_triattention,
    restore_triattention,
    update_triattention_metadata,
)
from modelopt.torch.sparsity.kv_cache.mode import KVCacheSparsityRegistry


def test_mode_registered():
    """TriAttention mode is registered in KVCacheSparsityRegistry."""
    assert "triattention" in KVCacheSparsityRegistry


def test_mode_descriptor_properties():
    """Mode descriptor has correct properties."""
    descriptor = KVCacheSparsityRegistry["triattention"]
    assert descriptor.name == "triattention"
    assert descriptor.config_class is TriAttentionConfig


def test_mode_discoverable_globally():
    """TriAttention mode is discoverable via get_from_any."""
    from modelopt.torch.opt.mode import _ModeRegistryCls

    descriptor = _ModeRegistryCls.get_from_any("triattention")
    assert descriptor is not None
    assert descriptor.name == "triattention"


def test_convert_returns_model_and_metadata():
    """Convert returns (model, metadata) without modifying weights."""
    model = nn.Linear(16, 16)
    original_weight = model.weight.data.clone()
    config = TriAttentionConfig()

    converted_model, metadata = convert_triattention(model, config)

    torch.testing.assert_close(converted_model.weight.data, original_weight)
    assert "triattention_config" in metadata


def test_convert_metadata_contains_config():
    """Metadata stores the config values."""
    model = nn.Linear(16, 16)
    config = TriAttentionConfig(budget=1024, prune_interval=64)

    _, metadata = convert_triattention(model, config)

    assert metadata["triattention_config"]["budget"] == 1024
    assert metadata["triattention_config"]["prune_interval"] == 64


def test_restore_returns_model():
    """Restore returns the model."""
    model = nn.Linear(16, 16)
    config = TriAttentionConfig()
    _, metadata = convert_triattention(model, config)

    restored = restore_triattention(model, config, metadata)
    assert restored is model


def test_update_metadata():
    """update_triattention_metadata updates config in metadata."""
    model = nn.Linear(16, 16)
    config = TriAttentionConfig(budget=512)
    metadata = {}

    update_triattention_metadata(model, config, metadata)

    assert metadata["triattention_config"]["budget"] == 512
