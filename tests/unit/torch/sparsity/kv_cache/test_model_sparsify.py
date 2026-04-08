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

"""Tests for TriAttention sparsify/calibrate entry API."""

import torch
import torch.nn as nn

from modelopt.torch.sparsity.kv_cache.config import TriAttentionConfig
from modelopt.torch.sparsity.kv_cache.model_sparsify import calibrate, sparsify


def test_sparsify_returns_model():
    """sparsify() returns the model."""
    model = nn.Linear(16, 16)
    result = sparsify(model, TriAttentionConfig())
    assert result is model


def test_sparsify_accepts_dict_config():
    """sparsify() accepts dict config."""
    model = nn.Linear(16, 16)
    result = sparsify(model, {"budget": 1024})
    assert result is model


def test_sparsify_preserves_weights():
    """sparsify() does not modify model weights."""
    model = nn.Linear(16, 16)
    original_weight = model.weight.data.clone()
    sparsify(model, TriAttentionConfig())
    torch.testing.assert_close(model.weight.data, original_weight)


def test_calibrate_returns_model():
    """calibrate() returns the model."""
    model = nn.Linear(16, 16)
    sparsify(model, TriAttentionConfig())
    result = calibrate(model)
    assert result is model


def test_sparsify_then_calibrate():
    """sparsify() followed by calibrate() works without error."""
    model = nn.Linear(16, 16)
    model = sparsify(model, TriAttentionConfig(budget=512))
    model = calibrate(model)
    assert isinstance(model, nn.Module)
