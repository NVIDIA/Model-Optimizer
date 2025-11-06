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

"""Tests for sparse attention mode registry."""

import pytest

pytest.importorskip("transformers")

from modelopt.torch.opt.mode import _ModeRegistryCls
from modelopt.torch.sparsity.attention_sparsity.mode import SparseAttentionModeRegistry


def test_sparse_attention_mode_exists():
    """Test that sparse_attention mode is registered."""
    assert "sparse_attention" in SparseAttentionModeRegistry


def test_sparse_attention_mode_descriptor():
    """Test sparse attention mode descriptor properties."""
    mode_descriptor = _ModeRegistryCls.get_from_any("sparse_attention")

    assert mode_descriptor is not None
    assert hasattr(mode_descriptor, "config_class")
    assert hasattr(mode_descriptor, "convert")


def test_mode_registry_get():
    """Test getting mode from registry."""
    mode = SparseAttentionModeRegistry["sparse_attention"]
    assert mode is not None
