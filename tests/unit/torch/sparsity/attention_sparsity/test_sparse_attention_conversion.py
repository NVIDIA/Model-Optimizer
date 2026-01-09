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

"""Tests for sparse attention conversion and replacement."""

import pytest

pytest.importorskip("transformers")

import torch.nn as nn
from _test_utils.torch.sparsity.sparse_attention_common import (
    FLASH_SKIP_SOFTMAX_DEFAULT_CFG,
    SimpleAttentionModel,
    SimpleTransformerEncoderLayer,
)

import modelopt.torch.opt as mto
import modelopt.torch.sparsity.attention_sparsity as sparse_attn
from modelopt.torch.sparsity.attention_sparsity.conversion import (
    disable_sparse_attention,
    enable_sparse_attention,
)
from modelopt.torch.sparsity.attention_sparsity.sparse_attention import SparseAttentionModule


class TestSparseAttentionReplacement:
    """Test module replacement logic."""

    def test_basic_replacement(self):
        """Test that attention modules are replaced with sparse versions."""
        model = SimpleAttentionModel()

        # Count original attention modules
        original_attention_count = sum(
            isinstance(m, nn.MultiheadAttention) for m in model.modules()
        )
        assert original_attention_count > 0

        # Apply sparse attention
        sparse_model = sparse_attn.sparsify(model, FLASH_SKIP_SOFTMAX_DEFAULT_CFG)

        # Count sparse attention modules
        sparse_attention_count = sum(
            isinstance(m, SparseAttentionModule) for m in sparse_model.modules()
        )

        # Verify replacement occurred
        assert sparse_attention_count > 0

    def test_enable_disable_toggle(self):
        """Test enabling and disabling sparse attention."""
        model = SimpleAttentionModel()
        model = sparse_attn.sparsify(model, FLASH_SKIP_SOFTMAX_DEFAULT_CFG)

        # Check initially enabled
        for module in model.modules():
            if isinstance(module, SparseAttentionModule):
                assert module.is_enabled

        # Disable all sparse attention modules
        disable_sparse_attention(model, "*")
        for module in model.modules():
            if isinstance(module, SparseAttentionModule):
                assert not module.is_enabled

        # Re-enable all sparse attention modules
        enable_sparse_attention(model, "*")
        for module in model.modules():
            if isinstance(module, SparseAttentionModule):
                assert module.is_enabled

    def test_pattern_based_replacement(self):
        """Test pattern-based selective replacement."""
        model = SimpleTransformerEncoderLayer()

        # Apply with pattern
        config = {
            "sparse_cfg": {
                "*self_attn*": {
                    "method": "flash_skip_softmax",
                    "threshold": 1e-4,
                    "br": 128,
                    "bc": 128,
                    "enable": True,
                },
                "default": {"enable": False},
            },
        }

        sparse_model = sparse_attn.sparsify(model, config)

        # Verify sparse modules exist
        has_sparse = any(isinstance(m, SparseAttentionModule) for m in sparse_model.modules())
        assert has_sparse


class TestConversionEdgeCases:
    """Test edge cases and error paths in conversion."""

    def test_callable_filter(self):
        """Test using callable filter instead of wildcard."""
        model = SimpleAttentionModel()

        # Use callable filter
        def filter_func(name):
            return "attn" in name

        config = {
            "sparse_cfg": {
                filter_func: {
                    "method": "flash_skip_softmax",
                    "threshold": 1e-3,
                    "enable": True,
                },
            },
        }

        sparse_model = sparse_attn.sparsify(model, config)
        has_sparse = any(isinstance(m, SparseAttentionModule) for m in sparse_model.modules())
        assert has_sparse

    def test_no_matching_modules(self):
        """Test pattern that matches nothing."""
        model = SimpleAttentionModel()

        config = {
            "sparse_cfg": {
                "*nonexistent*": {
                    "method": "flash_skip_softmax",
                    "threshold": 1e-3,
                    "enable": True,
                },
            },
        }

        # Should not error, even with no matches
        sparse_attn.sparsify(model, config)

    def test_disable_enable_functions(self):
        """Test disable/enable utility functions."""
        from modelopt.torch.sparsity.attention_sparsity.conversion import (
            disable_sparse_attention,
            enable_sparse_attention,
        )

        model = SimpleAttentionModel()
        model = sparse_attn.sparsify(model, FLASH_SKIP_SOFTMAX_DEFAULT_CFG)

        # Disable all
        disable_sparse_attention(model, "*")
        for module in model.modules():
            if isinstance(module, SparseAttentionModule):
                assert not module.is_enabled

        # Enable all
        enable_sparse_attention(model, "*")
        for module in model.modules():
            if isinstance(module, SparseAttentionModule):
                assert module.is_enabled

    def test_restore_sparse_attention_model(self):
        """Test save/restore via modelopt_state."""
        # Create and sparsify original model
        model_orig = SimpleAttentionModel()
        model_orig = sparse_attn.sparsify(model_orig, FLASH_SKIP_SOFTMAX_DEFAULT_CFG)

        # Save state
        state_dict = mto.modelopt_state(model_orig)

        # Restore to new model
        model_restored = SimpleAttentionModel()
        mto.restore_from_modelopt_state(model_restored, state_dict)

        # Verify restoration
        has_sparse = any(isinstance(m, SparseAttentionModule) for m in model_restored.modules())
        assert has_sparse

        # Verify module is configured
        for module in model_restored.modules():
            if isinstance(module, SparseAttentionModule):
                assert hasattr(module, "_method")
                assert module._method == "flash_skip_softmax"
