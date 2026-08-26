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

from unittest.mock import MagicMock, patch

import torch.nn as nn
from _test_utils.torch.sparsity.sparse_attention_common import (
    FLASH_SKIP_SOFTMAX_DEFAULT_CFG,
    SimpleAttentionModel,
    SimpleTransformerEncoderLayer,
)

import modelopt.torch.opt as mto
import modelopt.torch.sparsity.attention_sparsity as sparse_attn
from modelopt.torch.sparsity.attention_sparsity.conversion import (
    _set_attn_implementation,
    disable_sparse_attention,
    enable_sparse_attention,
    export_sparse_attention_config,
    print_sparse_attention_summary,
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

    def test_pattern_based_replacement(self):
        """Test pattern-based selective replacement."""
        model = SimpleTransformerEncoderLayer()

        # Apply with pattern
        config = {
            "sparse_cfg": {
                "*self_attn*": {
                    "method": "flash_skip_softmax",
                    "thresholds": {"prefill": [1e-4], "decode": [1e-4]},
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
                    "thresholds": {"prefill": [1e-3], "decode": [1e-4]},
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
                    "thresholds": {"prefill": [1e-3], "decode": [1e-4]},
                    "enable": True,
                },
            },
        }

        # Should not error, even with no matches
        sparse_attn.sparsify(model, config)

    def test_disable_enable_functions(self):
        """Test disable/enable utility functions."""

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

    def test_print_sparse_attention_summary(self, capsys):
        """Test print_sparse_attention_summary function."""
        model = SimpleAttentionModel()
        model = sparse_attn.sparsify(model, FLASH_SKIP_SOFTMAX_DEFAULT_CFG)

        # Print summary
        print_sparse_attention_summary(model)

        # Capture output
        captured = capsys.readouterr()
        assert "Sparse attention:" in captured.out
        assert "modules enabled" in captured.out

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


class TestSparseAttentionModuleMethods:
    """Test SparseAttentionModule methods."""

    def test_get_stats_with_stats_manager(self):
        """Test get_stats() when stats manager exists and is enabled."""
        model = SimpleAttentionModel()
        config = {
            "sparse_cfg": {
                "*attention*": {
                    "method": "flash_skip_softmax",
                    "thresholds": {"prefill": [0.001], "decode": [0.0001]},
                    "br": 64,
                    "bc": 64,
                    "collect_stats": True,  # Enable stats collection
                    "enable": True,
                }
            },
        }

        sparse_model = sparse_attn.sparsify(model, config)

        # Find sparse module
        sparse_module = None
        for module in sparse_model.modules():
            if isinstance(module, SparseAttentionModule):
                sparse_module = module
                break

        assert sparse_module is not None
        assert sparse_module._stats_manager is not None

        # Get stats (should return summary)
        stats = sparse_module.get_stats()

        assert isinstance(stats, dict)
        assert "module" in stats
        assert "total_calls" in stats
        assert "average_sparsity" in stats

    def test_get_stats_without_stats_manager(self):
        """Test get_stats() when stats manager is None."""
        model = SimpleAttentionModel()
        config = {
            "sparse_cfg": {
                "*attention*": {
                    "method": "flash_skip_softmax",
                    "thresholds": {"prefill": [0.001], "decode": [0.0001]},
                    "br": 64,
                    "bc": 64,
                    "collect_stats": False,  # Disable stats collection
                    "enable": True,
                }
            },
        }

        sparse_model = sparse_attn.sparsify(model, config)

        # Find sparse module
        for module in sparse_model.modules():
            if isinstance(module, SparseAttentionModule):
                # Stats manager should be None
                assert module._stats_manager is None

                # get_stats should return empty dict
                stats = module.get_stats()
                assert stats == {}
                break


class TestSetAttnImplementation:
    """Cover the _set_attn_implementation logic in conversion.py."""

    def test_triton_backend_sets_attn_impl(self):
        """triton backend sets _attn_implementation=modelopt_triton on model.config."""
        model = type(
            "M",
            (),
            {"config": type("C", (), {"_attn_implementation": "eager"})()},
        )()
        config = type("Cfg", (), {"sparse_cfg": {}})()
        config.sparse_cfg = {
            "*": {"method": "triton_skip_softmax", "backend": "triton"},
        }
        with patch(
            "modelopt.torch.kernels.sparsity.attention.register_triton_attention",
            MagicMock(return_value=True),
        ):
            _set_attn_implementation(model, config)
        assert model.config._attn_implementation == "modelopt_triton"

    def test_triton_backend_register_failure_raises(self):
        """When register_triton_attention returns False, a RuntimeError is raised."""
        model = type(
            "M",
            (),
            {"config": type("C", (), {"_attn_implementation": "eager"})()},
        )()
        config = type("Cfg", (), {"sparse_cfg": {}})()
        config.sparse_cfg = {"*": {"method": "triton_skip_softmax", "backend": "triton"}}
        with (
            patch(
                "modelopt.torch.kernels.sparsity.attention.register_triton_attention",
                MagicMock(return_value=False),
            ),
            pytest.raises(RuntimeError, match="Failed to register"),
        ):
            _set_attn_implementation(model, config)

    def test_triton_backend_no_triton_raises(self):
        """When register_triton_attention is None, ImportError is raised."""
        model = type(
            "M",
            (),
            {"config": type("C", (), {"_attn_implementation": "eager"})()},
        )()
        config = type("Cfg", (), {"sparse_cfg": {}})()
        config.sparse_cfg = {"*": {"method": "triton_skip_softmax", "backend": "triton"}}
        with (
            patch(
                "modelopt.torch.kernels.sparsity.attention.register_triton_attention",
                None,
            ),
            pytest.raises(ImportError, match="Triton backend requires"),
        ):
            _set_attn_implementation(model, config)

    def test_mixed_backends_raises(self):
        """Mixing pytorch and triton backends is not supported."""
        model = type("M", (), {"config": None})()
        config = type("Cfg", (), {"sparse_cfg": {}})()
        config.sparse_cfg = {
            "layer1": {"method": "triton_skip_softmax", "backend": "triton"},
            "layer2": {"method": "flash_skip_softmax", "backend": "pytorch"},
        }
        with pytest.raises(ValueError, match="Mixed backends"):
            _set_attn_implementation(model, config)

    def test_vsa_only_is_noop(self):
        """VSA-only configs do not change _attn_implementation."""
        model = type(
            "M",
            (),
            {"config": type("C", (), {"_attn_implementation": "eager"})()},
        )()
        config = type("Cfg", (), {"sparse_cfg": {}})()
        config.sparse_cfg = {"*": {"method": "vsa"}}
        _set_attn_implementation(model, config)
        # Should remain eager — VSA patches SDPA directly
        assert model.config._attn_implementation == "eager"

    def test_mixed_vsa_and_non_vsa_raises(self):
        """VSA + non-VSA methods are rejected."""
        model = type("M", (), {"config": None})()
        config = type("Cfg", (), {"sparse_cfg": {}})()
        config.sparse_cfg = {
            "layer1": {"method": "vsa"},
            "layer2": {"method": "flash_skip_softmax", "backend": "pytorch"},
        }
        with pytest.raises(ValueError, match="Cannot mix VSA"):
            _set_attn_implementation(model, config)


class TestExportSparseAttentionConfig:
    """Cover export_sparse_attention_config branches."""

    def test_returns_none_without_calibration(self):
        """When no module has calibration params, returns None."""
        model = SimpleAttentionModel()
        model = sparse_attn.sparsify(model, FLASH_SKIP_SOFTMAX_DEFAULT_CFG)
        out = export_sparse_attention_config(model)
        assert out is None

    def test_exports_when_calibration_present(self):
        """Calibration params are reflected in the exported config."""
        model = SimpleAttentionModel()
        model = sparse_attn.sparsify(model, FLASH_SKIP_SOFTMAX_DEFAULT_CFG)

        for module in model.modules():
            if isinstance(module, SparseAttentionModule):
                module._sparse_method_instance.calibration_params = {
                    "prefill": {"a": 3.14, "b": 7.5},
                    "decode": {"a": 0.5, "b": 9.0},
                }

        out = export_sparse_attention_config(model)
        assert out is not None
        assert "config_groups" in out
        tsf = out["threshold_scale_factor"]
        assert tsf["prefill"] == {"a": 3.14, "b": 7.5}
        assert tsf["decode"] == {"a": 0.5, "b": 9.0}
        assert out["producer"]["name"] == "modelopt"
