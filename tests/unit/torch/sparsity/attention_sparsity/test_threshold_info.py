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

"""Unit tests for threshold calibration functionality."""

import pytest

pytest.importorskip("transformers")

from _test_utils.torch_sparsity.sparse_attention_common import SimpleAttentionModel

from modelopt.torch.sparsity.attention_sparsity import sparsify
from modelopt.torch.sparsity.attention_sparsity.methods.flash_skip_softmax import FlashSkipSoftmax
from modelopt.torch.sparsity.attention_sparsity.sparse_attention import SparseAttentionModule


class TestFlashSkipSoftmaxThresholdInfo:
    """Test FlashSkipSoftmax.get_threshold_info() method."""

    def test_static_threshold(self):
        """Test threshold info for static threshold."""
        method = FlashSkipSoftmax(
            method_config={
                "threshold": 0.001,
                "br": 128,
                "bc": 128,
                "backend": "pytorch",
                "is_causal": True,
            }
        )

        info = method.get_threshold_info()

        assert info["type"] == "static"
        assert info["value"] == 0.001

    def test_phased_threshold(self):
        """Test threshold info for phase-specific thresholds."""
        method = FlashSkipSoftmax(
            method_config={
                "threshold": {"prefill": 0.001, "decode": 0.0001},
                "br": 128,
                "bc": 128,
                "backend": "pytorch",
                "is_causal": True,
            }
        )

        info = method.get_threshold_info()

        assert info["type"] == "static_phased"
        assert "thresholds" in info
        assert info["thresholds"]["prefill"] == 0.001
        assert info["thresholds"]["decode"] == 0.0001
        assert "current" in info

    def test_dynamic_calibrated_threshold(self):
        """Test threshold info for calibrated dynamic threshold."""
        method = FlashSkipSoftmax(
            method_config={
                "threshold": 0.001,
                "br": 128,
                "bc": 128,
                "backend": "pytorch",
                "is_causal": True,
            }
        )

        # Simulate calibration setting scale factor
        method.threshold_scale_factor = 437.5

        info = method.get_threshold_info()

        assert info["type"] == "dynamic"
        assert info["scale_factor"] == 437.5
        assert info["formula"] == "λ / length"
        assert "example_lengths" in info
        assert abs(info["example_lengths"][1024] - 437.5 / 1024) < 1e-6
        assert abs(info["example_lengths"][2048] - 437.5 / 2048) < 1e-6

    def test_threshold_info_structure(self):
        """Test that threshold info has expected structure."""
        method = FlashSkipSoftmax(
            method_config={
                "threshold": 0.001,
                "br": 128,
                "bc": 128,
                "backend": "pytorch",
                "is_causal": True,
            }
        )

        info = method.get_threshold_info()

        # Should always have 'type' key
        assert "type" in info
        assert isinstance(info, dict)


class TestSparseAttentionModuleThresholdInfo:
    """Test SparseAttentionModule.get_threshold_info() delegation."""

    def test_module_delegates_to_method(self):
        """Test that module correctly delegates to sparse method instance."""
        model = SimpleAttentionModel(hidden_size=64, num_heads=4)

        config = {
            "sparse_cfg": {
                "*attention*": {
                    "method": "flash_skip_softmax",
                    "threshold": 0.005,
                    "br": 64,
                    "bc": 64,
                    "enable": True,
                }
            },
        }

        sparse_model = sparsify(model, config)

        # Find sparse attention module
        sparse_module = None
        for module in sparse_model.modules():
            if isinstance(module, SparseAttentionModule):
                sparse_module = module
                break

        assert sparse_module is not None

        # Test get_threshold_info
        info = sparse_module.get_threshold_info()

        assert info["type"] == "static"
        assert info["value"] == 0.005

    def test_module_with_calibrated_threshold(self):
        """Test module reports calibrated threshold correctly."""
        model = SimpleAttentionModel(hidden_size=64, num_heads=4)

        config = {
            "sparse_cfg": {
                "*attention*": {
                    "method": "flash_skip_softmax",
                    "threshold": 0.001,
                    "br": 64,
                    "bc": 64,
                    "enable": True,
                }
            },
        }

        sparse_model = sparsify(model, config)

        # Find module and set calibrated threshold
        for module in sparse_model.modules():
            if isinstance(module, SparseAttentionModule):
                module._sparse_method_instance.threshold_scale_factor = 500.0
                break

        # Get threshold info
        info = module.get_threshold_info()

        assert info["type"] == "dynamic"
        assert info["scale_factor"] == 500.0

    def test_module_without_method_instance(self):
        """Test get_threshold_info when sparse method instance doesn't exist."""
        model = SimpleAttentionModel(hidden_size=64, num_heads=4)

        config = {
            "sparse_cfg": {
                "*attention*": {
                    "method": "flash_skip_softmax",
                    "threshold": 0.001,
                    "br": 64,
                    "bc": 64,
                    "enable": True,
                }
            },
        }

        sparse_model = sparsify(model, config)

        # Find module
        for module in sparse_model.modules():
            if isinstance(module, SparseAttentionModule):
                # Remove sparse method instance to test fallback
                delattr(module, "_sparse_method_instance")

                info = module.get_threshold_info()

                assert info["type"] == "none"
                assert info["value"] is None
                break


class TestPrintSparseAttentionSummaryIntegration:
    """Test integration with print_sparse_attention_summary."""

    def test_summary_displays_static_threshold(self, capsys):
        """Test that print function displays static thresholds."""
        from modelopt.torch.sparsity.attention_sparsity.conversion import (
            print_sparse_attention_summary,
        )

        model = SimpleAttentionModel(hidden_size=64, num_heads=4)

        config = {
            "sparse_cfg": {
                "*attention*": {
                    "method": "flash_skip_softmax",
                    "threshold": 0.001,
                    "br": 64,
                    "bc": 64,
                    "enable": True,
                }
            },
        }

        sparse_model = sparsify(model, config)
        print_sparse_attention_summary(sparse_model)

        captured = capsys.readouterr()
        assert "Static (1.00e-03)" in captured.out
        assert "flash_skip_softmax" in captured.out

    def test_summary_displays_dynamic_threshold(self, capsys):
        """Test that print function displays dynamic thresholds."""
        from modelopt.torch.sparsity.attention_sparsity.conversion import (
            print_sparse_attention_summary,
        )

        model = SimpleAttentionModel(hidden_size=64, num_heads=4)

        config = {
            "sparse_cfg": {
                "*attention*": {
                    "method": "flash_skip_softmax",
                    "threshold": 0.001,
                    "br": 64,
                    "bc": 64,
                    "enable": True,
                }
            },
        }

        sparse_model = sparsify(model, config)

        # Set calibrated threshold
        for module in sparse_model.modules():
            if isinstance(module, SparseAttentionModule):
                module._sparse_method_instance.threshold_scale_factor = 437.5

        print_sparse_attention_summary(sparse_model)

        captured = capsys.readouterr()
        assert "Dynamic (λ=437.500000)" in captured.out
        assert "flash_skip_softmax" in captured.out
