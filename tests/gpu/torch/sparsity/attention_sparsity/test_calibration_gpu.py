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

"""GPU tests for sparse attention calibration."""

import pytest
import torch
from _test_utils.torch.sparsity.sparse_attention_common import SimpleTransformerEncoderLayer

import modelopt.torch.opt as mto
from modelopt.torch.sparsity.attention_sparsity import sparsify
from modelopt.torch.sparsity.attention_sparsity.calibration import RulerDatasetBuilder
from modelopt.torch.sparsity.attention_sparsity.sparse_attention import SparseAttentionModule

# Skip all tests if no GPU available
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")


class TestRulerDatasetBuilderGPU:
    """Test RULER dataset generation with real tokenizers on GPU."""

    def test_ruler_generation_with_real_tokenizer(self):
        """Test RULER generation with GPT2 tokenizer."""
        builder = RulerDatasetBuilder(
            samples=6,  # Need at least 6 samples (1 per task)
            max_seqlen=1024,  # Generates: [1024]
            tokenizer_name_or_path="gpt2",
            seed=42,
        )

        dataset = builder.build_calibration_dataset()

        # Should generate 6 samples (1 per task)
        assert len(dataset) == 6

        # All samples should have valid structure
        for sample in dataset:
            assert "input" in sample
            assert "length" in sample
            assert sample["length"] > 0

    def test_generated_length_accuracy(self):
        """Test that generated token counts are accurate."""
        builder = RulerDatasetBuilder(
            samples=3,
            max_seqlen=1024,  # Generates: [1024]
            tokenizer_name_or_path="gpt2",
            seed=42,
        )

        dataset = builder.build_calibration_dataset()

        # Check that lengths are within reasonable range of target
        for sample in dataset:
            # RULER aims for 70-90% of target for context
            assert 700 < sample["length"] < 1400

    def test_multiple_subtasks(self):
        """Test generation with multiple RULER subtasks."""
        builder = RulerDatasetBuilder(
            samples=12,  # Need at least 6 for 1 per task, use 12 for 2 per task
            max_seqlen=1024,  # Generates: [1024]
            tokenizer_name_or_path="gpt2",
            seed=42,
        )

        dataset = builder.build_calibration_dataset()

        # Check task distribution (should have multiple tasks from RULER_TASKS)
        tasks_found = {s["task"] for s in dataset}
        assert len(tasks_found) >= 2  # At least 2 different tasks

    def test_large_context_lengths(self):
        """Test with larger context lengths."""
        builder = RulerDatasetBuilder(
            samples=24,  # 4 lengths * 6 tasks = need 24 for 1 per (length, task)
            max_seqlen=8192,  # Generates: [8192, 4096, 2048, 1024]
            tokenizer_name_or_path="gpt2",
            seed=42,
        )

        dataset = builder.build_calibration_dataset()

        assert len(dataset) == 24

        # Verify we have different lengths
        lengths = [s["length"] for s in dataset]
        # Should have variety of lengths across the bins
        assert len(set(lengths)) > 1  # At least 2 different target lengths used


class TestCalibrationGPU:
    """Test calibration with real models on GPU."""

    @pytest.fixture
    def simple_model(self):
        """Create simple attention model for testing."""
        model = SimpleTransformerEncoderLayer(d_model=256, nhead=8).cuda()
        return model

    def test_calibration_simple_model(self, simple_model):
        """Test calibration with simple attention model."""
        model = simple_model

        config = {
            "sparse_cfg": {
                "*attn*": {
                    "method": "flash_skip_softmax",
                    "threshold": {"prefill": 1e-3, "decode": 1e-4},
                    "br": 64,
                    "bc": 64,
                    "backend": "pytorch",
                    "enable": True,
                    "calibration": {
                        "target_sparse_ratio": {"prefill": 0.5, "decode": 0.0},
                        "samples": 4,
                        "max_seqlen": 1024,
                    },
                }
            },
        }

        def forward_loop(model):
            # Simple forward loop for calibration
            pass

        # Apply sparse attention with calibration
        sparse_model = sparsify(model, config, forward_loop=forward_loop)

        # Verify sparse attention modules exist
        sparse_modules = [m for m in sparse_model.modules() if isinstance(m, SparseAttentionModule)]
        assert len(sparse_modules) > 0

        # Verify calibration was applied
        for module in sparse_modules:
            method = module._sparse_method_instance
            # Check if calibrated threshold scale factor is set
            if hasattr(method, "threshold_scale_factor") and method.threshold_scale_factor:
                assert method.threshold_scale_factor > 0

    def test_calibration_pytorch_backend(self, simple_model):
        """Test calibration with pytorch backend."""
        model = simple_model

        config = {
            "sparse_cfg": {
                "*attn*": {
                    "method": "flash_skip_softmax",
                    "threshold": {"prefill": 1e-3, "decode": 1e-4},
                    "backend": "pytorch",
                    "enable": True,
                    "calibration": {
                        "target_sparse_ratio": {"prefill": 0.5, "decode": 0.0},
                        "samples": 2,
                        "max_seqlen": 1024,
                    },
                }
            },
        }

        def forward_loop(model):
            pass

        sparse_model = sparsify(model, config, forward_loop=forward_loop)

        # Check backend is set correctly
        for module in sparse_model.modules():
            if isinstance(module, SparseAttentionModule):
                method = module._sparse_method_instance
                assert hasattr(method, "backend")
                assert method.backend == "pytorch"

    def test_simplified_calibration(self, simple_model):
        """Test simplified calibration (prefill phase only)."""
        model = simple_model

        config = {
            "sparse_cfg": {
                "*attn*": {
                    "method": "flash_skip_softmax",
                    "threshold": {"prefill": 1e-3, "decode": 1e-4},
                    "enable": True,
                    "calibration": {
                        "target_sparse_ratio": {"prefill": 0.5, "decode": 0.0},
                        "samples": 4,
                        "max_seqlen": 1024,
                    },
                }
            },
        }

        def forward_loop(model):
            pass

        sparse_model = sparsify(model, config, forward_loop=forward_loop)

        # Should complete without errors
        assert sparse_model is not None

    def test_calibration_persistence(self, simple_model):
        """Test save and restore of calibrated model."""
        model = simple_model

        config = {
            "sparse_cfg": {
                "*attn*": {
                    "method": "flash_skip_softmax",
                    "threshold": {"prefill": 1e-3, "decode": 1e-4},
                    "enable": True,
                    "calibration": {
                        "target_sparse_ratio": {"prefill": 0.5, "decode": 0.0},
                        "samples": 2,
                        "max_seqlen": 1024,
                    },
                }
            },
        }

        def forward_loop(model):
            pass

        # Calibrate model
        sparse_model = sparsify(model, config, forward_loop=forward_loop)

        # Save modelopt state
        modelopt_state = mto.modelopt_state(sparse_model)

        # Create new model and restore
        model_restored = SimpleTransformerEncoderLayer(d_model=256, nhead=8).cuda()

        restored = mto.restore_from_modelopt_state(model_restored, modelopt_state)

        # Check that sparse attention is restored
        has_sparse = any(isinstance(m, SparseAttentionModule) for m in restored.modules())
        assert has_sparse


class TestCalibrationEndToEnd:
    """Integration tests with inference."""

    @pytest.fixture
    def simple_model_setup(self):
        """Setup simple model."""
        model = SimpleTransformerEncoderLayer(d_model=256, nhead=8).cuda()
        return model

    def test_calibrated_model_inference(self, simple_model_setup):
        """Test inference with calibrated model."""
        model = simple_model_setup

        config = {
            "sparse_cfg": {
                "*attn*": {
                    "method": "flash_skip_softmax",
                    "threshold": {"prefill": 1e-3, "decode": 1e-4},
                    "backend": "pytorch",
                    "enable": True,
                    "calibration": {
                        "target_sparse_ratio": {"prefill": 0.5, "decode": 0.0},
                        "samples": 2,
                        "max_seqlen": 1024,
                    },
                }
            },
        }

        def forward_loop(model):
            pass

        # Calibrate model
        sparse_model = sparsify(model, config, forward_loop=forward_loop)

        # Test inference
        test_input = SimpleTransformerEncoderLayer.get_input(d_model=256, seq_len=10).cuda()

        sparse_model.eval()
        with torch.no_grad():
            output = sparse_model(test_input)

        # Check output is valid
        assert output is not None
        assert not torch.isnan(output).any()

    def test_calibrated_vs_fixed_threshold(self, simple_model_setup):
        """Compare calibrated vs fixed threshold models."""
        # Config with calibration
        config_calibrated = {
            "sparse_cfg": {
                "*attn*": {
                    "method": "flash_skip_softmax",
                    "threshold": {"prefill": 1e-3, "decode": 1e-4},
                    "enable": True,
                    "calibration": {
                        "target_sparse_ratio": {"prefill": 0.5, "decode": 0.0},
                        "samples": 2,
                        "max_seqlen": 1024,
                    },
                }
            },
        }

        # Config with fixed threshold (no calibration)
        config_fixed = {
            "sparse_cfg": {
                "*attn*": {
                    "method": "flash_skip_softmax",
                    "threshold": {"prefill": 1e-3, "decode": 1e-4},
                    "enable": True,
                }
            },
        }

        def forward_loop(model):
            pass

        # Test both can be created
        model_calibrated = sparsify(
            SimpleTransformerEncoderLayer(d_model=256, nhead=8).cuda(),
            config_calibrated,
            forward_loop=forward_loop,
        )

        model_fixed = sparsify(
            SimpleTransformerEncoderLayer(d_model=256, nhead=8).cuda(),
            config_fixed,
        )

        # Both should work for inference
        test_input = SimpleTransformerEncoderLayer.get_input(d_model=256, seq_len=10).cuda()

        with torch.no_grad():
            output_calibrated = model_calibrated(test_input)
            output_fixed = model_fixed(test_input)

        assert output_calibrated is not None
        assert output_fixed is not None

    def test_memory_usage(self, simple_model_setup):
        """Test that calibration doesn't cause memory issues."""
        model = simple_model_setup

        # Clear cache before test
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated()

        config = {
            "sparse_cfg": {
                "*attn*": {
                    "method": "flash_skip_softmax",
                    "threshold": {"prefill": 1e-3, "decode": 1e-4},
                    "enable": True,
                    "calibration": {
                        "target_sparse_ratio": {"prefill": 0.5, "decode": 0.0},
                        "samples": 2,
                        "max_seqlen": 1024,
                    },
                }
            },
        }

        def forward_loop(model):
            pass

        # Calibrate
        sparsify(model, config, forward_loop=forward_loop)

        # Check memory didn't explode
        final_memory = torch.cuda.memory_allocated()
        memory_increase = final_memory - initial_memory

        # Memory should be reasonable (not more than 2GB increase)
        assert memory_increase < 2 * 1024**3  # 2GB


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
