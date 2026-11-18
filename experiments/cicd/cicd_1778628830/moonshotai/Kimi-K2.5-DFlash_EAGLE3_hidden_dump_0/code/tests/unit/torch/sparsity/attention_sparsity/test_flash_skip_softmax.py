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

"""Unit tests for FlashSkipSoftmax method internals."""

import pytest
import torch

pytest.importorskip("transformers")

from modelopt.torch.sparsity.attention_sparsity.methods.flash_skip_softmax import FlashSkipSoftmax


class TestFlashSkipSoftmaxMethod:
    """Test FlashSkipSoftmax method internals."""

    def test_phase_inference(self):
        """Test phase detection from attention score shape."""
        method = FlashSkipSoftmax(
            {
                "thresholds": {"prefill": [1e-3], "decode": [1e-4]},
                "br": 128,
                "bc": 128,
                "backend": "pytorch",
                "is_causal": True,
            }
        )

        # Prefill: seq_q > 1
        prefill_scores = torch.randn(2, 4, 64, 64)
        assert method._infer_phase(prefill_scores) == "prefill"

        # Decode: seq_q = 1
        decode_scores = torch.randn(2, 4, 1, 64)
        assert method._infer_phase(decode_scores) == "decode"

    def test_threshold_update_dict_config(self):
        """Test threshold updates with dict config."""
        method = FlashSkipSoftmax(
            {
                "thresholds": {"prefill": [1e-3], "decode": [1e-5]},
                "br": 128,
                "bc": 128,
                "backend": "pytorch",
                "is_causal": True,
            }
        )

        # Initially uses prefill thresholds
        initial_thresholds = method.thresholds

        # Update to decode
        method._update_thresholds("decode")
        assert method.thresholds == [1e-5]
        assert method.thresholds != initial_thresholds

        # Update back to prefill
        method._update_thresholds("prefill")
        assert method.thresholds == [1e-3]

    def test_block_reshaping_divisible(self):
        """Test block reshaping with divisible sequence lengths."""
        method = FlashSkipSoftmax(
            {
                "thresholds": {"prefill": [1e-3], "decode": [1e-4]},
                "br": 128,
                "bc": 128,
                "backend": "pytorch",
                "is_causal": True,
            }
        )

        # Seq lengths divisible by 128
        attn = torch.randn(2, 4, 256, 256)
        blocked, num_br, num_bc, padded_q, padded_k = method._reshape_to_blocks(attn, 128, 128)

        # Verify block dimensions
        assert blocked.shape == (2, 4, 2, 128, 2, 128)  # 256/128 = 2 blocks
        assert num_br == 2
        assert num_bc == 2
        assert padded_q == 256  # No padding
        assert padded_k == 256  # No padding

    def test_block_reshaping_with_padding(self):
        """Test block reshaping with non-divisible lengths."""
        method = FlashSkipSoftmax(
            {
                "thresholds": {"prefill": [1e-3], "decode": [1e-4]},
                "br": 128,
                "bc": 128,
                "backend": "pytorch",
                "is_causal": True,
            }
        )

        # Seq lengths NOT divisible by 128
        attn = torch.randn(2, 4, 200, 300)
        blocked, num_br, num_bc, padded_q, padded_k = method._reshape_to_blocks(attn, 128, 128)

        # Verify padding applied
        assert padded_q == 256  # ceil(200/128) * 128 = 2 * 128
        assert padded_k == 384  # ceil(300/128) * 128 = 3 * 128
        assert num_br == 2
        assert num_bc == 3
        assert blocked.shape == (2, 4, 2, 128, 3, 128)

    def test_correction_factor_calculation_prefill(self):
        """Test correction factor for prefill phase."""
        method = FlashSkipSoftmax(
            {
                "thresholds": {"prefill": [1e-3], "decode": [1e-4]},
                "br": 128,
                "bc": 128,
                "backend": "pytorch",
                "is_causal": True,
            }
        )

        # Create simple attention pattern
        attn = torch.randn(1, 1, 128, 256)

        mask, stats = method.calc_correction_factor_and_p(attn, "prefill")

        # Verify stats structure
        assert "correction_factor" in stats
        assert "sparsity" in stats
        assert "phase" in stats
        assert "total_blocks" in stats
        assert stats["phase"] == "prefill"
        assert 0 <= stats["correction_factor"] <= 1
        # sparsity is now a list (one entry per threshold)
        assert isinstance(stats["sparsity"], list)
        assert all(-1 <= s <= 1 for s in stats["sparsity"])

    def test_correction_factor_calculation_decode(self):
        """Test correction factor for decode phase."""
        method = FlashSkipSoftmax(
            {
                "thresholds": {"prefill": [1e-3], "decode": [1e-5]},
                "br": 128,
                "bc": 128,
                "backend": "pytorch",
                "is_causal": True,
            }
        )

        # Decode: single query
        attn = torch.randn(1, 1, 1, 256)

        mask, stats = method.calc_correction_factor_and_p(attn, "decode")

        # Verify stats structure
        assert stats["phase"] == "decode"
        assert "correction_factor" in stats
        assert isinstance(stats["sparsity"], list)
        assert all(0 <= s <= 1 for s in stats["sparsity"])
        assert mask.shape == (1, 1, 1, 256)

    def test_block_mask_correctness(self):
        """Test block mask shape and type."""
        method = FlashSkipSoftmax(
            {
                "thresholds": {"prefill": [1e-3], "decode": [1e-4]},
                "br": 128,
                "bc": 128,
                "backend": "pytorch",
                "is_causal": True,
            }
        )

        attn = torch.randn(2, 4, 128, 256)
        mask, _ = method.calc_correction_factor_and_p(attn, "prefill")

        # Verify mask properties
        assert mask.shape == attn.shape
        assert mask.dtype == torch.bool
        assert mask.device == attn.device

    def test_causal_vs_noncausal(self):
        """Test total_blocks calculation for causal vs non-causal."""
        config_base = {
            "thresholds": {"prefill": [1e-3], "decode": [1e-4]},
            "br": 128,
            "bc": 128,
            "backend": "pytorch",
        }

        method_causal = FlashSkipSoftmax({**config_base, "is_causal": True})
        method_noncausal = FlashSkipSoftmax({**config_base, "is_causal": False})

        attn = torch.randn(1, 1, 256, 256)  # 2x2 blocks

        _, stats_causal = method_causal.calc_correction_factor_and_p(attn, "prefill")
        _, stats_noncausal = method_noncausal.calc_correction_factor_and_p(attn, "prefill")

        # Causal: 2*(2+1)/2 = 3 blocks
        # Non-causal: 2*2 = 4 blocks
        assert stats_causal["total_blocks"] == 3
        assert stats_noncausal["total_blocks"] == 4

    def test_calculate_sparsity_assertions(self):
        """Test calculate_sparsity input validation."""
        method = FlashSkipSoftmax(
            {
                "thresholds": {"prefill": [1e-3], "decode": [1e-4]},
                "br": 128,
                "bc": 128,
                "backend": "pytorch",
                "is_causal": True,
            }
        )

        # Test: 4D shape required
        with pytest.raises(AssertionError, match="Expected 4D"):
            method.calculate_sparsity(attention_scores=torch.randn(2, 64, 64))  # 3D

    def test_apply_sparsity_with_mask(self):
        """Test apply_sparsity with pre-computed mask."""
        method = FlashSkipSoftmax(
            {
                "thresholds": {"prefill": [1e-3], "decode": [1e-4]},
                "br": 128,
                "bc": 128,
                "backend": "pytorch",
                "is_causal": True,
            }
        )

        attn = torch.randn(2, 4, 128, 256)

        # Calculate sparsity first
        sparse_mask, stats = method.calculate_sparsity(attn)

        # Apply sparsity with pre-computed mask
        sparse_attn = method.apply_sparsity(attn, sparse_mask)

        # Verify output shape matches input
        assert sparse_attn.shape == attn.shape

        # Verify masked positions have min value
        mask_value = torch.finfo(attn.dtype).min
        assert (sparse_attn[~sparse_mask] == mask_value).all()

    def test_apply_sparsity_without_mask(self):
        """Test apply_sparsity calculates mask internally when None."""
        method = FlashSkipSoftmax(
            {
                "thresholds": {"prefill": [1e-3], "decode": [1e-4]},
                "br": 128,
                "bc": 128,
                "backend": "pytorch",
                "is_causal": True,
            }
        )

        attn = torch.randn(2, 4, 128, 256)

        # Apply sparsity without pre-computed mask
        sparse_attn = method.apply_sparsity(attn)

        # Verify output shape matches input
        assert sparse_attn.shape == attn.shape

    def test_calibrated_path_prefill(self):
        """Dynamic calibrated threshold path is exercised when params/targets are set."""
        method = FlashSkipSoftmax(
            {
                "thresholds": {"prefill": [1e-3], "decode": [1e-4]},
                "br": 128,
                "bc": 128,
                "backend": "pytorch",
                "is_causal": False,
            }
        )
        method.calibration_params = {"prefill": {"a": 1.0, "b": 5.0}}
        method.target_sparse_ratio = {"prefill": 0.5}

        attn = torch.randn(1, 2, 128, 256)
        mask, stats = method.calc_correction_factor_and_p(attn, "prefill")
        # calibrated single-threshold path yields one sparsity entry
        assert len(stats["sparsity"]) == 1
        assert mask is not None

    def test_calibrated_path_decode(self):
        """Decode with calibrated params."""
        method = FlashSkipSoftmax(
            {
                "thresholds": {"prefill": [1e-3], "decode": [1e-4]},
                "br": 128,
                "bc": 128,
                "backend": "pytorch",
                "is_causal": False,
            }
        )
        method.calibration_params = {"decode": {"a": 0.5, "b": 4.0}}
        method.target_sparse_ratio = {"decode": 0.6}

        attn = torch.randn(1, 2, 1, 256)
        mask, stats = method.calc_correction_factor_and_p(attn, "decode")
        assert stats["phase"] == "decode"
        assert len(stats["sparsity"]) == 1

    def test_get_threshold_info_calibrated(self):
        """get_threshold_info returns dynamic_calibrated type when calibrated."""
        method = FlashSkipSoftmax(
            {
                "thresholds": {"prefill": [1e-3], "decode": [1e-4]},
                "br": 128,
                "bc": 128,
                "backend": "pytorch",
                "is_causal": True,
            }
        )
        method.calibration_params = {"prefill": {"a": 1.0, "b": 5.0}}
        method.target_sparse_ratio = {"prefill": 0.5}
        info = method.get_threshold_info()
        assert info["type"] == "dynamic_calibrated"
        assert "phases" in info
        assert "prefill" in info["phases"]
        assert "scale_factor" in info["phases"]["prefill"]

    def test_get_threshold_info_static(self):
        """get_threshold_info returns static type when no calibration."""
        method = FlashSkipSoftmax(
            {
                "thresholds": {"prefill": [1e-3], "decode": [1e-4]},
                "br": 128,
                "bc": 128,
                "backend": "pytorch",
                "is_causal": True,
            }
        )
        info = method.get_threshold_info()
        assert info["type"] == "static"
        assert "value" in info

    def test_get_sparse_context_patches_softmax(self):
        """get_sparse_context returns an ExitStack that patches F.softmax."""
        import torch.nn.functional as F

        method = FlashSkipSoftmax(
            {
                "thresholds": {"prefill": [1e-3], "decode": [1e-4]},
                "br": 64,
                "bc": 64,
                "backend": "pytorch",
                "is_causal": True,
            }
        )

        module = type("M", (), {"_last_stats": None})()
        original_softmax = F.softmax
        stack = method.get_sparse_context(module)
        with stack:
            # Inside the context, softmax should be patched
            assert F.softmax is not original_softmax
            # Call it once to exercise the sparse_softmax wrapper
            scores = torch.randn(1, 1, 64, 64)
            F.softmax(scores, dim=-1)
            assert module._last_stats is not None

        # After the context, softmax is restored
        assert F.softmax is original_softmax

    def test_calibration_mode_skips_apply(self):
        """In calibration mode, sparse_softmax wrapper does not apply mask."""
        import torch.nn.functional as F

        method = FlashSkipSoftmax(
            {
                "thresholds": {"prefill": [1e-3], "decode": [1e-4]},
                "br": 64,
                "bc": 64,
                "backend": "pytorch",
                "is_causal": True,
            }
        )
        method.set_calibration_mode(True)
        module = type("M", (), {"_last_stats": None})()

        with method.get_sparse_context(module):
            scores = torch.randn(1, 1, 64, 64)
            # Should not apply sparsity — output is regular softmax
            out = F.softmax(scores, dim=-1)
            assert torch.allclose(out.sum(dim=-1), torch.ones_like(out.sum(dim=-1)))
        method.set_calibration_mode(False)
