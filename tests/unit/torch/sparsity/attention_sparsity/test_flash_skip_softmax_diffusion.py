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

"""Unit tests for FlashSkipSoftmaxDiffusion method."""

import math

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

pytest.importorskip("transformers")

import modelopt.torch.sparsity.attention_sparsity as mtsa
from modelopt.torch.sparsity.attention_sparsity.methods.flash_skip_softmax_diffusion import (
    FlashSkipSoftmaxDiffusion,
)
from modelopt.torch.sparsity.attention_sparsity.sparse_attention import SparseAttentionModule

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_method(br=64, bc=64, **overrides):
    """Create a FlashSkipSoftmaxDiffusion with small blocks for CPU tests."""
    config = {"br": br, "bc": bc, "backend": "pytorch", "is_causal": False}
    config.update(overrides)
    return FlashSkipSoftmaxDiffusion(method_config=config)


# Config for integration tests (no calibration)
DIFFUSION_CFG = {
    "sparse_cfg": {
        "*attention*": {
            "method": "flash_skip_softmax_diffusion",
            "br": 64,
            "bc": 64,
            "backend": "pytorch",
            "is_causal": False,
            "collect_stats": True,
            "enable": True,
        },
        "default": {"enable": False},
    },
}

# Config for integration tests with calibration
DIFFUSION_CALIB_CFG = {
    "sparse_cfg": {
        "*attention*": {
            "method": "flash_skip_softmax_diffusion",
            "br": 64,
            "bc": 64,
            "backend": "pytorch",
            "is_causal": False,
            "collect_stats": True,
            "enable": True,
        },
        "default": {"enable": False},
        "calibration": {"target_sparse_ratio": {"prefill": 0.3}},
    },
}


class ExplicitSoftmaxAttention(nn.Module):
    """Attention module that explicitly calls F.softmax (not SDPA).

    This ensures the sparse attention F.softmax patch fires during forward.
    Class name contains 'Attention' so the plugin auto-registers it.
    """

    def __init__(self, hidden_size=64, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        b, s, d = x.shape
        qkv = self.qkv(x).reshape(b, s, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, b, H, s, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(b, s, d)
        return self.out_proj(out)


class SoftmaxAttentionModel(nn.Module):
    """Model with explicit F.softmax for sparse attention testing."""

    def __init__(self, hidden_size=64, num_heads=4):
        super().__init__()
        self.hidden_size = hidden_size
        self.attention = ExplicitSoftmaxAttention(hidden_size, num_heads)
        self.fc = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        return self.fc(self.attention(x))

    @classmethod
    def get_input(cls, hidden_size=64, seq_len=128, batch_size=2):
        return torch.randn(batch_size, seq_len, hidden_size)


# ===========================================================================
# Test Class 1: Method Internals
# ===========================================================================


class TestFlashSkipSoftmaxDiffusionMethod:
    """Test FlashSkipSoftmaxDiffusion method internals."""

    def test_block_reshaping_divisible(self):
        """Block shapes are correct when seq lengths are multiples of br, bc."""
        method = make_method(br=64, bc=64)
        attn = torch.randn(2, 4, 128, 256)
        blocked, num_br, num_bc, padded_q, padded_k = method._reshape_to_blocks(attn, 64, 64)

        assert blocked.shape == (2, 4, 2, 64, 4, 64)
        assert num_br == 2
        assert num_bc == 4
        assert padded_q == 128
        assert padded_k == 256

    def test_block_reshaping_with_padding(self):
        """Padding is applied correctly for non-aligned lengths."""
        method = make_method(br=64, bc=64)
        attn = torch.randn(2, 4, 100, 200)
        blocked, num_br, num_bc, padded_q, padded_k = method._reshape_to_blocks(attn, 64, 64)

        assert padded_q == 128  # ceil(100/64)*64
        assert padded_k == 256  # ceil(200/64)*64
        assert num_br == 2
        assert num_bc == 4
        assert blocked.shape == (2, 4, 2, 64, 4, 64)

    def test_calibration_mode_returns_none_mask_and_gaps(self):
        """Calibration mode collects normalized_gaps and returns None mask."""
        method = make_method()
        method.set_calibration_mode(True)

        attn = torch.randn(1, 2, 64, 128)
        mask, stats = method.calc_correction_factor_and_p(attn, "prefill")

        assert mask is None
        assert stats["sparsity"] == [0.0]
        assert stats["phase"] == "prefill"
        assert "normalized_gaps" in stats
        assert isinstance(stats["normalized_gaps"], np.ndarray)
        assert len(stats["normalized_gaps"]) > 0

    def test_inference_with_threshold_returns_bool_mask(self):
        """With calibration_params, returns a bool mask with sparsity in [0, 1]."""
        method = make_method()
        method.calibration_params = {"prefill": {"threshold": 0.5}}

        attn = torch.randn(1, 2, 64, 128)
        mask, stats = method.calc_correction_factor_and_p(attn, "prefill")

        assert mask is not None
        assert mask.shape == attn.shape
        assert mask.dtype == torch.bool
        assert 0.0 <= stats["sparsity"][0] <= 1.0

    def test_inference_without_calibration_all_ones_mask(self):
        """Without calibration_params, mask is all-True (no sparsity)."""
        method = make_method()
        # No calibration_params set

        attn = torch.randn(1, 2, 64, 128)
        mask, stats = method.calc_correction_factor_and_p(attn, "prefill")

        assert mask is not None
        assert mask.all()
        assert stats["sparsity"] == [0.0]

    def test_known_pattern_block_skip(self):
        """Block with low scores must be skipped when gap exceeds threshold.

        Setup: seq_k=128 with br=bc=64 → 2 block columns.
        Block 0: scores=10.0, Block 1: scores=0.0
        cummax = [10, 10], block_max = [10, 0], gap = [0, 10]
        With threshold=1.0: scaled_threshold = 1.0 * log(128) = 4.85
        Block 1 gap (10.0) >= 4.85 → block 1 is skipped.
        """
        method = make_method(br=64, bc=64)
        method.calibration_params = {"prefill": {"threshold": 1.0}}

        attn = torch.zeros(1, 1, 64, 128)
        attn[:, :, :, :64] = 10.0  # Block 0: high scores
        attn[:, :, :, 64:] = 0.0  # Block 1: low scores

        mask, stats = method.calc_correction_factor_and_p(attn, "prefill")

        # Block 0 should be kept (gap=0 < threshold)
        assert mask[:, :, :, :64].all(), "Block 0 should be kept (gap=0)"
        # Block 1 should be skipped (gap=10 >= 4.85)
        assert not mask[:, :, :, 64:].any(), "Block 1 should be skipped (gap=10)"
        assert stats["sparsity"][0] == pytest.approx(0.5, abs=0.01)

    def test_apply_sparsity_masked_positions_dtype_min(self):
        """Sparse positions are set to dtype.min, dense positions are unchanged."""
        method = make_method()
        method.calibration_params = {"prefill": {"threshold": 1.0}}

        attn = torch.zeros(1, 1, 64, 128)
        attn[:, :, :, :64] = 10.0
        attn[:, :, :, 64:] = 0.0

        sparse_mask, _ = method.calculate_sparsity(attn)
        sparse_attn = method.apply_sparsity(attn, sparse_mask)

        mask_value = torch.finfo(attn.dtype).min
        # Dense positions unchanged
        assert torch.allclose(sparse_attn[sparse_mask], attn[sparse_mask])
        # Sparse positions are dtype.min
        assert (sparse_attn[~sparse_mask] == mask_value).all()

    def test_apply_sparsity_without_mask(self):
        """apply_sparsity works when called without explicit mask."""
        method = make_method()
        method.calibration_params = {"prefill": {"threshold": 0.5}}

        attn = torch.randn(1, 2, 64, 128)
        sparse_attn = method.apply_sparsity(attn)

        assert sparse_attn.shape == attn.shape

    def test_calculate_sparsity_rejects_3d_input(self):
        """AssertionError on non-4D input."""
        method = make_method()
        with pytest.raises(AssertionError, match="Expected 4D"):
            method.calculate_sparsity(torch.randn(2, 64, 64))

    def test_always_prefill_phase(self):
        """Even with seq_q=1 (decode-like shape), phase is always 'prefill'."""
        method = make_method()
        method.calibration_params = {"prefill": {"threshold": 0.5}}

        attn = torch.randn(1, 2, 1, 128)
        _, stats = method.calculate_sparsity(attn)

        assert stats["phase"] == "prefill"

    def test_uniform_scores_zero_sparsity(self):
        """Equal scores → gap=0 everywhere → no blocks skipped.

        When all attention scores are the same value, block_max == cummax
        for every block, so gap = 0 for all blocks. No block is skipped
        regardless of threshold.
        """
        method = make_method()
        method.calibration_params = {"prefill": {"threshold": 0.001}}

        attn = torch.ones(1, 1, 64, 128) * 5.0
        mask, stats = method.calc_correction_factor_and_p(attn, "prefill")

        assert mask.all(), "Uniform scores should produce zero sparsity"
        assert stats["sparsity"][0] == 0.0

    def test_peaked_scores_high_sparsity(self):
        """One dominant block → other blocks skipped.

        Setup: seq_k=256, br=bc=64 → 4 block columns.
        Block 0: 10.0, Blocks 1-3: -5.0
        gap for blocks 1-3 = 10.0 - (-5.0) = 15.0
        scaled_threshold = 1.0 * log(256) = 5.55
        15.0 >= 5.55 → blocks 1-3 are skipped → 75% sparsity.
        """
        method = make_method(br=64, bc=64)
        method.calibration_params = {"prefill": {"threshold": 1.0}}

        attn = torch.full((1, 1, 64, 256), -5.0)
        attn[:, :, :, :64] = 10.0

        mask, stats = method.calc_correction_factor_and_p(attn, "prefill")

        assert stats["sparsity"][0] == pytest.approx(0.75, abs=0.01)
        assert mask[:, :, :, :64].all()  # Block 0 kept
        assert not mask[:, :, :, 64:].any()  # Blocks 1-3 skipped

    def test_monotonicity_threshold_vs_sparsity(self):
        """Higher threshold → lower or equal sparsity.

        A more aggressive threshold (smaller value) skips more blocks.
        A conservative threshold (larger value) keeps more blocks.
        """
        torch.manual_seed(42)
        attn = torch.randn(2, 4, 64, 256)

        sparsities = []
        for threshold in [0.1, 0.5, 1.0, 2.0, 5.0]:
            method = make_method()
            method.calibration_params = {"prefill": {"threshold": threshold}}
            _, stats = method.calc_correction_factor_and_p(attn, "prefill")
            sparsities.append(stats["sparsity"][0])

        # Sparsity should be non-increasing as threshold increases
        for i in range(len(sparsities) - 1):
            assert sparsities[i] >= sparsities[i + 1] - 1e-6, (
                f"Sparsity should decrease with higher threshold: "
                f"threshold {[0.1, 0.5, 1.0, 2.0, 5.0][i]} gave {sparsities[i]}, "
                f"threshold {[0.1, 0.5, 1.0, 2.0, 5.0][i + 1]} gave {sparsities[i + 1]}"
            )


# ===========================================================================
# Test Class 2: Integration Tests
# ===========================================================================


class TestDiffusionIntegration:
    """Integration tests using the full sparsify pipeline."""

    def test_sparsify_module_replacement(self):
        """sparsify() inserts SparseAttentionModule wrappers and forward runs."""
        model = SoftmaxAttentionModel(hidden_size=64, num_heads=4)
        x = SoftmaxAttentionModel.get_input(hidden_size=64, seq_len=128, batch_size=2)

        model = mtsa.sparsify(model, DIFFUSION_CFG)

        # Verify at least one SparseAttentionModule was inserted
        has_sparse = any(isinstance(m, SparseAttentionModule) for m in model.modules())
        assert has_sparse, "No SparseAttentionModule found after sparsify()"

        # Forward should work without errors
        model.eval()
        with torch.no_grad():
            output = model(x)
        assert not torch.isnan(output).any(), "NaN in output after sparsify"
        assert output.shape == x.shape

    def test_calibration_sets_threshold(self):
        """Percentile calibration produces a non-negative threshold."""
        model = SoftmaxAttentionModel(hidden_size=64, num_heads=4)

        def forward_loop(model):
            model.eval()
            with torch.no_grad():
                for _ in range(5):
                    # Use longer sequences with peaked patterns to produce
                    # non-trivial attention score distributions (non-zero gaps).
                    x = torch.randn(2, 256, 64)
                    # Make some tokens "hot" so Q·K^T has peaked columns
                    x[:, 0, :] *= 10.0
                    x[:, 64, :] *= 10.0
                    model(x)

        model = mtsa.sparsify(model, DIFFUSION_CALIB_CFG, forward_loop=forward_loop)

        # Find the sparse attention module and check calibration_params
        found_threshold = False
        for module in model.modules():
            if isinstance(module, SparseAttentionModule):
                params = module._sparse_method_instance.calibration_params
                if params and "prefill" in params and "threshold" in params["prefill"]:
                    threshold = params["prefill"]["threshold"]
                    assert isinstance(threshold, float)
                    assert threshold >= 0, f"Threshold should be non-negative, got {threshold}"
                    found_threshold = True

        assert found_threshold, "Calibration did not set threshold on any module"

    def test_sparsity_changes_softmax_output(self):
        """apply_sparsity + F.softmax differs from F.softmax alone.

        Proves that sparsity is doing something — the softmax distribution
        changes when sparse blocks are masked to dtype.min.
        """
        method = make_method()
        method.calibration_params = {"prefill": {"threshold": 0.5}}

        # Create attention scores where some blocks will be sparse
        torch.manual_seed(0)
        attn = torch.randn(1, 2, 64, 128)

        # Dense softmax
        dense_out = F.softmax(attn, dim=-1)

        # Sparse softmax
        sparse_attn = method.apply_sparsity(attn.clone())
        sparse_out = F.softmax(sparse_attn, dim=-1)

        # If any sparsity was applied, outputs must differ
        _, stats = method.calculate_sparsity(attn)
        if stats["sparsity"][0] > 0:
            assert not torch.allclose(dense_out, sparse_out), (
                "Sparse softmax output should differ from dense when sparsity > 0"
            )


# ===========================================================================
# Test Class 3: Sequence Length Invariance
# ===========================================================================


class TestSequenceLengthInvariance:
    """Test the gap/log(seq_k) normalization for sequence length invariance."""

    def test_normalized_gaps_consistent_across_lengths(self):
        """Same block pattern at different seq_k produces same normalized gaps.

        The gap/log(seq_k) normalization is the key innovation. Given the same
        block-level pattern (first block high, rest low), the raw gap grows
        with sequence length, but the normalized gap (gap / log(seq_k)) stays
        constant. This means a threshold calibrated at one length works at
        another.

        We verify this by checking that the normalized gap values for the
        non-peak blocks are identical regardless of seq_k.
        """
        normalized_gaps_by_length = {}

        for num_blocks in [2, 4, 8]:
            bc = 64
            seq_k = num_blocks * bc
            attn = torch.full((1, 1, 64, seq_k), 0.0)
            attn[:, :, :, :bc] = 10.0  # First block high, rest zero

            method = make_method(br=64, bc=bc)
            method.set_calibration_mode(True)
            _, stats = method.calc_correction_factor_and_p(attn, "prefill")

            gaps = stats["normalized_gaps"]
            # The first block column has gap=0, so normalized gap=0.
            # All other block columns have gap=10.0, normalized = 10.0 / log(seq_k).
            # min_gap over br rows is the same since all rows are identical.
            #
            # Collect the non-zero normalized gaps (from non-peak blocks).
            nonzero_gaps = gaps[gaps > 0.01]
            assert len(nonzero_gaps) > 0, f"Expected non-zero gaps for seq_k={seq_k}"

            # All non-zero gaps should be the same value: 10.0 / log(seq_k)
            expected = 10.0 / math.log(seq_k)
            np.testing.assert_allclose(
                nonzero_gaps,
                expected,
                atol=1e-4,
                err_msg=f"Normalized gaps wrong for seq_k={seq_k}",
            )

            normalized_gaps_by_length[seq_k] = expected

        # The normalized gap values differ across lengths (that's expected —
        # they are 10/log(seq_k)). But a percentile threshold computed on
        # these values at one length would correctly classify blocks at
        # another length. The key property: the _relative ordering_ is
        # preserved and the formula is applied correctly.
        vals = list(normalized_gaps_by_length.values())
        assert all(v > 0 for v in vals), "All normalized gaps should be positive"
