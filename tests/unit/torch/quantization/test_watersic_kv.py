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

"""Unit tests for the core ZSIC algorithm (WaterSIC KV-cache quantization)."""

from __future__ import annotations

import pytest
import torch

from modelopt.torch.quantization.algorithms.watersic_kv.zsic import (
    _compute_hessian_cholesky,
    binary_search_c,
    compute_entropy,
    compute_output_nmse,
    damp_for_rate,
    watersic_quantize,
    zsic_quantize,
)

# ---------------------------------------------------------------------------
# TestDampForRate
# ---------------------------------------------------------------------------


class TestDampForRate:
    """Tests for :func:`damp_for_rate`."""

    def test_below_knee_returns_base(self):
        """Rates below the knee should return the base value."""
        assert damp_for_rate(3.0) == pytest.approx(1e-4)

    def test_at_knee_returns_base(self):
        """Rate exactly at the knee should return the base value."""
        assert damp_for_rate(5.0) == pytest.approx(1e-4)

    def test_above_knee_decays(self):
        """Rate above the knee should decay: rate=6.0 gives base * 4^{-1} = 2.5e-5."""
        assert damp_for_rate(6.0) == pytest.approx(2.5e-5)

    def test_high_rate_very_small(self):
        """Very high rates should produce a very small damping value."""
        val = damp_for_rate(10.0)
        assert val < 1e-6


# ---------------------------------------------------------------------------
# TestComputeEntropy
# ---------------------------------------------------------------------------


class TestComputeEntropy:
    """Tests for :func:`compute_entropy`."""

    def test_single_value_zero_entropy(self):
        """A constant tensor has zero entropy."""
        Z = torch.full((10, 5), 3, dtype=torch.long)
        assert compute_entropy(Z) == pytest.approx(0.0, abs=1e-7)

    def test_uniform_distribution(self):
        """Four equally-likely values should give log2(4) = 2.0 bits."""
        Z = torch.tensor([0, 1, 2, 3] * 25, dtype=torch.long)
        assert compute_entropy(Z) == pytest.approx(2.0, abs=1e-5)

    def test_binary(self):
        """Half 0s, half 1s should give 1.0 bit."""
        Z = torch.tensor([0] * 50 + [1] * 50, dtype=torch.long)
        assert compute_entropy(Z) == pytest.approx(1.0, abs=1e-5)


# ---------------------------------------------------------------------------
# TestComputeHessianCholesky
# ---------------------------------------------------------------------------


class TestComputeHessianCholesky:
    """Tests for :func:`_compute_hessian_cholesky`."""

    def test_identity_activations(self):
        """With identity-like activations the Hessian should be PSD and L lower-triangular."""
        n = 8
        A = torch.eye(n, dtype=torch.float64)
        Sigma_X, L, perm = _compute_hessian_cholesky(A, sort_cols=False)

        # PSD: eigenvalues >= 0
        eigvals = torch.linalg.eigvalsh(Sigma_X)
        assert (eigvals >= -1e-8).all()

        # L is lower triangular
        assert torch.allclose(L, L.tril())

        # L @ L^T should approximate H = Sigma_X + damp*I
        damp = 1e-4 * Sigma_X.diag().mean()
        H = Sigma_X + damp * torch.eye(n, dtype=torch.float64)
        assert torch.allclose(L @ L.T, H, atol=1e-6)

    def test_with_column_sorting(self):
        """Column sorting should return a valid permutation with ascending diagonal."""
        torch.manual_seed(42)
        A = torch.randn(6, 20, dtype=torch.float64)
        Sigma_X, L, perm = _compute_hessian_cholesky(A, sort_cols=True)

        assert perm is not None
        # perm is a valid permutation of 0..n-1
        assert set(perm.tolist()) == set(range(6))
        # Diagonal of the reordered Hessian should be ascending
        diag_vals = Sigma_X.diag()
        assert (diag_vals[1:] >= diag_vals[:-1] - 1e-8).all()


# ---------------------------------------------------------------------------
# TestComputeOutputNmse
# ---------------------------------------------------------------------------


class TestComputeOutputNmse:
    """Tests for :func:`compute_output_nmse`."""

    def test_zero_error(self):
        """Perfect reconstruction should give NMSE = 0."""
        W = torch.randn(4, 8)
        A = torch.randn(8, 16)
        assert compute_output_nmse(W, W, A) == pytest.approx(0.0, abs=1e-7)

    def test_positive_error(self):
        """Perturbed reconstruction should give positive NMSE."""
        torch.manual_seed(0)
        W = torch.randn(4, 8)
        W_q = W + 0.1 * torch.randn_like(W)
        A = torch.randn(8, 16)
        nmse = compute_output_nmse(W, W_q, A)
        assert nmse > 0.0


# ---------------------------------------------------------------------------
# TestZsicQuantize
# ---------------------------------------------------------------------------


class TestZsicQuantize:
    """Tests for :func:`zsic_quantize`."""

    @pytest.fixture
    def setup(self):
        torch.manual_seed(123)
        a, n, T = 16, 8, 64
        W = torch.randn(a, n, dtype=torch.float64)
        A = torch.randn(n, T, dtype=torch.float64)
        Sigma_X, L, _ = _compute_hessian_cholesky(A, sort_cols=False)
        alpha = 0.5 / L.diag()
        return W, A, alpha, Sigma_X, L

    def test_produces_valid_output(self, setup):
        """Output should have correct shape, positive rate, and NMSE in (0, 1)."""
        W, A, alpha, Sigma_X, L = setup
        W_hat, rate, nmse, Z, gamma = zsic_quantize(W, A, alpha, Sigma_X, L, use_lmmse=False)

        assert W_hat.shape == W.shape
        assert rate > 0.0
        assert 0.0 < nmse < 1.0
        assert Z.shape == W.shape
        assert gamma.shape == (W.shape[1],)

    def test_lmmse_improves_nmse(self, setup):
        """LMMSE correction should reduce (or at least not increase) the NMSE."""
        W, A, alpha, Sigma_X, L = setup
        _, _, nmse_no, _, _ = zsic_quantize(W, A, alpha, Sigma_X, L, use_lmmse=False)
        _, _, nmse_yes, _, _ = zsic_quantize(W, A, alpha, Sigma_X, L, use_lmmse=True)

        assert nmse_yes <= nmse_no + 1e-8

    def test_rescaler_produces_valid_low_nmse(self, setup):
        """Rescaler path with n_rescaler_iters=5 should produce valid output with low NMSE."""
        W, A, alpha, Sigma_X, L = setup
        W_hat, rate, nmse, Z, gamma = zsic_quantize(
            W, A, alpha, Sigma_X, L, use_lmmse=True, n_rescaler_iters=5
        )

        assert W_hat.shape == W.shape
        assert rate > 0.0
        # The rescaler path should still achieve reasonable NMSE (< 0.5).
        assert 0.0 < nmse < 0.5
        # Reconstruction should be meaningfully close to original.
        assert (W - W_hat).norm() < W.norm()


# ---------------------------------------------------------------------------
# TestWatersicQuantize
# ---------------------------------------------------------------------------


class TestWatersicQuantize:
    """Tests for :func:`watersic_quantize`."""

    @pytest.fixture
    def data(self):
        torch.manual_seed(7)
        a, n, T = 32, 12, 100
        W = torch.randn(a, n, dtype=torch.float64)
        A = torch.randn(n, T, dtype=torch.float64)
        return W, A

    def test_basic_quantization(self, data):
        """Should return valid W_hat, rate, and NMSE."""
        W, A = data
        W_hat, rate, nmse, Z, gamma = watersic_quantize(W, A, c=0.5)
        assert W_hat.shape == W.shape
        assert rate > 0.0
        assert nmse > 0.0
        assert Z.shape == W.shape
        assert gamma.shape == (W.shape[1],)

    def test_smaller_c_gives_higher_rate(self, data):
        """Smaller c should produce finer quantization and a higher coding rate."""
        W, A = data
        _, rate_large, _, _, _ = watersic_quantize(W, A, c=2.0)
        _, rate_small, _, _, _ = watersic_quantize(W, A, c=0.1)
        assert rate_small > rate_large

    def test_permutation_roundtrip(self, data):
        """Columns should be correctly un-permuted so W_hat is in the original order."""
        W, A = data
        W_hat, _, _, _, _ = watersic_quantize(W, A, c=0.5)
        # The reconstruction error should be smaller than the weight norm
        # (i.e. it's not just garbage / misaligned columns).
        assert (W - W_hat).norm() < W.norm()


# ---------------------------------------------------------------------------
# TestBinarySearchC
# ---------------------------------------------------------------------------


class TestBinarySearchC:
    """Tests for :func:`binary_search_c`."""

    def test_achieves_target_rate(self):
        """The returned *c* should achieve a rate within 1.0 bit of the target."""
        torch.manual_seed(99)
        a, n, T = 32, 10, 80
        W = torch.randn(a, n, dtype=torch.float64)
        A = torch.randn(n, T, dtype=torch.float64)

        target = 4.0
        # Use sample_frac=1.0 so the search operates on the same rows as the
        # verification call (the matrix is small, so subsampling would cause
        # a large mismatch between search and evaluation rates).
        c = binary_search_c(W, A, target_rate=target, sample_frac=1.0)

        # Evaluate at full size to verify.
        _, rate, _, _, _ = watersic_quantize(W, A, c)
        assert abs(rate - target) < 1.0


# ---------------------------------------------------------------------------
# KV Quantizer Helper tests
# ---------------------------------------------------------------------------

from modelopt.torch.quantization.algorithms.watersic_kv.kv_quantizer import (
    WaterSICKVState,
    _compute_importance_weights,
    kl_divergence_logits,
)

# ---------------------------------------------------------------------------
# TestComputeImportanceWeights
# ---------------------------------------------------------------------------


class TestComputeImportanceWeights:
    """Tests for :func:`_compute_importance_weights`."""

    def test_uniform_attention_gives_uniform_weights(self):
        """Uniform attention matrix should produce equal importance weights."""
        N = 16
        P = torch.ones(8, N) / N  # uniform over tokens
        sqrt_w = _compute_importance_weights(P)

        assert sqrt_w.shape == (N, 1)
        # All weights should be identical (since input is uniform).
        assert torch.allclose(sqrt_w, sqrt_w[0].expand_as(sqrt_w))

    def test_peaked_attention_gives_high_weight(self):
        """When all attention is on token 0, token 0 should have the highest weight."""
        N = 16
        P = torch.zeros(8, N)
        P[:, 0] = 1.0  # all attention on token 0
        sqrt_w = _compute_importance_weights(P)

        assert sqrt_w.shape == (N, 1)
        # Token 0 should have the largest weight.
        assert sqrt_w[0, 0] == sqrt_w.max()

    def test_clipping(self):
        """Clipping should limit the maximum importance weight."""
        N = 16
        P = torch.zeros(8, N)
        P[:, 0] = 1.0  # all attention on token 0
        clip = 10.0
        sqrt_w = _compute_importance_weights(P, importance_clip=clip)

        import math

        assert sqrt_w.max().item() <= math.sqrt(clip) + 1e-6


# ---------------------------------------------------------------------------
# TestKlDivergenceLogits
# ---------------------------------------------------------------------------


class TestKlDivergenceLogits:
    """Tests for :func:`kl_divergence_logits`."""

    def test_identical_keys_zero_kl(self):
        """KL divergence should be zero when K_q == K."""
        torch.manual_seed(42)
        Q = torch.randn(2, 8, 4)
        K = torch.randn(2, 12, 4)
        kl = kl_divergence_logits(Q, K, K)
        assert kl == pytest.approx(0.0, abs=1e-7)

    def test_different_keys_positive_kl(self):
        """KL divergence should be positive when K_q != K."""
        torch.manual_seed(42)
        Q = torch.randn(2, 8, 4)
        K = torch.randn(2, 12, 4)
        K_q = K + 0.5 * torch.randn_like(K)
        kl = kl_divergence_logits(Q, K, K_q)
        assert kl > 0.0


# ---------------------------------------------------------------------------
# TestWaterSICKVState
# ---------------------------------------------------------------------------


class TestWaterSICKVState:
    """Tests for :class:`WaterSICKVState`."""

    def test_state_creation(self):
        """State dataclass should store all fields correctly."""
        Z = torch.randint(0, 10, (4, 32, 16))
        alpha = torch.randn(4, 16)
        gamma = torch.randn(4, 16)
        state = WaterSICKVState(Z=Z, alpha=alpha, gamma=gamma, perm=None, rate=2.5)

        assert state.Z is Z
        assert state.alpha is alpha
        assert state.gamma is gamma
        assert state.perm is None
        assert state.rate == 2.5
