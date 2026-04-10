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

"""Core ZSIC (Zero-Shot Integer Compression) algorithm for WaterSIC KV-cache quantization.

This is a pure math module with no Model-Optimizer dependencies.  It implements
the sequential integer coding algorithm described in the WaterSIC paper, ported
from psx-vfp commit 39073b1.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def damp_for_rate(target_rate: float, base: float = 1e-4, knee: float = 5.0) -> float:
    """Return a damping coefficient that decays for rates above *knee*.

    ``base * 4 ** (-max(0, target_rate - knee))``
    """
    return base * 4.0 ** (-max(0.0, target_rate - knee))


def compute_entropy(Z: Tensor) -> float:
    """Compute Shannon entropy (in bits) of integer-valued tensor *Z*."""
    # Flatten and count occurrences of each unique integer value.
    flat = Z.flatten().long()
    counts = torch.bincount(flat - flat.min())
    counts = counts[counts > 0]
    probs = counts.float() / counts.sum().float()
    return -(probs * probs.log2()).sum().item()


def compute_output_nmse(W: Tensor, W_q: Tensor, A: Tensor) -> float:
    """Normalised MSE measured in the output space: ``||err @ A||^2 / ||W @ A||^2``."""
    err = (W - W_q) @ A
    ref = W @ A
    return (err.norm() ** 2 / ref.norm() ** 2).item()


# ---------------------------------------------------------------------------
# Hessian / Cholesky
# ---------------------------------------------------------------------------


def _compute_hessian_cholesky(
    A: Tensor,
    damp_pct: float = 1e-4,
    sort_cols: bool = True,
) -> tuple[Tensor, Tensor, Tensor | None]:
    """Build the Hessian ``A A^T`` and return its damped Cholesky factor.

    Parameters
    ----------
    A : Tensor
        Activation matrix of shape ``(n, T)`` where *n* is the number of
        weight columns and *T* is the number of calibration tokens.
    damp_pct : float
        Fraction of the mean diagonal used as Tikhonov damping.
    sort_cols : bool
        If *True*, reorder columns by ascending diagonal of ``A A^T``
        (improves numerical stability of the sequential coding).

    Returns:
    -------
    Sigma_X : Tensor  – ``A A^T`` (possibly column-reordered), shape ``(n, n)``
    L : Tensor         – lower-triangular Cholesky factor of the damped
                         Hessian, shape ``(n, n)``
    perm : Tensor | None – permutation used (LongTensor of length *n*), or
                           *None* when ``sort_cols`` is *False*.
    """
    perm: Tensor | None = None

    if sort_cols:
        # Sort by ascending diagonal of A A^T  ≡  ascending row-norms of A.
        diag = (A * A).sum(dim=1)
        perm = torch.argsort(diag)
        A = A[perm]

    Sigma_X = A @ A.T

    damp = damp_pct * Sigma_X.diag().mean()
    H = Sigma_X + damp * torch.eye(Sigma_X.shape[0], device=A.device, dtype=A.dtype)

    try:
        L = torch.linalg.cholesky(H)
    except torch.linalg.LinAlgError:
        # Retry with 10x more damping if the first attempt was singular.
        H += 10 * damp * torch.eye(H.shape[0], device=A.device, dtype=A.dtype)
        L = torch.linalg.cholesky(H)

    return Sigma_X, L, perm


# ---------------------------------------------------------------------------
# Rescaler optimisation
# ---------------------------------------------------------------------------


def _optimize_rescalers(
    W_hat_0: Tensor,
    W: Tensor,
    Sigma_X: Tensor,
    gamma_init: Tensor,
    n_iters: int = 10,
) -> Tensor:
    """Alternating row / column rescaler optimisation.

    Starting from ``gamma_init`` (per-column), iterate:
    1. Fix gamma, solve for row rescalers *t*.
    2. Fix *t*, solve for column rescalers *gamma*.

    Returns the rescaled reconstruction ``diag(t) @ W_hat_0 @ diag(gamma)``.
    """
    gamma = gamma_init.clone()
    t = torch.ones(W.shape[0], device=W.device, dtype=W.dtype)

    for _ in range(n_iters):
        # --- Row rescalers (t) --- given gamma, minimise over t_i independently.
        # For each row i:  t_i = (W_hat_0[i] * gamma) . Sigma_X . W[i]
        #                       / (W_hat_0[i] * gamma) . Sigma_X . (W_hat_0[i] * gamma)
        scaled = W_hat_0 * gamma.unsqueeze(0)  # (a, n)
        num_t = (scaled @ Sigma_X * W).sum(dim=1)  # (a,)
        den_t = (scaled @ Sigma_X * scaled).sum(dim=1)  # (a,)
        t = num_t / den_t.clamp(min=1e-20)

        # --- Column rescalers (gamma) --- given t, minimise over gamma_j independently.
        # gamma_j = sum_i t_i * W_hat_0[i,j] * (Sigma_X[j,:] @ W[i,:].T)
        #         / sum_i (t_i * W_hat_0[i,j])^2 * Sigma_X[j,j]
        t_col = t.unsqueeze(1)  # (a, 1)
        tw = t_col * W_hat_0  # (a, n)
        # numerator: for each j, sum_i  tw[i,j] * (Sigma_X[j,:] @ W[i,:])
        num_g = (tw.T @ W @ Sigma_X.T).diag()  # (n,)  -- Sigma_X symmetric so .T ok
        den_g = (tw * tw).T @ torch.ones(W.shape[0], device=W.device, dtype=W.dtype)
        den_g = den_g * Sigma_X.diag()  # (n,)
        gamma = num_g / den_g.clamp(min=1e-20)

    return t.unsqueeze(1) * W_hat_0 * gamma.unsqueeze(0)


# ---------------------------------------------------------------------------
# Core sequential coding
# ---------------------------------------------------------------------------


def zsic_quantize(
    W: Tensor,
    A: Tensor,
    alpha: Tensor,
    Sigma_X: Tensor,
    L: Tensor,
    use_lmmse: bool = True,
    n_rescaler_iters: int = 0,
) -> tuple[Tensor, float, float, Tensor, Tensor]:
    """Run the ZSIC sequential integer coding loop.

    Parameters
    ----------
    W : Tensor (a, n)  – weight matrix (rows = output channels).
    A : Tensor (n, T)  – activation matrix.
    alpha : Tensor (n,) – per-column step sizes.
    Sigma_X : Tensor (n, n)  – ``A A^T``.
    L : Tensor (n, n)  – lower-triangular Cholesky factor.
    use_lmmse : bool
        Apply per-column LMMSE gain correction.
    n_rescaler_iters : int
        Number of alternating rescaler iterations (0 = disable).

    Returns:
    -------
    W_hat : Tensor (a, n)  – quantised reconstruction.
    rate : float           – estimated coding rate (bits per weight element).
    nmse : float           – output NMSE.
    Z : Tensor (a, n)     – integer codes.
    gamma : Tensor (n,)   – per-column LMMSE shrinkage gains.
    """
    a, n = W.shape

    # M_T = L^{-1} Sigma_X  (solve L M_T = Sigma_X for M_T)
    M_T = torch.linalg.solve_triangular(L, Sigma_X, upper=False)
    Y = W @ M_T.T  # (a, n)

    Z = torch.zeros(a, n, device=W.device, dtype=torch.long)
    gamma = torch.ones(n, device=W.device, dtype=W.dtype)

    for i in range(n - 1, -1, -1):
        d_i = alpha[i] * L[i, i]
        z_i = torch.round(Y[:, i] / d_i).long()
        Z[:, i] = z_i

        z_f = z_i.float().to(W.dtype)
        z_sq = z_f.dot(z_f)

        if use_lmmse and z_sq > 1e-20:
            gamma[i] = z_f.dot(Y[:, i]) / (d_i * z_sq)

        # Efficient rank-1 update: Y -= (gamma[i]*alpha[i]) * z_f outer L[i,:]
        Y.addr_(z_f, L[i], alpha=-(gamma[i] * alpha[i]).item())

    # --- Entropy and rate ---
    entropy = compute_entropy(Z)
    rate = entropy + 16.0 / a + 16.0 / n

    # --- Reconstruction ---
    W_hat_0 = Z.float().to(W.dtype) * alpha.unsqueeze(0)

    if n_rescaler_iters > 0:
        W_hat = _optimize_rescalers(W_hat_0, W, Sigma_X, gamma, n_iters=n_rescaler_iters)
    elif use_lmmse:
        W_hat = W_hat_0 * gamma.unsqueeze(0)
    else:
        W_hat = W_hat_0

    nmse = compute_output_nmse(W, W_hat, A)
    return W_hat, rate, nmse, Z, gamma


# ---------------------------------------------------------------------------
# WaterSIC interface
# ---------------------------------------------------------------------------


def watersic_quantize(
    W: Tensor,
    A: Tensor,
    c: float,
    damp_pct: float = 1e-4,
    use_lmmse: bool = True,
    n_rescaler_iters: int = 0,
    _precomputed: tuple[Tensor, Tensor, Tensor | None] | None = None,
) -> tuple[Tensor, float, float, Tensor, Tensor]:
    """Quantise *W* using the WaterSIC algorithm for a given scale factor *c*.

    Parameters
    ----------
    W : Tensor (a, n)
    A : Tensor (n, T)
    c : float – global scale factor that controls the rate/distortion trade-off.
    damp_pct : float
    use_lmmse : bool
    n_rescaler_iters : int
    _precomputed : tuple, optional
        ``(Sigma_X, L, perm)`` from a prior call to
        :func:`_compute_hessian_cholesky` to avoid redundant computation.

    Returns:
    -------
    W_hat : Tensor (a, n)  – quantised reconstruction (in original column order).
    rate : float
    nmse : float
    Z : Tensor (a, n)     – integer codes (in original column order).
    gamma : Tensor (n,)   – per-column LMMSE shrinkage gains (in original column order).
    """
    if _precomputed is not None:
        Sigma_X, L, perm = _precomputed
    else:
        Sigma_X, L, perm = _compute_hessian_cholesky(A, damp_pct=damp_pct)

    # Apply permutation to weight columns if used.
    if perm is not None:
        W = W[:, perm]
        A = A[perm]

    alpha = c / L.diag()

    W_hat, rate, nmse, Z, gamma = zsic_quantize(
        W, A, alpha, Sigma_X, L, use_lmmse=use_lmmse, n_rescaler_iters=n_rescaler_iters
    )

    # Undo permutation.
    if perm is not None:
        inv_perm = torch.argsort(perm)
        W_hat = W_hat[:, inv_perm]
        Z = Z[:, inv_perm]
        gamma = gamma[inv_perm]

    return W_hat, rate, nmse, Z, gamma


# ---------------------------------------------------------------------------
# Binary search for c
# ---------------------------------------------------------------------------


def binary_search_c(
    W: Tensor,
    A: Tensor,
    target_rate: float,
    damp_pct: float | None = None,
    use_lmmse: bool = True,
    n_rescaler_iters: int = 0,
    n_iters: int = 30,
    sample_frac: float = 0.1,
    _precomputed: tuple[Tensor, Tensor, Tensor | None] | None = None,
) -> float:
    """Find the scale factor *c* that achieves *target_rate* via log-space binary search.

    Parameters
    ----------
    W : Tensor (a, n)
    A : Tensor (n, T)
    target_rate : float – desired bits per weight element.
    damp_pct : float | None
        If *None*, determined automatically via :func:`damp_for_rate`.
    use_lmmse : bool
    n_rescaler_iters : int
    n_iters : int – number of binary-search iterations (default 30).
    sample_frac : float – fraction of rows to use (default 10%).
    _precomputed : tuple, optional

    Returns:
    -------
    c : float – optimal scale factor.
    """
    if damp_pct is None:
        damp_pct = damp_for_rate(target_rate)

    a = W.shape[0]
    n_sample = max(4, int(a * sample_frac))

    # Subsample rows for speed.
    if n_sample < a:
        idx = torch.randperm(a, device=W.device)[:n_sample]
        W_sub = W[idx]
    else:
        W_sub = W

    # Precompute Hessian / Cholesky (shared across iterations).
    if _precomputed is not None:
        precomputed = _precomputed
    else:
        precomputed = _compute_hessian_cholesky(A, damp_pct=damp_pct)

    log_c_lo = math.log(1e-6)
    log_c_hi = math.log(1e3)

    for _ in range(n_iters):
        log_c_mid = 0.5 * (log_c_lo + log_c_hi)
        c_mid = math.exp(log_c_mid)

        _, rate, _, _, _ = watersic_quantize(
            W_sub,
            A,
            c_mid,
            damp_pct=damp_pct,
            use_lmmse=use_lmmse,
            n_rescaler_iters=n_rescaler_iters,
            _precomputed=precomputed,
        )

        if rate > target_rate:
            # c is too small (finer quantization = higher rate), increase c.
            log_c_lo = log_c_mid
        else:
            log_c_hi = log_c_mid

    return math.exp(0.5 * (log_c_lo + log_c_hi))
