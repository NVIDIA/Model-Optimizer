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

"""WaterSIC KV-cache quantizer helper.

Wraps the core ZSIC math with attention-module hooking for real model
calibration.  The :class:`WaterSICKVHelper` patches
``_QuantAttention._quantized_attention`` to capture query / key activations,
then runs :func:`watersic_quantize` per head.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from modelopt.torch.utils import print_rank_0

from .zsic import (
    _compute_hessian_cholesky,
    binary_search_c,
    damp_for_rate,
    watersic_quantize,
)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class WaterSICKVState:
    """Per-layer quantisation state produced by :meth:`WaterSICKVHelper.quantize`."""

    Z: Tensor
    """Integer code-book indices."""
    alpha: Tensor
    """Per-column step sizes."""
    gamma: Tensor
    """Per-column LMMSE gains."""
    perm: Tensor | None
    """Column permutation (or *None*)."""
    rate: float
    """Achieved coding rate (bits per element)."""


# ---------------------------------------------------------------------------
# Importance weighting
# ---------------------------------------------------------------------------


def _compute_importance_weights(P: Tensor, importance_clip: float = 50.0) -> Tensor:
    """Derive per-token importance weights from an attention probability matrix.

    Parameters
    ----------
    P : Tensor (B, N)
        Attention probabilities summed (or averaged) over queries – i.e.
        ``P[b, n]`` is how much attention is paid to token *n* in sample *b*.
        Typically ``P = softmax(Q K^T / sqrt(d)).sum(dim=-2)``.
    importance_clip : float
        Clamp the normalised weights to ``[1/clip, clip]`` to prevent
        extreme outliers.

    Returns
    -------
    sqrt_w : Tensor (N, 1)
        Square-root importance weights, suitable for left-multiplying the
        activation matrix so that high-attention tokens contribute more to
        the Hessian.
    """
    # Sum across the batch (rows) to get a per-token importance score.
    w = P.sum(dim=0)  # (N,)

    # Normalise so that the mean weight is 1.
    w = w / w.mean().clamp(min=1e-30)

    # Clip to avoid extreme values.
    w = w.clamp(min=1.0 / importance_clip, max=importance_clip)

    return w.sqrt().unsqueeze(1)  # (N, 1)


# ---------------------------------------------------------------------------
# KL divergence in logit space
# ---------------------------------------------------------------------------


def kl_divergence_logits(
    Q: Tensor,
    K: Tensor,
    K_q: Tensor,
    temperature: float = 1.0,
) -> float:
    """Compute the KL divergence between attention distributions induced by *K* and *K_q*.

    Uses the logit identity to avoid materialising the full attention matrix:

        KL(P || P_q) = E_x[ P^T (s - s_q) + logsumexp(s_q) - logsumexp(s) ]

    where ``s = Q K^T / temperature`` and ``s_q = Q K_q^T / temperature``.

    Parameters
    ----------
    Q : Tensor (..., S, D)
    K : Tensor (..., N, D)
    K_q : Tensor (..., N, D)
    temperature : float

    Returns
    -------
    kl : float
        Mean KL divergence in **bits** (i.e. divided by ln 2).
    """
    Q64 = Q.double()
    K64 = K.double()
    Kq64 = K_q.double()

    s = Q64 @ K64.transpose(-2, -1) / temperature   # (..., S, N)
    s_q = Q64 @ Kq64.transpose(-2, -1) / temperature  # (..., S, N)

    log_Z = torch.logsumexp(s, dim=-1)      # (..., S)
    log_Z_q = torch.logsumexp(s_q, dim=-1)  # (..., S)

    P = torch.softmax(s, dim=-1)  # (..., S, N)

    # KL per query position: sum_n P_n (s_n - s_q_n) + log_Z_q - log_Z
    kl_per_query = (P * (s - s_q)).sum(dim=-1) + log_Z_q - log_Z  # (..., S)

    # Convert nats to bits and return mean.
    import math

    return (kl_per_query.mean() / math.log(2)).item()


# ---------------------------------------------------------------------------
# WaterSICKVHelper
# ---------------------------------------------------------------------------


class WaterSICKVHelper:
    """Hook-based helper that captures Q/K activations and runs WaterSIC quantisation.

    Usage::

        helper = WaterSICKVHelper(quant_attn_module, "layer.3")
        helper.setup()
        # ... run calibration forward passes ...
        state = helper.quantize(target_rate=4.0)
        helper.cleanup()
        helper.free()
    """

    def __init__(
        self,
        module,
        name: str,
        kl_aware: bool = False,
        importance_clip: float = 50.0,
    ):
        self.module = module
        self.name = name
        self.kl_aware = kl_aware
        self.importance_clip = importance_clip

        self.collected_Q: list[Tensor] = []
        self.collected_K: list[Tensor] = []

        self._original_fn = None

    # ----- patching --------------------------------------------------

    def setup(self):
        """Patch ``_quantized_attention`` on the module instance to capture Q/K."""
        # The original is a @staticmethod on the class – grab the underlying function.
        original_fn = type(self.module)._quantized_attention
        self._original_fn = original_fn

        helper = self  # closure reference

        def patched_fn(
            original_attention_interface,
            self_attn,
            query_states,
            key_states,
            value_states,
            *args,
            **kwargs,
        ):
            # Capture detached CPU copies before quantizers touch them.
            helper.collected_Q.append(query_states.detach().cpu())
            helper.collected_K.append(key_states.detach().cpu())

            # Call the original static method (not bound, pass all args).
            return original_fn(
                original_attention_interface,
                self_attn,
                query_states,
                key_states,
                value_states,
                *args,
                **kwargs,
            )

        # Patch on the *instance* so it shadows the class-level staticmethod.
        self.module._quantized_attention = patched_fn

    def cleanup(self):
        """Remove the instance-level override, restoring the class staticmethod."""
        if "_quantized_attention" in vars(self.module):
            delattr(self.module, "_quantized_attention")

    # ----- quantisation -----------------------------------------------

    def quantize(
        self,
        target_rate: float = 4.0,
        use_lmmse: bool = True,
        n_rescaler_iters: int = 0,
        sample_frac: float = 0.1,
    ) -> WaterSICKVState:
        """Run WaterSIC quantisation on the collected key activations.

        Parameters
        ----------
        target_rate : float
            Target coding rate in bits per element.
        use_lmmse : bool
            Whether to apply LMMSE gain correction.
        n_rescaler_iters : int
            Number of alternating rescaler iterations (0 = disable).
        sample_frac : float
            Fraction of rows used by :func:`binary_search_c`.

        Returns
        -------
        WaterSICKVState
        """
        # Concatenate collected activations across calibration batches.
        # Each tensor is (batch, n_heads, seq, d_head).
        Q_all = torch.cat(self.collected_Q, dim=0)  # (B_total, H, S_q, D)
        K_all = torch.cat(self.collected_K, dim=0)  # (B_total, H, S_k, D)

        B, H, S_k, D = K_all.shape

        # We'll store per-head results.
        Z_heads = []
        alpha_heads = []
        gamma_heads = []
        perm_heads = []
        rates = []

        damp_pct = damp_for_rate(target_rate)

        for h in range(H):
            # K_h shape: (B, S_k, D) → treat as weight matrix (a, n) where
            # a = B * S_k (token-batch dimension) and n = D (head dimension).
            K_h = K_all[:, h, :, :].reshape(-1, D).double()  # (B*S_k, D)

            # Activation matrix: use Q_h^T so the Hessian reflects query-key
            # interaction.  A shape: (D, B*S_q).
            Q_h = Q_all[:, h, :, :].reshape(-1, D).double()  # (B*S_q, D)
            A = Q_h.T  # (D, B*S_q)

            # Optional importance weighting — scale K rows (not A) so that
            # high-attention tokens contribute more to the quantisation objective.
            sqrt_w = None
            if self.kl_aware:
                # Compute attention probs: P = softmax(Q_h @ K_h^T / sqrt(D))
                scores = Q_h @ K_h.T / (D**0.5)
                P = torch.softmax(scores.double(), dim=-1).float()
                sqrt_w = _compute_importance_weights(P, self.importance_clip)
                K_h = K_h * sqrt_w  # Scale K rows by importance

            # Precompute Hessian / Cholesky.
            precomputed = _compute_hessian_cholesky(A, damp_pct=damp_pct)
            _, L, perm = precomputed

            # Binary search for the scale factor c.
            c = binary_search_c(
                K_h,
                A,
                target_rate=target_rate,
                damp_pct=damp_pct,
                use_lmmse=use_lmmse,
                n_rescaler_iters=n_rescaler_iters,
                sample_frac=sample_frac,
                _precomputed=precomputed,
            )

            # Full quantisation.
            W_hat, rate, nmse, Z_h, gamma_h = watersic_quantize(
                K_h,
                A,
                c,
                damp_pct=damp_pct,
                use_lmmse=use_lmmse,
                n_rescaler_iters=n_rescaler_iters,
                _precomputed=precomputed,
            )

            # Undo importance scaling after quantisation.
            if sqrt_w is not None:
                W_hat = W_hat / sqrt_w

            print_rank_0(
                f"  [{self.name}] head {h}: rate={rate:.2f} bpe, nmse={nmse:.4f}"
            )

            # Recover per-head state.
            # alpha = c / L.diag() (same as inside watersic_quantize).
            alpha_h = (c / L.diag()).float()

            Z_heads.append(Z_h)
            alpha_heads.append(alpha_h)
            gamma_heads.append(gamma_h.float())
            perm_heads.append(perm)
            rates.append(rate)

        mean_rate = sum(rates) / len(rates) if rates else 0.0

        state = WaterSICKVState(
            Z=torch.stack(Z_heads),           # (H, B*S_k, D)
            alpha=torch.stack(alpha_heads),    # (H, D)
            gamma=torch.stack(gamma_heads),    # (H, D)
            perm=torch.stack(perm_heads) if perm_heads and perm_heads[0] is not None else None,
            rate=mean_rate,
        )

        # Attach state to the module for downstream consumers.
        self.module._watersic_kv_state = state

        return state

    # ----- cleanup -----------------------------------------------------

    def free(self):
        """Release collected calibration data."""
        self.collected_Q.clear()
        self.collected_K.clear()
