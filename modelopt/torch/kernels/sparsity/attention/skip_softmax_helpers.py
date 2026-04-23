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


"""Skip-softmax / N:M sparse attention helpers.

These ``@triton.jit`` helpers are called conditionally from the baseline
flash-attention forward kernel in ``common/attention/triton_fa.py`` when the
user requests N:M sparsity or sink/window-aware dense regions.
"""

import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# N:M sparse softmax helpers
# ---------------------------------------------------------------------------
@triton.jit
def _sparse_nm_masks_m4(x0, x1, x2, x3, N: tl.constexpr):
    """Top-N of 4 selection via pure boolean logic (6 comparisons, no int casts).

    Uses ``>=`` so that ties are broken by index (lower index wins).
    Guarantees exactly N masks are True for any input including all-equal.

    Boolean formulas for "at least K of 3 wins":
      K=3 (N=1): AND of all   — must beat all 3 others
      K=2 (N=2): majority     — must beat at least 2 (sorting network)
      K=1 (N=3): OR of all    — must beat at least 1
    """
    c01 = x0 >= x1
    c02 = x0 >= x2
    c03 = x0 >= x3
    c12 = x1 >= x2
    c13 = x1 >= x3
    c23 = x2 >= x3

    nc01 = ~c01
    nc02 = ~c02
    nc03 = ~c03
    nc12 = ~c12
    nc13 = ~c13
    nc23 = ~c23

    if N == 1:
        # Keep max only: must beat all 3
        m0 = c01 & c02 & c03
        m1 = nc01 & c12 & c13
        m2 = nc02 & nc12 & c23
        m3 = nc03 & nc13 & nc23
    elif N == 2:
        # Majority vote: must beat at least 2 of 3
        m0 = (c01 & c02) | (c01 & c03) | (c02 & c03)
        m1 = (nc01 & c12) | (nc01 & c13) | (c12 & c13)
        m2 = (nc02 & nc12) | (nc02 & c23) | (nc12 & c23)
        m3 = (nc03 & nc13) | (nc03 & nc23) | (nc13 & nc23)
    elif N == 3:
        # Keep all but min: must beat at least 1
        m0 = c01 | c02 | c03
        m1 = nc01 | c12 | c13
        m2 = nc02 | nc12 | c23
        m3 = nc03 | nc13 | nc23
    else:
        tl.static_assert(False, "N must be 1, 2, or 3 for M=4")

    return m0, m1, m2, m3


@triton.jit
def _apply_sparse_nm_to_qk_tile(
    qk,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SPARSITY_N: tl.constexpr,
    SPARSITY_M: tl.constexpr,
):
    """Apply N:M sparse softmax to a QK score tile.

    For every ``SPARSITY_M`` consecutive elements along the N (key) dimension,
    keeps the top ``SPARSITY_N`` values and sets the rest to ``-inf``.
    ``BLOCK_N`` must be divisible by ``SPARSITY_M``.

    For M=4, exactly N values are retained (ties broken by position).
    For M=8, a threshold-based approach (``tl.sort``) may retain more
    than N values when ties straddle the threshold boundary.
    """
    tl.static_assert(SPARSITY_M == 4 or SPARSITY_M == 8, "SPARSITY_M must be 4 or 8")  # noqa: PLR1714
    MASK_VAL: tl.constexpr = float("-inf")

    if SPARSITY_M == 4:
        tl.static_assert(BLOCK_N % 4 == 0, "BLOCK_N must be divisible by 4")
        reshaped = tl.reshape(qk, (BLOCK_M, BLOCK_N // 4, 4))
        cols = tl.arange(0, 4)[None, None, :]
        x0 = tl.sum(tl.where(cols == 0, reshaped, 0.0), axis=2)
        x1 = tl.sum(tl.where(cols == 1, reshaped, 0.0), axis=2)
        x2 = tl.sum(tl.where(cols == 2, reshaped, 0.0), axis=2)
        x3 = tl.sum(tl.where(cols == 3, reshaped, 0.0), axis=2)

        m0, m1, m2, m3 = _sparse_nm_masks_m4(x0, x1, x2, x3, SPARSITY_N)

        out = tl.full((BLOCK_M, BLOCK_N // 4, 4), 0.0, dtype=qk.dtype)
        out = tl.where(cols == 0, tl.expand_dims(tl.where(m0, x0, MASK_VAL), 2), out)
        out = tl.where(cols == 1, tl.expand_dims(tl.where(m1, x1, MASK_VAL), 2), out)
        out = tl.where(cols == 2, tl.expand_dims(tl.where(m2, x2, MASK_VAL), 2), out)
        out = tl.where(cols == 3, tl.expand_dims(tl.where(m3, x3, MASK_VAL), 2), out)
        return tl.reshape(out, (BLOCK_M, BLOCK_N))

    else:  # SPARSITY_M == 8
        tl.static_assert(BLOCK_N % 8 == 0, "BLOCK_N must be divisible by 8")
        reshaped = tl.reshape(qk, (BLOCK_M, BLOCK_N // 8, 8))

        # Sort each group of 8 ascending; N-th largest is at index (8 - N)
        sorted_vals = tl.sort(reshaped, dim=2)
        KTH_IDX: tl.constexpr = SPARSITY_M - SPARSITY_N  # index of N-th largest in ascending order

        # Extract the threshold value at KTH_IDX via masked sum
        # Use 0.0 as fill (not -inf) so sum equals just the KTH element
        cols = tl.arange(0, 8)[None, None, :]
        threshold = tl.sum(tl.where(cols == KTH_IDX, sorted_vals, 0.0), axis=2)

        # Mask: keep elements >= threshold (may keep >N on ties — acceptable)
        mask = reshaped >= tl.expand_dims(threshold, 2)
        return tl.reshape(tl.where(mask, reshaped, MASK_VAL), (BLOCK_M, BLOCK_N))


# ---------------------------------------------------------------------------
# BLASST skip-softmax per-tile decision
# ---------------------------------------------------------------------------
@triton.jit
def _skip_softmax_decision(
    scores,
    row_max,
    SKIP_THRESHOLD_LOG2: tl.constexpr,
    Sparsity_total,
    Sparsity_skipped,
    MEASURE_SPARSITY: tl.constexpr,
):
    """BLASST skip-softmax per-tile decision (https://arxiv.org/pdf/2512.12087).

    During FlashAttention's block-wise computation we maintain a running
    maximum ``m_i^(j)`` across blocks. If a block's local maximum
    ``~m_i^(j)`` is significantly smaller than the running maximum
    (``~m_i^(j) - m_i^(j) < ln(lambda)``), then ``exp(~m_i^(j) - m_i^(j))
    < lambda ~= 0`` and the block's contribution to the output is negligible.
    The caller may then skip the softmax computation, V load, and BMM2.

    The threshold is pre-scaled to log2 space by the Python wrapper so it can
    be compared directly against the already-scaled scores.

    Returns:
        True when *all* Q rows in the tile satisfy the skip criterion.

    When ``MEASURE_SPARSITY`` is set, also records total/skipped tile counts
    via atomic adds on ``Sparsity_total`` / ``Sparsity_skipped``.
    """
    tile_row_max = tl.max(scores, 1)  # [BLOCK_M] — ~m_i^(j) (scaled)
    # Per-row: True if row's tile max is negligible vs running max
    can_skip = tile_row_max < (row_max + SKIP_THRESHOLD_LOG2)
    # Per-tile: skip entire tile only if ALL rows are negligible
    skip_tile = tl.min(can_skip.to(tl.int32)) == 1

    if MEASURE_SPARSITY:
        tl.atomic_add(Sparsity_total, 1)  # count every tile
        if skip_tile:
            tl.atomic_add(Sparsity_skipped, 1)  # count skipped tiles

    return skip_tile


# ---------------------------------------------------------------------------
# Sink/window dense-region check
# ---------------------------------------------------------------------------
@triton.jit
def _is_dense_region(
    kv_start,
    tile_q,
    seq_len_q,
    seq_len_kv,
    BLOCK_M: tl.constexpr,
    NUM_SINK_TOKENS: tl.constexpr,
    DENSE_WINDOW_SIZE: tl.constexpr,
):
    """Check if a KV tile falls in a dense region (sink tokens or local window).

    Uses absolute token positions so the result is BLOCK_N-independent,
    ensuring forward and backward (which may use different BLOCK_N) agree.

    Returns:
        True if the tile should be kept dense (skip N:M sparsification).
    """
    is_sink = kv_start < NUM_SINK_TOKENS
    causal_offset = seq_len_kv - seq_len_q
    q_abs_pos = tile_q * BLOCK_M + causal_offset
    token_distance = q_abs_pos - kv_start
    is_local = (token_distance >= 0) and (token_distance < DENSE_WINDOW_SIZE)
    return is_sink or is_local
