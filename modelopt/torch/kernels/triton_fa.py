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

# ruff: noqa: N803, N806 — Triton kernels use uppercase for constexpr and tensor args by convention

"""Triton flash attention kernel with variable-length sequences and GQA.

Based on the Flash Attention v2 algorithm (https://arxiv.org/abs/2307.08691).

Input format: flat packed [total_tokens, num_heads, head_dim] with per-sequence
metadata (b_start_loc, b_seq_len). Supports causal masking and autograd.
"""

import torch
import triton
import triton.language as tl

LOG2E: float = 1.44269504088896

# ---------------------------------------------------------------------------
# Autotune configs for forward kernel
# ---------------------------------------------------------------------------
_FWD_CONFIGS = [
    triton.Config({"BLOCK_M": bm, "BLOCK_N": bn}, num_stages=s, num_warps=w)
    for bm in [64, 128]
    for bn in [32, 64, 128]
    for s in [1, 2, 3]
    for w in [4, 8]
]

# Use a single config in testing for reproducibility
if "PYTEST_VERSION" in __import__("os").environ:
    _FWD_CONFIGS = [triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_stages=1, num_warps=4)]


# ---------------------------------------------------------------------------
# Masking helper
# ---------------------------------------------------------------------------
@triton.jit
def _apply_mask(
    scores,
    q_pos,
    kv_pos,
    seq_len_q,
    seq_len_kv,
    kv_start,
    IS_CAUSAL: tl.constexpr,
):
    """Apply causal mask and padding mask to a score tile."""
    if IS_CAUSAL:
        scores += tl.where(
            (kv_start + kv_pos[None, :] < seq_len_kv)
            & (q_pos[:, None] >= (kv_start + kv_pos[None, :])),
            0,
            float("-inf"),
        )
    else:
        scores += tl.where((kv_start + kv_pos[None, :]) < seq_len_kv, 0, float("-inf"))
    return scores


# ---------------------------------------------------------------------------
# V-mean precomputation kernel (for V2.5 fresh v_mean)
# ---------------------------------------------------------------------------
@triton.jit
def _precompute_vmean(
    V,  # [total_kv, num_kv_heads, head_dim]
    Vmean,  # [batch, num_kv_heads, num_kv_tiles, BLOCK_D] output
    b_start_loc_k,  # [batch] start offset of each KV sequence
    b_seq_len_k,  # [batch] length of each KV sequence
    stride_vbs,
    stride_vh,
    stride_vm_b,
    stride_vm_h,
    stride_vm_t,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    """Precompute mean of V vectors per KV tile.

    Grid: (batch, num_kv_heads, num_kv_tiles)
    Each thread block computes mean(V[tile]) → [BLOCK_D].
    """
    batch_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)
    tile_kv = tl.program_id(2)

    seq_len_kv = tl.load(b_seq_len_k + batch_idx)
    kv_offset = tl.load(b_start_loc_k + batch_idx)

    kv_start = tile_kv * BLOCK_N
    if kv_start >= seq_len_kv:
        return

    kv_pos = tl.arange(0, BLOCK_N)
    dim_pos = tl.arange(0, BLOCK_D)
    d_mask = dim_pos < HEAD_DIM
    kv_mask = (kv_start + kv_pos) < seq_len_kv

    # Load V tile [BLOCK_N, BLOCK_D]
    v_ptrs = (
        (kv_offset + kv_start + kv_pos[:, None]) * stride_vbs
        + kv_head_idx * stride_vh
        + dim_pos[None, :]
    )
    v = tl.load(V + v_ptrs, mask=kv_mask[:, None] & d_mask[None, :], other=0.0)

    # Count valid tokens for correct mean
    n_valid = tl.sum(kv_mask.to(tl.float32))
    n_valid = tl.maximum(n_valid, 1.0)  # avoid div by zero

    # mean(V, dim=0) → [BLOCK_D]
    v_mean = tl.sum(v.to(tl.float32), 0) / n_valid

    # Store
    out_ptr = (
        Vmean
        + batch_idx * stride_vm_b
        + kv_head_idx * stride_vm_h
        + tile_kv * stride_vm_t
        + dim_pos
    )
    tl.store(out_ptr, v_mean, mask=d_mask)


# ---------------------------------------------------------------------------
# Forward kernel body (shared by autotuned and fixed-config paths)
# ---------------------------------------------------------------------------
@triton.jit
def _attn_fwd_body(
    Q,  # [total_q, num_q_heads, head_dim] query tensor
    K,  # [total_kv, num_kv_heads, head_dim] key tensor
    V,  # [total_kv, num_kv_heads, head_dim] value tensor
    qk_scale,  # softmax_scale * log2(e)
    b_start_loc,  # [batch] start offset of each Q sequence
    b_seq_len,  # [batch] length of each Q sequence
    b_start_loc_k,  # [batch] start offset of each KV sequence
    b_seq_len_k,  # [batch] length of each KV sequence
    Out,  # [total_q, num_q_heads, head_dim] output tensor
    Lse,  # [total_q, num_q_heads] log-sum-exp
    stride_qbs,
    stride_qh,  # Q strides: per-token, per-head
    stride_kbs,
    stride_kh,  # K strides: per-token, per-head
    stride_vbs,
    stride_vh,  # V strides: per-token, per-head
    stride_obs,
    stride_oh,  # Output strides: per-token, per-head
    stride_lse_tok,
    stride_lse_head,  # LSE strides: per-token, per-head
    N_CTX,  # Max Q sequence length (autotune cache key only)
    kv_group_num: tl.constexpr,  # GQA ratio: num_q_heads // num_kv_heads
    BLOCK_M: tl.constexpr,  # Q tile size
    BLOCK_D: tl.constexpr,  # Head dim tile size (next_power_of_2(HEAD_DIM))
    BLOCK_N: tl.constexpr,  # KV tile size
    IS_CAUSAL: tl.constexpr,  # Whether to apply causal mask
    HEAD_DIM: tl.constexpr,  # Actual head dimension (for d_mask)
    STORE_LSE: tl.constexpr,  # Whether to save LSE for backward pass
    APPLY_SKIP_SOFTMAX: tl.constexpr = False,  # Skip KV tiles with negligible scores
    skip_threshold_log2=0.0,  # Effective log2-space threshold
    # --- V2.5 parameters (optional) ---
    APPLY_SKIP_V25: tl.constexpr = False,  # Pool-K approximate weight + fresh v_mean
    Vmean_cache=None,  # [batch, num_kv_heads, num_kv_tiles, BLOCK_D] float32 (precomputed)
    stride_vm_b=0,  # Vmean_cache strides
    stride_vm_h=0,
    stride_vm_t=0,
):
    # --- Grid: (batch, num_q_heads, num_q_tiles) ---
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    tile_q = tl.program_id(2)
    kv_head_idx = head_idx // kv_group_num

    # --- Load Q and KV varlen metadata ---
    seq_len_q = tl.load(b_seq_len + batch_idx)
    seq_len_kv = tl.load(b_seq_len_k + batch_idx)
    q_offset = tl.load(b_start_loc + batch_idx)
    kv_offset = tl.load(b_start_loc_k + batch_idx)

    if tile_q * BLOCK_M >= seq_len_q:
        return

    # --- Tile position indices ---
    q_pos = tile_q * BLOCK_M + tl.arange(0, BLOCK_M)
    kv_pos = tl.arange(0, BLOCK_N)
    dim_pos = tl.arange(0, BLOCK_D)
    d_mask = dim_pos < HEAD_DIM

    # --- Load Q tile [BLOCK_M, BLOCK_D]: stays in SRAM for the entire KV loop ---
    q_ptrs = (q_offset + q_pos[:, None]) * stride_qbs + head_idx * stride_qh + dim_pos[None, :]
    q = tl.load(Q + q_ptrs, mask=(q_pos[:, None] < seq_len_q) & d_mask[None, :], other=0.0)

    k_base = K + kv_head_idx * stride_kh
    v_base = V + kv_head_idx * stride_vh

    # --- Online softmax state (per Q row) ---
    row_max = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    row_sum = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    kv_bound = seq_len_kv if not IS_CAUSAL else tl.minimum((tile_q + 1) * BLOCK_M, seq_len_kv)

    # --- Main loop: iterate over KV tiles ---
    for kv_start in range(0, kv_bound, BLOCK_N):
        kv_start = tl.multiple_of(kv_start, BLOCK_N)

        # Load K^T [BLOCK_D, BLOCK_N]
        k_offs = (kv_offset + kv_start + kv_pos[None, :]) * stride_kbs + dim_pos[:, None]
        k = tl.load(
            k_base + k_offs,
            mask=((kv_start + kv_pos[None, :]) < seq_len_kv) & d_mask[:, None],
            other=0.0,
        )

        # scores = Q @ K^T * scale  [BLOCK_M, BLOCK_N]
        scores = tl.dot(q, k) * qk_scale
        scores = _apply_mask(scores, q_pos, kv_pos, seq_len_q, seq_len_kv, kv_start, IS_CAUSAL)

        if APPLY_SKIP_SOFTMAX:
            # --- Skip-softmax: check tile, skip V load if all rows negligible ---
            tile_row_max = tl.max(scores, 1)  # [BLOCK_M]
            can_skip = tile_row_max < (row_max + skip_threshold_log2)
            all_skip = tl.min(can_skip.to(tl.int32)) == 1

            if not all_skip:
                # Online softmax update
                m_new = tl.maximum(row_max, tile_row_max)
                p = tl.math.exp2(scores - m_new[:, None])
                p = tl.where(can_skip[:, None], 0.0, p)
                l_new = tl.sum(p, 1)
                correction = tl.math.exp2(row_max - m_new)
                row_sum = row_sum * correction + l_new
                acc = acc * correction[:, None]

                # Load V and accumulate
                v_offs = (kv_offset + kv_start + kv_pos[:, None]) * stride_vbs + dim_pos[None, :]
                v = tl.load(
                    v_base + v_offs,
                    mask=((kv_start + kv_pos[:, None]) < seq_len_kv) & d_mask[None, :],
                    other=0.0,
                )
                acc = tl.dot(p.to(v.dtype), v, acc)
                row_max = m_new
            # All rows negligible — skip V load, softmax update, accumulation
            elif APPLY_SKIP_V25:
                # V2.5: pool-K approximate weight + fresh precomputed v_mean
                # k is [BLOCK_D, BLOCK_N] (transposed K already in SRAM)
                k_mean = tl.sum(k.to(tl.float32), 1) / BLOCK_N  # [BLOCK_D]
                # Q @ k_mean → [BLOCK_M] per-row approximate score
                approx_score = tl.sum(q.to(tl.float32) * k_mean[None, :], 1) * qk_scale
                # BLOCK_M exponentials (not BLOCK_M*BLOCK_N)
                p_approx = tl.math.exp2(approx_score - row_max)  # [BLOCK_M]

                # Load fresh precomputed v_mean
                kv_tile_idx = kv_start // BLOCK_N
                vm_ptr = (
                    Vmean_cache
                    + batch_idx * stride_vm_b
                    + kv_head_idx * stride_vm_h
                    + kv_tile_idx * stride_vm_t
                    + dim_pos
                )
                fresh_v_mean = tl.load(vm_ptr, mask=d_mask, other=0.0)  # [BLOCK_D]

                # Update denominator and accumulator
                row_sum += p_approx
                acc += p_approx[:, None] * fresh_v_mean[None, :]
        else:
            # --- Standard path: no skip check ---
            m_new = tl.maximum(row_max, tl.max(scores, 1))
            p = tl.math.exp2(scores - m_new[:, None])
            l_new = tl.sum(p, 1)
            correction = tl.math.exp2(row_max - m_new)
            row_sum = row_sum * correction + l_new
            acc = acc * correction[:, None]

            v_offs = (kv_offset + kv_start + kv_pos[:, None]) * stride_vbs + dim_pos[None, :]
            v = tl.load(
                v_base + v_offs,
                mask=((kv_start + kv_pos[:, None]) < seq_len_kv) & d_mask[None, :],
                other=0.0,
            )
            acc = tl.dot(p.to(v.dtype), v, acc)
            row_max = m_new

    # --- Final normalization: output = acc / row_sum ---
    acc = acc / row_sum[:, None]

    if STORE_LSE:
        lse = row_max + tl.math.log2(row_sum)
        lse = tl.where(row_sum == 0.0, float("-inf"), lse)
        lse_ptrs = (q_offset + q_pos) * stride_lse_tok + head_idx * stride_lse_head
        tl.store(Lse + lse_ptrs, lse, mask=q_pos < seq_len_q)

    o_ptrs = (q_offset + q_pos[:, None]) * stride_obs + head_idx * stride_oh + dim_pos[None, :]
    tl.store(Out + o_ptrs, acc, mask=(q_pos[:, None] < seq_len_q) & d_mask[None, :])


# ---------------------------------------------------------------------------
# Forward kernel (autotuned wrapper around _attn_fwd_body)
# ---------------------------------------------------------------------------
@triton.autotune(configs=_FWD_CONFIGS, key=["N_CTX", "HEAD_DIM"])
@triton.jit
def _attn_fwd(
    Q,
    K,
    V,
    qk_scale,
    b_start_loc,
    b_seq_len,
    b_start_loc_k,
    b_seq_len_k,
    Out,
    Lse,
    stride_qbs,
    stride_qh,
    stride_kbs,
    stride_kh,
    stride_vbs,
    stride_vh,
    stride_obs,
    stride_oh,
    stride_lse_tok,
    stride_lse_head,
    N_CTX,
    kv_group_num: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    STORE_LSE: tl.constexpr,
    APPLY_SKIP_SOFTMAX: tl.constexpr = False,
    skip_threshold_log2=0.0,
):
    """Autotuned forward kernel (V1 skip-softmax, no V2.5 cache)."""
    _attn_fwd_body(
        Q,
        K,
        V,
        qk_scale,
        b_start_loc,
        b_seq_len,
        b_start_loc_k,
        b_seq_len_k,
        Out,
        Lse,
        stride_qbs,
        stride_qh,
        stride_kbs,
        stride_kh,
        stride_vbs,
        stride_vh,
        stride_obs,
        stride_oh,
        stride_lse_tok,
        stride_lse_head,
        N_CTX,
        kv_group_num=kv_group_num,
        BLOCK_M=BLOCK_M,
        BLOCK_D=BLOCK_D,
        BLOCK_N=BLOCK_N,
        IS_CAUSAL=IS_CAUSAL,
        HEAD_DIM=HEAD_DIM,
        STORE_LSE=STORE_LSE,
        APPLY_SKIP_SOFTMAX=APPLY_SKIP_SOFTMAX,
        skip_threshold_log2=skip_threshold_log2,
        APPLY_SKIP_V25=False,
    )


# ---------------------------------------------------------------------------
# Backward kernels
# ---------------------------------------------------------------------------
@triton.jit
def _attn_bwd_preprocess(
    Out,
    dO,
    Delta,
    stride_obs,
    stride_oh,
    stride_dobs,
    stride_doh,
    stride_delta_tok,
    stride_delta_head,
    total_tokens,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """Phase 1 of backward: compute delta_i = rowsum(O_i * dO_i).

    Delta is used in the dS computation: dS = P * (dP - delta).
    This avoids recomputing O in the dQ/dK/dV kernels.
    """
    head = tl.program_id(0)
    offs_tok = tl.program_id(1) * BLOCK_M + tl.arange(0, BLOCK_M)
    dim_pos = tl.arange(0, BLOCK_D)
    mask_tok = offs_tok < total_tokens
    mask_d = dim_pos < HEAD_DIM

    # Load O and dO tiles [BLOCK_M, BLOCK_D]
    o = tl.load(
        Out + offs_tok[:, None] * stride_obs + head * stride_oh + dim_pos[None, :],
        mask=mask_tok[:, None] & mask_d[None, :],
        other=0.0,
    )
    do = tl.load(
        dO + offs_tok[:, None] * stride_dobs + head * stride_doh + dim_pos[None, :],
        mask=mask_tok[:, None] & mask_d[None, :],
        other=0.0,
    )

    # delta_i = sum_d(O[i,d] * dO[i,d]) per token position
    delta = tl.sum(o * do, axis=1)
    tl.store(Delta + offs_tok * stride_delta_tok + head * stride_delta_head, delta, mask=mask_tok)


@triton.jit
def _attn_bwd_dq(
    Q,
    K,
    V,
    dO,
    dQ,
    Lse,
    Delta,
    b_start_loc,
    b_seq_len,
    b_start_loc_k,
    b_seq_len_k,
    qk_scale,
    sm_scale,
    stride_qbs,
    stride_qh,
    stride_kbs,
    stride_kh,
    stride_vbs,
    stride_vh,
    stride_dobs,
    stride_doh,
    stride_dqbs,
    stride_dqh,
    stride_lse_tok,
    stride_lse_head,
    kv_group_num: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    APPLY_SKIP_SOFTMAX: tl.constexpr = False,
    skip_threshold_log2=0.0,
):
    """Phase 3 of backward: compute dQ for one Q tile, looping over KV tiles.

    For each KV tile, recomputes attention scores S = Q @ K^T, then:
        P = softmax(S)  (via exp2 and saved LSE)
        dP = dO @ V^T
        dS = P * (dP - delta)
        dQ += dS @ K
    """
    # --- Grid: each thread block handles one (batch, q_head, q_tile) ---
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    tile_q = tl.program_id(2)
    kv_head_idx = head_idx // kv_group_num

    # --- Load per-sequence varlen metadata ---
    seq_len_q = tl.load(b_seq_len + batch_idx)
    seq_len_kv = tl.load(b_seq_len_k + batch_idx)
    q_offset = tl.load(b_start_loc + batch_idx)
    kv_offset = tl.load(b_start_loc_k + batch_idx)

    if tile_q * BLOCK_M >= seq_len_q:
        return

    q_pos = tile_q * BLOCK_M + tl.arange(0, BLOCK_M)
    kv_pos = tl.arange(0, BLOCK_N)
    dim_pos = tl.arange(0, BLOCK_D)
    d_mask = dim_pos < HEAD_DIM
    q_mask = q_pos < seq_len_q

    # --- Load Q, dO tiles: stay in SRAM for the entire KV loop ---
    q_ptrs = (q_offset + q_pos[:, None]) * stride_qbs + head_idx * stride_qh + dim_pos[None, :]
    q = tl.load(Q + q_ptrs, mask=q_mask[:, None] & d_mask[None, :], other=0.0)
    do_ptrs = (q_offset + q_pos[:, None]) * stride_dobs + head_idx * stride_doh + dim_pos[None, :]
    do = tl.load(dO + do_ptrs, mask=q_mask[:, None] & d_mask[None, :], other=0.0)

    # Load saved LSE and delta from forward pass (same [total_tokens, heads] layout)
    row_ptrs = (q_offset + q_pos) * stride_lse_tok + head_idx * stride_lse_head
    lse = tl.load(Lse + row_ptrs, mask=q_mask, other=0.0)
    row_delta = tl.load(Delta + row_ptrs, mask=q_mask, other=0.0)

    dq = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    kv_bound = seq_len_kv if not IS_CAUSAL else tl.minimum((tile_q + 1) * BLOCK_M, seq_len_kv)

    # --- Loop over KV tiles: recompute S, then compute dQ contribution ---
    for kv_start in range(0, kv_bound, BLOCK_N):
        kv_mask = (kv_start + kv_pos) < seq_len_kv

        # Load K^T and V for this KV tile
        k_ptrs = (
            (kv_offset + kv_start + kv_pos[None, :]) * stride_kbs
            + kv_head_idx * stride_kh
            + dim_pos[:, None]
        )
        kT = tl.load(K + k_ptrs, mask=kv_mask[None, :] & d_mask[:, None], other=0.0)
        v_ptrs = (
            (kv_offset + kv_start + kv_pos[:, None]) * stride_vbs
            + kv_head_idx * stride_vh
            + dim_pos[None, :]
        )
        v = tl.load(V + v_ptrs, mask=kv_mask[:, None] & d_mask[None, :], other=0.0)

        # Recompute attention: S = Q @ K^T, P = exp2(S - LSE)
        scores = tl.dot(q, kT) * qk_scale
        scores = _apply_mask(scores, q_pos, kv_pos, seq_len_q, seq_len_kv, kv_start, IS_CAUSAL)
        p = tl.math.exp2(scores - lse[:, None])

        # Re-apply skip-softmax: zero out rows that were skipped in forward
        if APPLY_SKIP_SOFTMAX:
            tile_row_max = tl.max(scores, 1)
            can_skip = tile_row_max < (lse + skip_threshold_log2)
            p = tl.where(can_skip[:, None], 0.0, p)

        # dP = dO @ V^T, dS = P * (dP - delta), dQ += dS @ K
        dp = tl.dot(do, tl.trans(v))
        ds = p * (dp - row_delta[:, None])
        dq += tl.dot(ds.to(kT.dtype), tl.trans(kT))

    # --- Store dQ (scaled by sm_scale since scores were pre-scaled by qk_scale) ---
    dq *= sm_scale
    dq_ptrs = (q_offset + q_pos[:, None]) * stride_dqbs + head_idx * stride_dqh + dim_pos[None, :]
    tl.store(dQ + dq_ptrs, dq.to(q.dtype), mask=q_mask[:, None] & d_mask[None, :])


@triton.jit
def _attn_bwd_dkdv(
    Q,
    K,
    V,
    dO,
    dK,
    dV,
    Lse,
    Delta,
    b_start_loc,
    b_seq_len,
    b_start_loc_k,
    b_seq_len_k,
    qk_scale,
    sm_scale,
    stride_qbs,
    stride_qh,
    stride_kbs,
    stride_kh,
    stride_vbs,
    stride_vh,
    stride_dobs,
    stride_doh,
    stride_dkbs,
    stride_dkh,
    stride_dvbs,
    stride_dvh,
    stride_lse_tok,
    stride_lse_head,
    kv_group_num: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    APPLY_SKIP_SOFTMAX: tl.constexpr = False,
    skip_threshold_log2=0.0,
):
    """Phase 2 of backward: compute dK, dV for one KV tile.

    Loops over all Q tiles (and GQA heads sharing this KV head), accumulating:
        dV += P^T @ dO
        dK += dS^T @ Q    where dS = P * (dO @ V^T - delta)
    """
    # --- Grid: each thread block handles one (batch, kv_head, kv_tile) ---
    batch_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)
    tile_kv = tl.program_id(2)

    # --- Load per-sequence varlen metadata ---
    seq_len_q = tl.load(b_seq_len + batch_idx)
    seq_len_kv = tl.load(b_seq_len_k + batch_idx)
    q_offset = tl.load(b_start_loc + batch_idx)
    kv_offset = tl.load(b_start_loc_k + batch_idx)

    kv_start = tile_kv * BLOCK_N
    if kv_start >= seq_len_kv:
        return

    kv_pos = tl.arange(0, BLOCK_N)  # Relative positions within this KV tile
    dim_pos = tl.arange(0, BLOCK_D)
    d_mask = dim_pos < HEAD_DIM
    kv_abs = kv_start + kv_pos  # Absolute positions for memory access
    kv_mask = kv_abs < seq_len_kv

    # --- Load K, V tiles: stay in SRAM throughout the Q loop ---
    kv_k_ptrs = (
        (kv_offset + kv_abs[:, None]) * stride_kbs + kv_head_idx * stride_kh + dim_pos[None, :]
    )
    kv_v_ptrs = (
        (kv_offset + kv_abs[:, None]) * stride_vbs + kv_head_idx * stride_vh + dim_pos[None, :]
    )
    k_tile = tl.load(K + kv_k_ptrs, mask=kv_mask[:, None] & d_mask[None, :], other=0.0)
    v_tile = tl.load(V + kv_v_ptrs, mask=kv_mask[:, None] & d_mask[None, :], other=0.0)
    kT = tl.trans(k_tile)

    # --- Accumulate dK, dV across all Q tiles ---
    dk = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)

    n_q_tiles = (seq_len_q + BLOCK_M - 1) // BLOCK_M
    # Causal: Q position i attends to KV 0..i, so this KV tile (at kv_start)
    # only receives gradients from Q tiles where q_pos >= kv_start. Skip earlier ones.
    first_q_tile = kv_start // BLOCK_M if IS_CAUSAL else 0
    q_pos_base = tl.arange(0, BLOCK_M)

    for qi in range(first_q_tile, n_q_tiles):
        q_pos = qi * BLOCK_M + q_pos_base
        q_mask = q_pos < seq_len_q

        # GQA: accumulate contributions from all Q heads sharing this KV head
        for g in range(kv_group_num):
            head_idx = kv_head_idx * kv_group_num + g

            # Load Q, dO, LSE, delta for this Q tile and head
            q_ptrs = (
                (q_offset + q_pos[:, None]) * stride_qbs + head_idx * stride_qh + dim_pos[None, :]
            )
            q_tile = tl.load(Q + q_ptrs, mask=q_mask[:, None] & d_mask[None, :], other=0.0)
            do_ptrs = (
                (q_offset + q_pos[:, None]) * stride_dobs + head_idx * stride_doh + dim_pos[None, :]
            )
            do_tile = tl.load(dO + do_ptrs, mask=q_mask[:, None] & d_mask[None, :], other=0.0)
            lse_ptrs = (q_offset + q_pos) * stride_lse_tok + head_idx * stride_lse_head
            lse = tl.load(Lse + lse_ptrs, mask=q_mask, other=0.0)
            row_delta = tl.load(Delta + lse_ptrs, mask=q_mask, other=0.0)

            # Recompute attention: S = Q @ K^T, P = exp2(S - LSE)
            scores = tl.dot(q_tile, kT) * qk_scale
            scores = _apply_mask(scores, q_pos, kv_pos, seq_len_q, seq_len_kv, kv_start, IS_CAUSAL)
            p = tl.math.exp2(scores - lse[:, None])

            # Re-apply skip-softmax: zero out rows that were skipped in forward
            if APPLY_SKIP_SOFTMAX:
                tile_row_max = tl.max(scores, 1)
                can_skip = tile_row_max < (lse + skip_threshold_log2)
                p = tl.where(can_skip[:, None], 0.0, p)

            # dV += P^T @ dO
            dv += tl.dot(tl.trans(p.to(do_tile.dtype)), do_tile)
            # dS = P * (dO @ V^T - delta), dK += dS^T @ Q
            dp = tl.dot(do_tile, tl.trans(v_tile))
            ds = p * (dp - row_delta[:, None])
            dk += tl.dot(tl.trans(ds.to(q_tile.dtype)), q_tile)

    # --- Store dK, dV (dK scaled by sm_scale) ---
    dk *= sm_scale
    tl.store(dK + kv_k_ptrs, dk.to(k_tile.dtype), mask=kv_mask[:, None] & d_mask[None, :])
    tl.store(dV + kv_v_ptrs, dv.to(v_tile.dtype), mask=kv_mask[:, None] & d_mask[None, :])


# ---------------------------------------------------------------------------
# Autograd wrapper + public API
# ---------------------------------------------------------------------------
class _Attention(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        b_start_loc,
        b_seq_len,
        max_input_len,
        is_causal,
        sm_scale,
        b_start_loc_k,
        b_seq_len_k,
        max_input_len_k,
        skip_softmax_threshold,
        skip_softmax_normalize_by_seqlen,
    ):
        HEAD_DIM = q.shape[2]
        num_q_heads = q.shape[1]
        num_kv_heads = k.shape[1]
        kv_group_num = num_q_heads // num_kv_heads
        batch = b_seq_len.shape[0]

        # Prefill: Q/K/V are the same packed tensor, reuse Q offsets for K/V.
        # Decode: K/V is a separate KV cache tensor, caller must pass explicit metadata.
        if b_seq_len_k is None:
            b_seq_len_k = b_seq_len
            b_start_loc_k = b_start_loc
            max_input_len_k = max_input_len

        # Pre-multiply scale by log2(e) so the kernel can use exp2()
        # exp(score * sm_scale) = exp2(score * sm_scale * log2(e))
        qk_scale = sm_scale * LOG2E
        # Triton tiles must be powers of 2; pad head dim
        BLOCK_D = triton.next_power_of_2(HEAD_DIM)

        # Skip-softmax: convert threshold to log2 space for the kernel.
        # Two modes:
        #   LLM (default): skip_threshold_log2 = log2(threshold)
        #     Skip if tile_row_max < row_max + log2(threshold)
        #   Diffusion (normalize_by_seqlen=True): skip_threshold_log2 = -threshold * log2(seq_k)
        #     Equivalent to: skip if gap >= threshold * log(seq_k)  (sequence-length-invariant)
        apply_skip = skip_softmax_threshold is not None and skip_softmax_threshold > 0.0
        if apply_skip:
            import math

            if skip_softmax_normalize_by_seqlen:
                skip_threshold_log2 = -skip_softmax_threshold * math.log2(max_input_len_k)
            else:
                skip_threshold_log2 = math.log2(skip_softmax_threshold)
        else:
            skip_threshold_log2 = 0.0

        o = torch.empty_like(q)
        lse = torch.empty(q.shape[0], num_q_heads, device=q.device, dtype=torch.float32)

        # Grid: (batch, q_heads, q_tiles). Uses a function because BLOCK_M is autotuned.
        def grid(META):
            return (batch, num_q_heads, triton.cdiv(max_input_len, META["BLOCK_M"]))

        _attn_fwd[grid](
            q,
            k,
            v,
            qk_scale,
            b_start_loc,
            b_seq_len,
            b_start_loc_k,
            b_seq_len_k,
            o,
            lse,
            q.stride(0),
            q.stride(1),
            k.stride(0),
            k.stride(1),
            v.stride(0),
            v.stride(1),
            o.stride(0),
            o.stride(1),
            lse.stride(0),
            lse.stride(1),
            N_CTX=max_input_len,
            kv_group_num=kv_group_num,
            BLOCK_D=BLOCK_D,
            IS_CAUSAL=is_causal,
            HEAD_DIM=HEAD_DIM,
            STORE_LSE=True,
            APPLY_SKIP_SOFTMAX=apply_skip,
            skip_threshold_log2=skip_threshold_log2,
            # BLOCK_M, BLOCK_N, num_warps, num_stages chosen by autotune
        )

        ctx.save_for_backward(q, k, v, o, lse, b_start_loc, b_seq_len, b_start_loc_k, b_seq_len_k)
        ctx.max_input_len = max_input_len
        ctx.max_input_len_k = max_input_len_k
        ctx.sm_scale = sm_scale
        ctx.qk_scale = qk_scale
        ctx.is_causal = is_causal
        ctx.HEAD_DIM = HEAD_DIM
        ctx.kv_group_num = kv_group_num
        ctx.num_q_heads = num_q_heads
        ctx.num_kv_heads = num_kv_heads
        ctx.batch = batch
        ctx.apply_skip = apply_skip
        ctx.skip_threshold_log2 = skip_threshold_log2
        return o

    @staticmethod
    def backward(ctx, grad_output):
        q, k, v, o, lse, b_start_loc, b_seq_len, b_start_loc_k, b_seq_len_k = ctx.saved_tensors
        HEAD_DIM = ctx.HEAD_DIM
        BLOCK = 64  # smaller block for backward to reduce shared memory pressure
        BLOCK_D = triton.next_power_of_2(HEAD_DIM)
        do = grad_output.contiguous()
        num_warps = 4

        # Phase 1: delta = rowsum(O * dO)
        delta = torch.empty_like(lse)
        _attn_bwd_preprocess[(ctx.num_q_heads, triton.cdiv(q.shape[0], BLOCK))](
            o,
            do,
            delta,
            o.stride(0),
            o.stride(1),
            do.stride(0),
            do.stride(1),
            delta.stride(0),
            delta.stride(1),
            q.shape[0],
            HEAD_DIM=HEAD_DIM,
            BLOCK_D=BLOCK_D,
            BLOCK_M=BLOCK,
        )

        dq = torch.zeros_like(q)
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)

        bwd_args = (
            q,
            k,
            v,
            do,
            lse,
            delta,
            b_start_loc,
            b_seq_len,
            b_start_loc_k,
            b_seq_len_k,
            ctx.qk_scale,
            ctx.sm_scale,
            q.stride(0),
            q.stride(1),
            k.stride(0),
            k.stride(1),
            v.stride(0),
            v.stride(1),
            do.stride(0),
            do.stride(1),
        )

        # Phase 2: dK, dV
        _attn_bwd_dkdv[(ctx.batch, ctx.num_kv_heads, triton.cdiv(ctx.max_input_len_k, BLOCK))](
            *bwd_args[:4],
            dk,
            dv,
            *bwd_args[4:],
            dk.stride(0),
            dk.stride(1),
            dv.stride(0),
            dv.stride(1),
            lse.stride(0),
            lse.stride(1),
            kv_group_num=ctx.kv_group_num,
            BLOCK_M=BLOCK,
            BLOCK_D=BLOCK_D,
            BLOCK_N=BLOCK,
            IS_CAUSAL=ctx.is_causal,
            HEAD_DIM=HEAD_DIM,
            APPLY_SKIP_SOFTMAX=ctx.apply_skip,
            skip_threshold_log2=ctx.skip_threshold_log2,
            num_warps=num_warps,
            num_stages=1,
        )

        # Phase 3: dQ
        _attn_bwd_dq[(ctx.batch, ctx.num_q_heads, triton.cdiv(ctx.max_input_len, BLOCK))](
            *bwd_args[:4],
            dq,
            *bwd_args[4:],
            dq.stride(0),
            dq.stride(1),
            lse.stride(0),
            lse.stride(1),
            kv_group_num=ctx.kv_group_num,
            BLOCK_M=BLOCK,
            BLOCK_D=BLOCK_D,
            BLOCK_N=BLOCK,
            IS_CAUSAL=ctx.is_causal,
            HEAD_DIM=HEAD_DIM,
            APPLY_SKIP_SOFTMAX=ctx.apply_skip,
            skip_threshold_log2=ctx.skip_threshold_log2,
            num_warps=num_warps,
            num_stages=1,
        )

        return dq, dk, dv, None, None, None, None, None, None, None, None, None, None


def _attention_v25_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    b_start_loc: torch.Tensor,
    b_seq_len: torch.Tensor,
    max_input_len: int,
    is_causal: bool,
    sm_scale: float,
    b_start_loc_k: torch.Tensor,
    b_seq_len_k: torch.Tensor,
    max_input_len_k: int,
    skip_threshold_log2: float,
    v_mean_cache: torch.Tensor,
) -> torch.Tensor:
    """V2.5 forward: pool-K approximate weights + fresh precomputed v_mean.

    No autograd — V2.5 is a forward-only inference optimization.
    """
    HEAD_DIM = q.shape[2]
    num_q_heads = q.shape[1]
    num_kv_heads = k.shape[1]
    kv_group_num = num_q_heads // num_kv_heads
    batch = b_seq_len.shape[0]
    qk_scale = sm_scale * LOG2E
    BLOCK_D = triton.next_power_of_2(HEAD_DIM)
    BLOCK_M = 128
    BLOCK_N = 64

    o = torch.empty_like(q)
    lse = torch.empty(q.shape[0], num_q_heads, device=q.device, dtype=torch.float32)

    # Precompute fresh v_mean for all KV tiles (VSA-style)
    num_kv_tiles = triton.cdiv(max_input_len_k, BLOCK_N)
    _precompute_vmean[(batch, num_kv_heads, num_kv_tiles)](
        v,
        v_mean_cache,
        b_start_loc_k,
        b_seq_len_k,
        v.stride(0),
        v.stride(1),
        v_mean_cache.stride(0),
        v_mean_cache.stride(1),
        v_mean_cache.stride(2),
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
        HEAD_DIM=HEAD_DIM,
    )

    grid = (batch, num_q_heads, triton.cdiv(max_input_len, BLOCK_M))

    _attn_fwd_body[grid](
        q,
        k,
        v,
        qk_scale,
        b_start_loc,
        b_seq_len,
        b_start_loc_k,
        b_seq_len_k,
        o,
        lse,
        q.stride(0),
        q.stride(1),
        k.stride(0),
        k.stride(1),
        v.stride(0),
        v.stride(1),
        o.stride(0),
        o.stride(1),
        lse.stride(0),
        lse.stride(1),
        N_CTX=max_input_len,
        kv_group_num=kv_group_num,
        BLOCK_M=BLOCK_M,
        BLOCK_D=BLOCK_D,
        BLOCK_N=BLOCK_N,
        IS_CAUSAL=is_causal,
        HEAD_DIM=HEAD_DIM,
        STORE_LSE=False,
        APPLY_SKIP_SOFTMAX=True,
        skip_threshold_log2=skip_threshold_log2,
        APPLY_SKIP_V25=True,
        Vmean_cache=v_mean_cache,
        stride_vm_b=v_mean_cache.stride(0),
        stride_vm_h=v_mean_cache.stride(1),
        stride_vm_t=v_mean_cache.stride(2),
        num_warps=4,
        num_stages=1,
    )
    return o


def attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    b_start_loc: torch.Tensor,
    b_seq_len: torch.Tensor,
    max_input_len: int,
    is_causal: bool = True,
    softmax_scale: float | None = None,
    b_start_loc_k: torch.Tensor | None = None,
    b_seq_len_k: torch.Tensor | None = None,
    max_input_len_k: int | None = None,
    *,
    skip_softmax_threshold: float | None = None,
    skip_softmax_normalize_by_seqlen: bool = False,
    v_mean_cache: torch.Tensor | None = None,
) -> torch.Tensor:
    """Variable-length flash attention with GQA, autograd, and optional skip-softmax.

    Args:
        q: [total_q_tokens, num_q_heads, head_dim]
        k: [total_kv_tokens, num_kv_heads, head_dim]
        v: [total_kv_tokens, num_kv_heads, head_dim]
        b_start_loc: [batch] start offset of each Q sequence in the flat tensor.
        b_seq_len: [batch] length of each Q sequence.
        max_input_len: Maximum Q sequence length (for grid sizing).
        is_causal: Whether to apply causal masking.
        softmax_scale: Scale factor (default: 1/sqrt(head_dim)).
        b_start_loc_k: [batch] start offset for K/V (None = same as Q).
        b_seq_len_k: [batch] length for K/V (None = same as Q).
        max_input_len_k: Maximum K/V sequence length (None = same as Q).
        skip_softmax_threshold: Skip KV tiles whose max attention score is
            below ``running_max * threshold`` for all Q rows.
        skip_softmax_normalize_by_seqlen: When True (diffusion mode), converts
            threshold via ``-threshold * log2(seq_k)`` for sequence-length
            invariance. When False (LLM mode), uses ``log2(threshold)``.
        v_mean_cache: V2.5 precomputed per-tile mean V vector. Shape
            ``[batch, num_kv_heads, num_kv_tiles, head_dim_padded]``.
            Populated by a precompute kernel before attention. Skipped tiles
            use pool-K approximate weight + this fresh v_mean.

    Returns:
        Output tensor [total_q_tokens, num_q_heads, head_dim].
    """
    import math

    sm_scale = 1.0 / (q.shape[2] ** 0.5) if softmax_scale is None else softmax_scale

    apply_skip = skip_softmax_threshold is not None and skip_softmax_threshold > 0.0
    use_v25 = apply_skip and v_mean_cache is not None

    if use_v25:
        # V2.5 path: no autograd, fixed block sizes, cache read/write
        if b_seq_len_k is None:
            b_seq_len_k = b_seq_len
            b_start_loc_k = b_start_loc
            max_input_len_k = max_input_len
        assert b_start_loc_k is not None
        assert max_input_len_k is not None
        assert skip_softmax_threshold is not None

        if skip_softmax_normalize_by_seqlen:
            skip_threshold_log2 = -skip_softmax_threshold * math.log2(max_input_len_k)
        else:
            skip_threshold_log2 = math.log2(skip_softmax_threshold)

        return _attention_v25_forward(
            q,
            k,
            v,
            b_start_loc,
            b_seq_len,
            max_input_len,
            is_causal,
            sm_scale,
            b_start_loc_k,
            b_seq_len_k,
            max_input_len_k,
            skip_threshold_log2,
            v_mean_cache,
        )

    # Standard path (V1 or no skip) with autograd support
    return _Attention.apply(
        q,
        k,
        v,
        b_start_loc,
        b_seq_len,
        max_input_len,
        is_causal,
        sm_scale,
        b_start_loc_k,
        b_seq_len_k,
        max_input_len_k,
        skip_softmax_threshold,
        skip_softmax_normalize_by_seqlen,
    )


__all__ = ["attention"]
