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

# Prefill kernel adapted from context_flashattention_nopad in SGLang / LightLLM.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Triton attention kernel for prefill and decode on flat packed tensors.

Supports variable sequence lengths, causal/non-causal masking, GQA, 2:4 sparse attention,
and autograd-compatible forward/backward.
"""

import torch
import triton
import triton.language as tl

LOG2E: float = 1.44269504088896


# ---------------------------------------------------------------------------
# 2:4 structured sparsity helpers
# ---------------------------------------------------------------------------
@triton.jit
def _sparse24_noabs_ops(x0, x1, x2, x3):
    """Compute 2:4 sparsity mask: for every 4 values, determine which 2 are largest."""
    (a1, a2, a3, a4, a5, a6) = (
        x0 > x1,
        x0 > x2,
        x0 > x3,
        x1 > x2,
        x1 > x3,
        x2 > x3,
    )
    na1 = a1 == 0
    na2 = a2 == 0
    na3 = a3 == 0
    na4 = a4 == 0
    na5 = a5 == 0
    na6 = a6 == 0
    m0 = a2 & a3 | a1 & a2 | a1 & a3
    m1 = na1 & a5 | a4 & a5 | na1 & a4
    m2 = na2 & na4 | na2 & a6 | na4 & a6
    m3 = na3 & na5 | na3 & na6 | na5 & na6
    return x0, x1, x2, x3, m0, m1, m2, m3


@triton.jit
def _apply_sparse24_to_qk_tile(
    qk,
    M: tl.constexpr,
    N: tl.constexpr,
    MASK_VAL: tl.constexpr,
):
    """Apply 2:4 sparsity to attention score tile [M, N]: keep top 2 of every 4 along N."""
    reshaped = tl.reshape(qk, (M, N // 4, 4))
    cols = tl.arange(0, 4)[None, None, :]
    x0 = tl.sum(tl.where(cols == 0, reshaped, 0.0), axis=2)
    x1 = tl.sum(tl.where(cols == 1, reshaped, 0.0), axis=2)
    x2 = tl.sum(tl.where(cols == 2, reshaped, 0.0), axis=2)
    x3 = tl.sum(tl.where(cols == 3, reshaped, 0.0), axis=2)
    _, _, _, _, m0, m1, m2, m3 = _sparse24_noabs_ops(x0, x1, x2, x3)
    s0 = tl.where(m0, x0, MASK_VAL)
    s1 = tl.where(m1, x1, MASK_VAL)
    s2 = tl.where(m2, x2, MASK_VAL)
    s3 = tl.where(m3, x3, MASK_VAL)
    sparse_reshaped = tl.full((M, N // 4, 4), 0.0, dtype=qk.dtype)
    sparse_reshaped = tl.where((cols == 0), tl.expand_dims(s0, 2), sparse_reshaped)
    sparse_reshaped = tl.where((cols == 1), tl.expand_dims(s1, 2), sparse_reshaped)
    sparse_reshaped = tl.where((cols == 2), tl.expand_dims(s2, 2), sparse_reshaped)
    sparse_reshaped = tl.where((cols == 3), tl.expand_dims(s3, 2), sparse_reshaped)
    sparse_qk = tl.reshape(sparse_reshaped, (M, N))
    return sparse_qk


# ---------------------------------------------------------------------------
# Shared: recompute masked S tile (used by forward inner loop and backward)
# ---------------------------------------------------------------------------
@triton.jit
def _mask_and_sparsify(
    qk,
    offs_m,
    offs_n,
    cur_batch_seq_len,
    cur_batch_kv_len,
    start_n,
    start_m,
    IS_CAUSAL: tl.constexpr,
    APPLY_SPARSE24: tl.constexpr,
    SKIP_DIAGONAL_BLOCKS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_FWD: tl.constexpr,
):
    """Apply causal mask, padding mask, and optional 2:4 sparsity to a QK tile.

    BLOCK_FWD is the forward kernel's block size, used for diagonal detection so that
    backward kernels (which may use smaller tiles) produce the same sparsity pattern.
    """
    if IS_CAUSAL:
        qk += tl.where(
            (start_n + offs_n[None, :] < cur_batch_kv_len)
            & (offs_m[:, None] >= (start_n + offs_n[None, :])),
            0,
            float("-inf"),
        )
    else:
        qk += tl.where((start_n + offs_n[None, :]) < cur_batch_kv_len, 0, float("-inf"))

    if APPLY_SPARSE24:
        if IS_CAUSAL and SKIP_DIAGONAL_BLOCKS:
            # Diagonal detection at BLOCK_FWD granularity (matches forward)
            q_pos_min = start_m * BLOCK_M
            fwd_q_tile_start = (q_pos_min // BLOCK_FWD) * BLOCK_FWD
            fwd_q_tile_end = fwd_q_tile_start + BLOCK_FWD
            fwd_k_tile_start = (start_n // BLOCK_FWD) * BLOCK_FWD
            fwd_k_tile_end = fwd_k_tile_start + BLOCK_FWD
            is_diagonal = (fwd_k_tile_start < fwd_q_tile_end) & (fwd_k_tile_end > fwd_q_tile_start)
            if not is_diagonal:
                qk = _apply_sparse24_to_qk_tile(qk, BLOCK_M, BLOCK_N, float("-inf"))
        else:
            qk = _apply_sparse24_to_qk_tile(qk, BLOCK_M, BLOCK_N, float("-inf"))
    return qk


# ---------------------------------------------------------------------------
# Forward kernel
# ---------------------------------------------------------------------------
@triton.jit
def _fwd_kernel_prefill(
    Q,
    K,
    V,
    qk_scale,
    B_Start_Loc,
    B_Seqlen,
    B_Start_Loc_K,
    B_Seqlen_K,
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
    kv_group_num: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    Lk: tl.constexpr,
    APPLY_SPARSE24: tl.constexpr,
    SKIP_DIAGONAL_BLOCKS: tl.constexpr,
    STORE_LSE: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    start_m = tl.program_id(2)
    cur_kv_head = cur_head // kv_group_num

    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_kv_len = tl.load(B_Seqlen_K + cur_batch)
    cur_batch_q_start = tl.load(B_Start_Loc + cur_batch)
    cur_batch_kv_start = tl.load(B_Start_Loc_K + cur_batch)

    block_start_loc = BLOCK_M * start_m
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_d = offs_d < Lk

    off_q = (
        (cur_batch_q_start + offs_m[:, None]) * stride_qbs + cur_head * stride_qh + offs_d[None, :]
    )
    q = tl.load(Q + off_q, mask=(offs_m[:, None] < cur_batch_seq_len) & mask_d[None, :], other=0.0)

    off_k = offs_n[None, :] * stride_kbs + cur_kv_head * stride_kh + offs_d[:, None]
    off_v = offs_n[:, None] * stride_vbs + cur_kv_head * stride_vh + offs_d[None, :]
    k_ptrs = K + off_k
    v_ptrs = V + off_v

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    block_mask = tl.where(block_start_loc < cur_batch_seq_len, 1, 0)
    end_n = (
        cur_batch_kv_len if not IS_CAUSAL else tl.minimum((start_m + 1) * BLOCK_M, cur_batch_kv_len)
    )

    for start_n in range(0, block_mask * end_n, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k = tl.load(
            k_ptrs + (cur_batch_kv_start + start_n) * stride_kbs,
            mask=((start_n + offs_n[None, :]) < cur_batch_kv_len) & mask_d[:, None],
            other=0.0,
        )
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)
        qk *= qk_scale
        qk = _mask_and_sparsify(
            qk,
            offs_m,
            offs_n,
            cur_batch_seq_len,
            cur_batch_kv_len,
            start_n,
            start_m,
            IS_CAUSAL,
            APPLY_SPARSE24,
            SKIP_DIAGONAL_BLOCKS,
            BLOCK_M,
            BLOCK_N,
            BLOCK_M,  # BLOCK_FWD = BLOCK_M in forward
        )

        # deferred-normalization online softmax (exp2)
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.math.exp2(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]
        v = tl.load(
            v_ptrs + (cur_batch_kv_start + start_n) * stride_vbs,
            mask=((start_n + offs_n[:, None]) < cur_batch_kv_len) & mask_d[None, :],
            other=0.0,
        )
        acc = tl.dot(p.to(v.dtype), v, acc)
        m_i = m_ij

    acc = acc / l_i[:, None]

    if STORE_LSE:
        lse_i = m_i + tl.math.log2(l_i)
        lse_i = tl.where(l_i == 0.0, float("-inf"), lse_i)
        off_lse = (cur_batch_q_start + offs_m) * stride_lse_tok + cur_head * stride_lse_head
        tl.store(Lse + off_lse, lse_i, mask=offs_m < cur_batch_seq_len)

    off_o = (
        (cur_batch_q_start + offs_m[:, None]) * stride_obs + cur_head * stride_oh + offs_d[None, :]
    )
    tl.store(Out + off_o, acc, mask=(offs_m[:, None] < cur_batch_seq_len) & mask_d[None, :])


# ---------------------------------------------------------------------------
# Backward kernels
# ---------------------------------------------------------------------------
@triton.jit
def _bwd_preprocess(
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
    Lk: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """Compute D_i = rowsum(O_i * dO_i) per query position per head."""
    head = tl.program_id(0)
    offs_tok = tl.program_id(1) * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    mask_tok = offs_tok < total_tokens
    mask_d = offs_d < Lk
    o = tl.load(
        Out + offs_tok[:, None] * stride_obs + head * stride_oh + offs_d[None, :],
        mask=mask_tok[:, None] & mask_d[None, :],
        other=0.0,
    )
    do = tl.load(
        dO + offs_tok[:, None] * stride_dobs + head * stride_doh + offs_d[None, :],
        mask=mask_tok[:, None] & mask_d[None, :],
        other=0.0,
    )
    delta = tl.sum(o * do, axis=1)
    tl.store(Delta + offs_tok * stride_delta_tok + head * stride_delta_head, delta, mask=mask_tok)


@triton.jit
def _bwd_kernel_dq(
    Q,
    K,
    V,
    dO,
    dQ,
    Lse,
    Delta,
    B_Start_Loc,
    B_Seqlen,
    B_Start_Loc_K,
    B_Seqlen_K,
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
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    Lk: tl.constexpr,
    APPLY_SPARSE24: tl.constexpr,
    SKIP_DIAGONAL_BLOCKS: tl.constexpr,
    BLOCK_FWD: tl.constexpr,
):
    """Backward: compute dQ for one Q tile, looping over KV tiles."""
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    start_m = tl.program_id(2)
    cur_kv_head = cur_head // kv_group_num

    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_kv_len = tl.load(B_Seqlen_K + cur_batch)
    cur_batch_q_start = tl.load(B_Start_Loc + cur_batch)
    cur_batch_kv_start = tl.load(B_Start_Loc_K + cur_batch)

    if start_m * BLOCK_M >= cur_batch_seq_len:
        return

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    mask_d = offs_d < Lk
    mask_m = offs_m < cur_batch_seq_len

    # Load Q, dO — stay in registers
    off_qm = (
        (cur_batch_q_start + offs_m[:, None]) * stride_qbs + cur_head * stride_qh + offs_d[None, :]
    )
    q = tl.load(Q + off_qm, mask=mask_m[:, None] & mask_d[None, :], other=0.0)
    off_dom = (
        (cur_batch_q_start + offs_m[:, None]) * stride_dobs
        + cur_head * stride_doh
        + offs_d[None, :]
    )
    do = tl.load(dO + off_dom, mask=mask_m[:, None] & mask_d[None, :], other=0.0)

    off_lse = (cur_batch_q_start + offs_m) * stride_lse_tok + cur_head * stride_lse_head
    lse = tl.load(Lse + off_lse, mask=mask_m, other=0.0)
    delta = tl.load(Delta + off_lse, mask=mask_m, other=0.0)

    dq = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    end_n = (
        cur_batch_kv_len if not IS_CAUSAL else tl.minimum((start_m + 1) * BLOCK_M, cur_batch_kv_len)
    )
    offs_n = tl.arange(0, BLOCK_N)

    for start_n in range(0, end_n, BLOCK_N):
        mask_n = (start_n + offs_n) < cur_batch_kv_len
        # Load K^T [BLOCK_DMODEL, BLOCK_N] and V [BLOCK_N, BLOCK_DMODEL]
        off_kn = (
            (cur_batch_kv_start + start_n + offs_n[None, :]) * stride_kbs
            + cur_kv_head * stride_kh
            + offs_d[:, None]
        )
        kT = tl.load(K + off_kn, mask=mask_n[None, :] & mask_d[:, None], other=0.0)
        off_vn = (
            (cur_batch_kv_start + start_n + offs_n[:, None]) * stride_vbs
            + cur_kv_head * stride_vh
            + offs_d[None, :]
        )
        v = tl.load(V + off_vn, mask=mask_n[:, None] & mask_d[None, :], other=0.0)

        # Recompute S [BLOCK_M, BLOCK_N]
        s = tl.dot(q, kT)
        s *= qk_scale
        s = _mask_and_sparsify(
            s,
            offs_m,
            offs_n,
            cur_batch_seq_len,
            cur_batch_kv_len,
            start_n,
            start_m,
            IS_CAUSAL,
            APPLY_SPARSE24,
            SKIP_DIAGONAL_BLOCKS,
            BLOCK_M,
            BLOCK_N,
            BLOCK_FWD,
        )
        p = tl.math.exp2(s - lse[:, None])

        # dP = dO @ V^T [BLOCK_M, BLOCK_N]
        dp = tl.dot(do, tl.trans(v))
        # dS = P * (dP - delta)
        ds = p * (dp - delta[:, None])
        # dQ += dS @ K (= dS @ kT^T)
        dq += tl.dot(ds.to(kT.dtype), tl.trans(kT))

    dq *= sm_scale
    off_dqm = (
        (cur_batch_q_start + offs_m[:, None]) * stride_dqbs
        + cur_head * stride_dqh
        + offs_d[None, :]
    )
    tl.store(dQ + off_dqm, dq.to(q.dtype), mask=mask_m[:, None] & mask_d[None, :])


@triton.jit
def _bwd_kernel_dkdv(
    Q,
    K,
    V,
    dO,
    dK,
    dV,
    Lse,
    Delta,
    B_Start_Loc,
    B_Seqlen,
    B_Start_Loc_K,
    B_Seqlen_K,
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
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    Lk: tl.constexpr,
    APPLY_SPARSE24: tl.constexpr,
    SKIP_DIAGONAL_BLOCKS: tl.constexpr,
    BLOCK_FWD: tl.constexpr,
):
    """Backward: compute dK, dV for one KV tile, looping over Q tiles and GQA heads."""
    cur_batch = tl.program_id(0)
    cur_kv_head = tl.program_id(1)
    start_n = tl.program_id(2)

    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_kv_len = tl.load(B_Seqlen_K + cur_batch)
    cur_batch_q_start = tl.load(B_Start_Loc + cur_batch)
    cur_batch_kv_start = tl.load(B_Start_Loc_K + cur_batch)

    kv_block_start = start_n * BLOCK_N
    if kv_block_start >= cur_batch_kv_len:
        return

    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    mask_d = offs_d < Lk
    mask_n = (kv_block_start + offs_n) < cur_batch_kv_len

    # Load K, V tiles [BLOCK_N, BLOCK_DMODEL] — stay in SRAM
    abs_offs_n = kv_block_start + offs_n
    off_kn = (
        (cur_batch_kv_start + abs_offs_n[:, None]) * stride_kbs
        + cur_kv_head * stride_kh
        + offs_d[None, :]
    )
    off_vn = (
        (cur_batch_kv_start + abs_offs_n[:, None]) * stride_vbs
        + cur_kv_head * stride_vh
        + offs_d[None, :]
    )
    k_tile = tl.load(K + off_kn, mask=mask_n[:, None] & mask_d[None, :], other=0.0)
    v_tile = tl.load(V + off_vn, mask=mask_n[:, None] & mask_d[None, :], other=0.0)
    kT_tile = tl.trans(k_tile)  # [BLOCK_DMODEL, BLOCK_N]

    dk = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)

    num_q_tiles = (cur_batch_seq_len + BLOCK_M - 1) // BLOCK_M
    q_tile_start = kv_block_start // BLOCK_M if IS_CAUSAL else 0
    offs_m_base = tl.arange(0, BLOCK_M)

    for q_tile_idx in range(q_tile_start, num_q_tiles):
        offs_m = q_tile_idx * BLOCK_M + offs_m_base
        mask_m = offs_m < cur_batch_seq_len

        for gqa_idx in range(kv_group_num):
            cur_head = cur_kv_head * kv_group_num + gqa_idx
            # Load Q, dO [BLOCK_M, BLOCK_DMODEL]
            off_qm = (
                (cur_batch_q_start + offs_m[:, None]) * stride_qbs
                + cur_head * stride_qh
                + offs_d[None, :]
            )
            q_tile = tl.load(Q + off_qm, mask=mask_m[:, None] & mask_d[None, :], other=0.0)
            off_dom = (
                (cur_batch_q_start + offs_m[:, None]) * stride_dobs
                + cur_head * stride_doh
                + offs_d[None, :]
            )
            do_tile = tl.load(dO + off_dom, mask=mask_m[:, None] & mask_d[None, :], other=0.0)
            # Load lse, delta [BLOCK_M]
            off_lse = (cur_batch_q_start + offs_m) * stride_lse_tok + cur_head * stride_lse_head
            lse = tl.load(Lse + off_lse, mask=mask_m, other=0.0)
            delta_val = tl.load(Delta + off_lse, mask=mask_m, other=0.0)

            # Recompute S [BLOCK_M, BLOCK_N] in ORIGINAL orientation
            s = tl.dot(q_tile, kT_tile)
            s *= qk_scale
            s = _mask_and_sparsify(
                s,
                offs_m,
                offs_n,
                cur_batch_seq_len,
                cur_batch_kv_len,
                start_n * BLOCK_N,
                q_tile_idx,
                IS_CAUSAL,
                APPLY_SPARSE24,
                SKIP_DIAGONAL_BLOCKS,
                BLOCK_M,
                BLOCK_N,
                BLOCK_FWD,
            )
            p = tl.math.exp2(s - lse[:, None])

            # dV += P^T @ dO
            dv += tl.dot(tl.trans(p.to(do_tile.dtype)), do_tile)
            # dP = dO @ V^T
            dp = tl.dot(do_tile, tl.trans(v_tile))
            # dS = P * (dP - delta)
            ds = p * (dp - delta_val[:, None])
            # dK += dS^T @ Q
            dk += tl.dot(tl.trans(ds.to(q_tile.dtype)), q_tile)

    dk *= sm_scale
    tl.store(dK + off_kn, dk.to(k_tile.dtype), mask=mask_n[:, None] & mask_d[None, :])
    tl.store(dV + off_vn, dv.to(v_tile.dtype), mask=mask_n[:, None] & mask_d[None, :])


# ---------------------------------------------------------------------------
# Python helpers
# ---------------------------------------------------------------------------
def _prepare_fwd_args(
    q,
    k,
    v,
    b_start_loc,
    b_seq_len,
    max_input_len,
    softmax_scale,
    b_start_loc_k,
    b_seq_len_k,
    max_input_len_k,
    apply_sparse24,
):
    """Validate inputs and derive common parameters."""
    if q.dim() != 3 or k.dim() != 3 or v.dim() != 3:
        raise ValueError(
            "q, k, v must be rank-3 [total_tokens, num_heads, head_dim]; "
            f"got q.dim()={q.dim()}, k.dim()={k.dim()}, v.dim()={v.dim()}."
        )
    head_dim = q.shape[2]
    num_kv_heads = k.shape[1]
    if num_kv_heads <= 0:
        raise ValueError(f"k.shape[1] (num_kv_heads) must be positive; got {num_kv_heads}.")
    if q.shape[1] % num_kv_heads != 0:
        raise ValueError(
            f"num_heads must be divisible by num_kv_heads; got {q.shape[1]} and {num_kv_heads}."
        )
    if b_seq_len_k is None:
        total_q = q.shape[0]
        if k.shape[0] != total_q or v.shape[0] != total_q:
            raise ValueError(
                "For self-attention, q, k, v must have same shape[0]; "
                f"got {q.shape[0]}, {k.shape[0]}, {v.shape[0]}."
            )
        b_seq_len_k = b_seq_len
        b_start_loc_k = b_start_loc
        max_input_len_k = max_input_len

    batch = b_seq_len.shape[0]
    if b_start_loc_k is None:
        b_start_loc_k = torch.zeros(batch + 1, device=q.device, dtype=torch.int32)
        b_start_loc_k[1:] = torch.cumsum(b_seq_len_k.to(torch.int64), dim=0)
        b_start_loc_k = b_start_loc_k[:batch]
    if max_input_len_k is None:
        max_input_len_k = int(b_seq_len_k.max().item())

    Lk = head_dim
    num_q_heads = q.shape[1]
    kv_group_num = num_q_heads // num_kv_heads
    sm_scale = 1.0 / (Lk**0.5) if softmax_scale is None else softmax_scale
    qk_scale = sm_scale * LOG2E

    capability = torch.cuda.get_device_capability()
    BLOCK_FWD = 128 if capability[0] >= 8 else 64
    BLOCK_BWD = 64  # backward holds more tiles in SRAM; smaller block avoids shared memory overflow
    if apply_sparse24 and BLOCK_BWD % 4 != 0:
        raise ValueError(f"sparse24 requires BLOCK divisible by 4, got {BLOCK_BWD}")
    num_warps_fwd = 4 if Lk <= 64 else 8
    num_warps_bwd = 4  # fewer warps to reduce shared memory pressure in backward

    return (
        b_start_loc_k,
        b_seq_len_k,
        max_input_len_k,
        sm_scale,
        qk_scale,
        BLOCK_FWD,
        BLOCK_BWD,
        num_warps_fwd,
        num_warps_bwd,
        Lk,
        kv_group_num,
        num_q_heads,
        num_kv_heads,
        batch,
    )


# ---------------------------------------------------------------------------
# Autograd wrapper
# ---------------------------------------------------------------------------
class _ContextAttentionFunc(torch.autograd.Function):
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
        softmax_scale,
        apply_sparse24,
        skip_diagonal_blocks,
        b_start_loc_k,
        b_seq_len_k,
        max_input_len_k,
    ):
        (
            b_start_loc_k,
            b_seq_len_k,
            max_input_len_k,
            sm_scale,
            qk_scale,
            BLOCK_FWD,
            BLOCK_BWD,
            num_warps_fwd,
            num_warps_bwd,
            Lk,
            kv_group_num,
            num_q_heads,
            num_kv_heads,
            batch,
        ) = _prepare_fwd_args(
            q,
            k,
            v,
            b_start_loc,
            b_seq_len,
            max_input_len,
            softmax_scale,
            b_start_loc_k,
            b_seq_len_k,
            max_input_len_k,
            apply_sparse24,
        )

        o = torch.empty_like(q)
        lse = torch.empty(q.shape[0], num_q_heads, device=q.device, dtype=torch.float32)
        BLOCK_DMODEL = triton.next_power_of_2(Lk)
        grid = (batch, num_q_heads, triton.cdiv(max_input_len, BLOCK_FWD))

        _fwd_kernel_prefill[grid](
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
            kv_group_num=kv_group_num,
            BLOCK_M=BLOCK_FWD,
            BLOCK_DMODEL=BLOCK_DMODEL,
            BLOCK_N=BLOCK_FWD,
            IS_CAUSAL=is_causal,
            Lk=Lk,
            APPLY_SPARSE24=apply_sparse24,
            SKIP_DIAGONAL_BLOCKS=skip_diagonal_blocks,
            STORE_LSE=True,
            num_warps=num_warps_fwd,
            num_stages=1,
        )

        ctx.save_for_backward(q, k, v, o, lse, b_start_loc, b_seq_len, b_start_loc_k, b_seq_len_k)
        ctx.max_input_len = max_input_len
        ctx.max_input_len_k = max_input_len_k
        ctx.sm_scale = sm_scale
        ctx.qk_scale = qk_scale
        ctx.is_causal = is_causal
        ctx.apply_sparse24 = apply_sparse24
        ctx.skip_diagonal_blocks = skip_diagonal_blocks
        ctx.BLOCK_FWD = BLOCK_FWD
        ctx.BLOCK_BWD = BLOCK_BWD
        ctx.num_warps_bwd = num_warps_bwd
        ctx.Lk = Lk
        ctx.kv_group_num = kv_group_num
        ctx.num_q_heads = num_q_heads
        ctx.num_kv_heads = num_kv_heads
        ctx.batch = batch
        return o

    @staticmethod
    def backward(ctx, grad_output):
        q, k, v, o, lse, b_start_loc, b_seq_len, b_start_loc_k, b_seq_len_k = ctx.saved_tensors
        BLOCK = ctx.BLOCK_BWD
        Lk = ctx.Lk
        BLOCK_DMODEL = triton.next_power_of_2(Lk)
        do = grad_output.contiguous()
        num_warps = ctx.num_warps_bwd

        # Phase 1: delta = rowsum(O * dO)
        delta = torch.empty_like(lse)
        pre_grid = (ctx.num_q_heads, triton.cdiv(q.shape[0], BLOCK))
        _bwd_preprocess[pre_grid](
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
            Lk=Lk,
            BLOCK_DMODEL=BLOCK_DMODEL,
            BLOCK_M=BLOCK,
        )

        dq = torch.zeros_like(q)
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)

        # Phase 2: dK, dV
        grid_dkdv = (ctx.batch, ctx.num_kv_heads, triton.cdiv(ctx.max_input_len_k, BLOCK))
        _bwd_kernel_dkdv[grid_dkdv](
            q,
            k,
            v,
            do,
            dk,
            dv,
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
            dk.stride(0),
            dk.stride(1),
            dv.stride(0),
            dv.stride(1),
            lse.stride(0),
            lse.stride(1),
            kv_group_num=ctx.kv_group_num,
            BLOCK_M=BLOCK,
            BLOCK_DMODEL=BLOCK_DMODEL,
            BLOCK_N=BLOCK,
            IS_CAUSAL=ctx.is_causal,
            Lk=Lk,
            APPLY_SPARSE24=ctx.apply_sparse24,
            SKIP_DIAGONAL_BLOCKS=ctx.skip_diagonal_blocks,
            BLOCK_FWD=ctx.BLOCK_FWD,
            num_warps=num_warps,
            num_stages=1,
        )

        # Phase 3: dQ
        grid_dq = (ctx.batch, ctx.num_q_heads, triton.cdiv(ctx.max_input_len, BLOCK))
        _bwd_kernel_dq[grid_dq](
            q,
            k,
            v,
            do,
            dq,
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
            dq.stride(0),
            dq.stride(1),
            lse.stride(0),
            lse.stride(1),
            kv_group_num=ctx.kv_group_num,
            BLOCK_M=BLOCK,
            BLOCK_DMODEL=BLOCK_DMODEL,
            BLOCK_N=BLOCK,
            IS_CAUSAL=ctx.is_causal,
            Lk=Lk,
            APPLY_SPARSE24=ctx.apply_sparse24,
            SKIP_DIAGONAL_BLOCKS=ctx.skip_diagonal_blocks,
            BLOCK_FWD=ctx.BLOCK_FWD,
            num_warps=num_warps,
            num_stages=1,
        )

        return dq, dk, dv, None, None, None, None, None, None, None, None, None, None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def context_attention_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    b_start_loc: torch.Tensor,
    b_seq_len: torch.Tensor,
    max_input_len: int,
    is_causal: bool = True,
    softmax_scale: float | None = None,
    apply_sparse24: bool = False,
    skip_diagonal_blocks: bool = True,
    b_start_loc_k: torch.Tensor | None = None,
    b_seq_len_k: torch.Tensor | None = None,
    max_input_len_k: int | None = None,
) -> None:
    """Inference-only attention (no backward). Writes output to ``o`` in-place."""
    if o.shape[0] != q.shape[0] or o.shape[1] != q.shape[1] or o.shape[2] != q.shape[2]:
        raise ValueError(f"o must match q shape; got o={o.shape}, q={q.shape}.")

    (
        b_start_loc_k,
        b_seq_len_k,
        max_input_len_k,
        sm_scale,
        qk_scale,
        BLOCK_FWD,
        _BLOCK_BWD,
        num_warps_fwd,
        _num_warps_bwd,
        Lk,
        kv_group_num,
        num_q_heads,
        num_kv_heads,
        batch,
    ) = _prepare_fwd_args(
        q,
        k,
        v,
        b_start_loc,
        b_seq_len,
        max_input_len,
        softmax_scale,
        b_start_loc_k,
        b_seq_len_k,
        max_input_len_k,
        apply_sparse24,
    )

    grid = (batch, num_q_heads, triton.cdiv(max_input_len, BLOCK_FWD))
    _fwd_kernel_prefill[grid](
        q,
        k,
        v,
        qk_scale,
        b_start_loc,
        b_seq_len,
        b_start_loc_k,
        b_seq_len_k,
        o,
        None,
        q.stride(0),
        q.stride(1),
        k.stride(0),
        k.stride(1),
        v.stride(0),
        v.stride(1),
        o.stride(0),
        o.stride(1),
        0,
        0,
        kv_group_num=kv_group_num,
        BLOCK_M=BLOCK_FWD,
        BLOCK_DMODEL=triton.next_power_of_2(Lk),
        BLOCK_N=BLOCK_FWD,
        IS_CAUSAL=is_causal,
        Lk=Lk,
        APPLY_SPARSE24=apply_sparse24,
        SKIP_DIAGONAL_BLOCKS=skip_diagonal_blocks,
        STORE_LSE=False,
        num_warps=num_warps_fwd,
        num_stages=1,
    )


def context_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    b_start_loc: torch.Tensor,
    b_seq_len: torch.Tensor,
    max_input_len: int,
    is_causal: bool = True,
    softmax_scale: float | None = None,
    apply_sparse24: bool = False,
    skip_diagonal_blocks: bool = True,
    b_start_loc_k: torch.Tensor | None = None,
    b_seq_len_k: torch.Tensor | None = None,
    max_input_len_k: int | None = None,
) -> torch.Tensor:
    """Attention with autograd support (training). Returns output tensor with grad_fn."""
    return _ContextAttentionFunc.apply(
        q,
        k,
        v,
        b_start_loc,
        b_seq_len,
        max_input_len,
        is_causal,
        softmax_scale,
        apply_sparse24,
        skip_diagonal_blocks,
        b_start_loc_k,
        b_seq_len_k,
        max_input_len_k,
    )


__all__ = ["context_attention", "context_attention_fwd"]
