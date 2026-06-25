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

import math
from typing import Any

import torch
import triton
import triton.language as tl

# Helpers for optional N:M sparsity and sink/window-aware dense regions live
# in the sparsity package. The baseline forward kernel below calls them
# conditionally under constexpr guards, so the unified single-kernel design
# stays intact while keeping feature-specific logic in its own subpackage.
#
# Lazy import: Triton resolves @triton.jit names at kernel compile time (first
# call), not at definition time, so populating the module globals before the
# first ``attention()`` call is sufficient. Deferring avoids a circular import
# (common.attention/__init__.py ↔ sparsity.attention/__init__.py via this file).
_apply_sparse_nm_to_qk_tile: Any = None
_is_dense_region: Any = None
_skip_softmax_decision: Any = None
_p_qdq_fp8: Any = None
_p_qdq_nvfp4: Any = None


def _load_sparsity_helpers() -> None:
    global _apply_sparse_nm_to_qk_tile, _is_dense_region, _skip_softmax_decision
    if _apply_sparse_nm_to_qk_tile is None:
        from modelopt.torch.kernels.sparsity.attention.skip_softmax_helpers import (
            _apply_sparse_nm_to_qk_tile as _nm,
        )
        from modelopt.torch.kernels.sparsity.attention.skip_softmax_helpers import (
            _is_dense_region as _dense,
        )
        from modelopt.torch.kernels.sparsity.attention.skip_softmax_helpers import (
            _skip_softmax_decision as _skip,
        )

        _apply_sparse_nm_to_qk_tile = _nm
        _is_dense_region = _dense
        _skip_softmax_decision = _skip


def _load_p_qdq_helpers() -> None:
    global _p_qdq_fp8, _p_qdq_nvfp4
    if _p_qdq_fp8 is None:
        from modelopt.torch.kernels.quantization.attention.p_qdq import _p_qdq_nvfp4 as _nvfp4
        from modelopt.torch.kernels.quantization.common.fp8_quant import fp8_scalar_qdq as _fp8

        _p_qdq_fp8 = _fp8
        _p_qdq_nvfp4 = _nvfp4


# Maps the public p_qdq option to the kernel's P_QDQ constexpr.
_P_QDQ_MODES = {None: 0, "fp8": 1, "nvfp4": 2}


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

_MEASURE_BLOCK_M = 128
# 128 so the kernel sparsity-measurement block matches the PyTorch
# flash_skip_softmax calibration block (br = bc = 128) and the Triton
# calibration kernel; otherwise the two measure at different granularities.
_MEASURE_BLOCK_N = 128
_MEASURE_NUM_STAGES = 1
_MEASURE_NUM_WARPS = 4


# ---------------------------------------------------------------------------
# Paged KV cache helpers
# ---------------------------------------------------------------------------
@triton.jit
def _load_paged_k_tile(
    K_cache,  # [num_blocks, page_size, num_kv_heads, head_dim]
    Block_table,  # [batch, max_blocks_per_seq]
    batch_idx,
    kv_head_idx,
    kv_start,
    kv_pos,  # [BLOCK_N] relative positions
    dim_pos,  # [BLOCK_D]
    seq_len_kv,
    stride_kc_block,
    stride_kc_pos,
    stride_kc_head,
    PAGE_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    max_blocks_per_seq,
):
    """Load K^T tile [BLOCK_D, BLOCK_N] from paged KV cache."""
    d_mask = dim_pos < HEAD_DIM
    kv_abs = kv_start + kv_pos  # absolute token positions
    kv_valid = kv_abs < seq_len_kv

    # Translate token positions -> (page_id, offset_in_page)
    page_local = kv_abs // PAGE_SIZE
    offset_in_page = kv_abs % PAGE_SIZE
    page_global = tl.load(
        Block_table + batch_idx * max_blocks_per_seq + page_local,
        mask=kv_valid,
        other=0,
    )

    # Load K values: K_cache[page_global, offset_in_page, kv_head_idx, dim]
    # K^T layout [BLOCK_D, BLOCK_N] for Q @ K^T matmul
    k_ptrs = (
        # int64: real KV pools have >2^31/stride blocks, so page_global*stride_kc_block
        # overflows int32 at high block IDs -> negative offset -> illegal memory access.
        page_global[None, :].to(tl.int64) * stride_kc_block
        + offset_in_page[None, :] * stride_kc_pos
        + kv_head_idx * stride_kc_head
        + dim_pos[:, None]
    )
    return tl.load(K_cache + k_ptrs, mask=kv_valid[None, :] & d_mask[:, None], other=0.0)


@triton.jit
def _load_paged_v_tile(
    V_cache,  # [num_blocks, page_size, num_kv_heads, head_dim]
    Block_table,  # [batch, max_blocks_per_seq]
    batch_idx,
    kv_head_idx,
    kv_start,
    kv_pos,  # [BLOCK_N] relative positions
    dim_pos,  # [BLOCK_D]
    seq_len_kv,
    stride_vc_block,
    stride_vc_pos,
    stride_vc_head,
    PAGE_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    max_blocks_per_seq,
):
    """Load V tile [BLOCK_N, BLOCK_D] from paged KV cache."""
    d_mask = dim_pos < HEAD_DIM
    kv_abs = kv_start + kv_pos
    kv_valid = kv_abs < seq_len_kv

    page_local = kv_abs // PAGE_SIZE
    offset_in_page = kv_abs % PAGE_SIZE
    page_global = tl.load(
        Block_table + batch_idx * max_blocks_per_seq + page_local,
        mask=kv_valid,
        other=0,
    )

    # V layout [BLOCK_N, BLOCK_D]
    v_ptrs = (
        # int64: see _load_paged_k_tile — avoid int32 overflow at high block IDs.
        page_global[:, None].to(tl.int64) * stride_vc_block
        + offset_in_page[:, None] * stride_vc_pos
        + kv_head_idx * stride_vc_head
        + dim_pos[None, :]
    )
    return tl.load(V_cache + v_ptrs, mask=kv_valid[:, None] & d_mask[None, :], other=0.0)


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
        # In chunked prefill or prefix-cache hits, Q is the latest suffix of KV
        # rather than starting at KV position 0.
        q_to_k_offset = seq_len_kv - seq_len_q
        scores += tl.where(
            (kv_start + kv_pos[None, :] < seq_len_kv)
            & (q_pos[:, None] + q_to_k_offset >= (kv_start + kv_pos[None, :])),
            0,
            float("-inf"),
        )
    else:
        scores += tl.where((kv_start + kv_pos[None, :]) < seq_len_kv, 0, float("-inf"))
    return scores


# ---------------------------------------------------------------------------
# Forward kernel
# ---------------------------------------------------------------------------
@triton.autotune(configs=_FWD_CONFIGS, key=["N_CTX", "HEAD_DIM"])
@triton.jit
def _attn_fwd(
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
    BLOCK_M: tl.constexpr,  # Q tile size (autotuned)
    BLOCK_D: tl.constexpr,  # Head dim tile size (next_power_of_2(HEAD_DIM))
    BLOCK_N: tl.constexpr,  # KV tile size (autotuned)
    IS_CAUSAL: tl.constexpr,  # Whether to apply causal mask
    HEAD_DIM: tl.constexpr,  # Actual head dimension (for d_mask)
    STORE_LSE: tl.constexpr,  # Whether to save LSE for backward pass
    SPARSITY_N: tl.constexpr = 0,  # N:M sparsity — keep top-N of every M elements (0 = disabled)
    SPARSITY_M: tl.constexpr = 4,  # N:M sparsity — group size (4 or 8)
    DENSE_SINK_TOKENS: tl.constexpr = 0,  # Leading KV tokens kept dense (attention sinks)
    DENSE_RECENT_TOKENS: tl.constexpr = 128,  # Recent KV tokens kept dense (BLOCK_N-independent)
    APPLY_SKIP_SOFTMAX: tl.constexpr = False,  # Skip KV tiles with negligible scores
    SKIP_THRESHOLD_LOG2: tl.constexpr = 0.0,  # log2(lambda) in the kernel's scaled log2 score space
    P_QDQ: tl.constexpr = 0,  # Fake quant-dequant of softmax P: 0=off, 1=FP8 E4M3, 2=NVFP4
    p_qdq_scale=1.0,  # Per-tensor scale for softmax qdq (runtime scalar; amax/448 or amax/(6*448))
    Sparsity_total=None,  # Optional int64 scalar for counting total tiles (atomic)
    Sparsity_skipped=None,  # Optional int64 scalar for counting skipped tiles (atomic)
    MEASURE_SPARSITY: tl.constexpr = False,  # When True, count total/skipped tiles via atomic adds
    IS_PAGED: tl.constexpr = False,  # Whether K/V are in paged cache
    K_cache=None,  # [num_blocks, page_size, num_kv_heads, head_dim] paged K
    V_cache=None,  # [num_blocks, page_size, num_kv_heads, head_dim] paged V
    Block_table=None,  # [batch, max_blocks_per_seq] page table
    stride_kc_block=0,
    stride_kc_pos=0,
    stride_kc_head=0,
    stride_vc_block=0,
    stride_vc_pos=0,
    stride_vc_head=0,
    PAGE_SIZE: tl.constexpr = 16,
    max_blocks_per_seq=0,
):
    # --- Grid: (batch, num_q_heads, num_q_tiles) ---
    # Example: batch=2, num_q_heads=32, seq_len=256, BLOCK_M=128
    #   grid = (2, 32, 2), 128 thread blocks launched in parallel
    #   block (1, 5, 0) handles: batch 1, Q head 5, tokens 0-127
    batch_idx = tl.program_id(0)  # 0..batch-1
    head_idx = tl.program_id(1)  # 0..num_q_heads-1
    tile_q = tl.program_id(2)  # 0..ceil(seq_len/BLOCK_M)-1
    kv_head_idx = head_idx // kv_group_num  # GQA: map Q head to shared KV head

    # --- Load Q and KV varlen metadata ---
    seq_len_q = tl.load(b_seq_len + batch_idx)
    seq_len_kv = tl.load(b_seq_len_k + batch_idx)
    q_offset = tl.load(b_start_loc + batch_idx)
    kv_offset = tl.load(b_start_loc_k + batch_idx)

    if tile_q * BLOCK_M >= seq_len_q:
        return  # This Q tile is past the sequence end

    # --- Tile position indices ---
    q_pos = tile_q * BLOCK_M + tl.arange(0, BLOCK_M)  # Absolute Q token positions
    kv_pos = tl.arange(0, BLOCK_N)  # Relative KV positions within a tile
    dim_pos = tl.arange(0, BLOCK_D)  # Head dimension positions
    d_mask = dim_pos < HEAD_DIM  # Mask for non-power-of-2 head dims

    # --- Load Q tile [BLOCK_M, BLOCK_D]: stays in SRAM for the entire KV loop ---
    q_ptrs = (q_offset + q_pos[:, None]) * stride_qbs + head_idx * stride_qh + dim_pos[None, :]
    q = tl.load(Q + q_ptrs, mask=(q_pos[:, None] < seq_len_q) & d_mask[None, :], other=0.0)

    # Base pointers for K and V at this KV head (per-tile offset added in loop)
    k_base = K + kv_head_idx * stride_kh
    v_base = V + kv_head_idx * stride_vh

    # --- Online softmax state (per Q row) ---
    row_max = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")  # Running max for stability
    row_sum = tl.zeros([BLOCK_M], dtype=tl.float32)  # Running sum of exp(scores)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)  # Running weighted sum of V

    # Causal bound: chunked/prefix prefill Q tiles are suffixes of the KV span.
    causal_offset = seq_len_kv - seq_len_q
    kv_bound = (
        seq_len_kv
        if not IS_CAUSAL
        else tl.minimum(causal_offset + (tile_q + 1) * BLOCK_M, seq_len_kv)
    )

    # --- Main loop: iterate over KV tiles ---
    for kv_start in range(0, kv_bound, BLOCK_N):
        kv_start = tl.multiple_of(kv_start, BLOCK_N)  # Compiler hint for alignment

        # Load K^T [BLOCK_D, BLOCK_N] (transposed layout for Q @ K^T matmul)
        if IS_PAGED:
            k = _load_paged_k_tile(
                K_cache,
                Block_table,
                batch_idx,
                kv_head_idx,
                kv_start,
                kv_pos,
                dim_pos,
                seq_len_kv,
                stride_kc_block,
                stride_kc_pos,
                stride_kc_head,
                PAGE_SIZE,
                BLOCK_N,
                BLOCK_D,
                HEAD_DIM,
                max_blocks_per_seq,
            )
        else:
            k_offs = (kv_offset + kv_start + kv_pos[None, :]) * stride_kbs + dim_pos[:, None]
            k = tl.load(
                k_base + k_offs,
                mask=((kv_start + kv_pos[None, :]) < seq_len_kv) & d_mask[:, None],
                other=0.0,
            )

        # scores = Q @ K^T * scale  [BLOCK_M, BLOCK_N]
        scores = tl.dot(q, k) * qk_scale
        scores = _apply_mask(scores, q_pos, kv_pos, seq_len_q, seq_len_kv, kv_start, IS_CAUSAL)

        # --- Optional N:M sparse softmax ---
        if SPARSITY_N > 0:
            if not _is_dense_region(
                kv_start,
                tile_q,
                seq_len_q,
                seq_len_kv,
                BLOCK_M,
                DENSE_SINK_TOKENS,
                DENSE_RECENT_TOKENS,
            ):
                scores = _apply_sparse_nm_to_qk_tile(
                    scores, BLOCK_M, BLOCK_N, SPARSITY_N, SPARSITY_M
                )

        # Optional skip-softmax decision — the decision logic (and optional
        # atomic counter updates) lives in sparsity/attention; this kernel
        # just consults it under its constexpr guard.
        skip_tile = False
        if APPLY_SKIP_SOFTMAX:
            skip_tile = _skip_softmax_decision(
                scores,
                row_max,
                q_pos,
                seq_len_q,
                SKIP_THRESHOLD_LOG2,
                Sparsity_total,
                Sparsity_skipped,
                MEASURE_SPARSITY,
            )

        if not skip_tile:
            # --- Online softmax update ---
            m_new = tl.maximum(row_max, tl.max(scores, 1))
            p = tl.math.exp2(scores - m_new[:, None])
            l_new = tl.sum(p, 1)
            correction = tl.math.exp2(row_max - m_new)
            row_sum = row_sum * correction + l_new
            acc = acc * correction[:, None]

            # --- Optional softmax quant-dequant (emulates quantized P @ V) ---
            # row_sum keeps the unquantized p: the softmax denominator stays in
            # fp32 and only the quantized P is fed to BMM2.
            if P_QDQ == 1:
                p = _p_qdq_fp8(p, p_qdq_scale)
            elif P_QDQ == 2:
                p = _p_qdq_nvfp4(p, p_qdq_scale, BLOCK_M, BLOCK_N)

            # Load V and accumulate
            if IS_PAGED:
                v = _load_paged_v_tile(
                    V_cache,
                    Block_table,
                    batch_idx,
                    kv_head_idx,
                    kv_start,
                    kv_pos,
                    dim_pos,
                    seq_len_kv,
                    stride_vc_block,
                    stride_vc_pos,
                    stride_vc_head,
                    PAGE_SIZE,
                    BLOCK_N,
                    BLOCK_D,
                    HEAD_DIM,
                    max_blocks_per_seq,
                )
            else:
                v_offs = (kv_offset + kv_start + kv_pos[:, None]) * stride_vbs + dim_pos[None, :]
                v = tl.load(
                    v_base + v_offs,
                    mask=((kv_start + kv_pos[:, None]) < seq_len_kv) & d_mask[None, :],
                    other=0.0,
                )
            acc = tl.dot(p.to(v.dtype), v, acc)
            row_max = m_new
        # else: tile skipped — no softmax, no V load, no BMM2 for this tile

    # --- Final normalization: output = acc / row_sum ---
    # Clamp denominator to avoid 0/0 NaN when skip-softmax skips all KV tiles.
    # Safe because acc is also 0 in that case (never accumulated), so 0/eps = 0.
    acc = acc / tl.maximum(row_sum[:, None], 1e-6)

    # Save LSE for backward pass (log2-space: lse = max + log2(sum))
    if STORE_LSE:
        lse = row_max + tl.math.log2(row_sum)
        lse = tl.where(row_sum == 0.0, float("-inf"), lse)
        lse_ptrs = (q_offset + q_pos) * stride_lse_tok + head_idx * stride_lse_head
        tl.store(Lse + lse_ptrs, lse, mask=q_pos < seq_len_q)

    # --- Store output [BLOCK_M, BLOCK_D] ---
    o_ptrs = (q_offset + q_pos[:, None]) * stride_obs + head_idx * stride_oh + dim_pos[None, :]
    tl.store(Out + o_ptrs, acc, mask=(q_pos[:, None] < seq_len_q) & d_mask[None, :])


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
    SPARSITY_N: tl.constexpr = 0,
    SPARSITY_M: tl.constexpr = 4,
    DENSE_SINK_TOKENS: tl.constexpr = 0,
    DENSE_RECENT_TOKENS: tl.constexpr = 128,
    APPLY_SKIP_SOFTMAX: tl.constexpr = False,
    SKIP_THRESHOLD_LOG2: tl.constexpr = 0.0,
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
    causal_offset = seq_len_kv - seq_len_q
    kv_bound = (
        seq_len_kv
        if not IS_CAUSAL
        else tl.minimum(causal_offset + (tile_q + 1) * BLOCK_M, seq_len_kv)
    )

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

        # Re-apply N:M sparse softmax to match forward pass
        if SPARSITY_N > 0:
            if not _is_dense_region(
                kv_start,
                tile_q,
                seq_len_q,
                seq_len_kv,
                BLOCK_M,
                DENSE_SINK_TOKENS,
                DENSE_RECENT_TOKENS,
            ):
                scores = _apply_sparse_nm_to_qk_tile(
                    scores, BLOCK_M, BLOCK_N, SPARSITY_N, SPARSITY_M
                )

        p = tl.math.exp2(scores - lse[:, None])

        # Skip-softmax backward: zero out P for rows with negligible contribution.
        # Per-row using final LSE because forward/backward tile sizes may differ
        # (forward autotunes BLOCK_N; backward uses a fixed size), so per-tile
        # skip masks from forward wouldn't align. LSE >= any intermediate running
        # max, so this conservatively zeros out at least what forward skipped.
        if APPLY_SKIP_SOFTMAX:
            tile_row_max = tl.max(scores, 1)
            can_skip = tile_row_max < (lse + SKIP_THRESHOLD_LOG2)
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
    SPARSITY_N: tl.constexpr = 0,
    SPARSITY_M: tl.constexpr = 4,
    DENSE_SINK_TOKENS: tl.constexpr = 0,
    DENSE_RECENT_TOKENS: tl.constexpr = 128,
    APPLY_SKIP_SOFTMAX: tl.constexpr = False,
    SKIP_THRESHOLD_LOG2: tl.constexpr = 0.0,
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
    # Causal with chunked/prefix prefill: Q positions are offset into KV space.
    causal_offset = seq_len_kv - seq_len_q
    first_q_tile = tl.maximum((kv_start - causal_offset) // BLOCK_M, 0) if IS_CAUSAL else 0
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

            # Re-apply N:M sparse softmax to match forward pass
            if SPARSITY_N > 0:
                if not _is_dense_region(
                    kv_start,
                    qi,
                    seq_len_q,
                    seq_len_kv,
                    BLOCK_M,
                    DENSE_SINK_TOKENS,
                    DENSE_RECENT_TOKENS,
                ):
                    scores = _apply_sparse_nm_to_qk_tile(
                        scores, BLOCK_M, BLOCK_N, SPARSITY_N, SPARSITY_M
                    )

            p = tl.math.exp2(scores - lse[:, None])

            # Skip-softmax backward: zero out P for rows with negligible contribution.
            # Per-row using final LSE because forward/backward tile sizes may differ
            # (forward autotunes BLOCK_N; backward uses a fixed size), so per-tile
            # skip masks from forward wouldn't align. LSE >= any intermediate running
            # max, so this conservatively zeros out at least what forward skipped.
            if APPLY_SKIP_SOFTMAX:
                tile_row_max = tl.max(scores, 1)
                can_skip = tile_row_max < (lse + SKIP_THRESHOLD_LOG2)
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
        sparsity_n,
        sparsity_m,
        dense_sink_tokens,
        dense_recent_tokens,
        skip_softmax_threshold,
        measure_sparsity,
        p_qdq_mode,
        p_qdq_scale,
        k_cache,
        v_cache,
        block_table,
        page_size,
    ):
        HEAD_DIM = q.shape[2]
        num_q_heads = q.shape[1]
        num_kv_heads = k.shape[1]
        kv_group_num = num_q_heads // num_kv_heads
        batch = b_seq_len.shape[0]

        is_paged = k_cache is not None

        # Backward indexes contiguous K/V via b_start_loc_k. In paged mode, callers
        # pass dummy k/v (e.g. torch.empty(0, ...)) and KV lives in k_cache/v_cache,
        # so dK/dV would be computed against the dummies — silently incorrect. Fail
        # fast instead of allowing autograd to produce wrong gradients.
        if is_paged and (q.requires_grad or k.requires_grad or v.requires_grad):
            raise NotImplementedError(
                "Paged KV cache path is forward-only; backward is not implemented."
            )

        # Prefill: Q/K/V are the same packed tensor, reuse Q offsets for K/V.
        # Decode: K/V is a separate KV cache tensor, caller must pass explicit metadata.
        if b_seq_len_k is None:
            b_seq_len_k = b_seq_len
            b_start_loc_k = b_start_loc
            max_input_len_k = max_input_len

        # Paged mode: b_start_loc_k may be None (KV is in paged cache, not contiguous).
        # Provide a dummy tensor so Triton can compile the tl.load (it won't be used).
        if b_start_loc_k is None:
            b_start_loc_k = torch.zeros_like(b_start_loc)

        # Pre-multiply scale by log2(e) so the kernel can use exp2()
        # exp(score * sm_scale) = exp2(score * sm_scale * log2(e))
        qk_scale = sm_scale * LOG2E
        # Triton tiles must be powers of 2; pad head dim
        BLOCK_D = triton.next_power_of_2(HEAD_DIM)

        # Convert the public lambda threshold to the kernel's log2 score space.
        if skip_softmax_threshold is not None and skip_softmax_threshold > 0.0:
            apply_skip = True
            # scores already include sm_scale and LOG2E, so the lambda cutoff is
            # just converted from natural-log probability space to log2 space.
            skip_threshold_log2 = math.log2(skip_softmax_threshold)
        else:
            apply_skip = False
            skip_threshold_log2 = 0.0

        o = torch.empty_like(q)
        lse = torch.empty(q.shape[0], num_q_heads, device=q.device, dtype=torch.float32)

        # Optional runtime sparsity counters (single int64 scalars for atomic adds)
        do_measure = measure_sparsity and apply_skip
        if do_measure:
            sparsity_total = torch.zeros(1, dtype=torch.int64, device=q.device)
            sparsity_skipped = torch.zeros(1, dtype=torch.int64, device=q.device)
        else:
            sparsity_total = None
            sparsity_skipped = None

        fwd_args = (
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
        )
        fwd_kwargs = {
            # Bucket the autotune key to a power-of-2 length REGIME, not the exact seq
            # len: keying on exact max_input_len re-ran autotune per unique prefill length
            # (the TTFT blowup on varying-length workloads). next_power_of_2 collapses that
            # to one tune per length regime. (N_CTX is key-only, never used in compute.)
            "N_CTX": triton.next_power_of_2(max(1, max_input_len)),
            "kv_group_num": kv_group_num,
            "BLOCK_D": BLOCK_D,
            "IS_CAUSAL": is_causal,
            "HEAD_DIM": HEAD_DIM,
            "STORE_LSE": True,
            "SPARSITY_N": sparsity_n,
            "SPARSITY_M": sparsity_m,
            "DENSE_SINK_TOKENS": dense_sink_tokens,
            "DENSE_RECENT_TOKENS": dense_recent_tokens,
            "APPLY_SKIP_SOFTMAX": apply_skip,
            "SKIP_THRESHOLD_LOG2": skip_threshold_log2,
            "P_QDQ": p_qdq_mode,
            "p_qdq_scale": p_qdq_scale,
            "Sparsity_total": sparsity_total,
            "Sparsity_skipped": sparsity_skipped,
            "MEASURE_SPARSITY": do_measure,
            "IS_PAGED": is_paged,
            "K_cache": k_cache,
            "V_cache": v_cache,
            "Block_table": block_table,
            "stride_kc_block": k_cache.stride(0) if is_paged else 0,
            "stride_kc_pos": k_cache.stride(1) if is_paged else 0,
            "stride_kc_head": k_cache.stride(2) if is_paged else 0,
            "stride_vc_block": v_cache.stride(0) if is_paged else 0,
            "stride_vc_pos": v_cache.stride(1) if is_paged else 0,
            "stride_vc_head": v_cache.stride(2) if is_paged else 0,
            "PAGE_SIZE": page_size,
            "max_blocks_per_seq": block_table.shape[1] if is_paged else 0,
        }

        # Grid: (batch, q_heads, q_tiles). Uses a function because BLOCK_M is autotuned.
        def grid(META):
            return (batch, num_q_heads, triton.cdiv(max_input_len, META["BLOCK_M"]))

        # Triton launches on torch.cuda.current_device(), which is not
        # necessarily the device the tensors live on (e.g. under accelerate
        # device_map="auto" sharding). Activate the tensor's device so the
        # kernel dereferences the right pointers instead of triggering an
        # illegal memory access.
        with torch.cuda.device(q.device):
            if do_measure:
                # Runtime counters mutate global tensors, so do not run them through
                # autotune candidate trials. Use one stable config for measurement.
                _attn_fwd.fn[grid](
                    *fwd_args,
                    **fwd_kwargs,
                    BLOCK_M=_MEASURE_BLOCK_M,
                    BLOCK_N=_MEASURE_BLOCK_N,
                    num_warps=_MEASURE_NUM_WARPS,
                    num_stages=_MEASURE_NUM_STAGES,
                )
            else:
                _attn_fwd[grid](
                    *fwd_args,
                    **fwd_kwargs,
                    # BLOCK_M, BLOCK_N, num_warps, num_stages chosen by autotune
                )

        # Store sparsity counters on the output tensor for retrieval by callers
        if do_measure:
            o._sparsity_total = sparsity_total.item()
            o._sparsity_skipped = sparsity_skipped.item()

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
        ctx.sparsity_n = sparsity_n
        ctx.sparsity_m = sparsity_m
        ctx.dense_sink_tokens = dense_sink_tokens
        ctx.dense_recent_tokens = dense_recent_tokens
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

        # Triton launches on torch.cuda.current_device(), which is not
        # necessarily the device the tensors live on (e.g. under accelerate
        # device_map="auto" sharding). Activate the tensor's device for each
        # launch so the kernels dereference the right pointers instead of
        # triggering an illegal memory access.

        # Phase 1: delta = rowsum(O * dO)
        delta = torch.empty_like(lse)
        with torch.cuda.device(q.device):
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
        with torch.cuda.device(q.device):
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
                SPARSITY_N=ctx.sparsity_n,
                SPARSITY_M=ctx.sparsity_m,
                DENSE_SINK_TOKENS=ctx.dense_sink_tokens,
                DENSE_RECENT_TOKENS=ctx.dense_recent_tokens,
                APPLY_SKIP_SOFTMAX=ctx.apply_skip,
                SKIP_THRESHOLD_LOG2=ctx.skip_threshold_log2,
                num_warps=num_warps,
                num_stages=1,
            )

        # Phase 3: dQ
        with torch.cuda.device(q.device):
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
                SPARSITY_N=ctx.sparsity_n,
                SPARSITY_M=ctx.sparsity_m,
                DENSE_SINK_TOKENS=ctx.dense_sink_tokens,
                DENSE_RECENT_TOKENS=ctx.dense_recent_tokens,
                APPLY_SKIP_SOFTMAX=ctx.apply_skip,
                SKIP_THRESHOLD_LOG2=ctx.skip_threshold_log2,
                num_warps=num_warps,
                num_stages=1,
            )

        return (
            dq,
            dk,
            dv,
            None,  # b_start_loc
            None,  # b_seq_len
            None,  # max_input_len
            None,  # is_causal
            None,  # sm_scale
            None,  # b_start_loc_k
            None,  # b_seq_len_k
            None,  # max_input_len_k
            None,  # sparsity_n
            None,  # sparsity_m
            None,  # dense_sink_tokens
            None,  # dense_recent_tokens
            None,  # skip_softmax_threshold
            None,  # measure_sparsity
            None,  # p_qdq_mode
            None,  # p_qdq_scale
            None,  # k_cache
            None,  # v_cache
            None,  # block_table
            None,  # page_size
        )


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
    sparsity_n: int = 0,
    sparsity_m: int = 4,
    dense_sink_tokens: int = 0,
    dense_recent_tokens: int = 64,
    skip_softmax_threshold: float | None = None,
    measure_sparsity: bool = False,
    p_qdq: str | None = None,
    p_qdq_amax: float = 1.0,
    k_cache: torch.Tensor | None = None,
    v_cache: torch.Tensor | None = None,
    block_table: torch.Tensor | None = None,
    page_size: int = 16,
) -> torch.Tensor:
    """Variable-length flash attention with GQA, autograd, optional sparsity, and paged KV.

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
        sparsity_n: N:M sparsity — keep top-N of every M attention scores
            along the key dimension. Set to 0 to disable. Examples:
            ``sparsity_n=2, sparsity_m=4`` for 2:4 sparsity;
            ``sparsity_n=4, sparsity_m=8`` for 4:8 sparsity.
        sparsity_m: N:M sparsity — group size (4 or 8).
        dense_sink_tokens: Leading KV tokens excluded from N:M sparsity and kept dense.
            Absolute token count, BLOCK_N-independent.
        dense_recent_tokens: Recent KV tokens excluded from N:M sparsity and kept dense.
            Absolute token count, BLOCK_N-independent. Default 64 tokens.
        skip_softmax_threshold: BLASST threshold lambda
            (https://arxiv.org/pdf/2512.12087). Skip KV tiles where
            ``exp(tile_max - running_max) < lambda``, meaning the tile's
            softmax contribution is negligible. Tiles are skipped entirely
            (no softmax, V load, or BMM2). Set to ``None`` or ``0`` to disable.
        measure_sparsity: When True and skip-softmax is active, count total
            and skipped tiles via atomic counters. The counts are stored as
            ``_sparsity_total`` and ``_sparsity_skipped`` attributes on the
            returned output tensor.
        p_qdq: Fake quant-dequant of the softmax probabilities ``P``
            before the ``P @ V`` matmul (BMM2), emulating quantized attention.
            ``"fp8"`` round-trips P through FP8 E4M3 with a static per-tensor
            scale (see ``p_qdq_amax``). ``"nvfp4"`` applies the two-level NVFP4
            recipe: E2M1 elements with one FP8 E4M3 scale per 16 elements along
            the key dimension (the BMM2 contraction axis; every autotuned
            BLOCK_N is a multiple of 16). The softmax denominator stays
            unquantized. The backward pass uses the straight-through estimator:
            gradients are computed from the unquantized P, matching QAT
            references that keep the backward dots in high precision.
            Set to ``None`` to disable.
        p_qdq_amax: Per-tensor amax for the softmax-P quant-dequant. The
            kernel's unnormalized P lies in [0, 1] (the max-subtraction caps
            every entry at ``exp2(0) = 1``), so 1 is the theoretical upper
            bound of its amax — hence the default of 1.0. It is converted to
            the standard per-tensor scale internally: ``amax / 448`` for FP8,
            and the global scale ``amax / (6 * 448)`` for NVFP4. A runtime
            scalar — user-set or calibrated values do not recompile the
            kernel. Values above amax saturate.
        k_cache: Paged K cache [num_blocks, page_size, num_kv_heads, head_dim].
            When provided, K/V are read from paged cache via block_table
            instead of from contiguous k/v tensors.
        v_cache: Paged V cache [num_blocks, page_size, num_kv_heads, head_dim].
        block_table: Page table [batch, max_blocks_per_seq] mapping sequence
            block indices to global page IDs.
        page_size: Number of tokens per page in the KV cache.

    Returns:
        Output tensor [total_q_tokens, num_q_heads, head_dim].

    Note:
        The paged KV path (``k_cache``/``v_cache`` not None) is forward-only —
        ``backward`` raises ``NotImplementedError`` if any of ``q``/``k``/``v``
        require grad, because the saved ``k``/``v`` are dummy tensors in paged
        mode and dK/dV would be silently incorrect.
    """
    # Both loaders must run unconditionally: Triton computes a kernel's
    # dependency hash once, on the first call, walking the full AST. If the
    # qdq helpers were still None at that point, their source would be
    # permanently excluded from the cache key and later edits to them would
    # silently reuse stale compiled kernels from the on-disk cache.
    _load_sparsity_helpers()
    _load_p_qdq_helpers()
    if p_qdq not in _P_QDQ_MODES:
        raise ValueError(
            f"p_qdq must be one of {sorted(k for k in _P_QDQ_MODES if k)} or None, got {p_qdq!r}"
        )
    p_qdq_mode = _P_QDQ_MODES[p_qdq]
    # Convert the per-tensor amax to the kernel's scale convention
    # (``q = cast(p / scale) * scale``): FP8 uses ``amax / 448``; NVFP4 uses the
    # global scale ``amax / (6 * 448)``. amax=1 (the default, the theoretical
    # upper bound of P's amax) therefore maps to the standard full-range scale.
    p_qdq_scale = 1.0
    if p_qdq_mode:
        if not (math.isfinite(p_qdq_amax) and p_qdq_amax > 0):
            raise ValueError(f"p_qdq_amax must be a finite positive value, got {p_qdq_amax}")
        p_qdq_scale = p_qdq_amax / 448.0 if p_qdq == "fp8" else p_qdq_amax / (6.0 * 448.0)
    sm_scale = 1.0 / (q.shape[2] ** 0.5) if softmax_scale is None else softmax_scale
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
        sparsity_n,
        sparsity_m,
        dense_sink_tokens,
        dense_recent_tokens,
        skip_softmax_threshold,
        measure_sparsity,
        p_qdq_mode,
        p_qdq_scale,
        k_cache,
        v_cache,
        block_table,
        page_size,
    )


__all__ = ["LOG2E", "_apply_mask", "attention"]
