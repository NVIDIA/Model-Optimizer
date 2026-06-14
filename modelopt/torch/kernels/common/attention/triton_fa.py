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

from modelopt.torch.kernels.quantization.attention.nvfp4_fakequant import (
    QUANT_NVFP4,
    fake_quant_fp4_k0,
    fake_quant_fp4_k1,
    tensor_global_scale_device,
)
from modelopt.torch.kernels.quantization.attention.softmax_fakequant import (
    ex2_fp16,
    resolve_softmax_mode,
    softmax_round,
)


def _resolve_softmax_modes(fp16_softmax, softmax_quant):
    """(fp16_softmax bool, per-point dict) -> (DIFF, EXP2, ACC, MIXED_FP16).

    ``fp16_softmax=True`` engages the reference mixed-precision softmax design: both exp2s (the
    softmax exp and the online correction) run in native fp16 (``ex2.approx.ftz.f16``) and the
    denominator accumulates the fp16 P in fp32 (unrounded). The optional ``softmax_quant`` dict
    ({"diff"/"exp2"/"acc": mode}) instead selects per-point round-based datapath quant (the
    fp8/bf16/fp16-round experiments) and takes precedence when given.
    """
    sq = softmax_quant or {}
    if fp16_softmax and not sq:
        return (0, 0, 0, True)
    default = "fp16_rne" if fp16_softmax else None
    return (
        resolve_softmax_mode(sq.get("diff", default)),
        resolve_softmax_mode(sq.get("exp2", default)),
        resolve_softmax_mode(sq.get("acc", default)),
        False,
    )


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
    NVFP4_Q: tl.constexpr = False,  # fakequant Q -> NVFP4 before BMM1
    NVFP4_K: tl.constexpr = False,  # fakequant K -> NVFP4 before BMM1 (= NVFP4 KV$ K side)
    NVFP4_P: tl.constexpr = False,  # fakequant softmax P -> NVFP4 before BMM2
    NVFP4_V: tl.constexpr = False,  # fakequant V -> NVFP4 before BMM2 (= NVFP4 KV$ V side)
    DIFF_QUANT: tl.constexpr = 0,  # softmax-datapath modes: DIFF (pre-exp2), EXP2, ACC (sum)
    EXP2_QUANT: tl.constexpr = 0,
    ACC_QUANT: tl.constexpr = 0,
    MIXED_FP16: tl.constexpr = False,  # reference mixed-precision softmax (native fp16 MUFU)
    PER_PAGE_SCALE: tl.constexpr = False,  # method 2: derive K/V global scale per tile from its own
    # amax (overrides the passed k/v_global_scale); complete tiles read as-is (baked), tail FQ'd.
    SCALE_PAGE: tl.constexpr = 128,  # per-page granularity = the on-write bake tile (128). The page
    # boundary uses THIS, not BLOCK_N, so a smaller autotuned BLOCK_N still reads baked 128-pages
    # correctly. (The trailing-page amax is then over BLOCK_N, a finer-but-consistent sub-page scale.)
    # Per-tensor NVFP4 global scales (amax/(6*448)). When the matching NVFP4_* flag is
    # set these are device 0-d tensors (pointers), read via ``tl.load`` so no host
    # ``.item()`` sync is needed (CUDA-graph-safe); otherwise an unused float default.
    q_global_scale=1.0,
    k_global_scale=1.0,
    p_global_scale=1.0,
    v_global_scale=1.0,
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
    # Per-page: tiles ending at/below this boundary are complete pages (baked on write -> read
    # as-is); only the trailing in-progress page is fakequantized on read, from its own amax. The
    # boundary is the 128-key bake page (SCALE_PAGE), independent of the autotuned BLOCK_N.
    page_boundary = (seq_len_kv // SCALE_PAGE) * SCALE_PAGE

    if tile_q * BLOCK_M >= seq_len_q:
        return  # This Q tile is past the sequence end

    # --- Tile position indices ---
    q_pos = tile_q * BLOCK_M + tl.arange(0, BLOCK_M)  # Absolute Q token positions
    kv_pos = tl.arange(0, BLOCK_N)  # Relative KV positions within a tile
    dim_pos = tl.arange(0, BLOCK_D)  # Head dimension positions
    d_mask = dim_pos < HEAD_DIM  # Mask for non-power-of-2 head dims

    # --- NVFP4 global scales: read on-device (no host sync) once, reuse in the KV loop.
    # The matching constexpr guard means the arg is only dereferenced when that operand
    # is actually quantized; otherwise it is an unused scalar default.
    q_gs = tl.load(q_global_scale) if NVFP4_Q else 1.0
    k_gs = tl.load(k_global_scale) if NVFP4_K else 1.0
    p_gs = tl.load(p_global_scale) if NVFP4_P else 1.0
    v_gs = tl.load(v_global_scale) if NVFP4_V else 1.0

    # --- Load Q tile [BLOCK_M, BLOCK_D]: stays in SRAM for the entire KV loop ---
    q_ptrs = (q_offset + q_pos[:, None]) * stride_qbs + head_idx * stride_qh + dim_pos[None, :]
    q = tl.load(Q + q_ptrs, mask=(q_pos[:, None] < seq_len_q) & d_mask[None, :], other=0.0)
    if NVFP4_Q:  # BMM1 query operand (A-side, GEMM-K = head dim = axis 1)
        q = fake_quant_fp4_k1(q, BLOCK_M, BLOCK_D, 16, q_gs, QUANT_NVFP4)

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

        if NVFP4_K and (  # BMM1 key operand (B-side, K^T [BLOCK_D, BLOCK_N], GEMM-K = axis 0)
            (not PER_PAGE_SCALE) or (not IS_PAGED) or (kv_start + BLOCK_N > page_boundary)
        ):
            # Per-page: complete tiles are baked (read as-is); only the trailing tile is FQ'd,
            # with its own per-tile amax. Frozen: every tile FQ'd with the per-tensor k_gs.
            kgs_eff = (tl.max(tl.abs(k)) / (6.0 * 448.0) + 1e-30) if PER_PAGE_SCALE else k_gs
            k = fake_quant_fp4_k0(k, BLOCK_D, BLOCK_N, 16, 1, kgs_eff, QUANT_NVFP4)

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
            # --- Online softmax update (with optional mixed-precision datapath quant) ---
            m_new = tl.maximum(row_max, tl.max(scores, 1))
            if MIXED_FP16:
                # Reference mixed-precision softmax: fp16 datapath, fp32 reductions. The exp2
                # input (scores - max) and the correction delta are converted to fp16 inside
                # ex2_fp16 (cvt.rn.f16.f32) and exponentiated by the native fp16 MUFU; P comes
                # out fp16-valued, and the denominator sums it in fp32 (unrounded).
                p = ex2_fp16(scores - m_new[:, None])  # FHADD2 -> fp16 ; MUFU.ex2.fp16 -> P fp16
                l_new = tl.sum(p, 1)  # row_sum accumulates fp16 P in fp32 (unrounded)
                correction = ex2_fp16(row_max - m_new)  # fp16 correction factor
            else:
                s_shift = softmax_round(scores - m_new[:, None], DIFF_QUANT)  # DIFF: input to exp2
                p = softmax_round(tl.math.exp2(s_shift), EXP2_QUANT)  # EXP2: output of exp2
                l_new = softmax_round(tl.sum(p, 1), ACC_QUANT)  # ACC: running softmax denom
                correction = tl.math.exp2(row_max - m_new)
            row_sum = row_sum * correction + l_new
            acc = acc * correction[:, None]

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
            if NVFP4_V and (  # BMM2 value operand (B-side, V [BLOCK_N, BLOCK_D], GEMM-K = axis 0)
                (not PER_PAGE_SCALE) or (not IS_PAGED) or (kv_start + BLOCK_N > page_boundary)
            ):
                # Per-page: complete tiles baked (read as-is); trailing tile FQ'd from its own amax.
                vgs_eff = (tl.max(tl.abs(v)) / (6.0 * 448.0) + 1e-30) if PER_PAGE_SCALE else v_gs
                v = fake_quant_fp4_k0(v, BLOCK_N, BLOCK_D, 16, 1, vgs_eff, QUANT_NVFP4)
            # BMM2 prob operand (A-side, P [BLOCK_M, BLOCK_N], GEMM-K = keys = axis 1) with a
            # per-tensor p_global_scale, matching mni/attnOpt: quantize the *unnormalized* exp
            # P tile, accumulate, normalize acc by row_sum after the loop. (Across KV tiles the
            # online-softmax rescaling makes this an approximation of full-P NVFP4 — same as
            # their flash kernel; the materialized/eager path is the exact reference.)
            p_dot = fake_quant_fp4_k1(p, BLOCK_M, BLOCK_N, 16, p_gs, QUANT_NVFP4) if NVFP4_P else p
            acc = tl.dot(p_dot.to(v.dtype), v, acc)
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
        k_cache,
        v_cache,
        block_table,
        page_size,
        nvfp4=None,
        fp16_softmax=False,
        softmax_quant=None,
        attn_global_scales=None,
        per_page_scale=False,
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
        _nvfp4 = nvfp4 or set()
        assert _nvfp4 <= {"q", "k", "p", "v"}, f"nvfp4 must be a subset of q/k/p/v, got {nvfp4}"
        _diff_q, _exp2_q, _acc_q, _mixed_fp16 = _resolve_softmax_modes(fp16_softmax, softmax_quant)
        # P global is a fixed constant (unnormalized exp P has per-row max ~1); keep it
        # as a 0-d device tensor so every scale enters the kernel uniformly by pointer.
        # Built copy-free with new_full (device-side fill) — torch.tensor(const,
        # device=cuda) does a host->device copy that is illegal during CUDA-graph capture.
        _p_global = q.new_full((), 1.0 / (6.0 * 448.0) + 1e-30, dtype=torch.float32)
        # In paged serving the caller passes precomputed per-tensor global scales
        # (from the small per-step q/k/v); falling back to scanning the full paged
        # K/V cache here would upcast it to fp32 and OOM. Eager/test launches leave
        # ``attn_global_scales`` unset and compute from the (small) q/k/v operands.
        _gs = attn_global_scales or {}
        # K/V NVFP4 global scale: DEFAULT = constant 1.0. The deprecated whole-cache amax scan OOMs
        # on a paged cache and, used as a frozen scale, saturates long context. per_page_scale derives
        # it per tile in-kernel (the passed scale is then ignored). Either way pass this cheap 0-d 1.0.
        _one_gs = q.new_full((), 1.0, dtype=torch.float32)
        fwd_kwargs = {
            # Bucket the autotune key to a power-of-2 length REGIME, not the exact
            # seq len: keying on exact max_input_len re-ran autotune (benchmark all
            # _FWD_CONFIGS) per unique prefill length — ~149 re-tunes on a varying-
            # length workload (random bench / agentic GDPval), the bulk of the TTFT
            # blowup. next_power_of_2 collapses that to one tune per length regime,
            # each still getting a length-appropriate config. (N_CTX is key-only,
            # never used in compute — see the kernel arg comment.)
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
            "NVFP4_Q": "q" in _nvfp4,
            "NVFP4_K": "k" in _nvfp4,
            "NVFP4_P": "p" in _nvfp4,
            "NVFP4_V": "v" in _nvfp4,
            "DIFF_QUANT": _diff_q,
            "EXP2_QUANT": _exp2_q,
            "ACC_QUANT": _acc_q,
            "MIXED_FP16": _mixed_fp16,
            # Device 0-d tensors (pointers) when quantized — read via tl.load in-kernel,
            # no host .item(). The caller's precomputed scales (paged serving) are also
            # device tensors; the fallback computes from the small q/k/v operands.
            "q_global_scale": (_gs["q"] if "q" in _gs else tensor_global_scale_device(q))
            if "q" in _nvfp4
            else 1.0,
            "k_global_scale": (_one_gs if per_page_scale else _gs.get("k", _one_gs))
            if "k" in _nvfp4
            else 1.0,
            "p_global_scale": _p_global if "p" in _nvfp4 else 1.0,
            "v_global_scale": (_one_gs if per_page_scale else _gs.get("v", _one_gs))
            if "v" in _nvfp4
            else 1.0,
            "PER_PAGE_SCALE": per_page_scale,
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
            elif per_page_scale:
                # Per-page bakes 128-key pages; the prefill KV tile MUST equal that page so the
                # trailing page's amax is computed over the FULL 128-page (matching decode/on-write),
                # not a 64-wide sub-tile. Pin BLOCK_N=128; small BLOCK_M + num_stages=1 keeps shared
                # memory in budget (bf16 serve ~84KB on A6000's 99KB; ample on GB300/B200).
                _attn_fwd.fn[grid](
                    *fwd_args, **fwd_kwargs, BLOCK_M=16, BLOCK_N=128, num_warps=4, num_stages=1
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
            None,  # k_cache
            None,  # v_cache
            None,  # block_table
            None,  # page_size
            None,  # nvfp4
            None,  # fp16_softmax
            None,  # softmax_quant
            None,  # attn_global_scales
            None,  # per_page_scale
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
    dense_recent_tokens: int = 128,
    skip_softmax_threshold: float | None = None,
    measure_sparsity: bool = False,
    nvfp4: set[str] | None = None,
    fp16_softmax: bool = False,
    softmax_quant: dict | None = None,
    attn_global_scales: dict | None = None,
    k_cache: torch.Tensor | None = None,
    v_cache: torch.Tensor | None = None,
    block_table: torch.Tensor | None = None,
    page_size: int = 16,
    per_page_scale: bool = False,
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
        k_cache: Paged K cache [num_blocks, page_size, num_kv_heads, head_dim].
            When provided, K/V are read from paged cache via block_table
            instead of from contiguous k/v tensors.
        v_cache: Paged V cache [num_blocks, page_size, num_kv_heads, head_dim].
        block_table: Page table [batch, max_blocks_per_seq] mapping sequence
            block indices to global page IDs.
        page_size: Number of tokens per page in the KV cache.
        fp16_softmax: Engage the reference mixed-precision softmax — both exp2s (the softmax
            exp and the online correction) run in native fp16 (``ex2.approx.ftz.f16``) and the
            denominator accumulates the fp16 P in fp32 (P fp16-in, sum unrounded). The matmul
            accumulators stay fp32. Ignored when ``softmax_quant`` is given.
        softmax_quant: Optional per-point datapath quant ``{"diff"/"exp2"/"acc": mode}`` for the
            round-based experiments (fp8/bf16/fp16-round); takes precedence over ``fp16_softmax``.

    Returns:
        Output tensor [total_q_tokens, num_q_heads, head_dim].

    Note:
        The paged KV path (``k_cache``/``v_cache`` not None) is forward-only —
        ``backward`` raises ``NotImplementedError`` if any of ``q``/``k``/``v``
        require grad, because the saved ``k``/``v`` are dummy tensors in paged
        mode and dK/dV would be silently incorrect.
    """
    _load_sparsity_helpers()
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
        k_cache,
        v_cache,
        block_table,
        page_size,
        nvfp4,
        fp16_softmax,
        softmax_quant,
        attn_global_scales,
        per_page_scale,
    )


__all__ = ["LOG2E", "_apply_mask", "attention"]
