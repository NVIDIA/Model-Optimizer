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

"""Triton decode attention with optional fused skip-softmax over a paged KV cache.

The general var-len flash-attention kernel (``triton_fa._attn_fwd``) is built for
prefill: it tiles queries into ``BLOCK_M`` rows. In decode there is one query
token per request, so 127/128 of every query tile is padding — the kernel does
~128x the needed query-side work. This kernel is decode-shaped: one query vector
per ``(request, query head)``, looping the paged KV cache.

Split-K (flash-decoding): decode batches are small, so a grid of just
``(batch, num_q_heads)`` leaves most SMs idle while each program walks the whole
KV cache serially. We add a third grid axis that partitions the KV sequence into
``num_kv_splits`` contiguous chunks, so a single ``(request, head)`` is computed
by ``num_kv_splits`` programs in parallel; a small second kernel combines their
partial softmaxes with the standard log-sum-exp rescaling (numerically exact).

Skip-softmax: a KV tile whose peak score is negligible versus the running max
(``tile_max - running_max < log(lambda)``) contributes ~0 to the softmax, so its
V load and accumulation are skipped — a bandwidth saving that is largest exactly
in long-context decode. The skip uses the same single-pass *prefix-max* criterion
as ``attention_calibrate`` (per query head), so the realized sparsity matches the
calibrated decode ``(a, b)``.

Skip vs split-K interaction: each split builds its own prefix max from
``-inf``, so the first tile of every split never skips and a split never sees a
dominant max living in an earlier split. Splitting therefore makes skipping
strictly *more conservative* (fewer skips) the more splits there are. Split-K is
the universal small-batch win (exact, parallelism); skip is most effective at low
split counts. ``num_kv_splits`` is exposed so callers can trade between them.
"""

import math

import torch
import triton
import triton.language as tl

from modelopt.torch.kernels.common.attention.triton_fa import (
    LOG2E,
    _load_paged_k_tile,
    _load_paged_v_tile,
)

# Cap on the auto-chosen split count. Decode KV reads dominate, so a handful of
# splits is enough to fill the SMs at small batch; more just fragments skipping.
MAX_KV_SPLITS = 8


@triton.jit
def _attn_decode_split_fwd(
    Q,  # [batch, num_q_heads, head_dim] — one query token per request
    qk_scale,  # softmax_scale * log2(e)
    B_seq_len_k,  # [batch] total KV length per request
    M_partial,  # [batch, num_q_heads, num_kv_splits] per-split running max
    L_partial,  # [batch, num_q_heads, num_kv_splits] per-split softmax denom
    Acc_partial,  # [batch, num_q_heads, num_kv_splits, BLOCK_D] per-split weighted V sum
    stride_qb,
    stride_qh,
    stride_mb,
    stride_mh,
    stride_ab,
    stride_ah,
    stride_as,
    K_cache,  # [num_blocks, page_size, num_kv_heads, head_dim]
    V_cache,
    Block_table,  # [batch, max_blocks_per_seq]
    stride_kc_block,
    stride_kc_pos,
    stride_kc_head,
    stride_vc_block,
    stride_vc_pos,
    stride_vc_head,
    Sparsity_total,  # optional int64 scalar (atomic) — total tiles
    Sparsity_skipped,  # optional int64 scalar (atomic) — skipped tiles
    kv_group_num: tl.constexpr,  # GQA ratio num_q_heads // num_kv_heads
    BLOCK_D: tl.constexpr,  # next_power_of_2(head_dim)
    BLOCK_N: tl.constexpr,  # KV tile size (128 to match the calibration granularity)
    HEAD_DIM: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    max_blocks_per_seq,
    NUM_KV_SPLITS: tl.constexpr,
    APPLY_SKIP: tl.constexpr,
    SKIP_THRESHOLD_LOG2: tl.constexpr,  # log2(lambda) in the scaled-log2 score space
    MEASURE_SPARSITY: tl.constexpr,
):
    """One (request, head, KV split): partial GEMV attention with skip."""
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    split_idx = tl.program_id(2)
    kv_head_idx = head_idx // kv_group_num

    seq_len_kv = tl.load(B_seq_len_k + batch_idx)

    # Partition whole BLOCK_N tiles (calibration-aligned) evenly across splits.
    num_tiles = (seq_len_kv + BLOCK_N - 1) // BLOCK_N
    tiles_per_split = (num_tiles + NUM_KV_SPLITS - 1) // NUM_KV_SPLITS
    tile_lo = split_idx * tiles_per_split
    tile_hi = tl.minimum(tile_lo + tiles_per_split, num_tiles)
    kv_lo = tile_lo * BLOCK_N
    kv_hi = tile_hi * BLOCK_N  # may exceed seq_len_kv; masked by kv_valid below

    dim_pos = tl.arange(0, BLOCK_D)
    d_mask = dim_pos < HEAD_DIM
    kv_pos = tl.arange(0, BLOCK_N)

    # Single query vector [BLOCK_D] for this (request, head); stays in registers.
    # Upcast to fp32 so the QK dot product accumulates in fp32 (matches torch matmul).
    q = tl.load(
        Q + batch_idx * stride_qb + head_idx * stride_qh + dim_pos, mask=d_mask, other=0.0
    ).to(tl.float32)

    m_i = -float("inf")  # running max (prefix, scalar) within this split
    l_i = 0.0  # running softmax denominator (scalar)
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)  # running weighted V sum

    for kv_start in range(kv_lo, kv_hi, BLOCK_N):
        kv_start = tl.multiple_of(kv_start, BLOCK_N)
        kv_valid = (kv_start + kv_pos) < seq_len_kv

        # K^T tile [BLOCK_D, BLOCK_N]; scores[BLOCK_N] = q . K^T (GEMV, M=1).
        kt = _load_paged_k_tile(
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
        scores = tl.sum(q[:, None] * kt.to(tl.float32), axis=0) * qk_scale  # [BLOCK_N], fp32 accum
        scores = tl.where(kv_valid, scores, -float("inf"))

        tile_max = tl.max(scores, axis=0)  # scalar

        skip = False
        if APPLY_SKIP:
            # Same prefix-max criterion as attention_calibrate (single query row).
            skip = tile_max < (m_i + SKIP_THRESHOLD_LOG2)
        if MEASURE_SPARSITY:
            tl.atomic_add(Sparsity_total, 1)
            if skip:
                tl.atomic_add(Sparsity_skipped, 1)

        if not skip:
            m_new = tl.maximum(m_i, tile_max)
            p = tl.math.exp2(scores - m_new)  # [BLOCK_N]
            p = tl.where(kv_valid, p, 0.0)
            correction = tl.math.exp2(m_i - m_new)
            l_i = l_i * correction + tl.sum(p, axis=0)
            acc = acc * correction
            vt = _load_paged_v_tile(
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
            acc += tl.sum(p[:, None] * vt.to(tl.float32), axis=0)  # [BLOCK_D], fp32 accum
            m_i = m_new

    # Store this split's partial softmax state (undivided acc + max + denom).
    off_ml = batch_idx * stride_mb + head_idx * stride_mh + split_idx
    tl.store(M_partial + off_ml, m_i)
    tl.store(L_partial + off_ml, l_i)
    off_a = batch_idx * stride_ab + head_idx * stride_ah + split_idx * stride_as + dim_pos
    tl.store(Acc_partial + off_a, acc, mask=d_mask)


@triton.jit
def _attn_decode_combine(
    M_partial,  # [batch, num_q_heads, num_kv_splits]
    L_partial,
    Acc_partial,  # [batch, num_q_heads, num_kv_splits, BLOCK_D]
    Out,  # [batch, num_q_heads, head_dim]
    stride_mb,
    stride_mh,
    stride_ab,
    stride_ah,
    stride_as,
    stride_ob,
    stride_oh,
    BLOCK_D: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NUM_KV_SPLITS: tl.constexpr,
):
    """Merge per-split partial softmaxes into the final output (exact)."""
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    dim_pos = tl.arange(0, BLOCK_D)
    d_mask = dim_pos < HEAD_DIM

    m = -float("inf")  # global running max across splits
    l_acc = 0.0  # global softmax denominator
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)

    base_ml = batch_idx * stride_mb + head_idx * stride_mh
    base_a = batch_idx * stride_ab + head_idx * stride_ah
    for s in range(NUM_KV_SPLITS):
        l_s = tl.load(L_partial + base_ml + s)
        if l_s > 0.0:  # skip empty splits (l == 0 -> contributed nothing)
            m_s = tl.load(M_partial + base_ml + s)
            acc_s = tl.load(Acc_partial + base_a + s * stride_as + dim_pos, mask=d_mask, other=0.0)
            m_new = tl.maximum(m, m_s)
            scale = tl.math.exp2(m - m_new)  # rescale the running totals
            scale_s = tl.math.exp2(m_s - m_new)  # rescale this split
            acc = acc * scale + acc_s * scale_s
            l_acc = l_acc * scale + l_s * scale_s
            m = m_new

    out = acc / tl.maximum(l_acc, 1e-6)  # 0/eps = 0 if every tile skipped
    tl.store(Out + batch_idx * stride_ob + head_idx * stride_oh + dim_pos, out, mask=d_mask)


def _auto_num_kv_splits(device: torch.device, num_programs: int) -> int:
    """Pick a split count that roughly fills the SMs without over-fragmenting."""
    num_sms = torch.cuda.get_device_properties(device).multi_processor_count
    # ceil(num_sms / num_programs), clamped to [1, MAX_KV_SPLITS].
    return max(1, min(MAX_KV_SPLITS, -(-num_sms // max(num_programs, 1))))


def attention_decode(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    b_seq_len_k: torch.Tensor,
    *,
    softmax_scale: float | None = None,
    skip_softmax_threshold: float | None = None,
    page_size: int = 16,
    num_kv_splits: int | None = None,
    measure_sparsity: bool = False,
) -> torch.Tensor:
    """Decode attention (one query token per request) over a paged KV cache.

    Args:
        q: ``[batch, num_q_heads, head_dim]`` — the single decode query per request.
        k_cache, v_cache: paged caches ``[num_blocks, page_size, num_kv_heads, head_dim]``.
        block_table: ``[batch, max_blocks_per_seq]`` page table.
        b_seq_len_k: ``[batch]`` total KV length per request (including the new token).
        softmax_scale: scale (default ``1/sqrt(head_dim)``).
        skip_softmax_threshold: BLASST lambda; skip KV tiles whose peak score is
            negligible versus the running max. ``None``/``0`` disables skipping
            (exact dense decode).
        page_size: tokens per page.
        num_kv_splits: split-K factor — how many programs cooperate on one
            ``(request, head)``. ``None`` auto-picks from the SM count and batch.
            More splits raise small-batch occupancy but make skipping more
            conservative (each split restarts its prefix max); pass ``1`` to keep
            skipping maximally effective.
        measure_sparsity: when skipping is active, count total/skipped tiles and
            attach them as ``_sparsity_total`` / ``_sparsity_skipped`` on the output.

    Returns:
        ``[batch, num_q_heads, head_dim]`` attention output.
    """
    assert q.dim() == 3, "decode query must be [batch, num_q_heads, head_dim]"
    q = q.contiguous()
    batch, num_q_heads, head_dim = q.shape
    num_kv_heads = k_cache.shape[2]
    kv_group_num = num_q_heads // num_kv_heads

    sm_scale = 1.0 / (head_dim**0.5) if softmax_scale is None else softmax_scale
    qk_scale = sm_scale * LOG2E
    BLOCK_D = triton.next_power_of_2(head_dim)
    BLOCK_N = 128  # match attention_calibrate's KV tile granularity

    if skip_softmax_threshold is not None and skip_softmax_threshold > 0.0:
        apply_skip = True
        skip_threshold_log2 = math.log2(skip_softmax_threshold)
    else:
        apply_skip = False
        skip_threshold_log2 = 0.0
    do_measure = measure_sparsity and apply_skip

    if num_kv_splits is None:
        num_kv_splits = _auto_num_kv_splits(q.device, batch * num_q_heads)
    num_kv_splits = max(1, num_kv_splits)

    # Per-split partial softmax state, merged by the combine kernel.
    m_partial = torch.empty(batch, num_q_heads, num_kv_splits, dtype=torch.float32, device=q.device)
    l_partial = torch.zeros(batch, num_q_heads, num_kv_splits, dtype=torch.float32, device=q.device)
    acc_partial = torch.empty(
        batch, num_q_heads, num_kv_splits, BLOCK_D, dtype=torch.float32, device=q.device
    )

    out = torch.empty_like(q)
    if do_measure:
        sparsity_total = torch.zeros(1, dtype=torch.int64, device=q.device)
        sparsity_skipped = torch.zeros(1, dtype=torch.int64, device=q.device)
    else:
        sparsity_total = None
        sparsity_skipped = None

    with torch.cuda.device(q.device):
        _attn_decode_split_fwd[(batch, num_q_heads, num_kv_splits)](
            q,
            qk_scale,
            b_seq_len_k,
            m_partial,
            l_partial,
            acc_partial,
            q.stride(0),
            q.stride(1),
            m_partial.stride(0),
            m_partial.stride(1),
            acc_partial.stride(0),
            acc_partial.stride(1),
            acc_partial.stride(2),
            k_cache,
            v_cache,
            block_table,
            k_cache.stride(0),
            k_cache.stride(1),
            k_cache.stride(2),
            v_cache.stride(0),
            v_cache.stride(1),
            v_cache.stride(2),
            sparsity_total,
            sparsity_skipped,
            kv_group_num=kv_group_num,
            BLOCK_D=BLOCK_D,
            BLOCK_N=BLOCK_N,
            HEAD_DIM=head_dim,
            PAGE_SIZE=page_size,
            max_blocks_per_seq=block_table.shape[1],
            NUM_KV_SPLITS=num_kv_splits,
            APPLY_SKIP=apply_skip,
            SKIP_THRESHOLD_LOG2=skip_threshold_log2,
            MEASURE_SPARSITY=do_measure,
            num_warps=4,
            num_stages=2,
        )
        _attn_decode_combine[(batch, num_q_heads)](
            m_partial,
            l_partial,
            acc_partial,
            out,
            m_partial.stride(0),
            m_partial.stride(1),
            acc_partial.stride(0),
            acc_partial.stride(1),
            acc_partial.stride(2),
            out.stride(0),
            out.stride(1),
            BLOCK_D=BLOCK_D,
            HEAD_DIM=head_dim,
            NUM_KV_SPLITS=num_kv_splits,
            num_warps=4,
        )

    if do_measure:
        out._sparsity_total = sparsity_total.item()
        out._sparsity_skipped = sparsity_skipped.item()
    return out


__all__ = ["attention_decode"]
