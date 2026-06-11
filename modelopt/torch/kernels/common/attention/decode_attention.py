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
    NVFP4_Q: tl.constexpr,  # fakequant Q -> NVFP4 before BMM1
    NVFP4_K: tl.constexpr,  # fakequant K -> NVFP4 before BMM1 (= NVFP4 KV$ K side)
    NVFP4_P: tl.constexpr,  # fakequant softmax P -> NVFP4 before BMM2
    NVFP4_V: tl.constexpr,  # fakequant V -> NVFP4 before BMM2 (= NVFP4 KV$ V side)
    DIFF_QUANT: tl.constexpr,  # softmax-datapath modes: DIFF (pre-exp2), EXP2, ACC (sum)
    EXP2_QUANT: tl.constexpr,
    ACC_QUANT: tl.constexpr,
    MIXED_FP16: tl.constexpr,  # reference mixed-precision softmax (native fp16 MUFU)
    K_CACHE_QUANTIZED: tl.constexpr,  # K cache pre-quantized (skip in-kernel K FQ entirely)
    V_CACHE_QUANTIZED: tl.constexpr,  # V cache pre-quantized for COMPLETE 16-key blocks (FQ only tail)
    # Per-tensor NVFP4 global scales (amax/(6*448)) as device 0-d tensors (pointers);
    # read via tl.load so no host .item() sync is needed (CUDA-graph-safe).
    q_global_scale,
    k_global_scale,
    p_global_scale,
    v_global_scale,
):
    """One (request, head, KV split): partial GEMV attention with skip + optional NVFP4."""
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    split_idx = tl.program_id(2)
    kv_head_idx = head_idx // kv_group_num

    seq_len_kv = tl.load(B_seq_len_k + batch_idx)

    # Quantize-on-write: K is fully fake-quantized in the cache and V is fake-quantized for
    # every COMPLETE 16-key NVFP4 block; only the in-progress tail block (the last
    # ``seq_len_kv % 16`` keys) is still raw and must be fakequantized in-kernel.
    v_dense_boundary = (seq_len_kv // 16) * 16

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

    # NVFP4 global scales: read on-device once (no host sync); only dereferenced when
    # the matching operand is quantized, else an unused scalar.
    q_gs = tl.load(q_global_scale) if NVFP4_Q else 1.0
    k_gs = tl.load(k_global_scale) if NVFP4_K else 1.0
    p_gs = tl.load(p_global_scale) if NVFP4_P else 1.0
    v_gs = tl.load(v_global_scale) if NVFP4_V else 1.0

    # Single query vector [BLOCK_D] for this (request, head); stays in registers.
    # Upcast to fp32 so the QK dot product accumulates in fp32 (matches torch matmul).
    q = tl.load(
        Q + batch_idx * stride_qb + head_idx * stride_qh + dim_pos, mask=d_mask, other=0.0
    ).to(tl.float32)
    if NVFP4_Q:  # BMM1 query (A-side, single row, GEMM-K = head dim)
        q = tl.reshape(
            fake_quant_fp4_k1(tl.reshape(q, (1, BLOCK_D)), 1, BLOCK_D, 16, q_gs, QUANT_NVFP4),
            (BLOCK_D,),
        )

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
        kt_f = kt.to(tl.float32)
        if NVFP4_K and not K_CACHE_QUANTIZED:  # pre-quantized K read as-is (every key's
            # head-dim block is complete at write time, so the whole cache is already on-grid)
            kt_f = fake_quant_fp4_k0(kt_f, BLOCK_D, BLOCK_N, 16, 1, k_gs, QUANT_NVFP4)
        scores = tl.sum(q[:, None] * kt_f, axis=0) * qk_scale  # [BLOCK_N], fp32 accum
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
            if MIXED_FP16:
                # Reference mixed-precision softmax: native fp16 MUFU exp + fp32 unrounded sum.
                p = ex2_fp16(scores - m_new)  # FHADD2 -> fp16 ; MUFU.ex2.fp16 -> P fp16
                p = tl.where(kv_valid, p, 0.0)
                correction = ex2_fp16(m_i - m_new)  # fp16 correction factor
                l_i = l_i * correction + tl.sum(p, axis=0)  # row_sum in fp32 (unrounded)
            else:
                s_shift = softmax_round(scores - m_new, DIFF_QUANT)  # DIFF: input to exp2
                p = softmax_round(tl.math.exp2(s_shift), EXP2_QUANT)  # EXP2: output of exp2
                p = tl.where(kv_valid, p, 0.0)
                correction = tl.math.exp2(m_i - m_new)
                l_i = l_i * correction + softmax_round(tl.sum(p, axis=0), ACC_QUANT)  # ACC: sum
            acc = acc * correction
            # P operand of BMM2 -> NVFP4. NVFP4 is homogeneous (NVFP4(c*x)=c*NVFP4(x)),
            # so quantizing the unnormalized exp here equals quantizing normalized P,
            # since the final acc is divided by l_i in the combine kernel.
            p_bmm = (
                tl.reshape(
                    fake_quant_fp4_k1(
                        tl.reshape(p, (1, BLOCK_N)), 1, BLOCK_N, 16, p_gs, QUANT_NVFP4
                    ),
                    (BLOCK_N,),
                )
                if NVFP4_P
                else p
            )
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
            vt_f = vt.to(tl.float32)
            if NVFP4_V:  # BMM2 value (B-side, V [BLOCK_N, BLOCK_D], GEMM-K = keys = axis 0)
                # Pre-quantized completed V blocks read as-is; only tiles reaching the raw tail
                # (kv_start+BLOCK_N > boundary) are fakequantized. Without on-write, FQ every tile.
                if (not V_CACHE_QUANTIZED) or (kv_start + BLOCK_N > v_dense_boundary):
                    vt_f = fake_quant_fp4_k0(vt_f, BLOCK_N, BLOCK_D, 16, 1, v_gs, QUANT_NVFP4)
            acc += tl.sum(p_bmm[:, None] * vt_f, axis=0)  # [BLOCK_D], fp32 accum
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
    nvfp4: set[str] | None = None,
    fp16_softmax: bool = False,
    softmax_quant: dict | None = None,
    attn_global_scales: dict | None = None,
    k_cache_quantized: bool = False,  # K cache holds fake-quantized values (skip in-kernel K FQ)
    v_cache_quantized: bool = False,  # V cache pre-quantized for complete 16-key blocks (FQ tail only)
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
        nvfp4: subset of ``{"q", "k", "p", "v"}`` whose BMM operands are fake-quantized
            to NVFP4 (E2M1) before the dot. ``{"k", "v"}`` == NVFP4 KV$; the full set
            == NVFP4 BMM1+BMM2. ``None`` disables (exact). Accuracy sim only — the dot
            still runs in fp32.
        fp16_softmax: engage the reference mixed-precision softmax — both exp2s (the softmax exp
            and the online correction) run in native fp16 (``ex2.approx.ftz.f16``) with the
            denominator accumulated in fp32 (P fp16-in, sum unrounded). The matmul accumulators
            stay fp32. Ignored when ``softmax_quant`` is given (that selects per-point round-based
            datapath quant instead).
        softmax_quant: optional per-point datapath quant ``{"diff"/"exp2"/"acc": mode}`` for the
            round-based experiments (fp8/bf16/fp16-round); takes precedence over ``fp16_softmax``.

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

    nvfp4 = nvfp4 or set()
    assert nvfp4 <= {"q", "k", "p", "v"}, f"nvfp4 must be a subset of q/k/p/v, got {nvfp4}"
    assert BLOCK_D % 16 == 0 and BLOCK_N % 16 == 0, "NVFP4 needs dims divisible by 16"
    # Per-tensor NVFP4 global scales (amax/(6*448)) as 0-d device tensors, read in-kernel
    # via tl.load (no host .item()). Paged serving passes precomputed scales (from the
    # small per-step q/k/v); scanning the full paged K/V cache here would OOM.
    _gs = attn_global_scales or {}
    q_gs = (_gs["q"] if "q" in _gs else tensor_global_scale_device(q)) if "q" in nvfp4 else 1.0
    k_gs = (
        (_gs["k"] if "k" in _gs else tensor_global_scale_device(k_cache)) if "k" in nvfp4 else 1.0
    )
    v_gs = (
        (_gs["v"] if "v" in _gs else tensor_global_scale_device(v_cache)) if "v" in nvfp4 else 1.0
    )
    # P global is a fixed constant (unnormalized exp P max ~1); 0-d device tensor for
    # uniformity. Built copy-free with new_full (device-side fill) — torch.tensor(const,
    # device=cuda) does a host->device copy that is illegal during CUDA-graph capture.
    p_gs = q.new_full((), 1.0 / (6.0 * 448.0) + 1e-30, dtype=torch.float32) if "p" in nvfp4 else 1.0
    # Softmax-datapath modes. ``fp16_softmax`` (with no per-point override) engages the reference
    # mixed-precision design (native fp16 MUFU exp + fp32 unrounded sum); a ``softmax_quant`` dict
    # instead selects per-point round-based datapath quant and takes precedence.
    _sq = softmax_quant or {}
    _mixed_fp16 = bool(fp16_softmax) and not _sq
    _sm_default = "fp16_rne" if fp16_softmax else None
    diff_q = resolve_softmax_mode(_sq.get("diff", _sm_default))
    exp2_q = resolve_softmax_mode(_sq.get("exp2", _sm_default))
    acc_q = resolve_softmax_mode(_sq.get("acc", _sm_default))

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
            NVFP4_Q="q" in nvfp4,
            NVFP4_K="k" in nvfp4,
            NVFP4_P="p" in nvfp4,
            NVFP4_V="v" in nvfp4,
            DIFF_QUANT=diff_q,
            EXP2_QUANT=exp2_q,
            ACC_QUANT=acc_q,
            MIXED_FP16=_mixed_fp16,
            K_CACHE_QUANTIZED=k_cache_quantized,
            V_CACHE_QUANTIZED=v_cache_quantized,
            q_global_scale=q_gs,
            k_global_scale=k_gs,
            p_global_scale=p_gs,
            v_global_scale=v_gs,
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


@triton.jit
def _prequant_kv_kernel(
    K,  # flat [total_keys, n_kv_heads, head_dim] (head_dim contiguous)
    V,
    seq_len,  # per-request KV length (uniform across requests here)
    stride_req,  # element stride between consecutive requests' key blocks
    stride_key,
    stride_head,
    k_global_scale,
    v_global_scale,
    NVFP4_K: tl.constexpr,
    NVFP4_V: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    """Fake-quantize a (contiguous) paged KV cache in place = the quantize-on-write op.

    K: every key's head-dim 16-block is complete at write time → fakequant ALL keys.
    V: NVFP4 blocks span 16 keys → fakequant only tiles whose keys are all in COMPLETE blocks
    (``kv_start + BLOCK_N <= (seq_len // 16) * 16``); the in-progress tail tile is left raw for the
    decode kernel to fakequant on read. This partitions the cache so the decode kernel with
    ``KV_CACHE_QUANTIZED=True`` reproduces fakequant-on-read exactly (same op, same global scale).
    """
    r = tl.program_id(0)
    h = tl.program_id(1)
    t = tl.program_id(2)
    kv_start = t * BLOCK_N
    key_off = tl.arange(0, BLOCK_N)
    dim_off = tl.arange(0, BLOCK_D)
    d_mask = dim_off < HEAD_DIM
    key_valid = (kv_start + key_off) < seq_len
    mask = key_valid[:, None] & d_mask[None, :]
    base = r * stride_req + kv_start * stride_key + h * stride_head
    ptr = base + key_off[:, None] * stride_key + dim_off[None, :]
    if NVFP4_K:
        k_gs = tl.load(k_global_scale)
        kt = tl.load(K + ptr, mask=mask, other=0.0).to(tl.float32)
        kq = fake_quant_fp4_k1(
            kt, BLOCK_N, BLOCK_D, 16, k_gs, QUANT_NVFP4
        )  # per-key head-dim blocks
        tl.store(K + ptr, kq.to(K.dtype.element_ty), mask=mask)
    if NVFP4_V:
        v_dense_boundary = (seq_len // 16) * 16
        if kv_start + BLOCK_N <= v_dense_boundary:  # whole tile is complete 16-key blocks
            v_gs = tl.load(v_global_scale)
            vt = tl.load(V + ptr, mask=mask, other=0.0).to(tl.float32)
            vq = fake_quant_fp4_k0(vt, BLOCK_N, BLOCK_D, 16, 1, v_gs, QUANT_NVFP4)
            tl.store(V + ptr, vq.to(V.dtype.element_ty), mask=mask)


def fake_quant_kv_cache(
    k_cache: torch.Tensor,  # [num_blocks, page_size, n_kv_heads, head_dim]
    v_cache: torch.Tensor,
    seq_len: int,  # uniform per-request KV length
    num_requests: int,
    blocks_per_req: int,
    *,
    page_size: int = 16,
    k_global_scale=None,
    v_global_scale=None,
    nvfp4: set[str] | None = None,
) -> None:
    """In-place quantize-on-write over a *contiguous* paged cache (block_table = arange).

    The serving path performs this incrementally per token; this batch form is the reference /
    test entrypoint. K is fully fake-quantized; V only for complete 16-key blocks (tail left raw).
    """
    nvfp4 = nvfp4 or set()
    if not ({"k", "v"} & nvfp4):
        return
    nb, ps, nkv, hd = k_cache.shape
    assert ps == page_size
    kf = k_cache.view(nb * ps, nkv, hd)
    vf = v_cache.view(nb * ps, nkv, hd)
    BLOCK_D = triton.next_power_of_2(hd)
    BLOCK_N = 128
    stride_req = blocks_per_req * ps * kf.stride(0)
    n_tiles = (seq_len + BLOCK_N - 1) // BLOCK_N
    kgs = k_global_scale if k_global_scale is not None else kf.new_full((), 1.0)
    vgs = v_global_scale if v_global_scale is not None else kf.new_full((), 1.0)
    with torch.cuda.device(kf.device):
        _prequant_kv_kernel[(num_requests, nkv, n_tiles)](
            kf,
            vf,
            seq_len,
            stride_req,
            kf.stride(0),
            kf.stride(1),
            kgs,
            vgs,
            NVFP4_K="k" in nvfp4,
            NVFP4_V="v" in nvfp4,
            BLOCK_N=BLOCK_N,
            BLOCK_D=BLOCK_D,
            HEAD_DIM=hd,
            num_warps=4,
        )


@triton.jit
def _onwrite_fq_paged_kernel(
    K_cache,
    V_cache,
    Block_table,
    K_lo,  # [batch] per-request K quant range [lo, hi) in logical key positions
    K_hi,
    V_lo,  # [batch] per-request V quant range (block-aligned, complete 16-key blocks)
    V_hi,
    stride_kc_block,
    stride_kc_pos,
    stride_kc_head,
    stride_vc_block,
    stride_vc_pos,
    stride_vc_head,
    k_global_scale,
    v_global_scale,
    NVFP4_K: tl.constexpr,
    NVFP4_V: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    max_blocks_per_seq,
):
    """In-place fake-quant of a paged KV cache over per-request key ranges (quantize-on-write).

    K range = newly-written keys (head-dim 16-blocks, per key). V range = newly-COMPLETE 16-key
    blocks (keys 16-blocks). Reads the tile, fakequants with the SAME ``fake_quant_fp4_k0`` the
    decode kernel would use, and stores back only the in-range valid keys via the block table.

    The tile each program touches is derived *per request* from on-device bounds
    (``base_tile = min(k_lo, v_lo) // BLOCK_N``, then ``+ program_id(2)``), so the launch grid
    never depends on a host ``.item()``. A pure-decode step appends one key whose tile also holds
    the at-most-one just-completed 128-block, so a fixed grid of ``(batch, n_kv, 1)`` covers it and
    is CUDA-graph-capturable.
    """
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    klo = tl.load(K_lo + batch_idx)
    khi = tl.load(K_hi + batch_idx)
    vlo = tl.load(V_lo + batch_idx)
    vhi = tl.load(V_hi + batch_idx)
    base_tile = tl.minimum(klo, vlo) // BLOCK_N
    tile = base_tile + tl.program_id(2)
    kv_start = tile * BLOCK_N
    kv_pos = tl.arange(0, BLOCK_N)
    dim_pos = tl.arange(0, BLOCK_D)
    d_mask = dim_pos < HEAD_DIM
    kv_abs = kv_start + kv_pos

    page_local = kv_abs // PAGE_SIZE
    offset_in_page = kv_abs % PAGE_SIZE
    in_any = ((kv_abs >= klo) & (kv_abs < khi)) | ((kv_abs >= vlo) & (kv_abs < vhi))
    page_global = tl.load(
        Block_table + batch_idx * max_blocks_per_seq + page_local, mask=in_any, other=0
    )

    if NVFP4_K:
        k_gs = tl.load(k_global_scale)
        k_ptrs = (
            page_global[None, :].to(tl.int64) * stride_kc_block
            + offset_in_page[None, :] * stride_kc_pos
            + head_idx * stride_kc_head
            + dim_pos[:, None]
        )
        k_in = (kv_abs >= klo) & (kv_abs < khi)
        kt = tl.load(K_cache + k_ptrs, mask=k_in[None, :] & d_mask[:, None], other=0.0).to(
            tl.float32
        )
        kq = fake_quant_fp4_k0(kt, BLOCK_D, BLOCK_N, 16, 1, k_gs, QUANT_NVFP4)
        tl.store(
            K_cache + k_ptrs, kq.to(K_cache.dtype.element_ty), mask=k_in[None, :] & d_mask[:, None]
        )

    if NVFP4_V:
        v_gs = tl.load(v_global_scale)
        v_ptrs = (
            page_global[:, None].to(tl.int64) * stride_vc_block
            + offset_in_page[:, None] * stride_vc_pos
            + head_idx * stride_vc_head
            + dim_pos[None, :]
        )
        v_in = (kv_abs >= vlo) & (kv_abs < vhi)
        vt = tl.load(V_cache + v_ptrs, mask=v_in[:, None] & d_mask[None, :], other=0.0).to(
            tl.float32
        )
        vq = fake_quant_fp4_k0(vt, BLOCK_N, BLOCK_D, 16, 1, v_gs, QUANT_NVFP4)
        tl.store(
            V_cache + v_ptrs, vq.to(V_cache.dtype.element_ty), mask=v_in[:, None] & d_mask[None, :]
        )


def fake_quant_kv_onwrite(
    k_cache,  # [num_blocks, page_size, n_kv_heads, head_dim]
    v_cache,
    block_table,  # [batch, max_blocks_per_seq]
    k_lo,  # [batch] int32 per-request K range [lo, hi)
    k_hi,
    v_lo,  # [batch] int32 per-request V range (16-block-aligned)
    v_hi,
    *,
    page_size: int = 16,
    k_global_scale=None,
    v_global_scale=None,
    nvfp4: set[str] | None = None,
    decode: bool = False,
) -> None:
    """Paged quantize-on-write: fakequant K (k_lo:k_hi) and V (v_lo:v_hi) per request in place.

    Set ``decode=True`` for a pure single-token decode step: the grid is fixed to one tile per
    request, so the launch needs no host ``.item()`` and is CUDA-graph-capturable. ``decode=False``
    (prefill / mixed batch, run eager) sizes the grid to the largest per-request span.
    """
    nvfp4 = nvfp4 or set()
    if not ({"k", "v"} & nvfp4):
        return
    batch, max_blocks = block_table.shape
    nkv, hd = k_cache.shape[2], k_cache.shape[3]
    BLOCK_D = triton.next_power_of_2(hd)
    BLOCK_N = 128
    if decode:
        # One appended key per request; its tile also holds the at-most-one just-completed
        # 128-block, so a single per-request tile suffices. Fixed grid => graph-safe (no sync).
        n_tiles = 1
    else:
        # Eager prefill/mixed: span the largest per-request range (base = min(k_lo,v_lo)//BLOCK_N).
        base = torch.minimum(k_lo, v_lo) // BLOCK_N
        span = int((torch.maximum(k_hi, v_hi) - base * BLOCK_N).max().item())
        if span <= 0:
            return
        n_tiles = (span + BLOCK_N - 1) // BLOCK_N
    kgs = k_global_scale if k_global_scale is not None else k_cache.new_full((), 1.0)
    vgs = v_global_scale if v_global_scale is not None else k_cache.new_full((), 1.0)
    with torch.cuda.device(k_cache.device):
        _onwrite_fq_paged_kernel[(batch, nkv, n_tiles)](
            k_cache,
            v_cache,
            block_table,
            k_lo,
            k_hi,
            v_lo,
            v_hi,
            k_cache.stride(0),
            k_cache.stride(1),
            k_cache.stride(2),
            v_cache.stride(0),
            v_cache.stride(1),
            v_cache.stride(2),
            kgs,
            vgs,
            NVFP4_K="k" in nvfp4,
            NVFP4_V="v" in nvfp4,
            PAGE_SIZE=page_size,
            BLOCK_N=BLOCK_N,
            BLOCK_D=BLOCK_D,
            HEAD_DIM=hd,
            max_blocks_per_seq=max_blocks,
            num_warps=4,
        )


__all__ = ["attention_decode", "fake_quant_kv_cache", "fake_quant_kv_onwrite"]
