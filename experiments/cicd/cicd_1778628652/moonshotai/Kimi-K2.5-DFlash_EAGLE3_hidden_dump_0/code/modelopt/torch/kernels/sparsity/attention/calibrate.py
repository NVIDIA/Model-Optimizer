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


"""Skip-softmax multi-threshold calibration kernel and Python API.

Runs a full attention forward (identical to dense attention) while measuring
how many KV tiles would be skipped at each candidate threshold. Used by the
sparse-attention calibration workflow in
``modelopt.torch.sparsity.attention_sparsity`` to fit a skip threshold.
"""

import math

import torch
import triton
import triton.language as tl

from modelopt.torch.kernels.common.attention.triton_fa import LOG2E, _apply_mask


# ---------------------------------------------------------------------------
# Calibration kernel: collect multi-threshold skip-softmax sparsity stats
# ---------------------------------------------------------------------------
@triton.jit
def _attn_fwd_calibrate(
    Q,
    K,
    V,
    qk_scale,
    b_start_loc,
    b_seq_len,
    b_start_loc_k,
    b_seq_len_k,
    Out,
    stride_qbs,
    stride_qh,
    stride_kbs,
    stride_kh,
    stride_vbs,
    stride_vh,
    stride_obs,
    stride_oh,
    Threshold_trials,  # [NUM_THRESHOLDS] float32 — pre-scaled to log2 space
    Per_program_totals,  # [num_programs * NUM_THRESHOLDS] int32 — per-program tile counts
    Per_program_skipped,  # [num_programs * NUM_THRESHOLDS] int32 — per-program skip counts
    kv_group_num: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NUM_THRESHOLDS: tl.constexpr,
    PADDED_THRESHOLDS: tl.constexpr,  # next_power_of_2(NUM_THRESHOLDS) for tl.arange
):
    """Forward kernel with multi-threshold sparsity measurement.

    Computes full attention (no skipping) while counting how many KV tiles
    would be skipped at each threshold. Each program writes its local counts
    to ``Per_program_totals`` and ``Per_program_skipped``; the Python wrapper
    sums across programs afterward. This avoids global atomic contention.
    """
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    tile_q = tl.program_id(2)
    kv_head_idx = head_idx // kv_group_num

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

    q_ptrs = (q_offset + q_pos[:, None]) * stride_qbs + head_idx * stride_qh + dim_pos[None, :]
    q = tl.load(Q + q_ptrs, mask=(q_pos[:, None] < seq_len_q) & d_mask[None, :], other=0.0)

    k_base = K + kv_head_idx * stride_kh
    v_base = V + kv_head_idx * stride_vh

    row_max = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    row_sum = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    # Pre-load all thresholds once (vectorized, stays in registers).
    # tl.arange requires power-of-2 size, so use PADDED_THRESHOLDS with masking.
    thresh_offs = tl.arange(0, PADDED_THRESHOLDS)
    thresh_mask = thresh_offs < NUM_THRESHOLDS
    thresholds = tl.load(Threshold_trials + thresh_offs, mask=thresh_mask, other=float("inf"))

    # Per-program local counters: avoid global atomic contention in inner loop.
    # Each program accumulates locally, then writes once to Per_program buffers.
    local_skipped = tl.zeros([PADDED_THRESHOLDS], dtype=tl.int32)
    num_tiles = 0

    kv_bound = seq_len_kv if not IS_CAUSAL else tl.minimum((tile_q + 1) * BLOCK_M, seq_len_kv)

    for kv_start in range(0, kv_bound, BLOCK_N):
        kv_start = tl.multiple_of(kv_start, BLOCK_N)

        k_offs = (kv_offset + kv_start + kv_pos[None, :]) * stride_kbs + dim_pos[:, None]
        k = tl.load(
            k_base + k_offs,
            mask=((kv_start + kv_pos[None, :]) < seq_len_kv) & d_mask[:, None],
            other=0.0,
        )

        scores = tl.dot(q, k) * qk_scale
        scores = _apply_mask(scores, q_pos, kv_pos, seq_len_q, seq_len_kv, kv_start, IS_CAUSAL)

        tile_row_max = tl.max(scores, 1)

        # --- Vectorized multi-threshold sparsity measurement ---
        # A tile is skipped iff ALL Q rows satisfy: tile_row_max < row_max + thresh.
        # Equivalently: max(tile_row_max - row_max) < thresh (worst-case row
        # must still be below threshold for the tile to be skippable).
        max_gap = tl.max(tile_row_max - row_max)  # scalar
        skip_mask = (max_gap < thresholds).to(tl.int32)  # [PADDED_THRESHOLDS]
        local_skipped += skip_mask
        num_tiles += 1

        # --- Always compute full attention (no skipping) ---
        m_new = tl.maximum(row_max, tile_row_max)
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

    # --- Write per-program counters (no atomics, just stores) ---
    # Compute unique flat program index for this (batch, head, q_tile).
    # Use tl.num_programs(2) (grid z dim = cdiv(max_input_len, BLOCK_M)) so the
    # stride matches the wrapper's buffer layout for any batch order. Loading
    # b_seq_len[0] would collide with later batches when batch 0 is shorter.
    num_q_tiles = tl.num_programs(2)
    num_heads = tl.num_programs(1)
    prog_idx = batch_idx * num_heads * num_q_tiles + head_idx * num_q_tiles + tile_q
    base = prog_idx * NUM_THRESHOLDS
    tl.store(
        Per_program_totals + base + thresh_offs,
        tl.full([PADDED_THRESHOLDS], num_tiles, dtype=tl.int32),
        mask=thresh_mask,
    )
    tl.store(
        Per_program_skipped + base + thresh_offs,
        local_skipped,
        mask=thresh_mask,
    )

    acc = acc / tl.maximum(row_sum[:, None], 1e-6)
    o_ptrs = (q_offset + q_pos[:, None]) * stride_obs + head_idx * stride_oh + dim_pos[None, :]
    tl.store(Out + o_ptrs, acc, mask=(q_pos[:, None] < seq_len_q) & d_mask[None, :])


def attention_calibrate(
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
    threshold_trials: list[float] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Flash attention with multi-threshold skip-softmax sparsity measurement.

    Computes full attention (identical output to dense attention) while
    measuring how many KV tiles would be skipped at each threshold in
    ``threshold_trials``. No autograd — forward only.

    Args:
        q, k, v, b_start_loc, b_seq_len, max_input_len, is_causal,
        softmax_scale, b_start_loc_k, b_seq_len_k, max_input_len_k:
            Same as :func:`modelopt.torch.kernels.common.attention.attention`.
        threshold_trials: List of threshold values to measure sparsity for.
            Each value is converted to log2-scaled space for the kernel.

    Returns:
        Tuple of (output, sparsity_counters):
        - output: ``[total_q_tokens, num_q_heads, head_dim]``
        - sparsity_counters: ``[num_thresholds, 2]`` int64 tensor where
          ``[:, 0]`` = total tile evaluations, ``[:, 1]`` = skipped tiles.
          Sparsity per threshold = ``counters[:, 1] / counters[:, 0]``.
    """
    if threshold_trials is None or len(threshold_trials) == 0:
        raise ValueError("threshold_trials must be a non-empty list")

    # Calibration has only been validated with uniform-length batches (current
    # diffusion + RULER paths). Varlen inputs would exercise code paths in the
    # kernel that have not been tested — fail loudly rather than silently
    # produce wrong sparsity counts.
    if b_seq_len.numel() > 1 and not torch.all(b_seq_len == b_seq_len[0]).item():
        raise NotImplementedError(
            "attention_calibrate currently supports only uniform-length batches. "
            f"Got b_seq_len={b_seq_len.tolist()}. Varlen calibration is untested — "
            "validate the kernel against a reference before removing this guard."
        )
    if int(b_seq_len[0].item()) != max_input_len:
        raise ValueError(
            "attention_calibrate expects max_input_len to equal b_seq_len[0] "
            f"(uniform batching). Got max_input_len={max_input_len}, "
            f"b_seq_len[0]={int(b_seq_len[0].item())}."
        )
    if (
        b_seq_len_k is not None
        and b_seq_len_k.data_ptr() != b_seq_len.data_ptr()
        and b_seq_len_k.numel() > 1
        and not torch.all(b_seq_len_k == b_seq_len_k[0]).item()
    ):
        raise NotImplementedError(
            "attention_calibrate currently supports only uniform-length batches. "
            f"Got b_seq_len_k={b_seq_len_k.tolist()}. Varlen calibration is untested."
        )

    HEAD_DIM = q.shape[2]
    num_q_heads = q.shape[1]
    num_kv_heads = k.shape[1]
    kv_group_num = num_q_heads // num_kv_heads
    batch = b_seq_len.shape[0]
    sm_scale = 1.0 / (HEAD_DIM**0.5) if softmax_scale is None else softmax_scale
    qk_scale = sm_scale * LOG2E
    BLOCK_D = triton.next_power_of_2(HEAD_DIM)
    BLOCK_M = 128
    BLOCK_N = 64

    if b_seq_len_k is None:
        b_seq_len_k = b_seq_len
        b_start_loc_k = b_start_loc

    num_thresholds = len(threshold_trials)

    # Convert thresholds to log2-scaled space: log2(lambda) * sm_scale
    threshold_tensor = torch.tensor(
        [math.log2(t) * sm_scale for t in threshold_trials],
        dtype=torch.float32,
        device=q.device,
    )

    o = torch.empty_like(q)

    num_q_tiles = triton.cdiv(max_input_len, BLOCK_M)
    grid = (batch, num_q_heads, num_q_tiles)
    num_programs = batch * num_q_heads * num_q_tiles

    # Per-program output buffers (no atomics needed — each program writes its own row)
    per_program_totals = torch.zeros(
        num_programs * num_thresholds, dtype=torch.int32, device=q.device
    )
    per_program_skipped = torch.zeros(
        num_programs * num_thresholds, dtype=torch.int32, device=q.device
    )

    _attn_fwd_calibrate[grid](
        q,
        k,
        v,
        qk_scale,
        b_start_loc,
        b_seq_len,
        b_start_loc_k,
        b_seq_len_k,
        o,
        q.stride(0),
        q.stride(1),
        k.stride(0),
        k.stride(1),
        v.stride(0),
        v.stride(1),
        o.stride(0),
        o.stride(1),
        threshold_tensor,
        per_program_totals,
        per_program_skipped,
        kv_group_num=kv_group_num,
        BLOCK_M=BLOCK_M,
        BLOCK_D=BLOCK_D,
        BLOCK_N=BLOCK_N,
        IS_CAUSAL=is_causal,
        HEAD_DIM=HEAD_DIM,
        NUM_THRESHOLDS=num_thresholds,
        PADDED_THRESHOLDS=triton.next_power_of_2(num_thresholds),
        num_warps=4,
        num_stages=1,
    )

    # Reduce across programs: sum per-program counts → [num_thresholds]
    totals = per_program_totals.view(num_programs, num_thresholds).sum(dim=0).to(torch.int64)
    skipped = per_program_skipped.view(num_programs, num_thresholds).sum(dim=0).to(torch.int64)
    sparsity_counters = torch.stack([totals, skipped], dim=1)  # [num_thresholds, 2]

    return o, sparsity_counters
