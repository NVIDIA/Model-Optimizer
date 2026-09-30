# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


"""Split-K decode attention for the ModelOpt paged NVFP4 serving path.

P QDQ operates on split-local, unnormalized online-softmax probabilities. Its
numerics therefore include the fixed split count as part of the kernel schedule.
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
from modelopt.torch.kernels.quantization.attention.bmm2_qdq import _p_qdq_nvfp4, _v_qdq_nvfp4
from modelopt.torch.kernels.quantization.common.fp8_quant import fp8_scalar_qdq

__all__ = ["attention_decode"]

_BLOCK_N = 128
_DEFAULT_KV_SPLITS = 32
_MAX_KV_SPLITS = 32
_P_QDQ_MODES = {None: 0, "fp8": 1, "nvfp4": 2}
_V_QDQ_MODES = {None, "nvfp4"}


@triton.jit
def _decode_split_kernel(
    Q,
    B_seq_len_k,
    M_partial,
    L_partial,
    Acc_partial,
    K_cache,
    V_cache,
    Block_table,
    qk_scale,
    stride_qb,
    stride_qh,
    stride_mb,
    stride_mh,
    stride_ab,
    stride_ah,
    stride_as,
    stride_kc_block,
    stride_kc_pos,
    stride_kc_head,
    stride_vc_block,
    stride_vc_pos,
    stride_vc_head,
    p_qdq_scale,
    v_qdq_scale,
    kv_group_num: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    max_blocks_per_seq,
    NUM_KV_SPLITS: tl.constexpr,
    P_QDQ: tl.constexpr,
    V_QDQ: tl.constexpr,
    V_CACHE_QUANTIZED: tl.constexpr,
):
    """Compute one partial softmax for one request, query head, and KV split."""
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    split_idx = tl.program_id(2)
    kv_head_idx = head_idx // kv_group_num
    seq_len_kv = tl.load(B_seq_len_k + batch_idx)
    v_quantized_boundary = (seq_len_kv // 16) * 16

    num_tiles = tl.cdiv(seq_len_kv, BLOCK_N)
    tiles_per_split = tl.cdiv(num_tiles, NUM_KV_SPLITS)
    tile_lo = split_idx * tiles_per_split
    tile_hi = tl.minimum(tile_lo + tiles_per_split, num_tiles)
    kv_lo = tile_lo * BLOCK_N
    kv_hi = tile_hi * BLOCK_N

    dim_pos = tl.arange(0, BLOCK_D)
    d_mask = dim_pos < HEAD_DIM
    kv_pos = tl.arange(0, BLOCK_N)
    q = tl.load(
        Q + batch_idx * stride_qb + head_idx * stride_qh + dim_pos,
        mask=d_mask,
        other=0.0,
    ).to(tl.float32)

    running_max = -float("inf")
    running_sum = 0.0
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)

    for kv_start in range(kv_lo, kv_hi, BLOCK_N):
        kv_start = tl.multiple_of(kv_start, BLOCK_N)
        kv_valid = kv_start + kv_pos < seq_len_kv
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
        ).to(tl.float32)
        scores = tl.sum(q[:, None] * k, axis=0) * qk_scale
        scores = tl.where(kv_valid, scores, -float("inf"))
        tile_max = tl.max(scores, axis=0)
        new_max = tl.maximum(running_max, tile_max)
        p = tl.math.exp2(scores - new_max)
        p = tl.where(kv_valid, p, 0.0)
        correction = tl.math.exp2(running_max - new_max)
        running_sum = running_sum * correction + tl.sum(p, axis=0)
        acc *= correction

        if P_QDQ == 1:
            # FP8 E4M3 per-tensor QDQ of the split-local unnormalized P
            # (elementwise -> split-count- and batch-shape-invariant).
            p = fp8_scalar_qdq(p, p_qdq_scale)
        elif P_QDQ == 2:
            p = tl.reshape(
                _p_qdq_nvfp4(
                    tl.reshape(p.to(V_cache.dtype.element_ty).to(tl.float32), (1, BLOCK_N)),
                    p_qdq_scale,
                    1,
                    BLOCK_N,
                ),
                (BLOCK_N,),
            )

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
        ).to(tl.float32)
        if V_QDQ and ((not V_CACHE_QUANTIZED) or kv_start + BLOCK_N > v_quantized_boundary):
            v_qdq = _v_qdq_nvfp4(v, v_qdq_scale, BLOCK_N, BLOCK_D)
            v_qdq = v_qdq.to(V_cache.dtype.element_ty).to(tl.float32)
            if V_CACHE_QUANTIZED:
                use_qdq = kv_start + kv_pos >= v_quantized_boundary
                v = tl.where(use_qdq[:, None], v_qdq, v)
            else:
                v = v_qdq

        acc += tl.sum(p[:, None] * v, axis=0)
        running_max = new_max

    partial_offset = batch_idx * stride_mb + head_idx * stride_mh + split_idx
    tl.store(M_partial + partial_offset, running_max)
    tl.store(L_partial + partial_offset, running_sum)
    acc_offset = batch_idx * stride_ab + head_idx * stride_ah + split_idx * stride_as + dim_pos
    tl.store(Acc_partial + acc_offset, acc, mask=d_mask)


@triton.jit
def _decode_combine_kernel(
    M_partial,
    L_partial,
    Acc_partial,
    Out,
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
    """Merge split-local online-softmax states."""
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    dim_pos = tl.arange(0, BLOCK_D)
    d_mask = dim_pos < HEAD_DIM
    base_ml = batch_idx * stride_mb + head_idx * stride_mh
    base_acc = batch_idx * stride_ab + head_idx * stride_ah

    running_max = -float("inf")
    running_sum = 0.0
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)
    for split_idx in range(NUM_KV_SPLITS):
        split_sum = tl.load(L_partial + base_ml + split_idx)
        if split_sum > 0.0:
            split_max = tl.load(M_partial + base_ml + split_idx)
            split_acc = tl.load(
                Acc_partial + base_acc + split_idx * stride_as + dim_pos,
                mask=d_mask,
                other=0.0,
            )
            new_max = tl.maximum(running_max, split_max)
            correction = tl.math.exp2(running_max - new_max)
            split_correction = tl.math.exp2(split_max - new_max)
            acc = acc * correction + split_acc * split_correction
            running_sum = running_sum * correction + split_sum * split_correction
            running_max = new_max

    output = acc / tl.maximum(running_sum, 1e-6)
    tl.store(
        Out + batch_idx * stride_ob + head_idx * stride_oh + dim_pos,
        output,
        mask=d_mask,
    )


def _qdq_scale(mode: str | None, amax: float | None, operand: str) -> float:
    allowed = _P_QDQ_MODES if operand == "p" else _V_QDQ_MODES
    if mode not in allowed:
        raise ValueError(
            f"{operand}_qdq must be one of {sorted(m for m in allowed if m)} or None, got {mode!r}"
        )
    if mode is None:
        return 1.0
    if amax is None:
        return 1.0
    if not (math.isfinite(amax) and amax > 0.0):
        raise ValueError(f"{operand}_qdq_amax must be finite and positive, got {amax}")
    # FP8 uses the per-tensor convention ``amax / 448``; NVFP4 the two-level
    # global scale ``amax / (6 * 448)`` (matches the prefill kernel).
    return amax / 448.0 if mode == "fp8" else amax / (6.0 * 448.0)


def attention_decode(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    b_seq_len_k: torch.Tensor,
    *,
    softmax_scale: float | None = None,
    page_size: int = 16,
    num_kv_splits: int = _DEFAULT_KV_SPLITS,
    p_qdq: str | None = None,
    p_qdq_amax: float = 1.0,
    v_qdq: str | None = None,
    v_qdq_amax: float | None = None,
    v_cache_quantized: bool = False,
) -> torch.Tensor:
    """Decode one query token per request over a paged KV cache.

    Q and K are expected to be fake-quantized before this call. Dynamic NVFP4
    Q should use an FP32 QDQ carrier; K may remain BF16 when its global scale is
    one. P is rounded to the model/cache dtype before native-style quantization,
    then its QDQ result remains FP32. Complete block-16 V groups may be finalized
    in the cache; only the pristine partial group is then quantized on read.
    P QDQ intentionally follows the split-local online-softmax schedule; changing
    ``num_kv_splits`` can therefore change quantized results.
    """
    if q.ndim != 3:
        raise ValueError(f"q must have shape [batch, heads, head_dim], got {tuple(q.shape)}")
    if page_size != k_cache.shape[1] or page_size != v_cache.shape[1]:
        raise ValueError("page_size must match both paged KV cache tensors")
    if not 1 <= num_kv_splits <= _MAX_KV_SPLITS:
        raise ValueError(f"num_kv_splits must be in [1, {_MAX_KV_SPLITS}], got {num_kv_splits}")
    batch, num_q_heads, head_dim = q.shape
    num_kv_heads = k_cache.shape[2]
    if num_q_heads % num_kv_heads:
        raise ValueError("num_q_heads must be divisible by num_kv_heads")
    if b_seq_len_k.shape != (batch,) or block_table.shape[0] != batch:
        raise ValueError("decode metadata batch dimension must match q")

    p_qdq_scale = _qdq_scale(p_qdq, p_qdq_amax, "p")
    v_qdq_scale = _qdq_scale(v_qdq, v_qdq_amax, "v")
    if v_cache_quantized and v_qdq != "nvfp4":
        raise ValueError("v_cache_quantized requires v_qdq='nvfp4'")
    q = q.contiguous()
    block_d = triton.next_power_of_2(head_dim)
    if (p_qdq == "nvfp4" or v_qdq == "nvfp4") and head_dim % 16:
        raise ValueError("NVFP4 decode requires dimensions divisible by 16")
    qk_scale = (head_dim**-0.5 if softmax_scale is None else softmax_scale) * LOG2E
    m_partial = torch.empty(batch, num_q_heads, num_kv_splits, dtype=torch.float32, device=q.device)
    l_partial = torch.empty_like(m_partial)
    acc_partial = torch.empty(
        batch, num_q_heads, num_kv_splits, block_d, dtype=torch.float32, device=q.device
    )
    output = torch.empty_like(q)

    with torch.cuda.device(q.device):
        _decode_split_kernel[(batch, num_q_heads, num_kv_splits)](
            q,
            b_seq_len_k,
            m_partial,
            l_partial,
            acc_partial,
            k_cache,
            v_cache,
            block_table,
            qk_scale,
            q.stride(0),
            q.stride(1),
            m_partial.stride(0),
            m_partial.stride(1),
            acc_partial.stride(0),
            acc_partial.stride(1),
            acc_partial.stride(2),
            k_cache.stride(0),
            k_cache.stride(1),
            k_cache.stride(2),
            v_cache.stride(0),
            v_cache.stride(1),
            v_cache.stride(2),
            p_qdq_scale,
            v_qdq_scale,
            kv_group_num=num_q_heads // num_kv_heads,
            BLOCK_D=block_d,
            BLOCK_N=_BLOCK_N,
            HEAD_DIM=head_dim,
            PAGE_SIZE=page_size,
            max_blocks_per_seq=block_table.shape[1],
            NUM_KV_SPLITS=num_kv_splits,
            P_QDQ=_P_QDQ_MODES[p_qdq],
            V_QDQ=v_qdq == "nvfp4",
            V_CACHE_QUANTIZED=v_cache_quantized,
            num_warps=4,
            num_stages=2,
        )
        _decode_combine_kernel[(batch, num_q_heads)](
            m_partial,
            l_partial,
            acc_partial,
            output,
            m_partial.stride(0),
            m_partial.stride(1),
            acc_partial.stride(0),
            acc_partial.stride(1),
            acc_partial.stride(2),
            output.stride(0),
            output.stride(1),
            BLOCK_D=block_d,
            HEAD_DIM=head_dim,
            NUM_KV_SPLITS=num_kv_splits,
            num_warps=4,
        )
    return output
