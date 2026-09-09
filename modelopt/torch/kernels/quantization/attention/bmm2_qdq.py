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

"""NVFP4 operand helpers for the attention ``P @ V`` matmul (BMM2).

P and V share the low-level ``nvfp4_scalar_qdq`` primitive, but retain thin
operand-specific wrappers because their layouts and amax reductions differ.
P is nonnegative with layout ``[M, K]``; V is signed with layout ``[K, N]``.
Both use block-16 scaling along the BMM2 contraction axis.
"""

import math

import torch
import triton
import triton.language as tl

from modelopt.torch.kernels.quantization.common.nvfp4_quant import nvfp4_scalar_qdq

__all__ = ["fake_quant_v_onwrite"]

_BLOCK_N = 16


@triton.jit
def _p_qdq_nvfp4(
    p,
    global_scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """NVFP4 fake quant-dequant of softmax probabilities.

    Two-level scaling per the NVFP4 recipe: E2M1 elements with one FP8 E4M3
    scale per 16 contiguous elements along the key dimension (the contraction
    axis of ``P @ V``), and a per-tensor ``global_scale`` (runtime scalar,
    ``amax / (6 * 448)``; ``attention()`` derives it from ``p_qdq_amax``,
    which defaults to 1, the theoretical upper bound of P's amax).

    ``p >= 0``, so the block amax is a plain max (no ``abs``), and
    ``nvfp4_scalar_qdq`` guards the degenerate all-zero blocks of fully
    masked or padded positions.
    """
    tl.static_assert(BLOCK_N % 16 == 0, "BLOCK_N must be divisible by 16 for NVFP4")

    grouped = tl.reshape(p, (BLOCK_M, BLOCK_N // 16, 16))
    block_amax = tl.expand_dims(tl.max(grouped, axis=2), 2)  # p >= 0, so max == amax
    q = nvfp4_scalar_qdq(grouped, block_amax, global_scale, 16)
    return tl.reshape(q, (BLOCK_M, BLOCK_N))


@triton.jit
def _v_qdq_nvfp4(v, global_scale, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr):
    """Fake-quantize signed V in block-16 groups along its key axis."""
    tl.static_assert(BLOCK_N % 16 == 0, "BLOCK_N must be divisible by 16 for NVFP4")
    grouped = tl.reshape(v, (BLOCK_N // 16, 16, BLOCK_D))
    block_amax = tl.expand_dims(tl.max(tl.abs(grouped), axis=1), 1)
    return tl.reshape(nvfp4_scalar_qdq(grouped, block_amax, global_scale, 16), (BLOCK_N, BLOCK_D))


@triton.jit
def _fake_quant_v_onwrite_kernel(
    V_cache,
    Block_table,
    V_lo,
    V_hi,
    stride_vc_block,
    stride_vc_pos,
    stride_vc_head,
    v_qdq_scale,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    max_blocks_per_seq,
):
    """Finalize one block-16 V group for one request and KV head."""
    batch_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)
    v_lo = tl.load(V_lo + batch_idx)
    v_hi = tl.load(V_hi + batch_idx)
    kv_start = (v_lo // BLOCK_N + tl.program_id(2)) * BLOCK_N
    kv_abs = kv_start + tl.arange(0, BLOCK_N)
    dim_pos = tl.arange(0, BLOCK_D)
    mask = (kv_abs >= v_lo) & (kv_abs < v_hi)
    page_local = kv_abs // PAGE_SIZE
    page_offset = kv_abs % PAGE_SIZE
    page = tl.load(Block_table + batch_idx * max_blocks_per_seq + page_local, mask=mask, other=0)
    ptrs = (
        page[:, None].to(tl.int64) * stride_vc_block
        + page_offset[:, None] * stride_vc_pos
        + kv_head_idx * stride_vc_head
        + dim_pos[None, :]
    )
    load_mask = mask[:, None] & (dim_pos[None, :] < HEAD_DIM)
    v = tl.load(V_cache + ptrs, mask=load_mask, other=0.0).to(tl.float32)
    v = _v_qdq_nvfp4(v, v_qdq_scale, BLOCK_N, BLOCK_D)
    tl.store(V_cache + ptrs, v.to(V_cache.dtype.element_ty), mask=load_mask)


def fake_quant_v_onwrite(
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    v_lo: torch.Tensor,
    v_hi: torch.Tensor,
    *,
    max_new_tokens: int,
    page_size: int = 16,
    v_qdq_scale: float = 1.0,
) -> None:
    """NVFP4-finalize complete block-16 groups in ``[v_lo, v_hi)`` in place.

    ``max_new_tokens`` is host metadata used to size the masked launch grid.
    The grid covers every group that the largest query chunk can complete
    without reading device metadata. ``v_lo`` and ``v_hi`` must describe
    aligned, completed block-16 boundaries; their device values are not
    host-validated.
    """
    if page_size != v_cache.shape[1]:
        raise ValueError(f"page_size {page_size} must match v_cache.shape[1] {v_cache.shape[1]}")
    if not (math.isfinite(v_qdq_scale) and v_qdq_scale > 0):
        raise ValueError(f"v_qdq_scale must be finite and positive, got {v_qdq_scale}")
    if max_new_tokens < 1:
        raise ValueError(f"max_new_tokens must be positive, got {max_new_tokens}")

    batch, max_blocks = block_table.shape
    num_kv_heads, head_dim = v_cache.shape[2:]
    num_groups = triton.cdiv(max_new_tokens, _BLOCK_N)

    with torch.cuda.device(v_cache.device):
        _fake_quant_v_onwrite_kernel[(batch, num_kv_heads, num_groups)](
            v_cache,
            block_table,
            v_lo,
            v_hi,
            v_cache.stride(0),
            v_cache.stride(1),
            v_cache.stride(2),
            v_qdq_scale,
            BLOCK_N=_BLOCK_N,
            BLOCK_D=triton.next_power_of_2(head_dim),
            HEAD_DIM=head_dim,
            PAGE_SIZE=page_size,
            max_blocks_per_seq=max_blocks,
            num_warps=4,
        )
