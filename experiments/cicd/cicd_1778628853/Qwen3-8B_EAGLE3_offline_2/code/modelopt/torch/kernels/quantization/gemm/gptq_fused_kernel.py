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

"""Fused Triton kernels for GPTQ blockwise weight-update.

A kernel for scalar (NVFP4) quantization with inline two-level scale computation.
Fuses scale computation + quantization + per-column GPTQ error propagation into
one launch per GPTQ block, avoiding the Python-level per-column loop.

Architecture:
  - One Triton program per output row.
  - ``w_full [BLOCK_SIZE]`` register tensor holds working weights.
  - Per-column: calls ``nvfp4_scalar_qdq()`` for FP4 QDQ with inline scale
    computation, then propagates error via ``w_full -= err * h_inv_row``.
"""

import torch
import triton
import triton.language as tl

from .nvfp4_quant import nvfp4_scalar_qdq

__all__ = ["gptq_fused_block_scalar"]


# ---------------------------------------------------------------------------
# Scalar kernel — NVFP4 QDQ + error propagation
# ---------------------------------------------------------------------------


@triton.jit
def _gptq_scalar_kernel(
    w_ptr,
    qw_ptr,
    err_ptr,
    amax_ptr,
    global_scale,
    hinv_ptr,
    num_rows,
    n_amax_blocks,
    quant_block_size,
    block_start,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= num_rows:
        return

    w_base = w_ptr + row * BLOCK_SIZE
    qw_base = qw_ptr + row * BLOCK_SIZE
    err_base = err_ptr + row * BLOCK_SIZE
    amax_base = amax_ptr + row * n_amax_blocks

    j_range = tl.arange(0, BLOCK_SIZE)
    w_full = tl.load(w_base + j_range)

    for col in range(0, BLOCK_SIZE, 1):
        block_amax = tl.load(amax_base + (block_start + col) // quant_block_size)

        w_scalar = tl.sum(tl.where(j_range == col, w_full, 0.0))
        q_scalar = tl.sum(
            nvfp4_scalar_qdq(
                tl.full([1], w_scalar, dtype=tl.float32),
                block_amax,
                global_scale,
                1,
            )
        )

        d_val = tl.load(hinv_ptr + col * BLOCK_SIZE + col)
        err_val = (w_scalar - q_scalar) / d_val
        tl.store(err_base + col, err_val)
        tl.store(qw_base + col, q_scalar)

        remaining = j_range > col
        hinv_row = tl.load(hinv_ptr + col * BLOCK_SIZE + j_range, mask=remaining, other=0.0)
        w_full = w_full - err_val * hinv_row


def gptq_fused_block_scalar(
    w_block: torch.Tensor,
    block_amax: torch.Tensor,
    global_scale: float,
    h_inv_cho_blk: torch.Tensor,
    quant_block_size: int,
    block_start: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run scalar GPTQ (NVFP4) column loop for one block in a single Triton kernel launch.

    Computes FP8-quantized scales from per-block amax inline via
    :func:`nvfp4_scalar_qdq`, then performs NVFP4 fake quantization and
    GPTQ error propagation per column.

    Args:
        w_block:         Working weights ``[num_rows, block_size]`` (float32).
        block_amax:      Per-block amax values ``[num_rows, n_amax_blocks]`` (float32).
        global_scale:    Pre-computed ``global_amax / (6.0 * 448.0)`` (scalar).
        h_inv_cho_blk:   Block of upper-Cholesky inverse Hessian ``[block_size, block_size]``.
        quant_block_size: Number of elements sharing one scale factor.
        block_start:     Column offset of this block in the full weight matrix.

    Returns:
        ``(qw_block, err_block)`` each ``[num_rows, block_size]``.
    """
    num_rows, block_size = w_block.shape

    qw_block = torch.empty_like(w_block)
    err_block = torch.empty_like(w_block)

    _gptq_scalar_kernel[(num_rows,)](
        w_block.contiguous(),
        qw_block,
        err_block,
        block_amax.contiguous(),
        global_scale,
        h_inv_cho_blk.contiguous(),
        num_rows,
        block_amax.shape[1],
        quant_block_size,
        block_start,
        BLOCK_SIZE=block_size,
    )

    return qw_block, err_block
