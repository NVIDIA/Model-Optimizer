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

"""Fused Triton kernel for the GPTQ blockwise weight-update inner loop.

The standard GPTQ inner loop launches ~10-15 CUDA kernels per column
(amax lookup, FP4 quantization, error computation, rank-1 update).
For ``block_size=128`` that is ~1 500 kernel launches per block, each with
~5-10 us of launch overhead dominating actual compute.

This module fuses the entire inner loop into a **single** Triton kernel per
block.  Rows are independent and map to Triton programs; columns are processed
sequentially inside each program so the rank-1 error update is carried forward
without synchronisation.

Supported quantisation format: **NVFP4 static block quantisation** (two-level
scaling with per-group amax and a global amax).
"""

import torch
import triton
import triton.language as tl

__all__ = ["gptq_fused_block"]

# -- NVFP4 constants used by the kernel ------------------------------------
# Maximum representable FP4-E2M1 value (1 + 1 + 0.5 = 6.0 when decoded via
# the standard E2M1 table: {0, 0.5, 1, 1.5, 2, 3, 4, 6}).
_FP4_MAX = 6.0
# FP8-E4M3 has max representable value 448.
_FP8_E4M3_MAX = 448.0


@triton.jit
def _gptq_fused_block_kernel(
    w_ptr,  # [num_rows, BLOCK_SIZE] working weight block (in-place)
    qw_ptr,  # [num_rows, BLOCK_SIZE] output: quantized weights
    err_ptr,  # [num_rows, BLOCK_SIZE] output: quantization errors
    amax_ptr,  # [num_rows, num_groups] per-group amax, row-major
    global_amax_ptr,  # scalar float32 on device
    hinv_ptr,  # [BLOCK_SIZE, BLOCK_SIZE] upper Cholesky of H^{-1}
    num_rows,
    num_groups,
    group_size: tl.constexpr,
    block_start,  # column offset of this block in the full weight matrix
    n_cols,  # actual columns in this block (may be < BLOCK_SIZE)
    BLOCK_SIZE: tl.constexpr,
):
    """One program per row; sequentially quantizes columns, propagating errors."""
    row = tl.program_id(0)
    if row >= num_rows:
        return

    # Base pointers for this row
    w_base = w_ptr + row * BLOCK_SIZE
    qw_base = qw_ptr + row * BLOCK_SIZE
    err_base = err_ptr + row * BLOCK_SIZE
    amax_row_base = amax_ptr + row * num_groups

    # Pre-compute global FP8 scale factors (constant across columns)
    global_amax = tl.load(global_amax_ptr).to(tl.float32)
    global_scale = global_amax / 6.0  # _FP4_MAX
    fp8_inv_scale = tl.where(global_scale > 0.0, 1.0 / (448.0 / global_scale), 0.0)

    j_range = tl.arange(0, BLOCK_SIZE)

    for i in range(BLOCK_SIZE):
        wi = tl.load(w_base + i)

        # -- Compute NVFP4 two-level scale for this column's group -----------
        col_idx = block_start + i
        group_idx = col_idx // group_size
        raw_amax = tl.load(amax_row_base + group_idx).to(tl.float32)
        raw_scale = raw_amax / 6.0  # _FP4_MAX

        # FP8-quantize the block scale: scale * fp8_scale -> cast E4M3 -> back
        fp8_scale = tl.where(global_scale > 0.0, 448.0 / global_scale, 1.0)
        si = (raw_scale * fp8_scale).to(tl.float8e4nv).to(tl.float32) * fp8_inv_scale

        # Guard: replace zero / nan / inf scale with 1.0
        # NOTE: ``si != si`` is the standard NaN check in Triton (no math.isnan).
        si_safe = tl.where(
            (si == 0.0) | (si != si) | (tl.abs(si) == float("inf")),  # noqa: PLR0124
            1.0,
            si,
        )

        # -- FP4-E2M1 fake quantization (nearest-round to 8 levels) ----------
        abs_scaled = tl.abs(wi) / si_safe
        q_val = tl.where(
            abs_scaled <= 0.25,
            0.0,
            tl.where(
                abs_scaled < 0.75,
                0.5,
                tl.where(
                    abs_scaled <= 1.25,
                    1.0,
                    tl.where(
                        abs_scaled < 1.75,
                        1.5,
                        tl.where(
                            abs_scaled <= 2.5,
                            2.0,
                            tl.where(abs_scaled < 3.5, 3.0, tl.where(abs_scaled <= 5.0, 4.0, 6.0)),
                        ),
                    ),
                ),
            ),
        )

        qi = q_val * si_safe * tl.where(wi >= 0.0, 1.0, -1.0)
        tl.store(qw_base + i, qi)

        # -- GPTQ error and rank-1 update ------------------------------------
        di = tl.load(hinv_ptr + i * BLOCK_SIZE + i)
        err_i = (wi - qi) / di
        tl.store(err_base + i, err_i)

        j_mask = (j_range > i) & (j_range < n_cols)
        hinv_row = tl.load(hinv_ptr + i * BLOCK_SIZE + j_range, mask=j_mask, other=0.0)
        w_rem = tl.load(w_base + j_range, mask=j_mask, other=0.0)
        w_rem = w_rem - err_i * hinv_row
        tl.store(w_base + j_range, w_rem, mask=j_mask)


def gptq_fused_block(
    w_block: torch.Tensor,
    amax_grouped: torch.Tensor,
    global_amax: torch.Tensor,
    h_inv_cho_blk: torch.Tensor,
    group_size: int,
    block_start: int,
    n_cols: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the GPTQ column loop for one block in a single Triton kernel launch.

    Args:
        w_block: Working weight block of shape ``[num_rows, block_size]`` (will be cloned).
        amax_grouped: Per-group amax of shape ``[num_rows, num_groups]``.
        global_amax: Scalar tensor with the global amax.
        h_inv_cho_blk: Upper Cholesky factor of H^{-1}, shape ``[block_size, block_size]``.
        group_size: NVFP4 quantization group size (typically 16).
        block_start: Column offset of this block in the full weight matrix.
        n_cols: Actual number of columns in this block (``<= block_size``).

    Returns:
        Tuple of ``(qw_block, err_block)`` each of shape ``[num_rows, block_size]``.
    """
    num_rows, block_size = w_block.shape
    num_groups = amax_grouped.shape[1]

    w_block = w_block.contiguous()
    amax_grouped = amax_grouped.contiguous()
    h_inv_cho_blk = h_inv_cho_blk.contiguous()

    qw_block = torch.empty_like(w_block)
    err_block = torch.empty_like(w_block)

    grid = (num_rows,)
    with torch.cuda.device(w_block.device):
        _gptq_fused_block_kernel[grid](
            w_block,
            qw_block,
            err_block,
            amax_grouped,
            global_amax,
            h_inv_cho_blk,
            num_rows,
            num_groups,
            group_size,
            block_start,
            n_cols,
            BLOCK_SIZE=block_size,
        )

    return qw_block, err_block
