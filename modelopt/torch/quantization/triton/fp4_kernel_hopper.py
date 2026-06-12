#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""NVFP4 Fake Quantization Triton kernels requiring compute capability >= 8.9 (Hopper+).

These kernels use tl.float8e4nv which requires native FP8 hardware support.
"""

import torch
import triton
import triton.language as tl

from .fp4_kernel import _torch_dtype_to_tl

__all__ = ["fp4_fake_quant_block"]


@triton.jit
def fp4_fake_quant_kernel(
    x_ptr,
    y_ptr,
    M,
    N,
    global_scale_ptr,
    stride_xm,
    stride_xn,
    stride_ym,
    stride_yn,
    BLOCK_SIZE: tl.constexpr,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
    NUM_FP4_BLOCKS: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
):
    """Applies FP4 fake quantization using block pointers for memory addressing."""
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    row_start = pid_m * TILE_M
    col_start = pid_n * TILE_N

    x_block_ptr = tl.make_block_ptr(
        base=x_ptr,
        shape=(M, N),
        strides=(stride_xm, stride_xn),
        offsets=(row_start, col_start),
        block_shape=(TILE_M, TILE_N),
        order=(1, 0),
    )
    y_block_ptr = tl.make_block_ptr(
        base=y_ptr,
        shape=(M, N),
        strides=(stride_ym, stride_yn),
        offsets=(row_start, col_start),
        block_shape=(TILE_M, TILE_N),
        order=(1, 0),
    )

    global_scale = tl.load(global_scale_ptr).to(tl.float32)
    global_scale_safe = tl.where(global_scale > 0.0, global_scale, 1e-12)

    tile = tl.load(x_block_ptr, boundary_check=(0, 1), padding_option="zero").to(tl.float32)

    tile_reshaped = tl.reshape(tile, (TILE_M, NUM_FP4_BLOCKS, BLOCK_SIZE))
    x_abs = tl.abs(tile_reshaped)

    block_max = tl.max(x_abs, axis=2, keep_dims=True)

    block_max_scaled = block_max / (6.0 * global_scale_safe)
    block_max_scaled = tl.minimum(block_max_scaled, 448.0)
    block_max_quant = block_max_scaled.to(tl.float8e4nv).to(tl.float32) * global_scale
    block_max_quant = tl.where(block_max_quant >= 1e-5, block_max_quant, 1.0)

    block_max_quant_broadcast = tl.broadcast_to(
        block_max_quant, (TILE_M, NUM_FP4_BLOCKS, BLOCK_SIZE)
    )

    abs_scaled = x_abs / block_max_quant_broadcast

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
                        tl.where(
                            abs_scaled < 3.5,
                            3.0,
                            tl.where(abs_scaled <= 5.0, 4.0, 6.0),
                        ),
                    ),
                ),
            ),
        ),
    )

    x_rescaled = q_val * block_max_quant_broadcast
    x_rescaled = tl.where(tile_reshaped >= 0, x_rescaled, -x_rescaled)

    tile_quant = tl.reshape(x_rescaled, (TILE_M, TILE_N))

    tl.store(y_block_ptr, tile_quant.to(OUT_DTYPE), boundary_check=(0, 1))


def fp4_fake_quant_block(
    x: torch.Tensor,
    global_amax: torch.Tensor,
    block_size: int = 16,
    tile_rows: int = 16,
    tile_cols: int = 64,
    num_warps: int | None = None,
    num_stages: int | None = None,
) -> torch.Tensor:
    """FP4 fake quantization implementation using block-pointer tiling.

    Args:
        x (torch.Tensor): Input tensor of shape ``(M, N)`` or higher.
        global_amax (torch.Tensor): Global maximum value tensor for scaling.
        block_size (int): Number of elements per FP4 block.
        tile_rows (int, optional): Row tile size. Defaults to 16.
        tile_cols (int, optional): Column tile size. Defaults to 64. Rounded up to
            the nearest multiple of ``block_size`` internally.
        num_warps (int | None, optional): Override for Triton warps. Autotuned when ``None``.
        num_stages (int | None, optional): Override for pipeline stages. Autotuned when ``None``.

    Returns:
        torch.Tensor: Fake-quantized tensor matching the input shape and dtype.
    """
    x_shape = x.shape
    x_dtype = x.dtype
    x = x.reshape(-1, x_shape[-1]).contiguous()

    M, N = x.shape
    y = torch.empty_like(x)

    stride_xm, stride_xn = x.stride()
    stride_ym, stride_yn = y.stride()

    tile_cols = max(tile_cols, block_size)
    tile_cols_aligned = ((tile_cols + block_size - 1) // block_size) * block_size
    num_fp4_blocks = tile_cols_aligned // block_size

    global_scale = (global_amax.float() / (6.0 * 448.0)).to(x.device)

    grid = lambda *_: (triton.cdiv(M, tile_rows), triton.cdiv(N, tile_cols_aligned))

    launch_kwargs = {
        "BLOCK_SIZE": block_size,
        "TILE_M": tile_rows,
        "TILE_N": tile_cols_aligned,
        "NUM_FP4_BLOCKS": num_fp4_blocks,
        "OUT_DTYPE": _torch_dtype_to_tl(x_dtype),
    }
    if num_warps is not None:
        launch_kwargs["num_warps"] = num_warps
    if num_stages is not None:
        launch_kwargs["num_stages"] = num_stages
    with torch.cuda.device(x.device):
        fp4_fake_quant_kernel[grid](
            x,
            y,
            M,
            N,
            global_scale,
            stride_xm,
            stride_xn,
            stride_ym,
            stride_yn,
            **launch_kwargs,
        )

    y = y.view(*x_shape)
    return y
