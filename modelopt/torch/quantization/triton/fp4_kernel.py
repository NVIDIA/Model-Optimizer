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

"""NVFP4 Fake Quantization Triton Implementation.

This module provides high-performance GPU implementations of NVFP4 fake quantization
operations using Triton kernels.
"""

import torch
import triton
import triton.language as tl

__all__ = ["fp4_fake_quant_block"]


_TORCH_TO_TL_DTYPE = {
    torch.float32: tl.float32,
    torch.float: tl.float32,
    torch.float16: tl.float16,
    torch.half: tl.float16,
    torch.bfloat16: tl.bfloat16,
}


def _torch_dtype_to_tl(dtype: torch.dtype):
    if dtype not in _TORCH_TO_TL_DTYPE:
        raise ValueError(f"Unsupported dtype for fp4 fake quantization: {dtype}")
    return _TORCH_TO_TL_DTYPE[dtype]


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
        tile_rows (int, optional): Row tile size. Defaults to 64.
        tile_cols (int, optional): Column tile size. Defaults to 128. Rounded up to
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

    global_scale = global_amax.float() / (6.0 * 448.0)

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


@triton.jit
def fp4_dequantize_kernel(
    packed_ptr,
    scale_ptr,
    global_scale_ptr,
    output_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
    TILE_SIZE: tl.constexpr,
):
    """Dequantizes FP4 packed data using per-block scaling factors.

    Args:
        packed_ptr (tl.pointer): Pointer to packed uint8 tensor (M x N//2)
        scale_ptr (tl.pointer): Pointer to per-block scale tensor (M x N//BLOCK_SIZE)
        output_ptr (tl.pointer): Pointer to output tensor (M x N)
        global_scale_ptr (tl.pointer): Pointer to global scale tensor
        N (int): Number of columns in unpacked tensor
        BLOCK_SIZE (tl.constexpr): Size of each FP4 quantization block
        TILE_SIZE (tl.constexpr): Size of the processing tile (in packed elements)
    """
    # Get program ID for processing packed elements
    pid = tl.program_id(0)

    # Calculate packed element offsets (each packed element contains 2 FP4 values)
    packed_start = pid * TILE_SIZE
    packed_offs = packed_start + tl.arange(0, TILE_SIZE)

    # Calculate 2D coordinates for packed data
    packed_row_idx = packed_offs // (N // 2)
    packed_col_idx = packed_offs % (N // 2)

    # Create mask for packed data bounds checking
    packed_mask = packed_col_idx < (N // 2)

    # Load global scale
    global_scale = tl.load(global_scale_ptr)

    # Load packed data
    packed_data = tl.load(packed_ptr + packed_offs, mask=packed_mask, other=0)

    # Unpack packed FP4 values (uint8) to float16x2
    x_f16x2_packed = tl.inline_asm_elementwise(
        asm="""
        {
            .reg .b8 byte0, byte1, byte2, byte3;
            mov.b32 {byte0, byte1, byte2, byte3}, $4;
            cvt.rn.f16x2.e2m1x2 $0, byte0;
            cvt.rn.f16x2.e2m1x2 $1, byte1;
            cvt.rn.f16x2.e2m1x2 $2, byte2;
            cvt.rn.f16x2.e2m1x2 $3, byte3;
        }
        """,
        constraints="=r,=r,=r,=r,r",
        args=[packed_data],
        dtype=tl.uint32,
        is_pure=True,
        pack=4,
    )
    val_low = (
        (x_f16x2_packed & 0xFFFF).cast(tl.uint16).cast(tl.float16, bitcast=True).cast(tl.float32)
    )
    val_high = (
        (x_f16x2_packed >> 16).cast(tl.uint16).cast(tl.float16, bitcast=True).cast(tl.float32)
    )

    # Calculate output positions for both values
    out_col_low = packed_col_idx * 2
    out_col_high = packed_col_idx * 2 + 1
    out_offs_low = packed_row_idx * N + out_col_low
    out_offs_high = packed_row_idx * N + out_col_high

    # Calculate block indices for scaling
    block_col_low = out_col_low // BLOCK_SIZE
    block_col_high = out_col_high // BLOCK_SIZE
    scale_offs_low = packed_row_idx * (N // BLOCK_SIZE) + block_col_low
    scale_offs_high = packed_row_idx * (N // BLOCK_SIZE) + block_col_high

    # Load scaling factors
    scale_low = tl.load(scale_ptr + scale_offs_low, mask=packed_mask & (out_col_low < N), other=1.0)
    scale_high = tl.load(
        scale_ptr + scale_offs_high, mask=packed_mask & (out_col_high < N), other=1.0
    )

    # Apply scaling
    result_low = val_low * scale_low.to(tl.float32) * global_scale
    result_high = val_high * scale_high.to(tl.float32) * global_scale

    # Store results
    out_mask_low = packed_mask & (out_col_low < N)
    out_mask_high = packed_mask & (out_col_high < N)

    tl.store(output_ptr + out_offs_low, result_low, mask=out_mask_low)
    tl.store(output_ptr + out_offs_high, result_high, mask=out_mask_high)


def fp4_dequantize(
    packed_tensor: torch.Tensor,
    scale_tensor: torch.Tensor,
    global_scale: torch.Tensor,
    block_size: int = 16,
    tile_size: int = 128,
    dtype: torch.dtype = torch.get_default_dtype(),
) -> torch.Tensor:
    """Dequantizes FP4 packed tensor using per-block scaling factors.

    Args:
        packed_tensor (torch.Tensor): Packed uint8 tensor of shape (M, N//2)
        scale_tensor (torch.Tensor): Per-block scale tensor of shape (M, N//block_size)
        global_scale (torch.Tensor): Global scaling factor tensor
        block_size (int): Size of FP4 quantization blocks
        tile_size (int): Size of processing tiles

    Returns:
        torch.Tensor: Dequantized tensor of shape (M, N)
    """
    packed_N = packed_tensor.shape[-1]
    N = packed_N * 2
    # Create output tensor with proper shape handling
    output_shape = list(packed_tensor.shape)
    output_shape[-1] = N
    output = torch.empty(output_shape, dtype=dtype, device=packed_tensor.device)

    # Calculate total number of elements and grid size
    grid = lambda meta: (triton.cdiv(packed_tensor.numel(), meta["TILE_SIZE"]),)

    fp4_dequantize_kernel[grid](
        packed_tensor,
        scale_tensor,
        global_scale,
        output,
        N,
        BLOCK_SIZE=block_size,
        TILE_SIZE=tile_size,
    )

    return output
