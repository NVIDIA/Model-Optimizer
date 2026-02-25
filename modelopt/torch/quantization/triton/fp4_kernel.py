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

__all__ = ["fp4_dequantize", "static_blockwise_fp4_fake_quant"]


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


@triton.jit
def static_blockwise_fp4_fake_quant_kernel(
    x_ptr,  # [NUM_FP4_BLOCKS * BLOCK_SIZE]
    y_ptr,  # [NUM_FP4_BLOCKS * BLOCK_SIZE]
    scale_ptr,  # [NUM_FP4_BLOCKS]
    NUM_FP4_BLOCKS,
    BLOCK_SIZE: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    if pid >= NUM_FP4_BLOCKS:
        return

    block_offset = pid * BLOCK_SIZE
    idx = block_offset + tl.arange(0, BLOCK_SIZE)

    scale = tl.load(scale_ptr + pid).to(tl.float32)

    x = tl.load(x_ptr + idx).to(tl.float32)

    x_abs = tl.abs(x)
    # If scale is 0, inf, or nan, use 1.0 (matching CUDA kernel behavior)
    # Note: (x != x) checks if x is NaN per IEEE 754
    scale_safe = tl.where(
        (scale == 0) | (scale != scale) | (tl.abs(scale) == float("inf")),  # noqa: PLR0124
        1.0,
        scale,
    )
    abs_scaled = x_abs / scale_safe

    # FP4 values: 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0
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

    x_rescaled = q_val * scale_safe
    x_quant = tl.where(x >= 0, x_rescaled, -x_rescaled)

    tl.store(y_ptr + idx, x_quant.to(OUT_DTYPE))


def static_blockwise_fp4_fake_quant(
    x: torch.Tensor,
    amax: torch.Tensor,
    global_amax: torch.Tensor | None = None,
    quantize_block_scales: bool = True,
    out_dtype: torch.dtype | None = None,
):
    """Static blockwise FP4 fake quantization using Triton kernel.

    Args:
        x: [NUM_FP4_BLOCKS, BLOCK_SIZE] on CUDA.
        amax: [NUM_FP4_BLOCKS] or [NUM_FP4_BLOCKS, 1] per-block amax values.
        global_amax: FP32 scalar global amax. If provided, used to compute scale_fp8_quant_amax.
        quantize_block_scales: If True, quantize block scales to FP8.
        out_dtype: Output dtype. Defaults to x.dtype if None.
    """
    assert x.ndim == 2
    NUM_FP4_BLOCKS, BLOCK_SIZE = x.shape

    if out_dtype is None:
        out_dtype = x.dtype

    amax = amax.float()  # Requires to be in float32
    scale = amax / 6.0  # FP4 max representable value is 6.0

    if quantize_block_scales:
        from modelopt.torch.quantization.tensor_quant import scaled_e4m3_impl
        from modelopt.torch.quantization.utils import reduce_amax

        if global_amax is None:
            global_amax = reduce_amax(amax, axis=None, keepdims=False, squeeze_scalar=True)

        global_amax = global_amax.float()
        scale_fp8_quant_amax = global_amax / 6.0
        scale = scaled_e4m3_impl(scale, scale_fp8_quant_amax)

    x_flat = x.contiguous().view(-1)
    y_flat = torch.empty_like(x_flat, dtype=out_dtype)
    scale_flat = scale.view(NUM_FP4_BLOCKS).contiguous()

    tl_out_dtype = _torch_dtype_to_tl(out_dtype)

    grid = (NUM_FP4_BLOCKS,)

    with torch.cuda.device(x.device):
        static_blockwise_fp4_fake_quant_kernel[grid](
            x_flat,
            y_flat,
            scale_flat,
            NUM_FP4_BLOCKS,
            BLOCK_SIZE,
            OUT_DTYPE=tl_out_dtype,
        )

    return y_flat.view_as(x)
