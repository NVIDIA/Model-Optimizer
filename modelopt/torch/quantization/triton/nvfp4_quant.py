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

"""Composable Triton JIT functions for NVFP4 (E2M1) fake quantization.

Single source of truth for FP4 decision-boundary rounding.  Used by:
  - ``fp4_kernel.py``         (standalone blockwise fake quant)
  - ``fp4_kernel_hopper.py``  (Hopper block-pointer variant)
  - ``gptq_fused_kernel.py``  (fused GPTQ scalar path)

FP4 (E2M1) representable magnitudes: {0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}
"""

import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def fp4_round_magnitude(abs_scaled):
    """Round ``|x| / scale`` to the nearest FP4 (E2M1) magnitude.

    Works with any tensor shape — the caller is responsible for computing
    ``abs_scaled = |x| / scale`` beforehand.

    Returns:
        Tensor of same shape as *abs_scaled* with values in
        {0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}.
    """
    return tl.where(
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


@triton.jit
def nvfp4_scalar_quant(
    x,  # [N] float32, already loaded
    scale,  # float32 scalar: pre-computed block scale (amax / 6.0)
    N: tl.constexpr,
):
    """NVFP4 scalar fake quantization for a group of elements sharing one scale.

    Quantizes each element independently: divide by scale, round to nearest
    FP4 (E2M1) value via ``fp4_round_magnitude``, multiply by scale.

    Args:
        x:     [N] float32 tensor of values to quantize (already in registers).
        scale: float32 scalar block scale.
        N:     Compile-time number of elements.

    Returns:
        x_quant: [N] float32, fake-quantized values.
    """
    x_abs = tl.abs(x)
    # Guard against degenerate scale (matching CUDA kernel behavior)
    scale_safe = tl.where(
        (scale == 0.0) | libdevice.isnan(scale) | (tl.abs(scale) == float("inf")),
        1.0,
        scale,
    )
    abs_scaled = x_abs / scale_safe
    q_val = fp4_round_magnitude(abs_scaled)
    x_rescaled = q_val * scale_safe
    x_quant = tl.where(x >= 0, x_rescaled, -x_rescaled)
    return x_quant


@triton.jit
def fp8_quantize_scale(block_amax, global_scale):
    """FP8 E4M3 fake-quantize the per-block NVFP4 scale.

    Computes ``scale = block_amax / 6.0``, then round-trips it through
    FP8 E4M3 using ``global_scale`` for the second-level scaling.

    Works with any tensor shape (scalar, 1-D, or higher) since all ops
    are element-wise.

    Args:
        block_amax:   Per-block amax value(s).
        global_scale: Pre-computed ``global_amax / (6.0 * 448.0)``.

    Returns:
        FP8-quantized per-block scale(s), same shape as ``block_amax``.
    """
    FP8_E4M3_MAX: tl.constexpr = 448.0
    scale_in_fp8_range = block_amax / (6.0 * global_scale)
    scale_clamped = tl.minimum(scale_in_fp8_range, FP8_E4M3_MAX)
    return scale_clamped.to(tl.float8e4nv).to(tl.float32) * global_scale


@triton.jit
def nvfp4_scalar_qdq(
    x,  # [N] float32, already loaded
    block_amax,  # float32 scalar: per-block amax
    global_scale,  # float32 scalar: pre-computed global_amax / (6.0 * 448.0)
    N: tl.constexpr,
):
    """NVFP4 scalar fake quantization with inline two-level scale computation.

    Computes the per-block FP8-quantized scale from ``block_amax`` via
    :func:`fp8_quantize_scale`, then quantizes each element to the nearest
    FP4 (E2M1) value.

    Args:
        x:            [N] float32 tensor of values to quantize.
        block_amax:   Per-block amax (absolute maximum of the block).
        global_scale: Pre-computed ``global_amax / (6.0 * 448.0)``.
        N:            Compile-time number of elements.

    Returns:
        x_quant: [N] float32, fake-quantized values.
    """
    scale = fp8_quantize_scale(block_amax, global_scale)
    return nvfp4_scalar_quant(x, scale, N)
