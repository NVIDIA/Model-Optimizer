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

"""Composable Triton JIT functions for FP8 (E4M3) fake quantization.

Counterpart of ``nvfp4_quant.py`` for per-tensor FP8. Used by the unified
flash-attention kernel's softmax-P qdq and linear-attention state/factor qdq.
"""

import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

__all__ = ["fp8_dynamic_qdq", "fp8_scalar_qdq"]


@triton.jit
def _fp8_cast(value, NATIVE_FP8: tl.constexpr):
    value = tl.clamp(value, -448.0, 448.0)
    if NATIVE_FP8:
        return value.to(tl.float8e4nv).to(tl.float32)
    # GPUs before SM89 need a software E4M3 conversion, including subnormals and ties.
    bits = tl.abs(value).to(tl.int32, bitcast=True)
    exponent = ((bits >> 23) & 255) - 127
    step = ((tl.maximum(exponent, -6) - 3 + 127) << 23).to(tl.float32, bitcast=True)
    return libdevice.nearbyint(value / step) * step


@triton.jit
def fp8_dynamic_qdq(value, NATIVE_FP8: tl.constexpr = True):
    """Dynamic per-tile FP8 QDQ with TensorQuantizer scaling and returned decode scale."""
    amax = tl.max(tl.abs(value))
    safe_amax = tl.where(amax <= 2.0**-24, 1.0, amax)
    quant_scale = tl.div_rn(448.0, safe_amax)
    decode_scale = tl.div_rn(1.0, quant_scale)
    return _fp8_cast(value * quant_scale, NATIVE_FP8) * decode_scale, decode_scale


@triton.jit
def fp8_scalar_qdq(x, scale):
    """Per-tensor FP8 E4M3 fake quant-dequant: ``cast(x / scale) * scale``.

    Standard quantizer convention with ``scale = amax / 448``. Works with any
    tensor shape and sign (all ops are element-wise); out-of-range values
    saturate to +-448 like a real quantizer.

    Args:
        x:     Tensor of values to fake-quantize.
        scale: Per-tensor scale (runtime scalar or broadcastable tensor).

    Returns:
        Fake-quantized tensor of the same shape as ``x``, in float32.
    """
    return _fp8_cast(x / scale, True) * scale
