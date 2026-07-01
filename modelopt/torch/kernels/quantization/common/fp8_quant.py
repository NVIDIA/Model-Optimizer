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
flash-attention kernel's softmax-P qdq (``common/attention/triton_fa.py``).
"""

import triton
import triton.language as tl


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
    FP8_E4M3_MAX: tl.constexpr = 448.0
    x_scaled = tl.clamp(x / scale, -FP8_E4M3_MAX, FP8_E4M3_MAX)
    return x_scaled.to(tl.float8e4nv).to(tl.float32) * scale
