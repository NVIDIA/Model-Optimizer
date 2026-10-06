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

"""Composable signed narrow-range INT8 QDQ matching CUDA TensorQuantizer."""

import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

__all__ = []


@triton.jit
def int8_scalar_qdq(value, amax):
    """QDQ with a scalar or broadcastable amax, using CUDA TensorQuantizer arithmetic."""
    # Match gemm/tensor_quant_gpu.cu: tiny groups become zero; ties round to even.
    tiny = amax < 2.0**-24
    scale = tl.div_rn(127.0, tl.where(tiny, 1.0, amax))
    codes = tl.clamp(libdevice.nearbyint(value * scale), -127.0, 127.0)
    return tl.where(tiny, 0.0, tl.div_rn(codes, scale))


@triton.jit
def int8_block_qdq(value, GROUP_SIZE: tl.constexpr):
    """Dynamic INT8 QDQ per row and last-axis group of a two-dimensional tile."""
    groups = tl.reshape(value, (value.shape[0], value.shape[1] // GROUP_SIZE, GROUP_SIZE))
    amax = tl.max(tl.abs(groups), axis=2, keep_dims=True)
    return tl.reshape(int8_scalar_qdq(groups, amax), value.shape)
