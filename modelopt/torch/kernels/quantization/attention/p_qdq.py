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

"""Softmax-P quant-dequant helpers for the unified flash attention kernel.

These ``@triton.jit`` helpers fake-quantize the softmax probabilities ``P``
before the ``P @ V`` matmul (BMM2) — the in-kernel counterpart of the
``p_bmm_quantizer`` config. They are called conditionally from the baseline
flash-attention kernel in ``common/attention/triton_fa.py`` under the
``P_QDQ`` constexpr guard, following the same composition pattern as
the sparsity helpers in ``sparsity/attention/skip_softmax_helpers.py``.

Only NVFP4 needs a P-specific helper (tiling policy and block amaxes); the
per-tensor FP8 mode uses ``quantization/common/fp8_quant.fp8_scalar_qdq``
directly. What is P-specific here: the kernel's online-softmax ``p`` is
unnormalized and bounded (``0 <= p <= 1``), so the per-tensor amax is
exactly 1.0 by construction (default scale 1.0 means an effective global
amax of 6 * 448), block amaxes need no ``abs``, and the NVFP4 scale blocks
of 16 run along the key dimension — the contraction axis of ``P @ V``.
"""

import triton
import triton.language as tl

from modelopt.torch.kernels.quantization.gemm.nvfp4_quant import nvfp4_scalar_qdq


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
    conventionally ``amax / (6 * 448)``). The default of 1.0 means an
    effective global amax of 6 * 448 = 2688.

    ``p >= 0``, so the block amax is a plain max (no ``abs``), and
    ``nvfp4_scalar_qdq`` guards the degenerate all-zero blocks of fully
    masked or padded positions.
    """
    tl.static_assert(BLOCK_N % 16 == 0, "BLOCK_N must be divisible by 16 for NVFP4")

    grouped = tl.reshape(p, (BLOCK_M, BLOCK_N // 16, 16))
    block_amax = tl.expand_dims(tl.max(grouped, axis=2), 2)  # p >= 0, so max == amax
    q = nvfp4_scalar_qdq(grouped, block_amax, global_scale, 16)
    return tl.reshape(q, (BLOCK_M, BLOCK_N))
