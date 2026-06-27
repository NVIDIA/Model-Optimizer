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

"""BMM2-operand (softmax-P and value) quant-dequant helpers for flash attention.

These ``@triton.jit`` helpers fake-quantize the two operands of the ``P @ V``
matmul (BMM2) in-kernel — ``P`` (:func:`_p_qdq_nvfp4`, the counterpart of
``p_bmm_quantizer``) and ``V`` (:func:`_v_qdq_nvfp4`, the counterpart of
``v_bmm_quantizer``). They are called conditionally from the baseline
flash-attention kernel in ``common/attention/triton_fa.py`` (and the paged
decode kernel) under the ``P_QDQ`` / ``V_QDQ`` constexpr guards, following the
same composition pattern as the sparsity helpers in
``sparsity/attention/skip_softmax_helpers.py``. Both block 16 along the key
dimension — the contraction axis of ``P @ V`` — which a per-token cache write
cannot produce, so V is fake-quantized here on read rather than on write.

Only NVFP4 needs a P-specific helper (tiling policy and block amaxes); the
per-tensor FP8 mode uses ``quantization/common/fp8_quant.fp8_scalar_qdq``
directly. What is P-specific here: the kernel's online-softmax ``p`` is
unnormalized and bounded (``0 <= p <= 1``, since the max-subtraction caps
every entry at ``exp2(0) = 1``), so 1 is the theoretical upper bound of its
amax; block amaxes need no ``abs``; and the NVFP4 scale blocks of 16 run
along the key dimension — the contraction axis of ``P @ V``. The caller
(``attention()``) converts the amax to the ``global_scale`` below.
"""

import triton
import triton.language as tl

from modelopt.torch.kernels.quantization.common.nvfp4_quant import nvfp4_scalar_qdq


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
    ``amax / (6 * 448)``; ``attention()`` derives it from ``p_qdq_amax``,
    which defaults to 1, the theoretical upper bound of P's amax).

    ``p >= 0``, so the block amax is a plain max (no ``abs``), and
    ``nvfp4_scalar_qdq`` guards the degenerate all-zero blocks of fully
    masked or padded positions.
    """
    tl.static_assert(BLOCK_N % 16 == 0, "BLOCK_N must be divisible by 16 for NVFP4")

    grouped = tl.reshape(p, (BLOCK_M, BLOCK_N // 16, 16))
    block_amax = tl.expand_dims(tl.max(grouped, axis=2), 2)  # p >= 0, so max == amax
    q = nvfp4_scalar_qdq(grouped, block_amax, global_scale, 16)
    return tl.reshape(q, (BLOCK_M, BLOCK_N))


@triton.jit
def _v_qdq_nvfp4(
    v,
    global_scale,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """NVFP4 fake quant-dequant of the value operand ``V`` of ``P @ V`` (BMM2).

    Same two-level NVFP4 scaling as :func:`_p_qdq_nvfp4`, but ``V``'s blocks of
    16 run along the *key* dimension — which for the loaded tile
    ``v [BLOCK_N keys, BLOCK_D head_dim]`` is axis 0 — so 16 contiguous keys
    share a scale, per head-dim channel (the contraction axis of ``P @ V``).
    Unlike ``P``, ``V`` is signed, so the block amax uses ``abs``.

    Relies on the caller masking out-of-range keys to 0 (``_load_paged_v_tile``
    loads with ``other=0.0``): a partial trailing tile then cannot poison a
    block amax — zeros never raise it — and ``nvfp4_scalar_qdq`` guards the
    resulting all-zero blocks. The per-tensor ``global_scale`` (``amax/(6*448)``)
    barely affects V — the dynamic per-16 block amax carries the range and V
    does not saturate E4M3 — so callers may pass the constant ``1.0``.
    """
    tl.static_assert(BLOCK_N % 16 == 0, "BLOCK_N must be divisible by 16 for NVFP4")

    grouped = tl.reshape(v, (BLOCK_N // 16, 16, BLOCK_D))  # 16-key blocks along axis 1 (keys)
    block_amax = tl.expand_dims(tl.max(tl.abs(grouped), axis=1), 1)  # V is signed -> abs
    q = nvfp4_scalar_qdq(grouped, block_amax, global_scale, 16)
    return tl.reshape(q, (BLOCK_N, BLOCK_D))
