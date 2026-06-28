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

"""Value-operand (V) quant-dequant helper for flash attention.

The in-kernel counterpart of ``v_bmm_quantizer``: :func:`_v_qdq_nvfp4` fake-quantizes
the ``V`` operand of the ``P @ V`` matmul (BMM2). It is the sibling of
``attention/p_qdq._p_qdq_nvfp4`` (the ``P`` operand) — both block 16 along the key
dimension, the contraction axis of ``P @ V``, which a per-token cache write cannot
produce. V's keys axis is axis 0 of the loaded tile and V is signed, so unlike P its
block amax uses ``abs``. Called under the ``V_QDQ`` constexpr guard from the baseline
flash-attention kernel (``common/attention/triton_fa.py``) and the paged decode kernel
(``common/attention/decode_attention.py``); per-tensor FP8 uses
``quantization/common/fp8_quant.fp8_scalar_qdq`` directly.
"""

import triton
import triton.language as tl

from modelopt.torch.kernels.quantization.common.nvfp4_quant import nvfp4_scalar_qdq


@triton.jit
def _v_qdq_nvfp4(
    v,
    global_scale,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """NVFP4 fake quant-dequant of the value operand ``V`` of ``P @ V`` (BMM2).

    Two-level NVFP4 scaling like ``p_qdq._p_qdq_nvfp4``, but ``V``'s blocks of 16 run
    along the *key* dimension — which for the loaded tile ``v [BLOCK_N keys, BLOCK_D
    head_dim]`` is axis 0 — so 16 contiguous keys share a scale, per head-dim channel
    (the contraction axis of ``P @ V``). Unlike ``P``, ``V`` is signed, so the block
    amax uses ``abs``.

    Relies on the caller masking out-of-range keys to 0 (``_load_paged_v_tile`` loads
    with ``other=0.0``): a partial trailing tile then cannot poison a block amax — zeros
    never raise it — and ``nvfp4_scalar_qdq`` guards the resulting all-zero blocks. The
    per-tensor ``global_scale`` (``amax/(6*448)``) barely affects V — the dynamic per-16
    block amax carries the range and V does not saturate E4M3 — so callers may pass the
    constant ``1.0``.
    """
    tl.static_assert(BLOCK_N % 16 == 0, "BLOCK_N must be divisible by 16 for NVFP4")

    grouped = tl.reshape(v, (BLOCK_N // 16, 16, BLOCK_D))  # 16-key blocks along axis 1 (keys)
    block_amax = tl.expand_dims(tl.max(tl.abs(grouped), axis=1), 1)  # V is signed -> abs
    q = nvfp4_scalar_qdq(grouped, block_amax, global_scale, 16)
    return tl.reshape(q, (BLOCK_N, BLOCK_D))
