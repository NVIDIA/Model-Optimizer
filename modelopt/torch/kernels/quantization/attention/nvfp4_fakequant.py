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

# ruff: noqa: N803 — Triton kernels use uppercase for constexpr by convention

"""In-kernel NVFP4 (E2M1) fake quantization for attention BMM operands.

Tile-level wrappers over the scalar FP4 device functions in
``kernels/quantization/gemm/nvfp4_quant.py`` (the single source of truth for the
E2M1 decision-boundary rounding). They quantize a Triton tile to NVFP4 along its
*contraction* axis in groups of ``BLK`` (= 16, the NVFP4 block size): a per-block
FP8-E4M3 scale plus a per-tile global scale (``global_amax / (6 * 448)``), exactly
as ModelOpt's NVFP4 quantizer does.

This is *fake* quantization: values are rounded to the NVFP4 grid and dequantized
back to fp32, so the downstream ``tl.dot`` still runs in bf16/fp32 — the intent is
to measure the *accuracy* impact of NVFP4 attention BMMs, not to accelerate them.

The per-block E4M3 scale is computed with an exact fp32 emulation of E4M3 (rather
than ``tl.float8e4nv``) so the kernel runs on pre-sm_89 GPUs as well; the emulation
is bit-exact to ``torch.float8_e4m3fn``. The global scale is computed per tile
(dynamic), which captures the dominant FP4-rounding error; a calibrated per-tensor
global scale is a follow-up (would be passed in).
"""

import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

from modelopt.torch.kernels.quantization.gemm.nvfp4_quant import nvfp4_scalar_quant

# NVFP4 second-level normalizer: E2M1 max (6.0) * FP8-E4M3 max (448.0). Must use the
# tl.constexpr(...) instantiation form so Triton @jit functions can read it as a global.
_NVFP4_GLOBAL_NORM = tl.constexpr(6.0 * 448.0)


@triton.jit
def e4m3_emulate(x):
    """Round ``x`` to the nearest E4M3 (E4M3fn, max 448) value in fp32.

    Hardware-free equivalent of ``x.to(float8_e4m3fn).to(float32)`` — bit-exact to
    ``torch.float8_e4m3fn`` over the representable range, so the NVFP4 block scale
    can be quantized on GPUs without native fp8 (pre-sm_89).
    """
    ax = tl.minimum(tl.abs(x), 448.0)
    nz = ax > 0.0
    safe = tl.where(nz, ax, 1.0)  # avoid log2(0)
    exp = tl.maximum(tl.floor(tl.log2(safe)), -6.0)  # clamp to min normal exponent
    step = tl.exp2(exp - 3.0)  # 3 mantissa bits -> ulp = 2^(exp-3)
    q = libdevice.nearbyint(safe / step) * step  # round-half-to-even (matches torch)
    q = tl.minimum(tl.where(nz, q, 0.0), 448.0)
    return tl.where(x >= 0, q, -q)


@triton.jit
def _nvfp4_block_scale(block_amax, global_scale):
    """Per-block NVFP4 scale: E4M3-quantized ``block_amax/6`` with the global scale."""
    scale_in_fp8_range = tl.minimum(block_amax / (6.0 * global_scale), 448.0)
    return e4m3_emulate(scale_in_fp8_range) * global_scale


@triton.jit
def nvfp4_qdq_1d(x, NB: tl.constexpr, BLK: tl.constexpr):
    """NVFP4 fake-quantize a 1-D tile ``x`` of length ``NB*BLK`` along its only axis."""
    global_scale = tl.max(tl.abs(x)) / _NVFP4_GLOBAL_NORM
    xr = tl.reshape(x, (NB, BLK))
    block_amax = tl.max(tl.abs(xr), axis=1, keep_dims=True)  # [NB, 1]
    scale = _nvfp4_block_scale(block_amax, global_scale)  # [NB, 1], broadcasts over BLK
    xq = nvfp4_scalar_quant(xr, scale, BLK)
    return tl.reshape(xq, (NB * BLK,))


@triton.jit
def nvfp4_qdq_lastdim_2d(x, M: tl.constexpr, NB: tl.constexpr, BLK: tl.constexpr):
    """NVFP4 fake-quantize a 2-D tile ``x`` ``[M, NB*BLK]`` along axis 1 (contraction).

    Used for operands whose contraction dimension is the column dimension, e.g. the
    query tile ``[BLOCK_M, head_dim]`` (contract over head_dim) and the probability
    tile ``[BLOCK_M, BLOCK_N]`` (contract over BLOCK_N) — each row is quantized
    independently in blocks of ``BLK`` columns.
    """
    global_scale = tl.max(tl.abs(x)) / _NVFP4_GLOBAL_NORM
    xr = tl.reshape(x, (M, NB, BLK))
    block_amax = tl.max(tl.abs(xr), axis=2, keep_dims=True)  # [M, NB, 1]
    scale = _nvfp4_block_scale(block_amax, global_scale)  # [M, NB, 1], broadcasts over BLK
    xq = nvfp4_scalar_quant(xr, scale, BLK)
    return tl.reshape(xq, (M, NB * BLK))


@triton.jit
def nvfp4_qdq_lastdim_2d_perrow(x, M: tl.constexpr, NB: tl.constexpr, BLK: tl.constexpr):
    """NVFP4 fake-quantize ``[M, NB*BLK]`` along axis 1 with a *per-row* global scale.

    Used for the softmax probabilities P: a flash kernel quantizes the unnormalized
    ``exp`` and divides the output by the row sum afterward. Because NVFP4 is
    homogeneous (``NVFP4(c*x) = c*NVFP4(x)``), a *per-row* global makes that equal to
    quantizing the normalized P — but only with a per-row (per-query-token) global,
    since each query row has its own normalizer. (A per-tile global would mix rows
    with different row-sums and not commute with the deferred normalization.)
    """
    row_amax = tl.max(tl.abs(x), axis=1, keep_dims=True)  # [M, 1] per-row global
    global_scale = row_amax / _NVFP4_GLOBAL_NORM
    xr = tl.reshape(x, (M, NB, BLK))
    block_amax = tl.max(tl.abs(xr), axis=2, keep_dims=True)  # [M, NB, 1]
    scale = _nvfp4_block_scale(block_amax, global_scale[:, :, None])  # broadcasts over NB, BLK
    xq = nvfp4_scalar_quant(xr, scale, BLK)
    return tl.reshape(xq, (M, NB * BLK))


@triton.jit
def nvfp4_qdq_axis0(x, NB: tl.constexpr, BLK: tl.constexpr, M: tl.constexpr):
    """NVFP4 fake-quantize a 2-D tile ``x`` ``[NB*BLK, M]`` along axis 0 (contraction).

    Used for operands whose contraction dimension is the row dimension, e.g. K^T
    ``[head_dim, BLOCK_N]`` (contract over head_dim) and V ``[BLOCK_N, head_dim]``
    (contract over BLOCK_N) — each column is quantized independently in blocks of
    ``BLK`` rows.
    """
    global_scale = tl.max(tl.abs(x)) / _NVFP4_GLOBAL_NORM
    xr = tl.reshape(x, (NB, BLK, M))
    block_amax = tl.max(tl.abs(xr), axis=1, keep_dims=True)  # [NB, 1, M]
    scale = _nvfp4_block_scale(block_amax, global_scale)  # [NB, 1, M], broadcasts over BLK
    xq = nvfp4_scalar_quant(xr, scale, BLK)
    return tl.reshape(xq, (NB * BLK, M))
