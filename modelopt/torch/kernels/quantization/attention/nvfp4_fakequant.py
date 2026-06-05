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

# ruff: noqa: N803, N806 — Triton kernels use uppercase for constexpr by convention

"""In-kernel FP4 fake quantization for attention BMM operands.

**Design-aligned with the customized vLLM** ``mni/attnOpt`` attention-quant
framework (``vllm/v1/attention/ops/triton_quant/triton_quant_utils.py``): same
``_fake_quant_fp4_k1`` / ``_fake_quant_fp4_k0`` signatures, the same per-tensor
``global_scale`` convention (computed host-side by :func:`tensor_global_scale`,
the mirror of their ``tensor_scale``), the same block-scale formula
(``absmax/6 + 1e-30``), the same E2M1 ``>=`` decision boundaries, and the same
``SCALE_TYPE`` enum. So our NVFP4 numerics match theirs bit-for-bit.

Two intentional portability tweaks (numerically identical to theirs):
  - ``A-side``/``B-side`` use ``k1``/``k0`` for Q,P / K,V (GEMM-K = axis 1 / 0).
  - the E4M3 per-block scale is computed with an fp32 E4M3 emulation that is
    bit-exact to ``torch.float8_e4m3fn`` (== their ``.to(tl.float8e4nv)``), so the
    NVFP4 path also runs on pre-sm_89 GPUs where ``tl.float8e4nv`` is unsupported;
    the E2M1 value rounding uses their PTX ``cvt`` on sm_100 and the portable
    ``>=`` fallback elsewhere.

This is *fake* quantization: operands are rounded to the FP4 grid and dequantized
to fp32; the ``tl.dot`` still runs in bf16/fp32 (accuracy study, not speed).
"""

import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice
from triton.language.target_info import cuda_capability_geq

# SCALE_TYPE enum — values mirror mni/attnOpt's triton_quant_utils for parity.
QUANT_NVFP4 = tl.constexpr(2)  # FP4 E2M1, E4M3 per-block scale x FP32 global (true NVFP4)
QUANT_MXFP4_RUP = tl.constexpr(3)  # FP4 E2M1, E8M0 per-block scale, round up
QUANT_MXFP4_RNE = tl.constexpr(4)  # FP4 E2M1, E8M0 per-block scale, round to nearest
QUANT_NVFP4_SFP32 = tl.constexpr(6)  # FP4 E2M1, full FP32 per-block scale (no scale quant)

E2M1_MAX = 6.0
FP8_E4M3_MAX = 448.0


def tensor_global_scale(tensor, scale_type: int = 2) -> float:
    """Per-tensor NVFP4 global scale ``amax / (6*448)`` (mirror of their ``tensor_scale``).

    Computed host-side and passed into the kernels as a constant. ``scale_type`` is
    accepted for signature parity; only NVFP4/SFP32 use a global (others ignore it).
    """
    return tensor.float().abs().max().item() / (E2M1_MAX * FP8_E4M3_MAX) + 1e-30


@triton.jit
def e4m3_emulate(x):
    """fp32 round to nearest E4M3 (E4M3fn, max 448) — bit-exact to torch.float8_e4m3fn.

    Portable equivalent of ``x.to(tl.float8e4nv).to(tl.float32)`` for the per-block
    scale, so the NVFP4 path runs on GPUs without native fp8 (pre-sm_89).
    """
    ax = tl.minimum(tl.abs(x), 448.0)
    nz = ax > 0.0
    safe = tl.where(nz, ax, 1.0)
    exp = tl.maximum(tl.floor(tl.log2(safe)), -6.0)
    step = tl.exp2(exp - 3.0)
    q = libdevice.nearbyint(safe / step) * step
    q = tl.minimum(tl.where(nz, q, 0.0), 448.0)
    return tl.where(x >= 0, q, -q)


@triton.jit
def _round_e2m1(x_s):
    """Round x_s (= value / scale) to the nearest FP4 E2M1 magnitude (signed).

    PTX ``cvt.rn.satfinite.e2m1x2.f32`` on sm_100+ (matches mni/attnOpt); portable
    ``>=`` decision-boundary fallback elsewhere (identical grid + tie direction).
    """
    if cuda_capability_geq(10, 0):
        return tl.inline_asm_elementwise(
            """{
                .reg .b8 e2m1;
                .reg .b32 f16x2;
                .reg .b16 lo, hi;
                cvt.rn.satfinite.e2m1x2.f32 e2m1, $2, $3;
                cvt.rn.f16x2.e2m1x2 f16x2, e2m1;
                mov.b32 {lo, hi}, f16x2;
                cvt.f32.f16 $0, hi;
                cvt.f32.f16 $1, lo;
            }""",
            "=r,=r,r,r",
            args=[x_s],
            dtype=tl.float32,
            is_pure=True,
            pack=2,
        )
    a = tl.abs(x_s)
    sgn = tl.where(x_s >= 0.0, 1.0, -1.0)
    return sgn * tl.where(
        a >= 5.0,
        6.0,
        tl.where(
            a >= 3.5,
            4.0,
            tl.where(
                a >= 2.5,
                3.0,
                tl.where(
                    a >= 1.75,
                    2.0,
                    tl.where(
                        a >= 1.25, 1.5, tl.where(a >= 0.75, 1.0, tl.where(a >= 0.25, 0.5, 0.0))
                    ),
                ),
            ),
        ),
    )


@triton.jit
def _block_scale(scale_f32, global_scale, SCALE_TYPE: tl.constexpr):
    """Quantize the per-block scale per SCALE_TYPE (mirror of mni/attnOpt)."""
    if SCALE_TYPE == QUANT_NVFP4:
        # E4M3 block scale relative to the FP32 global (emulated E4M3 == fp8e4nv).
        return e4m3_emulate(scale_f32 / global_scale) * global_scale
    elif SCALE_TYPE == QUANT_MXFP4_RUP:
        return tl.exp2(tl.ceil(tl.log2(scale_f32)))
    elif SCALE_TYPE == QUANT_MXFP4_RNE:
        return tl.exp2(tl.floor(tl.log2(scale_f32) + 0.5))
    else:  # QUANT_NVFP4_SFP32: full fp32 per-block scale
        return scale_f32


@triton.jit
def fake_quant_fp4_k1(
    x,
    M: tl.constexpr,
    K: tl.constexpr,
    K_BLOCK: tl.constexpr,
    global_scale,
    SCALE_TYPE: tl.constexpr,
):
    """Fake-quantize ``x [M, K]`` to FP4 E2M1, 1×K_BLOCK blocks along K (axis 1).

    A-side operands (Q, P) where the GEMM-K dim is axis 1. Per-tensor ``global_scale``.
    """
    NG: tl.constexpr = K // K_BLOCK
    x_r = tl.reshape(x.to(tl.float32), (M, NG, K_BLOCK))
    scale_f32 = tl.max(tl.abs(x_r), axis=2) / 6.0 + 1e-30  # [M, NG]
    scale = _block_scale(scale_f32, global_scale, SCALE_TYPE)
    scale = tl.broadcast_to(tl.expand_dims(scale, 2), (M, NG, K_BLOCK))
    x_q = _round_e2m1(x_r / scale)
    return tl.reshape(x_q * scale, (M, K)).to(x.dtype)


@triton.jit
def fake_quant_fp4_k0(
    x,
    K: tl.constexpr,
    N: tl.constexpr,
    K_BLOCK: tl.constexpr,
    N_BLOCK: tl.constexpr,
    global_scale,
    SCALE_TYPE: tl.constexpr,
):
    """Fake-quantize ``x [K, N]`` to FP4 E2M1, K_BLOCK×N_BLOCK blocks.

    B-side operands (K^T, V) where the GEMM-K dim is axis 0. ``N_BLOCK=1`` for NVFP4.
    """
    KG: tl.constexpr = K // K_BLOCK
    NG: tl.constexpr = N // N_BLOCK
    x_r = tl.reshape(x.to(tl.float32), (KG, K_BLOCK, NG, N_BLOCK))
    scale_f32 = tl.max(tl.max(tl.abs(x_r), axis=3), axis=1) / 6.0 + 1e-30  # [KG, NG]
    scale = _block_scale(scale_f32, global_scale, SCALE_TYPE)
    scale = tl.broadcast_to(tl.expand_dims(tl.expand_dims(scale, 1), 3), (KG, K_BLOCK, NG, N_BLOCK))
    x_q = _round_e2m1(x_r / scale)
    return tl.reshape(x_q * scale, (K, N)).to(x.dtype)
