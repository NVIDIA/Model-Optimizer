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

"""Softmax-datapath fake quantization for attention (mixed-precision softmax).

Design-aligned, mode-value-identical with the customized vLLM ``mni/attnOpt``
framework: the softmax internal datapath can be rounded to FP16/BF16 (RNE/RZ, with
optional flush-to-zero) at separate points — **DIFF** (input to exp2), **EXP2**
(output of exp2), and **ACC** (the running sum) — to model the hardware MUFU/PTX
softmax precision rather than the conservative FP32 reference.

The per-mode rounding functions are bit-exact copies of theirs (same PTX
``cvt.rz``/``cvt.rn.ftz`` and Triton casts), so a "mixed-FP16 softmax" run here
matches theirs at each point. Mode integer values mirror their enum exactly.
"""

import triton
import triton.language as tl

# Mode enum — values identical to mni/attnOpt triton_quant_utils (QUANT_NONE / SMQUANT_*).
SMQUANT_NONE = tl.constexpr(0)
SMQUANT_BF16_RZ = tl.constexpr(9)
SMQUANT_FP16_RZ = tl.constexpr(10)
SMQUANT_BF16_RNE = tl.constexpr(12)
SMQUANT_FP16_RNE = tl.constexpr(13)
SMQUANT_FP16_RNE_FTZ = tl.constexpr(17)
SMQUANT_FP16_RZ_FTZ = tl.constexpr(18)
SMQUANT_BF16_RNE_FTZ = tl.constexpr(19)
SMQUANT_BF16_RZ_FTZ = tl.constexpr(20)

# Python-side string -> mode map (mirror of their _SMQUANT_ALL_MODES subset).
SOFTMAX_MODE_MAP = {
    None: 0,
    "none": 0,
    "fp32": 0,
    "bypass": 0,
    "bf16_rz": 9,
    "fp16_rz": 10,
    "bf16": 12,
    "fp16": 13,
    "fp16_ftz": 17,
    "fp16_rz_ftz": 18,
    "bf16_ftz": 19,
    "bf16_rz_ftz": 20,
    # convenience aliases
    "fp16_rne": 13,
    "bf16_rne": 12,
}


@triton.jit
def _rz_bf16(x):
    return tl.inline_asm_elementwise(
        "{ .reg .b16 b; cvt.rz.bf16.f32 b, $1; cvt.f32.bf16 $0, b; }",
        "=r,r",
        args=[x],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _rz_fp16(x):
    return tl.inline_asm_elementwise(
        "{ .reg .b16 h; cvt.rz.f16.f32 h, $1; cvt.f32.f16 $0, h; }",
        "=r,r",
        args=[x],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _rne_bf16_ftz(x):
    return tl.inline_asm_elementwise(
        "{ .reg .b16 b; cvt.rn.ftz.bf16.f32 b, $1; cvt.f32.bf16 $0, b; }",
        "=r,r",
        args=[x],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _rz_bf16_ftz(x):
    return tl.inline_asm_elementwise(
        "{ .reg .b16 b; cvt.rz.ftz.bf16.f32 b, $1; cvt.f32.bf16 $0, b; }",
        "=r,r",
        args=[x],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _rne_fp16_ftz(x):
    return tl.inline_asm_elementwise(
        "{ .reg .b16 h; cvt.rn.ftz.f16.f32 h, $1; cvt.f32.f16 $0, h; }",
        "=r,r",
        args=[x],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _rz_fp16_ftz(x):
    return tl.inline_asm_elementwise(
        "{ .reg .b16 h; cvt.rz.ftz.f16.f32 h, $1; cvt.f32.f16 $0, h; }",
        "=r,r",
        args=[x],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def softmax_round(x, MODE: tl.constexpr):
    """Round ``x`` per the softmax-datapath MODE (no-op for MODE==0/NONE)."""
    if MODE == SMQUANT_FP16_RNE:
        return (x.to(tl.float16)).to(tl.float32)  # Triton RNE cast
    elif MODE == SMQUANT_BF16_RNE:
        return (x.to(tl.bfloat16)).to(tl.float32)
    elif MODE == SMQUANT_FP16_RZ:
        return _rz_fp16(x)
    elif MODE == SMQUANT_BF16_RZ:
        return _rz_bf16(x)
    elif MODE == SMQUANT_FP16_RNE_FTZ:
        return _rne_fp16_ftz(x)
    elif MODE == SMQUANT_FP16_RZ_FTZ:
        return _rz_fp16_ftz(x)
    elif MODE == SMQUANT_BF16_RNE_FTZ:
        return _rne_bf16_ftz(x)
    elif MODE == SMQUANT_BF16_RZ_FTZ:
        return _rz_bf16_ftz(x)
    else:
        return x  # NONE / fp32 — no rounding


def resolve_softmax_mode(mode) -> int:
    """Map a softmax-quant mode string (or None) to its integer enum value."""
    if isinstance(mode, int):
        return mode
    if mode not in SOFTMAX_MODE_MAP:
        valid = sorted(k for k in SOFTMAX_MODE_MAP if isinstance(k, str))
        raise ValueError(f"Unknown softmax quant mode {mode!r}; choose from {valid}")
    return SOFTMAX_MODE_MAP[mode]
