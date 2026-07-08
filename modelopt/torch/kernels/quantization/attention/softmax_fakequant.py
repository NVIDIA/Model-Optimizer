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

"""Mixed-FP16 softmax helpers for Triton attention kernels."""

import triton
import triton.language as tl

__all__ = ["SOFTMAX_MODES", "ex2_fp16", "resolve_softmax_mode"]

SOFTMAX_MODES = frozenset(("fp32", "mixed_fp16"))


def resolve_softmax_mode(mode: str) -> bool:
    """Validate ``mode`` and report whether mixed-FP16 softmax is enabled."""
    if mode not in SOFTMAX_MODES:
        raise ValueError(f"softmax_mode must be one of {sorted(SOFTMAX_MODES)}, got {mode!r}")
    return mode == "mixed_fp16"


@triton.jit
def ex2_fp16(x):
    """Evaluate base-2 exponentiation in native FP16 on sm_75 or newer and return FP32."""
    return tl.inline_asm_elementwise(
        "{ .reg .b16 h; cvt.rn.f16.f32 h, $1; ex2.approx.f16 h, h; cvt.f32.f16 $0, h; }",
        "=r,r",
        args=[x],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
