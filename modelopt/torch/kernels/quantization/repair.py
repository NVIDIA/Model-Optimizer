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

"""Fused application/reduction of immutable integer levels and learned block scales."""

import math

import torch
import triton
import triton.language as tl

__all__ = ["apply_block_scales", "decode_iq_levels"]


@triton.jit
def _apply(levels, scales, output, n: tl.constexpr, group: tl.constexpr, block: tl.constexpr):
    index = tl.program_id(0).to(tl.int64) * block + tl.arange(0, block)
    valid = index < n
    value = tl.load(levels + index, valid, 0).to(tl.float32)
    scale = tl.load(scales + index // group, valid, 0).to(tl.float32)
    tl.store(output + index, value * scale, valid)


@triton.jit
def _reduce(
    gradient, levels, output, count: tl.constexpr, group: tl.constexpr, groups: tl.constexpr
):
    rows = tl.program_id(0).to(tl.int64) * groups + tl.arange(0, groups)
    indices = rows[:, None] * group + tl.arange(0, group)[None, :]
    valid = rows[:, None] < count
    grad = tl.load(gradient + indices, valid, 0).to(tl.float32)
    value = tl.load(levels + indices, valid, 0).to(tl.float32)
    result = tl.sum(grad * value, axis=1)
    tl.store(output + rows, result, rows < count)


@triton.jit
def _decode_iq(
    packed,
    grid,
    output,
    n: tl.constexpr,
    kind: tl.constexpr,
    block_bytes: tl.constexpr,
    block: tl.constexpr,
):
    index = tl.program_id(0).to(tl.int64) * block + tl.arange(0, block)
    valid = index < n
    base, offset = (index // 256) * block_bytes, index % 256
    group, element = offset // 8, offset % 8
    if kind == "iq1_m" or kind == "iq1_s":  # noqa: PLR1714 - Triton constexpr dispatch.
        if kind == "iq1_s":
            low = tl.load(packed + base + 2 + group, valid, 0).to(tl.int32)
            word_offset = base + 34 + (group // 4) * 2
            word = tl.load(packed + word_offset, valid, 0).to(tl.int32)
            word |= tl.load(packed + word_offset + 1, valid, 0).to(tl.int32) << 8
            entry = low | (((word >> ((group % 4) * 3)) & 7) << 8)
            negative_delta = (word & 32768) != 0
        else:
            low = tl.load(packed + base + group, valid, 0).to(tl.int32)
            high = tl.load(packed + base + 32 + group // 2, valid, 0).to(tl.int32)
            nibble = (high >> ((group % 2) * 4)) & 15
            entry = low | ((nibble & 7) << 8)
            negative_delta = (nibble & 8) != 0
        value = tl.load(grid + entry * 8 + element).to(tl.int32) * 8
        value += tl.where(negative_delta, -1, 1)
    else:
        if kind == "iq2_s":
            low = tl.load(packed + base + 2 + group, valid, 0).to(tl.int32)
            high = tl.load(packed + base + 66 + group // 4, valid, 0).to(tl.int32)
            entry = low | (((high >> ((group % 4) * 2)) & 3) << 8)
            signs = tl.load(packed + base + 34 + group, valid, 0).to(tl.int32)
        else:
            low = tl.load(packed + base + 2 + 2 * group, valid, 0).to(tl.int32)
            high = tl.load(packed + base + 3 + 2 * group, valid, 0).to(tl.int32)
            code = low | (high << 8)
            entry, sign_index = code & 511, code >> 9
            parity = sign_index ^ (sign_index >> 4)
            parity ^= parity >> 2
            parity ^= parity >> 1
            signs = sign_index | ((parity & 1) << 7)
        value = tl.load(grid + entry * 8 + element).to(tl.int32)
        value *= tl.where(((signs >> element) & 1) != 0, -1, 1)
    tl.store(output + index, value, valid)


def decode_iq_levels(packed, format_name, shape):
    """Decode immutable IQ levels in one native launch, without reference chunks."""
    if format_name in ("iq1_m", "iq1_s"):
        from modelopt.torch.quantization.ggml.iq1_s import iq1_s_grid

        grid = iq1_s_grid(packed.device)
        block_bytes = 56 if format_name == "iq1_m" else 50
    elif format_name == "iq2_s":
        from modelopt.torch.quantization.ggml.iq2_s import iq2_s_grid

        grid, block_bytes = iq2_s_grid(packed.device), 82
    elif format_name == "iq2_xs":
        from modelopt.torch.quantization.ggml.iq2_xs import iq2_xs_grid

        grid, block_bytes = iq2_xs_grid(packed.device), 74
    else:
        raise ValueError(f"No native repair-level decoder for {format_name!r}.")
    elements = math.prod(shape)
    if (
        not packed.is_cuda
        or packed.dtype != torch.uint8
        or elements % 256
        or packed.numel() != elements // 256 * block_bytes
    ):
        raise ValueError("Native IQ decode requires CUDA bytes matching complete 256-value blocks.")
    packed = packed.contiguous()
    output = torch.empty(shape, device=packed.device, dtype=torch.int8)
    _decode_iq[(triton.cdiv(output.numel(), 1024),)](
        packed, grid, output, output.numel(), format_name, block_bytes, 1024
    )
    return output


class _BlockScales(torch.autograd.Function):
    @staticmethod
    def forward(ctx, levels, scales, dtype):
        levels, scales = levels.contiguous(), scales.contiguous()
        group, remainder = divmod(levels.numel(), scales.numel())
        if remainder or group < 1 or group & (group - 1):
            raise ValueError(
                "Scale groups must evenly partition immutable levels in powers of two."
            )
        output = torch.empty(levels.shape, device=levels.device, dtype=dtype)
        _apply[(triton.cdiv(levels.numel(), 1024),)](
            levels, scales, output, levels.numel(), group, 1024, enable_fp_fusion=False
        )
        ctx.save_for_backward(levels)
        ctx.scale_shape, ctx.scale_dtype, ctx.group = scales.shape, scales.dtype, group
        return output

    @staticmethod
    def backward(ctx, gradient):
        (levels,) = ctx.saved_tensors
        gradient = gradient.contiguous()
        count = levels.numel() // ctx.group
        result = torch.empty(ctx.scale_shape, device=levels.device, dtype=ctx.scale_dtype)
        groups = max(1, 256 // ctx.group)
        _reduce[(triton.cdiv(count, groups),)](
            gradient, levels, result, count, ctx.group, groups, enable_fp_fusion=False
        )
        return None, result, None


def apply_block_scales(levels, scales, dtype):
    """Decode projected weights without materializing a full FP32 level tensor.

    Autograd retains integer levels and reduces scale gradients directly. Native
    code selection and scale-field projection remain owned by the format adapter.
    """
    return _BlockScales.apply(levels, scales, dtype)
