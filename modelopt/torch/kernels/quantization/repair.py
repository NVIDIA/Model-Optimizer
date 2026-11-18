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

# ruff: noqa: PLR1714 -- Triton constexpr dispatch uses explicit scalar comparisons.

import math

import torch
import triton
import triton.language as tl

from modelopt.torch.quantization.ggml import IQ_FORMAT_REGISTRY
from modelopt.torch.quantization.ggml.iq1_s import iq1_s_grid
from modelopt.torch.quantization.ggml.iq2_s import iq2_s_grid
from modelopt.torch.quantization.ggml.iq2_xs import iq2_xs_grid
from modelopt.torch.quantization.ggml.iq2_xxs import iq2_xxs_grid

__all__ = ["apply_block_scales", "decode_iq_levels", "masked_mse_sum"]

# Codec registration alone does not establish a packed CUDA implementation.
SUPPORTED_FORMATS = frozenset(("iq1_s", "iq1_m", "iq2_xxs", "iq2_xs", "iq2_s"))


@triton.jit
def _mse(P, T, MASK, SUM, N, FEATURES: tl.constexpr, MASKED: tl.constexpr):
    index = tl.program_id(0).to(tl.int64) * 1024 + tl.arange(0, 1024)
    active = index < N
    if MASKED:
        active &= tl.load(MASK + index // FEATURES, index < N, 0)
    difference = tl.load(P + index, active, 0).to(tl.float32) - tl.load(T + index, active, 0).to(
        tl.float32
    )
    tl.store(SUM + tl.program_id(0), tl.sum(difference * difference, 0))


@triton.jit
def _mse_backward(P, T, MASK, G, DP, N, FEATURES: tl.constexpr, MASKED: tl.constexpr):
    index = tl.program_id(0).to(tl.int64) * 1024 + tl.arange(0, 1024)
    active = index < N
    if MASKED:
        active &= tl.load(MASK + index // FEATURES, index < N, 0)
    difference = tl.load(P + index, active, 0).to(tl.float32) - tl.load(T + index, active, 0).to(
        tl.float32
    )
    value = difference * (2 * tl.load(G).to(tl.float32))
    tl.store(DP + index, value, index < N)


class _MaskedMSE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, prediction, target, mask, features):
        prediction, target = prediction.contiguous(), target.contiguous()
        mask = None if mask is None else mask.contiguous()
        partial = torch.empty(
            triton.cdiv(prediction.numel(), 1024), device=prediction.device, dtype=torch.float32
        )
        if partial.numel():
            _mse[(partial.numel(),)](
                prediction,
                target,
                mask,
                partial,
                prediction.numel(),
                features,
                mask is not None,
                enable_fp_fusion=False,
            )
        ctx.save_for_backward(prediction, target, mask)
        ctx.features = features
        return partial.sum()

    @staticmethod
    def backward(ctx, gradient):
        prediction, target, mask = ctx.saved_tensors
        dp = torch.empty_like(prediction)
        if prediction.numel():
            _mse_backward[(triton.cdiv(prediction.numel(), 1024),)](
                prediction,
                target,
                mask,
                gradient,
                dp,
                prediction.numel(),
                ctx.features,
                mask is not None,
                enable_fp_fusion=False,
            )
        return dp, -dp.to(target.dtype) if ctx.needs_input_grad[1] else None, None, None


def masked_mse_sum(prediction, target, mask=None, features=1):
    """Sum valid squared errors while saving only the original narrow activations."""
    return _MaskedMSE.apply(prediction, target, mask, features)


@triton.jit
def _apply(
    levels,
    scales,
    output,
    n,
    group: tl.constexpr,
    block: tl.constexpr,
):
    index = tl.program_id(0).to(tl.int64) * block + tl.arange(0, block)
    valid = index < n
    value = tl.load(levels + index, valid, 0).to(tl.float32)
    scale = tl.load(scales + index // group, valid, 0).to(tl.float32)
    value *= scale
    tl.store(output + index, value, valid)


@triton.jit
def iq_levels(packed, grid, index, valid, kind: tl.constexpr, block_bytes: tl.constexpr):
    """Decode arbitrary element tiles; shared by expanded and packed GEMM paths.

    Mirrors ``modelopt/torch/quantization/ggml/`` modules
    ``iq{1_s,1_m,2_xxs,2_xs,2_s}.py::_unpack_int8``. IQ1's factor 8 is
    ``1 / _IQ1_*_DELTA``, with the signed delta represented by +/-1.
    """
    base, offset = (index // 256) * block_bytes, index % 256
    group, element = offset // 8, offset % 8
    if kind == "iq1_m" or kind == "iq1_s":
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
            if kind == "iq2_xxs":
                subblock = base + 2 + (group // 4) * 8
                entry = tl.load(packed + subblock + group % 4, valid, 0).to(tl.int32)
                auxiliary = tl.load(packed + subblock + 4, valid, 0).to(tl.int32)
                auxiliary |= tl.load(packed + subblock + 5, valid, 0).to(tl.int32) << 8
                auxiliary |= tl.load(packed + subblock + 6, valid, 0).to(tl.int32) << 16
                auxiliary |= tl.load(packed + subblock + 7, valid, 0).to(tl.int32) << 24
                sign_index = (auxiliary >> ((group % 4) * 7)) & 127
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
    return value


@triton.jit
def _decode_iq(
    packed,
    grid,
    output,
    n,
    kind: tl.constexpr,
    block_bytes: tl.constexpr,
    block: tl.constexpr,
):
    index = tl.program_id(0).to(tl.int64) * block + tl.arange(0, block)
    valid = index < n
    value = iq_levels(packed, grid, index, valid, kind, block_bytes)
    tl.store(output + index, value, valid)


def iq_metadata(device, format_name):
    if format_name not in SUPPORTED_FORMATS:
        raise ValueError(f"No native repair-level decoder for {format_name!r}.")
    fmt = IQ_FORMAT_REGISTRY[format_name]
    if fmt.block_size != 256:
        raise ValueError("Native IQ kernels require 256-element blocks.")
    if format_name in ("iq1_m", "iq1_s"):
        grid = iq1_s_grid(device)
    elif format_name == "iq2_s":
        grid = iq2_s_grid(device)
    elif format_name == "iq2_xs":
        grid = iq2_xs_grid(device)
    else:
        grid = iq2_xxs_grid(device)
    return grid, fmt.block_bytes


def decode_iq_levels(packed, format_name, shape):
    """Decode immutable GGML levels in one native launch, without reference chunks."""
    grid, block_bytes = iq_metadata(packed.device, format_name)
    fmt = IQ_FORMAT_REGISTRY[format_name]
    elements = math.prod(shape)
    if (
        not packed.is_cuda
        or packed.dtype != torch.uint8
        or elements % fmt.block_size
        or packed.numel() != elements // fmt.block_size * block_bytes
    ):
        raise ValueError("Native IQ decode requires CUDA bytes matching complete native blocks.")
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
            levels,
            scales,
            output,
            levels.numel(),
            group,
            1024,
            enable_fp_fusion=False,
        )
        ctx.save_for_backward(levels)
        ctx.scale_shape, ctx.scale_dtype, ctx.group = scales.shape, scales.dtype, group
        return output

    @staticmethod
    def backward(ctx, gradient):
        (levels,) = ctx.saved_tensors
        # Match native reconstruction's broadcast-multiply reduction, including
        # its projection-dtype dW before conversion to FP32. Bound the temporary
        # product without changing the final group reduction dimension.
        if not ctx.needs_input_grad[1]:
            return None, None, None
        shape = (-1, ctx.scale_shape[-1], ctx.group)
        gradient, levels = gradient.reshape(shape), levels.reshape(shape)
        result = torch.empty(ctx.scale_shape, device=levels.device, dtype=ctx.scale_dtype)
        rows = result.reshape(-1, ctx.scale_shape[-1])
        chunk_size = max(1, 1048576 // (ctx.scale_shape[-1] * ctx.group))
        for start in range(0, len(rows), chunk_size):
            chunk = slice(start, start + chunk_size)
            product = gradient[chunk].float() * levels[chunk].float()
            rows[chunk] = product.sum(dim=-1, keepdim=True).reshape_as(rows[chunk])
        return None, result, None


def apply_block_scales(levels, scales, dtype):
    """Decode projected weights without materializing a full FP32 level tensor.

    Autograd retains integer levels and reduces scale gradients directly. Native
    code selection and scale-field projection remain owned by the format adapter.
    """
    if not levels.is_cuda or levels.device != scales.device:
        raise ValueError("Integer levels and block scales must share a CUDA device.")
    return _BlockScales.apply(levels, scales, dtype)
