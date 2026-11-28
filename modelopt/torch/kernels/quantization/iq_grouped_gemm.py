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

"""Persistent GGML expert tiles using GPU routing offsets and compact pointer tables."""

import torch
import triton
import triton.language as tl

from modelopt.torch.quantization.ggml import IQ_FORMAT_REGISTRY

from .iq_gemm import (
    _SCALE_TILE_N,
    _TILE_K,
    _TILE_M,
    _TILE_N,
    _forward,
    _input_backward,
    _scale_backward,
)
from .repair import iq_metadata

__all__ = ["iq_grouped_linear"]


@triton.jit
def _grouped(
    X,
    G,
    POINTERS,
    S,
    GRID,
    OFFS,
    BIAS,
    Y,
    ROWS,
    N: tl.constexpr,
    K: tl.constexpr,
    E: tl.constexpr,
    GROUP: tl.constexpr,
    KIND: tl.constexpr,
    BYTES: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BACKWARD: tl.constexpr,
    BE: tl.constexpr,
    BM: tl.constexpr = _TILE_M,
    BN: tl.constexpr = _TILE_N,
    BK: tl.constexpr = _TILE_K,
):
    experts = tl.arange(0, BE)
    ends = tl.load(OFFS + experts, experts < E, 0).to(tl.int64)
    starts = tl.load(OFFS + experts - 1, (experts > 0) & (experts < E), 0).to(tl.int64)
    ends = tl.minimum(tl.maximum(ends, 0), ROWS)
    starts = tl.minimum(tl.maximum(starts, 0), ROWS)
    tile_ends = tl.cumsum(tl.where(experts < E, tl.cdiv(tl.maximum(ends - starts, 0), BM), 0))
    columns: tl.constexpr = tl.cdiv(K, BK) if BACKWARD else tl.cdiv(N, BN)
    total_tiles = tl.sum(tl.where(experts == E - 1, tile_ends, 0)) * columns
    for task in range(tl.program_id(0), total_tiles, tl.num_programs(0)):
        ordinal = task // columns
        expert = tl.sum(((tile_ends <= ordinal) & (experts < E)).to(tl.int32))
        start = tl.sum(tl.where(experts == expert, starts, 0))
        end = tl.sum(tl.where(experts == expert, ends, 0))
        prior = tl.sum(tl.where(experts == expert - 1, tile_ends, 0))
        mi, ni = ordinal - prior, task % columns
        packed = tl.load(POINTERS + expert).to(tl.pointer_type(tl.uint8))
        steps = S + expert.to(tl.int64) * (N * K // GROUP)
        if BACKWARD:
            _input_backward(
                G + start * N,
                packed,
                steps,
                GRID,
                Y + start * K,
                end - start,
                N,
                K,
                GROUP,
                KIND,
                BYTES,
                MI=mi,
                NI=ni,
                EXPLICIT=True,
                BM=BM,
                BN=BN,
                BK=BK,
            )
        else:
            _forward(
                X + start * K,
                packed,
                steps,
                GRID,
                BIAS + expert * N,
                Y + start * N,
                end - start,
                N,
                K,
                GROUP,
                KIND,
                BYTES,
                HAS_BIAS,
                MI=mi,
                NI=ni,
                EXPLICIT=True,
                BM=BM,
                BN=BN,
                BK=BK,
            )


@triton.jit
def _grouped_scales(
    X,
    G,
    POINTERS,
    GRID,
    OFFS,
    DS,
    ROWS,
    N: tl.constexpr,
    K: tl.constexpr,
    GROUP: tl.constexpr,
    KIND: tl.constexpr,
    BYTES: tl.constexpr,
    BM: tl.constexpr = _TILE_M,
    BN: tl.constexpr = _SCALE_TILE_N,
    BK: tl.constexpr = _TILE_K,
):
    expert = tl.program_id(0)
    start = tl.load(OFFS + expert - 1, expert > 0, 0).to(tl.int64)
    end = tl.load(OFFS + expert).to(tl.int64)
    start = tl.minimum(tl.maximum(start, 0), ROWS)
    end = tl.maximum(start, tl.minimum(tl.maximum(end, 0), ROWS))
    packed = tl.load(POINTERS + expert).to(tl.pointer_type(tl.uint8))
    _scale_backward(
        X + start * K,
        G + start * N,
        packed,
        GRID,
        DS + expert.to(tl.int64) * (N * K // GROUP),
        end - start,
        N,
        K,
        GROUP,
        KIND,
        BYTES,
        NI=tl.program_id(1),
        KI=tl.program_id(2),
        EXPLICIT=True,
        BM=BM,
        BN=BN,
        BK=BK,
    )


@triton.jit
def _grouped_bias(G, OFFS, DB, N: tl.constexpr, ROWS):
    expert = tl.program_id(0)
    n = tl.program_id(1) * 64 + tl.arange(0, 64)
    m = tl.arange(0, 32)
    start = tl.load(OFFS + expert - 1, expert > 0, 0).to(tl.int64)
    end = tl.load(OFFS + expert).to(tl.int64)
    start = tl.minimum(tl.maximum(start, 0), ROWS)
    end = tl.maximum(start, tl.minimum(tl.maximum(end, 0), ROWS))
    acc = tl.full((32, 64), 0, tl.float32)
    for first in range(start, end, 32):
        rows = first + m
        acc += tl.load(
            G + rows[:, None] * N + n[None, :], (rows[:, None] < end) & (n[None, :] < N), 0
        ).to(tl.float32)
    tl.store(DB + expert * N + n, tl.sum(acc, 0), n < N)


class _IQGrouped(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, steps, pointers, offsets, name, shape, bias, packeds):
        n, k = shape
        inputs, steps = inputs.contiguous(), steps.contiguous()
        grid, block_bytes = iq_metadata(inputs.device, name)
        group = len(packeds) * n * k // steps.numel()
        # Zero ownerless tails; routing never consumes them and backward must ignore them.
        output = inputs.new_zeros((inputs.shape[0], n))
        meta = {
            "N": n,
            "K": k,
            "ROWS": inputs.shape[0],
            "GROUP": group,
            "KIND": name,
            "BYTES": block_bytes,
        }
        programs = torch.cuda.get_device_properties(inputs.device).multi_processor_count * 2
        common = {
            "E": len(packeds),
            "BE": triton.next_power_of_2(len(packeds)),
        }
        with torch.cuda.device(inputs.device):
            _grouped[(programs,)](
                inputs,
                output,
                pointers,
                steps,
                grid,
                offsets,
                bias if bias is not None else output,
                output,
                HAS_BIAS=bias is not None,
                BACKWARD=False,
                enable_fp_fusion=False,
                **meta,
                **common,
            )
        ctx.save_for_backward(inputs, steps, pointers, offsets, grid, *packeds)
        ctx.meta, ctx.common, ctx.programs = meta, common, programs
        ctx.has_bias = bias is not None
        return output

    @staticmethod
    def backward(ctx, gradient):
        inputs, steps, pointers, offsets, grid, *packeds = ctx.saved_tensors
        gradient = gradient.contiguous()
        dx, ds, db = None, None, None
        if ctx.needs_input_grad[0]:
            dx = torch.zeros_like(inputs)
            with torch.cuda.device(inputs.device):
                _grouped[(ctx.programs,)](
                    inputs,
                    gradient,
                    pointers,
                    steps,
                    grid,
                    offsets,
                    None,
                    dx,
                    HAS_BIAS=False,
                    BACKWARD=True,
                    enable_fp_fusion=False,
                    **ctx.meta,
                    **ctx.common,
                )
        if ctx.needs_input_grad[1]:
            ds = torch.empty_like(steps)
            n, k = ctx.meta["N"], ctx.meta["K"]
            with torch.cuda.device(gradient.device):
                _grouped_scales[
                    (len(packeds), triton.cdiv(n, _SCALE_TILE_N), triton.cdiv(k, _TILE_K))
                ](
                    inputs,
                    gradient,
                    pointers,
                    grid,
                    offsets,
                    ds,
                    enable_fp_fusion=False,
                    **ctx.meta,
                )
        if ctx.has_bias and ctx.needs_input_grad[6]:
            n = ctx.meta["N"]
            db = inputs.new_empty((len(packeds), n))
            with torch.cuda.device(gradient.device):
                _grouped_bias[(len(packeds), triton.cdiv(n, 64))](
                    gradient, offsets, db, n, inputs.shape[0]
                )
        return dx, ds, None, None, None, None, db, None


def iq_grouped_linear(
    inputs,
    packeds,
    steps,
    offsets,
    format_name,
    weight_shape,
    bias=None,
    *,
    packed_pointers,
):
    """Use native expert-sorted rows without host token counts or expanded expert weights.

    Offsets are cumulative, monotonic CUDA expert endpoints from HF routing.
    Bounds are clipped to available rows on device; ordering remains the caller's contract.
    The caller retains ownership of the immutable packed pointer table.
    """
    fmt = IQ_FORMAT_REGISTRY[format_name]
    shape = tuple(weight_shape)
    if (
        not packeds
        or len(shape) != 2
        or shape[1] % fmt.block_size
        or inputs.ndim != 2
        or inputs.shape[1] != shape[1]
        or not inputs.is_cuda
        or inputs.dtype not in (torch.float16, torch.bfloat16)
        or offsets.device != inputs.device
        or offsets.shape != (len(packeds),)
        or offsets.dtype not in (torch.int32, torch.int64)
        or not offsets.is_contiguous()
        or packed_pointers.device != inputs.device
        or packed_pointers.dtype != torch.int64
        or packed_pointers.shape != (len(packeds),)
        or not packed_pointers.is_contiguous()
        or steps.device != inputs.device
        or steps.dtype != torch.float32
        or steps.numel() * fmt.int8_group_size != len(packeds) * shape[0] * shape[1]
        or any(
            p.device != inputs.device
            or p.dtype != torch.uint8
            or not p.is_contiguous()
            or p.numel() != shape[0] * shape[1] // fmt.block_size * fmt.block_bytes
            for p in packeds
        )
    ):
        raise ValueError(
            "Grouped IQ requires matching native CUDA matrices, FP32 scales and integer routing endpoints."
        )
    if bias is not None:
        if (
            bias.shape != (len(packeds), weight_shape[0])
            or bias.device != inputs.device
            or bias.dtype != inputs.dtype
        ):
            raise ValueError(
                "Grouped IQ bias must match the expert/output layout, device and compute dtype."
            )
        bias = bias.contiguous()
    return _IQGrouped.apply(
        inputs,
        steps,
        packed_pointers,
        offsets,
        format_name,
        tuple(weight_shape),
        bias,
        tuple(packeds),
    )
