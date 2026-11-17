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

from .iq_gemm import _forward, _input_backward, _scale_backward
from .repair import iq_metadata

__all__ = ["iq_grouped_linear"]


@triton.jit
def _grouped(
    X,
    G,
    POINTERS,
    S,
    WOFFS,
    GRID,
    OFFS,
    BIAS,
    Y,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    E: tl.constexpr,
    GROUP: tl.constexpr,
    KIND: tl.constexpr,
    BYTES: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BACKWARD: tl.constexpr,
    BE: tl.constexpr,
    HAS_OFFSETS: tl.constexpr,
):
    experts = tl.arange(0, BE)
    ends = tl.load(OFFS + experts, experts < E, 0).to(tl.int64)
    starts = tl.load(OFFS + experts - 1, (experts > 0) & (experts < E), 0).to(tl.int64)
    tile_ends = tl.cumsum(tl.where(experts < E, tl.cdiv(ends - starts, 32), 0))
    columns: tl.constexpr = tl.cdiv(K, 64) if BACKWARD else tl.cdiv(N, 32)
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
        weight_offsets = WOFFS + expert.to(tl.int64) * (N * K // GROUP) if HAS_OFFSETS else WOFFS
        if BACKWARD:
            _input_backward(
                G + start * N,
                packed,
                steps,
                weight_offsets,
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
                HAS_OFFSETS=HAS_OFFSETS,
            )
        else:
            _forward(
                X + start * K,
                packed,
                steps,
                weight_offsets,
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
                HAS_OFFSETS=HAS_OFFSETS,
            )


@triton.jit
def _grouped_scales(
    X,
    G,
    POINTERS,
    GRID,
    OFFS,
    DS,
    N: tl.constexpr,
    K: tl.constexpr,
    GROUP: tl.constexpr,
    KIND: tl.constexpr,
    BYTES: tl.constexpr,
):
    expert = tl.program_id(0)
    start = tl.load(OFFS + expert - 1, expert > 0, 0).to(tl.int64)
    end = tl.load(OFFS + expert).to(tl.int64)
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
    )


@triton.jit
def _grouped_bias(G, OFFS, DB, N: tl.constexpr):
    expert = tl.program_id(0)
    n = tl.program_id(1) * 64 + tl.arange(0, 64)
    m = tl.arange(0, 32)
    start = tl.load(OFFS + expert - 1, expert > 0, 0).to(tl.int64)
    end = tl.load(OFFS + expert).to(tl.int64)
    acc = tl.full((32, 64), 0, tl.float32)
    for first in range(start, end, 32):
        rows = first + m
        acc += tl.load(
            G + rows[:, None] * N + n[None, :], (rows[:, None] < end) & (n[None, :] < N), 0
        ).to(tl.float32)
    tl.store(DB + expert * N + n, tl.sum(acc, 0), n < N)


class _IQGrouped(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, steps, pointers, offsets, name, shape, bias, packeds, weight_offsets):
        n, k = shape
        inputs, steps = inputs.contiguous(), steps.contiguous()
        if weight_offsets is not None:
            weight_offsets = weight_offsets.contiguous()
        grid, block_bytes = iq_metadata(inputs.device, name)
        group = len(packeds) * n * k // steps.numel()
        # Zero ownerless tails; routing never consumes them and backward must ignore them.
        output = inputs.new_zeros((inputs.shape[0], n))
        meta = {"N": n, "K": k, "GROUP": group, "KIND": name, "BYTES": block_bytes}
        programs = torch.cuda.get_device_properties(inputs.device).multi_processor_count * 2
        common = {
            "E": len(packeds),
            "BE": triton.next_power_of_2(len(packeds)),
            "M": inputs.shape[0],
        }
        _grouped[(programs,)](
            inputs,
            output,
            pointers,
            steps,
            weight_offsets,
            grid,
            offsets,
            bias if bias is not None else output,
            output,
            HAS_BIAS=bias is not None,
            BACKWARD=False,
            HAS_OFFSETS=weight_offsets is not None,
            enable_fp_fusion=False,
            **meta,
            **common,
        )
        ctx.save_for_backward(inputs, steps, pointers, offsets, grid, weight_offsets, *packeds)
        ctx.has_offsets = weight_offsets is not None
        ctx.meta, ctx.common, ctx.programs = meta, common, programs
        ctx.has_bias = bias is not None
        return output

    @staticmethod
    def backward(ctx, gradient):
        inputs, steps, pointers, offsets, grid, weight_offsets, *packeds = ctx.saved_tensors
        gradient = gradient.contiguous()
        dx, ds, db = None, None, None
        if ctx.needs_input_grad[0]:
            dx = torch.zeros_like(inputs)
            _grouped[(ctx.programs,)](
                inputs,
                gradient,
                pointers,
                steps,
                weight_offsets,
                grid,
                offsets,
                None,
                dx,
                HAS_BIAS=False,
                BACKWARD=True,
                HAS_OFFSETS=ctx.has_offsets,
                enable_fp_fusion=False,
                **ctx.meta,
                **ctx.common,
            )
        if ctx.needs_input_grad[1]:
            ds = torch.empty_like(steps)
            n, k = ctx.meta["N"], ctx.meta["K"]
            _grouped_scales[(len(packeds), triton.cdiv(n, 16), triton.cdiv(k, 64))](
                inputs, gradient, pointers, grid, offsets, ds, **ctx.meta
            )
        if ctx.has_bias and ctx.needs_input_grad[6]:
            n = ctx.meta["N"]
            db = inputs.new_empty((len(packeds), n))
            _grouped_bias[(len(packeds), triton.cdiv(n, 64))](gradient, offsets, db, n)
        return dx, ds, None, None, None, None, db, None, None


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
    weight_offsets=None,
):
    """Use native expert-sorted rows without host token counts or expanded expert weights.

    Offsets are cumulative, monotonic CUDA expert endpoints from HF routing.
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
    if weight_offsets is not None and (
        weight_offsets.shape != steps.shape
        or weight_offsets.device != steps.device
        or weight_offsets.requires_grad
    ):
        raise ValueError("Native offsets must be frozen and match the scale groups.")
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
        weight_offsets,
    )
