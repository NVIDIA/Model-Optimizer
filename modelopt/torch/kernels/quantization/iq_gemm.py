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

"""Packed GGML projections with bounded tile storage in both autograd directions."""

import math

import torch
import triton
import triton.language as tl

from .repair import iq_levels, iq_metadata

__all__ = ["iq_linear"]


@triton.jit
def _weights(
    packed,
    grid,
    steps,
    offsets,
    rows,
    cols,
    valid,
    K: tl.constexpr,
    GROUP: tl.constexpr,
    KIND: tl.constexpr,
    BYTES: tl.constexpr,
    DTYPE: tl.constexpr,
    HAS_OFFSETS: tl.constexpr,
):
    index = rows[:, None].to(tl.int64) * K + cols[None, :]
    levels = iq_levels(packed, grid, index, valid, KIND, BYTES).to(tl.float32)
    scales = tl.load(steps + index // GROUP, valid, 0)
    values = levels * scales
    if HAS_OFFSETS:
        values += tl.load(offsets + index // GROUP, valid, 0).to(tl.float32)
    return values.to(DTYPE)


@triton.jit
def _forward(
    X,
    P,
    S,
    WOFFS,
    GRID,
    BIAS,
    Y,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    GROUP: tl.constexpr,
    KIND: tl.constexpr,
    BYTES: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BM: tl.constexpr = 32,
    BN: tl.constexpr = 32,
    BK: tl.constexpr = 64,
    MI=0,
    NI=0,
    EXPLICIT: tl.constexpr = False,
    HAS_OFFSETS: tl.constexpr = False,
):
    m = ((MI if EXPLICIT else tl.program_id(0)) * BM + tl.arange(0, BM)).to(tl.int64)
    n = ((NI if EXPLICIT else tl.program_id(1)) * BN + tl.arange(0, BN)).to(tl.int64)
    k = tl.arange(0, BK)
    acc = tl.full((BM, BN), 0, tl.float32)
    for start in range(tl.cdiv(K, BK)):
        cols = start * BK + k
        x = tl.load(X + m[:, None] * K + cols[None, :], (m[:, None] < M) & (cols[None, :] < K), 0)
        w = _weights(
            P,
            GRID,
            S,
            WOFFS,
            n,
            cols,
            (n[:, None] < N) & (cols[None, :] < K),
            K,
            GROUP,
            KIND,
            BYTES,
            x.dtype,
            HAS_OFFSETS,
        )
        acc += tl.dot(x, tl.trans(w))
    if HAS_BIAS:
        acc += tl.load(BIAS + n, n < N, 0)[None, :]
    tl.store(Y + m[:, None] * N + n[None, :], acc, (m[:, None] < M) & (n[None, :] < N))


@triton.jit
def _input_backward(
    G,
    P,
    S,
    WOFFS,
    GRID,
    DX,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    GROUP: tl.constexpr,
    KIND: tl.constexpr,
    BYTES: tl.constexpr,
    BM: tl.constexpr = 32,
    BN: tl.constexpr = 32,
    BK: tl.constexpr = 64,
    MI=0,
    NI=0,
    EXPLICIT: tl.constexpr = False,
    HAS_OFFSETS: tl.constexpr = False,
):
    m = ((MI if EXPLICIT else tl.program_id(0)) * BM + tl.arange(0, BM)).to(tl.int64)
    k = ((NI if EXPLICIT else tl.program_id(1)) * BK + tl.arange(0, BK)).to(tl.int64)
    n = tl.arange(0, BN)
    acc = tl.full((BM, BK), 0, tl.float32)
    for start in range(tl.cdiv(N, BN)):
        rows = start * BN + n
        g = tl.load(G + m[:, None] * N + rows[None, :], (m[:, None] < M) & (rows[None, :] < N), 0)
        w = _weights(
            P,
            GRID,
            S,
            WOFFS,
            rows,
            k,
            (rows[:, None] < N) & (k[None, :] < K),
            K,
            GROUP,
            KIND,
            BYTES,
            g.dtype,
            HAS_OFFSETS,
        )
        acc += tl.dot(g, w)
    tl.store(DX + m[:, None] * K + k[None, :], acc, (m[:, None] < M) & (k[None, :] < K))


@triton.jit
def _scale_backward(
    X,
    G,
    P,
    GRID,
    DS,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    GROUP: tl.constexpr,
    KIND: tl.constexpr,
    BYTES: tl.constexpr,
    BM: tl.constexpr = 32,
    BN: tl.constexpr = 16,
    BK: tl.constexpr = 64,
    NI=0,
    KI=0,
    EXPLICIT: tl.constexpr = False,
):
    n = ((NI if EXPLICIT else tl.program_id(0)) * BN + tl.arange(0, BN)).to(tl.int64)
    k = (KI if EXPLICIT else tl.program_id(1)) * BK + tl.arange(0, BK)
    m = tl.arange(0, BM)
    acc = tl.full((BN, BK), 0, tl.float32)
    for start in range(tl.cdiv(M, BM)):
        rows = (start * BM + m).to(tl.int64)
        x = tl.load(X + rows[:, None] * K + k[None, :], (rows[:, None] < M) & (k[None, :] < K), 0)
        g = tl.load(G + rows[None, :] * N + n[:, None], (rows[None, :] < M) & (n[:, None] < N), 0)
        acc += tl.dot(g, x)
    index = n[:, None].to(tl.int64) * K + k[None, :]
    valid = (n[:, None] < N) & (k[None, :] < K)
    levels = iq_levels(P, GRID, index, valid, KIND, BYTES).to(tl.float32)
    # Match materialized F.linear's rounded dW before the scale reduction.
    values = acc.to(X.dtype.element_ty).to(tl.float32) * levels
    reduced = tl.sum(tl.reshape(values, (BN, BK // GROUP, GROUP)), 2)
    groups = (KI if EXPLICIT else tl.program_id(1)) * (BK // GROUP) + tl.arange(0, BK // GROUP)
    tl.store(
        DS + n[:, None] * (K // GROUP) + groups[None, :],
        reduced,
        (n[:, None] < N) & (groups[None, :] < K // GROUP),
    )


class _IQLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, packed, steps, name, shape, bias, weight_offsets):
        n, k = shape
        x = inputs.contiguous().view(-1, k)
        steps = steps.contiguous()
        if weight_offsets is not None:
            weight_offsets = weight_offsets.contiguous()
        grid, block_bytes = iq_metadata(packed.device, name)
        group = n * k // steps.numel()
        output = inputs.new_empty((x.shape[0], n))
        meta = {"GROUP": group, "KIND": name, "BYTES": block_bytes}
        if x.shape[0]:
            _forward[(triton.cdiv(x.shape[0], 32), triton.cdiv(n, 32))](
                x,
                packed,
                steps,
                weight_offsets,
                grid,
                bias,
                output,
                x.shape[0],
                n,
                k,
                HAS_BIAS=bias is not None,
                HAS_OFFSETS=weight_offsets is not None,
                enable_fp_fusion=False,
                **meta,
            )
        ctx.save_for_backward(x, packed, steps, grid, weight_offsets)
        ctx.has_offsets = weight_offsets is not None
        ctx.meta, ctx.shape, ctx.input_shape = meta, shape, inputs.shape
        ctx.has_bias = bias is not None
        return output.view(*inputs.shape[:-1], n)

    @staticmethod
    def backward(ctx, gradient):
        x, packed, steps, grid, weight_offsets = ctx.saved_tensors
        n, k = ctx.shape
        gradient = gradient.contiguous().view(-1, n)
        dx, ds = None, None
        if ctx.needs_input_grad[0]:
            dx = torch.empty_like(x)
            if x.shape[0]:
                _input_backward[(triton.cdiv(x.shape[0], 32), triton.cdiv(k, 64))](
                    gradient,
                    packed,
                    steps,
                    weight_offsets,
                    grid,
                    dx,
                    x.shape[0],
                    n,
                    k,
                    HAS_OFFSETS=ctx.has_offsets,
                    enable_fp_fusion=False,
                    **ctx.meta,
                )
        if ctx.needs_input_grad[2]:
            ds = torch.empty_like(steps) if x.shape[0] else torch.zeros_like(steps)
            if x.shape[0]:
                _scale_backward[(triton.cdiv(n, 16), triton.cdiv(k, 64))](
                    x, gradient, packed, grid, ds, x.shape[0], n, k, **ctx.meta
                )
        db = gradient.sum(0) if ctx.has_bias and ctx.needs_input_grad[5] else None
        return None if dx is None else dx.view(ctx.input_shape), None, ds, None, None, db, None


def iq_linear(inputs, packed, steps, format_name, weight_shape, bias=None, *, weight_offsets=None):
    """Apply a native GGML matrix without expanding weights or saving a full dW.

    ``steps`` are the differentiable projected native fields, in flattened
    row-major scale-group order. Packed codes remain immutable.
    """
    from modelopt.torch.quantization.ggml import IQ_FORMAT_REGISTRY

    fmt = IQ_FORMAT_REGISTRY[format_name]
    shape = tuple(weight_shape)
    if (
        len(shape) != 2
        or shape[1] % fmt.block_size
        or inputs.shape[-1] != shape[1]
        or not inputs.is_cuda
        or inputs.dtype not in (torch.float16, torch.bfloat16)
        or packed.device != inputs.device
        or steps.device != inputs.device
        or packed.dtype != torch.uint8
        or not packed.is_contiguous()
    ):
        raise ValueError(
            "Packed IQ GEMM requires CUDA FP16/BF16 inputs and contiguous native matrix bytes."
        )
    _, block_bytes = iq_metadata(packed.device, format_name)
    group = fmt.int8_group_size
    if packed.numel() != math.prod(
        shape
    ) // fmt.block_size * block_bytes or steps.numel() * group != math.prod(shape):
        raise ValueError("Packed bytes/scale groups do not match the IQ matrix.")
    if weight_offsets is not None and (
        weight_offsets.shape != steps.shape
        or weight_offsets.device != steps.device
        or weight_offsets.requires_grad
    ):
        raise ValueError("Native offsets must be frozen and match the scale groups.")
    if bias is not None:
        if bias.shape != (shape[0],) or bias.device != inputs.device or bias.dtype != inputs.dtype:
            raise ValueError(
                "Packed IQ bias must match the output width, device and compute dtype."
            )
        bias = bias.contiguous()
    return _IQLinear.apply(inputs, packed, steps, format_name, shape, bias, weight_offsets)
