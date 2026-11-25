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

from modelopt.torch.quantization.ggml import IQ_FORMAT_REGISTRY

from .repair import iq_levels, iq_metadata

__all__ = ["iq_linear"]

_TILE_M, _TILE_N, _TILE_K, _SCALE_TILE_N = 32, 32, 64, 16


@triton.jit
def _weights(
    packed,
    grid,
    steps,
    rows,
    cols,
    valid,
    K,
    GROUP: tl.constexpr,
    KIND: tl.constexpr,
    BYTES: tl.constexpr,
    DTYPE: tl.constexpr,
):
    index = rows[:, None].to(tl.int64) * K + cols[None, :]
    levels = iq_levels(packed, grid, index, valid, KIND, BYTES).to(tl.float32)
    scales = tl.load(steps + index // GROUP, valid, 0)
    values = levels * scales
    return values.to(DTYPE)


@triton.jit
def _forward(
    X,
    P,
    S,
    GRID,
    BIAS,
    Y,
    M,
    N,
    K,
    GROUP: tl.constexpr,
    KIND: tl.constexpr,
    BYTES: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BM: tl.constexpr = _TILE_M,
    BN: tl.constexpr = _TILE_N,
    BK: tl.constexpr = _TILE_K,
):
    m = (tl.program_id(0) * BM + tl.arange(0, BM)).to(tl.int64)
    n = (tl.program_id(1) * BN + tl.arange(0, BN)).to(tl.int64)
    k = tl.arange(0, BK)
    acc = tl.full((BM, BN), 0, tl.float32)
    for start in range(tl.cdiv(K, BK)):
        cols = start * BK + k
        x = tl.load(X + m[:, None] * K + cols[None, :], (m[:, None] < M) & (cols[None, :] < K), 0)
        w = _weights(
            P,
            GRID,
            S,
            n,
            cols,
            (n[:, None] < N) & (cols[None, :] < K),
            K,
            GROUP,
            KIND,
            BYTES,
            x.dtype,
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
    GRID,
    DX,
    M,
    N,
    K,
    GROUP: tl.constexpr,
    KIND: tl.constexpr,
    BYTES: tl.constexpr,
    BM: tl.constexpr = _TILE_M,
    BN: tl.constexpr = _TILE_N,
    BK: tl.constexpr = _TILE_K,
):
    m = (tl.program_id(0) * BM + tl.arange(0, BM)).to(tl.int64)
    k = (tl.program_id(1) * BK + tl.arange(0, BK)).to(tl.int64)
    n = tl.arange(0, BN)
    acc = tl.full((BM, BK), 0, tl.float32)
    for start in range(tl.cdiv(N, BN)):
        rows = start * BN + n
        g = tl.load(G + m[:, None] * N + rows[None, :], (m[:, None] < M) & (rows[None, :] < N), 0)
        w = _weights(
            P,
            GRID,
            S,
            rows,
            k,
            (rows[:, None] < N) & (k[None, :] < K),
            K,
            GROUP,
            KIND,
            BYTES,
            g.dtype,
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
    N,
    K,
    GROUP: tl.constexpr,
    KIND: tl.constexpr,
    BYTES: tl.constexpr,
    BM: tl.constexpr = _TILE_M,
    BN: tl.constexpr = _SCALE_TILE_N,
    BK: tl.constexpr = _TILE_K,
):
    n = (tl.program_id(0) * BN + tl.arange(0, BN)).to(tl.int64)
    k = tl.program_id(1) * BK + tl.arange(0, BK)
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
    # Preserve native F.linear autograd parity: round dW to the compute dtype
    # before the FP32 reduction into projected scale gradients.
    values = acc.to(X.dtype.element_ty).to(tl.float32) * levels
    reduced = tl.sum(tl.reshape(values, (BN, BK // GROUP, GROUP)), 2)
    groups = tl.program_id(1) * (BK // GROUP) + tl.arange(0, BK // GROUP)
    tl.store(
        DS + n[:, None] * (K // GROUP) + groups[None, :],
        reduced,
        (n[:, None] < N) & (groups[None, :] < K // GROUP),
    )


class _IQLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, packed, steps, name, shape, bias):
        n, k = shape
        x = inputs.contiguous().view(-1, k)
        steps = steps.contiguous()
        grid, block_bytes = iq_metadata(packed.device, name)
        group = n * k // steps.numel()
        output = inputs.new_empty((x.shape[0], n))
        meta = {"GROUP": group, "KIND": name, "BYTES": block_bytes}
        if x.shape[0]:
            _forward[(triton.cdiv(x.shape[0], _TILE_M), triton.cdiv(n, _TILE_N))](
                x,
                packed,
                steps,
                grid,
                bias,
                output,
                x.shape[0],
                n,
                k,
                HAS_BIAS=bias is not None,
                enable_fp_fusion=False,
                **meta,
            )
        ctx.save_for_backward(x if ctx.needs_input_grad[2] else None, packed, steps, grid)
        ctx.meta, ctx.shape, ctx.input_shape = meta, shape, inputs.shape
        ctx.has_bias = bias is not None
        return output.view(*inputs.shape[:-1], n)

    @staticmethod
    def backward(ctx, gradient):
        x, packed, steps, grid = ctx.saved_tensors
        n, k = ctx.shape
        gradient = gradient.contiguous().view(-1, n)
        m = gradient.shape[0]
        dx, ds = None, None
        if ctx.needs_input_grad[0]:
            dx = gradient.new_empty((m, k))
            if m:
                _input_backward[(triton.cdiv(m, _TILE_M), triton.cdiv(k, _TILE_K))](
                    gradient,
                    packed,
                    steps,
                    grid,
                    dx,
                    m,
                    n,
                    k,
                    enable_fp_fusion=False,
                    **ctx.meta,
                )
        if ctx.needs_input_grad[2]:
            ds = torch.empty_like(steps) if m else torch.zeros_like(steps)
            if m:
                _scale_backward[(triton.cdiv(n, _SCALE_TILE_N), triton.cdiv(k, _TILE_K))](
                    x, gradient, packed, grid, ds, m, n, k, enable_fp_fusion=False, **ctx.meta
                )
        db = gradient.sum(0) if ctx.has_bias and ctx.needs_input_grad[5] else None
        return None if dx is None else dx.view(ctx.input_shape), None, ds, None, None, db


def iq_linear(inputs, packed, steps, format_name, weight_shape, bias=None):
    """Apply a native GGML matrix without expanding weights or saving a full dW.

    ``steps`` are differentiable FP32 native fields in row-major block/group order,
    shaped ``(prod(weight_shape) // block_size, block_size // int8_group_size)``.
    Packed codes remain immutable. Forward and input VJP decode weights again for
    each 32-token row tile; larger token counts increase decode work.
    """
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
    blocks = math.prod(shape) // fmt.block_size
    if (
        packed.numel() != blocks * block_bytes
        or steps.dtype != torch.float32
        or steps.shape != (blocks, fmt.block_size // group)
    ):
        raise ValueError("Packed bytes/native FP32 scales do not match the IQ matrix.")
    if bias is not None:
        if bias.shape != (shape[0],) or bias.device != inputs.device or bias.dtype != inputs.dtype:
            raise ValueError(
                "Packed IQ bias must match the output width, device and compute dtype."
            )
        bias = bias.contiguous()
    return _IQLinear.apply(inputs, packed, steps, format_name, shape, bias)
