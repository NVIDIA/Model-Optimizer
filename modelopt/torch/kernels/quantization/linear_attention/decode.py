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

"""Fused FP32 token/replay training recurrence with state checkpoint recomputation."""

import torch
import triton
import triton.language as tl
from torch.autograd.function import once_differentiable
from triton.language.extra.cuda import libdevice

from .int8 import int8_scalar_qdq

__all__ = ["fused_recurrence"]


@triton.jit
def _fp8_qdq(value):
    amax = tl.max(tl.abs(value))
    scale = tl.where(amax > 0, amax * (1.0 / 448.0), 1.0)
    normalized = tl.div_rn(value, scale)
    normalized = tl.where(
        normalized < -448.0, -448.0, tl.where(normalized > 448.0, 448.0, normalized)
    )
    bits = tl.abs(normalized).to(tl.int32, bitcast=True)
    exponent = ((bits >> 23) & 255) - 127
    step = ((tl.maximum(exponent, -6) - 3 + 127) << 23).to(tl.float32, bitcast=True)
    rounded = libdevice.nearbyint(normalized / step) * step
    return rounded * scale, scale


@triton.jit
def _sum_keys(value):
    BK: tl.constexpr = value.shape[0]
    rows = tl.arange(0, BK)
    for level in tl.static_range(0, tl.constexpr(triton.next_power_of_2(BK).bit_length() - 1)):
        offset = BK >> (level + 1)
        value = value + tl.gather(
            value, tl.broadcast_to((rows ^ offset)[:, None], value.shape), axis=0
        )
    return tl.sum(tl.where(rows[:, None] == 0, value, 0.0), axis=0)


@triton.jit
def _advance(
    state,
    key,
    value,
    gate,
    beta,
    write_state,
    CHANNEL_GATE: tl.constexpr,
    STATE_QDQ: tl.constexpr,
    FACTOR_QDQ: tl.constexpr,
):
    decay = libdevice.exp(gate)
    if CHANNEL_GATE:
        decayed = state * decay[:, None]
    else:
        decayed = state * decay
    residual = value - _sum_keys(key[:, None] * decayed)
    update = beta * residual
    update_scale = 1.0
    if FACTOR_QDQ:
        update, update_scale = _fp8_qdq(update)
    working = decayed + key[:, None] * update[None, :]
    stored = working
    if STATE_QDQ == 3:
        state_scale = tl.full((working.shape[0], working.shape[1] // 32), 1.0, tl.float16)
    else:
        state_scale = 1.0
    if STATE_QDQ:
        if write_state:
            if STATE_QDQ == 3:
                groups = tl.reshape(working, (working.shape[0], working.shape[1] // 32, 32))
                scale = tl.maximum(tl.max(tl.abs(groups), 2) * (1.0 / 127.0), 6e-8)
                normalized = tl.div_rn(groups, scale[:, :, None])
                codes = tl.floor(tl.abs(normalized) + 0.5)
                codes = tl.where(normalized < 0, -codes, codes)
                state_scale = scale.to(tl.float16)
                stored = tl.reshape(
                    tl.minimum(tl.maximum(codes, -127.0), 127.0)
                    * state_scale.to(tl.float32)[:, :, None],
                    working.shape,
                )
            elif STATE_QDQ == 2:
                amax = tl.max(tl.abs(working))
                # Match Torch scalar division: FP32 multiply by the rounded reciprocal.
                state_scale = tl.where(amax > 0, amax * (1.0 / 127.0), 1.0)
                stored = int8_scalar_qdq(working, state_scale)
            else:
                stored, state_scale = _fp8_qdq(working)
    return stored, working, decayed, residual, update, update_scale, state_scale


@triton.jit
def _decode_fwd(
    Q,
    K,
    V,
    G,
    Beta,
    Initial,
    Out,
    Final,
    Updates,
    Anchor,
    AnchorScale,
    UpdateScale,
    Checkpoints,
    T: tl.constexpr,
    H: tl.constexpr,
    DK: tl.constexpr,
    DV: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NV: tl.constexpr,
    CHANNEL_GATE: tl.constexpr,
    STATE_QDQ: tl.constexpr,
    REPLAY: tl.constexpr,
    FACTOR_QDQ: tl.constexpr,
    WINDOW: tl.constexpr,
    CURSOR: tl.constexpr,
    READ_STORED: tl.constexpr,
    SCALE: tl.constexpr,
    INTERVAL: tl.constexpr,
):
    h = tl.program_id(0)
    tile = tl.program_id(1)
    keys = tl.arange(0, BK)
    values = tile * BV + tl.arange(0, BV)
    offsets = h * DK * DV + keys[:, None] * DV + values[None, :]
    mask = (keys[:, None] < DK) & (values[None, :] < DV)
    state = tl.load(Initial + offsets, mask, 0)
    tl.store(Anchor + offsets, state, mask)
    if STATE_QDQ == 3:
        groups = tile * (BV // 32) + tl.arange(0, BV // 32)
        scale_offsets = (h * DK + keys[:, None]) * (DV // 32) + groups[None, :]
        scale_mask = (keys[:, None] < DK) & (groups[None, :] < DV // 32)
        tl.store(AnchorScale + scale_offsets, 1.0, scale_mask)
    else:
        tl.store(AnchorScale + h * NV + tile, 1.0)
    for t in range(T):
        if t % INTERVAL == 0:
            tl.store(Checkpoints + (t // INTERVAL) * H * DK * DV + offsets, state, mask)
        query = tl.load(Q + (t * H + h) * DK + keys, keys < DK, 0)
        key = tl.load(K + (t * H + h) * DK + keys, keys < DK, 0)
        value = tl.load(V + (t * H + h) * DV + values, values < DV, 0)
        beta = tl.load(Beta + t * H + h)
        if CHANNEL_GATE:
            gate = tl.load(G + (t * H + h) * DK + keys, keys < DK, 0)
        else:
            gate = tl.load(G + t * H + h)
        write_state = not REPLAY or (CURSOR + t + 1) % WINDOW == 0
        state, working, decayed, residual, update, update_scale, state_scale = _advance(
            state,
            key,
            value,
            gate,
            beta,
            write_state,
            CHANNEL_GATE,
            STATE_QDQ,
            FACTOR_QDQ,
        )
        read = state if READ_STORED else working
        output = _sum_keys(query[:, None] * read) * SCALE
        tl.store(Out + (t * H + h) * DV + values, output, values < DV)
        tl.store(Updates + (t * H + h) * DV + values, update, values < DV)
        tl.store(UpdateScale + (t * H + h) * NV + tile, update_scale)
        if write_state:
            tl.store(Anchor + offsets, state, mask)
            if STATE_QDQ == 3:
                tl.store(AnchorScale + scale_offsets, state_scale, scale_mask)
            else:
                tl.store(AnchorScale + h * NV + tile, state_scale)
    tl.store(Final + offsets, state, mask)


@triton.jit
def _decode_bwd(
    Q,
    K,
    V,
    G,
    Beta,
    Checkpoints,
    DOut,
    DFinal,
    DUpdates,
    DAnchor,
    DQ,
    DKOut,
    DVOut,
    DG,
    DBeta,
    DInitial,
    T: tl.constexpr,
    H: tl.constexpr,
    DK: tl.constexpr,
    DV: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NV: tl.constexpr,
    CHANNEL_GATE: tl.constexpr,
    STATE_QDQ: tl.constexpr,
    REPLAY: tl.constexpr,
    FACTOR_QDQ: tl.constexpr,
    WINDOW: tl.constexpr,
    CURSOR: tl.constexpr,
    READ_STORED: tl.constexpr,
    SCALE: tl.constexpr,
    INTERVAL: tl.constexpr,
):
    h = tl.program_id(0)
    tile = tl.program_id(1)
    keys = tl.arange(0, BK)
    values = tile * BV + tl.arange(0, BV)
    offsets = h * DK * DV + keys[:, None] * DV + values[None, :]
    mask = (keys[:, None] < DK) & (values[None, :] < DV)
    adjoint = tl.load(DFinal + offsets, mask, 0)
    LAST_REFRESH: tl.constexpr = ((T + CURSOR) // WINDOW) * WINDOW - CURSOR - 1
    for t in range(T - 1, -1, -1):
        start = (t // INTERVAL) * INTERVAL
        state = tl.load(Checkpoints + (t // INTERVAL) * H * DK * DV + offsets, mask, 0)
        for j in range(start, t):
            key = tl.load(K + (j * H + h) * DK + keys, keys < DK, 0)
            value = tl.load(V + (j * H + h) * DV + values, values < DV, 0)
            beta = tl.load(Beta + j * H + h)
            if CHANNEL_GATE:
                gate = tl.load(G + (j * H + h) * DK + keys, keys < DK, 0)
            else:
                gate = tl.load(G + j * H + h)
            write_state = not REPLAY or (CURSOR + j + 1) % WINDOW == 0
            state, _, _, _, _, _, _ = _advance(
                state,
                key,
                value,
                gate,
                beta,
                write_state,
                CHANNEL_GATE,
                STATE_QDQ,
                FACTOR_QDQ,
            )
        query = tl.load(Q + (t * H + h) * DK + keys, keys < DK, 0)
        key = tl.load(K + (t * H + h) * DK + keys, keys < DK, 0)
        value = tl.load(V + (t * H + h) * DV + values, values < DV, 0)
        beta = tl.load(Beta + t * H + h)
        if CHANNEL_GATE:
            gate = tl.load(G + (t * H + h) * DK + keys, keys < DK, 0)
        else:
            gate = tl.load(G + t * H + h)
        write_state = not REPLAY or (CURSOR + t + 1) % WINDOW == 0
        stored, working, decayed, residual, update, _, _ = _advance(
            state,
            key,
            value,
            gate,
            beta,
            write_state,
            CHANNEL_GATE,
            STATE_QDQ,
            FACTOR_QDQ,
        )
        if REPLAY:
            if t == LAST_REFRESH:
                adjoint += tl.load(DAnchor + offsets, mask, 0)
        elif t == T - 1:
            adjoint += tl.load(DAnchor + offsets, mask, 0)
        doutput = tl.load(DOut + (t * H + h) * DV + values, values < DV, 0)
        read = stored if READ_STORED else working
        dquery = tl.sum(read * doutput[None, :], 1) * SCALE
        dstate = adjoint + query[:, None] * (doutput[None, :] * SCALE)
        dupdate = tl.sum(key[:, None] * dstate, 0)
        dupdate += tl.load(DUpdates + (t * H + h) * DV + values, values < DV, 0)
        dbeta = tl.sum(dupdate * residual, 0)
        dvalue = beta * dupdate
        dkey = tl.sum(update[None, :] * dstate - decayed * dvalue[None, :], 1)
        ddecayed = dstate - key[:, None] * dvalue[None, :]
        dgate = tl.sum(ddecayed * decayed, 1)
        decay = libdevice.exp(gate)
        if CHANNEL_GATE:
            adjoint = ddecayed * decay[:, None]
            tl.store(DG + ((tile * T + t) * H + h) * DK + keys, dgate, keys < DK)
        else:
            adjoint = ddecayed * decay
            tl.store(DG + (tile * T + t) * H + h, tl.sum(dgate, 0))
        tl.store(DQ + ((tile * T + t) * H + h) * DK + keys, dquery, keys < DK)
        tl.store(DKOut + ((tile * T + t) * H + h) * DK + keys, dkey, keys < DK)
        tl.store(DVOut + (t * H + h) * DV + values, dvalue, values < DV)
        tl.store(DBeta + (tile * T + t) * H + h, dbeta)
    tl.store(DInitial + offsets, adjoint, mask)


class _FusedRecurrence(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        g,
        beta,
        initial,
        state_qdq,
        block_v,
        replay,
        factor_qdq,
        window,
        cursor,
        read_stored,
        scale,
        interval,
    ):
        length, heads, keys = q.shape
        values = v.shape[-1]
        tiles = triton.cdiv(values, block_v)
        checkpoints = torch.empty(
            (triton.cdiv(length, interval), heads, keys, values), device=q.device, dtype=q.dtype
        )
        output, updates = torch.empty_like(v), torch.empty_like(v)
        final, anchor = torch.empty_like(initial), torch.empty_like(initial)
        scale_shape = (heads, keys, values // 32) if state_qdq == 3 else (heads, tiles)
        anchor_scales = torch.empty(
            scale_shape, device=q.device, dtype=torch.float16 if state_qdq == 3 else q.dtype
        )
        update_scales = torch.empty((length, heads, tiles), device=q.device, dtype=q.dtype)
        kwargs = {
            "T": length,
            "H": heads,
            "DK": keys,
            "DV": values,
            "BK": triton.next_power_of_2(keys),
            "BV": block_v,
            "NV": tiles,
            "CHANNEL_GATE": g.ndim == 3,
            "STATE_QDQ": state_qdq,
            "REPLAY": replay,
            "FACTOR_QDQ": factor_qdq,
            "WINDOW": window,
            "CURSOR": cursor,
            "READ_STORED": read_stored,
            "SCALE": scale,
            "INTERVAL": interval,
            "num_warps": 4 if triton.next_power_of_2(keys) * block_v <= 4096 else 8,
            "enable_fp_fusion": False,
        }
        _decode_fwd[(heads, tiles)](
            q,
            k,
            v,
            g,
            beta,
            initial,
            output,
            final,
            updates,
            anchor,
            anchor_scales,
            update_scales,
            checkpoints,
            **kwargs,
        )
        ctx.save_for_backward(q, k, v, g, beta, initial, checkpoints)
        ctx.kwargs = kwargs
        ctx.mark_non_differentiable(anchor_scales, update_scales)
        return output, final, updates, anchor, anchor_scales, update_scales

    @staticmethod
    @once_differentiable
    def backward(ctx, doutput, dfinal, dupdates, danchor, _danchor_scales, _dupdate_scales):
        q, k, v, g, beta, initial, checkpoints = ctx.saved_tensors
        tiles = ctx.kwargs["NV"]
        dq = torch.empty((tiles, *q.shape), device=q.device, dtype=q.dtype)
        dk = torch.empty_like(dq)
        dv = torch.empty_like(v)
        dg = torch.empty((tiles, *g.shape), device=q.device, dtype=q.dtype)
        dbeta = torch.empty((tiles, *beta.shape), device=q.device, dtype=q.dtype)
        dinitial = torch.empty_like(initial)
        doutput = torch.zeros_like(v) if doutput is None else doutput.contiguous()
        dfinal = torch.zeros_like(initial) if dfinal is None else dfinal.contiguous()
        dupdates = torch.zeros_like(v) if dupdates is None else dupdates.contiguous()
        danchor = torch.zeros_like(initial) if danchor is None else danchor.contiguous()
        _decode_bwd[(q.shape[1], tiles)](
            q,
            k,
            v,
            g,
            beta,
            checkpoints,
            doutput,
            dfinal,
            dupdates,
            danchor,
            dq,
            dk,
            dv,
            dg,
            dbeta,
            dinitial,
            **ctx.kwargs,
        )
        return dq.sum(0), dk.sum(0), dv, dg.sum(0), dbeta.sum(0), dinitial, *([None] * 9)


def fused_recurrence(
    q,
    k,
    v,
    g,
    beta,
    initial,
    *,
    state_qdq=False,
    state_format="fp8_e4m3",
    state_codec="tile",
    block_v=16,
    replay=False,
    factor_qdq=False,
    window=8,
    cursor=0,
    read_stored=True,
    scale=None,
    checkpoint_interval=8,
):
    """Run a nonempty activated sequence; encoding keys/gates and carry metadata is external."""
    if not all(
        x.is_cuda and x.dtype == torch.float32 and x.device == q.device
        for x in (q, k, v, g, beta, initial)
    ):
        raise ValueError("Fused recurrence requires CUDA FP32 inputs on one device")
    if q.ndim != 3 or q.shape != k.shape or v.shape[:2] != q.shape[:2] or beta.shape != q.shape[:2]:
        raise ValueError("Expected aligned [T,H,D] q/k/v and [T,H] beta")
    length, heads, keys = q.shape
    if length == 0 or not 1 <= keys <= 128 or initial.shape != (heads, keys, v.shape[-1]):
        raise ValueError("Require nonempty sequence, Dk<=128, and matching initial state")
    if g.shape not in (q.shape, beta.shape) or block_v not in (16, 32, 64, 128):
        raise ValueError("Invalid gate shape or state value-block grouping")
    if not 1 <= window <= 64 or not 0 <= cursor < window or not 1 <= checkpoint_interval <= 32:
        raise ValueError("Invalid replay cursor/window or recomputation interval")
    if state_format not in ("fp8_e4m3", "int8"):
        raise ValueError("State format must be fp8_e4m3 or int8")
    if state_codec not in ("tile", "int8_hadamard32"):
        raise ValueError("State codec must be tile or int8_hadamard32")
    if state_codec == "int8_hadamard32":
        if (state_qdq and state_format != "int8") or v.shape[-1] % 32 or block_v < 32:
            raise ValueError("Hadamard codec requires INT8, Dv divisible by 32 and block_v >= 32")
    state_mode = 3 if state_codec == "int8_hadamard32" else (2 if state_format == "int8" else 1)
    state_qdq = state_mode if state_qdq else 0
    scale = keys**-0.5 if scale is None else scale
    return _FusedRecurrence.apply(
        *(x.contiguous() for x in (q, k, v, g, beta, initial)),
        state_qdq,
        block_v,
        replay,
        factor_qdq,
        window,
        cursor,
        read_stored,
        scale,
        checkpoint_interval,
    )
