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

"""GDN replay forward corresponding to the ModelOpt two-precision window adjoint.

Within a window, outputs use a rounded checkpoint and cached FP16 residuals.
Every 16th token solves the raw-input triangular system to refresh the FP32
anchor. Quantized reads never participate in that checkpoint update.
"""

import triton
import triton.language as tl

from ..common.fp8_quant import fp8_scalar_qdq

__all__ = []


@triton.jit
def _flush(anchor, raw_keys, values, cumulative, betas):
    keys = raw_keys / tl.sqrt(tl.sum(raw_keys * raw_keys, 1) + 1e-6)[:, None]
    keys = keys.to(tl.bfloat16)
    gram = tl.dot(keys, tl.trans(keys))
    projected = tl.trans(tl.dot(anchor.to(tl.bfloat16), tl.trans(keys)))
    residual = (values - tl.exp(cumulative)[:, None] * projected).to(tl.bfloat16).to(tl.float32)
    positions = tl.arange(0, 16)
    solved = tl.full(values.shape, 0, tl.float32)
    for row in range(16):
        # (I + tril(beta_i * exp(G_i-G_j) * <k_i,k_j>, -1)) u = beta * residual.
        gate = tl.sum(tl.where(positions == row, cumulative, 0))
        coefficients = tl.sum(tl.where(positions[:, None] == row, gram, 0), 0)
        coefficients = tl.where(positions < row, tl.exp(gate - cumulative) * coefficients, 0)
        correction = tl.sum(solved * coefficients[:, None], 0)
        rhs = tl.sum(tl.where(positions[:, None] == row, residual, 0), 0)
        beta = tl.sum(tl.where(positions == row, betas, 0))
        solved = tl.where(positions[:, None] == row, beta * (rhs - correction)[None, :], solved)
    return solved, keys


@triton.jit
def _replay_step(
    Anchor,
    Read,
    Rings,
    Amax,
    QKV,
    A,
    B,
    Rate,
    Bias,
    Slots,
    Positions,
    Windows,
    Output,
    query_scale,
    STRIDES,
    KEY_HEADS: tl.constexpr,
    HEADS: tl.constexpr,
    V: tl.constexpr,
    READ_MODE: tl.constexpr,
    NULL: tl.constexpr,
):
    tile, row, head = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    slot = tl.load(Slots + row).to(tl.int64)
    if slot == NULL:
        return
    position = tl.load(Positions + row).to(tl.int64)
    first = (position == 0) & (tl.load(Windows + row) == 0)
    flush = position == 15
    key_head = head // (HEADS // KEY_HEADS)
    Keys, Values, Gates, Betas, Updates = Rings
    ss, sk, sv, sg, sb, su, sa = STRIDES
    channels = tl.arange(0, 128)
    rows = tile * 64 + tl.arange(0, 64)
    times = tl.arange(0, 16)
    packed = QKV + row * (2 * KEY_HEADS * 128 + HEADS * V)
    q = tl.load(packed + key_head * 128 + channels).to(tl.float32)
    k = tl.load(packed + (KEY_HEADS + key_head) * 128 + channels).to(tl.float32)
    v = tl.load(packed + 2 * KEY_HEADS * 128 + head * V + rows, rows < V, 0).to(tl.float32)
    raw_gate = tl.load(A + row * HEADS + head).to(tl.float32) + tl.load(Bias + head)
    softplus = tl.where(raw_gate > 20, raw_gate, tl.extra.cuda.libdevice.log1p(tl.exp(raw_gate)))
    gate = -tl.exp(tl.load(Rate + head)) * softplus
    beta = tl.sigmoid(tl.load(B + row * HEADS + head).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
    key_offsets = slot * sk + (key_head * 16 + times[:, None]) * 128 + channels[None, :]
    value_offsets = (head * 16 + times[:, None]) * V + rows[None, :]
    gate_offsets = head * 16 + times
    key_window = tl.load(Keys + key_offsets, times[:, None] < position, 0).to(tl.float32)
    key_window = tl.where(times[:, None] == position, k[None, :], key_window)
    gate_window = tl.load(Gates + slot * sg + gate_offsets, times < position, 0)
    gate_window = tl.where(times == position, gate, gate_window)
    cumulative = tl.cumsum(gate_window)
    total = tl.sum(gate_window)
    decay = tl.exp(total)
    weights = tl.exp(total - cumulative)
    qnorm = tl.sqrt(tl.sum(q * q) + 1e-6)
    knorm = tl.sqrt(tl.sum(k * k) + 1e-6)
    query = q / qnorm * query_scale
    key = k / knorm
    norms = tl.sqrt(tl.sum(key_window * key_window, 1) + 1e-6)
    key_query = tl.sum(key_window * query[None, :], 1) / norms
    state_offsets = slot * ss + (head * V + rows[:, None]) * 128 + channels[None, :]
    checkpoint = Anchor
    if READ_MODE != "s32_rs":
        if not (first | flush):
            checkpoint = Read
    state = tl.load(checkpoint + state_offsets, rows[:, None] < V, 0)
    # Two matrix-vector products use BF16 tensor-core operands; normalization
    # follows the contraction, as in the differentiable window implementation.
    operands = tl.where(times[None, :] == 0, q[:, None], 0)
    operands = tl.where(times[None, :] == 1, k[:, None], operands)
    products = tl.dot(state.to(tl.bfloat16), operands.to(tl.bfloat16))
    projected_q = tl.sum(tl.where(times[None, :] == 0, products, 0), 1) / qnorm * query_scale
    if flush:
        values = tl.load(
            Values + slot * sv + value_offsets, (times[:, None] < position) & (rows[None, :] < V), 0
        ).to(tl.float32)
        values = tl.where(times[:, None] == position, v[None, :], values)
        betas = tl.load(Betas + slot * sb + gate_offsets, times < position, 0)
        betas = tl.where(times == position, beta, betas)
        solved, normalized_keys = _flush(state, key_window, values, cumulative, betas)
        output = decay * projected_q + tl.sum(solved * (weights * key_query)[:, None], 0)
        weighted = (solved * weights[:, None]).to(tl.bfloat16)
        state = decay * state + tl.dot(tl.trans(weighted), normalized_keys)
        tl.store(Anchor + state_offsets, state, rows[:, None] < V)
    else:
        key_key = tl.sum(key_window * key[None, :], 1) / norms
        cached = tl.load(
            Updates + slot * su + value_offsets,
            (times[:, None] < position) & (rows[None, :] < V),
            0,
        ).to(tl.float32)
        weights = tl.where(times < position, weights, 0)
        projected_k = tl.sum(tl.where(times[None, :] == 1, products, 0), 1) / knorm
        delta = beta * (
            v - (decay * projected_k + tl.sum(cached * (weights * key_key)[:, None], 0))
        )
        output = decay * projected_q + tl.sum(cached * (weights * key_query)[:, None], 0)
        output += delta * tl.sum(key * query)
        write = (head * 16 + position) * V + rows
        tl.store(Updates + slot * su + write, delta.to(tl.float16), rows < V)
        tl.store(Values + slot * sv + write, v, rows < V)
        if tile == 0:
            tl.store(Gates + slot * sg + head * 16 + position, gate)
            tl.store(Betas + slot * sb + head * 16 + position, beta)
            if head % (HEADS // KEY_HEADS) == 0:
                tl.store(Keys + slot * sk + (key_head * 16 + position) * 128 + channels, k)
    tl.store(Output + (row * HEADS + head) * V + rows, output, rows < V)
    if first | flush:
        if READ_MODE == "tp8_hi32":
            scale = tl.max(tl.abs(state), 1, keep_dims=True) / 448.0
            scale = tl.where(scale > 0, scale, 1.0)
            rounded = fp8_scalar_qdq(tl.div_rn(state, scale), 1.0, EMULATE=True) * scale
            tl.store(Read + state_offsets, rounded, rows[:, None] < V)
        elif READ_MODE == "tp4_hi32":
            tiles: tl.constexpr = triton.next_power_of_2(triton.cdiv(V, 64))
            tl.store(Amax + slot * sa + head * tiles + tile, tl.max(tl.abs(state)))
