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

"""Functional training launches of the optional quantized-ReplaySSM serving kernels."""

import torch
from vllm.model_executor.layers.fla.ops.fused_recurrent_replayssm import (
    _apply_h_value_axis,
    fused_recurrent_gated_delta_rule_replayssm,
    prefill_write_checkpoint,
)


def _storage(value, enabled):
    heads, keys, values = value.shape
    state = torch.zeros(
        2, heads, values, keys, device=value.device, dtype=torch.int8 if enabled else torch.float32
    )
    scales = (
        torch.zeros(2, heads, values // 32, keys, device=value.device, dtype=torch.float16)
        if enabled
        else None
    )
    indices = torch.ones(1, device=value.device, dtype=torch.int32)
    return state, scales, indices


def _decoded(state, scales):
    values = state[1].transpose(-1, -2).float()
    if scales is None:
        return values, None
    metadata = scales[1].transpose(-1, -2).contiguous()
    return (values.unflatten(-1, (-1, 32)) * metadata.float().unsqueeze(-1)).flatten(-2), metadata


def checkpoint(value, enabled):
    """Return the serving checkpoint's decoded Hadamard-basis values and FP16 scales."""
    state, scales, indices = _storage(value, enabled)
    original = value.transpose(-1, -2).unsqueeze(0).contiguous().float()
    if enabled:
        prefill_write_checkpoint(
            original, state, None, None, scales, None, None, indices, 8, hadamard=True
        )
    else:
        state[1] = _apply_h_value_axis(original)[0]
    return _decoded(state, scales)


def original_basis(value):
    """Use serving's FP32 checkpoint transform when returning a dense state."""
    return _apply_h_value_axis(value.transpose(-1, -2).unsqueeze(0).contiguous())[0].transpose(
        -1, -2
    )


def step(q, k, v, gate, beta, carry, window, enabled, scale, normalize, gate_inputs):
    """Launch one token on private cache buffers; never mutate tensors saved for backward."""
    anchor = carry.anchor
    state, scales, indices = _storage(anchor.values, enabled)
    if enabled:
        metadata = anchor.scales
        codes = (anchor.values.unflatten(-1, (-1, 32)) / metadata.float().unsqueeze(-1)).round()
        state[1] = codes.flatten(-2).transpose(-1, -2).to(torch.int8)
        scales[1] = metadata.transpose(-1, -2)
    else:
        state[1] = anchor.values.transpose(-1, -2)
    heads, keys, values = anchor.values.shape
    d = torch.zeros(2, heads, window, values, device=q.device, dtype=torch.bfloat16)
    kc = torch.zeros(2, heads, window, keys, device=q.device, dtype=torch.bfloat16)
    channel = gate.ndim == 2
    gc = torch.zeros(
        (2, heads, window, keys) if channel else (2, heads, window),
        device=q.device,
        dtype=torch.float32,
    )
    for i, entry in enumerate(carry.entries):
        d[1, :, i] = entry.update.values
        kc[1, :, i] = entry.key.values
        gc[1, :, i] = entry.log_retention
    mixed = torch.cat([x.to(torch.bfloat16).flatten() for x in (q, k, v)])[None]
    if channel:
        a, b, a_log, bias = gate.flatten()[None], beta[None], None, None
    else:
        raw_gate, raw_beta, a_log, bias = gate_inputs
        a, b = raw_gate[None].contiguous(), raw_beta[None].contiguous()
    output = torch.empty(1, heads, values, device=q.device, dtype=torch.bfloat16)
    cursor = carry.cursor
    fused_recurrent_gated_delta_rule_replayssm(
        mixed,
        a,
        b,
        a_log,
        bias,
        scale,
        state,
        d,
        kc,
        gc,
        output,
        indices,
        torch.full((1,), cursor, device=q.device, dtype=torch.int32),
        use_qk_l2norm_in_kernel=normalize,
        quant_state_bits=8 if enabled else 0,
        state_scale=scales,
        hadamard_value_basis=True,
        block_v=64,
        nk=2 if keys == 32 else None,
        _is_kda=channel,
    )
    decoded, metadata = _decoded(state, scales)
    return (
        output[0].float(),
        decoded,
        metadata,
        kc[1, :, cursor].float(),
        d[1, :, cursor].float(),
        gc[1, :, cursor],
    )
