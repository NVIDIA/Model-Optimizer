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

"""Differentiable explicit per-sequence prefill/decode phase handoff."""

from contextlib import contextmanager
from itertools import pairwise

import torch

from ._chunk_prefill import chunk_gdn, chunk_kda
from .decode import recurrent_decode

__all__ = ["linear_attention_training_phase"]


def _lengths(values):
    if isinstance(values, torch.Tensor):
        values = values.tolist()
    values = tuple(values)
    if any(type(value) is not int or value < 0 for value in values):
        raise ValueError("Prefill lengths must be nonnegative integers")
    return values


@contextmanager
def linear_attention_training_phase(model, prefill_lengths):
    """Supply explicit sequence phases through forward and checkpointed backward.

    Runtime phase metadata is local to the converted layers and restored on exit.
    Keep this context active through backward when activation checkpointing is used.
    """
    # The plugin imports this numerical package during quantization initialization.
    from ..plugins.linear_attention import _LinearAttentionQuantMixin

    lengths = _lengths(prefill_lengths)
    layers = [
        m
        for m in model.modules()
        if isinstance(m, _LinearAttentionQuantMixin)
        and m.linear_attention_config.decode is not None
    ]
    if not layers:
        raise ValueError("The model has no converted decode-aware linear-attention layers")
    previous = [getattr(m, "_linear_attention_prefill_lengths", None) for m in layers]
    try:
        for module in layers:
            module._linear_attention_prefill_lengths = lengths
        yield model
    finally:
        for module, original in zip(layers, previous):
            module._linear_attention_prefill_lengths = original


def _decode_prefill(
    q,
    k,
    v,
    g,
    beta,
    *,
    policy,
    state_qdq,
    state_format,
    scale,
    initial_state,
    output_final_state,
    cu_seqlens,
    cu_seqlens_cpu,
    state_v_first,
    output_dtype,
    prefill_lengths,
):
    if prefill_lengths is None:
        raise ValueError("Decode-aware training requires explicit per-sequence prefill lengths")
    if q.ndim != 4 or k.shape != q.shape or v.ndim != 4 or v.shape[:2] != q.shape[:2]:
        raise ValueError("q/k and v must have compatible [B,T,H,D] shapes")
    batch, length, key_heads, keys = q.shape
    heads, values = v.shape[2:]
    if batch < 1 or key_heads < 1 or heads % key_heads or beta.shape != (batch, length, heads):
        raise ValueError("Invalid batch/head dimensions or beta shape")
    if g.shape not in (beta.shape, (*beta.shape, keys)):
        raise ValueError("Invalid GDN/KDA log-retention shape")
    q, k = (x.repeat_interleave(heads // key_heads, dim=2) for x in (q, k))
    boundaries = cu_seqlens_cpu if cu_seqlens_cpu is not None else cu_seqlens
    if boundaries is None:
        sequences = [(b, 0, length) for b in range(batch)]
    else:
        bounds = _lengths(boundaries)
        if (
            batch != 1
            or len(bounds) < 2
            or bounds[0] != 0
            or bounds[-1] != length
            or any(a > b for a, b in pairwise(bounds))
        ):
            raise ValueError(
                "Packed boundaries must partition a batch of one, allowing empty entries"
            )
        sequences = [(0, a, b) for a, b in pairwise(bounds)]
    prefixes = _lengths(prefill_lengths)
    if len(prefixes) != len(sequences) or any(
        p > end - start for p, (_, start, end) in zip(prefixes, sequences)
    ):
        raise ValueError("Supply one valid prefill length per sequence")
    if initial_state is None:
        states = q.new_zeros(len(sequences), heads, keys, values)
    else:
        states = initial_state.transpose(-1, -2) if state_v_first else initial_state
        states = states.to(q.dtype)
        if states.shape != (len(sequences), heads, keys, values):
            raise ValueError("Initial state shape does not match sequence/head dimensions")
    active = [n for n, prefix in enumerate(prefixes) if prefix]
    prefix_outputs, prefix_states = {}, {}
    if active:
        packed = [
            torch.cat(
                [
                    tensor[sequences[n][0], sequences[n][1] : sequences[n][1] + prefixes[n]]
                    for n in active
                ]
            ).unsqueeze(0)
            for tensor in (q, k, v, g, beta)
        ]
        offsets = [0]
        for n in active:
            offsets.append(offsets[-1] + prefixes[n])
        prefix_fn = chunk_kda if g.ndim == 4 else chunk_gdn
        with torch.autocast(device_type=q.device.type, enabled=False):
            output, final = prefix_fn(
                *packed,
                state_qdq=state_qdq and policy.decode.prefill_state_qdq,
                state_format=state_format,
                scale=scale,
                initial_state=states[active],
                cu_seqlens=torch.tensor(offsets),
                state_v_first=False,
                chunk_size=policy.chunk_size,
                state_qdq_block_v=policy.state.block_v,
            )
        for index, n in enumerate(active):
            prefix_outputs[n] = output[0, offsets[index] : offsets[index + 1]]
            prefix_states[n] = final[index]
    outputs, finals = [], []
    for n, (b, start, end) in enumerate(sequences):
        split = start + prefixes[n]
        suffix, carry = recurrent_decode(
            *(x[b, split:end] for x in (q, k, v, g, beta)),
            config=policy.decode,
            state_qdq=state_qdq,
            state_format=state_format,
            block_v=policy.state.block_v,
            initial_state=prefix_states.get(n, states[n]),
            position=prefixes[n],
            scale=scale,
        )
        prefix = prefix_outputs.get(n, q.new_empty(0, heads, values))
        outputs.append(torch.cat((prefix, suffix)))
        finals.append(carry.reconstruct())
    output = torch.stack(outputs) if boundaries is None else torch.cat(outputs).unsqueeze(0)
    final = torch.stack(finals)
    if state_v_first:
        final = final.transpose(-1, -2)
    return output.to(output_dtype), final if output_final_state else None
