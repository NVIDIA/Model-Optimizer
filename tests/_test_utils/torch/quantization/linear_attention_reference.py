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

"""Small differentiable oracles, independent of FLA and Triton.

These references prioritize explicit arithmetic and gradients over speed. Gates are
natural logarithms; inputs are already normalized/activated. Packed sequence boundaries
are read on the CPU. Accumulation follows the input dtype (use float64 for algebra tests).
"""

from collections.abc import Callable
from itertools import pairwise

import torch

__all__ = ["chunk_gdn_reference", "recurrent_delta_rule_reference", "state_fp8_qdq_reference"]


def state_fp8_qdq_reference(state: torch.Tensor, block_v: int = 64) -> torch.Tensor:
    """Dynamic E4M3 QDQ with identity STE and one scale per ``[Dk, block_v]`` tile.

    ``state`` is in key-first layout ``[..., Dk, Dv]``. Scales and rounding do not
    contribute derivatives. The zero tile uses scale one; partial value tiles are valid.
    """
    if block_v not in (16, 32, 64, 128):
        raise ValueError("block_v must be 16, 32, 64, or 128")
    with torch.no_grad():
        rounded = []
        for tile in state.float().split(block_v, dim=-1):
            amax = tile.abs().amax(dim=(-2, -1), keepdim=True)
            scale = torch.where(amax > 0, amax / 448.0, torch.ones_like(amax))
            rounded.append((tile / scale).clamp(-448, 448).to(torch.float8_e4m3fn).float() * scale)
        quantized = torch.cat(rounded, dim=-1).to(state.dtype)
    return state + (quantized - state).detach()


def _prepare(q, k, v, g, beta, initial_state, cu_seqlens, state_v_first):
    if q.ndim != 4 or k.shape != q.shape or v.shape[:2] != q.shape[:2]:
        raise ValueError("q/k must have shape [B,T,Hk,Dk] and v shape [B,T,Hv,Dv]")
    batch, length, heads, keys = q.shape
    value_heads, values = v.shape[2:]
    if length == 0 or value_heads % heads:
        raise ValueError("nonempty sequences and Hv divisible by Hk are required")
    if beta.shape != (batch, length, value_heads) or g.shape not in (
        beta.shape,
        (*beta.shape, keys),
    ):
        raise ValueError("beta must be [B,T,Hv]; g must be [B,T,Hv] or [B,T,Hv,Dk]")
    if cu_seqlens is None:
        sequences = [(b, 0, length) for b in range(batch)]
    else:
        boundaries = cu_seqlens.tolist()
        if (
            batch != 1
            or len(boundaries) < 2
            or boundaries[0] != 0
            or boundaries[-1] != length
            or any(a >= b for a, b in pairwise(boundaries))
        ):
            raise ValueError("cu_seqlens must partition a packed batch of size one")
        sequences = [(0, a, b) for a, b in pairwise(boundaries)]
    if initial_state is None:
        state = q.new_zeros(len(sequences), value_heads, keys, values)
    else:
        state = initial_state.transpose(-1, -2) if state_v_first else initial_state
        if state.shape != (len(sequences), value_heads, keys, values):
            raise ValueError("initial_state shape does not match sequence/head dimensions")
    return (
        q.repeat_interleave(value_heads // heads, dim=2),
        k.repeat_interleave(value_heads // heads, dim=2),
        state,
        sequences,
    )


def recurrent_delta_rule_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    cu_seqlens: torch.Tensor | None = None,
    state_v_first: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """GDN (scalar decay) or KDA (per-key decay) exact token recurrence.

    Inputs have shapes ``q,k:[B,T,Hk,Dk]``, ``v:[B,T,Hv,Dv]``,
    ``beta:[B,T,Hv]``, and ``g:[B,T,Hv]`` (GDN) or ``[B,T,Hv,Dk]`` (KDA).
    State is ``[N,Hv,Dk,Dv]``, or ``[N,Hv,Dv,Dk]`` with ``state_v_first``.
    Returns output and final state, preserving gradients through the initial state.
    """
    q, k, states, sequences = _prepare(q, k, v, g, beta, initial_state, cu_seqlens, state_v_first)
    scale = q.shape[-1] ** -0.5 if scale is None else scale
    outputs, finals = [], []
    for n, (b, start, end) in enumerate(sequences):
        state = states[n]
        sequence_output = []
        for t in range(start, end):
            decay = g[b, t].exp()
            if g.ndim == 3:
                decay = decay.unsqueeze(-1)
            decayed = state * decay.unsqueeze(-1)
            residual = v[b, t] - (k[b, t].unsqueeze(-1) * decayed).sum(-2)
            update = beta[b, t].unsqueeze(-1) * residual
            state = decayed + k[b, t].unsqueeze(-1) * update.unsqueeze(-2)
            sequence_output.append((q[b, t].unsqueeze(-1) * state).sum(-2) * scale)
        outputs.append(torch.stack(sequence_output))
        finals.append(state)
    output = torch.stack(outputs) if cu_seqlens is None else torch.cat(outputs).unsqueeze(0)
    final = torch.stack(finals)
    return output, final.transpose(-1, -2) if state_v_first else final


def chunk_gdn_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    chunk_size: int = 64,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    cu_seqlens: torch.Tensor | None = None,
    state_v_first: bool = False,
    state_qdq: bool = False,
    state_qdq_block_v: int = 64,
    w_quantizer: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact GDN chunk algebra with optional state/W fake quantization.

    The solve is a unit-lower triangular solve. ``w_quantizer`` sees the complete
    materialized ``[B,T,Hv,Dk]`` WY operand once, with its own autograd semantics.
    State QDQ occurs on the initial state and each chunk's final state, after readout.
    """
    if g.ndim != 3 or chunk_size <= 0:
        raise ValueError("chunk_gdn_reference requires scalar GDN gates and positive chunk_size")
    q, k, states, sequences = _prepare(q, k, v, g, beta, initial_state, cu_seqlens, state_v_first)
    scale = q.shape[-1] ** -0.5 if scale is None else scale
    chunks, all_w = [], []
    for n, (b, start, end) in enumerate(sequences):
        for lo in range(start, end, chunk_size):
            hi = min(lo + chunk_size, end)
            qc, kc, vc = (x[b, lo:hi].transpose(0, 1) for x in (q, k, v))
            gc = g[b, lo:hi].transpose(0, 1).cumsum(-1)
            bc = beta[b, lo:hi].transpose(0, 1).unsqueeze(-1)
            # Mask before exp: upper-triangle positive differences can overflow for long decay.
            causal = torch.ones(hi - lo, hi - lo, device=q.device, dtype=torch.bool).tril()
            decay = (gc.unsqueeze(-1) - gc.unsqueeze(-2)).masked_fill(~causal, 0).exp()
            gram = kc @ kc.transpose(-1, -2)
            lower = (bc * gram * decay).tril(-1)
            matrix = lower + torch.eye(hi - lo, device=q.device, dtype=q.dtype)
            rhs = torch.cat((bc * vc, bc * kc * gc.exp().unsqueeze(-1)), dim=-1)
            solved = torch.linalg.solve_triangular(matrix, rhs, upper=False, unitriangular=True)
            u, w = solved.split((v.shape[-1], k.shape[-1]), dim=-1)
            chunks.append((n, qc, kc, gc, decay, u))
            all_w.append(w.transpose(0, 1))
    w = torch.cat(all_w).reshape(q.shape)
    if w_quantizer is not None:
        w = w_quantizer(w)
    w = w.reshape(-1, *w.shape[2:])
    outputs, finals, offset, previous_n = [], [], 0, -1
    for n, qc, kc, gc, decay, u in chunks:
        if n != previous_n:
            state = states[n]
            if state_qdq:
                state = state_fp8_qdq_reference(state, state_qdq_block_v)
            previous_n = n
        length = qc.shape[1]
        wc = w[offset : offset + length].transpose(0, 1)
        offset += length
        updated_values = u - wc @ state  # state_read
        local_scores = ((qc * scale) @ kc.transpose(-1, -2) * decay).tril()
        output = (qc * (scale * gc.exp()).unsqueeze(-1)) @ state
        output = output + local_scores @ updated_values  # local_readout
        outputs.append(output.transpose(0, 1))
        weighted_keys = kc * (gc[..., -1:] - gc).exp().unsqueeze(-1)
        state = state * gc[..., -1].exp()[:, None, None]
        state = state + weighted_keys.transpose(-1, -2) @ updated_values  # state_update
        if state_qdq:
            state = state_fp8_qdq_reference(state, state_qdq_block_v)
        if len(finals) <= n:
            finals.append(state)
        else:
            finals[n] = state
    output = torch.cat(outputs).reshape(*v.shape)
    final = torch.stack(finals)
    return output, final.transpose(-1, -2) if state_v_first else final
