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

from itertools import pairwise

import torch

__all__ = [
    "chunk_delta_rule_reference",
    "recurrent_delta_rule_reference",
    "state_qdq_reference",
]


def state_qdq_reference(state: torch.Tensor, block_v: int = 64, state_format: str = "fp8_e4m3"):
    """Dynamic state-tile QDQ with detached scales and identity STE."""
    if state_format not in ("fp8_e4m3", "int8"):
        raise ValueError("State format must be fp8_e4m3 or int8")
    if block_v not in (16, 32, 64, 128):
        raise ValueError("block_v must be 16, 32, 64, or 128")
    with torch.no_grad():
        rounded = []
        for tile in state.float().split(block_v, dim=-1):
            amax = tile.abs().amax(dim=(-2, -1), keepdim=True)
            if state_format == "int8":
                # CUDA TensorQuantizer uses a quantization multiplier and zeros tiny groups.
                tiny = amax < 2**-24
                quant_scale = 127.0 / torch.where(tiny, torch.ones_like(amax), amax)
                codes = (tile * quant_scale).round().clamp(-127, 127)
                decoded = torch.where(tiny, 0.0, codes / quant_scale)
            else:
                safe_amax = torch.where(amax <= 2**-24, torch.ones_like(amax), amax)
                quant_scale = torch.div(448.0, safe_amax)
                scale = quant_scale.reciprocal()
                codes = (tile * quant_scale).clamp(-448, 448).to(torch.float8_e4m3fn).float()
                decoded = codes * scale
            rounded.append(decoded)
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


def chunk_gdn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    chunk_size: int = 64,
    scale: float | None = None,
    initial_state: torch.Tensor,
    state_qdq: bool = False,
    state_qdq_block_v: int = 64,
    state_format: str = "fp8_e4m3",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute one prepared, nonempty GDN prefix [T,H,D] with a key-first state."""
    scale = q.shape[-1] ** -0.5 if scale is None else scale
    state = initial_state
    if state_qdq:
        state = state_qdq_reference(state, state_qdq_block_v, state_format)
    pieces = []
    for lo in range(0, len(q), chunk_size):
        hi = min(lo + chunk_size, len(q))
        qc, kc, vc = (x[lo:hi].transpose(0, 1) for x in (q, k, v))
        gc = g[lo:hi].transpose(0, 1).cumsum(-1)
        bc = beta[lo:hi].transpose(0, 1).unsqueeze(-1)
        # Mask before exp: upper-triangle positive differences can overflow for long decay.
        causal = torch.ones(hi - lo, hi - lo, device=q.device, dtype=torch.bool).tril()
        decay = (gc.unsqueeze(-1) - gc.unsqueeze(-2)).masked_fill(~causal, 0).exp()
        lower = (bc * (kc @ kc.transpose(-1, -2)) * decay).tril(-1)
        matrix = lower + torch.eye(hi - lo, device=q.device, dtype=q.dtype)
        gate = gc.exp().unsqueeze(-1)
        rhs = torch.cat((bc * vc, bc * kc * gate), dim=-1)
        solved = torch.linalg.solve_triangular(matrix, rhs, upper=False, unitriangular=True)
        u, w = solved.split((v.shape[-1], k.shape[-1]), dim=-1)
        updated_values = u - w @ state
        local_scores = ((qc * scale @ kc.transpose(-1, -2)) * decay).tril()
        output = (qc * (scale * gc.exp()).unsqueeze(-1)) @ state
        pieces.append((output + local_scores @ updated_values).transpose(0, 1))
        weighted_keys = kc * (gc[..., -1:] - gc).exp().unsqueeze(-1)
        state = state * gc[..., -1].exp()[:, None, None]
        state = state + weighted_keys.transpose(-1, -2) @ updated_values
        if state_qdq:
            state = state_qdq_reference(state, state_qdq_block_v, state_format)
    return torch.cat(pieces), state


def chunk_kda(
    q,
    k,
    v,
    g,
    beta,
    *,
    chunk_size=64,
    scale=None,
    initial_state,
    state_qdq=False,
    state_qdq_block_v=64,
    state_format="fp8_e4m3",
):
    """Compute one prepared, nonempty KDA prefix [T,H,D] with a key-first state."""
    scale = q.shape[-1] ** -0.5 if scale is None else scale
    state = initial_state
    if state_qdq:
        state = state_qdq_reference(state, state_qdq_block_v, state_format)
    pieces = []
    for lo in range(0, len(q), chunk_size):
        hi = min(lo + chunk_size, len(q))
        qc, kc, vc = (x[lo:hi].transpose(0, 1) for x in (q, k, v))
        prefix = g[lo:hi].transpose(0, 1).cumsum(-2)
        bc = beta[lo:hi].transpose(0, 1).unsqueeze(-1)
        gate = prefix.exp()
        lower_rows, score_rows = [], []
        for row in range(hi - lo):
            decayed_keys = (
                kc[..., : row + 1, :]
                * (prefix[..., row : row + 1, :] - prefix[..., : row + 1, :]).exp()
            )
            right = decayed_keys.transpose(-1, -2)
            left = kc[..., row : row + 1, :] * bc[..., row : row + 1, :]
            interaction = left @ right
            score = qc[..., row : row + 1, :] * scale @ right
            padding = hi - lo - row - 1
            lower_rows.append(torch.nn.functional.pad(interaction, (0, padding)))
            score_rows.append(torch.nn.functional.pad(score, (0, padding)))
        lower = torch.cat(lower_rows, dim=-2).tril(-1)
        scores = torch.cat(score_rows, dim=-2)
        identity = torch.eye(hi - lo, device=q.device, dtype=q.dtype).expand_as(lower)
        inverse = torch.linalg.solve_triangular(
            identity + lower, identity, upper=False, unitriangular=True
        )
        u = inverse @ (bc * vc)
        w = inverse @ (bc * kc * gate)
        updated = u - w @ state
        pieces.append(((qc * scale * gate) @ state + scores @ updated).transpose(0, 1))
        weighted_keys = kc * (prefix[..., -1:, :] - prefix).exp()
        state = state * gate[..., -1, :, None] + weighted_keys.transpose(-1, -2) @ updated
        if state_qdq:
            state = state_qdq_reference(state, state_qdq_block_v, state_format)
    return torch.cat(pieces), state


def chunk_delta_rule_reference(
    q, k, v, g, beta, *, initial_state=None, cu_seqlens=None, state_v_first=False, chunk_size=64
):
    """Evaluate GDN/KDA chunk algebra independently for each packed or batched sequence."""
    q, k, states, sequences = _prepare(q, k, v, g, beta, initial_state, cu_seqlens, state_v_first)
    chunk = chunk_kda if g.ndim == 4 else chunk_gdn
    outputs, finals = [], []
    for n, (b, start, end) in enumerate(sequences):
        output, state = chunk(
            *(x[b, start:end] for x in (q, k, v, g, beta)),
            initial_state=states[n],
            chunk_size=chunk_size,
        )
        outputs.append(output)
        finals.append(state)
    output = torch.stack(outputs) if cu_seqlens is None else torch.cat(outputs).unsqueeze(0)
    final = torch.stack(finals)
    return output, final.transpose(-1, -2) if state_v_first else final
