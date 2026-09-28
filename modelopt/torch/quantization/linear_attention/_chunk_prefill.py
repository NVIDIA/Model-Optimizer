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

"""Exact chunked prefix computation for decode-aware training."""

import torch

from .utils import _prepare, _state_qdq

__all__ = []


def chunk_gdn(
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
    state_format: str = "fp8_e4m3",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute a GDN prefix with optional QDQ on the initial state and chunk writes."""
    if g.ndim != 3 or chunk_size <= 0:
        raise ValueError("chunk_gdn requires scalar GDN gates and positive chunk_size")
    q, k, states, sequences = _prepare(q, k, v, g, beta, initial_state, cu_seqlens, state_v_first)
    scale = q.shape[-1] ** -0.5 if scale is None else scale
    outputs, finals = [], []
    for n, (b, start, end) in enumerate(sequences):
        state = states[n]
        if state_qdq:
            state = _state_qdq(state, state_qdq_block_v, state_format)
        pieces = []
        for lo in range(start, end, chunk_size):
            hi = min(lo + chunk_size, end)
            qc, kc, vc = (x[b, lo:hi].transpose(0, 1) for x in (q, k, v))
            gc = g[b, lo:hi].transpose(0, 1).cumsum(-1)
            bc = beta[b, lo:hi].transpose(0, 1).unsqueeze(-1)
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
                state = _state_qdq(state, state_qdq_block_v, state_format)
        outputs.append(torch.cat(pieces))
        finals.append(state)
    output = torch.stack(outputs) if cu_seqlens is None else torch.cat(outputs).unsqueeze(0)
    final = torch.stack(finals)
    return output, final.transpose(-1, -2) if state_v_first else final


def chunk_kda(
    q,
    k,
    v,
    g,
    beta,
    *,
    chunk_size=64,
    scale=None,
    initial_state=None,
    cu_seqlens=None,
    state_v_first=False,
    state_qdq=False,
    state_qdq_block_v=64,
    state_format="fp8_e4m3",
):
    """Compute a KDA prefix with optional QDQ on the initial state and chunk writes."""
    if g.ndim != 4 or chunk_size <= 0:
        raise ValueError("chunk_kda requires per-key gates and positive chunk_size")
    q, k, states, sequences = _prepare(q, k, v, g, beta, initial_state, cu_seqlens, state_v_first)
    scale = q.shape[-1] ** -0.5 if scale is None else scale
    outputs, finals = [], []
    for n, (b, start, end) in enumerate(sequences):
        state = states[n]
        if state_qdq:
            state = _state_qdq(state, state_qdq_block_v, state_format)
        pieces = []
        for lo in range(start, end, chunk_size):
            hi = min(lo + chunk_size, end)
            qc, kc, vc = (x[b, lo:hi].transpose(0, 1) for x in (q, k, v))
            prefix = g[b, lo:hi].transpose(0, 1).cumsum(-2)
            bc = beta[b, lo:hi].transpose(0, 1).unsqueeze(-1)
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
                state = _state_qdq(state, state_qdq_block_v, state_format)
        outputs.append(torch.cat(pieces))
        finals.append(state)
    output = torch.stack(outputs) if cu_seqlens is None else torch.cat(outputs).unsqueeze(0)
    final = torch.stack(finals)
    return output, final.transpose(-1, -2) if state_v_first else final
