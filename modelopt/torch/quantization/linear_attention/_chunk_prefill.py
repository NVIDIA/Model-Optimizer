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

from collections.abc import Callable

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
    w_quantizer: Callable[[torch.Tensor], torch.Tensor] | None = None,
    matmul: Callable[[str, torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    arithmetic: Callable[[str, torch.Tensor], torch.Tensor] | None = None,
    inverse_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact GDN chunk algebra with optional state/W fake quantization.

    The solve is a unit-lower triangular solve. ``w_quantizer`` sees the complete
    materialized ``[B,T,Hv,Dk]`` WY operand once, with its own autograd semantics.
    State QDQ occurs on the initial state and each chunk's final state, after readout.
    Optional callbacks expose the eight named matmuls and elementwise boundaries.
    Matmul callbacks receive conventional ``lhs @ rhs`` shapes, without padding.
    """
    if g.ndim != 3 or chunk_size <= 0:
        raise ValueError("chunk_gdn requires scalar GDN gates and positive chunk_size")
    q, k, states, sequences = _prepare(q, k, v, g, beta, initial_state, cu_seqlens, state_v_first)
    scale = q.shape[-1] ** -0.5 if scale is None else scale
    mm = matmul if matmul is not None else lambda name, lhs, rhs: lhs @ rhs
    arithmetic = arithmetic if arithmetic is not None else lambda name, x: x
    chunks, all_w = [], []
    for n, (b, start, end) in enumerate(sequences):
        for lo in range(start, end, chunk_size):
            hi = min(lo + chunk_size, end)
            qc, kc, vc = (x[b, lo:hi].transpose(0, 1) for x in (q, k, v))
            gc = arithmetic("gate_prefix", g[b, lo:hi].transpose(0, 1).cumsum(-1))
            bc = beta[b, lo:hi].transpose(0, 1).unsqueeze(-1)
            # Mask before exp: upper-triangle positive differences can overflow for long decay.
            causal = torch.ones(hi - lo, hi - lo, device=q.device, dtype=torch.bool).tril()
            decay = arithmetic(
                "gate_exp", (gc.unsqueeze(-1) - gc.unsqueeze(-2)).masked_fill(~causal, 0).exp()
            )
            if matmul is None:
                lower = (bc * (kc @ kc.transpose(-1, -2)) * decay).tril(-1)
            else:
                lower = (mm("key_interaction", bc * kc, kc.transpose(-1, -2)) * decay).tril(-1)
            matrix = lower + torch.eye(hi - lo, device=q.device, dtype=q.dtype)
            gate = arithmetic("gate_exp", gc.exp()).unsqueeze(-1)
            if matmul is None and inverse_fn is None:
                rhs = torch.cat((bc * vc, bc * kc * gate), dim=-1)
                solved = torch.linalg.solve_triangular(matrix, rhs, upper=False, unitriangular=True)
                u, w = solved.split((v.shape[-1], k.shape[-1]), dim=-1)
            else:
                inverse = (
                    inverse_fn(lower)
                    if inverse_fn is not None
                    else torch.linalg.solve_triangular(
                        matrix,
                        torch.eye(hi - lo, device=q.device, dtype=q.dtype).expand_as(matrix),
                        upper=False,
                        unitriangular=True,
                    )
                )
                u = mm("wy_value", inverse, bc * vc)
                w = mm("wy_key", inverse, bc * kc * gate)
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
                state = _state_qdq(state, state_qdq_block_v, state_format)
            previous_n = n
        length = qc.shape[1]
        wc = w[offset : offset + length].transpose(0, 1)
        offset += length
        updated_values = arithmetic("value_residual", u - mm("state_read", wc, state))
        local_scores = (mm("output_score", qc * scale, kc.transpose(-1, -2)) * decay).tril()
        output = mm(
            "output_state", qc * (scale * arithmetic("gate_exp", gc.exp())).unsqueeze(-1), state
        )
        output = arithmetic("output_add", output + mm("output_value", local_scores, updated_values))
        outputs.append(output.transpose(0, 1))
        weighted_keys = kc * arithmetic("gate_exp", (gc[..., -1:] - gc).exp()).unsqueeze(-1)
        state = arithmetic(
            "state_decay", state * arithmetic("gate_exp", gc[..., -1].exp())[:, None, None]
        )
        state = arithmetic(
            "state_add", state + mm("state_update", weighted_keys.transpose(-1, -2), updated_values)
        )
        if state_qdq:
            state = _state_qdq(state, state_qdq_block_v, state_format)
        if len(finals) <= n:
            finals.append(state)
        else:
            finals[n] = state
    output = torch.cat(outputs).reshape(*v.shape)
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
    w_quantizer=None,
    matmul=None,
    inverse_fn=None,
):
    """KDA chunk computation with causal per-channel decays and optional operand QDQ.

    The eight-site callback receives conventional three-dimensional matmul
    operands. Interaction sites are invoked one query row at a time with decay
    already applied to the right-hand keys. This avoids inverse-prefix factors.
    """
    if g.ndim != 4 or chunk_size <= 0:
        raise ValueError("chunk_kda requires per-key gates and positive chunk_size")
    q, k, states, sequences = _prepare(q, k, v, g, beta, initial_state, cu_seqlens, state_v_first)
    scale = q.shape[-1] ** -0.5 if scale is None else scale
    mm = matmul if matmul is not None else lambda name, lhs, rhs: lhs @ rhs
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
                interaction = mm("key_interaction", left, right)
                score = mm("output_score", qc[..., row : row + 1, :] * scale, right)
                # Zero padding follows the computation and never contributes a scale statistic.
                padding = hi - lo - row - 1
                lower_rows.append(torch.nn.functional.pad(interaction, (0, padding)))
                score_rows.append(torch.nn.functional.pad(score, (0, padding)))
            lower = torch.cat(lower_rows, dim=-2).tril(-1)
            scores = torch.cat(score_rows, dim=-2)
            identity = torch.eye(hi - lo, device=q.device, dtype=q.dtype).expand_as(lower)
            inverse = (
                inverse_fn(lower)
                if inverse_fn is not None
                else torch.linalg.solve_triangular(
                    identity + lower, identity, upper=False, unitriangular=True
                )
            )
            u = mm("wy_value", inverse, bc * vc)
            w = mm("wy_key", inverse, bc * kc * gate)
            if w_quantizer is not None:
                w = w_quantizer(w)
            updated = u - mm("state_read", w, state)
            pieces.append(
                (
                    mm("output_state", qc * scale * gate, state)
                    + mm("output_value", scores, updated)
                ).transpose(0, 1)
            )
            weighted_keys = kc * (prefix[..., -1:, :] - prefix).exp()
            state = state * gate[..., -1, :, None] + mm(
                "state_update", weighted_keys.transpose(-1, -2), updated
            )
            if state_qdq:
                state = _state_qdq(state, state_qdq_block_v, state_format)
        outputs.append(torch.cat(pieces))
        finals.append(state)
    output = torch.stack(outputs) if cu_seqlens is None else torch.cat(outputs).unsqueeze(0)
    final = torch.stack(finals)
    return output, final.transpose(-1, -2) if state_v_first else final

