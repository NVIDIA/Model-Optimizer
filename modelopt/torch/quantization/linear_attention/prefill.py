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

"""Exact GDN chunk prefill with operand QDQ at its eight real GEMMs."""

from itertools import pairwise
from numbers import Integral

import torch

from .config import LinearAttentionSolveConfig
from .matmul import GDNMatmulSites
from .solve import triangular_inverse

__all__ = []


def _as_lengths(values, name):
    if isinstance(values, torch.Tensor):
        values = values.tolist()
    values = tuple(values)
    if any(
        isinstance(value, bool) or not isinstance(value, Integral) or value < 0 for value in values
    ):
        raise ValueError(f"{name} must contain nonnegative integers")
    return tuple(int(value) for value in values)


def _sequences(q, cu_seqlens, cu_seqlens_cpu, sequence_lengths):
    batch, length = q.shape[:2]
    if hasattr(cu_seqlens_cpu, "resolve"):
        boundaries = cu_seqlens_cpu.resolve(cu_seqlens)
    else:
        boundaries = cu_seqlens_cpu if cu_seqlens_cpu is not None else cu_seqlens
    if boundaries is None:
        segments = tuple((index, 0, length) for index in range(batch))
    else:
        bounds = _as_lengths(boundaries, "Packed boundaries")
        if batch != 1 or len(bounds) < 2 or bounds[0] != 0 or bounds[-1] != length:
            raise ValueError("Packed boundaries must partition a batch of one")
        if any(left > right for left, right in pairwise(bounds)):
            raise ValueError("Packed boundaries must be nondecreasing")
        segments = tuple((0, left, right) for left, right in pairwise(bounds))
    valid = (
        tuple(end - start for _, start, end in segments)
        if sequence_lengths is None
        else _as_lengths(sequence_lengths, "Sequence lengths")
    )
    if len(valid) != len(segments) or any(
        size > end - start for size, (_, start, end) in zip(valid, segments)
    ):
        raise ValueError("Supply one sequence length within each physical segment")
    return segments, valid


def _chunk_gdn(q, k, v, g, beta, state, sites, solve, scale):
    """Run one sequence, carrying an FP32 [H,K,V] state across 64-token chunks."""

    def mm(name, lhs, rhs):
        return sites.matmul(name, lhs.unsqueeze(0), rhs.unsqueeze(0))[0]

    outputs = []
    for start in range(0, len(q), 64):
        stop = min(start + 64, len(q))
        count = stop - start

        def padded(value):
            value = value[start:stop]
            return torch.nn.functional.pad(value, (0, 0) * (value.ndim - 1) + (0, 64 - count))

        # [T,H,D] -> [H,64,D]; scalar gates/beta become [H,64].
        qc, kc, vc = (padded(value).transpose(0, 1) for value in (q, k, v))
        gc = padded(g.unsqueeze(-1)).squeeze(-1).transpose(0, 1).cumsum(-1)
        bc = padded(beta.unsqueeze(-1)).squeeze(-1).transpose(0, 1)
        gate = gc.exp()
        causal = torch.ones(64, 64, dtype=torch.bool, device=q.device).tril()
        # Mask positive upper-triangle differences before exp; strong forgetting can
        # otherwise overflow values that the causal mask discards only afterward.
        relative = (gc.unsqueeze(-1) - gc.unsqueeze(-2)).masked_fill(~causal, 0)
        decay = relative.exp().masked_fill(~causal, 0)
        lower = (mm("key_interaction", bc.unsqueeze(-1) * kc, kc) * decay).tril(-1)
        scores = (mm("output_score", qc * scale, kc) * decay).tril()
        inverse = triangular_inverse(lower, solve)
        u = mm("wy_value", inverse, (bc.unsqueeze(-1) * vc).transpose(-1, -2))
        w = mm(
            "wy_key",
            inverse,
            (bc.unsqueeze(-1) * kc * gate.unsqueeze(-1)).transpose(-1, -2),
        )
        state_t = state.transpose(-1, -2)
        updated = u - mm("state_read", w, state_t)
        memory = mm("output_state", qc * scale * gate.unsqueeze(-1), state_t)
        local = mm("output_value", scores, updated.transpose(-1, -2))
        outputs.append((memory + local)[:, :count].transpose(0, 1))
        weighted_keys = kc * (gc[..., -1:] - gc).exp().unsqueeze(-1)
        update = mm("state_update", weighted_keys.transpose(-1, -2), updated.transpose(-1, -2))
        state = state * gate[..., -1, None, None] + update
    return torch.cat(outputs) if outputs else q.new_zeros((0, v.shape[1], v.shape[-1])), state


def gdn_prefill_qat(
    q,
    k,
    v,
    g,
    beta,
    *,
    sites: GDNMatmulSites,
    solve=None,
    scale=None,
    initial_state=None,
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,
    cu_seqlens=None,
    cu_seqlens_cpu=None,
    state_v_first=False,
    chunk_size=64,
    prefill_lengths=None,
    sequence_lengths=None,
):
    """Execute GDN prefill QAT without inventing non-GEMM interaction operands."""
    if chunk_size != 64 or q.device.type != "cuda":
        raise ValueError("GDN prefill operand QDQ requires chunk_size=64 on CUDA")
    if any(value.dtype != torch.bfloat16 for value in (q, k, v)):
        raise ValueError("GDN prefill operand QDQ requires BF16 Q/K/V projections")
    if q.ndim != 4 or k.shape != q.shape or v.ndim != 4 or v.shape[:2] != q.shape[:2]:
        raise ValueError("Q/K/V must have compatible [B,T,H,D] shapes")
    batch, length, key_heads, keys = q.shape
    heads, values = v.shape[2:]
    if heads % key_heads or g.shape != (batch, length, heads) or beta.shape != g.shape:
        raise ValueError("GDN prefill requires scalar gates and compatible grouped heads")
    sites.validate()
    solve = LinearAttentionSolveConfig() if solve is None else solve
    output_dtype = q.dtype
    q, k, v, g, beta = (value.float() for value in (q, k, v, g, beta))
    if use_qk_l2norm_in_kernel:
        q, k = (value * (value.square().sum(-1, keepdim=True) + 1e-6).rsqrt() for value in (q, k))
    q, k = (value.repeat_interleave(heads // key_heads, dim=2) for value in (q, k))
    scale = keys**-0.5 if scale is None else scale
    segments, valid = _sequences(q, cu_seqlens, cu_seqlens_cpu, sequence_lengths)
    if prefill_lengths is not None and _as_lengths(prefill_lengths, "Prefill lengths") != valid:
        raise ValueError("GDN operand QDQ supports prefill only; every valid token must be prefix")
    if initial_state is None:
        states = q.new_zeros(len(segments), heads, keys, values)
    else:
        states = initial_state.transpose(-1, -2) if state_v_first else initial_state
        if states.shape != (len(segments), heads, keys, values):
            raise ValueError("Initial state must provide one [H,K,V] matrix per sequence")
        states = states.float()
    output = q.new_zeros(batch, length, heads, values)
    final_states = []
    for index, ((batch_index, start, _), size) in enumerate(zip(segments, valid)):
        sequence_output, final = _chunk_gdn(
            q[batch_index, start : start + size],
            k[batch_index, start : start + size],
            v[batch_index, start : start + size],
            g[batch_index, start : start + size],
            beta[batch_index, start : start + size],
            states[index],
            sites,
            solve,
            scale,
        )
        output[batch_index, start : start + size] = sequence_output
        final_states.append(final)
    final = torch.stack(final_states)
    if state_v_first:
        final = final.transpose(-1, -2)
    return output.to(output_dtype), final if output_final_state else None
