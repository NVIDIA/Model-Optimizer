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

"""Shared helpers for linear-attention quantization."""

from itertools import pairwise

import torch

from ..nn import TensorQuantizer

__all__ = []


def validate_gdn_quantizer(
    quantizer: TensorQuantizer,
    *,
    name: str,
    num_bits: tuple[int | tuple[int, int], ...] = ((4, 3),),
) -> None:
    """Check supported formats and the custom backward's identity STE."""
    if not isinstance(quantizer, TensorQuantizer):
        raise ValueError(f"{name} requires a single TensorQuantizer")
    if not (
        quantizer._dynamic
        and quantizer.num_bits in num_bits
        and (quantizer.num_bits != 8 or (not quantizer.unsigned and quantizer.narrow_range))
        and quantizer.block_sizes is None
        and quantizer.fake_quant
        and quantizer._pass_through_bwd
        and not quantizer.rotate_is_enabled
        and quantizer.pre_quant_scale is None
        and quantizer.backend is None
        and not quantizer._bias
        and not quantizer._use_constant_amax
    ):
        raise ValueError(
            f"{name} supports only dynamic fake quantization with num_bits in {num_bits}, "
            "pass_through_bwd=True, no block_sizes, rotation, pre-scaling, bias, constant "
            "amax, or custom backend. Other gradient rules and formats are not implemented."
        )


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

def _state_qdq(state: torch.Tensor, block_v: int = 64, state_format: str = "fp8_e4m3"):
    """Dynamic state-tile QDQ with detached scales and identity STE."""
    if state_format not in ("fp8_e4m3", "int8"):
        raise ValueError("State format must be fp8_e4m3 or int8")
    if block_v not in (16, 32, 64, 128):
        raise ValueError("block_v must be 16, 32, 64, or 128")
    with torch.no_grad():
        rounded = []
        for tile in state.float().split(block_v, dim=-1):
            amax = tile.abs().amax(dim=(-2, -1), keepdim=True)
            limit = 127.0 if state_format == "int8" else 448.0
            scale = torch.where(amax > 0, amax / limit, torch.ones_like(amax))
            normalized = (tile / scale).clamp(-limit, limit)
            codes = (
                normalized.round()
                if state_format == "int8"
                else normalized.to(torch.float8_e4m3fn).float()
            )
            rounded.append(codes * scale)
        quantized = torch.cat(rounded, dim=-1).to(state.dtype)
    return state + (quantized - state).detach()
