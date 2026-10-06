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

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from ..nn import TensorQuantizer

__all__ = []

_STATE_FORMATS: dict[int | tuple[int, int], str] = {(4, 3): "fp8_e4m3", 8: "int8"}


class _ForwardValue(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, rounded):
        return rounded.to(value.dtype)

    @staticmethod
    def backward(ctx, grad):
        return grad, None


def forward_value(value, rounded):
    # Unlike value + (rounded - value).detach(), this cannot lose low bits by cancellation.
    return _ForwardValue.apply(value, rounded)


def validate_gdn_quantizer(
    quantizer: TensorQuantizer,
    *,
    name: str,
    num_bits: tuple[int | tuple[int, int], ...] = ((4, 3),),
    block_sizes: tuple[int, ...] = (),
) -> None:
    """Check supported formats and the custom backward's identity STE."""
    # Numerical helpers load while QuantizeConfig initializes; defer the quantizer import.
    from ..nn import TensorQuantizer

    if not isinstance(quantizer, TensorQuantizer):
        raise ValueError(f"{name} requires a single TensorQuantizer")
    if not (
        quantizer._dynamic
        and quantizer.num_bits in num_bits
        and (quantizer.num_bits != 8 or (not quantizer.unsigned and quantizer.narrow_range))
        and (
            quantizer.block_sizes is None
            or (
                quantizer.num_bits == 8
                and quantizer.block_sizes in ({-1: size} for size in block_sizes)
            )
        )
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
            f"INT8 block_sizes in {block_sizes} (or no blocks), "
            "pass_through_bwd=True, no rotation, pre-scaling, bias, constant "
            "amax, or custom backend. Other gradient rules and formats are not implemented."
        )


def state_quantizer_config(
    quantizer: TensorQuantizer, *, name="state_quantizer"
) -> tuple[str, int]:
    """Validate the state quantizer and derive its format and last-axis group size.

    A zero group size retains the legacy policy's full-key state tiles.
    """
    validate_gdn_quantizer(
        quantizer, name=name, num_bits=tuple(_STATE_FORMATS), block_sizes=(16, 32, 64)
    )
    if quantizer.block_sizes is not None:
        return _STATE_FORMATS[quantizer.num_bits], quantizer.block_sizes[-1]
    if quantizer.axis != (0, 1):
        raise ValueError(f"{name} supports only axis=(0, 1) with state.block_v tiling")
    return _STATE_FORMATS[quantizer.num_bits], 0


def _make_state_quantizer(state_format):
    """Build a standalone-call quantizer; converted modules supply their registered instance."""
    # QuantizeConfig imports this module before the quantizer classes are initialized.
    from ..config import QuantizerAttributeConfig
    from ..nn import TensorQuantizer

    if state_format not in _STATE_FORMATS.values():
        raise ValueError("State format must be fp8_e4m3 or int8")
    return TensorQuantizer(
        QuantizerAttributeConfig(
            num_bits=(4, 3) if state_format == "fp8_e4m3" else 8,
            type="dynamic",
            axis=(0, 1),
            narrow_range=True,
            pass_through_bwd=True,
        )
    )


def _resolve_state_quantizer(state_quantizer, state_qdq, state_format):
    """Resolve legacy flags and registered quantizer settings at either training entry point."""
    if state_quantizer is None and state_qdq:
        state_quantizer = _make_state_quantizer(state_format)
    if state_quantizer is not None:
        state_qdq = state_quantizer.is_enabled and state_quantizer._if_quant
        if state_quantizer.is_enabled:
            state_format, _ = state_quantizer_config(state_quantizer)
    return state_quantizer, state_qdq, state_format


def _state_qdq(
    state: torch.Tensor,
    block_v: int = 64,
    state_format: str = "fp8_e4m3",
    state_quantizer: TensorQuantizer | None = None,
):
    """Dynamic state-tile QDQ with detached scales and identity STE."""
    if state_quantizer is not None and state_quantizer.block_sizes is not None:
        return state_quantizer(state)
    if state_format not in ("fp8_e4m3", "int8"):
        raise ValueError("State format must be fp8_e4m3 or int8")
    if block_v not in (16, 32, 64, 128):
        raise ValueError("block_v must be 16, 32, 64, or 128")
    quantized, _ = _tile_qdq(state, block_v, state_format, state_quantizer=state_quantizer)
    return quantized


def _tile_qdq(value, block_v, state_format, *, state_quantizer=None):
    """Return TensorQuantizer tile QDQ with identity STE and detached scales."""
    quantizer = (
        state_quantizer if state_quantizer is not None else _make_state_quantizer(state_format)
    )
    rounded, scales = [], []
    for part in value.split(block_v, dim=-1):
        # Canonical [N, H, K*BV] shape preserves per-head tile scales for both state ranks.
        tensor = part.flatten(-2)
        inputs = tensor.reshape(1, -1, tensor.shape[-1])
        decoded = quantizer(inputs)
        amax = quantizer._get_amax(inputs).float()
        if quantizer.num_bits == 8:
            scale = amax / quantizer.maxbound
        else:
            safe_amax = torch.where(amax <= 2**-24, torch.ones_like(amax), amax)
            scale = torch.div(quantizer.maxbound, safe_amax).reciprocal()
        rounded.append(decoded.reshape_as(part))
        scales.append(scale.reshape(tensor.shape[:-1]))
    return torch.cat(rounded, dim=-1).to(value.dtype), torch.stack(scales, dim=-1)
