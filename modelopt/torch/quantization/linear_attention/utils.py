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


def _validate_gate_inputs(g, beta, gate_inputs):
    """Check raw projection shapes and types before dispatching native kernels."""
    if gate_inputs is None:
        return
    if not isinstance(gate_inputs, (tuple, list)) or len(gate_inputs) != 4:
        raise ValueError("gate_inputs must contain raw gate, raw beta, A_log, and bias")
    raw_g, raw_beta, rate, bias = gate_inputs
    for value in (raw_g, raw_beta, rate, bias):
        if value is not None and (
            not isinstance(value, torch.Tensor)
            or value.device != g.device
            or value.dtype not in (torch.bfloat16, torch.float32)
        ):
            raise ValueError("Raw gate inputs must be BF16/FP32 tensors on the gate device")
    if raw_g is None or raw_beta is None or raw_g.shape != g.shape or raw_beta.shape != beta.shape:
        raise ValueError("Raw gate and beta shapes must match their prepared operands")
    heads = beta.shape[-1]
    channels = g.shape[-1] if g.ndim == beta.ndim + 1 else 1
    if rate is None or rate.numel() != heads or rate.ndim not in (1, 2):
        raise ValueError("Raw gate A_log must contain one rate per value head")
    if (bias is None and g.ndim == beta.ndim) or (
        bias is not None and (bias.ndim != 1 or bias.numel() != heads * channels)
    ):
        raise ValueError("Raw gate bias must contain one value per gate channel")


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
    if quantizer.block_sizes is not None and quantizer.block_sizes.get("type") == "dynamic":
        raise ValueError(
            f"{name} does not support block_sizes['type']='dynamic'. For INT8 state QDQ, "
            "use top-level type='dynamic' with block_sizes={-1: N}; "
            "quantized block scales are not supported."
        )
    if not (
        quantizer._dynamic
        and quantizer.num_bits in num_bits
        and (quantizer.num_bits != 8 or (not quantizer.unsigned and quantizer.narrow_range))
        and (
            quantizer.block_sizes is None
            or (quantizer.block_sizes in ({-1: size} for size in block_sizes))
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
            f"block_sizes in {block_sizes} (or no blocks), "
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
        quantizer, name=name, num_bits=tuple(_STATE_FORMATS), block_sizes=(16, 32, 64, 128)
    )
    if quantizer.block_sizes is not None:
        return _STATE_FORMATS[quantizer.num_bits], quantizer.block_sizes[-1]
    if quantizer.axis != (0, 1):
        raise ValueError(f"{name} supports only axis=(0, 1) with state_block_v tiling")
    return _STATE_FORMATS[quantizer.num_bits], 0


def _resolve_state_quantizer(state_quantizer):
    """Validate an enabled quantizer once; return None when state QDQ is inactive."""
    if state_quantizer is not None and state_quantizer.is_enabled:
        state_quantizer_config(state_quantizer)
        if state_quantizer._if_quant:
            return state_quantizer
    return None


def _state_qdq(state: torch.Tensor, state_quantizer: TensorQuantizer | None, block_v: int = 64):
    """Apply state QDQ; TensorQuantizer owns the format, scales, and backward rule."""
    if state_quantizer is None:
        return state
    if state_quantizer.block_sizes is not None:
        return state_quantizer(state)
    rounded = []
    for part in state.split(block_v, dim=-1):
        # Canonical [N, H, K*BV] shape preserves per-head tile scales for both state ranks.
        tensor = part.flatten(-2)
        inputs = tensor.reshape(1, -1, tensor.shape[-1])
        rounded.append(state_quantizer(inputs).reshape_as(part))
    return torch.cat(rounded, dim=-1).to(state.dtype)
