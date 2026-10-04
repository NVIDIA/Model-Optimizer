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


def _fp8_quantize(value: torch.Tensor, axis):
    """Apply ModelOpt FP8 QDQ and return its detached dequantization scales."""
    # QuantizeConfig imports this module before the quantizer classes are initialized.
    from ..config import QuantizerAttributeConfig
    from ..nn import TensorQuantizer

    quantizer = TensorQuantizer(
        QuantizerAttributeConfig(num_bits=(4, 3), type="dynamic", axis=axis, pass_through_bwd=True)
    )
    quantized = quantizer(value)
    amax = quantizer._get_amax(value).float()
    safe_amax = torch.where(amax <= 2**-24, torch.ones_like(amax), amax)
    return quantized, torch.div(448.0, safe_amax).reciprocal()


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
    quantized, _ = _tile_qdq(state, block_v, state_format, state=True)
    if state_format == "fp8_e4m3":
        return quantized
    return state + (quantized - state).detach()


def _tile_qdq(value, block_v, state_format, *, state=False):
    """Return tile-rounded values and detached scales; INT8 callers supply their STE."""
    rounded, scales = [], []
    for part in value.split(block_v, dim=-1):
        tensor = part.flatten(-2) if state else part
        if state_format == "fp8_e4m3":
            axis = tuple(range(tensor.ndim - 1)) or None
            decoded, scale = _fp8_quantize(tensor, axis)
        else:
            with torch.no_grad():
                tensor = tensor.float()
                amax = tensor.abs().amax(dim=-1, keepdim=True)
                scale = torch.where(amax > 0, amax / 127.0, torch.ones_like(amax))
                decoded = (tensor / scale).round().clamp(-127, 127) * scale
        rounded.append(decoded.reshape_as(part))
        scales.append(scale.squeeze(-1))
    return torch.cat(rounded, dim=-1).to(value.dtype), torch.stack(scales, dim=-1)
