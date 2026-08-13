# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Non-mutating export helpers for quantized checkpoint weights."""

from collections import OrderedDict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import torch
import torch.nn as nn

from modelopt.torch.quantization.qtensor import NVFP4QTensor
from modelopt.torch.quantization.utils import quantizer_attr_names

from .convert_hf_config import convert_hf_quant_config_format
from .model_config import QUANTIZATION_NONE, QUANTIZATION_NVFP4, QUANTIZATION_W4A16_NVFP4
from .quant_utils import (
    build_quant_config,
    get_quantization_format_from_quantizers,
    to_quantized_weight,
)

__all__ = [
    "QuantizedWeightExport",
    "QuantizedWeightExportState",
    "build_hf_quantization_config",
    "capture_quantized_weight_export_state",
    "export_quantized_weight",
]

_SUPPORTED_FORMATS = {QUANTIZATION_NVFP4, QUANTIZATION_W4A16_NVFP4}


@dataclass(frozen=True)
class QuantizedWeightExportState:
    """Scalar ModelOpt state for one logical floating-point weight."""

    quantization_format: str
    block_size: int
    weight_amax: torch.Tensor
    input_amax: torch.Tensor | None = None


@dataclass(frozen=True)
class QuantizedWeightExport:
    """Packed checkpoint tensors for one quantized weight."""

    weight: torch.Tensor
    weight_scale: torch.Tensor
    weight_scale_2: torch.Tensor
    input_scale: torch.Tensor | None = None

    def named_tensors(self, weight_name: str = "weight") -> OrderedDict[str, torch.Tensor]:
        """Return checkpoint tensors using ModelOpt's canonical relative names."""
        attrs = quantizer_attr_names(weight_name)
        tensors = OrderedDict(
            (
                (weight_name, self.weight),
                (attrs.weight_scale, self.weight_scale),
                (attrs.weight_scale_2, self.weight_scale_2),
            )
        )
        if self.input_scale is not None:
            tensors[attrs.input_scale] = self.input_scale
        return tensors


@dataclass(frozen=True)
class _Nvfp4WeightQuantizerView:
    _amax: torch.Tensor
    block_sizes: Mapping[object, object]


@dataclass(frozen=True)
class _Nvfp4InputQuantizerView:
    _amax: torch.Tensor
    is_enabled: bool = True
    maxbound: float = 6.0

    def export_amax(self) -> torch.Tensor:
        return self._amax


def _clone_amax(quantizer: object, *, input_quantizer: bool = False) -> torch.Tensor | None:
    if quantizer is None or not bool(getattr(quantizer, "is_enabled", False)):
        return None
    if input_quantizer:
        value = getattr(quantizer, "_amax", None)
    else:
        value = getattr(quantizer, "global_amax", None)
        if value is None:
            value = getattr(quantizer, "_global_amax", None)
        if value is None:
            value = getattr(quantizer, "_amax", None)
    if not isinstance(value, torch.Tensor):
        return None
    return value.detach().clone()


def capture_quantized_weight_export_state(
    module: nn.Module,
    weight_name: str = "weight",
    *,
    weight_quantizer: object | None = None,
    input_quantizer: object | None = None,
) -> QuantizedWeightExportState:
    """Capture scalar quantized-weight export state without changing the module."""
    attrs = quantizer_attr_names(weight_name)
    if weight_quantizer is None:
        weight_quantizer = getattr(module, attrs.weight_quantizer, None)
    if weight_quantizer is None:
        raise RuntimeError(f"Missing weight quantizer for {weight_name!r}")
    if NVFP4QTensor._is_static_quantizer(weight_quantizer):
        raise NotImplementedError(
            "Pure distributed export does not support static per-block NVFP4 state."
        )

    weight_amax = _clone_amax(weight_quantizer)
    if weight_amax is None:
        raise RuntimeError(f"Missing calibrated weight amax for {weight_name!r}")
    if weight_amax.numel() != 1:
        raise NotImplementedError(
            "Pure quantized-weight export currently requires scalar weight amax"
        )
    if input_quantizer is None:
        input_quantizer = getattr(module, attrs.input_quantizer, None)
    quantization_format = get_quantization_format_from_quantizers(
        module,
        weight_quantizer,
        input_quantizer,
        weight_name=weight_name,
    )
    if quantization_format not in _SUPPORTED_FORMATS:
        raise NotImplementedError(f"Unsupported pure export format: {quantization_format!r}")

    input_amax = _clone_amax(
        input_quantizer,
        input_quantizer=True,
    )
    block_sizes = getattr(weight_quantizer, "block_sizes", None) or {}
    if getattr(weight_quantizer, "num_bits", None) != (2, 1) or block_sizes.get("scale_bits") != (
        4,
        3,
    ):
        raise NotImplementedError("Pure quantized-weight export requires NVFP4 weights")
    if quantization_format == QUANTIZATION_NVFP4 and input_amax is None:
        raise RuntimeError(f"Missing calibrated input amax for {weight_name!r}")
    if input_amax is not None and input_amax.numel() != 1:
        raise NotImplementedError(
            "Pure quantized-weight export currently requires scalar input amax"
        )

    return QuantizedWeightExportState(
        quantization_format=quantization_format,
        block_size=block_sizes[-1],
        weight_amax=weight_amax,
        input_amax=input_amax,
    )


def _require_positive_finite(name: str, value: torch.Tensor) -> None:
    if value.numel() == 0 or not torch.isfinite(value).all() or not torch.all(value > 0):
        raise RuntimeError(f"Invalid {name}: {value}")


def export_quantized_weight(
    weight: torch.Tensor,
    state: QuantizedWeightExportState,
    *,
    dtype: torch.dtype,
) -> QuantizedWeightExport:
    """Pack one canonical weight without mutating it or its quantizers."""
    if state.quantization_format not in _SUPPORTED_FORMATS:
        raise NotImplementedError(f"Unsupported pure export format: {state.quantization_format!r}")
    if state.block_size <= 0:
        raise ValueError(f"Invalid block size: {state.block_size}")

    weight_amax = state.weight_amax.to(device=weight.device, dtype=torch.float32)
    _require_positive_finite("weight amax", weight_amax)
    quantizer = _Nvfp4WeightQuantizerView(
        _amax=weight_amax,
        block_sizes={-1: state.block_size, "type": "dynamic", "scale_bits": (4, 3)},
    )
    weight_scale_2 = NVFP4QTensor.get_weights_scaling_factor_2_from_quantizer(quantizer)
    weight_scale = NVFP4QTensor.get_weights_scaling_factor(
        weight,
        block_size=state.block_size,
        weights_scaling_factor_2=weight_scale_2.to(weight.device),
    )[0]
    packed_weight = to_quantized_weight(
        weight.to(dtype),
        weight_scale,
        state.quantization_format,
        weight_scale_2,
        state.block_size,
    )

    input_scale = None
    if state.quantization_format == QUANTIZATION_NVFP4:
        if state.input_amax is None:
            raise RuntimeError("Missing input amax for NVFP4 W4A4 export")
        input_amax = state.input_amax.to(device=weight.device, dtype=torch.float32)
        _require_positive_finite("input amax", input_amax)
        input_scale = NVFP4QTensor.get_activation_scaling_factor(
            _Nvfp4InputQuantizerView(input_amax)
        ).squeeze()

    return QuantizedWeightExport(
        weight=packed_weight,
        weight_scale=weight_scale,
        weight_scale_2=weight_scale_2.squeeze(),
        input_scale=input_scale,
    )


def build_hf_quantization_config(
    layer_states: Mapping[str, QuantizedWeightExportState | None]
    | Iterable[tuple[str, QuantizedWeightExportState | None]],
) -> dict:
    """Build the canonical ModelOpt HF config from canonical layer formats."""
    states = dict(layer_states)
    quantized_formats = {
        state.quantization_format for state in states.values() if state is not None
    }
    unsupported = quantized_formats.difference(_SUPPORTED_FORMATS)
    if unsupported:
        raise NotImplementedError(f"Unsupported quantized layer formats: {sorted(unsupported)}")
    layer_config = {}
    for name, state in states.items():
        layer_config[f"{name}.quantization"] = (
            state.quantization_format if state is not None else QUANTIZATION_NONE
        )
        layer_config[f"{name}.awq_block_size"] = state.block_size if state is not None else 0
    return convert_hf_quant_config_format(build_quant_config(layer_config))
