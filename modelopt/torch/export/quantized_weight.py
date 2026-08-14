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

"""Non-mutating export helpers for quantized checkpoint weights."""

from collections import OrderedDict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn as nn

from modelopt import __version__
from modelopt.torch.quantization.qtensor import NVFP4QTensor
from modelopt.torch.quantization.utils import quantizer_attr_names

from .convert_hf_config import convert_hf_quant_config_format
from .model_config import QUANTIZATION_NONE, QUANTIZATION_NVFP4, QUANTIZATION_W4A16_NVFP4
from .quant_utils import process_layer_quant_config, to_quantized_weight

__all__ = [
    "build_hf_quantization_config",
    "capture_quantized_weight_export_state",
    "export_quantized_weight",
    "quantized_weight_export_states_compatible",
    "quantized_weight_export_states_equal",
    "replicate_quantized_weight_export_state",
    "synchronize_quantized_weight_export_state",
]

_SUPPORTED_FORMATS = {QUANTIZATION_NVFP4, QUANTIZATION_W4A16_NVFP4}


@dataclass(frozen=True)
class _QuantizedWeightExportState:
    """Opaque export state for one logical floating-point weight."""

    quantization_format: str
    block_size: int
    weight_amax: torch.Tensor
    input_amax: torch.Tensor | None = None


@dataclass(frozen=True)
class _QuantizedWeightExport:
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


def _clone_amax(quantizer: object | None) -> torch.Tensor | None:
    if quantizer is None or not bool(getattr(quantizer, "is_enabled", False)):
        return None
    export_amax = getattr(quantizer, "export_amax", None)
    value = export_amax() if callable(export_amax) else getattr(quantizer, "amax", None)
    if not isinstance(value, torch.Tensor):
        return None
    return value.detach().clone()


def _get_nvfp4_export_format(
    module: nn.Module,
    weight_quantizer: object,
    input_quantizer: object | None,
) -> str:
    if not bool(getattr(weight_quantizer, "is_enabled", False)):
        raise RuntimeError("Weight quantizer is disabled")
    if NVFP4QTensor._is_static_quantizer(weight_quantizer):
        raise NotImplementedError("Static NVFP4 export state is not supported")
    if hasattr(weight_quantizer, "svdquant_lora_a") or getattr(
        module, "fused_with_prequant", False
    ):
        raise NotImplementedError("AWQ and SVDQuant export state is not supported")

    block_sizes = getattr(weight_quantizer, "block_sizes", None)
    if (
        getattr(weight_quantizer, "num_bits", None) != (2, 1)
        or not isinstance(block_sizes, Mapping)
        or block_sizes.get("type") != "dynamic"
        or block_sizes.get("scale_bits") != (4, 3)
        or block_sizes.get("four_over_six", False)
    ):
        raise NotImplementedError("Only dynamic block-16 NVFP4 weights are supported")

    if input_quantizer is None or not bool(getattr(input_quantizer, "is_enabled", False)):
        return QUANTIZATION_W4A16_NVFP4
    input_blocks = getattr(input_quantizer, "block_sizes", None)
    if (
        hasattr(input_quantizer, "_pre_quant_scale")
        or getattr(input_quantizer, "num_bits", None) != (2, 1)
        or not isinstance(input_blocks, Mapping)
        or input_blocks.get("type") != "dynamic"
        or input_blocks.get("scale_bits") != (4, 3)
        or input_blocks.get("four_over_six", False)
    ):
        raise NotImplementedError("Only dynamic NVFP4 or disabled input quantizers are supported")
    return QUANTIZATION_NVFP4


def capture_quantized_weight_export_state(
    module: nn.Module,
    weight_name: str = "weight",
    *,
    weight_quantizer: object | None = None,
    input_quantizer: object | None = None,
) -> _QuantizedWeightExportState:
    """Capture export state without changing the module or its quantizers."""
    attrs = quantizer_attr_names(weight_name)
    if weight_quantizer is None:
        weight_quantizer = getattr(module, attrs.weight_quantizer, None)
    if weight_quantizer is None:
        raise RuntimeError(f"Missing weight quantizer for {weight_name!r}")
    if input_quantizer is None:
        input_quantizer = getattr(module, attrs.input_quantizer, None)

    quantization_format = _get_nvfp4_export_format(module, weight_quantizer, input_quantizer)
    weight_amax = _clone_amax(weight_quantizer)
    if weight_amax is None or weight_amax.numel() != 1:
        raise RuntimeError(f"Missing scalar calibrated weight amax for {weight_name!r}")

    input_amax = _clone_amax(input_quantizer)
    if quantization_format == QUANTIZATION_NVFP4 and (
        input_amax is None or input_amax.numel() != 1
    ):
        raise RuntimeError(f"Missing scalar calibrated input amax for {weight_name!r}")

    block_sizes = getattr(weight_quantizer, "block_sizes", None)
    if not isinstance(block_sizes, Mapping):
        raise RuntimeError(f"Missing NVFP4 block sizes for {weight_name!r}")
    block_size = block_sizes[-1]
    if block_size != 16:
        raise NotImplementedError(f"Unsupported NVFP4 block size: {block_size}")
    return _QuantizedWeightExportState(
        quantization_format=quantization_format,
        block_size=block_size,
        weight_amax=weight_amax,
        input_amax=input_amax,
    )


def _require_state(state: object) -> _QuantizedWeightExportState:
    if not isinstance(state, _QuantizedWeightExportState):
        raise TypeError("Expected ModelOpt quantized-weight export state")
    return state


def _clone_state(
    state: _QuantizedWeightExportState,
    *,
    device: torch.device | str | None = None,
) -> _QuantizedWeightExportState:
    def clone(tensor: torch.Tensor) -> torch.Tensor:
        tensor = tensor.detach().clone()
        return tensor if device is None else tensor.to(device=device)

    return _QuantizedWeightExportState(
        quantization_format=state.quantization_format,
        block_size=state.block_size,
        weight_amax=clone(state.weight_amax),
        input_amax=(None if state.input_amax is None else clone(state.input_amax)),
    )


def synchronize_quantized_weight_export_state(
    state: object,
    *,
    group=None,
    device: torch.device | str | None = None,
) -> object:
    """Max-reduce scalar state for ranks that shard one logical weight."""
    state = _require_state(state)
    synchronized = _clone_state(state)
    if dist.is_available() and dist.is_initialized() and dist.get_world_size(group) > 1:
        dist.all_reduce(synchronized.weight_amax, op=dist.ReduceOp.MAX, group=group)
        if synchronized.input_amax is not None:
            dist.all_reduce(synchronized.input_amax, op=dist.ReduceOp.MAX, group=group)
    return _clone_state(synchronized, device=device)


def replicate_quantized_weight_export_state(
    state: object,
    projection_count: int,
) -> tuple[object, ...]:
    """Authorize scalar state replication across a fused projection split."""
    if projection_count < 1:
        raise ValueError("projection_count must be positive")
    state = _require_state(state)
    return tuple(_clone_state(state) for _ in range(projection_count))


def quantized_weight_export_states_equal(
    left: object,
    right: object,
) -> bool:
    """Return whether two opaque export states describe the same checkpoint state."""
    left = _require_state(left)
    right = _require_state(right)
    if (
        left.quantization_format != right.quantization_format
        or left.block_size != right.block_size
        or not torch.equal(left.weight_amax, right.weight_amax)
        or (left.input_amax is None) != (right.input_amax is None)
    ):
        return False
    return left.input_amax is None or torch.equal(left.input_amax, right.input_amax)


def quantized_weight_export_states_compatible(left: object, right: object) -> bool:
    """Return whether two states share one deployment format and tensor layout."""
    left = _require_state(left)
    right = _require_state(right)
    return (
        left.quantization_format == right.quantization_format
        and left.block_size == right.block_size
        and (left.input_amax is None) == (right.input_amax is None)
    )


def _require_positive_finite(name: str, value: torch.Tensor) -> None:
    if value.numel() == 0 or not torch.isfinite(value).all() or not torch.all(value > 0):
        raise RuntimeError(f"Invalid {name}: {value}")


def export_quantized_weight(
    weight: torch.Tensor,
    state: object,
    *,
    dtype: torch.dtype | None = None,
) -> _QuantizedWeightExport:
    """Pack one canonical weight without mutating it or its quantizers."""
    state = _require_state(state)
    if state.quantization_format not in _SUPPORTED_FORMATS:
        raise NotImplementedError(f"Unsupported export format: {state.quantization_format!r}")
    if state.block_size != 16:
        raise NotImplementedError(f"Unsupported NVFP4 block size: {state.block_size}")

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
        weight.to(dtype or weight.dtype),
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

    return _QuantizedWeightExport(
        weight=packed_weight,
        weight_scale=weight_scale,
        weight_scale_2=weight_scale_2.squeeze(),
        input_scale=input_scale,
    )


def build_hf_quantization_config(
    layer_states: Mapping[str, object | None] | Iterable[tuple[str, object | None]],
) -> dict:
    """Build the canonical ModelOpt HF config from canonical module states."""
    layer_config = {}
    for name, state in dict(layer_states).items():
        state = None if state is None else _require_state(state)
        layer_config[f"{name}.quantization"] = (
            state.quantization_format if state is not None else QUANTIZATION_NONE
        )
        layer_config[f"{name}.awq_block_size"] = state.block_size if state is not None else 0
    quantization = process_layer_quant_config(layer_config)
    quantization["kv_cache_quant_algo"] = QUANTIZATION_NONE
    return convert_hf_quant_config_format(
        {
            "producer": {"name": "modelopt", "version": __version__},
            "quantization": quantization,
        }
    )
