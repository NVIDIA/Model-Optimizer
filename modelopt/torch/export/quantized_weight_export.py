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

"""Capture state for functional quantized-weight export."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from modelopt.torch.quantization.qtensor import NVFP4QTensor, QTensorWrapper
from modelopt.torch.quantization.utils import quantizer_attr_names, representative_weight_quantizer

from ..quantization.nn import SequentialQuantizer, TensorQuantizer
from .quant_format import (
    QUANTIZATION_FP8,
    QUANTIZATION_FP8_PB_WO,
    QUANTIZATION_FP8_PC_PT,
    QUANTIZATION_MXFP4,
    QUANTIZATION_MXFP8,
    QUANTIZATION_NVFP4,
    QUANTIZATION_W4A8_MXFP4_FP8,
    QUANTIZATION_W4A8_NVFP4_FP8,
    QUANTIZATION_W4A16_MXFP4,
    QUANTIZATION_W4A16_NVFP4,
)
from .quant_utils import _get_quantization_from_quantizers, get_scaling_factor

__all__ = [
    "capture_quantized_weight_export_state",
    "get_quantized_weight_export_spec",
]

_NVFP4_EXPORT_FORMATS = {
    QUANTIZATION_NVFP4,
    QUANTIZATION_W4A8_NVFP4_FP8,
    QUANTIZATION_W4A16_NVFP4,
}
_MXFP4_EXPORT_FORMATS = {QUANTIZATION_MXFP4, QUANTIZATION_W4A8_MXFP4_FP8}
_FUNCTIONAL_WEIGHT_EXPORT_FORMATS = (
    _NVFP4_EXPORT_FORMATS
    | _MXFP4_EXPORT_FORMATS
    | {
        QUANTIZATION_FP8,
        QUANTIZATION_FP8_PC_PT,
        QUANTIZATION_FP8_PB_WO,
        QUANTIZATION_MXFP8,
    }
)


class _UnsupportedQuantizedWeightExportFormatError(NotImplementedError):
    pass


@dataclass(frozen=True)
class _ExportStateTensor:
    name: str
    value: torch.Tensor
    axes: tuple[int, ...] = ()
    block_sizes: tuple[int, ...] = ()


@dataclass(frozen=True)
class _QuantizedWeightExportState:
    """Opaque state required to export one logical quantized weight."""

    quantization_format: str
    block_size: int
    weight_shape: tuple[int, ...]
    tensors: tuple[_ExportStateTensor, ...]
    packing_permutation: tuple[int, ...]
    static_nvfp4: bool = False
    four_over_six: bool = False


@dataclass(frozen=True)
class _QuantizedWeightExportSpec:
    """Stable deployment metadata for one logical quantized weight."""

    quantization_format: str
    block_size: int
    input_quantizer_enabled: bool


def _same_storage(left: object, right: object) -> bool:
    if left is right:
        return True
    if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
        return False
    if left.device.type == "meta" or right.device.type == "meta":
        return False
    return (
        left.device == right.device
        and left.numel() == right.numel()
        and left.untyped_storage().data_ptr() == right.untyped_storage().data_ptr()
        and left.storage_offset() == right.storage_offset()
    )


def _resolve_weight_quantizer(
    module: nn.Module, weight_name: str
) -> tuple[torch.Tensor, TensorQuantizer | SequentialQuantizer] | None:
    weight = getattr(module, weight_name)
    iter_weights = getattr(module, "iter_weights_for_calibration", None)
    if iter_weights is not None:
        for candidate, quantizer in iter_weights():
            if _same_storage(candidate, weight):
                return candidate, quantizer

    quantizer = representative_weight_quantizer(module, weight_name)
    if quantizer is None and weight_name.startswith("weight"):
        quantizer = representative_weight_quantizer(module)
        if quantizer is not None and weight_name.removeprefix("weight").isdigit():
            raise NotImplementedError(
                f"Unable to resolve quantizer for numbered weight {weight_name!r}"
            )
    if quantizer is None:
        return None
    # A representative quantizer cannot supply another expert's calibrated state.
    if iter_weights is not None:
        raise NotImplementedError(
            f"Calibration iterator does not expose a matching full weight view for {weight_name!r}"
        )
    return weight, quantizer


def _packing_permutation(weight: torch.Tensor, quantized_view: torch.Tensor) -> tuple[int, ...]:
    ndim = weight.ndim
    identity = tuple(range(ndim))
    if ndim >= 2:
        transposed = weight.transpose(-1, -2)
        if (
            tuple(quantized_view.shape) == tuple(transposed.shape)
            and quantized_view.stride() == transposed.stride()
        ):
            return (*range(ndim - 2), ndim - 1, ndim - 2)
    if (
        tuple(quantized_view.shape) == tuple(weight.shape)
        and quantized_view.stride() == weight.stride()
    ):
        return identity
    raise NotImplementedError(
        "Unsupported quantized weight view "
        f"shape/stride={tuple(quantized_view.shape)}/{quantized_view.stride()} for "
        f"shape/stride={tuple(weight.shape)}/{weight.stride()}"
    )


def _state_tensor(
    name: str,
    value: torch.Tensor,
    *,
    axes: tuple[int, ...] = (),
    block_sizes: tuple[int, ...] = (),
    cpu: bool = True,
) -> _ExportStateTensor:
    value = value.detach().cpu().clone() if cpu else value.detach().clone()
    if value.numel() == 1 and not axes and not block_sizes:
        return _ExportStateTensor(name, value.reshape(()))
    if len(axes) != value.ndim or len(block_sizes) != value.ndim:
        raise ValueError(
            f"Explicit axis and block metadata is required for non-scalar {name}: "
            f"shape={tuple(value.shape)}, axes={axes}, block_sizes={block_sizes}"
        )
    return _ExportStateTensor(name, value, axes, block_sizes)


def _axis_scale_state(
    name: str,
    value: torch.Tensor,
    packed_shape: tuple[int, ...],
    axis: int | Sequence[int],
    *,
    cpu: bool = True,
) -> _ExportStateTensor:
    axes = (axis,) if isinstance(axis, int) else tuple(axis)
    axes = tuple(_normalize_weight_dim(dim, len(packed_shape)) for dim in axes)
    expected_shape = tuple(packed_shape[dim] for dim in axes)
    if value.numel() != torch.Size(expected_shape).numel():
        raise RuntimeError(
            f"Scale {name!r} shape {tuple(value.shape)} does not match axes {axes} "
            f"of packed weight shape {packed_shape}"
        )
    return _state_tensor(
        name,
        value.reshape(expected_shape),
        axes=axes,
        block_sizes=(1,) * len(axes),
        cpu=cpu,
    )


def _block_scale_state(
    name: str,
    value: torch.Tensor,
    packed_shape: tuple[int, ...],
    block_config: Mapping[int | str, Any],
    *,
    cpu: bool = True,
) -> _ExportStateTensor:
    ndim = len(packed_shape)
    block_sizes = [1] * ndim
    for dim, block_size in block_config.items():
        if isinstance(dim, int) and block_size is not None:
            block_sizes[_normalize_weight_dim(dim, ndim)] = int(block_size)
    expected_shape = tuple(
        (size + block_size - 1) // block_size for size, block_size in zip(packed_shape, block_sizes)
    )
    if value.numel() != torch.Size(expected_shape).numel():
        raise RuntimeError(
            f"Scale {name!r} shape {tuple(value.shape)} does not match block sizes "
            f"{tuple(block_sizes)} of packed weight shape {packed_shape}"
        )

    expanded_shape = []
    expanded_axes = []
    expanded_block_sizes = []
    for axis, (size, block_size) in enumerate(zip(expected_shape, block_sizes)):
        expanded_shape.append(size)
        expanded_axes.append(axis)
        expanded_block_sizes.append(block_size)
        if block_size != 1:
            expanded_shape.append(1)
            expanded_axes.append(axis)
            expanded_block_sizes.append(1)
    if tuple(value.shape) == tuple(expanded_shape):
        return _state_tensor(
            name,
            value,
            axes=tuple(expanded_axes),
            block_sizes=tuple(expanded_block_sizes),
            cpu=cpu,
        )
    return _state_tensor(
        name,
        value.reshape(expected_shape),
        axes=tuple(range(ndim)),
        block_sizes=tuple(block_sizes),
        cpu=cpu,
    )


def _input_quantizer(module: nn.Module, weight_name: str):
    quantizer = getattr(module, quantizer_attr_names(weight_name).input_quantizer, None)
    if quantizer is None:
        quantizer = getattr(module, "input_quantizer", None)
    return quantizer


def _resolve_quantized_weight_export_inputs(
    module: nn.Module,
    weight_name: str,
) -> (
    tuple[
        torch.Tensor,
        torch.Tensor,
        TensorQuantizer,
        TensorQuantizer | None,
        str,
    ]
    | None
):
    weight = getattr(module, weight_name)
    if isinstance(weight, QTensorWrapper):
        raise NotImplementedError("Functional export requires an uncompressed source weight")

    resolved = _resolve_weight_quantizer(module, weight_name)
    if resolved is None:
        return None
    quantized_view, weight_quantizer = resolved
    input_quantizer = _input_quantizer(module, weight_name)
    if not weight_quantizer.is_enabled:
        return None
    quantization_format = _get_quantization_from_quantizers(
        module, weight_quantizer, input_quantizer
    )
    if quantization_format is None:
        raise RuntimeError(f"Unable to resolve quantization format for {weight_name!r}")
    if isinstance(weight_quantizer, SequentialQuantizer):
        raise _UnsupportedQuantizedWeightExportFormatError(
            f"Functional export does not support {quantization_format!r}"
        )
    if quantization_format == QUANTIZATION_W4A16_MXFP4:
        raise _UnsupportedQuantizedWeightExportFormatError(
            "Functional export does not support weight-only MXFP4"
        )
    if quantization_format == QUANTIZATION_FP8_PB_WO and weight_quantizer.block_sizes != {
        -2: 128,
        -1: 128,
    }:
        raise _UnsupportedQuantizedWeightExportFormatError(
            "Functional export supports FP8 2D blockwise weight quantization only with "
            "128x128 blocks"
        )
    if quantization_format not in _FUNCTIONAL_WEIGHT_EXPORT_FORMATS:
        raise _UnsupportedQuantizedWeightExportFormatError(
            f"Functional export does not support {quantization_format!r}"
        )
    return weight, quantized_view, weight_quantizer, input_quantizer, quantization_format


def capture_quantized_weight_export_state(
    module: nn.Module,
    weight_name: str = "weight",
    *,
    cpu: bool = True,
) -> _QuantizedWeightExportState | None:
    """Capture detached state for one weight, or ``None`` when it is not quantized."""
    resolved = _resolve_quantized_weight_export_inputs(module, weight_name)
    if resolved is None:
        return None
    weight, quantized_view, weight_quantizer, input_quantizer, quantization_format = resolved

    permutation = _packing_permutation(weight, quantized_view)
    packed_shape = tuple(weight.shape[axis] for axis in permutation)
    block_config = getattr(weight_quantizer, "block_sizes", None) or {}
    block_size = int(block_config.get(-1, 0)) if isinstance(block_config, dict) else 0
    tensors = []
    static_nvfp4 = (
        quantization_format in _NVFP4_EXPORT_FORMATS
        and NVFP4QTensor._is_static_quantizer(weight_quantizer)
    )

    if static_nvfp4:
        if not block_size or packed_shape[-1] % block_size:
            raise RuntimeError(f"Invalid static NVFP4 block size for {weight_name!r}")
        per_block_amax = weight_quantizer.amax
        global_amax = NVFP4QTensor._get_static_global_amax(weight_quantizer)
        if per_block_amax is None or global_amax is None:
            raise RuntimeError(f"Missing calibrated static NVFP4 state for {weight_name!r}")
        block_shape = (*packed_shape[:-1], packed_shape[-1] // block_size)
        tensors.append(
            _state_tensor(
                "weight_block_amax",
                per_block_amax.reshape(block_shape),
                axes=tuple(range(len(packed_shape))),
                block_sizes=(1,) * (len(packed_shape) - 1) + (block_size,),
                cpu=cpu,
            )
        )
        tensors.append(_state_tensor("weight_global_amax", global_amax, cpu=cpu))
    elif quantization_format in _NVFP4_EXPORT_FORMATS:
        weight_scale_2 = (
            weight_quantizer._amax.float() / 448.0
            if quantization_format == QUANTIZATION_W4A8_NVFP4_FP8
            else NVFP4QTensor.get_weights_scaling_factor_2_from_quantizer(weight_quantizer)
        )
        tensors.append(_state_tensor("weight_scale_2", weight_scale_2, cpu=cpu))
    elif quantization_format == QUANTIZATION_FP8:
        weight_scale = get_scaling_factor(weight_quantizer)
        if weight_scale is None:
            raise RuntimeError(f"Missing calibrated weight scale for {weight_name!r}")
        tensors.append(_state_tensor("weight_scale", weight_scale, cpu=cpu))
    elif quantization_format == QUANTIZATION_FP8_PC_PT:
        weight_scale = get_scaling_factor(weight_quantizer)
        if weight_scale is None or weight_quantizer.axis is None:
            raise RuntimeError(f"Missing calibrated per-channel scale for {weight_name!r}")
        tensors.append(
            _axis_scale_state(
                "weight_scale",
                weight_scale,
                packed_shape,
                weight_quantizer.axis,
                cpu=cpu,
            )
        )
    elif quantization_format == QUANTIZATION_FP8_PB_WO:
        weight_scale = get_scaling_factor(weight_quantizer)
        if weight_scale is None:
            raise RuntimeError(f"Missing calibrated block scale for {weight_name!r}")
        tensors.append(
            _block_scale_state("weight_scale", weight_scale, packed_shape, block_config, cpu=cpu)
        )
    elif quantization_format == QUANTIZATION_MXFP8:
        cached_scale = getattr(weight_quantizer, "_scale", None)
        if cached_scale is not None:
            tensors.append(
                _block_scale_state(
                    "weight_scale", cached_scale, packed_shape, block_config, cpu=cpu
                )
            )
    elif quantization_format not in _MXFP4_EXPORT_FORMATS:
        raise AssertionError(f"Unexpected functional export format {quantization_format!r}")

    if input_quantizer is not None and input_quantizer.is_enabled:
        input_amax = input_quantizer.export_amax()
        if input_amax is not None:
            input_scale = (
                NVFP4QTensor.get_activation_scaling_factor(input_quantizer)
                if quantization_format == QUANTIZATION_NVFP4
                else get_scaling_factor(input_quantizer)
            )
            tensors.append(_state_tensor("input_scale", input_scale, cpu=cpu))

    return _QuantizedWeightExportState(
        quantization_format=quantization_format,
        block_size=block_size,
        weight_shape=tuple(weight.shape),
        tensors=tuple(tensors),
        packing_permutation=permutation,
        static_nvfp4=static_nvfp4,
        four_over_six=bool(block_config.get("four_over_six", False)),
    )


def get_quantized_weight_export_spec(
    module: nn.Module,
    weight_name: str = "weight",
) -> _QuantizedWeightExportSpec | None:
    """Return stable deployment metadata without retaining quantizer tensors."""
    resolved = _resolve_quantized_weight_export_inputs(module, weight_name)
    if resolved is None:
        return None
    _, _, weight_quantizer, input_quantizer, quantization_format = resolved
    block_config = getattr(weight_quantizer, "block_sizes", None) or {}
    block_size = int(block_config.get(-1, 0)) if isinstance(block_config, dict) else 0
    return _QuantizedWeightExportSpec(
        quantization_format,
        block_size,
        input_quantizer is not None and input_quantizer.is_enabled,
    )


def _normalize_weight_dim(weight_dim: int, ndim: int) -> int:
    if not -ndim <= weight_dim < ndim:
        raise IndexError(f"Weight dimension {weight_dim} is invalid for a rank-{ndim} weight")
    return weight_dim % ndim
