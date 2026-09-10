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

"""Functional export of quantized weight tensors."""

import re
from collections import OrderedDict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn

from modelopt import __version__
from modelopt.torch.quantization.qtensor import (
    MXFP4QTensor,
    MXFP8QTensor,
    NVFP4QTensor,
    QTensorWrapper,
)
from modelopt.torch.quantization.utils import quantizer_attr_names, representative_weight_quantizer

from ..quantization.nn import SequentialQuantizer, TensorQuantizer
from .convert_hf_config import convert_hf_quant_config_format
from .model_config import (
    QUANTIZATION_FP8,
    QUANTIZATION_FP8_PB_WO,
    QUANTIZATION_FP8_PC_PT,
    QUANTIZATION_MXFP4,
    QUANTIZATION_MXFP8,
    QUANTIZATION_NONE,
    QUANTIZATION_NVFP4,
    QUANTIZATION_W4A8_MXFP4_FP8,
    QUANTIZATION_W4A8_NVFP4_FP8,
    QUANTIZATION_W4A16_NVFP4,
)
from .quant_utils import (
    _get_quantization_from_quantizers,
    get_scaling_factor,
    process_layer_quant_config,
    to_quantized_weight,
)

__all__ = [
    "build_hf_quantization_config",
    "capture_quantized_weight_export_state",
    "export_quantized_weight_tensors",
    "get_quantized_weight_export_spec",
    "merge_quantized_weight_export_states",
    "restore_quantized_weight_export_state",
    "select_quantized_weight_export_state",
    "split_quantized_weight_export_state",
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


@dataclass(frozen=True)
class _ExportStateTensorMetadata:
    name: str
    axes: tuple[int, ...]
    block_sizes: tuple[int, ...]


@dataclass(frozen=True)
class _QuantizedWeightExportMetadata:
    quantization_format: str
    block_size: int
    weight_shape: tuple[int, ...]
    tensors: tuple[_ExportStateTensorMetadata, ...]
    packing_permutation: tuple[int, ...]
    static_nvfp4: bool
    four_over_six: bool


def _same_storage(left: object, right: object) -> bool:
    if left is right:
        return True
    if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
        return False
    if left.device.type == "meta" or right.device.type == "meta":
        return False
    return (
        left.device == right.device
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
    if quantizer is None:
        return None
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
    if quantization_format == QUANTIZATION_MXFP4 and (
        input_quantizer is None or not input_quantizer.is_enabled
    ):
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
    _, _, weight_quantizer, _, quantization_format = resolved
    block_config = getattr(weight_quantizer, "block_sizes", None) or {}
    block_size = int(block_config.get(-1, 0)) if isinstance(block_config, dict) else 0
    return _QuantizedWeightExportSpec(quantization_format, block_size)


def split_quantized_weight_export_state(
    state: _QuantizedWeightExportState,
) -> tuple[object, tuple[torch.Tensor, ...]]:
    """Separate opaque state metadata from tensor values for typed transport."""
    metadata = _QuantizedWeightExportMetadata(
        state.quantization_format,
        state.block_size,
        state.weight_shape,
        tuple(
            _ExportStateTensorMetadata(record.name, record.axes, record.block_sizes)
            for record in state.tensors
        ),
        state.packing_permutation,
        state.static_nvfp4,
        state.four_over_six,
    )
    return metadata, tuple(record.value for record in state.tensors)


def restore_quantized_weight_export_state(
    metadata: object,
    tensors: Sequence[torch.Tensor],
) -> _QuantizedWeightExportState:
    """Restore opaque export state after its tensor values have been transported."""
    if not isinstance(metadata, _QuantizedWeightExportMetadata):
        raise TypeError(f"Invalid quantized weight export metadata: {type(metadata).__name__}")
    if len(tensors) != len(metadata.tensors):
        raise ValueError(f"Expected {len(metadata.tensors)} export tensors, got {len(tensors)}")
    return _QuantizedWeightExportState(
        metadata.quantization_format,
        metadata.block_size,
        metadata.weight_shape,
        tuple(
            _ExportStateTensor(spec.name, tensor, spec.axes, spec.block_sizes)
            for spec, tensor in zip(metadata.tensors, tensors, strict=True)
        ),
        metadata.packing_permutation,
        metadata.static_nvfp4,
        metadata.four_over_six,
    )


def _normalize_weight_dim(weight_dim: int, ndim: int) -> int:
    if not -ndim <= weight_dim < ndim:
        raise IndexError(f"Weight dimension {weight_dim} is invalid for a rank-{ndim} weight")
    return weight_dim % ndim


def _merge_state_tensors(
    records: Sequence[_ExportStateTensor], packed_dim: int
) -> _ExportStateTensor:
    reference = records[0]
    if any(
        record.name != reference.name
        or record.axes != reference.axes
        or record.block_sizes != reference.block_sizes
        for record in records[1:]
    ):
        raise ValueError("Quantized weight shards have incompatible export state")
    if packed_dim in reference.axes:
        tensor_dim = reference.axes.index(packed_dim)
        value = torch.cat([record.value for record in records], dim=tensor_dim)
    else:
        if len({tuple(record.value.shape) for record in records}) != 1:
            raise ValueError("Replicated export tensors have incompatible shapes")
        value = torch.stack([record.value for record in records]).amax(dim=0)
    return _ExportStateTensor(reference.name, value, reference.axes, reference.block_sizes)


def _validate_merge_block_boundaries(
    states: Sequence[_QuantizedWeightExportState], packed_dim: int
) -> None:
    block_sizes = {
        block_size
        for record in states[0].tensors
        for axis, block_size in zip(record.axes, record.block_sizes, strict=True)
        if axis == packed_dim and block_size > 1
    }
    logical_dim = states[0].packing_permutation[packed_dim]
    boundary = 0
    for state in states[:-1]:
        boundary += state.weight_shape[logical_dim]
        if misaligned := sorted(block_size for block_size in block_sizes if boundary % block_size):
            raise ValueError(
                "Quantized weight shard boundary "
                f"{boundary} is not aligned to export-state block sizes {misaligned}"
            )


def merge_quantized_weight_export_states(
    states: Sequence[_QuantizedWeightExportState],
    weight_dim: int,
) -> _QuantizedWeightExportState:
    """Merge state for logical weight shards concatenated along ``weight_dim``."""
    if not states:
        raise ValueError("At least one quantized weight state is required")
    reference = states[0]
    ndim = len(reference.weight_shape)
    weight_dim = _normalize_weight_dim(weight_dim, ndim)
    if any(
        state.quantization_format != reference.quantization_format
        or state.block_size != reference.block_size
        or state.packing_permutation != reference.packing_permutation
        or state.static_nvfp4 != reference.static_nvfp4
        or state.four_over_six != reference.four_over_six
        or len(state.tensors) != len(reference.tensors)
        or len(state.weight_shape) != ndim
        or any(
            size != reference.weight_shape[axis]
            for axis, size in enumerate(state.weight_shape)
            if axis != weight_dim
        )
        for state in states[1:]
    ):
        raise ValueError("Quantized weight shards are incompatible")

    shape = list(reference.weight_shape)
    shape[weight_dim] = sum(state.weight_shape[weight_dim] for state in states)
    packed_dim = reference.packing_permutation.index(weight_dim)
    _validate_merge_block_boundaries(states, packed_dim)
    tensors = tuple(
        _merge_state_tensors([state.tensors[index] for state in states], packed_dim)
        for index in range(len(reference.tensors))
    )
    return _QuantizedWeightExportState(
        reference.quantization_format,
        reference.block_size,
        tuple(shape),
        tensors,
        reference.packing_permutation,
        reference.static_nvfp4,
        reference.four_over_six,
    )


def _select_state_tensor(
    record: _ExportStateTensor,
    packed_dim: int,
    indices: torch.Tensor,
) -> _ExportStateTensor:
    if packed_dim not in record.axes:
        return record
    tensor_dim = record.axes.index(packed_dim)
    block_size = record.block_sizes[tensor_dim]
    if block_size == 1:
        selected = indices
    else:
        selected = _selected_block_indices(indices, block_size)
    value = record.value.index_select(tensor_dim, selected.to(record.value.device))
    return _ExportStateTensor(record.name, value, record.axes, record.block_sizes)


def _selected_block_indices(indices: torch.Tensor, block_size: int) -> torch.Tensor:
    if indices.numel() % block_size:
        raise ValueError("Weight selection must preserve complete quantization blocks")
    blocks = indices.reshape(-1, block_size)
    block_indices = torch.div(blocks, block_size, rounding_mode="floor")
    selected = block_indices[:, 0]
    expected = selected[:, None] * block_size + torch.arange(block_size)
    if not torch.all(block_indices == selected[:, None]) or not torch.equal(
        blocks.sort(dim=1).values, expected
    ):
        raise ValueError("Weight selection must preserve complete quantization blocks")
    return selected


def select_quantized_weight_export_state(
    state: _QuantizedWeightExportState,
    weight_dim: int,
    indices: Iterable[int] | torch.Tensor,
) -> _QuantizedWeightExportState:
    """Select logical indices that preserve complete quantization blocks."""
    ndim = len(state.weight_shape)
    weight_dim = _normalize_weight_dim(weight_dim, ndim)
    indices = torch.as_tensor(list(indices) if not isinstance(indices, torch.Tensor) else indices)
    indices = indices.to(dtype=torch.long, device="cpu").reshape(-1)
    if indices.numel() == 0:
        raise ValueError("Weight selection cannot be empty")
    if indices.min() < 0 or indices.max() >= state.weight_shape[weight_dim]:
        raise IndexError("Weight selection is out of range")

    shape = list(state.weight_shape)
    shape[weight_dim] = indices.numel()
    packed_dim = state.packing_permutation.index(weight_dim)
    if state.block_size > 1 and packed_dim == ndim - 1:
        _selected_block_indices(indices, state.block_size)
    return _QuantizedWeightExportState(
        state.quantization_format,
        state.block_size,
        tuple(shape),
        tuple(_select_state_tensor(record, packed_dim, indices) for record in state.tensors),
        state.packing_permutation,
        state.static_nvfp4,
        state.four_over_six,
    )


def _restore_packing_permutation(tensor: torch.Tensor, permutation: tuple[int, ...]):
    if permutation == tuple(range(len(permutation))):
        return tensor
    return tensor.permute(tuple(permutation.index(dim) for dim in range(len(permutation))))


def _restore_state_tensor(
    record: _ExportStateTensor,
    permutation: tuple[int, ...],
) -> torch.Tensor:
    if not record.axes:
        return record.value
    logical_axes = tuple(permutation[axis] for axis in record.axes)
    order = tuple(sorted(range(len(logical_axes)), key=logical_axes.__getitem__))
    if order == tuple(range(len(order))):
        return record.value
    return record.value.permute(order)


def export_quantized_weight_tensors(
    weight: torch.Tensor,
    state: _QuantizedWeightExportState,
    dtype: torch.dtype,
    weight_name: str = "weight",
) -> OrderedDict[str, torch.Tensor]:
    """Pack a logical weight into canonical ModelOpt checkpoint tensors."""
    if tuple(weight.shape) != state.weight_shape:
        raise ValueError(
            f"Weight shape {tuple(weight.shape)} does not match export state {state.weight_shape}"
        )
    packed_weight = weight.permute(state.packing_permutation)
    records = {record.name: record for record in state.tensors}

    weight_scale_2 = None
    if state.static_nvfp4:
        block_amax = records["weight_block_amax"].value
        global_amax = records["weight_global_amax"].value
        quantizer = SimpleNamespace(
            block_sizes={
                -1: state.block_size,
                "scale_bits": (4, 3),
                "four_over_six": state.four_over_six,
            },
            _amax=block_amax,
            _global_amax=global_amax,
            global_amax=global_amax,
        )
        weight_scale_2 = NVFP4QTensor.get_weights_scaling_factor_2_from_quantizer(quantizer)
        weight_scale = NVFP4QTensor.get_weights_scaling_factor_from_quantizer(
            quantizer, packed_weight, weight_scale_2
        )[0]
    elif state.quantization_format in _NVFP4_EXPORT_FORMATS:
        weight_scale_2 = records["weight_scale_2"].value
        weight_scale = NVFP4QTensor.get_weights_scaling_factor(
            packed_weight,
            state.block_size,
            weights_scaling_factor_2=weight_scale_2.to(packed_weight.device),
        )[0]
    elif "weight_scale" in records:
        weight_scale = records["weight_scale"].value
    elif state.quantization_format == QUANTIZATION_MXFP8:
        weight_scale = MXFP8QTensor.get_weights_scaling_factor(packed_weight)
    elif state.quantization_format in _MXFP4_EXPORT_FORMATS:
        quantized_weight, weight_scale = MXFP4QTensor.quantize(
            packed_weight.to(dtype), block_size=state.block_size
        )
        quantized_weight = quantized_weight._quantized_data
    else:
        raise _UnsupportedQuantizedWeightExportFormatError(
            f"Functional export does not support {state.quantization_format!r}"
        )

    weight_scale = weight_scale.to(packed_weight.device)
    if weight_scale_2 is not None:
        weight_scale_2 = weight_scale_2.to(packed_weight.device)

    if state.quantization_format not in _MXFP4_EXPORT_FORMATS:
        quantized_weight = to_quantized_weight(
            packed_weight.to(dtype),
            weight_scale,
            state.quantization_format,
            weight_scale_2,
            state.block_size,
        )
    attrs = quantizer_attr_names(weight_name)
    output = OrderedDict(
        ((weight_name, _restore_packing_permutation(quantized_weight, state.packing_permutation)),)
    )

    if "weight_scale" in records:
        scale_state = records["weight_scale"]
        weight_scale_record = _ExportStateTensor(
            scale_state.name,
            weight_scale,
            scale_state.axes,
            scale_state.block_sizes,
        )
    elif (
        state.quantization_format
        in _NVFP4_EXPORT_FORMATS
        | {
            QUANTIZATION_MXFP8,
        }
        | _MXFP4_EXPORT_FORMATS
    ):
        weight_scale_record = _block_scale_state(
            "weight_scale",
            weight_scale,
            tuple(packed_weight.shape),
            {-1: state.block_size},
            cpu=False,
        )
    else:
        weight_scale_record = _state_tensor("weight_scale", weight_scale, cpu=False)
    output[attrs.weight_scale] = _restore_state_tensor(
        weight_scale_record, state.packing_permutation
    )
    if weight_scale_2 is not None:
        scale_2_record = records.get("weight_scale_2") or records["weight_global_amax"]
        output[attrs.weight_scale_2] = _restore_state_tensor(
            _ExportStateTensor(
                scale_2_record.name,
                weight_scale_2.squeeze(),
                scale_2_record.axes,
                scale_2_record.block_sizes,
            ),
            state.packing_permutation,
        )
    if "input_scale" in records:
        output[attrs.input_scale] = (
            _restore_state_tensor(records["input_scale"], state.packing_permutation)
            .squeeze()
            .to(weight.device)
        )
    return output


def _quantized_layer_name(weight_name: str) -> str:
    name = weight_name.removesuffix(".weight")
    return re.sub(r"(\.experts)\.\d+(?=\.)", r"\1", name)


def build_hf_quantization_config(
    named_states: Mapping[str, _QuantizedWeightExportState | _QuantizedWeightExportSpec | None]
    | Iterable[tuple[str, _QuantizedWeightExportState | _QuantizedWeightExportSpec | None]],
) -> dict[str, Any]:
    """Build canonical ModelOpt HF configuration from named states or specs."""
    layers: dict[str, tuple[str, int] | None] = {}
    for weight_name, state in dict(named_states).items():
        layer_name = _quantized_layer_name(weight_name)
        value = None if state is None else (state.quantization_format, state.block_size)
        previous = layers.setdefault(layer_name, value)
        if previous != value:
            raise ValueError(f"Inconsistent quantization state for {layer_name}")

    layer_config = {}
    for layer_name, value in layers.items():
        layer_config[f"{layer_name}.quantization"] = (
            QUANTIZATION_NONE if value is None else value[0]
        )
        layer_config[f"{layer_name}.awq_block_size"] = 0 if value is None else value[1]
    config = {
        "producer": {"name": "modelopt", "version": __version__},
        "quantization": process_layer_quant_config(layer_config),
    }
    config["quantization"].setdefault("kv_cache_quant_algo", QUANTIZATION_NONE)
    return convert_hf_quant_config_format(config)
