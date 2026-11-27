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

"""Partition native repair fields and immutable codes in logical weight coordinates."""

import copy
import math
from dataclasses import dataclass

import torch

from .ggml.common import PINNED_PAYLOAD_PREFIX
from .ggml.registry import GGML_FORMAT_REGISTRY
from .nn import TensorQuantizer
from .qtensor import FP8QTensor, NVFP4QTensor, QTensorWrapper
from .qtensor.base_qtensor import get_native_tensor_class
from .qtensor.nvfp4_tensor import cutlass_fp4_scale_to_modelopt_fp4_scale
from .utils.native_weight import (
    _canonical,
    _format_key,
    get_pinned_native_weight,
    pin_native_weight,
)

__all__ = []

_NATIVE_TP_CODECS = (FP8QTensor, NVFP4QTensor)


@dataclass(frozen=True)
class NativeWeightPartition:
    """Describe ordinary or packed gate/up ownership of a logical matrix."""

    shape: tuple[int, int]
    dim: int
    rank: int
    size: int
    split_factor: int = 1

    def __post_init__(self):
        if (
            len(self.shape) != 2
            or any(width <= 0 for width in self.shape)
            or self.dim not in (0, 1)
            or self.size <= 0
            or self.split_factor <= 0
            or not 0 <= self.rank < self.size
        ):
            raise ValueError(
                "Native TP fields require a positive matrix, axis 0 or 1, positive partition "
                "counts and a rank within the partition group."
            )
        if self.shape[self.dim] % (self.size * self.split_factor):
            raise ValueError("Native TP fields require evenly divided physical weight partitions.")

    @property
    def local_shape(self):
        return tuple(n // self.size if axis == self.dim else n for axis, n in enumerate(self.shape))

    def intervals(self, block=1):
        width = self.shape[self.dim] // (self.size * self.split_factor)
        if width % block:
            raise ValueError("TP partitions must align with complete native quantization blocks.")
        return [
            ((part * self.size + self.rank) * width // block, width // block)
            for part in range(self.split_factor)
        ]

    def select(self, value, block=1):
        dtype = value.dtype
        if value.element_size() == 1 and value.is_floating_point():
            value = value.view(torch.uint8)
        pieces = [value.narrow(self.dim, start, width) for start, width in self.intervals(block)]
        selected = pieces[0].clone() if len(pieces) == 1 else torch.cat(pieces, dim=self.dim)
        return selected.view(dtype)


def _pinned_native_fields(quantizer, view, fields, source):
    meta = getattr(quantizer, "_native_weight_meta", None)
    if meta is None:
        return
    if _format_key(quantizer, meta["source_shape"]) != _canonical(meta["format"]):
        raise ValueError("Pinned native format changed before partitioning.")
    if tuple(meta["source_shape"]) == view.local_shape:
        if get_pinned_native_weight(quantizer, source) is None:
            raise ValueError("Pinned local native source changed before partitioning.")
        return
    if tuple(meta["source_shape"]) != view.shape:
        raise ValueError("Pinned native payload does not match its physical weight geometry.")
    global_source = tuple(source.shape) == view.shape
    if global_source:
        if get_pinned_native_weight(quantizer, source) is None:
            raise ValueError("Pinned native source changed before partitioning.")
        source = view.select(source)
    elif tuple(source.shape) != view.local_shape:
        raise ValueError(
            "Partition pinned TP fields before FSDP, or enter the full FSDP weight owner."
        )
    cls = get_native_tensor_class(meta["class"])
    if cls not in _NATIVE_TP_CODECS:
        raise ValueError("This native codec needs its own physical payload TP partitioner.")
    packed = cls is NVFP4QTensor
    elements_per_code = 2 if packed else 1
    codes = quantizer._native_weight_codes.view(meta["code_dtype"])
    if codes.numel() % view.shape[0]:
        raise ValueError("Native code padding cannot be partitioned into complete logical rows.")
    physical_columns = codes.numel() // view.shape[0]
    if physical_columns * elements_per_code < view.shape[1]:
        raise ValueError("Native code payload is smaller than the logical matrix.")
    codes = view.select(
        codes.reshape(view.shape[0], physical_columns), elements_per_code if view.dim == 1 else 1
    )
    if len(meta["code_shape"]) == 1:
        codes = codes.flatten()
    native = cls(view.local_shape, meta["dtype"], codes)
    native.metadata.update({**meta.get("native_metadata", {}), "shape": view.local_shape})
    decoded = native.dequantize(
        dtype=source.dtype,
        block_sizes=quantizer.block_sizes,
        **{
            name: fields.get("_" + name, getattr(quantizer, "_" + name, None))
            for name in ("scale", "double_scale", "scale_zeros")
        },
    )
    if not global_source and not torch.equal(decoded, source):
        raise ValueError(
            "Provide the original global source to partition an unrounded native pin; "
            "a local source must match its decoded staged master."
        )
    staged = copy.copy(quantizer)
    staged._buffers = dict(quantizer._buffers)
    pin_native_weight(staged, native, source)
    if meta.get("native_scale_buffers"):
        staged._native_weight_meta["native_scale_buffers"] = True
    fields.update(
        {
            name: value
            for name, value in staged._buffers.items()
            if name.startswith("_native_weight_")
        }
    )
    fields["_native_weight_meta"] = staged._native_weight_meta


def native_weight_partition(quantizer, view, source=None):
    """Plan native fields using the original global source or a decoded local staged master."""
    if not quantizer.fake_quant:
        raise ValueError("Stage packed weights before applying tensor parallelism.")
    blocks = {
        axis % 2: size
        for axis, size in (quantizer.block_sizes or {}).items()
        if isinstance(axis, int) and size is not None
    }
    if quantizer.backend == "ggml":
        blocks = {1: GGML_FORMAT_REGISTRY[quantizer.num_bits].block_size}
    view.intervals(blocks.get(view.dim, 1))
    if quantizer.rotate_is_enabled and view.dim == 1:
        rotation_block = quantizer.rotate_block_size or view.shape[-1]
        view.intervals(rotation_block)
    grid = tuple(math.ceil(n / blocks.get(axis, 1)) for axis, n in enumerate(view.shape))
    local_grid = tuple(
        math.ceil(n / blocks.get(axis, 1)) for axis, n in enumerate(view.local_shape)
    )
    fields = {}
    for name in (
        "_amax",
        "_global_amax",
        "_scale",
        "_double_scale",
        "_scale_zeros",
        "_bias_value",
        "_pre_quant_scale",
    ):
        value = getattr(quantizer, name, None)
        if value is None:
            continue
        if hasattr(value, "to_local"):
            value = value.to_local()
        block_field = name == "_scale" or (name == "_amax" and quantizer.is_static_block_quant)
        if block_field and blocks and value.numel() in (math.prod(grid), math.prod(local_grid)):
            value = (
                view.select(value.reshape(grid), blocks.get(view.dim, 1))
                if value.numel() == math.prod(grid)
                else value.reshape(local_grid)
            )
            if name == "_amax" and quantizer.is_static_block_quant:
                target = (
                    (-1, 1)
                    if set(blocks) == {1}
                    else tuple(
                        n
                        for axis, size in enumerate(local_grid)
                        for n in ((size, 1) if axis in blocks else (size,))
                    )
                )
                value = value.reshape(target)
        elif value.numel() > 1:
            if name == "_pre_quant_scale" and value.shape[-1] == view.shape[-1] and view.dim == 1:
                value = view.select(value.reshape(-1, view.shape[-1])).reshape(
                    *value.shape[:-1], -1
                )
            elif value.ndim == 2 and value.shape[view.dim] == view.shape[view.dim]:
                value = view.select(value)
        fields[name] = value
    if quantizer.backend == "ggml":
        fmt = GGML_FORMAT_REGISTRY[quantizer.num_bits]
        if view.shape[-1] % fmt.block_size:
            raise ValueError("GGML TP partitions require complete logical row blocks.")
        payload_name = PINNED_PAYLOAD_PREFIX + fmt.name
        # Main's source fingerprints and search scales share native block ownership.
        # Dropping them would re-encode calibrated codes after TP.
        for name, value in quantizer.named_buffers(recurse=False):
            if not name.startswith(payload_name):
                continue
            tail = (
                (fmt.block_bytes,)
                if name == payload_name
                else (value.shape[-1],)
                if name.endswith("_source")
                else ()
            )
            global_shape, local_shape = (*grid, *tail), (*local_grid, *tail)
            if value.numel() == math.prod(global_shape):
                value = view.select(
                    value.reshape(global_shape), fmt.block_size if view.dim == 1 else 1
                )
            elif value.numel() != math.prod(local_shape):
                raise ValueError("Pinned IQ fields do not match the physical TP block geometry")
            fields[name] = value.reshape(local_shape)
        importance_name = "_ggml_importance_" + fmt.name
        importance = getattr(quantizer, importance_name, None)
        if importance is not None and view.dim == 1:
            if importance.numel() == view.shape[-1]:
                importance = view.select(importance.reshape(1, -1)).flatten()
            elif importance.numel() != view.local_shape[-1]:
                raise ValueError("IQ importance does not match global or local input columns")
            fields[importance_name] = importance
    elif getattr(quantizer, "_native_weight_meta", None) is not None:
        if source is None:
            raise ValueError("Partitioning a native pin requires its source weight.")
        _pinned_native_fields(quantizer, view, fields, source)
    axis = quantizer.axis
    axes = (axis,) if isinstance(axis, int) else axis or ()
    # Dynamic maxima are preserved only when the split does not divide their reduction.
    dynamic_is_local = (
        quantizer._dynamic
        and not quantizer.block_sizes
        and view.dim in {value % 2 for value in axes}
    )
    if (
        not quantizer.is_mx_format
        and not dynamic_is_local
        and not quantizer._use_constant_amax
        and quantizer.backend != "ggml"
        and getattr(quantizer, "_amax", None) is None
        and ("_scale" not in fields or quantizer._dynamic)
    ):
        raise ValueError(
            "Native TP fields require calibrated thresholds, shard-local dynamic reductions "
            "or pinned native scales."
        )
    layouts = {"global_delta": True, "local_delta": True} if quantizer.backend == "ggml" else {}
    return fields, layouts, bool(blocks) and quantizer.is_static_block_quant


@torch.inference_mode(False)
@torch.no_grad()
def prepare_packed_weights_for_tp(model):
    """Stage packed parameters as logical masters while retaining authoritative native bytes.

    Call before HF shards parameters. Returned metadata records the original packed
    deployment representation; native pins remain the source of codes during repair.
    Masters retain version counters even when the caller stages under inference mode.
    Atomic rollback retains the original packed parameters until all layers succeed;
    peak staging storage includes those payloads, new masters/pins and decoder scratch.
    """
    candidates, aliases, owners = [], {}, {}
    for name, module in model.named_modules():
        weight = module._parameters.get("weight")
        if not isinstance(weight, QTensorWrapper):
            continue
        quantizer = getattr(module, "weight_quantizer", None)
        if not isinstance(quantizer, TensorQuantizer):
            raise ValueError("Packed TP staging requires one physical weight quantizer.")
        native = weight.get_qtensor()
        if len(native.metadata["shape"]) != 2:
            raise ValueError("Packed TP staging requires a logical matrix parameter.")
        if type(native) not in _NATIVE_TP_CODECS:
            raise ValueError("This native codec needs its own physical payload TP partitioner.")
        key = id(weight)
        if aliases.setdefault(key, quantizer) is not quantizer:
            raise ValueError("Tied packed weights require shared native quantizer fields.")
        if owners.setdefault(id(quantizer), key) != key:
            raise ValueError("One native quantizer cannot own multiple physical packed weights.")
        candidates.append((name, module, quantizer, native, key))
    metadata, journal, masters, representations = {}, [], {}, {}
    names = ("_fake_quant", "_enable_pre_quant_scale", "_rotate", "_native_weight_meta")
    try:
        for name, module, quantizer, native, key in candidates:
            scale = quantizer._scale
            swizzled = isinstance(native, NVFP4QTensor) and scale.dtype == torch.uint8
            if swizzled:
                scale = cutlass_fp4_scale_to_modelopt_fp4_scale(
                    scale, native.metadata["shape"][-2:]
                )
            master = None
            if key not in masters:
                master = native.dequantize(
                    dtype=native.metadata["dtype"],
                    block_sizes=quantizer.block_sizes,
                    scale=scale,
                    double_scale=getattr(quantizer, "_double_scale", None),
                    scale_zeros=getattr(quantizer, "_scale_zeros", None),
                )
            journal.append(
                (
                    module,
                    module._parameters["weight"],
                    quantizer,
                    dict(quantizer._buffers),
                    {key: quantizer.__dict__[key] for key in names if key in quantizer.__dict__},
                    getattr(module, "_use_real_quant_gemm", None),
                )
            )
            if key not in masters:
                representations[key] = {
                    "native_metadata": dict(native.metadata),
                    "fake_quant": quantizer.fake_quant,
                    "enable_pre_quant_scale": quantizer._enable_pre_quant_scale,
                    "rotate": quantizer._rotate,
                }
                if swizzled:
                    representations[key]["scale_layout"] = tuple(quantizer._scale.shape)
                    quantizer._set_buffer("_scale", scale)
                quantizer._fake_quant = True
                quantizer._enable_pre_quant_scale = False
                quantizer.disable_rotate()
                pin_native_weight(quantizer, native, master)
                quantizer._native_weight_meta["native_scale_buffers"] = True
                masters[key] = torch.nn.Parameter(master, requires_grad=False)
            metadata[name] = dict(representations[key])
            metadata[name]["real_quant_gemm"] = getattr(module, "_use_real_quant_gemm", None)
            # Bypass compressed-checkpoint loading, which rewraps same-shaped FP8 parameters.
            dict.__setitem__(module._parameters, "weight", masters[key])
            if hasattr(module, "_use_real_quant_gemm"):
                module._use_real_quant_gemm = False
    except BaseException:
        for module, weight, quantizer, buffers, attributes, real_gemm in reversed(journal):
            quantizer._buffers = buffers
            for attribute in names:
                if attribute in attributes:
                    setattr(quantizer, attribute, attributes[attribute])
                else:
                    quantizer.__dict__.pop(attribute, None)
            module._parameters["weight"] = weight
            if real_gemm is not None:
                module._use_real_quant_gemm = real_gemm
        raise
    return metadata
