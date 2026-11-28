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

"""Preserve an existing native payload through fake-quant checkpoint round-trips.

The payload is authoritative only while its source weight and format still
match. Scale buffers are live codec fields: changing them changes the native
weight without re-encoding its codes. Amax controls fake-quant fallback, not the
native representation. To recalibrate/re-encode codes, clear _native_weight_meta
first. Cache validation keeps weak references, never gathered floating weights.
"""

import weakref

import torch

from ..config import RotateConfig
from ..ggml.common import _cache_base, _input_cache_key, _row_keys

_VALIDATED = weakref.WeakKeyDictionary()
_SOURCE_DTYPES = {torch.float16: "fp16", torch.bfloat16: "bf16", torch.float32: "fp32"}
_POLICY_FIELDS = (
    "backend_extra_args",
    "_unsigned",
    "_narrow_range",
    "_fake_quant",
    "_dynamic",
    "_bias",
    "_use_constant_amax",
    "_constant_amax",
)


def _canonical(value):
    if isinstance(value, RotateConfig):
        value = value.model_dump()
    if isinstance(value, dict):
        return {str(key): _canonical(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    return value


def _format_key(quantizer, source_shape):
    result = {
        name: _canonical(getattr(quantizer, name, None))
        for name in (
            "_num_bits",
            "_axis",
            "_block_sizes",
            "backend",
            "_rotate",
            "_enable_pre_quant_scale",
            *_POLICY_FIELDS,
        )
    }
    blocks = quantizer.block_sizes
    if blocks and blocks.get("type") != "dynamic":
        # Static block setup derives axis from the block layout on first forward.
        # Treat that initialization as equivalent, while retaining later overrides.
        rank = len(source_shape)
        axes = {axis % rank: size for axis, size in blocks.items() if isinstance(axis, int)}
        if any(size is not None for size in axes.values()):
            derived, offset = [], 0
            for axis in range(rank):
                derived.append(offset)
                offset += 2 if axes.get(axis) is not None else 1
            if len(axes) == 1 and axes.get(rank - 1) is not None:
                derived = [0]
            if not hasattr(quantizer, "_block_reshape_size") or quantizer.axis == tuple(derived):
                result["_axis"] = {"block_axes": derived}
    return result


def _has_input_transform(quantizer):
    return (
        quantizer.pre_quant_scale is not None
        or quantizer.bias is not None
        or quantizer.rotate_is_enabled
        or quantizer.rotate_back_is_enabled
    )


def _source_keys(source, dtype=None):
    rows = source.detach().reshape(-1, source.shape[-1])
    keys = torch.empty(rows.shape[0], dtype=torch.int64, device=source.device)
    chunk_size = max(1, 1048576 // rows.shape[-1])
    for start in range(0, len(rows), chunk_size):
        values = rows[start : start + chunk_size]
        keys[start : start + chunk_size] = _row_keys(values.to(dtype) if dtype else values)
    return keys.reshape(source.shape[:-1])


def pin_native_weight(quantizer, native, source):
    """Pin native codes/scales to the current source without adding a model branch.

    Scale buffers remain live and are not fingerprinted. After changing a scale,
    update the logical source from the decoded native fields and re-pin them
    together. To recalibrate amax, clear ``_native_weight_meta`` first: amax alone
    does not replace authoritative native codes. Re-pin after recalibration.
    """
    if _has_input_transform(quantizer):
        raise ValueError("Native payload pinning requires disabled input transforms.")
    payload = native._quantized_data
    buffers = {"_native_weight_codes": payload.contiguous().view(torch.uint8).clone()}
    buffers.update(
        {
            "_native_weight_source_" + label: _source_keys(source, dtype)
            for dtype, label in _SOURCE_DTYPES.items()
        }
    )
    meta = {
        "class": f"{type(native).__module__}.{type(native).__qualname__}",
        "shape": tuple(native.metadata["shape"]),
        "dtype": native.metadata["dtype"],
        "code_shape": tuple(payload.shape),
        "code_dtype": payload.dtype,
        "source_shape": tuple(source.shape),
        "format": _format_key(quantizer, source.shape),
        "native_metadata": dict(native.metadata),
    }
    previous_buffers = dict(quantizer._buffers)
    previous_meta = getattr(quantizer, "_native_weight_meta", None)
    if previous_meta is not None and previous_meta.get("native_scale_buffers"):
        # Staging describes the logical master, independently of a pin refresh.
        meta["native_scale_buffers"] = True
    _VALIDATED.pop(quantizer, None)
    try:
        for name, value in buffers.items():
            quantizer._set_buffer(name, value)
        quantizer._native_weight_meta = meta
    except BaseException:
        quantizer._buffers = previous_buffers
        if previous_meta is None:
            quantizer.__dict__.pop("_native_weight_meta", None)
        else:
            quantizer._native_weight_meta = previous_meta
        raise


def get_pinned_native_weight(quantizer, source):
    """Return the native QTensor while its source matches, otherwise ``None``.

    Unversioned inference tensors are rechecked on every read to detect in-place
    changes. Packed staging creates versioned masters even inside inference mode.
    """
    # Deferred: base_qtensor imports the quantization utils package at module scope.
    from ..qtensor.base_qtensor import get_native_tensor_class

    meta = getattr(quantizer, "_native_weight_meta", None)
    if (
        not meta
        or _has_input_transform(quantizer)
        or tuple(source.shape) != tuple(meta["source_shape"])
        or source.dtype not in _SOURCE_DTYPES
    ):
        return None
    expected_format = _canonical(meta["format"])
    if _format_key(quantizer, source.shape) != expected_format:
        return None
    expected = getattr(quantizer, "_native_weight_source_" + _SOURCE_DTYPES[source.dtype])
    key = (_input_cache_key(source), _input_cache_key(expected))
    bases = (_cache_base(source), _cache_base(expected))
    previous = _VALIDATED.get(quantizer)
    if (
        previous is None
        or any(ref() is not base for ref, base in zip(previous[0], bases))
        or previous[1] != key
    ):
        if not torch.equal(_source_keys(source), expected):
            return None
        if all(value is not None for value in key):
            _VALIDATED[quantizer] = (tuple(weakref.ref(base) for base in bases), key)
    payload = quantizer._native_weight_codes.view(meta["code_dtype"]).reshape(meta["code_shape"])
    native = get_native_tensor_class(meta["class"])(meta["shape"], meta["dtype"], payload)
    native.metadata.update(meta.get("native_metadata", {}))
    return native


def dequantize_pinned_native(quantizer, native, dtype):
    """Decode pinned fields through their native codec and original scale layout."""
    return native.dequantize(
        dtype=dtype,
        scale=quantizer._scale,
        double_scale=getattr(quantizer, "_double_scale", None),
        scale_zeros=getattr(quantizer, "_scale_zeros", None),
        block_sizes=quantizer.block_sizes,
    )
