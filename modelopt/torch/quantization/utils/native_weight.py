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
match. Cache validation keeps weak references, never gathered floating weights.
"""

import weakref

import torch

from ..config import RotateConfig
from .weight_source import _cache_base, _input_cache_key, _row_keys

_VALIDATED = weakref.WeakKeyDictionary()
_SOURCE_DTYPES = {torch.float16: "fp16", torch.bfloat16: "bf16", torch.float32: "fp32"}
_POLICY_DEFAULTS = {
    "backend_extra_args": {},
    "_unsigned": False,
    "_narrow_range": False,
    "_fake_quant": True,
    "_dynamic": False,
    "_bias": None,
    "_use_constant_amax": False,
    "_constant_amax": None,
}


def _canonical(value):
    if isinstance(value, RotateConfig):
        value = value.model_dump()
    if isinstance(value, dict):
        return {str(key): _canonical(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    return value


def _format_key(quantizer):
    return {
        name: _canonical(getattr(quantizer, name, None))
        for name in (
            "_num_bits",
            "_axis",
            "_block_sizes",
            "backend",
            "_rotate",
            "_rotate_back",
            "_rotate_block_size",
            "_rotate_is_fp32",
            *_POLICY_DEFAULTS,
        )
    }


def pin_native_weight(quantizer, native, source):
    """Pin native codes/scales to the current source without adding a model branch."""
    payload = native._quantized_data
    quantizer._set_buffer("_native_weight_codes", payload.contiguous().view(torch.uint8).clone())
    for dtype, label in _SOURCE_DTYPES.items():
        quantizer._set_buffer(
            "_native_weight_source_" + label, _row_keys(source.detach().to(dtype))
        )
    quantizer._native_weight_meta = {
        "class": type(native).__name__,
        "shape": tuple(native.metadata["shape"]),
        "dtype": native.metadata["dtype"],
        "code_shape": tuple(payload.shape),
        "code_dtype": payload.dtype,
        "source_shape": tuple(source.shape),
        "format": _format_key(quantizer),
        "native_metadata": dict(native.metadata),
    }
    _VALIDATED.pop(quantizer, None)


def get_pinned_native_weight(quantizer, source):
    """Return the native QTensor while its source matches, otherwise ``None``."""
    from ..qtensor.base_qtensor import get_native_tensor_class

    meta = getattr(quantizer, "_native_weight_meta", None)
    if (
        not meta
        or tuple(source.shape) != tuple(meta["source_shape"])
        or source.dtype not in _SOURCE_DTYPES
    ):
        return None
    expected_format = _canonical(meta["format"])
    if "_backend" in expected_format:
        # Older repair checkpoints stored a nonexistent _backend attribute.
        # Their native class identifies the backend without re-encoding codes.
        expected_format.pop("_backend")
        expected_format["backend"] = getattr(
            get_native_tensor_class(meta["class"]), "quantization_backend", None
        )
        expected_format = {**_POLICY_DEFAULTS, **expected_format}
    if _format_key(quantizer) != expected_format:
        return None
    key = _input_cache_key(source)
    previous = _VALIDATED.get(quantizer)
    if previous is None or previous[0]() is None or previous[1] != key:
        expected = getattr(quantizer, "_native_weight_source_" + _SOURCE_DTYPES[source.dtype])
        if not torch.equal(_row_keys(source), expected):
            return None
        if key is not None:
            _VALIDATED[quantizer] = (weakref.ref(_cache_base(source)), key)
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
