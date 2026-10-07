# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Map ONNX ``TensorProto`` element types to NumPy dtypes and readable names.

Use this when matching quantized tensor / zero-point types (e.g. in QDQ removal).
See :func:`tensorproto_dtype_table` for the full table and :func:`parse_qdq_quantized_dtype_list`
to convert user input into ``TensorProto`` enum integers.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

import numpy as np
from onnx import TensorProto


def tensorproto_members() -> dict[str, int]:
    """Return every ``TensorProto`` uppercase name -> integer (from the installed ONNX)."""
    out: dict[str, int] = {}
    for name in dir(TensorProto):
        if not name.isupper() or name.startswith("_"):
            continue
        val = getattr(TensorProto, name)
        if isinstance(val, int):
            out[name] = val
    return out


_TENSORPROTO_BY_UPPER_NAME = tensorproto_members()

# Standard ONNX DataType -> NumPy dtype where there is a direct counterpart
TENSORPROTO_TO_NUMPY: dict[int, np.dtype | None] = {
    TensorProto.FLOAT: np.dtype(np.float32),
    TensorProto.UINT8: np.dtype(np.uint8),
    TensorProto.INT8: np.dtype(np.int8),
    TensorProto.UINT16: np.dtype(np.uint16),
    TensorProto.INT16: np.dtype(np.int16),
    TensorProto.INT32: np.dtype(np.int32),
    TensorProto.INT64: np.dtype(np.int64),
    TensorProto.STRING: np.dtype(np.str_),
    TensorProto.BOOL: np.dtype(np.bool_),
    TensorProto.FLOAT16: np.dtype(np.float16),
    TensorProto.DOUBLE: np.dtype(np.float64),
    TensorProto.UINT32: np.dtype(np.uint32),
    TensorProto.UINT64: np.dtype(np.uint64),
    TensorProto.COMPLEX64: np.dtype(np.complex64),
    TensorProto.COMPLEX128: np.dtype(np.complex128),
}

with contextlib.suppress(AttributeError, TypeError):  # ONNX 1.14+
    TENSORPROTO_TO_NUMPY[TensorProto.BFLOAT16] = np.dtype("bfloat16")

for _tp_id in _TENSORPROTO_BY_UPPER_NAME.values():
    TENSORPROTO_TO_NUMPY.setdefault(_tp_id, None)

# NumPy dtype kind -> TensorProto (for reverse resolution from string like "float32")
_NUMPY_NAME_TO_TENSORPROTO: dict[str, int] = {
    "float32": TensorProto.FLOAT,
    "float64": TensorProto.DOUBLE,
    "float16": TensorProto.FLOAT16,
    "uint8": TensorProto.UINT8,
    "int8": TensorProto.INT8,
    "uint16": TensorProto.UINT16,
    "int16": TensorProto.INT16,
    "uint32": TensorProto.UINT32,
    "int32": TensorProto.INT32,
    "uint64": TensorProto.UINT64,
    "int64": TensorProto.INT64,
    "bool": TensorProto.BOOL,
    "str": TensorProto.STRING,
    "complex64": TensorProto.COMPLEX64,
    "complex128": TensorProto.COMPLEX128,
}

# Aliases users might type (lowercase -> TensorProto int)
DTYPE_NAME_ALIASES: dict[str, int] = {
    **_NUMPY_NAME_TO_TENSORPROTO,
    "float": TensorProto.FLOAT,
    "double": TensorProto.DOUBLE,
    "fp32": TensorProto.FLOAT,
    "fp16": TensorProto.FLOAT16,
    "u8": TensorProto.UINT8,
    "i8": TensorProto.INT8,
    "u16": TensorProto.UINT16,
    "i16": TensorProto.INT16,
    "u32": TensorProto.UINT32,
    "i32": TensorProto.INT32,
    "u64": TensorProto.UINT64,
    "i64": TensorProto.INT64,
}

# TensorProto names lowercased (UINT16 -> uint16, BFLOAT16 -> bfloat16, …)
for _upper, _tid in _TENSORPROTO_BY_UPPER_NAME.items():
    DTYPE_NAME_ALIASES.setdefault(_upper.lower(), _tid)


def numpy_dtype_for_tensorproto(tensorproto_type: int) -> np.dtype | None:
    """NumPy dtype equivalent for ``tensorproto_type``, or ``None`` if not mapped."""
    if tensorproto_type in TENSORPROTO_TO_NUMPY:
        return TENSORPROTO_TO_NUMPY[tensorproto_type]
    return None


def tensorproto_to_name(tensorproto_type: int) -> str:
    """Best-effort ``TensorProto`` enum name (e.g. ``UINT16``)."""
    for upper, tid in _TENSORPROTO_BY_UPPER_NAME.items():
        if tid == tensorproto_type:
            return upper
    return f"DATA_TYPE({tensorproto_type})"


def tensorproto_dtype_table() -> list[tuple[int, str, str | None]]:
    """Rows ``(enum_int, TensorProto name, numpy dtype string or None)`` for documentation."""
    rows: list[tuple[int, str, str | None]] = []
    seen: set[int] = set()
    for upper, tid in sorted(_TENSORPROTO_BY_UPPER_NAME.items(), key=lambda x: x[1]):
        if tid in seen:
            continue
        seen.add(tid)
        np_d = numpy_dtype_for_tensorproto(tid)
        np_s = str(np_d) if np_d is not None else None
        rows.append((tid, upper, np_s))
    return rows


def as_tensorproto_int(spec: int | str) -> int:
    """Convert ``TensorProto.UINT16``-style int, name ``UINT16`` / ``uint16``, or alias to enum int."""
    if isinstance(spec, int):
        return spec

    key = str(spec).strip()
    if key.upper() in _TENSORPROTO_BY_UPPER_NAME:
        return _TENSORPROTO_BY_UPPER_NAME[key.upper()]
    lowered = key.lower()
    if lowered in DTYPE_NAME_ALIASES:
        return DTYPE_NAME_ALIASES[lowered]
    raise ValueError(
        f"Unknown dtype {spec!r}. Use TensorProto name (e.g. UINT16), numpy-like name (uint16), "
        f"or integer TensorProto code. See tensorproto_dtype_table()."
    )


def parse_qdq_quantized_dtype_list(dtypes: Iterable[int | str]) -> frozenset[int]:
    """Normalize user-provided dtypes to a frozenset of ``TensorProto`` ints."""
    return frozenset(as_tensorproto_int(x) for x in dtypes)


DEFAULT_QDQ_REMOVE_QUANT_TYPES: frozenset[int] = frozenset({TensorProto.UINT16, TensorProto.INT16})
