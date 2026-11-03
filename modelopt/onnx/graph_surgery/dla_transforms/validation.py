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

"""Validate the intentionally small input contract before constructing the IR."""

from collections.abc import Iterable

import onnx

__all__ = []


def validate(model: onnx.ModelProto) -> None:
    """Accept supported top-level graph and opset forms before transformation."""
    versions = {item.version for item in model.opset_import if item.domain in {"", "ai.onnx"}}
    if len(versions) != 1:
        raise ValueError("Require one unambiguous standard ONNX opset version")
    opset = versions.pop()
    if opset not in (17, 18, 20, 21):
        raise ValueError(
            f"Unsupported ONNX opset {opset}; implement support beyond 17, 18, 20 and 21"
        )
    if model.functions or model.training_info or model.graph.sparse_initializer:
        raise ValueError("TODO: support local functions, training graphs and sparse initializers")
    if any(
        attr.type in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS)
        for node in model.graph.node
        for attr in node.attribute
    ):
        raise ValueError("TODO: support nested graphs in an additional DLA transform")


def quantized_types(values: Iterable[int | str] | None) -> set[int] | None:
    """Resolve explicit storage type names once at the API boundary."""
    if values is None:
        return None
    if isinstance(values, (str, bytes)):
        raise TypeError("qdq_quantized_dtypes must be an iterable of dtype IDs or names")
    result = set()
    for value in values:
        if isinstance(value, str):
            value = (
                int(value) if value.isdecimal() else onnx.TensorProto.DataType.Value(value.upper())
            )
        if not isinstance(value, int) or value not in onnx.TensorProto.DataType.values():
            raise ValueError(f"Invalid ONNX storage dtype: {value}")
        result.add(value)
    return result
