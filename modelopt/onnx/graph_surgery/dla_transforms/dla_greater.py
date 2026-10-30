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

"""Prepare FLOAT operands only where conversion preserves comparison ordering."""

from __future__ import annotations

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from ..utils.graph_utils import GraphCache, add_unique_initializers, add_value_info
from .dla_constants_to_initializers import _add_legacy_initializer_inputs
from .dla_indexing import _insert_support_nodes, _operator_nodes
from .dla_rank_conversion import _runtime_input_names

__all__ = []

_INT_TYPES = frozenset({TensorProto.INT32, TensorProto.INT64})
_EXACT_TYPES = frozenset(
    {
        TensorProto.FLOAT,
        TensorProto.FLOAT16,
        TensorProto.INT8,
        TensorProto.UINT8,
        TensorProto.INT16,
        TensorProto.UINT16,
    }
)

_SUPPORTED_TYPES = _EXACT_TYPES | _INT_TYPES


def _apply_greater(model: onnx.ModelProto) -> onnx.ModelProto:
    """Convert comparisons whose ordering survives FLOAT operand conversion."""
    graph = model.graph
    cache = GraphCache(graph)
    selected = {id(node) for node in _operator_nodes(model, {"Greater"})}
    runtime = _runtime_input_names(model)
    unique_init = set(cache.init_names)
    converted, safe_thresholds = {}, {}
    nodes = []
    for node in list(graph.node):
        if id(node) not in selected:
            nodes.append(node)
            continue
        types = {cache.get_dtype(name) for name in node.input}
        safe = types <= _EXACT_TYPES
        if not safe and types <= _SUPPORTED_TYPES:
            for name in node.input:
                if name not in safe_thresholds:
                    tensor = cache.get_init(name) if name not in runtime else None
                    array = numpy_helper.to_array(tensor) if tensor is not None else None
                    # Interior thresholds retain separation from either adjacent integer.
                    safe_thresholds[name] = array is not None and (
                        array.size == 0 or (array.min() > -(2**24) and array.max() < 2**24)
                    )
                safe |= safe_thresholds[name]
        if not safe:
            nodes.append(node)
            continue
        for index, name in enumerate(node.input):
            if cache.get_dtype(name) == TensorProto.FLOAT:
                continue
            if name not in converted:
                output = cache.unique_name(
                    name + "_cast"
                    if cache.is_init(name) and name not in runtime
                    else name + "_to_float32"
                )
                if cache.is_init(name) and name not in runtime:
                    array = cache.get_init_array(name)
                    assert array is not None
                    add_unique_initializers(
                        graph,
                        unique_init,
                        [numpy_helper.from_array(array.astype(np.float32), output)],
                    )
                else:
                    nodes.append(
                        helper.make_node(
                            "Cast",
                            [name],
                            [output],
                            name=cache.unique_name(name + "_cast_to_float32"),
                            to=TensorProto.FLOAT,
                        )
                    )
                    shape = cache.get_shape(name)
                    if shape is not None:
                        add_value_info(graph, output, TensorProto.FLOAT, shape)
                converted[name] = output
            node.input[index] = converted[name]
        nodes.append(node)
    _insert_support_nodes(graph, nodes)
    _add_legacy_initializer_inputs(model)
    return model
