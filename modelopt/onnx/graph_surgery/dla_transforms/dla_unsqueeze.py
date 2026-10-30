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

"""Prepare proven FLOAT consumer bindings while preserving integer Unsqueeze owners."""

import onnx
from onnx import TensorProto, helper

from ..utils.graph_utils import GraphCache, add_value_info
from .dla_indexing import _insert_support_nodes, _operator_nodes


def _float_input(cache, node, port, source, version):
    """Require a FLOAT schema binding instead of guessing which ports accept integers."""
    if node.op_type == "Cast":
        return False
    try:
        schema = onnx.defs.get_schema(node.op_type, version, "")
    except onnx.defs.SchemaError:
        return False
    if not schema.inputs:
        return False
    formal = schema.inputs[min(port, len(schema.inputs) - 1)].type_str
    if formal == "tensor(float)":
        return True
    constraints = {item.type_param_str: item.allowed_type_strs for item in schema.type_constraints}
    if "tensor(float)" not in constraints.get(formal, []):
        return False
    bindings = [
        cache.get_dtype(name)
        for index, name in enumerate(node.input)
        if name
        and name != source
        and schema.inputs[min(index, len(schema.inputs) - 1)].type_str == formal
    ]
    bindings.extend(
        dtype
        for index, name in enumerate(node.output)
        if name
        and schema.outputs[min(index, len(schema.outputs) - 1)].type_str == formal
        and (dtype := cache.get_dtype(name)) is not None
    )
    return bool(bindings) and all(dtype == TensorProto.FLOAT for dtype in bindings)


def _apply_unsqueeze(model):
    """Add private FLOAT views only for consumers whose schema already requires FLOAT."""
    graph = model.graph
    cache = GraphCache(graph)
    standard = {id(node) for node in _operator_nodes(model, {node.op_type for node in graph.node})}
    version = next((op.version for op in model.opset_import if op.domain in {"", "ai.onnx"}), 1)
    after = {}
    for node in _operator_nodes(model, {"Unsqueeze"}):
        if cache.get_dtype(node.input[0]) not in (TensorProto.INT32, TensorProto.INT64):
            continue
        consumers = [
            (consumer, port)
            for consumer, port in cache.get_consumers(node.output[0])
            if id(consumer) in standard
            and _float_input(cache, consumer, port, node.output[0], version)
        ]
        if not consumers:
            continue
        output = cache.unique_name(node.output[0] + "_float32")
        cast = helper.make_node(
            "Cast",
            [node.output[0]],
            [output],
            to=TensorProto.FLOAT,
            name=cache.unique_name(node.output[0] + "_cast_to_float32"),
        )
        add_value_info(graph, output, TensorProto.FLOAT, cache.get_shape(node.output[0]))
        for consumer, port in consumers:
            consumer.input[port] = output
        after[id(node)] = cast
    nodes = []
    for node in graph.node:
        nodes.append(node)
        if id(node) in after:
            nodes.append(after[id(node)])
    _insert_support_nodes(graph, nodes)
    return model
