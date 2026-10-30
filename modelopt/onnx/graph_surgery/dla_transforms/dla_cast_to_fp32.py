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

"""Redirect eligible standard Cast targets to FP32 using ONNX type constraints.

Shared type variables connect operands and outputs across operations. Fixed
non-FP32 operands, integer or boolean controls, opaque operations and lexical
captures prevent conversion of their connected type groups. Compatible Casts
can convert together; their connected tensor metadata is updated as well.
"""

from __future__ import annotations

import onnx
from onnx import TensorProto

from ...logging_config import logger
from ..utils.graph_utils import GraphCache, _captured_names
from .dla_topk import _topk_plans


def _fp32_cast_outputs(model: onnx.ModelProto, *, preserve_topk_indices: bool = False) -> set[str]:
    """Find tensor type groups that can become FP32 without changing fixed operands."""
    graph = model.graph
    candidates = {
        node.output[0]
        for node in graph.node
        if node.op_type == "Cast"
        and node.domain in {"", "ai.onnx"}
        and not node.overload
        and node.output
        and any(attr.name == "to" and attr.i != TensorProto.FLOAT for attr in node.attribute)
    }
    if not candidates:
        return set()
    parents: dict[str, str] = {}

    def find(name: str) -> str:
        """Find and compress a tensor's shared-type component."""
        parents.setdefault(name, name)
        while parents[name] != name:
            parents[name] = parents[parents[name]]
            name = parents[name]
        return name

    blocked: set[str] = set()
    if preserve_topk_indices and any(node.op_type == "TopK" for node in graph.node):
        blocked.update(
            cast.output[0]
            for _, cast, *_ in _topk_plans(model, GraphCache(graph), rank_converted=True)
        )
    types = {
        value.name: value.type.tensor_type.elem_type
        for value in (*graph.input, *graph.value_info, *graph.output)
        if value.type.HasField("tensor_type")
    }
    types.update((value.name, value.data_type) for value in graph.initializer)
    fixed = {value.name for value in graph.input} | {value.name for value in graph.initializer}
    fixed.update(value.values.name for value in graph.sparse_initializer)
    types.update((value.values.name, value.values.data_type) for value in graph.sparse_initializer)
    opsets = {opset.domain: opset.version for opset in model.opset_import}
    opsets[""] = opsets.get("", opsets.get("ai.onnx", 1))
    schemas = {}
    for node in graph.node:
        for attribute in node.attribute:
            subgraphs = (
                [attribute.g] if attribute.type == onnx.AttributeProto.GRAPH else attribute.graphs
            )
            for subgraph in subgraphs:
                blocked.update(_captured_names(subgraph))
        domain = "" if node.domain == "ai.onnx" else node.domain
        key = (domain, node.op_type)
        if key not in schemas:
            try:
                schemas[key] = onnx.defs.get_schema(node.op_type, opsets.get(domain, 1), domain)
            except onnx.defs.SchemaError:
                schemas[key] = None
        schema = schemas[key]
        if (
            schema is None
            or node.overload
            # SequenceInsert couples sequence and tensor types across separate schema variables.
            or (domain == "" and node.op_type == "SequenceInsert")
            or any(
                parameter.option == onnx.defs.OpSchema.FormalParameterOption.Variadic
                and not parameter.is_homogeneous
                for parameter in (*schema.inputs, *schema.outputs)
            )
        ):
            blocked.update(name for name in (*node.input, *node.output) if name)
            continue
        if (
            domain == ""
            and node.op_type == "Mod"
            and schema.since_version < 28
            and not any(
                attribute.name == "fmod" and attribute.i == 1 for attribute in node.attribute
            )
        ):
            # Older Mod schemas require integer inputs in the default attribute mode.
            blocked.update(name for name in (*node.input, *node.output) if name)
            continue
        allowed = {
            constraint.type_param_str: constraint.allowed_type_strs
            for constraint in schema.type_constraints
        }
        groups: dict[str, list[str]] = {}
        input_types: set[str] = set()
        for values, parameters, is_output in (
            (node.input, schema.inputs, False),
            (node.output, schema.outputs, True),
        ):
            for index, name in enumerate(values):
                if not name:
                    continue
                parameter = parameters[min(index, len(parameters) - 1)]
                type_name = parameter.type_str
                groups.setdefault(type_name, []).append(name)
                if "tensor(float)" not in allowed.get(type_name, [type_name]):
                    blocked.add(name)
                if not is_output:
                    input_types.add(type_name)
                elif type_name not in input_types:
                    fixed.add(name)
                    if type_name == "tensor(float)":
                        types[name] = TensorProto.FLOAT
                    elif type_name in allowed and node.op_type != "Cast":
                        # Independent output variables may still default to an input's dtype.
                        blocked.update(name for name in node.input if name)
        for names in groups.values():
            for name in names[1:]:
                parents[find(name)] = find(names[0])
        if node.op_type == "Cast" and domain == "":
            for attribute in node.attribute:
                if attribute.name == "to":
                    types[node.output[0]] = attribute.i
        elif node.op_type == "Constant" and domain == "":
            for attribute in node.attribute:
                if attribute.name == "value":
                    types[node.output[0]] = attribute.t.data_type
        # Runtime attention inputs have implementation-specific precision requirements.
        if node.op_type == "GroupQueryAttention":
            blocked.update(name for name in node.input if name)
    blocked.update(name for name in fixed - candidates if types.get(name) != TensorProto.FLOAT)
    blocked_roots = {find(name) for name in blocked}
    promoted_roots = {find(name) for name in candidates if find(name) not in blocked_roots}
    return {name for name in parents if find(name) in promoted_roots}


def _apply_cast_to_fp32(
    model: onnx.ModelProto, *, preserve_topk_indices: bool = False
) -> onnx.ModelProto:
    """Change every eligible Cast node's ``to`` attribute to FLOAT (float32).

    Preserve connected type requirements, custom operations and subgraph captures.
    Update intermediate and graph-output descriptors for converted type groups.
    """
    graph = model.graph
    promoted = _fp32_cast_outputs(model, preserve_topk_indices=preserve_topk_indices)
    value_info_by_name = {}
    for value_info in graph.value_info:
        value_info_by_name.setdefault(value_info.name, value_info)
    outputs_by_name = {output.name: output for output in graph.output}
    cnt = 0
    for node in graph.node:
        if node.op_type != "Cast":
            continue
        if not node.output or node.output[0] not in promoted:
            continue
        for attr in node.attribute:
            if attr.name == "to" and attr.i != TensorProto.FLOAT:
                attr.i = TensorProto.FLOAT
                cnt += 1
                break

    for name in promoted:
        for value_info in (value_info_by_name.get(name), outputs_by_name.get(name)):
            if value_info is not None:
                value_info.type.tensor_type.elem_type = TensorProto.FLOAT

    logger.debug("cast_to_fp32: redirected %d Cast node(s) to float32.", cnt)
    return model
