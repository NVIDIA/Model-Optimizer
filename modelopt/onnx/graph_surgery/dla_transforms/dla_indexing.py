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

"""DLA indexing, reduction and attention passes with deferred rank promotion."""

import numpy as np
from onnx import TensorProto, helper, numpy_helper

from ..utils.graph_utils import (
    GraphCache,
    _attribute,
    _shape_before_boundary_promotion,
    add_initializer,
    batch_replace_nodes,
    pad4d,
)
from .dla_constants_to_initializers import _add_legacy_initializer_inputs
from .dla_graph_cleanup import _subgraph_captures

__all__ = []


def _insert_support_nodes(graph, nodes):
    """Insert generated support nodes without copying existing protobuf payloads."""
    positions = {id(node): index for index, node in enumerate(graph.node)}
    batch = {}
    pending = []
    for node in nodes:
        index = positions.get(id(node))
        if index is None:
            pending.append(node)
        elif pending:
            batch[index] = ([], pending)
            pending = []
    if pending:
        batch[len(graph.node)] = ([], pending)
    batch_replace_nodes(graph, batch)


def _operator_nodes(model, operators, domains=("", "ai.onnx")):
    """Select supported schemas without interpreting opaque model-local functions."""
    functions = {(fn.domain, fn.name, fn.overload) for fn in model.functions}
    return [
        node
        for node in model.graph.node
        if node.op_type in operators
        and node.domain in domains
        and not node.overload
        and (node.domain, node.op_type, node.overload) not in functions
    ]


def _apply_gather(model):
    """Prepare Gather indices for the subsequent ordered data-rank promotion."""
    graph = model.graph
    cache = GraphCache(graph)
    nodes = []
    squeezed_indices = {}
    selected = _operator_nodes(model, {"Gather"})
    for candidate in selected:
        if cache.get_shape(candidate.input[0]) is None:
            raise ValueError(f"{candidate.name!r}: Gather requires a static data shape")
        if cache.get_init(candidate.input[1]) is None and cache.get_dtype(
            candidate.input[1]
        ) not in (TensorProto.INT32, TensorProto.INT64):
            producer = cache.get_producer(candidate.input[1])
            if producer is None or producer.op_type != "Cast":
                raise ValueError(
                    f"{candidate.name!r}: dynamic Gather indices require an integer Cast"
                )
    selected_ids = {id(node) for node in selected}
    for node in graph.node:
        if id(node) not in selected_ids:
            nodes.append(node)
            continue
        indices = cache.get_init(node.input[1])
        if indices is not None and node.input[1] not in cache.graph_input_names:
            if indices.name not in squeezed_indices:
                values = numpy_helper.to_array(indices).reshape(-1).astype(np.int32)
                squeezed_indices[indices.name] = add_initializer(
                    graph,
                    cache.unique_name(indices.name + "_gather_indices"),
                    values,
                    dtype=TensorProto.INT32,
                    with_value_info=False,
                )
            node.input[1] = squeezed_indices[indices.name]
        else:
            original = node.input[1]
            index_shape = cache.get_shape(original)
            if index_shape is not None and len(index_shape) <= 1:
                if original not in squeezed_indices:
                    prepared = cache.unique_name(original + "_gather_int32")
                    nodes.append(
                        helper.make_node(
                            "Cast", [original], [prepared], name=prepared, to=TensorProto.INT32
                        )
                    )
                    squeezed_indices[original] = prepared
            if original in squeezed_indices:
                node.input[1] = squeezed_indices[original]
                axis = _attribute(node, "axis")
                shape = cache.get_shape(node.input[0])
                if axis is None:
                    node.attribute.append(helper.make_attribute("axis", 0))
                    axis = _attribute(node, "axis")
                if axis.i >= 0 and shape:
                    axis.i += {2: 2, 3: 1}.get(len(shape), 0)
                node.name = cache.unique_name((node.name or node.output[0]) + "_axis_updated")
                nodes.append(node)
                continue
            squeezed = cache.unique_name(original + "_squeezed")
            squeeze_inputs = [original]
            producer = cache.get_producer(original)
            if producer is not None and producer.op_type == "Cast":
                target = _attribute(producer, "to")
                shared = (
                    original in cache.graph_output_names or len(cache.get_consumers(original)) > 1
                )
                if shared:
                    integer = cache.unique_name(original + "_gather_int32")
                    nodes.append(
                        helper.make_node(
                            "Cast",
                            [producer.input[0]],
                            [integer],
                            name=integer,
                            to=TensorProto.INT32,
                        )
                    )
                    squeeze_inputs[0] = integer
                else:
                    target.i = TensorProto.INT32
                    for value in graph.value_info:
                        if value.name == original:
                            value.type.tensor_type.elem_type = TensorProto.INT32
                axes = add_initializer(
                    graph,
                    cache.unique_name(node.name + "_indices_squeeze_axes"),
                    np.asarray([0, 1, 2], dtype=np.int64),
                    dtype=TensorProto.INT64,
                    with_value_info=False,
                )
                squeeze_inputs.append(axes)
            elif cache.get_dtype(original) not in (TensorProto.INT32, TensorProto.INT64):
                raise ValueError(f"{node.name!r}: dynamic Gather indices require an integer Cast")
            nodes.append(
                helper.make_node("Squeeze", squeeze_inputs, [squeezed], name=node.name + "_squeeze")
            )
            node.input[1] = squeezed
            squeezed_indices[original] = squeezed
            graph.value_info.append(
                helper.make_tensor_value_info(
                    squeezed, cache.get_dtype(original) or TensorProto.FLOAT, None
                )
            )
        if _attribute(node, "axis") is None:
            node.attribute.append(helper.make_attribute("axis", 0))
        axis = _attribute(node, "axis")
        shape = cache.get_shape(node.input[0])
        if axis is not None:
            if axis.i >= 0 and shape:
                axis.i += {2: 2, 3: 1}.get(len(shape), 0)
            node.name = cache.unique_name((node.name or node.output[0]) + "_axis_updated")
        nodes.append(node)
    _insert_support_nodes(graph, nodes)
    _add_legacy_initializer_inputs(model)
    return model


def _apply_gather_elements(model):
    """Prepare matching index ranks and preserve native axis and runtime-value contracts."""
    graph = model.graph
    cache = GraphCache(graph)
    converted_indices = {}
    replacements = {}
    selected_ids = {id(node) for node in _operator_nodes(model, {"GatherElements"})}
    for index, node in enumerate(graph.node):
        if id(node) not in selected_ids:
            continue
        original = node.input[1]
        shape = cache.get_shape(original)
        tensor = cache.get_init(original)
        rank = len(tensor.dims) if tensor is not None else len(shape) if shape is not None else None
        added = []
        if original in converted_indices:
            prepared, rank = converted_indices[original]
        elif tensor is not None and original not in cache.graph_input_names:
            array = numpy_helper.to_array(tensor).astype(np.int32)
            if rank in (2, 3):
                array = array.reshape([*([1] * (4 - rank)), *array.shape])
            elif rank == 0:
                array = array.reshape(1)
            prepared = add_initializer(
                graph,
                cache.unique_name(tensor.name + "_gatherelements_indices"),
                array,
                dtype=TensorProto.INT32,
                with_value_info=False,
            )
        else:
            prepared = cache.unique_name(original + "_cast_to_int32")
            added.append(
                helper.make_node(
                    "Cast",
                    [original],
                    [prepared],
                    name=cache.unique_name(original + "_cast_to_int32_node"),
                    to=TensorProto.INT32,
                )
            )
            # Other intermediate producers are promoted by the subsequent rank stages.
            if rank in (2, 3) and original in cache.graph_input_names:
                axes = add_initializer(
                    graph,
                    cache.unique_name(original + "_gatherelements_axes"),
                    np.arange(4 - rank, dtype=np.int64),
                    dtype=TensorProto.INT64,
                    with_value_info=False,
                )
                promoted = cache.unique_name(original + "_gatherelements_indices")
                added.append(
                    helper.make_node(
                        "Unsqueeze",
                        [prepared, axes],
                        [promoted],
                        name=cache.unique_name(original + "_indices_unsqueeze"),
                    )
                )
                prepared = promoted
        converted_indices[original] = (prepared, rank)
        node.input[1] = prepared
        if rank in (2, 3):
            axis = _attribute(node, "axis")
            if axis is None:
                node.attribute.append(helper.make_attribute("axis", 0))
                axis = _attribute(node, "axis")
            if axis.i >= 0:
                axis.i += 4 - rank
            node.name = cache.unique_name((node.name or node.output[0]) + "_axis_updated")
        replacements[index] = ([node], [*added, node])
    batch_replace_nodes(graph, replacements)
    live = {name for node in graph.node for name in node.input}
    live.update(cache.graph_input_names | cache.graph_output_names | _subgraph_captures(graph))
    for original in converted_indices.keys() - live:
        tensor = cache.get_init(original)
        if tensor is not None:
            graph.initializer.remove(tensor)
    _add_legacy_initializer_inputs(model)
    return model


def _apply_standalone_reduce_operations(model):
    """Plan native reduction axes and preserve their output layouts during rank promotion."""
    graph = model.graph
    cache = GraphCache(graph)
    nodes = []
    count = 0
    plans = {}
    for node in _operator_nodes(model, {"ReduceSum", "ReduceMean", "ReduceMax", "ReduceMin"}):
        shape = _shape_before_boundary_promotion(cache, node.input[0])
        if not shape:
            keepdims = _attribute(node, "keepdims")
            if keepdims is not None and keepdims.i == 0:
                raise ValueError(f"Missing static reduction input shape for {node.name!r}")
            continue  # Already rank-preserving, including generated 4D reduction branches.
        tensor = cache.get_init(node.input[1]) if len(node.input) > 1 else None
        if len(node.input) > 1 and (tensor is None or tensor.name in cache.graph_input_names):
            raise ValueError(f"Dynamic reduction axes are unsupported for {node.name!r}")
        attribute = _attribute(node, "axes")
        raw_axes = (
            numpy_helper.to_array(tensor).reshape(-1).tolist()
            if tensor is not None
            else list(attribute.ints)
            if attribute is not None
            else list(range(len(shape)))
        )
        reduced = {value % len(shape) for value in raw_axes or range(len(shape))}
        output_shape = cache.get_shape(node.output[0])
        if output_shape is None and node.output[0].endswith("_squeeze_input"):
            output_shape = cache.get_shape(node.output[0].removesuffix("_squeeze_input"))
        if output_shape is None:
            output_shape = [dim for index, dim in enumerate(shape) if index not in reduced]
        plans[id(node)] = (shape, raw_axes, output_shape)
    for node in graph.node:
        nodes.append(node)
        if id(node) not in plans:
            continue
        shape, raw_axes, output_shape = plans[id(node)]
        keepdims = _attribute(node, "keepdims")
        reshape = keepdims is not None and keepdims.i == 0
        if keepdims is not None:
            keepdims.i = 1
        delta = {2: 2, 3: 1}.get(len(shape), 0)
        values = [value + delta if value >= 0 else value for value in raw_axes]
        if len(node.input) > 1:
            node.input[1] = add_initializer(
                graph,
                cache.unique_name((node.name or node.output[0]) + "_axes"),
                np.asarray(values, dtype=np.int64),
                dtype=TensorProto.INT64,
                with_value_info=False,
            )
        else:
            attribute = _attribute(node, "axes")
            if attribute is not None:
                del attribute.ints[:]
                attribute.ints.extend(values)
        if reshape:
            target = pad4d(output_shape)
            name = add_initializer(
                graph,
                cache.unique_name(f"reshape_shape_{count}"),
                np.asarray(target, dtype=np.int64),
                dtype=TensorProto.INT64,
                with_value_info=False,
            )
            original = node.output[0]
            node.output[0] = cache.unique_name(original + "_temp")
            nodes.append(
                helper.make_node(
                    "Reshape",
                    [node.output[0], name],
                    [original],
                    name=cache.unique_name((node.name or original) + "_reshape"),
                )
            )
            count += 1
    _insert_support_nodes(graph, nodes)
    _add_legacy_initializer_inputs(model)
    return model


def _apply_group_query_attention(model):
    """Restore native attention ranks across internal tensors while retaining public boundaries."""
    graph = model.graph
    cache = GraphCache(graph)
    nodes = []
    boundaries = cache.graph_input_names | cache.init_names | cache.graph_output_names
    squeezed_inputs = {}
    axes_names = {}

    def axes_for(name):
        """Reuse a private axes control for adapters of the same attention tensor."""
        if name not in axes_names:
            axes_names[name] = add_initializer(
                graph,
                cache.unique_name(name + "_axis"),
                np.asarray([0], dtype=np.int64),
                dtype=TensorProto.INT64,
                with_value_info=False,
            )
        return axes_names[name]

    selected_ids = {
        id(node) for node in _operator_nodes(model, {"GroupQueryAttention"}, ("com.microsoft",))
    }
    for node in graph.node:
        if id(node) not in selected_ids:
            nodes.append(node)
            continue
        for index, name in enumerate(node.input[:3]):
            if name and name not in boundaries:
                if name in squeezed_inputs:
                    node.input[index] = squeezed_inputs[name]
                    continue
                axes = axes_for(name)
                squeezed = cache.unique_name(name + "_squeezed")
                squeezed_inputs[name] = squeezed
                nodes.append(
                    helper.make_node(
                        "Squeeze",
                        [name, axes],
                        [squeezed],
                        name=cache.unique_name(node.name + "_" + name + "_squeeze"),
                    )
                )
                node.input[index] = squeezed
        nodes.append(node)
        for index, name in enumerate(node.output[:1]):
            if name and name not in cache.graph_output_names:
                axes = axes_for(name)
                node.output[index] = cache.unique_name(name + "_squeezed")
                nodes.append(
                    helper.make_node(
                        "Unsqueeze",
                        [node.output[index], axes],
                        [name],
                        name=cache.unique_name(node.name + "_" + name + "_unsqueeze"),
                    )
                )
    _insert_support_nodes(graph, nodes)
    _add_legacy_initializer_inputs(model)
    return model
