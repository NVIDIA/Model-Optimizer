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

"""Promote model boundaries at their explicit positions in the DLA pipeline."""

import numpy as np
from onnx import TensorProto, helper, numpy_helper

from ..utils.graph_utils import GraphCache, add_initializer, batch_replace_nodes, pad4d
from .dla_constants_to_initializers import _add_legacy_initializer_inputs
from .dla_graph_cleanup import _subgraph_captures
from .dla_reorder_nodes import priority_topological_order

__all__ = []


def _require_adapter_opset(model):
    """Refuse input-form axes adapters before changing a legacy graph."""
    version = next(
        (entry.version for entry in model.opset_import if entry.domain in {"", "ai.onnx"}), 1
    )
    if version < 13:
        raise ValueError("DLA boundary rank adapters require ONNX opset 13 or newer")


def _apply_non4d_model_inputs(model):
    """Promote selected boundary uses while preserving explicit adapter results."""
    graph = model.graph
    cache = GraphCache(graph)
    captured = _subgraph_captures(graph)
    replacements = {}
    prepend = []
    promotions = []
    unsqueezed_shapes = {}
    for value in graph.input:
        shape = cache.get_shape(value.name)
        if shape is None or len(shape) not in (2, 3):
            continue
        consumers = [
            node
            for node, _ in cache.get_consumers(value.name)
            if (node.input[0] == value.name or node.op_type == "Concat")
            and not (node.op_type == "Squeeze" and "Gather" in node.name)
            and node.op_type != "InstanceNormalization"
        ]
        if not consumers:
            continue
        _require_adapter_opset(model)
        for consumer in consumers:
            if consumer.op_type != "Unsqueeze":
                continue
            tensor = cache.get_init(consumer.input[1])
            if tensor is None or tensor.name in cache.graph_input_names:
                raise ValueError(f"Unsqueeze axes must be fixed for {consumer.name!r}")
            output_shape = list(shape)
            axes = numpy_helper.to_array(tensor).reshape(-1).tolist()
            output_rank = len(shape) + len(axes)
            for axis in sorted(axis % output_rank for axis in axes):
                output_shape.insert(axis, 1)
            if (
                pad4d(shape)[-len(output_shape) :] == output_shape
                and consumer.output[0] in cache.graph_output_names | captured
            ):
                raise ValueError(
                    f"Cannot remove graph-output or captured Unsqueeze {consumer.name!r}"
                )
            unsqueezed_shapes[id(consumer)] = output_shape
        promotions.append((value, shape, consumers))
    for value, shape, consumers in promotions:
        axes = list(range(4 - len(shape)))
        axes_name = add_initializer(
            graph,
            cache.unique_name(value.name + "_unsqueeze_axes"),
            np.asarray(axes, dtype=np.int64),
            dtype=TensorProto.INT64,
            with_value_info=False,
        )
        output = cache.unique_name(value.name + "_unsqueeze_input")
        prepend.insert(
            0,
            helper.make_node(
                "Unsqueeze",
                [value.name, axes_name],
                [output],
                name=cache.unique_name(value.name + "_unsqueeze"),
            ),
        )
        promoted_shape = pad4d(shape)
        graph.value_info.append(
            helper.make_tensor_value_info(output, value.type.tensor_type.elem_type, promoted_shape)
        )
        for consumer in consumers:
            if consumer.op_type == "Unsqueeze":
                output_shape = unsqueezed_shapes[id(consumer)]
                if promoted_shape[-len(output_shape) :] == output_shape:
                    old_output = consumer.output[0]
                    for downstream, _ in cache.get_consumers(old_output):
                        for index, name in enumerate(downstream.input):
                            if name == old_output:
                                downstream.input[index] = output
                    replacements[id(consumer)] = []
                else:
                    target = add_initializer(
                        graph,
                        cache.unique_name((consumer.name or consumer.output[0]) + "_reshape_shape"),
                        np.asarray(output_shape, dtype=np.int64),
                        dtype=TensorProto.INT64,
                        with_value_info=False,
                    )
                    replacements[id(consumer)] = [
                        helper.make_node(
                            "Reshape",
                            [output, target],
                            list(consumer.output),
                            name=cache.unique_name(
                                (consumer.name or consumer.output[0]) + "_reshape"
                            ),
                        )
                    ]
            else:
                for index, name in enumerate(consumer.input):
                    if name == value.name:
                        consumer.input[index] = output
    batch = {
        index: ([node], replacements[id(node)])
        for index, node in enumerate(graph.node)
        if id(node) in replacements
    }
    if prepend:
        removed, added = batch.get(0, ([], []))
        batch[0] = (removed, [*prepend, *added])
    batch_replace_nodes(graph, batch)
    _add_legacy_initializer_inputs(model)
    return model


def _apply_non4d_model_outputs(model, *, defer_rank_conversion=False):
    """Restore four-dimensional outputs, optionally deferring checks to later rank stages."""
    graph = model.graph
    cache = GraphCache(graph)
    captured = _subgraph_captures(graph)
    boundary_names = cache.graph_input_names | cache.init_names
    append = []
    plans = []
    for value in graph.output:
        shape = cache.get_shape(value.name)
        if shape is None or len(shape) >= 4 or value.name in boundary_names:
            continue
        producer = cache.get_producer(value.name)
        if (
            producer is not None
            and producer.op_type == "Identity"
            and producer.input[0] in boundary_names
        ):
            continue
        if len(shape) == 0:
            raise ValueError(f"Unsupported scalar model output {value.name!r}")
        axes = []
        existing = producer if producer is not None and producer.op_type == "Squeeze" else None
        if existing is not None:
            if len(existing.input) > 1 and existing.input[1]:
                initializer = cache.get_init(existing.input[1])
                if initializer is None or initializer.name in cache.graph_input_names:
                    raise ValueError(f"Output Squeeze axes must be fixed for {value.name!r}")
                axes = numpy_helper.to_array(initializer).reshape(-1).tolist()
            else:
                input_shape = cache.get_shape(existing.input[0])
                if input_shape is None:
                    raise ValueError(f"Output Squeeze input shape must be known for {value.name!r}")
                axes = [index for index, dim in enumerate(input_shape) if dim == 1]
        source_shape = cache.get_shape(existing.input[0] if existing is not None else value.name)
        if not defer_rank_conversion and (source_shape is None or len(source_shape) != 4):
            raise ValueError(
                f"Output restoration requires a known rank-4 producer for {value.name!r}"
            )
        delta = 4 - len(axes) - len(shape)
        if delta < 0:
            raise ValueError(f"Output Squeeze input rank exceeds 4 for {value.name!r}")
        axes = list(range(delta)) + [axis + delta if axis >= 0 else axis for axis in axes]
        plans.append((value, shape, existing, axes))
    if plans:
        _require_adapter_opset(model)
    for value, shape, existing, axes in plans:
        axes_name = None
        if len(shape) > 1 or shape == [1]:
            axes_name = add_initializer(
                graph,
                cache.unique_name(value.name + "_squeeze_axes"),
                np.asarray(axes, dtype=np.int64),
                dtype=TensorProto.INT64,
                with_value_info=False,
            )
        if existing is not None:
            data_input = existing.input[0]
            del existing.input[:]
            existing.input.extend([data_input] + ([axes_name] if axes_name else []))
            continue
        data_input = cache.unique_name(value.name + "_squeeze_input")
        for node in graph.node:
            for index, name in enumerate(node.input):
                if name == value.name:
                    node.input[index] = data_input
            for index, name in enumerate(node.output):
                if name == value.name:
                    node.output[index] = data_input
        append.append(
            helper.make_node(
                "Squeeze",
                [data_input] + ([axes_name] if axes_name else []),
                [value.name],
                name=cache.unique_name(value.name + "_squeeze"),
            )
        )
    graph.node.extend(append)
    if any(value.name in captured for value, _, _, _ in plans):
        order, _, _ = priority_topological_order(graph)
        nodes = list(graph.node)
        ranks = {id(nodes[index]): rank for rank, index in enumerate(order)}
        graph.node.sort(key=lambda node: ranks[id(node)])
    _add_legacy_initializer_inputs(model)
    return model


def _apply_remove_intermediary_squeeze_unsqueeze(model, *, rank_converted=False):
    """Remove no-op adapters, or reconcile adapters within the ordered rank-converted pipeline."""
    graph = model.graph
    cache = GraphCache(graph)
    captured = _subgraph_captures(graph)
    removed = set()
    for node in list(graph.node):
        if node.op_type not in {"Squeeze", "Unsqueeze"}:
            continue
        if (
            node.input[0] in cache.graph_input_names | cache.init_names | cache.graph_output_names
            or node.output[0] in cache.graph_output_names | captured
            or node.input[0] in captured
        ):
            continue
        input_shape = cache.get_shape(node.input[0])
        output_shape = cache.get_shape(node.output[0])
        if not input_shape or not output_shape:
            continue
        if not rank_converted and input_shape != output_shape:
            continue
        short, long = (
            (output_shape, input_shape)
            if node.op_type == "Squeeze"
            else (input_shape, output_shape)
        )
        if len(short) <= len(long) and long[-len(short) :] == short:
            old, new = node.input[0], node.output[0]
            for other in graph.node:
                if other is node:
                    continue
                for index, name in enumerate(other.output):
                    if name == old:
                        other.output[index] = new
                for index, name in enumerate(other.input):
                    if name == old:
                        other.input[index] = new
            removed.add(id(node))
        else:
            target = add_initializer(
                graph,
                cache.unique_name((node.name or node.output[0]) + "_reshape_shape"),
                np.asarray(pad4d(output_shape), dtype=np.int64),
                dtype=TensorProto.INT64,
                with_value_info=False,
            )
            node.CopyFrom(
                helper.make_node(
                    "Reshape",
                    [node.input[0], target],
                    list(node.output),
                    name=cache.unique_name((node.name or node.output[0]) + "_to_reshape"),
                )
            )
    batch_replace_nodes(
        graph, {index: ([node], []) for index, node in enumerate(graph.node) if id(node) in removed}
    )
    _add_legacy_initializer_inputs(model)
    return model


def _apply_fix_instancenorm_channels(model):
    """Append spatial singleton dimensions while retaining normalization channels and outputs."""
    graph = model.graph
    cache = GraphCache(graph)
    replacements = {}
    for index, node in enumerate(graph.node):
        if node.op_type == "InstanceNormalization":
            shape = cache.get_shape(node.input[0])
            if shape is not None and len(shape) == 3:
                target = add_initializer(
                    graph,
                    cache.unique_name((node.name or node.output[0]) + "_rotateleft_shape"),
                    np.asarray([*shape, 1], dtype=np.int64),
                    dtype=TensorProto.INT64,
                    with_value_info=False,
                )
                output = cache.unique_name(node.input[0] + "_rotateleft")
                reshape = helper.make_node(
                    "Reshape",
                    [node.input[0], target],
                    [output],
                    name=cache.unique_name((node.name or node.output[0]) + "_rotateleft"),
                )
                node.input[0] = output
                replacements[index] = ([node], [reshape, node])
                for consumer, _ in cache.get_consumers(node.output[0]):
                    if (
                        consumer.op_type != "Squeeze"
                        or consumer.output[0] not in cache.graph_output_names
                        or len(consumer.input) < 2
                    ):
                        continue
                    axes = cache.get_init(consumer.input[1])
                    if axes is not None and numpy_helper.to_array(axes).reshape(-1).tolist() == [0]:
                        consumer.input[1] = add_initializer(
                            graph,
                            cache.unique_name(axes.name + "_trailing"),
                            np.asarray([3], np.int64),
                            dtype=TensorProto.INT64,
                            with_value_info=False,
                        )
    batch_replace_nodes(graph, replacements)
    _add_legacy_initializer_inputs(model)
    return model
