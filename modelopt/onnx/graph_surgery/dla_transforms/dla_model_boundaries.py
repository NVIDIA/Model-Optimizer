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

from ..utils.graph_utils import GraphCache, add_initializer, pad4d

__all__ = []


def _apply_non4d_model_inputs(model):
    graph = model.graph
    cache = GraphCache(graph)
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
        ]
        if not consumers:
            continue
        for consumer in consumers:
            if consumer.op_type != "Unsqueeze":
                continue
            tensor = cache.get_init(consumer.input[1])
            if tensor is None:
                raise ValueError(f"Missing Unsqueeze axes for {consumer.name!r}")
            output_shape = list(shape)
            for axis in numpy_helper.to_array(tensor).reshape(-1).tolist():
                output_shape.insert(axis if axis >= 0 else len(output_shape) + axis + 1, 1)
            if (
                pad4d(shape)[-len(output_shape) :] == output_shape
                and consumer.output[0] in cache.graph_output_names
            ):
                raise ValueError(f"Cannot remove graph-output Unsqueeze {consumer.name!r}")
            unsqueezed_shapes[id(consumer)] = output_shape
        promotions.append((value, shape, consumers))
    for value, shape, consumers in promotions:
        axes = list(range(4 - len(shape)))
        axes_name = add_initializer(
            graph,
            value.name + "_unsqueeze_axes",
            np.asarray(axes, dtype=np.int64),
            dtype=TensorProto.INT64,
            with_value_info=False,
        )
        output = value.name + "_unsqueeze_input"
        prepend.insert(
            0,
            helper.make_node(
                "Unsqueeze", [value.name, axes_name], [output], name=value.name + "_unsqueeze"
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
                        consumer.name + "_reshape_shape",
                        np.asarray(output_shape, dtype=np.int64),
                        dtype=TensorProto.INT64,
                        with_value_info=False,
                    )
                    replacements[id(consumer)] = [
                        helper.make_node(
                            "Reshape",
                            [output, target],
                            list(consumer.output),
                            name=consumer.name + "_reshape",
                        )
                    ]
            else:
                for index, name in enumerate(consumer.input):
                    if name == value.name:
                        consumer.input[index] = output
    nodes = list(graph.node)
    del graph.node[:]
    graph.node.extend(prepend)
    for node in nodes:
        graph.node.extend(replacements.get(id(node), [node]))
    return model


def _apply_non4d_model_outputs(model):
    graph = model.graph
    cache = GraphCache(graph)
    boundary_names = cache.graph_input_names | cache.init_names
    append = []
    for value in graph.output:
        shape = cache.get_shape(value.name)
        if shape is None or len(shape) >= 4:
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
        if existing is not None and len(existing.input) > 1:
            initializer = cache.get_init(existing.input[1])
            if initializer is not None:
                axes = numpy_helper.to_array(initializer).reshape(-1).tolist()
        delta = 4 - len(axes) - len(shape)
        axes = list(range(delta)) + [axis + delta for axis in axes]
        axes_name = None
        if len(shape) > 1:
            axes_name = add_initializer(
                graph,
                value.name + "_squeeze_axes",
                np.asarray(axes, dtype=np.int64),
                dtype=TensorProto.INT64,
                with_value_info=False,
            )
        if existing is not None:
            data_input = existing.input[0]
            del existing.input[:]
            existing.input.extend([data_input] + ([axes_name] if axes_name else []))
            continue
        data_input = value.name + "_squeeze_input"
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
                name=value.name + "_squeeze",
            )
        )
    graph.node.extend(append)
    return model


def _apply_remove_intermediary_squeeze_unsqueeze(model):
    graph = model.graph
    cache = GraphCache(graph)
    removed = set()
    for node in list(graph.node):
        if node.op_type not in {"Squeeze", "Unsqueeze"}:
            continue
        if (
            node.input[0] in cache.graph_input_names | cache.init_names | cache.graph_output_names
            or node.output[0] in cache.graph_output_names
        ):
            continue
        input_shape = cache.get_shape(node.input[0])
        output_shape = cache.get_shape(node.output[0])
        if not input_shape or not output_shape:
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
                node.name + "_reshape_shape",
                np.asarray(pad4d(output_shape), dtype=np.int64),
                dtype=TensorProto.INT64,
                with_value_info=False,
            )
            node.CopyFrom(
                helper.make_node(
                    "Reshape",
                    [node.input[0], target],
                    list(node.output),
                    name=node.name + "_to_reshape",
                )
            )
    remaining = [node for node in graph.node if id(node) not in removed]
    del graph.node[:]
    graph.node.extend(remaining)
    return model


def _apply_fix_instancenorm_channels(model):
    graph = model.graph
    cache = GraphCache(graph)
    nodes = []
    for node in graph.node:
        if node.op_type == "InstanceNormalization":
            shape = cache.get_shape(node.input[0])
            if shape is not None and len(shape) == 3:
                target = add_initializer(
                    graph,
                    node.name + "_rotateleft_shape",
                    np.asarray([*shape, 1], dtype=np.int64),
                    dtype=TensorProto.INT64,
                    with_value_info=False,
                )
                output = cache.unique_name(node.input[0] + "_rotateleft")
                nodes.append(
                    helper.make_node(
                        "Reshape", [node.input[0], target], [output], name=node.name + "_rotateleft"
                    )
                )
                node.input[0] = output
        nodes.append(node)
    del graph.node[:]
    graph.node.extend(nodes)
    return model
