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

"""Apply the C++ FP16 activation policy while retaining FP32 public boundaries."""

import copy
import math

import numpy as np
from onnx import TensorProto, helper, numpy_helper

from ..utils.graph_utils import GraphCache, _attribute, convert_initializers_to_dtype
from .dla_rank_conversion import _apply_non4d_lpnorm

__all__ = []


def _cast(source, output, dtype):
    name = source + "_to_fp16" if dtype == TensorProto.FLOAT16 else output + "_to_fp32"
    return helper.make_node("Cast", [source], [output], name=name, to=dtype)


def _is_cast(node, dtype):
    target = _attribute(node, "to") if node is not None else None
    return node is not None and node.op_type == "Cast" and target is not None and target.i == dtype


def _apply_fp16_lpnorm(model):
    """Decompose last-axis L2 normalization followed by a static gamma multiply."""
    graph = model.graph
    cache = GraphCache(graph)
    static = cache.init_names | {n.output[0] for n in graph.node if n.op_type == "Constant"}
    opset = next((o.version for o in model.opset_import if o.domain in ("", "ai.onnx")), 1)
    nodes = []
    for node in graph.node:
        if node.op_type != "LpNormalization":
            nodes.append(node)
            continue
        axis_attr, p_attr = _attribute(node, "axis"), _attribute(node, "p")
        axis = axis_attr.i if axis_attr is not None else -1
        shape = cache.get_shape(node.input[0])
        consumers = cache.get_consumers(node.output[0])
        scaled = (
            len(consumers) == 1
            and consumers[0][0].op_type == "Mul"
            and any(name in static for name in consumers[0][0].input if name != node.output[0])
        )
        if (
            (p_attr is not None and p_attr.i != 2)
            or not scaled
            or not (axis == -1 or (shape is not None and axis == len(shape) - 1))
        ):
            nodes.append(node)
            continue
        base = node.name or node.output[0] + "_lpnorm"
        axes = cache.unique_name(base + "_axes")
        epsilon = cache.unique_name(base + "_epsilon")
        if opset >= 13:
            graph.initializer.append(numpy_helper.from_array(np.array([axis], np.int64), axes))
        graph.initializer.append(numpy_helper.from_array(np.float32(2**-14), epsilon))
        power, total, norm, safe = [
            cache.unique_name(base + suffix)
            for suffix in ("_power_output", "_sum_output", "_norm_output", "_safe_norm_output")
        ]
        nodes.extend(
            [
                helper.make_node(
                    "Mul", [node.input[0], node.input[0]], [power], name=base + "_square"
                ),
                helper.make_node(
                    "ReduceSum",
                    [power, axes] if opset >= 13 else [power],
                    [total],
                    name=base + "_sum",
                    **({} if opset >= 13 else {"axes": [axis]}),
                    keepdims=1,
                ),
                helper.make_node("Sqrt", [total], [norm], name=base + "_sqrt"),
                helper.make_node("Max", [norm, epsilon], [safe], name=base + "_safe_norm"),
                helper.make_node(
                    "Div", [node.input[0], safe], list(node.output), name=base + "_divide"
                ),
            ]
        )
    del graph.node[:]
    graph.node.extend(nodes)
    return _apply_non4d_lpnorm(model)


def _apply_force_fp16_activations(model):
    """Run after the ordered DLA passes and their final symbolic shape inference."""
    graph = model.graph
    cache = GraphCache(graph)
    fp32, fp16 = TensorProto.FLOAT, TensorProto.FLOAT16
    preserved_inits = {
        name
        for name in cache.init_names
        if any(n.op_type == "Resize" and port in (1, 2) for n, port in cache.get_consumers(name))
    }
    excluded_inits = [init for init in graph.initializer if init.name in preserved_inits]
    for init in excluded_inits:
        graph.initializer.remove(init)
    try:
        convert_initializers_to_dtype(graph, "float16")
    finally:
        graph.initializer.extend(excluded_inits)
    # Conversion replaces initializer objects, invalidating the cached references.
    cache = GraphCache(graph)
    preserved = set()
    terminal_concats = set()
    for node in graph.node:
        if not _is_cast(node, fp32) or cache.get_dtype(node.input[0]) in (fp32, fp16, None):
            continue
        for concat, _ in cache.get_consumers(node.output[0]):
            if concat.op_type != "Concat" or len(concat.output) != 1:
                continue
            uses = cache.get_consumers(concat.output[0])
            if len(uses) == 1:
                boundary, port = uses[0]
                if (
                    port == 0
                    and boundary.op_type == "Squeeze"
                    and len(boundary.output) == 1
                    and (boundary.output[0] in cache.graph_output_names)
                ):
                    preserved.add(concat.output[0])
                    terminal_concats.add(id(concat))
    for node in graph.node:
        if _is_cast(node, fp32) and node.output[0] not in cache.graph_output_names | preserved:
            _attribute(node, "to").i = fp16
    for value in graph.value_info:
        if (
            value.type.HasField("tensor_type")
            and value.type.tensor_type.elem_type == fp32
            and (value.name not in cache.graph_output_names | preserved | preserved_inits)
        ):
            value.type.tensor_type.elem_type = fp16
    for value in graph.input:
        if (
            value.name in cache.init_names - preserved_inits
            and value.type.tensor_type.elem_type == fp32
        ):
            value.type.tensor_type.elem_type = fp16
    before, after = {}, {}
    front, back = [], []
    for node in graph.node:
        if id(node) not in terminal_concats:
            continue
        for port, source in enumerate(node.input):
            if source not in preserved:
                output = cache.unique_name(source + "_concat_fp32")
                before.setdefault(id(node), []).append(_cast(source, output, fp32))
                node.input[port] = output
                preserved.add(output)
    for value in graph.input:
        if value.name in cache.init_names or value.type.tensor_type.elem_type != fp32:
            continue
        direct = None
        rank = len(value.type.tensor_type.shape.dim)
        for consumer, port in cache.get_consumers(value.name):
            if _is_cast(consumer, fp16):
                continue
            shape_info = cache.vi_map.get(consumer.output[0])
            adapter = (
                rank < 4
                and port == 0
                and consumer.op_type == "Unsqueeze"
                and (shape_info is not None and len(shape_info.type.tensor_type.shape.dim) == 4)
            )
            if adapter:
                assert shape_info is not None
                source = consumer.output[0]
                output = cache.unique_name(source + "_fp16")
                after.setdefault(id(consumer), []).append(_cast(source, output, fp16))
                for downstream, slot in cache.get_consumers(source):
                    if not _is_cast(downstream, fp16):
                        downstream.input[slot] = output
                shape_info.type.tensor_type.elem_type = fp32
            else:
                if direct is None:
                    direct = cache.unique_name(value.name + "_fp16")
                    front.append(_cast(value.name, direct, fp16))
                consumer.input[port] = direct
    if any(
        n.op_type in {"QuantizeLinear", "DequantizeLinear"} and n.domain in ("", "ai.onnx")
        for n in graph.node
    ):
        opsets = [o for o in model.opset_import if o.domain in ("", "ai.onnx")]
        if opsets:
            opsets[0].version = max(19, opsets[0].version)
        else:
            model.opset_import.append(helper.make_opsetid("", 19))
    for value in graph.output:
        if value.type.tensor_type.elem_type != fp32:
            continue
        producer = cache.get_producer(value.name)
        if producer is not None and producer.op_type == "Squeeze":
            source = producer.input[0]
            if source not in preserved:
                output = cache.unique_name(source + "_to_fp32_fp16")
                before.setdefault(id(producer), []).append(_cast(source, output, fp32))
                producer.input[0] = output
            continue
        if _is_cast(producer, fp32):
            continue
        tensor = cache.get_init(value.name) if producer is None else None
        if producer is None and (tensor is None or tensor.data_type != fp16):
            continue
        internal = cache.unique_name(value.name + "_fp16")
        if producer is None:
            assert tensor is not None
            tensor.name = internal
        else:
            for port, name in enumerate(producer.output):
                if name == value.name:
                    producer.output[port] = internal
                    break
        for annotation in graph.value_info:
            if annotation.name == value.name:
                annotation.name = internal
                annotation.type.tensor_type.elem_type = fp16
        for consumer, port in cache.get_consumers(value.name):
            consumer.input[port] = internal
        back.append(_cast(internal, value.name, fp32))
    nodes = list(front)
    for node in graph.node:
        nodes.extend(before.get(id(node), []))
        nodes.append(node)
        nodes.extend(after.get(id(node), []))
    nodes.extend(back)
    del graph.node[:]
    graph.node.extend(nodes)
    _rewrite_terminal_projections(graph)
    return model


def _rewrite_terminal_projections(graph):
    """Bound terminal projection channels to DLA's 8192-element dimension limit."""
    cache = GraphCache(graph)
    removed, removed_values, removed_inits = set(), set(), set()
    replacements, additions = {}, []

    def attributes(node, key, expected):
        value = _attribute(node, key)
        return (
            value is not None
            and (list(value.ints) if isinstance(expected, list) else value.i) == expected
        )

    def axes(node, expected):
        value = cache.get_init_array(node.input[1]) if len(node.input) == 2 else None
        return value is not None and value.reshape(-1).tolist() == expected

    for output in graph.output:
        if output.type.tensor_type.elem_type != TensorProto.FLOAT:
            continue
        chain = []
        current = output.name
        for op in ("Squeeze", "Cast", "Concat", "ReduceSum", "Concat", "Transpose", "Conv"):
            node = cache.get_producer(current)
            if node is None or node.op_type != op:
                break
            chain.append(node)
            current = node.input[0]
        if len(chain) != 7:
            continue
        squeeze, cast, final, reduce, initial, transpose, conv = chain
        if not (
            axes(squeeze, [0])
            and _is_cast(cast, TensorProto.FLOAT)
            and len(final.input) == len(initial.input) == 1
            and axes(reduce, [0])
            and attributes(reduce, "keepdims", 1)
            and attributes(transpose, "perm", [0, 3, 2, 1])
            and len(conv.input) == 2
            and len(conv.output) == 1
        ):
            continue
        weight = cache.get_init(conv.input[1])
        dq = cache.get_producer(conv.input[1]) if weight is None else None
        if dq is not None:
            parameters = [cache.get_init(name) for name in dq.input[1:]]
            if (
                dq.op_type != "DequantizeLinear"
                or len(dq.input) < 2
                or any(
                    parameter is None or math.prod(parameter.dims) != 1 for parameter in parameters
                )
            ):
                continue
            weight = cache.get_init(dq.input[0])
            chain.append(dq)
        if (
            weight is None
            or len(weight.dims) != 4
            or weight.dims[0] <= 8192
            or not weight.HasField("raw_data")
        ):
            continue
        if weight.data_type != (TensorProto.UINT8 if dq is not None else TensorProto.FLOAT16):
            continue
        chain_ids = {id(n) for n in chain}
        if any(
            id(consumer) not in chain_ids
            for name in (weight.name, squeeze.input[1], reduce.input[1])
            for consumer, _ in cache.get_consumers(name)
        ):
            continue
        if any(len(cache.get_consumers(n.output[0])) != 1 for n in chain[1:]):
            continue
        channels = weight.dims[0]
        if cache.get_shape(output.name) != [1, 1, channels]:
            continue
        bytes_per_channel = math.prod(weight.dims[1:]) * (1 if dq is not None else 2)
        if bytes_per_channel <= 0 or len(weight.raw_data) != channels * bytes_per_channel:
            continue
        nodes, outputs = [], []
        for index, start in enumerate(range(0, channels, 8192)):
            count = min(8192, channels - start)
            suffix = f"_dla_chunk_{index}"
            chunk = TensorProto(
                name=cache.unique_name(weight.name + suffix),
                data_type=weight.data_type,
                dims=[count, *weight.dims[1:]],
            )
            chunk.raw_data = weight.raw_data[
                start * bytes_per_channel : (start + count) * bytes_per_channel
            ]
            additions.append(chunk)
            cloned_conv = copy.deepcopy(conv)
            cloned_conv.name = cache.unique_name(conv.name + suffix)
            if dq is not None:
                cloned_dq = copy.deepcopy(dq)
                cloned_dq.name = cache.unique_name(dq.name + suffix)
                cloned_dq.input[0] = chunk.name
                cloned_dq.output[0] = cache.unique_name(dq.output[0] + suffix)
                nodes.append(cloned_dq)
                cloned_conv.input[1] = cloned_dq.output[0]
            else:
                cloned_conv.input[1] = chunk.name
            cloned_conv.output[0] = cache.unique_name(conv.output[0] + suffix)
            nodes.append(cloned_conv)
            shape = cache.unique_name(conv.output[0] + suffix + "_bounded_shape")
            additions.append(numpy_helper.from_array(np.array([1, 1, count], np.int64), shape))
            bounded = cache.unique_name(conv.output[0] + suffix + "_bounded")
            nodes.append(
                helper.make_node(
                    "Reshape",
                    [cloned_conv.output[0], shape],
                    [bounded],
                    name=cache.unique_name(conv.name + suffix + "_bounded_reshape"),
                )
            )
            result = cache.unique_name(conv.output[0] + suffix + "_fp32")
            nodes.append(_cast(bounded, result, TensorProto.FLOAT))
            outputs.append(result)
        nodes.append(
            helper.make_node(
                "Concat",
                outputs,
                [output.name],
                axis=2,
                name=cache.unique_name(output.name + "_dla_terminal_concat"),
            )
        )
        replacements[id(conv)] = nodes
        removed.update(id(n) for n in chain)
        removed_values.update(name for n in chain for name in n.output if name != output.name)
        removed_inits.update([weight.name, squeeze.input[1], reduce.input[1]])
    if not replacements:
        return
    nodes = []
    for node in graph.node:
        nodes.extend(replacements.get(id(node), []))
        if id(node) not in removed:
            nodes.append(node)
    retained_inits = [t for t in graph.initializer if t.name not in removed_inits]
    retained_info = [v for v in graph.value_info if v.name not in removed_values]
    del graph.node[:]
    graph.node.extend(nodes)
    del graph.initializer[:]
    graph.initializer.extend([*retained_inits, *additions])
    del graph.value_info[:]
    graph.value_info.extend(retained_info)
