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
from onnx import AttributeProto, TensorProto, defs, helper, numpy_helper

from ..utils.graph_utils import (
    GraphCache,
    _attribute,
    _shape_before_boundary_promotion,
    batch_replace_nodes,
    convert_initializers_to_dtype,
)
from .dla_constants_to_initializers import _add_legacy_initializer_inputs
from .dla_graph_cleanup import _subgraph_captures
from .dla_indexing import _insert_support_nodes, _operator_nodes
from .dla_rank_conversion import (
    _apply_non4d_lpnorm,
    _runtime_input_names,
    _specialized_collapsed_axis,
)

__all__ = []


def _cast(source, output, dtype, cache):
    """Allocate a boundary cast without colliding with existing graph owners."""
    name = source + "_to_fp16" if dtype == TensorProto.FLOAT16 else output + "_to_fp32"
    return helper.make_node("Cast", [source], [output], name=cache.unique_name(name), to=dtype)


def _fp16_operator_nodes(model, operations):
    """Select standard operations and supported contrib schemas for FP16 policy."""
    nodes = _operator_nodes(model, operations)
    if any(op.domain == "com.microsoft" and op.version == 1 for op in model.opset_import):
        nodes += _operator_nodes(
            model, operations & {"QuantizeLinear", "DequantizeLinear", "Gelu"}, ("com.microsoft",)
        )
    return nodes


def _is_cast(node, dtype):
    """Match a standard Cast with the requested destination type."""
    target = _attribute(node, "to") if node is not None else None
    return (
        node is not None
        and node.domain in {"", "ai.onnx"}
        and not node.overload
        and node.op_type == "Cast"
        and target is not None
        and target.i == dtype
    )


def _apply_fp16_lpnorm(model):
    """Decompose last-axis L2 normalization followed by a static gamma multiply."""
    graph = model.graph
    cache = GraphCache(graph)
    standard = {id(node) for node in _operator_nodes(model, {"LpNormalization", "Mul", "Constant"})}
    static = cache.init_names - _runtime_input_names(model) | {
        node.output[0] for node in graph.node if node.op_type == "Constant" and id(node) in standard
    }
    opset = next((o.version for o in model.opset_import if o.domain in ("", "ai.onnx")), 1)
    reshapes = {id(node) for node in _operator_nodes(model, {"Reshape"})}
    mapped_axes = {}
    for node in _operator_nodes(model, {"LpNormalization"}):
        attribute = _attribute(node, "axis")
        axis = attribute.i if attribute is not None else -1
        shape = _shape_before_boundary_promotion(cache, node.input[0])
        mapped_axes[id(node)] = (
            _specialized_collapsed_axis(cache, node, shape, axis, reshapes)
            if shape and len(shape) == 5
            else -1
        )
    nodes = []
    for node in graph.node:
        if node.op_type != "LpNormalization" or id(node) not in standard:
            nodes.append(node)
            continue
        axis_attr, p_attr = _attribute(node, "axis"), _attribute(node, "p")
        axis = axis_attr.i if axis_attr is not None else -1
        shape = cache.get_shape(node.input[0])
        consumers = cache.get_consumers(node.output[0])
        scaled = (
            len(consumers) == 1
            and consumers[0][0].op_type == "Mul"
            and id(consumers[0][0]) in standard
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
        axis = mapped_axes[id(node)]
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
                    "Mul",
                    [node.input[0], node.input[0]],
                    [power],
                    name=cache.unique_name(base + "_square"),
                ),
                helper.make_node(
                    "ReduceSum",
                    [power, axes] if opset >= 13 else [power],
                    [total],
                    name=cache.unique_name(base + "_sum"),
                    **({} if opset >= 13 else {"axes": [axis]}),
                    keepdims=1,
                ),
                helper.make_node("Sqrt", [total], [norm], name=cache.unique_name(base + "_sqrt")),
                helper.make_node(
                    "Max", [norm, epsilon], [safe], name=cache.unique_name(base + "_safe_norm")
                ),
                helper.make_node(
                    "Div",
                    [node.input[0], safe],
                    list(node.output),
                    name=cache.unique_name(base + "_divide"),
                ),
            ]
        )
    retained = {id(node) for node in nodes}
    batch_replace_nodes(
        graph,
        {index: ([node], []) for index, node in enumerate(graph.node) if id(node) not in retained},
    )
    _insert_support_nodes(graph, nodes)
    _add_legacy_initializer_inputs(model)
    return _apply_non4d_lpnorm(model)


def _validate_float_weights(graph):
    """Validate all FLOAT payloads before calling the existing weight converter."""
    for tensor in graph.initializer:
        if tensor.data_type != TensorProto.FLOAT:
            continue
        count = math.prod(tensor.dims)
        raw = tensor.HasField("raw_data")
        if (
            count < 0
            or (raw and len(tensor.raw_data) != count * 4)
            or (not raw and len(tensor.float_data) != count)
            or (tensor.data_location == TensorProto.EXTERNAL and not raw)
        ):
            raise ValueError(f"Invalid or unloaded FLOAT initializer {tensor.name!r}")


def _fp32_control(node, port, native, required):
    """Retain schema-mandated FLOAT inputs and inputs of native FP32 regions."""
    return id(node) in native or (id(node), port) in required


def _fp32_regions(model, cache):
    """Preserve native floating dependencies and schema-required FLOAT bindings."""
    supported = _fp16_operator_nodes(model, {node.op_type for node in model.graph.node})
    contrib = {id(node) for node in supported if node.domain == "com.microsoft"}
    standard = {id(node) for node in supported}
    native = {
        id(node)
        for node in model.graph.node
        if id(node) not in standard
        or any(
            attr.type in {AttributeProto.GRAPH, AttributeProto.GRAPHS} for attr in node.attribute
        )
        or any(
            (info := cache.vi_map.get(name)) is not None
            and info.type.WhichOneof("value") not in {None, "tensor_type"}
            for name in [*node.input, *node.output]
        )
    }
    version = next((op.version for op in model.opset_import if op.domain in {"", "ai.onnx"}), 1)
    if _operator_nodes(model, {"QuantizeLinear", "DequantizeLinear"}):
        version = max(version, 19)
    pending = list(_subgraph_captures(model.graph))
    schemas, required, outputs = {}, set(), set()
    for node in model.graph.node:
        if id(node) in native:
            pending.extend([*node.input, *node.output])
        if id(node) not in standard:
            continue
        schema_version = version
        if id(node) in contrib:
            # Contrib v1 supports HALF without upgrading the standard opset.
            schema_version = 20 if node.op_type == "Gelu" else 19
        key = node.op_type, schema_version
        if key not in schemas:
            schema = defs.get_schema(node.op_type, schema_version)
            only_fp32 = {
                item.type_param_str
                for item in schema.type_constraints
                if "tensor(float)" in item.allowed_type_strs
                and "tensor(float16)" not in item.allowed_type_strs
            }
            schemas[key] = schema, only_fp32 | {"tensor(float)"}
        schema, only_fp32 = schemas[key]
        for port, name in enumerate(node.input):
            if name and (
                schema.inputs[min(port, len(schema.inputs) - 1)].type_str in only_fp32
                or (node.op_type == "Resize" and port in (1, 2))
            ):
                required.add((id(node), port))
                pending.append(name)
        float_outputs = {name for name in node.output if cache.get_dtype(name) == TensorProto.FLOAT}
        dtype = _attribute(node, "dtype")
        if (dtype is not None and dtype.i == TensorProto.FLOAT) or (
            node.op_type != "Cast"
            and not any(cache.get_dtype(name) == TensorProto.FLOAT for name in node.input)
        ):
            outputs.update(float_outputs)
        outputs.update(
            name
            for port, name in enumerate(node.output)
            if name in float_outputs
            and schema.outputs[min(port, len(schema.outputs) - 1)].type_str in only_fp32
        )
    values = set()
    while pending:
        name = pending.pop()
        if name in values or cache.get_dtype(name) != TensorProto.FLOAT:
            continue
        values.add(name)
        producer = cache.get_producer(name)
        if producer is not None:
            native.add(id(producer))
            pending.extend([*producer.input, *producer.output])
    return values | outputs, native, required


def _upgrade_fp16_quantizers(model, cache):
    """Migrate inherited reduction schemas when FP16 quantizers require opset 19."""
    if not _operator_nodes(model, {"QuantizeLinear", "DequantizeLinear"}):
        return
    opset = next(op for op in model.opset_import if op.domain in {"", "ai.onnx"})
    if opset.version >= 19:
        return
    functions = {(fn.domain, fn.name, fn.overload) for fn in model.functions}
    pending = [model.graph]
    while pending:
        graph = pending.pop()
        allocator = cache if graph is model.graph else GraphCache(graph)
        allocator._names.update(_subgraph_captures(graph))
        for node in graph.node:
            for attr in node.attribute:
                if attr.type == AttributeProto.GRAPH:
                    pending.append(attr.g)
                elif attr.type == AttributeProto.GRAPHS:
                    pending.extend(attr.graphs)
            if (
                node.domain not in {"", "ai.onnx"}
                or node.overload
                or (node.domain, node.op_type, node.overload) in functions
            ):
                continue
            axes = _attribute(node, "axes")
            if axes is None or not node.op_type.startswith("Reduce"):
                continue
            if any(value.name == "axes" for value in defs.get_schema(node.op_type, 19).inputs):
                control = allocator.unique_name((node.name or node.output[0]) + "_axes")
                length = len(axes.ints)
                graph.initializer.append(
                    numpy_helper.from_array(np.asarray(axes.ints, np.int64), control)
                )
                node.input.append(control)
                node.attribute.remove(axes)
                if model.ir_version < 4:
                    graph.input.append(
                        helper.make_tensor_value_info(control, TensorProto.INT64, [length])
                    )
    opset.version = 19


def _apply_force_fp16_activations(model):
    """Run after the ordered DLA passes and their final symbolic shape inference."""
    graph = model.graph
    cache = GraphCache(graph)
    opset = next((op.version for op in model.opset_import if op.domain in {"", "ai.onnx"}), 1)
    if opset < 13 and _operator_nodes(model, {"QuantizeLinear", "DequantizeLinear"}):
        raise ValueError("FP16 quantizer migration requires ONNX opset 13 or newer")
    fp32, fp16 = TensorProto.FLOAT, TensorProto.FLOAT16
    runtime = _runtime_input_names(model)
    native_values, native_nodes, fp32_inputs = _fp32_regions(model, cache)
    preserved_inits = {
        name
        for name in cache.init_names
        if name in runtime
        or name in native_values
        or any(
            _fp32_control(node, port, native_nodes, fp32_inputs)
            for node, port in cache.get_consumers(name)
        )
    }
    _validate_float_weights(graph)
    if any(
        tensor.data_type == fp32 and tensor.name not in preserved_inits
        for tensor in graph.initializer
    ):
        excluded = [
            tensor
            for tensor in graph.initializer
            if tensor.data_type == fp32 and tensor.name in preserved_inits
        ]
        for tensor in excluded:
            graph.initializer.remove(tensor)
        try:
            convert_initializers_to_dtype(graph, "float16")
        finally:
            graph.initializer.extend(excluded)
        cache = GraphCache(graph)
    preserved = set(native_values)
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
    for tensor in graph.initializer:
        if tensor.name not in preserved_inits - runtime or tensor.data_type != fp32:
            continue
        consumers = [
            (node, port)
            for node, port in cache.get_consumers(tensor.name)
            if not _fp32_control(node, port, native_nodes, fp32_inputs) and not _is_cast(node, fp16)
        ]
        if consumers:
            output = cache.unique_name(tensor.name + "_fp16")
            front.append(_cast(tensor.name, output, fp16, cache))
            for node, port in consumers:
                node.input[port] = output
    for node in graph.node:
        for name in node.output:
            if name not in native_values:
                continue
            consumers = [
                (consumer, port)
                for consumer, port in cache.get_consumers(name)
                if not _fp32_control(consumer, port, native_nodes, fp32_inputs)
                and not _is_cast(consumer, fp16)
            ]
            if consumers:
                output = cache.unique_name(name + "_fp16")
                after.setdefault(id(node), []).append(_cast(name, output, fp16, cache))
                for consumer, port in consumers:
                    consumer.input[port] = output
    for node in graph.node:
        if id(node) not in terminal_concats:
            continue
        for port, source in enumerate(node.input):
            if source not in preserved:
                output = cache.unique_name(source + "_concat_fp32")
                before.setdefault(id(node), []).append(_cast(source, output, fp32, cache))
                node.input[port] = output
                preserved.add(output)
    for value in graph.input:
        if value.name in cache.init_names - runtime or value.type.tensor_type.elem_type != fp32:
            continue
        direct = None
        rank = len(value.type.tensor_type.shape.dim)
        for consumer, port in cache.get_consumers(value.name):
            if _fp32_control(consumer, port, native_nodes, fp32_inputs) or _is_cast(consumer, fp16):
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
                after.setdefault(id(consumer), []).append(_cast(source, output, fp16, cache))
                for downstream, slot in cache.get_consumers(source):
                    if not _is_cast(downstream, fp16):
                        downstream.input[slot] = output
                shape_info.type.tensor_type.elem_type = fp32
            else:
                if direct is None:
                    direct = cache.unique_name(value.name + "_fp16")
                    front.append(_cast(value.name, direct, fp16, cache))
                consumer.input[port] = direct
    _upgrade_fp16_quantizers(model, cache)
    for value in graph.output:
        if value.type.tensor_type.elem_type != fp32 or value.name in native_values:
            continue
        producer = cache.get_producer(value.name)
        if producer is not None and producer.op_type == "Squeeze":
            source = producer.input[0]
            if source not in preserved:
                output = cache.unique_name(source + "_to_fp32_fp16")
                before.setdefault(id(producer), []).append(_cast(source, output, fp32, cache))
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
            if model.ir_version < 4:
                for declaration in graph.input:
                    if declaration.name == value.name:
                        declaration.name = internal
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
        back.append(_cast(internal, value.name, fp32, cache))
    nodes = list(front)
    for node in graph.node:
        nodes.extend(before.get(id(node), []))
        nodes.append(node)
        nodes.extend(after.get(id(node), []))
    nodes.extend(back)
    _insert_support_nodes(graph, nodes)
    _rewrite_terminal_projections(model)
    _add_legacy_initializer_inputs(model)
    return model


def _rewrite_terminal_projections(model):
    """Bound private fixed projection channels while preserving native owners."""
    graph = model.graph
    cache = GraphCache(graph)
    standard = {
        id(node) for node in _fp16_operator_nodes(model, {node.op_type for node in graph.node})
    }
    protected = cache.graph_output_names | _subgraph_captures(graph)
    runtime = _runtime_input_names(model)
    removed, removed_values, removed_inits = set(), set(), set()
    replacements, additions = {}, []

    def attributes(node, key, expected):
        """Match one fixed operator attribute used by the projection pattern."""
        value = _attribute(node, key)
        return (
            value is not None
            and (list(value.ints) if isinstance(expected, list) else value.i) == expected
        )

    def axes(node, expected):
        """Match fixed axes without changing their initializer owner."""
        value = cache.get_init_array(node.input[1]) if len(node.input) == 2 else None
        return value is not None and value.reshape(-1).tolist() == expected

    for output in graph.output:
        if output.type.tensor_type.elem_type != TensorProto.FLOAT:
            continue
        chain = []
        current = output.name
        for op in ("Squeeze", "Cast", "Concat", "ReduceSum", "Concat", "Transpose", "Conv"):
            node = cache.get_producer(current)
            if node is None or id(node) not in standard or node.op_type != op:
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
            and ((_attribute(conv, "group") is None) or _attribute(conv, "group").i == 1)
        ):
            continue
        weight = cache.get_init(conv.input[1])
        dq = cache.get_producer(conv.input[1]) if weight is None else None
        if dq is not None:
            parameters = [cache.get_init(name) for name in dq.input[1:]]
            if (
                dq.op_type != "DequantizeLinear"
                or id(dq) not in standard
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
            name in protected | runtime for name in (weight.name, squeeze.input[1], reduce.input[1])
        ):
            continue
        if any(name in protected for item in chain[1:] for name in item.output):
            continue
        if any(
            id(consumer) not in chain_ids
            for name in (weight.name, squeeze.input[1], reduce.input[1])
            for consumer, _ in cache.get_consumers(name)
        ):
            continue
        if any(len(cache.get_consumers(n.output[0])) != 1 for n in chain[1:]):
            continue
        channels = weight.dims[0]
        if cache.get_shape(output.name) != [1, 1, channels] or cache.get_shape(conv.output[0]) != [
            1,
            channels,
            1,
            1,
        ]:
            continue
        bytes_per_channel = math.prod(weight.dims[1:]) * (1 if dq is not None else 2)
        payload = weight.raw_data
        if bytes_per_channel <= 0 or len(payload) != channels * bytes_per_channel:
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
            chunk.raw_data = payload[
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
            nodes.append(_cast(bounded, result, TensorProto.FLOAT, cache))
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
    batch_replace_nodes(
        graph,
        {
            index: ([node], replacements.get(id(node), []))
            for index, node in enumerate(graph.node)
            if id(node) in removed
        },
    )
    for index in range(len(graph.initializer) - 1, -1, -1):
        if graph.initializer[index].name in removed_inits:
            del graph.initializer[index]
    graph.initializer.extend(additions)
    for index in range(len(graph.value_info) - 1, -1, -1):
        if graph.value_info[index].name in removed_values:
            del graph.value_info[index]
    if model.ir_version < 4:
        for index in range(len(graph.input) - 1, -1, -1):
            if graph.input[index].name in removed_inits:
                del graph.input[index]
