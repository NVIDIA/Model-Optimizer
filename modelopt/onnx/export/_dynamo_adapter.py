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

"""Dynamo-only normalization and validation for quantized ONNX graphs."""

from collections.abc import Iterable

import numpy as np
import onnx
from onnx import numpy_helper

from modelopt.onnx.quantization.graph_indexing import (
    get_tensor_consumer_nodes,
    get_tensor_producer_nodes,
)

__all__ = []

_DYNAMO_INT4_NODE_PREFIX = "__modelopt_dynamo_int4__"
_DYNAMO_MXFP8_NODE_PREFIX = "__modelopt_dynamo_mxfp8__"


def _attribute(node: onnx.NodeProto, name: str):
    attribute = next((item for item in node.attribute if item.name == name), None)
    return onnx.helper.get_attribute_value(attribute) if attribute is not None else None


def _single_consumer(consumers, name: str, weight_name: str) -> onnx.NodeProto:
    matches = consumers.get(name, [])
    if len(matches) != 1:
        raise NotImplementedError(
            f"Unsupported Dynamo quantized weight topology for '{weight_name}': "
            f"expected one consumer; found {len(matches)}."
        )
    return matches[0]


def _constant_tensor(producers, name: str) -> onnx.TensorProto:
    producer = producers.get(name)
    if isinstance(producer, onnx.TensorProto):
        return producer
    if isinstance(producer, onnx.NodeProto) and producer.op_type == "Constant":
        value = _attribute(producer, "value")
        if isinstance(value, onnx.TensorProto):
            return value
    raise NotImplementedError("Dynamo ONNX export requires constant weight shapes and scales.")


def _replace_initializer(graph: onnx.GraphProto, tensor: onnx.TensorProto) -> None:
    existing = next((item for item in graph.initializer if item.name == tensor.name), None)
    if existing is None:
        graph.initializer.append(tensor)
    else:
        existing.CopyFrom(tensor)


def _remove_nodes(graph: onnx.GraphProto, nodes: Iterable[onnx.NodeProto]) -> None:
    output_names = {output for node in nodes for output in node.output}
    for index in reversed(range(len(graph.node))):
        if not output_names.isdisjoint(graph.node[index].output):
            del graph.node[index]


def _path_after_marker(
    consumers, value_name: str, weight_name: str
) -> tuple[list[onnx.NodeProto], onnx.NodeProto]:
    path = []
    passthrough_order = {"Identity": 0, "Reshape": 1, "Cast": 1, "Transpose": 2}
    seen = set()
    previous_order = -1
    node = _single_consumer(consumers, value_name, weight_name)
    while node.op_type in passthrough_order:
        current_order = passthrough_order[node.op_type]
        if node.op_type in seen or current_order < previous_order:
            raise NotImplementedError(
                f"Unsupported Dynamo quantized weight topology for '{weight_name}': "
                f"unexpected {node.op_type} ordering."
            )
        seen.add(node.op_type)
        previous_order = current_order
        path.append(node)
        value_name = node.output[0]
        node = _single_consumer(consumers, value_name, weight_name)
    if node.op_type not in {"MatMul", "Gemm"} or len(node.input) < 2 or node.input[1] != value_name:
        raise NotImplementedError(
            f"Unsupported Dynamo quantized weight topology for '{weight_name}': "
            "expected terminal MatMul/Gemm at input 1."
        )
    return path, node


def _weight_source(graph, producers, consumers, marker: onnx.NodeProto):
    initializers = {item.name: item for item in graph.initializer}
    marker_input = marker.input[0]
    producer = producers.get(marker_input)
    reshape = (
        producer if isinstance(producer, onnx.NodeProto) and producer.op_type == "Reshape" else None
    )
    weight_name = reshape.input[0] if reshape is not None else marker_input
    if weight_name not in initializers:
        raise NotImplementedError(
            f"Unsupported Dynamo quantized weight '{marker_input}': "
            "data input must be a static initializer."
        )
    _single_consumer(consumers, weight_name, weight_name)
    if reshape is not None:
        _single_consumer(consumers, marker_input, weight_name)
        _constant_tensor(producers, reshape.input[1])
    return initializers[weight_name], reshape


def _is_static_value(producers, name: str, seen: set[str] | None = None) -> bool:
    producer = producers.get(name)
    if isinstance(producer, onnx.TensorProto):
        return True
    if not isinstance(producer, onnx.NodeProto):
        return False
    if producer.op_type == "Constant":
        return True
    inputs = [input_name for input_name in producer.input if input_name]
    if not inputs or name in (seen or set()):
        return False
    visited = {*seen, name} if seen is not None else {name}
    return all(_is_static_value(producers, input_name, visited) for input_name in inputs)


def _has_static_weight_source(producers, name: str) -> bool:
    producer = producers.get(name)
    if isinstance(producer, onnx.TensorProto):
        return True
    if (
        isinstance(producer, onnx.NodeProto)
        and producer.op_type == "Reshape"
        and isinstance(producers.get(producer.input[0]), onnx.TensorProto)
    ):
        return True
    # Shape and index inputs can be dynamic; only input 0 carries the weight data.
    data_name = name
    seen = set()
    while (
        isinstance(producer, onnx.NodeProto)
        and producer.op_type
        in {
            "Identity",
            "Reshape",
            "Cast",
            "Transpose",
            "Slice",
            "Gather",
            "GatherElements",
            "GatherND",
            "Expand",
            "Squeeze",
            "Unsqueeze",
            "Flatten",
        }
        and producer.input[0] not in seen
    ):
        data_name = producer.input[0]
        seen.add(data_name)
        producer = producers.get(data_name)
    if _is_static_value(producers, data_name):
        raise NotImplementedError(
            f"Unsupported Dynamo quantized weight '{name}': unsupported static or dynamic transformation "
            "before the quantization marker."
        )
    return False


def _validate_restoring_reshape(
    path, weight: onnx.TensorProto, producers, format_name: str
) -> None:
    reshape = next((node for node in path if node.op_type == "Reshape"), None)
    if reshape is None:
        return
    restored_shape = [
        int(dim) for dim in numpy_helper.to_array(_constant_tensor(producers, reshape.input[1]))
    ]
    if restored_shape != list(weight.dims):
        raise NotImplementedError(
            f"Unsupported Dynamo {format_name} weight '{weight.name}': restoring Reshape does "
            "not match the source initializer shape."
        )


def _fold_prefix_reshape(
    graph: onnx.GraphProto,
    marker: onnx.NodeProto,
    weight: onnx.TensorProto,
    reshape: onnx.NodeProto | None,
    producers,
) -> None:
    if reshape is None:
        return
    shape = [
        int(dim) for dim in numpy_helper.to_array(_constant_tensor(producers, reshape.input[1]))
    ]
    values = numpy_helper.to_array(weight).reshape(shape)
    _replace_initializer(graph, numpy_helper.from_array(values, weight.name))
    marker.input[0] = weight.name
    _remove_nodes(graph, [reshape])


def _split_scale(
    graph, producers, consumers, marker: onnx.NodeProto, *, materialize_initializer: bool
) -> None:
    if len(marker.input) < 2:
        return
    scale_name = marker.input[1]
    tensor = _constant_tensor(producers, scale_name)
    if (
        materialize_initializer
        and len(consumers.get(scale_name, [])) == 1
        and isinstance(producers.get(scale_name), onnx.TensorProto)
    ):
        return
    array = np.asarray(numpy_helper.to_array(tensor))
    unique_name = f"{scale_name}_{marker.output[0]}"
    marker.input[1] = unique_name
    scale_tensor = numpy_helper.from_array(array, unique_name)
    if materialize_initializer:
        _replace_initializer(graph, scale_tensor)
    else:
        marker_index = next(
            index for index, node in enumerate(graph.node) if node.output[0] == marker.output[0]
        )
        graph.node.insert(
            marker_index,
            onnx.helper.make_node(
                "Constant", [], [unique_name], name=f"{unique_name}_constant", value=scale_tensor
            ),
        )


def _remove_identity(graph, path: list[onnx.NodeProto], marker_output: str) -> list[onnx.NodeProto]:
    if not path or path[0].op_type != "Identity":
        return path
    identity = path.pop(0)
    replacement = identity.output[0]
    for node in graph.node:
        for index, input_name in enumerate(node.input):
            if input_name == replacement:
                node.input[index] = marker_output
    _remove_nodes(graph, [identity])
    return path


def _normalize_int4_path(
    graph: onnx.GraphProto,
    marker: onnx.NodeProto,
    weight: onnx.TensorProto,
    prefix_reshape: onnx.NodeProto | None,
    path: list[onnx.NodeProto],
    terminal: onnx.NodeProto,
    producers,
) -> None:
    original_shape = list(weight.dims)
    block_size = int(_attribute(marker, "block_size") or 0)
    if block_size <= 0 or not original_shape or original_shape[-1] % block_size:
        raise NotImplementedError(
            f"Unsupported Dynamo INT4 weight '{weight.name}': invalid block size {block_size}."
        )

    if prefix_reshape is not None:
        blocked_shape = [
            int(dim)
            for dim in numpy_helper.to_array(_constant_tensor(producers, prefix_reshape.input[1]))
        ]
        if not blocked_shape or blocked_shape[-1] != block_size:
            raise NotImplementedError(
                f"Unsupported Dynamo INT4 weight '{weight.name}': pre-Reshape must use "
                f"block size {block_size}."
            )
    else:
        blocked_shape = [-1, block_size]
    marker_rank = len(blocked_shape if prefix_reshape is not None else original_shape)
    marker_axis = _attribute(marker, "axis")
    if (1 if marker_axis is None else marker_axis) not in {-1, marker_rank - 1}:
        raise NotImplementedError(
            f"Unsupported Dynamo INT4 weight '{weight.name}': only last-axis blocks are supported."
        )
    _validate_restoring_reshape(path, weight, producers, "INT4")
    blocked_weight = numpy_helper.to_array(weight).reshape(blocked_shape)
    _replace_initializer(graph, numpy_helper.from_array(blocked_weight, weight.name))
    marker.input[0] = weight.name

    path = _remove_identity(graph, path, marker.output[0])
    reshape = next((node for node in path if node.op_type == "Reshape"), None)

    block_axis = len(original_shape) - 1
    marker.attribute.append(onnx.helper.make_attribute("_target_shape", original_shape))
    transpose = next((node for node in path if node.op_type == "Transpose"), None)
    if transpose is not None:
        perm = list(_attribute(transpose, "perm") or [])
        if sorted(perm) != list(range(len(original_shape))):
            raise NotImplementedError(
                f"Unsupported Dynamo INT4 weight '{weight.name}': invalid transpose permutation."
            )
        marker.attribute.append(onnx.helper.make_attribute("_transpose_perm", perm))
        block_axis = perm.index(block_axis)
    axis = next((attr for attr in marker.attribute if attr.name == "axis"), None)
    if axis is None:
        marker.attribute.append(onnx.helper.make_attribute("axis", block_axis))
    else:
        axis.i = block_axis

    cast = next((node for node in path if node.op_type == "Cast"), None)
    if cast is not None:
        cast.input[0] = marker.output[0]
        weight_output = cast.output[0]
    else:
        weight_output = marker.output[0]
    terminal.input[1] = weight_output
    _remove_nodes(
        graph,
        [node for node in (prefix_reshape, reshape, transpose) if node is not None],
    )

    scale = next(item for item in graph.initializer if item.name == marker.input[1])
    scale_values = numpy_helper.to_array(scale).reshape(-1, 1)
    _replace_initializer(graph, numpy_helper.from_array(scale_values, scale.name))


def _normalize_nvfp4_path(
    graph: onnx.GraphProto,
    marker: onnx.NodeProto,
    weight: onnx.TensorProto,
    prefix_reshape: onnx.NodeProto | None,
    path: list[onnx.NodeProto],
    terminal: onnx.NodeProto,
    producers,
) -> None:
    path = _remove_identity(graph, path, marker.output[0])
    removable = []
    prefix_nodes = []
    reshape = next((node for node in path if node.op_type == "Reshape"), None)
    if prefix_reshape is None and reshape is not None:
        raise NotImplementedError(
            f"Unsupported Dynamo NVFP4 weight '{weight.name}': restoring Reshape requires a "
            "pre-Reshape."
        )
    if prefix_reshape is not None:
        if reshape is None:
            raise NotImplementedError(
                f"Unsupported Dynamo NVFP4 weight '{weight.name}': missing restoring Reshape."
            )
        _validate_restoring_reshape(path, weight, producers, "NVFP4")
        marker.input[0] = weight.name
        prefix_nodes.append(prefix_reshape)
        removable.append(reshape)
        path.remove(reshape)
    cast = next((node for node in path if node.op_type == "Cast"), None)
    if cast is not None:
        removable.append(cast)
        path.remove(cast)

    for node in removable:
        output = node.output[0]
        for candidate in graph.node:
            for index, input_name in enumerate(candidate.input):
                if input_name == output:
                    candidate.input[index] = marker.output[0]
    _remove_nodes(graph, [*prefix_nodes, *removable])

    if path and path[0].op_type == "Transpose" and terminal.op_type == "Gemm":
        transpose = path[0]
        if list(_attribute(transpose, "perm") or []) != [1, 0]:
            raise NotImplementedError(
                f"Unsupported Dynamo NVFP4 weight '{weight.name}': only a rank-2 transpose is supported."
            )
        trans_b = next((item for item in terminal.attribute if item.name == "transB"), None)
        if trans_b is None:
            terminal.attribute.append(onnx.helper.make_attribute("transB", 1))
        else:
            trans_b.i = 1 - trans_b.i
        terminal.input[1] = marker.output[0]
        _remove_nodes(graph, [transpose])


def _validate_marker_domain(marker: onnx.NodeProto, expected_domain: str) -> None:
    if marker.domain != expected_domain:
        raise NotImplementedError(
            f"Unsupported Dynamo quantization marker '{marker.op_type}': unexpected domain "
            f"'{marker.domain}'."
        )


def normalize_dynamo_weight_paths(model: onnx.ModelProto) -> onnx.ModelProto:
    """Normalize supported Dynamo static-weight paths for the legacy format packers."""
    graph = model.graph
    producers = get_tensor_producer_nodes(graph, get_initializer_producers=True)
    consumers = get_tensor_consumer_nodes(graph)

    for marker in list(graph.node):
        kind = None
        marker_output = marker.output[0]
        if marker.op_type == "TRT_FP4QDQ":
            _validate_marker_domain(marker, "trt")
            kind = "nvfp4"
        elif marker.op_type == "TRT_MXFP8DequantizeLinear" and _has_static_weight_source(
            producers, marker.input[0]
        ):
            _validate_marker_domain(marker, "trt")
            kind = "mxfp8"
        elif marker.op_type == "DequantizeLinear" and (_attribute(marker, "block_size") or 0) > 0:
            producer = producers.get(marker.input[0])
            if (
                not isinstance(producer, onnx.NodeProto)
                or producer.op_type != "TRT_FP4DynamicQuantize"
            ):
                _validate_marker_domain(marker, "")
                kind = "int4"
        elif marker.op_type in {
            "TRT_FP8QuantizeLinear",
            "QuantizeLinear",
        } and _has_static_weight_source(producers, marker.input[0]):
            expected_domain = "trt" if marker.op_type == "TRT_FP8QuantizeLinear" else ""
            _validate_marker_domain(marker, expected_domain)
            dq = _single_consumer(consumers, marker.output[0], marker.input[0])
            expected_dq = (
                "TRT_FP8DequantizeLinear"
                if marker.op_type == "TRT_FP8QuantizeLinear"
                else "DequantizeLinear"
            )
            if dq.op_type != expected_dq or dq.domain != marker.domain:
                raise NotImplementedError(
                    f"Unsupported Dynamo quantized weight '{marker.input[0]}': expected a matching "
                    f"{expected_dq}."
                )
            if list(dq.input[1:]) != list(marker.input[1:]):
                raise NotImplementedError(
                    f"Unsupported Dynamo quantized weight '{marker.input[0]}': "
                    "Q/DQ scale and zero-point inputs must match."
                )
            kind = "fp8" if marker.op_type.startswith("TRT_") else "int8"
            marker_output = dq.output[0]
        if kind is None:
            continue

        if kind in {"fp8", "int8"}:
            direct_consumers = consumers.get(marker_output, [])
            if (
                len(direct_consumers) == 1
                and direct_consumers[0].op_type == "Conv"
                and len(direct_consumers[0].input) > 1
                and direct_consumers[0].input[1] == marker_output
            ):
                continue

        weight, prefix_reshape = _weight_source(graph, producers, consumers, marker)
        path, terminal = _path_after_marker(consumers, marker_output, weight.name)
        if kind in {"fp8", "int8", "mxfp8"}:
            _validate_restoring_reshape(path, weight, producers, kind.upper())
            _fold_prefix_reshape(graph, marker, weight, prefix_reshape, producers)
        if kind in {"int4", "mxfp8"}:
            _split_scale(
                graph,
                producers,
                consumers,
                marker,
                materialize_initializer=kind == "int4",
            )
        if kind == "int4":
            if not marker.name.startswith(_DYNAMO_INT4_NODE_PREFIX):
                marker.name = f"{_DYNAMO_INT4_NODE_PREFIX}{marker.name}"
            _normalize_int4_path(graph, marker, weight, prefix_reshape, path, terminal, producers)
        elif kind == "mxfp8" and not marker.name.startswith(_DYNAMO_MXFP8_NODE_PREFIX):
            marker.name = f"{_DYNAMO_MXFP8_NODE_PREFIX}{marker.name}"
        elif kind == "nvfp4":
            _normalize_nvfp4_path(graph, marker, weight, prefix_reshape, path, terminal, producers)

    return model
