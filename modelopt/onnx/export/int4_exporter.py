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

"""INT4 quantization exporter."""

import onnx
from onnx import numpy_helper

from modelopt.onnx.logging_config import logger
from modelopt.onnx.quantization.graph_utils import (
    get_tensor_consumer_nodes,
    get_tensor_producer_nodes,
)
from modelopt.onnx.quantization.qdq_utils import cast_initializer_to_dtype
from modelopt.onnx.quantization.quant_utils import pack_weights_to_int4

from .base_exporter import (
    ONNXQuantExporter,
    _materialize_initializer_input,
    _replace_initializer,
    _single_consumer,
)


def _optional_attribute(node: onnx.NodeProto, name: str):
    attr = next((attr for attr in node.attribute if attr.name == name), None)
    return onnx.helper.get_attribute_value(attr) if attr else None


def _constant_array(producers, name: str):
    producer = producers.get(name)
    if isinstance(producer, onnx.TensorProto):
        return numpy_helper.to_array(producer)
    value = (
        _optional_attribute(producer, "value")
        if producer and producer.op_type == "Constant"
        else None
    )
    if value is None:
        raise NotImplementedError("Dynamo ONNX export does not support dynamic weight shapes.")
    return numpy_helper.to_array(value)


def _weight_dq_nodes(graph: onnx.GraphProto) -> list[onnx.NodeProto]:
    initializers = {initializer.name for initializer in graph.initializer}
    producers = get_tensor_producer_nodes(graph)
    consumers = get_tensor_consumer_nodes(graph)
    result = []
    for node in graph.node:
        if node.op_type != "DequantizeLinear" or len(node.input) != 2:
            continue
        producer = producers.get(node.input[0])
        initializer_backed = node.input[0] in initializers or (
            producer is not None
            and producer.op_type == "Reshape"
            and producer.input[0] in initializers
        )
        marker = _optional_attribute(node, "block_size") is not None
        legacy = node.domain == "" and (
            _optional_attribute(node, "_target_shape") is not None
            or any(consumer.op_type == "Reshape" for consumer in consumers.get(node.output[0], []))
        )
        if initializer_backed and (marker or legacy):
            result.append(node)
        elif marker and (producer is None or producer.op_type != "TRT_FP4DynamicQuantize"):
            raise NotImplementedError(
                f"Unsupported Dynamo INT4 weight '{node.input[0]}': "
                "data input must be a static initializer."
            )
    return result


def _normalize_weight_paths(graph: onnx.GraphProto) -> list[onnx.NodeProto]:
    """Normalize the supported Dynamo weight prefix to the legacy INT4 shape."""
    initializers = {initializer.name: initializer for initializer in graph.initializer}
    producers = get_tensor_producer_nodes(graph, get_initializer_producers=True)
    consumers = get_tensor_consumer_nodes(graph)
    removed_outputs = set()

    for node in _weight_dq_nodes(graph):
        weight_name = node.input[0]
        producer = producers.get(weight_name)
        if isinstance(producer, onnx.NodeProto):
            source_name = producer.input[0]
            _single_consumer(consumers, source_name, source_name)
            _single_consumer(consumers, weight_name, source_name)
            blocked_shape = [int(dim) for dim in _constant_array(producers, producer.input[1])]
            weight = numpy_helper.to_array(initializers[source_name]).reshape(blocked_shape)
            _replace_initializer(graph, numpy_helper.from_array(weight, source_name))
            node.input[0] = source_name
            weight_name = source_name
            removed_outputs.update(producer.output)
        else:
            _single_consumer(consumers, weight_name, weight_name)
        if _optional_attribute(node, "block_size") is None:
            node.attribute.append(
                onnx.helper.make_attribute("block_size", initializers[weight_name].dims[-1])
            )

        scale_name = node.input[1]
        scale_producer = producers.get(scale_name)
        if scale_name not in initializers and (
            not isinstance(scale_producer, onnx.NodeProto) or scale_producer.op_type != "Constant"
        ):
            raise NotImplementedError(
                f"Unsupported Dynamo INT4 weight '{weight_name}': scale must be constant."
            )

    if removed_outputs:
        retained = [node for node in graph.node if removed_outputs.isdisjoint(node.output)]
        del graph.node[:]
        graph.node.extend(retained)
    return _weight_dq_nodes(graph)


class INT4QuantExporter(ONNXQuantExporter):
    """Exporter for INT4 quantization."""

    @staticmethod
    def pre_process(onnx_model: onnx.ModelProto) -> onnx.ModelProto:
        """Pre-processes the ONNX model for INT4 quantization."""
        graph = onnx_model.graph
        value_info_map = {value_info.name: value_info for value_info in graph.value_info}
        weight_dq_nodes = _normalize_weight_paths(graph)
        tensor_producer_map = get_tensor_producer_nodes(graph, get_initializer_producers=True)
        tensor_consumers = get_tensor_consumer_nodes(graph)

        outputs_to_remove = set()
        for node in weight_dq_nodes:
            weight_name = node.input[0]
            logger.debug(f"Restructuring graph for weight {weight_name}")

            next_node = _single_consumer(tensor_consumers, node.output[0], weight_name)
            weight_output = node.output[0]
            path_input = node.output[0]
            cast_node = None
            reshape_node = None
            target_shape_attr = next(
                (attr for attr in node.attribute if attr.name == "_target_shape"), None
            )
            weight_shape = (
                list(target_shape_attr.ints)
                if target_shape_attr is not None
                else list(next(item for item in graph.initializer if item.name == weight_name).dims)
            )
            for _ in range(2):
                if next_node.op_type == "Cast" and cast_node is None:
                    cast_node = next_node
                elif next_node.op_type == "Reshape" and reshape_node is None:
                    reshape_node = next_node
                    outputs_to_remove.update(reshape_node.output)
                    reshape_output = value_info_map.get(reshape_node.output[0])
                    weight_shape = (
                        [dim.dim_value for dim in reshape_output.type.tensor_type.shape.dim]
                        if reshape_output is not None
                        else [
                            int(dim)
                            for dim in _constant_array(tensor_producer_map, reshape_node.input[1])
                        ]
                    )
                    shape_producer = tensor_producer_map[reshape_node.input[1]]
                    if (
                        isinstance(shape_producer, onnx.NodeProto)
                        and len(tensor_consumers.get(reshape_node.input[1], [])) == 1
                    ):
                        outputs_to_remove.update(shape_producer.output)
                else:
                    break
                path_input = next_node.output[0]
                next_node = _single_consumer(tensor_consumers, path_input, weight_name)

            target_shape = onnx.helper.make_attribute("_target_shape", weight_shape)
            if target_shape_attr is None:
                node.attribute.append(target_shape)
            else:
                target_shape_attr.CopyFrom(target_shape)

            if cast_node is not None:
                value_info = value_info_map.get(node.output[0])
                source_dtype = (
                    value_info.type.tensor_type.elem_type
                    if value_info is not None
                    else onnx.TensorProto.FLOAT
                )
                if _optional_attribute(cast_node, "to") == source_dtype:
                    outputs_to_remove.update(cast_node.output)
                else:
                    cast_node.input[0] = node.output[0]
                    weight_output = cast_node.output[0]

            if next_node.op_type == "Transpose":
                transpose_node = next_node
                outputs_to_remove.update(transpose_node.output)
                path_input = transpose_node.output[0]
                perm = _optional_attribute(transpose_node, "perm")
                assert perm is not None, f"Permutation not found for {node.name}"

                node.attribute.append(onnx.helper.make_attribute("_transpose_perm", perm))

                matmul_node = _single_consumer(
                    tensor_consumers, transpose_node.output[0], weight_name
                )
            else:
                perm = None
                matmul_node = next_node

            if (
                matmul_node.op_type not in ["MatMul", "Gemm"]
                or len(matmul_node.input) < 2
                or matmul_node.input[1] != path_input
            ):
                raise NotImplementedError(
                    f"Unsupported Dynamo INT4 weight topology for '{weight_name}': "
                    "expected terminal MatMul/Gemm at input 1."
                )
            axis = len(weight_shape) - 1
            axis = perm.index(axis) if perm is not None else axis
            axis_attr = next((attr for attr in node.attribute if attr.name == "axis"), None)
            if axis_attr is None:
                node.attribute.append(onnx.helper.make_attribute("axis", axis))
            else:
                axis_attr.i = axis
            matmul_node.input[1] = weight_output

        # Remove transpose, reshape, and constant nodes
        new_nodes = [node for node in graph.node if outputs_to_remove.isdisjoint(node.output)]
        del graph.node[:]
        graph.node.extend(new_nodes)

        return onnx_model

    @staticmethod
    def compute_scales(onnx_model: onnx.ModelProto) -> onnx.ModelProto:
        """Computes the scales for the weights in the ONNX model for INT4 quantization."""
        graph = onnx_model.graph
        initializer_map = {initializer.name: initializer for initializer in graph.initializer}
        weight_dq_nodes = _weight_dq_nodes(graph)
        tensor_producer_map = get_tensor_producer_nodes(graph, get_initializer_producers=True)

        for node in weight_dq_nodes:
            weight_name = node.input[0]
            scale_name = node.input[1]
            logger.debug(f"Computing scales for weight {weight_name}")

            # Load weight and scale tensors
            weight = numpy_helper.to_array(initializer_map[weight_name])
            scale = _constant_array(tensor_producer_map, scale_name)

            # Dequantize weight
            weight = weight / scale
            block_size = _optional_attribute(node, "block_size") or weight.shape[-1]

            target_shape = _optional_attribute(node, "_target_shape")
            transpose_perm = _optional_attribute(node, "_transpose_perm")
            assert target_shape is not None, f"Target shape not found for {node.name}"

            # Reshape weights and scales
            weight = weight.reshape(target_shape)
            assert target_shape[-1] % block_size == 0, (
                f"Block size {block_size} is not divisible by {target_shape[-1]}"
            )
            scale_shape = [*target_shape[:-1], target_shape[-1] // block_size]
            scale = scale.reshape(scale_shape)

            # Transpose weights and scales if permutation was stored
            if transpose_perm is not None:
                weight = weight.transpose(transpose_perm)
                scale = scale.transpose(transpose_perm)

            if scale_name not in initializer_map:
                scale_name = scale_name.replace("Constant_output_0", "scale")
            _materialize_initializer_input(
                graph, node, 1, onnx.numpy_helper.from_array(scale, scale_name)
            )
            _replace_initializer(graph, numpy_helper.from_array(weight, weight_name))

            logger.debug(f"Computed scales for weight {weight_name} for INT4 quantization")

        # Clean up metadata attributes from DequantizeLinear nodes
        for node in weight_dq_nodes:
            attrs_to_keep = [attr for attr in node.attribute if not attr.name.startswith("_")]
            del node.attribute[:]
            node.attribute.extend(attrs_to_keep)

        return onnx_model

    @staticmethod
    def compress_weights(onnx_model: onnx.ModelProto) -> onnx.ModelProto:
        """Compresses the weights in the ONNX model for INT4 quantization."""
        graph = onnx_model.graph
        initializer_map = {initializer.name: initializer for initializer in graph.initializer}
        weight_dq_nodes = _weight_dq_nodes(graph)

        for node in weight_dq_nodes:
            weight_name = node.input[0]
            weight = numpy_helper.to_array(initializer_map[weight_name])
            weight_shape = weight.shape
            weights_int4_np = pack_weights_to_int4(weight)
            weights_int4_onnx = onnx.numpy_helper.from_array(weights_int4_np, weight_name)
            weights_int4_onnx.data_type = onnx.TensorProto.INT4
            weights_int4_onnx.dims[0] = weight_shape[0]
            _replace_initializer(graph, weights_int4_onnx)
            logger.debug(f"Converted {weight_name} to INT4 precision")

        return onnx_model

    @staticmethod
    def post_process(onnx_model: onnx.ModelProto) -> onnx.ModelProto:
        """Post-processes the ONNX model for INT4 quantization."""

        def is_pre_quant_scale_node(node: onnx.NodeProto) -> bool:
            has_pqs_input = any(input for input in node.input if "_pre_quant_scale" in input)
            return node.op_type == "Mul" and has_pqs_input

        graph = onnx_model.graph
        initializer_map = {initializer.name: initializer for initializer in graph.initializer}
        nodes_to_remove = []

        def is_fp32_cast(node: onnx.NodeProto) -> bool:
            return node.op_type == "Cast" and any(
                attr.name == "to" and attr.i == onnx.TensorProto.FLOAT for attr in node.attribute
            )

        # Remove Cast nodes after specific operators
        for node in graph.node:
            if node.op_type in ["Transpose", "Reshape", "Sqrt", "Add", "Gelu"]:
                child_nodes = [n for n in graph.node if node.output[0] in n.input]
                if len(child_nodes) == 1 and is_fp32_cast(child_nodes[0]):
                    cast_node = child_nodes[0]
                    node.output.clear()
                    node.output.extend(cast_node.output)
                    nodes_to_remove.append(cast_node.name)

        # Remove unnecessay Cast after Pre-quant scale
        for node in graph.node:
            if is_pre_quant_scale_node(node):
                pqs_child_nodes = [n for n in graph.node if node.output[0] in n.input]
                assert len(pqs_child_nodes) == 1, f"Expected exactly one child node for {node.name}"
                cast_node = pqs_child_nodes[0]
                if cast_node.op_type == "Cast":
                    node.output.clear()
                    node.output.extend(cast_node.output)
                    nodes_to_remove.append(cast_node.name)

        # Remove unnecessary casts
        new_nodes = [node for node in graph.node if node.name not in nodes_to_remove]
        del graph.node[:]
        graph.node.extend(new_nodes)

        # Cast bias to float16
        for node in graph.node:
            if node.op_type == "Add" and "proj/Add" in node.name:
                cast_initializer_to_dtype(node, "Half", initializer_map)

        # Cast pre quant scales of o_proj and down_proj to float16
        for node in graph.node:
            if node.op_type == "Mul" and (
                any(
                    x in node.name
                    for x in ("o_proj/input_quantizer/Mul", "down_proj/input_quantizer/Mul")
                )
            ):
                cast_initializer_to_dtype(node, "Half", initializer_map)

        return onnx_model
