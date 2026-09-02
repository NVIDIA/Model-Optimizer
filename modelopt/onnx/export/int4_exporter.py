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
from modelopt.onnx.quantization.graph_utils import get_tensor_producer_nodes
from modelopt.onnx.quantization.qdq_utils import cast_initializer_to_dtype
from modelopt.onnx.quantization.quant_utils import pack_weights_to_int4

from .base_exporter import ONNXQuantExporter


class INT4QuantExporter(ONNXQuantExporter):
    """Exporter for INT4 quantization."""

    @staticmethod
    def pre_process(onnx_model: onnx.ModelProto) -> onnx.ModelProto:
        """Pre-processes the ONNX model for INT4 quantization."""
        graph = onnx_model.graph
        value_info_map = {value_info.name: value_info for value_info in graph.value_info}
        weight_dq_nodes = [node for node in graph.node if node.op_type == "DequantizeLinear"]
        tensor_producer_map = get_tensor_producer_nodes(graph, get_initializer_producers=True)

        nodes_to_remove = []
        for node in weight_dq_nodes:
            weight_name = node.input[0]
            logger.debug(f"Restructuring graph for weight {weight_name}")

            ## Convert DequantizeLinear -> Reshape -> Transpose -> MatMul/Gemm to DequantizeLinear -> Matmul/Gemm
            dq_child_nodes = [n for n in graph.node if node.output[0] in n.input]
            reshape_node = dq_child_nodes[0]
            nodes_to_remove.append(reshape_node.name)
            assert reshape_node.op_type == "Reshape", f"Expected Reshape node for {node.name}"
            reshape_node_output = reshape_node.output[0]

            # Remove constant node from reshape node
            shape_constant_name = next(input for input in reshape_node.input if "Constant" in input)
            nodes_to_remove.append(tensor_producer_map[shape_constant_name].name)

            # Get the shape of the output of the reshape node - store for compute_scales
            reshape_output_value_info = value_info_map.get(reshape_node_output)
            if reshape_output_value_info is not None:
                weight_shape = [
                    dim.dim_value for dim in reshape_output_value_info.type.tensor_type.shape.dim
                ]
            else:
                raise ValueError(f"Unable to determine shape of weight tensor {weight_name}")

            # Store target shape as attribute on DequantizeLinear node
            target_shape_attr = node.attribute.add()
            target_shape_attr.name = "_target_shape"
            target_shape_attr.ints.extend(weight_shape)

            reshape_child_nodes = [n for n in graph.node if reshape_node.output[0] in n.input]
            assert len(reshape_child_nodes) == 1, f"Expected exactly one child node for {node.name}"

            # Check if there's an optional Cast node between Reshape and Transpose/MatMul/Gemm
            next_node = reshape_child_nodes[0]
            if next_node.op_type == "Cast":
                # Remove unnecessary Cast node
                cast_node = next_node
                nodes_to_remove.append(cast_node.name)
                cast_child_nodes = [n for n in graph.node if cast_node.output[0] in n.input]
                next_node = cast_child_nodes[0]

            # Store transpose permutation if present
            if next_node.op_type == "Transpose":
                transpose_node = next_node
                nodes_to_remove.append(transpose_node.name)
                assert transpose_node.op_type == "Transpose", (
                    f"Expected Transpose node for {node.name}"
                )
                perm = None
                for attr in transpose_node.attribute:
                    if attr.name == "perm":
                        perm = [x for x in attr.ints]  # noqa: C416
                assert perm is not None, f"Permutation not found for {node.name}"

                # Store permutation as attribute on DequantizeLinear node
                perm_attr = node.attribute.add()
                perm_attr.name = "_transpose_perm"
                perm_attr.ints.extend(perm)

                transpose_child_nodes = [
                    n for n in graph.node if transpose_node.output[0] in n.input
                ]
                assert len(transpose_child_nodes) == 1, (
                    f"Expected exactly one matmul node for {node.name}"
                )
                matmul_node = transpose_child_nodes[0]
            else:
                matmul_node = next_node

            assert matmul_node.op_type in ["MatMul", "Gemm"], (
                f"Expected MatMul or Gemm node for {node.name}"
            )
            # Rewire MatMul to use DequantizeLinear output directly
            matmul_node.input[1] = node.output[0]

        # Remove transpose, reshape, and constant nodes
        new_nodes = [node for node in graph.node if node.name not in nodes_to_remove]
        del graph.node[:]
        graph.node.extend(new_nodes)

        return onnx_model

    @staticmethod
    def compute_scales(onnx_model: onnx.ModelProto) -> onnx.ModelProto:
        """Computes the scales for the weights in the ONNX model for INT4 quantization."""
        graph = onnx_model.graph
        initializer_map = {initializer.name: initializer for initializer in graph.initializer}
        weight_dq_nodes = [node for node in graph.node if node.op_type == "DequantizeLinear"]
        tensor_producer_map = get_tensor_producer_nodes(graph, get_initializer_producers=True)

        for node in weight_dq_nodes:
            weight_name = node.input[0]
            scale_name = node.input[1]
            logger.debug(f"Computing scales for weight {weight_name}")

            # Load weight and scale tensors
            weight = numpy_helper.to_array(initializer_map[weight_name])
            if scale_name in initializer_map:
                scale = numpy_helper.to_array(initializer_map[scale_name])
            else:
                scale_constant_node = tensor_producer_map[scale_name]
                for attr in scale_constant_node.attribute:
                    if attr.name == "value":
                        tensor = attr.t
                        scale = numpy_helper.to_array(tensor)

            # Dequantize weight
            weight = weight / scale
            block_size = weight.shape[-1]

            # Get target shape from metadata stored in pre_process
            target_shape = None
            transpose_perm = None
            for attr in node.attribute:
                if attr.name == "_target_shape":
                    target_shape = list(attr.ints)
                elif attr.name == "_transpose_perm":
                    transpose_perm = list(attr.ints)

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

            # Handle scale tensor creation/update
            if scale_name not in initializer_map:
                # Remove scale producer if it's a Constant node
                scale_producer = tensor_producer_map[scale_name]
                if scale_producer.op_type == "Constant":
                    graph.node.remove(scale_producer)

                # Create a new scale tensor
                scale_name = scale_name.replace("Constant_output_0", "scale")
                scale_tensor = onnx.numpy_helper.from_array(scale, scale_name)
                graph.initializer.append(scale_tensor)
                node.input[1] = scale_name
            else:
                scale_tensor = onnx.numpy_helper.from_array(scale, scale_name)
                initializer_map[scale_name].CopyFrom(scale_tensor)

            # Update weight tensor
            weight_tensor = numpy_helper.from_array(weight, weight_name)
            initializer_map[weight_name].CopyFrom(weight_tensor)

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
        weight_dq_nodes = [node for node in graph.node if node.op_type == "DequantizeLinear"]

        for node in weight_dq_nodes:
            weight_name = node.input[0]
            weight = numpy_helper.to_array(initializer_map[weight_name])
            weight_shape = weight.shape
            weights_int4_np = pack_weights_to_int4(weight)
            weights_int4_onnx = onnx.numpy_helper.from_array(weights_int4_np, weight_name)
            weights_int4_onnx.data_type = onnx.TensorProto.INT4
            weights_int4_onnx.dims[0] = weight_shape[0]
            initializer_map[weight_name].CopyFrom(weights_int4_onnx)
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
                assert cast_node.op_type == "Cast", f"Expected Cast node for {node.name}"
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
