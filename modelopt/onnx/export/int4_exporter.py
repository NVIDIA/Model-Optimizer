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

import math

import onnx
from onnx import numpy_helper

from modelopt.onnx.logging_config import logger
from modelopt.onnx.quantization.graph_utils import (
    get_tensor_consumer_nodes,
    get_tensor_producer_nodes,
)
from modelopt.onnx.quantization.qdq_utils import cast_initializer_to_dtype
from modelopt.onnx.quantization.quant_utils import pack_weights_to_int4

from .base_exporter import ONNXQuantExporter


def _get_weight_dq_nodes(graph: onnx.GraphProto) -> list[onnx.NodeProto]:
    initializer_names = {initializer.name for initializer in graph.initializer}
    tensor_producer_map = get_tensor_producer_nodes(graph)

    def _has_initializer_source(tensor_name: str) -> bool:
        if tensor_name in initializer_names:
            return True
        producer = tensor_producer_map.get(tensor_name)
        return (
            producer is not None
            and producer.op_type == "Reshape"
            and producer.input[0] in initializer_names
        )

    return [
        node
        for node in graph.node
        if node.op_type == "DequantizeLinear"
        and node.domain == "trt"
        and any(attr.name == "block_size" for attr in node.attribute)
        and _has_initializer_source(node.input[0])
    ]


class INT4QuantExporter(ONNXQuantExporter):
    """Exporter for INT4 quantization."""

    @staticmethod
    def pre_process(onnx_model: onnx.ModelProto) -> onnx.ModelProto:
        """Pre-processes the ONNX model for INT4 quantization."""
        graph = onnx_model.graph
        value_info_map = {
            value_info.name: value_info
            for value_info in (*graph.input, *graph.value_info, *graph.output)
        }
        weight_dq_nodes = _get_weight_dq_nodes(graph)
        tensor_producer_map = get_tensor_producer_nodes(graph)
        initializer_map = {initializer.name: initializer for initializer in graph.initializer}
        tensor_consumer_map = get_tensor_consumer_nodes(graph)
        node_outputs_to_remove = set()
        constant_outputs_to_remove = set()
        initializer_candidates_to_remove = set()
        tensor_names = {
            name for node in graph.node for name in (*node.input, *node.output) if name
        } | {initializer.name for initializer in graph.initializer}

        def _get_value_info_shape(tensor_name: str) -> list[int] | None:
            value_info = value_info_map.get(tensor_name)
            if value_info is None:
                return None
            dims = value_info.type.tensor_type.shape.dim
            if not all(dim.HasField("dim_value") for dim in dims):
                return None
            return [dim.dim_value for dim in dims]

        def _get_value_info_dtype(tensor_name: str) -> int | None:
            value_info = value_info_map.get(tensor_name)
            return None if value_info is None else value_info.type.tensor_type.elem_type

        def _get_constant_values(tensor_name: str) -> list[int] | None:
            if tensor_name in initializer_map:
                return numpy_helper.to_array(initializer_map[tensor_name]).reshape(-1).tolist()
            producer = tensor_producer_map.get(tensor_name)
            if producer is None or producer.op_type != "Constant":
                return None
            value = next((attr.t for attr in producer.attribute if attr.name == "value"), None)
            return None if value is None else numpy_helper.to_array(value).reshape(-1).tolist()

        def _get_reshape_output_shape(
            reshape_node: onnx.NodeProto, input_shape: list[int]
        ) -> list[int]:
            output_shape = _get_value_info_shape(reshape_node.output[0])
            if output_shape is not None:
                return output_shape

            requested_shape = _get_constant_values(reshape_node.input[1])
            if requested_shape is None:
                raise ValueError(f"Unable to determine shape for Reshape node {reshape_node.name}")

            allowzero = next(
                (attr.i for attr in reshape_node.attribute if attr.name == "allowzero"), 0
            )
            output_shape = []
            inferred_axis = None
            for axis, dim in enumerate(requested_shape):
                if dim == 0 and not allowzero:
                    dim = input_shape[axis]
                elif dim == -1:
                    if inferred_axis is not None:
                        raise ValueError(f"Multiple inferred dimensions in {reshape_node.name}")
                    inferred_axis = axis
                    dim = 1
                elif dim < 0:
                    raise ValueError(f"Invalid dimension {dim} in {reshape_node.name}")
                output_shape.append(dim)

            input_size = math.prod(input_shape)
            known_output_size = math.prod(output_shape)
            if inferred_axis is not None:
                if known_output_size == 0 or input_size % known_output_size:
                    raise ValueError(f"Invalid shape for Reshape node {reshape_node.name}")
                output_shape[inferred_axis] = input_size // known_output_size
            elif input_size != known_output_size:
                raise ValueError(f"Invalid shape for Reshape node {reshape_node.name}")
            return output_shape

        def _mark_shape_input_for_removal(reshape_node: onnx.NodeProto):
            shape_name = reshape_node.input[1]
            initializer_candidates_to_remove.add(shape_name)
            producer = tensor_producer_map.get(shape_name)
            if producer is not None and producer.op_type == "Constant":
                constant_outputs_to_remove.update(producer.output)

        def _get_only_child(tensor_name: str, parent_name: str) -> onnx.NodeProto:
            child_nodes = tensor_consumer_map.get(tensor_name, [])
            assert len(child_nodes) == 1, f"Expected exactly one child node for {parent_name}"
            return child_nodes[0]

        def _clone_shared_initializer_input(
            node: onnx.NodeProto, input_index: int, path_index: int
        ) -> str:
            tensor_name = node.input[input_index]
            if len(tensor_consumer_map[tensor_name]) <= 1:
                return tensor_name

            tensor = initializer_map.get(tensor_name)
            if tensor is None:
                producer = tensor_producer_map.get(tensor_name)
                if producer is None or producer.op_type != "Constant":
                    raise ValueError(f"Expected a constant shared input for {node.name}")
                tensor = next((attr.t for attr in producer.attribute if attr.name == "value"), None)
                if tensor is None:
                    raise ValueError(f"Expected a tensor value for {producer.name}")
                constant_outputs_to_remove.update(producer.output)
            else:
                initializer_candidates_to_remove.add(tensor_name)

            base_name = f"{tensor_name}_int4_{path_index}"
            unique_name = base_name
            suffix = 0
            while unique_name in tensor_names:
                suffix += 1
                unique_name = f"{base_name}_{suffix}"
            tensor_names.add(unique_name)

            cloned_tensor = onnx.TensorProto()
            cloned_tensor.CopyFrom(tensor)
            cloned_tensor.name = unique_name
            graph.initializer.append(cloned_tensor)
            initializer_map[unique_name] = cloned_tensor
            node.input[input_index] = unique_name
            return unique_name

        for path_index, node in enumerate(weight_dq_nodes):
            weight_name = node.input[0]
            logger.debug(f"Restructuring graph for weight {weight_name}")

            if weight_name in initializer_map:
                weight_shape = list(initializer_map[weight_name].dims)
            else:
                pre_reshape = tensor_producer_map.get(weight_name)
                if (
                    pre_reshape is None
                    or pre_reshape.op_type != "Reshape"
                    or pre_reshape.input[0] not in initializer_map
                ):
                    raise ValueError(
                        f"Expected an initializer or constant Reshape input for {node.name}"
                    )

                source_name = pre_reshape.input[0]
                source_initializer = initializer_map[source_name]
                weight_shape = list(source_initializer.dims)
                blocked_shape = _get_reshape_output_shape(pre_reshape, weight_shape)
                if math.prod(weight_shape) != math.prod(blocked_shape):
                    raise ValueError(f"Invalid blocked weight shape for {node.name}")

                blocked_initializer = onnx.TensorProto()
                blocked_initializer.CopyFrom(source_initializer)
                blocked_initializer.name = weight_name
                del blocked_initializer.dims[:]
                blocked_initializer.dims.extend(blocked_shape)
                graph.initializer.append(blocked_initializer)
                initializer_map[weight_name] = blocked_initializer
                weight_shape = blocked_shape

                node_outputs_to_remove.update(pre_reshape.output)
                initializer_candidates_to_remove.add(source_name)
                _mark_shape_input_for_removal(pre_reshape)

            weight_name = _clone_shared_initializer_input(node, 0, path_index)
            _clone_shared_initializer_input(node, 1, path_index)

            next_node = _get_only_child(node.output[0], node.name)
            path_dtype = _get_value_info_dtype(node.output[0])
            preserved_cast = None
            while next_node.op_type in {"Cast", "Reshape"}:
                if next_node.op_type == "Reshape":
                    weight_shape = _get_reshape_output_shape(next_node, weight_shape)
                    if path_dtype is None:
                        path_dtype = _get_value_info_dtype(next_node.output[0])
                    node_outputs_to_remove.update(next_node.output)
                    _mark_shape_input_for_removal(next_node)
                else:
                    cast_dtype = next(attr.i for attr in next_node.attribute if attr.name == "to")
                    if path_dtype == cast_dtype:
                        node_outputs_to_remove.update(next_node.output)
                    else:
                        assert preserved_cast is None, (
                            f"Expected at most one precision Cast node for {node.name}"
                        )
                        preserved_cast = next_node
                    path_dtype = cast_dtype
                next_node = _get_only_child(next_node.output[0], node.name)

            target_shape_attr = node.attribute.add()
            target_shape_attr.name = "_target_shape"
            target_shape_attr.ints.extend(weight_shape)

            # Store transpose permutation if present
            if next_node.op_type == "Transpose":
                transpose_node = next_node
                node_outputs_to_remove.update(transpose_node.output)
                perm = None
                for attr in transpose_node.attribute:
                    if attr.name == "perm":
                        perm = [x for x in attr.ints]  # noqa: C416
                assert perm is not None, f"Permutation not found for {node.name}"

                # Store permutation as attribute on DequantizeLinear node
                perm_attr = node.attribute.add()
                perm_attr.name = "_transpose_perm"
                perm_attr.ints.extend(perm)

                matmul_node = _get_only_child(transpose_node.output[0], node.name)
                quant_axis = perm.index(len(weight_shape) - 1)
                output_shape = [weight_shape[axis] for axis in perm]
            else:
                matmul_node = next_node
                quant_axis = len(weight_shape) - 1
                output_shape = weight_shape

            assert matmul_node.op_type in ["MatMul", "Gemm"], (
                f"Expected MatMul or Gemm node for {node.name}"
            )
            weight_output = node.output[0]
            if preserved_cast is not None:
                preserved_cast.input[0] = node.output[0]
                weight_output = preserved_cast.output[0]
            axis_attr = next((attr for attr in node.attribute if attr.name == "axis"), None)
            if axis_attr is None:
                axis_attr = node.attribute.add()
                axis_attr.name = "axis"
            axis_attr.i = quant_axis
            output_value_info = value_info_map.get(node.output[0])
            if output_value_info is None:
                output_value_info = onnx.helper.make_tensor_value_info(
                    node.output[0], initializer_map[weight_name].data_type, output_shape
                )
                graph.value_info.append(output_value_info)
                value_info_map[node.output[0]] = output_value_info
            output_dims = output_value_info.type.tensor_type.shape.dim
            del output_dims[:]
            for dim_value in output_shape:
                output_dims.add().dim_value = dim_value
            cast_output_value_info = value_info_map.get(weight_output)
            if cast_output_value_info is not None:
                output_dims = cast_output_value_info.type.tensor_type.shape.dim
                del output_dims[:]
                for dim_value in output_shape:
                    output_dims.add().dim_value = dim_value
            # Rewire MatMul to use the normalized weight output.
            matmul_node.input[1] = weight_output

        new_nodes = [
            node
            for node in graph.node
            if not any(output in node_outputs_to_remove for output in node.output)
        ]
        used_tensors = {input_name for node in new_nodes for input_name in node.input}
        new_nodes = [
            node
            for node in new_nodes
            if not (
                node.op_type == "Constant"
                and any(output in constant_outputs_to_remove for output in node.output)
                and not any(output in used_tensors for output in node.output)
            )
        ]
        used_tensors = {input_name for node in new_nodes for input_name in node.input}
        protected_tensors = used_tensors | {value.name for value in (*graph.input, *graph.output)}
        new_initializers = [
            initializer
            for initializer in graph.initializer
            if initializer.name not in initializer_candidates_to_remove
            or initializer.name in protected_tensors
        ]
        del graph.node[:]
        graph.node.extend(new_nodes)
        del graph.initializer[:]
        graph.initializer.extend(new_initializers)

        return onnx_model

    @staticmethod
    def compute_scales(onnx_model: onnx.ModelProto) -> onnx.ModelProto:
        """Computes the scales for the weights in the ONNX model for INT4 quantization."""
        graph = onnx_model.graph
        initializer_map = {initializer.name: initializer for initializer in graph.initializer}
        value_info_map = {
            value_info.name: value_info
            for value_info in (*graph.input, *graph.value_info, *graph.output)
        }
        weight_dq_nodes = _get_weight_dq_nodes(graph)
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

            scale_value_info = value_info_map.get(scale_name)
            if scale_value_info is not None:
                tensor_type = scale_value_info.type.tensor_type
                tensor_type.elem_type = scale_tensor.data_type
                del tensor_type.shape.dim[:]
                for dim_value in scale_tensor.dims:
                    tensor_type.shape.dim.add().dim_value = dim_value

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
        value_info_map = {
            value_info.name: value_info
            for value_info in (*graph.input, *graph.value_info, *graph.output)
        }
        weight_dq_nodes = _get_weight_dq_nodes(graph)

        for node in weight_dq_nodes:
            weight_name = node.input[0]
            weight = numpy_helper.to_array(initializer_map[weight_name])
            weight_shape = weight.shape
            weights_int4_np = pack_weights_to_int4(weight)
            weights_int4_onnx = onnx.numpy_helper.from_array(weights_int4_np, weight_name)
            weights_int4_onnx.data_type = onnx.TensorProto.INT4
            weights_int4_onnx.dims[0] = weight_shape[0]
            initializer_map[weight_name].CopyFrom(weights_int4_onnx)
            if weight_name in value_info_map:
                tensor_type = value_info_map[weight_name].type.tensor_type
                tensor_type.elem_type = onnx.TensorProto.INT4
                del tensor_type.shape.dim[:]
                for dim_value in weight_shape:
                    tensor_type.shape.dim.add().dim_value = dim_value
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
                if len(pqs_child_nodes) == 1 and pqs_child_nodes[0].op_type == "Cast":
                    cast_node = pqs_child_nodes[0]
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
