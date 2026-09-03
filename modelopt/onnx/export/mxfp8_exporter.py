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

"""MXFP8 quantization exporter."""

import numpy as np
import onnx
from onnx import numpy_helper

from modelopt.onnx.logging_config import logger
from modelopt.onnx.quantization.graph_utils import (
    get_tensor_consumer_nodes,
    get_tensor_producer_nodes,
)
from modelopt.onnx.quantization.qdq_utils import _cast_fp8, onnx_dtype_map
from modelopt.onnx.quantization.quant_utils import compute_e8m0, get_amax
from modelopt.onnx.utils import get_attribute, has_attribute

from .base_exporter import ONNXQuantExporter

E8_M0_BIAS = 127
DEFAULT_BLOCK_SIZE = 32
DEFAULT_QUANT_AXIS = -1


def _sync_initializer_metadata(graph: onnx.GraphProto, initializer: onnx.TensorProto) -> None:
    """Synchronize existing type and shape declarations for an initializer."""
    for value_info in (*graph.input, *graph.value_info, *graph.output):
        if value_info.name != initializer.name:
            continue
        tensor_type = value_info.type.tensor_type
        tensor_type.elem_type = initializer.data_type
        del tensor_type.shape.dim[:]
        for dim_value in initializer.dims:
            tensor_type.shape.dim.add().dim_value = dim_value


def _get_weight_dq_nodes(graph: onnx.GraphProto) -> list[onnx.NodeProto]:
    """Get weight DequantizeLinear nodes from the graph."""
    initializer_names = {initializer.name for initializer in graph.initializer}
    return [
        node
        for node in graph.node
        if node.op_type == "TRT_MXFP8DequantizeLinear" and node.input[0] in initializer_names
    ]


def _get_quant_params(node: onnx.NodeProto) -> tuple[int, int]:
    """Extract quantization axis and block size from a node."""
    if has_attribute(node, "axis"):
        quant_axis = int(get_attribute(node, "axis"))
    else:
        quant_axis = DEFAULT_QUANT_AXIS
        logger.warning(
            "axis attribute not found for MXFP8DequantizeLinear node. Setting axis to -1"
        )

    if has_attribute(node, "block_size"):
        block_size = int(get_attribute(node, "block_size"))
    else:
        block_size = DEFAULT_BLOCK_SIZE
        logger.warning(
            "block_size attribute not found for MXFP8DequantizeLinear node. "
            "Setting block_size to 32"
        )

    return quant_axis, block_size


class MXFP8QuantExporter(ONNXQuantExporter):
    """Exporter for MXFP8 quantization."""

    @staticmethod
    def pre_process(onnx_model: onnx.ModelProto) -> onnx.ModelProto:
        """Pre-processes the ONNX model for MXFP8 quantization."""
        graph = onnx_model.graph
        weight_dq_nodes = _get_weight_dq_nodes(graph)
        tensor_consumer_map = get_tensor_consumer_nodes(graph)
        tensor_producer_map = get_tensor_producer_nodes(graph)
        initializer_map = {initializer.name: initializer for initializer in graph.initializer}
        tensor_names = {
            name for node in graph.node for name in (*node.input, *node.output) if name
        } | set(initializer_map)
        initializer_candidates_to_remove = set()
        constant_outputs_to_remove = set()

        def _clone_shared_input(node: onnx.NodeProto, input_index: int, path_index: int):
            tensor_name = node.input[input_index]
            if len(tensor_consumer_map[tensor_name]) <= 1:
                return

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

            base_name = f"{tensor_name}_mxfp8_{path_index}"
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

        for path_index, node in enumerate(weight_dq_nodes):
            _clone_shared_input(node, 0, path_index)
            _clone_shared_input(node, 1, path_index)

        used_tensors = {input_name for node in graph.node for input_name in node.input}
        protected_tensors = used_tensors | {value.name for value in (*graph.input, *graph.output)}
        new_initializers = [
            initializer
            for initializer in graph.initializer
            if initializer.name not in initializer_candidates_to_remove
            or initializer.name in protected_tensors
        ]
        new_nodes = [
            node
            for node in graph.node
            if not (
                node.op_type == "Constant"
                and any(output in constant_outputs_to_remove for output in node.output)
                and not any(output in used_tensors for output in node.output)
            )
        ]
        del graph.initializer[:]
        graph.initializer.extend(new_initializers)
        del graph.node[:]
        graph.node.extend(new_nodes)
        return onnx_model

    @staticmethod
    def compute_scales(onnx_model: onnx.ModelProto) -> onnx.ModelProto:
        """Computes the e8m0 scales for weights in the ONNX model for MXFP8 quantization."""
        logger.info("Computing MXFP8 scales for weights")
        graph = onnx_model.graph
        initializer_map = {init.name: init for init in graph.initializer}
        tensor_producer_map = get_tensor_producer_nodes(graph)

        for node in _get_weight_dq_nodes(graph):
            weight_name = node.input[0]
            logger.debug(f"Computing MXFP8 scale for weight {weight_name}")

            weight = numpy_helper.to_array(initializer_map[weight_name])
            quant_axis, block_size = _get_quant_params(node)

            # Compute scales
            amax = get_amax(weight, quant_axis, block_size)
            se8m0_fp32 = compute_e8m0(amax, weight.shape, quant_axis, block_size)
            se8m0 = se8m0_fp32.astype(np.uint8)

            scale_name = node.input[1]
            if scale_name in initializer_map:
                scale_tensor = onnx.numpy_helper.from_array(se8m0, scale_name)
                initializer_map[scale_name].CopyFrom(scale_tensor)
            else:
                scale_producer = tensor_producer_map[scale_name]
                if scale_producer.op_type == "Constant":
                    graph.node.remove(scale_producer)

                scale_name_new = scale_name.replace("Constant_output_0", "scale")
                scale_tensor = onnx.numpy_helper.from_array(se8m0, scale_name_new)
                graph.initializer.append(scale_tensor)
                initializer_map[scale_name_new] = scale_tensor
                node.input[1] = scale_name_new
            _sync_initializer_metadata(graph, scale_tensor)

        return onnx_model

    @staticmethod
    def compress_weights(onnx_model: onnx.ModelProto) -> onnx.ModelProto:
        """Compresses the weights in the ONNX model to FP8 format for MXFP8 quantization."""
        logger.info("Compressing weights to MXFP8 format")
        graph = onnx_model.graph
        initializer_map = {init.name: init for init in graph.initializer}

        for node in _get_weight_dq_nodes(graph):
            weight_name = node.input[0]
            scale_name = node.input[1]
            logger.debug(f"Compressing weight {weight_name} to MXFP8")

            weight = numpy_helper.to_array(initializer_map[weight_name])
            quant_axis, block_size = _get_quant_params(node)

            # Get scale and convert back to fp32 for computation
            se8m0 = numpy_helper.to_array(initializer_map[scale_name])
            se8m0_fp32 = se8m0.astype(np.float32)

            # Expand block array so that it can be broadcasted with weight
            se8m0_fp32_expanded = np.repeat(se8m0_fp32, block_size, axis=quant_axis)
            scaled_weight = weight / np.exp2(se8m0_fp32_expanded - E8_M0_BIAS)

            # Create FP8 weight tensor
            weights_e4m3 = onnx.helper.make_tensor(
                name=weight_name,
                data_type=onnx_dtype_map["Float8"],
                dims=[*scaled_weight.shape],
                vals=_cast_fp8(scaled_weight).tobytes(),
                raw=True,
            )
            initializer_map[weight_name].CopyFrom(weights_e4m3)
            _sync_initializer_metadata(graph, initializer_map[weight_name])
            logger.debug(f"Converted {weight_name} to MXFP8")

        return onnx_model

    @staticmethod
    def post_process(onnx_model: onnx.ModelProto) -> onnx.ModelProto:
        """Post-processes the ONNX model for MXFP8 quantization.

        Sets DQ output type to FP16 and updates GELU nodes to use tanh approximation.
        """
        logger.info("Post-processing MXFP8 quantized model")
        graph = onnx_model.graph

        # Set output type of DQ to FP16
        for node in graph.node:
            if node.op_type == "TRT_MXFP8DequantizeLinear":
                for attr in node.attribute:
                    if attr.name == "output_dtype":
                        attr.i = onnx_dtype_map["Half"]

        # Currently only tanh approximation is supported for Gelu
        for node in graph.node:
            if node.op_type == "Gelu":
                for attr in node.attribute:
                    if attr.name == "approximate":
                        attr.s = b"tanh"
                        logger.debug(f"Updated GELU node {node.name} to use tanh approximation")

        # Insert cast to fp16 after Sqrt nodes
        cast_nodes_to_insert = []
        for idx, node in enumerate(graph.node):
            if node.op_type == "Sqrt":
                sqrt_output = node.output[0]
                cast_output = f"{sqrt_output}_cast_fp16"

                # Create Cast node
                cast_node = onnx.helper.make_node(
                    "Cast",
                    inputs=[sqrt_output],
                    outputs=[cast_output],
                    to=onnx_dtype_map["Half"],
                    name=f"{node.name}_cast_fp16",
                )
                cast_nodes_to_insert.append((idx + 1, cast_node))

                # Update consumers to use cast output
                for consumer in graph.node:
                    if consumer == node:
                        continue
                    for i, inp in enumerate(consumer.input):
                        if inp == sqrt_output:
                            consumer.input[i] = cast_output

        # Insert Cast nodes in reverse order to preserve indices
        for offset, (pos, cast_node) in enumerate(cast_nodes_to_insert):
            graph.node.insert(pos + offset, cast_node)
            logger.debug(f"Inserted Cast to FP16 after {cast_node.input[0]}")

        return onnx_model
