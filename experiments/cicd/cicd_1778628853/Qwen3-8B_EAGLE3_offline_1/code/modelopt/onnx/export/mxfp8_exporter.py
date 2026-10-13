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
from modelopt.onnx.quantization.graph_utils import get_tensor_producer_nodes
from modelopt.onnx.quantization.qdq_utils import _cast_fp8, onnx_dtype_map
from modelopt.onnx.quantization.quant_utils import compute_e8m0, get_amax
from modelopt.onnx.utils import get_attribute, has_attribute

from .base_exporter import ONNXQuantExporter

E8_M0_BIAS = 127
DEFAULT_BLOCK_SIZE = 32
DEFAULT_QUANT_AXIS = -1


def _get_weight_dq_nodes(graph: onnx.GraphProto) -> list[onnx.NodeProto]:
    """Get weight DequantizeLinear nodes from the graph."""
    return [
        node
        for node in graph.node
        if node.op_type == "TRT_MXFP8DequantizeLinear"
        and any(".weight" in inp for inp in node.input)
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

            # Remove scale producer if it's a Constant node
            scale_name = node.input[1]
            scale_producer = tensor_producer_map[scale_name]
            if scale_producer.op_type == "Constant":
                graph.node.remove(scale_producer)

            # Create and add new scale tensor
            scale_name_new = scale_name.replace("Constant_output_0", "scale")
            scale_tensor = onnx.numpy_helper.from_array(se8m0, scale_name_new)
            graph.initializer.append(scale_tensor)
            node.input[1] = scale_name_new

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
