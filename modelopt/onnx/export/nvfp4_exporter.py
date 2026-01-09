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

"""NVFP4 quantization exporter."""

import numpy as np
import onnx
import torch
from onnx import numpy_helper

from modelopt.onnx import utils
from modelopt.onnx.logging_config import logger
from modelopt.onnx.quantization.graph_utils import get_tensor_consumer_nodes
from modelopt.onnx.quantization.qdq_utils import onnx_dtype_map
from modelopt.onnx.quantization.quant_utils import (
    get_weights_scaling_factor,
    get_weights_scaling_factor_2,
    quantize,
)
from modelopt.torch.quantization.qtensor import NVFP4QTensor

from .base_exporter import ONNXQuantExporter


def _cast_fp4(array: np.ndarray) -> np.ndarray:
    """Cast a numpy array to FLOAT4E2M1 using PyTorch.

    Note: The first dimension of the array must be divisible by 2
    as two FP4 values are packed into a single byte.
    """
    array_f32_t = torch.from_numpy(array)
    array_f32_t_shape = array_f32_t.shape
    assert array_f32_t_shape[0] % 2 == 0, "array_f32_t_shape[0] must be divisible by 2"
    array_f4_t_shape = (array_f32_t_shape[0] // 2, *array_f32_t_shape[1:])
    if torch.cuda.is_available():
        array_f32_t = array_f32_t.cuda()
    array_f4_t = NVFP4QTensor._cast_fp4(array_f32_t)
    array_f4_t = array_f4_t.flatten()
    array_f4_t_packed = (array_f4_t[::2] | (array_f4_t[1::2] << 4)).reshape(array_f4_t_shape)
    array_f4 = array_f4_t_packed.cpu().numpy().astype(np.uint8)
    return array_f4


def _cast_fp8(array: np.ndarray) -> np.ndarray:
    """Cast a numpy array to FLOAT8E4M3FN using PyTorch."""
    array_f32_t = torch.from_numpy(array)
    if torch.cuda.is_available():
        array_f32_t = array_f32_t.cuda()
    array_f8_t = array_f32_t.clamp(min=-448, max=448).to(torch.float8_e4m3fn).view(torch.uint8)
    array_f8 = array_f8_t.cpu().numpy().astype(np.uint8)
    return array_f8


def _replace_fp4qdq_with_2dq(
    graph: onnx.GraphProto,
    node: onnx.NodeProto,
    initializer_indices: dict[str, int],
    value_info_map: dict[str, onnx.ValueInfoProto],
    graph_inputs: set[str],
    w_f4: np.ndarray,
    sw_f32_per_tensor: np.ndarray,
    sw_f8_per_block: np.ndarray,
    block_size: int,
):
    """Replaces the given node in the ONNX graph with a subgraph consisting of two DequantizeLinear nodes.

    Args:
        graph: The ONNX graph containing the node to replace.
        node: The node to be replaced.
        initializer_indices: A dictionary mapping initializer names to their indices in the graph.
        value_info_map: A dictionary mapping value info names to their ValueInfoProto objects.
        graph_inputs: A set of graph input names.
        w_f4: NumPy array for w_f4.
        sw_f32_per_tensor: NumPy array for sw_f32_per_tensor.
        sw_f8_per_block: NumPy array for sw_f8_per_block.
        block_size: Block size used in block quantization.
    """

    def _add_initializer(initializer):
        if initializer.name not in initializer_indices:
            graph.initializer.append(initializer)

    def _add_input_value_info(graph, tensor_proto):
        assert tensor_proto.name not in graph_inputs, (
            f"{tensor_proto.name} already in graph inputs."
        )
        assert tensor_proto.name not in value_info_map, (
            f"{tensor_proto.name} already in value info."
        )

        value_info = onnx.helper.make_tensor_value_info(
            tensor_proto.name, tensor_proto.data_type, tensor_proto.dims
        )
        graph.input.append(value_info)

    # Remove the original node from the graph
    graph.node.remove(node)
    weight_name = node.input[0]

    # Generate unique names for the initializers
    w_f4_name = weight_name + "_f4"
    sw_f8_per_block_name = weight_name + "_f8_scale"
    sw_f32_per_tensor_name = sw_f8_per_block_name + "_f32_scale"

    # Create TensorProto for initializers
    w_f4_proto = onnx.helper.make_tensor(
        name=w_f4_name,
        data_type=onnx_dtype_map["Float4"],
        dims=[w_f4.shape[0] * 2, *w_f4.shape[1:]],
        vals=w_f4.tobytes(),
        raw=True,
    )
    sw_f32_per_tensor_proto = onnx.numpy_helper.from_array(
        sw_f32_per_tensor, sw_f32_per_tensor_name
    )
    sw_f8_per_block_proto = onnx.numpy_helper.from_array(sw_f8_per_block, sw_f8_per_block_name)
    sw_f8_per_block_proto = onnx.helper.make_tensor(
        name=sw_f8_per_block_name,
        data_type=onnx_dtype_map["Float8"],
        dims=[*sw_f8_per_block.shape],
        vals=sw_f8_per_block.tobytes(),
        raw=True,
    )

    # Add ValueInfo for the initializers if not present
    _add_input_value_info(graph, w_f4_proto)
    _add_input_value_info(graph, sw_f32_per_tensor_proto)
    _add_input_value_info(graph, sw_f8_per_block_proto)

    # Add the initializers to the graph
    _add_initializer(w_f4_proto)
    _add_initializer(sw_f32_per_tensor_proto)
    _add_initializer(sw_f8_per_block_proto)

    # Create DequantizeLinear_1 node: (sw_f8_per_block, sw_f32_per_tensor) -> sw_f32
    sw_f32_name = weight_name + "_f32_scale"
    dequant1 = onnx.helper.make_node(
        "DequantizeLinear",
        inputs=[sw_f8_per_block_proto.name, sw_f32_per_tensor_proto.name],
        outputs=[sw_f32_name],
        name=weight_name + "_DequantizeLinear",
    )

    # Create DequantizeLinear_2 node: (w_f4, sw_f32) -> w_32
    w32_name = node.output[0]
    dequant2 = onnx.helper.make_node(
        "DequantizeLinear",
        inputs=[w_f4_proto.name, sw_f32_name],
        outputs=[w32_name],
        name=weight_name + "_DequantizeLinear_1",
        axis=-1,
        block_size=block_size,
    )

    # Add value_info for sw_f32
    # Assuming sw_f16 has the same shape as sw_f8_per_block
    sw_f32_type_proto = onnx.helper.make_tensor_type_proto(
        elem_type=onnx_dtype_map["Float"], shape=sw_f8_per_block.shape
    )
    sw_f16_value_info = onnx.helper.make_value_info(name=sw_f32_name, type_proto=sw_f32_type_proto)
    graph.value_info.append(sw_f16_value_info)

    # Change the data type of w16 (output of 2nd DQ) to model weight precision type
    if w32_name in value_info_map:
        value_info_map[w32_name].type.tensor_type.elem_type = onnx_dtype_map["Float"]
    else:
        raise ValueError(f"ValueInfo for {w32_name} not found.")

    # Add the new nodes to the graph
    graph.node.extend([dequant1, dequant2])


class NVFP4QuantExporter(ONNXQuantExporter):
    """Exporter for NVFP4 quantization.

    Converts FP32/FP16 weights of an ONNX model to FP4 weights and scaling factors.
    TRT_FP4QDQ nodes will get removed from the weights and replaced with two DQ nodes
    with converted FP4 weights and scaling factors.
    """

    @staticmethod
    def pre_process(onnx_model: onnx.ModelProto) -> onnx.ModelProto:
        """Pre-processes the ONNX model for NVFP4 quantization.

        This is a no-op for NVFP4 quantization as no pre-processing is needed.
        """
        return onnx_model

    @staticmethod
    def compute_scales(onnx_model: onnx.ModelProto) -> onnx.ModelProto:
        """Computes the scales for the weights in the ONNX model for NVFP4 quantization.

        Stores computed scales as node attributes for use in compress_weights.
        """
        logger.info("Computing scales for NVFP4 quantization")
        graph = onnx_model.graph
        initializers = graph.initializer
        initializer_indices = {
            initializer.name: idx for idx, initializer in enumerate(graph.initializer)
        }

        fp4_qdq_nodes = [node for node in graph.node if node.op_type == "TRT_FP4QDQ"]
        logger.debug(f"Found {len(fp4_qdq_nodes)} FP4QDQ nodes to process")

        for node in fp4_qdq_nodes:
            idx = initializer_indices.get(node.input[0], None)
            assert idx is not None, f"Initializer for weight '{node.input[0]}' not found."

            tensor = initializers[idx]
            w32 = utils.read_f16_tensor_as_fp32(tensor)

            # Compute scales
            sw_f32_per_tensor = get_weights_scaling_factor_2(w32)
            block_size = node.attribute[0].i
            sw_f32_per_block = get_weights_scaling_factor(w32, block_size, sw_f32_per_tensor)

            logger.debug(f"Computed scales for weight {node.input[0]} with block size {block_size}")

            # Store scales as node attributes for use in compress_weights
            sw_per_tensor_attr = node.attribute.add()
            sw_per_tensor_attr.name = "_sw_f32_per_tensor"
            sw_per_tensor_attr.floats.extend(sw_f32_per_tensor.flatten().tolist())

            sw_per_block_attr = node.attribute.add()
            sw_per_block_attr.name = "_sw_f32_per_block"
            sw_per_block_attr.floats.extend(sw_f32_per_block.flatten().tolist())

            sw_per_block_shape_attr = node.attribute.add()
            sw_per_block_shape_attr.name = "_sw_f32_per_block_shape"
            sw_per_block_shape_attr.ints.extend(sw_f32_per_block.shape)

        return onnx_model

    @staticmethod
    def compress_weights(onnx_model: onnx.ModelProto) -> onnx.ModelProto:
        """Compresses the weights in the ONNX model for NVFP4 quantization.

        Converts weights to FP4 format and scales to FP8 format.
        """
        logger.info("Compressing weights for NVFP4 quantization")
        graph = onnx_model.graph
        initializers = graph.initializer
        initializer_indices = {
            initializer.name: idx for idx, initializer in enumerate(graph.initializer)
        }

        fp4_qdq_nodes = [node for node in graph.node if node.op_type == "TRT_FP4QDQ"]

        for node in fp4_qdq_nodes:
            idx = initializer_indices.get(node.input[0], None)
            assert idx is not None, f"Initializer for weight '{node.input[0]}' not found."

            tensor = initializers[idx]
            w32 = utils.read_f16_tensor_as_fp32(tensor)
            block_size = node.attribute[0].i

            # Retrieve scales from node attributes
            sw_f32_per_tensor = None
            sw_f32_per_block = None
            sw_per_block_shape = None

            for attr in node.attribute:
                if attr.name == "_sw_f32_per_tensor":
                    sw_f32_per_tensor = np.array(list(attr.floats), dtype=np.float32)
                elif attr.name == "_sw_f32_per_block":
                    sw_f32_per_block = np.array(list(attr.floats), dtype=np.float32)
                elif attr.name == "_sw_f32_per_block_shape":
                    sw_per_block_shape = tuple(attr.ints)

            assert sw_f32_per_tensor is not None, f"Scales not found for {node.input[0]}"
            assert sw_f32_per_block is not None, f"Block scales not found for {node.input[0]}"
            assert sw_per_block_shape is not None, (
                f"Block scale shape not found for {node.input[0]}"
            )

            sw_f32_per_block = sw_f32_per_block.reshape(sw_per_block_shape)

            # Quantize weights
            w_f32 = quantize(w32, block_size, sw_f32_per_block, sw_f32_per_tensor)

            # Cast to FP4 and FP8
            w_f4 = _cast_fp4(w_f32)
            sw_f8_per_block = _cast_fp8(sw_f32_per_block)

            # Store compressed data as node attributes for post_process
            w_f4_attr = node.attribute.add()
            w_f4_attr.name = "_w_f4"
            w_f4_attr.t.CopyFrom(numpy_helper.from_array(w_f4, "w_f4"))

            sw_f8_attr = node.attribute.add()
            sw_f8_attr.name = "_sw_f8_per_block"
            sw_f8_attr.t.CopyFrom(numpy_helper.from_array(sw_f8_per_block, "sw_f8"))

            logger.debug(f"Compressed weight {node.input[0]} to FP4")

        return onnx_model

    @staticmethod
    def post_process(onnx_model: onnx.ModelProto) -> onnx.ModelProto:
        """Post-processes the ONNX model for NVFP4 quantization.

        Replaces TRT_FP4QDQ nodes with two DequantizeLinear nodes and handles
        precision casting for inputs.
        """
        logger.info("Post-processing NVFP4 quantization")
        graph = onnx_model.graph
        initializers_to_delete = []
        tensor_consumers = get_tensor_consumer_nodes(graph)
        initializer_indices = {
            initializer.name: idx for idx, initializer in enumerate(graph.initializer)
        }
        value_info_map = {vi.name: vi for vi in graph.value_info}
        graph_inputs = {inp.name for inp in graph.input}

        def _get_precision_dtype() -> str:
            # Check initializers to determine the precision of the weights
            precision_dtype = "Half"
            for initializer in graph.initializer:
                if initializer.data_type == 16:
                    precision_dtype = "BFloat16"
                    break  # Assuming all weights are of the same precision
            return precision_dtype

        def _cast_input_dtypes(node: onnx.NodeProto, precision_dtype: str):
            # Change the input types to match weight precision (precision_dtype)
            if node.op_type == "Transpose":
                maybe_matmul = tensor_consumers[node.output[0]][0]
                assert maybe_matmul.op_type == "MatMul"
                node = maybe_matmul

            # Create Cast nodes for each input of the target node except bias
            for i, input_name in enumerate(node.input[:2]):
                cast_output_name = input_name + "_f16"  # Unique name for the cast output

                # Create a Cast node to convert the input to FP16/BF16
                cast_node = onnx.helper.make_node(
                    "Cast",
                    inputs=[input_name],  # Original input of the target node
                    outputs=[cast_output_name],
                    to=onnx_dtype_map[precision_dtype],  # Cast to FP16/BF16
                )

                # Insert the Cast node into the graph
                graph.node.extend([cast_node])

                # Update the target node input to use the cast node output
                node.input[i] = cast_output_name

        precision_dtype = _get_precision_dtype()
        logger.debug(f"Using precision dtype: {precision_dtype}")

        fp4_qdq_nodes = [node for node in graph.node if node.op_type == "TRT_FP4QDQ"]
        logger.debug(f"Found {len(fp4_qdq_nodes)} FP4QDQ nodes to convert")

        for node in fp4_qdq_nodes:
            idx = initializer_indices.get(node.input[0], None)
            assert idx is not None, f"Initializer for weight '{node.input[0]}' not found."
            initializers_to_delete.append(graph.initializer[idx].name)

            # Retrieve compressed data from node attributes
            block_size = node.attribute[0].i
            w_f4 = None
            sw_f8_per_block = None
            sw_f32_per_tensor = None

            for attr in node.attribute:
                if attr.name == "_w_f4":
                    w_f4 = numpy_helper.to_array(attr.t)
                elif attr.name == "_sw_f8_per_block":
                    sw_f8_per_block = numpy_helper.to_array(attr.t)
                elif attr.name == "_sw_f32_per_tensor":
                    sw_f32_per_tensor = np.array(list(attr.floats), dtype=np.float32)

            assert w_f4 is not None, f"Compressed weights not found for {node.input[0]}"
            assert sw_f8_per_block is not None, f"FP8 scales not found for {node.input[0]}"
            assert sw_f32_per_tensor is not None, f"Per-tensor scales not found for {node.input[0]}"

            logger.debug(f"Replacing FP4QDQ node for weight {node.input[0]} with 2 DQ nodes")

            _replace_fp4qdq_with_2dq(
                graph,
                node,
                initializer_indices,
                value_info_map,
                graph_inputs,
                w_f4,
                sw_f32_per_tensor,
                sw_f8_per_block,
                block_size,
            )

            # Cast input dtypes for the next node
            next_node = tensor_consumers[node.output[0]][0]
            _cast_input_dtypes(next_node, precision_dtype)

        # Remove old initializers
        new_initializers = [
            init for init in graph.initializer if init.name not in initializers_to_delete
        ]
        graph.ClearField("initializer")
        graph.initializer.extend(new_initializers)
        logger.info(f"Removed {len(initializers_to_delete)} initializers")

        return onnx_model
