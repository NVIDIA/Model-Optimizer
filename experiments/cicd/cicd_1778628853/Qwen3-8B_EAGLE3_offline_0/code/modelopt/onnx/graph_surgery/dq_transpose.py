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

"""Transpose DequantizeLinear weights for column-major storage optimization.

This module provides functionality to transform quantized ONNX models by
transposing the weights and scales in DequantizeLinear nodes and adding
corresponding Transpose nodes after them. This is useful for optimizing
inference with backends that prefer column-major weight storage (e.g., NvTensorRtRtx).

The transformation:
1. For each DequantizeLinear node feeding into MatMul/Gemm:
   - Transpose the quantized weights (input 0)
   - Transpose the scales (input 1)
   - Transpose zero points if present (input 2)
   - Update the axis attribute (0 -> 1 for 2D tensors)
   - Add a Transpose node after DequantizeLinear to recover original shape
"""

import os

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from ..logging_config import logger


def _is_int4_type(data_type: int) -> bool:
    """Check if the data type is INT4 or UINT4."""
    return data_type in [TensorProto.INT4, TensorProto.UINT4]


def _unpack_int4(packed_data: bytes, shape: tuple, signed: bool = True) -> np.ndarray:
    """Unpack INT4/UINT4 packed bytes to int8 array.

    INT4 is packed with 2 values per byte (low nibble first).
    """
    packed_arr = np.frombuffer(packed_data, dtype=np.uint8)

    # Extract low and high nibbles
    low_nibbles = packed_arr & 0x0F
    high_nibbles = (packed_arr >> 4) & 0x0F

    # Interleave: low nibble comes first
    unpacked = np.empty(len(packed_arr) * 2, dtype=np.int8)
    unpacked[0::2] = low_nibbles
    unpacked[1::2] = high_nibbles

    # Handle sign extension for INT4
    if signed:
        # Values >= 8 are negative (two's complement)
        unpacked = np.where(unpacked >= 8, unpacked - 16, unpacked).astype(np.int8)

    # Reshape to original shape
    total_elements = np.prod(shape)
    return unpacked[:total_elements].reshape(shape)


def _pack_int4(arr: np.ndarray, signed: bool = True) -> bytes:
    """Pack int8 array back to INT4/UINT4 packed bytes.

    INT4 is packed with 2 values per byte (low nibble first).
    """
    flat = arr.flatten().astype(np.int8)

    # Handle negative values for signed INT4
    if signed:
        flat = np.where(flat < 0, flat + 16, flat).astype(np.uint8)
    else:
        flat = flat.astype(np.uint8)

    # Ensure we have even number of elements (pad if needed)
    if len(flat) % 2 != 0:
        flat = np.append(flat, np.uint8(0))

    # Pack: low nibble first, then high nibble
    low_nibbles = flat[0::2] & 0x0F
    high_nibbles = flat[1::2] & 0x0F
    packed = low_nibbles | (high_nibbles << 4)

    return packed.astype(np.uint8).tobytes()


def _transpose_tensor_proto(tensor: onnx.TensorProto, perm: list[int]) -> onnx.TensorProto:
    """Transpose an ONNX TensorProto, handling INT4 packed format."""
    original_shape = list(tensor.dims)
    data_type = tensor.data_type

    if _is_int4_type(data_type):
        # Handle INT4/UINT4 specially
        signed = data_type == TensorProto.INT4
        unpacked = _unpack_int4(tensor.raw_data, tuple(original_shape), signed=signed)
        transposed = np.transpose(unpacked, perm)
        packed_data = _pack_int4(transposed, signed=signed)

        new_shape = [original_shape[p] for p in perm]

        new_tensor = onnx.TensorProto()
        new_tensor.name = tensor.name + "_transposed"
        new_tensor.data_type = data_type
        new_tensor.dims.extend(new_shape)
        new_tensor.raw_data = packed_data
        return new_tensor
    else:
        # Standard handling for other types
        arr = numpy_helper.to_array(tensor)
        transposed = np.transpose(arr, perm)
        new_tensor = numpy_helper.from_array(transposed, name=tensor.name + "_transposed")
        return new_tensor


def _find_initializer(model: onnx.ModelProto, name: str) -> onnx.TensorProto | None:
    """Find initializer by name."""
    for init in model.graph.initializer:
        if init.name == name:
            return init
    return None


def _find_node_by_output(model: onnx.ModelProto, output_name: str) -> onnx.NodeProto | None:
    """Find node that produces the given output."""
    for node in model.graph.node:
        if output_name in node.output:
            return node
    return None


def _get_consumers(model: onnx.ModelProto, tensor_name: str) -> list[onnx.NodeProto]:
    """Find all nodes that consume the given tensor."""
    return [node for node in model.graph.node if tensor_name in node.input]


def transpose_dequantize_linear_weights(
    model_path: str,
    output_path: str,
    use_external_data: bool = True,
    external_data_name: str | None = None,
    verbose: bool = True,
) -> onnx.ModelProto:
    """Transpose weights in DequantizeLinear nodes for column-major storage.

    This function transforms a quantized ONNX model by:
    1. Finding all DequantizeLinear nodes that feed into MatMul/Gemm
    2. Transposing the quantized weights, scales, and zero points
    3. Updating the axis attribute (0 -> 1)
    4. Adding Transpose nodes after DequantizeLinear to recover original shape

    This optimization is useful for backends that prefer column-major weight
    storage, such as NvTensorRtRtx.

    Args:
        model_path: Path to input quantized ONNX model.
        output_path: Path to save modified model.
        use_external_data: Whether to save weights as external data.
        external_data_name: Name for external data file.
        verbose: Whether to print progress messages.

    Returns:
        Modified ONNX model.

    Example:
        >>> from modelopt.onnx.graph_surgery import transpose_dequantize_linear_weights
        >>> model = transpose_dequantize_linear_weights(
        ...     model_path="model_quantized.onnx",
        ...     output_path="model_quantized_transposed.onnx",
        ... )
    """
    if verbose:
        logger.info(f"Loading model from: {model_path}")

    model = onnx.load(model_path, load_external_data=True)
    graph = model.graph

    # Statistics
    stats = {
        "dq_nodes_processed": 0,
        "transpose_nodes_added": 0,
        "weights_transposed": 0,
        "scales_transposed": 0,
        "zero_points_transposed": 0,
    }

    # Find all DequantizeLinear nodes
    dq_nodes = [node for node in graph.node if node.op_type == "DequantizeLinear"]

    if verbose:
        logger.info(f"Found {len(dq_nodes)} DequantizeLinear nodes")

    # Track which DQ nodes feed into MatMul/Gemm as weight input
    dq_nodes_to_process = []
    for dq_node in dq_nodes:
        if len(dq_node.output) == 0:
            continue

        dq_output = dq_node.output[0]
        consumers = _get_consumers(model, dq_output)

        for consumer in consumers:
            if consumer.op_type in ["MatMul", "Gemm"]:
                # Check if DQ output is the weight input (input[1] for MatMul/Gemm)
                if len(consumer.input) > 1 and consumer.input[1] == dq_output:
                    dq_nodes_to_process.append((dq_node, consumer))
                    break

    if verbose:
        logger.info(
            f"Found {len(dq_nodes_to_process)} DequantizeLinear nodes feeding into MatMul/Gemm"
        )

    # Track initializers to add/remove
    initializers_to_remove = []
    initializers_to_add = []
    nodes_to_add = []
    processed_dq_names = set()

    for dq_node, consumer_node in dq_nodes_to_process:
        if dq_node.name in processed_dq_names:
            continue
        processed_dq_names.add(dq_node.name)

        if len(dq_node.input) < 2:
            if verbose:
                logger.warning(f"Skipping {dq_node.name}: insufficient inputs")
            continue

        weight_name = dq_node.input[0]
        scale_name = dq_node.input[1]

        # Find initializers
        weight_init = _find_initializer(model, weight_name)
        scale_init = _find_initializer(model, scale_name)

        if weight_init is None or scale_init is None:
            if verbose:
                logger.debug(f"Skipping {dq_node.name}: weights or scale not constant")
            continue

        # Check if 2D
        if len(weight_init.dims) != 2:
            if verbose:
                logger.debug(
                    f"Skipping {dq_node.name}: weights not 2D (shape: {list(weight_init.dims)})"
                )
            continue

        original_shape = list(weight_init.dims)

        if verbose:
            is_int4 = _is_int4_type(weight_init.data_type)
            logger.debug(f"Processing {dq_node.name}: shape={original_shape}, INT4={is_int4}")

        # Transpose weights
        transposed_weight = _transpose_tensor_proto(weight_init, [1, 0])
        initializers_to_remove.append(weight_init)
        initializers_to_add.append(transposed_weight)

        # Update DQ node input to use transposed weight
        for i, inp in enumerate(dq_node.input):
            if inp == weight_name:
                dq_node.input[i] = transposed_weight.name
                break
        stats["weights_transposed"] += 1

        # Transpose scale if 2D
        if len(scale_init.dims) == 2:
            transposed_scale = _transpose_tensor_proto(scale_init, [1, 0])
            initializers_to_remove.append(scale_init)
            initializers_to_add.append(transposed_scale)

            for i, inp in enumerate(dq_node.input):
                if inp == scale_name:
                    dq_node.input[i] = transposed_scale.name
                    break
            stats["scales_transposed"] += 1

        # Transpose zero point if present and 2D
        if len(dq_node.input) > 2:
            zp_name = dq_node.input[2]
            zp_init = _find_initializer(model, zp_name)
            if zp_init is not None and len(zp_init.dims) == 2:
                transposed_zp = _transpose_tensor_proto(zp_init, [1, 0])
                initializers_to_remove.append(zp_init)
                initializers_to_add.append(transposed_zp)

                for i, inp in enumerate(dq_node.input):
                    if inp == zp_name:
                        dq_node.input[i] = transposed_zp.name
                        break
                stats["zero_points_transposed"] += 1

        # Update axis attribute (0 -> 1 after transpose)
        for attr in dq_node.attribute:
            if attr.name == "axis":
                old_axis = attr.i
                if old_axis == 0:
                    attr.i = 1
                elif old_axis == 1:
                    attr.i = 0
                if verbose:
                    logger.debug(f"  Updated axis: {old_axis} -> {attr.i}")
                break

        # Create intermediate output for DQ (transposed shape)
        dq_output_name = dq_node.output[0]
        new_dq_output_name = f"{dq_output_name}_before_transpose"

        # Update DQ node output
        dq_node.output[0] = new_dq_output_name

        # Create Transpose node to convert back to original shape
        transpose_node = helper.make_node(
            "Transpose",
            inputs=[new_dq_output_name],
            outputs=[dq_output_name],  # Use original output name
            name=f"{dq_node.name}_transpose_back",
            perm=[1, 0],
        )
        nodes_to_add.append(transpose_node)
        stats["transpose_nodes_added"] += 1
        stats["dq_nodes_processed"] += 1

        if verbose:
            transposed_shape = [original_shape[1], original_shape[0]]
            logger.debug(f"  Transposed: {original_shape} -> {transposed_shape}")

    # Apply changes to graph
    # Remove old initializers
    for init in initializers_to_remove:
        graph.initializer.remove(init)

    # Add new initializers
    graph.initializer.extend(initializers_to_add)

    # Add transpose nodes
    graph.node.extend(nodes_to_add)

    if verbose:
        logger.info("\nTransformation statistics:")
        logger.info(f"  DequantizeLinear nodes processed: {stats['dq_nodes_processed']}")
        logger.info(f"  Transpose nodes added: {stats['transpose_nodes_added']}")
        logger.info(f"  Weights transposed: {stats['weights_transposed']}")
        logger.info(f"  Scales transposed: {stats['scales_transposed']}")
        logger.info(f"  Zero points transposed: {stats['zero_points_transposed']}")

    # Save model
    if verbose:
        logger.info(f"\nSaving modified model to: {output_path}")

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    if use_external_data:
        if external_data_name is None:
            external_data_name = os.path.basename(output_path) + "_data"

        if verbose:
            logger.info(f"  Saving weights to external file: {external_data_name}")

        onnx.save(
            model,
            output_path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=external_data_name,
            size_threshold=1024,
        )
    else:
        onnx.save(model, output_path)

    if verbose:
        logger.info("Done!")

    return model
