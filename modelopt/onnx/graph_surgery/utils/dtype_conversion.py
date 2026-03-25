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

"""Data type conversion utilities for ONNX models.

This module provides functions for converting ONNX models between different
floating-point precisions, particularly FP16 to BF16 conversion.
"""

import os

import numpy as np
import onnx
from onnx import TensorProto, numpy_helper

from ...logging_config import logger


def fp16_to_bf16_array(fp16_array: np.ndarray) -> np.ndarray:
    """Convert a float16 numpy array to bfloat16.

    BF16 has 1 sign bit, 8 exponent bits, 7 mantissa bits.
    FP16 has 1 sign bit, 5 exponent bits, 10 mantissa bits.

    We go FP16 -> FP32 -> BF16 to avoid precision loss.

    Args:
        fp16_array: Input float16 numpy array.

    Returns:
        BF16 data as uint16 numpy array (raw bit representation).
    """
    # Convert FP16 to FP32
    fp32_array = fp16_array.astype(np.float32)

    # View as uint32 to manipulate bits
    uint32_view = fp32_array.view(np.uint32)

    # BF16 is just the upper 16 bits of FP32
    # Round to nearest even
    rounding = (uint32_view >> 16) & 1
    uint32_view = uint32_view + 0x7FFF + rounding

    # Shift right by 16 to get BF16 as uint16
    bf16_uint16 = (uint32_view >> 16).astype(np.uint16)

    return bf16_uint16


def _convert_initializer_to_bf16(initializer: onnx.TensorProto) -> onnx.TensorProto:
    """Convert an FP16 initializer to BF16.

    Args:
        initializer: ONNX TensorProto initializer.

    Returns:
        New initializer with BF16 data type.
    """
    if initializer.data_type != TensorProto.FLOAT16:
        return initializer

    # Get the FP16 data
    fp16_array = numpy_helper.to_array(initializer)

    # Convert to BF16 (stored as uint16)
    bf16_uint16 = fp16_to_bf16_array(fp16_array)

    # Create new initializer with BF16 type
    new_initializer = onnx.TensorProto()
    new_initializer.name = initializer.name
    new_initializer.data_type = TensorProto.BFLOAT16
    new_initializer.dims.extend(initializer.dims)

    # Store BF16 data as raw bytes
    new_initializer.raw_data = bf16_uint16.tobytes()

    return new_initializer


def _convert_constant_node_to_bf16(node: onnx.NodeProto) -> bool:
    """Convert a Constant node's FP16 value to BF16.

    Args:
        node: ONNX NodeProto to convert.

    Returns:
        True if conversion was performed, False otherwise.
    """
    if node.op_type != "Constant":
        return False

    for attr in node.attribute:
        if attr.name == "value" and attr.t.data_type == TensorProto.FLOAT16:
            # Get the FP16 tensor
            fp16_array = numpy_helper.to_array(attr.t)

            # Convert to BF16
            bf16_uint16 = fp16_to_bf16_array(fp16_array)

            # Update the tensor in place
            attr.t.data_type = TensorProto.BFLOAT16
            attr.t.ClearField("raw_data")
            attr.t.ClearField("float_data")
            attr.t.ClearField("int32_data")
            attr.t.raw_data = bf16_uint16.tobytes()

            return True

        # Handle value_float attribute
        if attr.name == "value_float":
            fp32_val = np.array([attr.f], dtype=np.float32)
            bf16_uint16 = fp16_to_bf16_array(fp32_val.astype(np.float16))
            new_tensor = onnx.TensorProto()
            new_tensor.data_type = TensorProto.BFLOAT16
            new_tensor.raw_data = bf16_uint16.tobytes()
            attr.t.CopyFrom(new_tensor)
            attr.name = "value"
            return True

    return False


def convert_fp16_to_bf16(
    input_path: str,
    output_path: str,
    external_data: bool = True,
    verbose: bool = True,
) -> dict[str, int]:
    """Convert an ONNX model from FP16 to BF16.

    This function converts:
    1. All FP16 initializers (weights) to BF16
    2. All FP16 value_info (intermediate tensors) to BF16
    3. All FP16 graph inputs/outputs to BF16
    4. All Cast nodes that target FP16 to target BF16

    Args:
        input_path: Path to input FP16 ONNX model.
        output_path: Path to output BF16 ONNX model.
        external_data: Whether to save weights as external data.
        verbose: Whether to print progress messages.

    Returns:
        Dictionary with conversion statistics.

    Example:
        >>> stats = convert_fp16_to_bf16(
        ...     input_path="model_fp16.onnx",
        ...     output_path="model_bf16.onnx",
        ... )
        >>> logger.info(f"Converted {stats['initializers_converted']} initializers")
    """
    if verbose:
        logger.info(f"Loading model from: {input_path}")

    # Load model with external data
    model = onnx.load(input_path, load_external_data=True)
    graph = model.graph

    # Statistics
    stats = {
        "initializers_converted": 0,
        "constants_converted": 0,
        "casts_converted": 0,
        "value_info_converted": 0,
        "inputs_converted": 0,
        "outputs_converted": 0,
    }

    # 1. Convert initializers (weights)
    if verbose:
        logger.info("Converting initializers...")
    new_initializers = []
    for init in graph.initializer:
        if init.data_type == TensorProto.FLOAT16:
            new_init = _convert_initializer_to_bf16(init)
            new_initializers.append(new_init)
            stats["initializers_converted"] += 1
        else:
            new_initializers.append(init)

    # Clear and replace initializers
    while len(graph.initializer) > 0:
        graph.initializer.pop()
    graph.initializer.extend(new_initializers)

    # 2. Convert Constant nodes and Cast nodes
    if verbose:
        logger.info("Converting Constant nodes and Cast nodes...")
    for node in graph.node:
        if _convert_constant_node_to_bf16(node):
            stats["constants_converted"] += 1

        # Convert Cast nodes that cast to FP16
        if node.op_type == "Cast":
            for attr in node.attribute:
                if attr.name == "to" and attr.i == TensorProto.FLOAT16:
                    attr.i = TensorProto.BFLOAT16
                    stats["casts_converted"] += 1

    # 3. Convert value_info (intermediate tensors)
    if verbose:
        logger.info("Converting value_info...")
    for vi in graph.value_info:
        if vi.type.HasField("tensor_type"):
            if vi.type.tensor_type.elem_type == TensorProto.FLOAT16:
                vi.type.tensor_type.elem_type = TensorProto.BFLOAT16
                stats["value_info_converted"] += 1

    # 4. Convert graph inputs
    if verbose:
        logger.info("Converting graph inputs...")
    for inp in graph.input:
        if inp.type.HasField("tensor_type"):
            if inp.type.tensor_type.elem_type == TensorProto.FLOAT16:
                inp.type.tensor_type.elem_type = TensorProto.BFLOAT16
                stats["inputs_converted"] += 1

    # 5. Convert graph outputs
    if verbose:
        logger.info("Converting graph outputs...")
    for out in graph.output:
        if out.type.HasField("tensor_type"):
            if out.type.tensor_type.elem_type == TensorProto.FLOAT16:
                out.type.tensor_type.elem_type = TensorProto.BFLOAT16
                stats["outputs_converted"] += 1

    # Print statistics
    if verbose:
        logger.info("\nConversion statistics:")
        logger.info(f"  Initializers converted: {stats['initializers_converted']}")
        logger.info(f"  Constants converted: {stats['constants_converted']}")
        logger.info(f"  Cast nodes converted: {stats['casts_converted']}")
        logger.info(f"  Value_info converted: {stats['value_info_converted']}")
        logger.info(f"  Inputs converted: {stats['inputs_converted']}")
        logger.info(f"  Outputs converted: {stats['outputs_converted']}")

    # Save model
    if verbose:
        logger.info(f"\nSaving model to: {output_path}")

    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    if external_data:
        external_data_path = os.path.basename(output_path) + ".data"
        onnx.save(
            model,
            output_path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=external_data_path,
            size_threshold=0,
        )
        if verbose:
            logger.info(f"  External data saved to: {external_data_path}")
    else:
        onnx.save(model, output_path)

    if verbose:
        logger.info("Done!")

    return stats
