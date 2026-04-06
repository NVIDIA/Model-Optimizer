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

"""Utilities for QDQ (Quantize-Dequantize) node identification and manipulation."""

import onnx

from modelopt.onnx.logging_config import logger


def get_quantized_tensors(onnx_model: onnx.ModelProto) -> set[str]:
    """Get the names of all quantized tensors from an ONNX model.

    This function identifies all QuantizeLinear nodes in the ONNX model
    and extracts the names of tensors being quantized (the first input of
    each QuantizeLinear node, excluding scale and zero-point inputs).

    Args:
        onnx_model: ONNX model protobuf to analyze

    Returns:
        Set of tensor names that are inputs to QuantizeLinear nodes
        (i.e., the tensors being quantized)

    Example:
        >>> import onnx
        >>> from modelopt.onnx.quantization.autotune.qdq_utils import get_quantized_tensors
        >>>
        >>> # Load a quantized model
        >>> model = onnx.load("quantized_model.onnx")
        >>>
        >>> # Get all quantized tensor names
        >>> quantized_tensors = get_quantized_tensors(model)
        >>> print(f"Found {len(quantized_tensors)} quantized tensors")
        >>>
        >>> # Use with autotuner to import insertion points
        >>> from modelopt.onnx.quantization.autotune import QDQAutotuner
        >>> autotuner = QDQAutotuner(new_model)
        >>> autotuner.initialize()
        >>> autotuner.import_insertion_points(quantized_tensors)
    """
    quantized_tensors = set()

    for node in onnx_model.graph.node:
        if node.op_type == "QuantizeLinear":
            # First input is the tensor being quantized
            # (inputs[1] is scale, inputs[2] is zero-point)
            if node.input:
                quantized_tensors.add(node.input[0])

    logger.debug(f"Found {len(quantized_tensors)} quantized tensors in ONNX model")
    return quantized_tensors
