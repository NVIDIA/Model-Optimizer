# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


import onnx
import pytest
import torch
from _test_utils.onnx.lib_test_models import UNet, export_as_onnx

from modelopt.onnx.quantization import quantize


@pytest.fixture
def model_and_input():
    """Create model and dummy input."""
    model = UNet()
    dummy_input = torch.randn(2, 1, 256, 256)
    return model, dummy_input


@pytest.fixture
def onnx_paths(tmp_path):
    """Create paths for ONNX files."""
    return {
        "onnx_path": tmp_path / "model.onnx",
        "quant_onnx_path": tmp_path / "model.quant.onnx",
    }


@pytest.mark.parametrize("high_precision_type", ["fp16", "fp32"])
def test_convtranspose_qdq(model_and_input, onnx_paths, high_precision_type):
    """Test that ConvTranspose weight inputs don't have QuantizeLinear nodes."""
    model, dummy_input = model_and_input
    onnx_path = onnx_paths["onnx_path"]
    quant_onnx_path = onnx_paths["quant_onnx_path"]

    # Step 1: Export ONNX
    export_as_onnx(model, dummy_input, onnx_filename=onnx_path, opset=17)

    # Step 2: Quantize
    quantize(
        str(onnx_path),
        quantize_mode="int8",
        output_path=str(quant_onnx_path),
        high_precision_type=high_precision_type,
    )

    # Step 3: Load and verify quantized model
    quant_model = onnx.load(str(quant_onnx_path))

    # Check that weight inputs to ConvTranspose don't have QuantizeLinear
    conv_transpose_nodes = [
        node for node in quant_model.graph.node if node.op_type == "ConvTranspose"
    ]
    for node in conv_transpose_nodes:
        # Get the weight input (usually the second input for ConvTranspose)
        weight_input = node.input[1]

        # Find all nodes that produce this weight input
        weight_producers = [n for n in quant_model.graph.node if weight_input in n.output]

        # Check that none of the producers are QuantizeLinear
        for producer in weight_producers:
            assert producer.op_type != "QuantizeLinear", (
                f"Weight input {weight_input} to ConvTranspose {node.name} comes from QuantizeLinear node"
            )
