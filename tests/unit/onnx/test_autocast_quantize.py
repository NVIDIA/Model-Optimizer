# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import os

import onnx
import onnx_graphsurgeon as gs
import pytest
import torch
from _test_utils.onnx.lib_test_models import SimpleMLP, export_as_onnx

from modelopt.onnx.autocast import convert_to_mixed_precision
from modelopt.onnx.quantization import quantize


def assert_nodes_are_quantized(nodes):
    for node in nodes:
        for inp_idx, inp in enumerate(node.inputs):
            if isinstance(inp, gs.Variable):
                assert node.i(inp_idx).op == "DequantizeLinear", (
                    f"Input '{inp.name}' of node '{node.name}' is not quantized but should be!"
                )
    return True


@pytest.mark.parametrize("keep_io_types", [True, False])
@pytest.mark.parametrize("bias_add", [True, False])
def test_autocast_quantize_int8(tmp_path, keep_io_types, bias_add):
    model_torch = SimpleMLP(bias_add=bias_add)
    input_tensor = torch.randn(2, 16, 16)
    low_precision_type = "fp16"

    onnx_path = os.path.join(tmp_path, f"model{'_biasAdd' if bias_add else ''}.onnx")
    export_as_onnx(model_torch, input_tensor, onnx_filename=onnx_path)

    # Convert model to low precision
    converted_model = convert_to_mixed_precision(
        onnx_path, keep_io_types=keep_io_types, low_precision_type=low_precision_type
    )
    converted_model_path = onnx_path.replace(
        ".onnx", f".{low_precision_type}{'_keepIOTypes' if keep_io_types else ''}.onnx"
    )
    onnx.save(converted_model, converted_model_path)

    # Quantize converted model
    quantize(converted_model_path, quantize_mode="int8", high_precision_dtype=low_precision_type)

    # Output model should be produced in the same tmp_path
    output_onnx_path = converted_model_path.replace(".onnx", ".quant.onnx")

    # Check that quantized explicit model is generated
    assert os.path.isfile(output_onnx_path)

    # Load the output model and check QDQ node placements
    graph = gs.import_onnx(onnx.load(output_onnx_path))

    # Check that all MatMul nodes are quantized
    mm_nodes = [n for n in graph.nodes if n.op == "MatMul"]
    assert assert_nodes_are_quantized(mm_nodes)
