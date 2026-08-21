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

import os
import sys

import onnx
import onnx_graphsurgeon as gs
from _test_utils.onnx.lib_test_models import build_conv_concat_model

from modelopt.onnx.quantization.quantize import quantize


def assert_nodes_are_quantized(nodes):
    for node in nodes:
        for inp_idx, inp in enumerate(node.inputs):
            if isinstance(inp, gs.Variable) and node.i(inp_idx).op != "Identity":
                assert node.i(inp_idx).op == "DequantizeLinear", (
                    f"Input '{inp.name}' of node '{node.name}' is not quantized but should be!"
                )
    return True


def _check_concat_qdq_status(onnx_path, quantize_mode):
    # Quantize the input model
    quantize(onnx_path, quantize_mode=quantize_mode, passes="concat_elimination")

    # Output model should be produced in the same tmpdir
    output_onnx_path = onnx_path.replace(".onnx", ".quant.onnx")

    # Check that quantized explicit model is generated
    assert os.path.isfile(output_onnx_path)

    # Load the output model and check QDQ node placements
    graph = gs.import_onnx(onnx.load(output_onnx_path))

    #   Check that all Conv nodes are quantized
    conv_nodes = [n for n in graph.nodes if n.op == "Conv"]
    assert assert_nodes_are_quantized(conv_nodes)

    check_num = 0
    for node in graph.nodes:
        if node.op == "Concat":
            assert node.o(0).op == "QuantizeLinear"
            for inp_idx, inp in enumerate(node.inputs):
                if isinstance(inp, gs.Variable) and node.i(inp_idx).op == "DequantizeLinear":
                    assert node.o(0).inputs[1].values == node.i(inp_idx).inputs[1].values, (
                        f"Concat output scale '{node.o(0).inputs[1].values}'\
                          should be equal to input scale '{node.i(inp_idx).inputs[1].values}'"
                    )
                    assert node.o(0).inputs[2].values == node.i(inp_idx).inputs[2].values, (
                        f"Concat output zero point '{node.o(0).inputs[2].values}'\
                          should be equal to input zero point '{node.i(inp_idx).inputs[2].values}'"
                    )
                    check_num = check_num + 1
    assert check_num == 2


def test_concat_elim_int8(tmp_path):
    # Copy the test model to the tmp_path
    model = build_conv_concat_model()
    this_function_name = sys._getframe().f_code.co_name
    onnx_path = os.path.join(tmp_path, f"{this_function_name}.onnx")
    onnx.save(model, onnx_path)
    _check_concat_qdq_status(onnx_path, "int8")


def test_concat_elim_fp8(tmp_path):
    # Copy the test model to the tmp_path
    model = build_conv_concat_model()
    this_function_name = sys._getframe().f_code.co_name
    onnx_path = os.path.join(tmp_path, f"{this_function_name}.onnx")
    onnx.save(model, onnx_path)
    _check_concat_qdq_status(onnx_path, "fp8")
