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

import onnx
import onnx_graphsurgeon as gs
import pytest
import torch
from _test_utils.onnx.lib_test_models import SimpleMLP, export_as_onnx
from onnxruntime.quantization.calibrate import CalibrationDataReader

import modelopt.onnx.quantization as moq


def assert_nodes_are_quantized(nodes):
    for node in nodes:
        for inp_idx, inp in enumerate(node.inputs):
            if isinstance(inp, gs.Variable):
                assert node.i(inp_idx).op == "DequantizeLinear", (
                    f"Input '{inp.name}' of node '{node.name}' is not quantized but should be!"
                )
    return True


def int8_test_helper(tmp_path, high_precision_dtype, **kwargs):
    model_torch = SimpleMLP()
    input_tensor = torch.randn(2, 16, 16)

    onnx_path = os.path.join(tmp_path, "model.onnx")
    export_as_onnx(model_torch, input_tensor, onnx_filename=onnx_path)
    moq.quantize(
        onnx_path, quantize_mode="int8", high_precision_dtype=high_precision_dtype, **kwargs
    )

    # Output model should be produced in the same tmp_path
    output_onnx_path = onnx_path.replace(".onnx", ".quant.onnx")

    # Check that quantized explicit model is generated
    assert os.path.isfile(output_onnx_path)

    # Load the output model and check QDQ node placements
    graph = gs.import_onnx(onnx.load(output_onnx_path))

    # Check that all MatMul nodes are quantized
    mm_nodes = [n for n in graph.nodes if n.op == "MatMul"]
    assert assert_nodes_are_quantized(mm_nodes)


@pytest.mark.parametrize("high_precision_dtype", ["fp32", "fp16", "bf16"])
def test_int8(tmp_path, high_precision_dtype):
    int8_test_helper(tmp_path, high_precision_dtype)


@pytest.mark.parametrize("high_precision_dtype", ["fp32", "fp16", "bf16"])
def test_int8_with_calibration_reader(tmp_path, high_precision_dtype):
    input_tensor = torch.randn(2, 16, 16)

    # Calibration data comes from a custom data reader, enabling iterator based reading functionality
    class ExampleCalibrationDataReader(CalibrationDataReader):
        def __init__(self, input_data):
            self.data_list = [{"input": input_data.numpy()}]
            self.iter = iter(self.data_list)
            self.get_first_calls = 0
            self.get_next_calls = 0

        def get_next(self):
            self.get_next_calls += 1
            return next(self.iter, None)

        def get_first(self):
            self.get_first_calls += 1
            return self.data_list[0]

        def rewind(self):
            self.iter = iter(self.data_list)

    calibration_reader = ExampleCalibrationDataReader(input_tensor)
    int8_test_helper(tmp_path, high_precision_dtype, calibration_data_reader=calibration_reader)
    assert calibration_reader.get_first_calls > 0 or calibration_reader.get_next_calls > 0


@pytest.mark.parametrize("high_precision_dtype", ["fp32", "fp16", "bf16"])
def test_int8_with_calibration_data(tmp_path, high_precision_dtype):
    input_tensor = torch.randn(2, 16, 16)

    # test pre-allocated calibration data pathway
    calibration_data = {"input": input_tensor.numpy()}
    int8_test_helper(tmp_path, high_precision_dtype, calibration_data=calibration_data)
