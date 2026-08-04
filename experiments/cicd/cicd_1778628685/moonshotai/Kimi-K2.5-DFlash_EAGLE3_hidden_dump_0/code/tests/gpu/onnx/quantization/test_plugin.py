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

import numpy as np
import onnx
import onnx_graphsurgeon as gs
from _test_utils.import_helper import skip_if_no_libcudnn, skip_if_no_tensorrt
from _test_utils.onnx.autocast.utils import _assert_tensors_are_fp16
from _test_utils.onnx.quantization.utils import assert_nodes_are_quantized

from modelopt.onnx.autocast import convert_to_mixed_precision
from modelopt.onnx.autocast.graphsanitizer import GraphSanitizer
from modelopt.onnx.quantization.quantize import quantize
from modelopt.onnx.trt_utils import load_onnx_model

skip_if_no_libcudnn()
skip_if_no_tensorrt()


def _create_test_model_trt():
    opset_version = 13
    tensor_shapes = [1, 16, 128, 768]

    # Instantiate ONNX variables
    input_0 = gs.Variable(name="input_0", dtype=np.float32, shape=tensor_shapes)
    conv_0_weight = gs.Constant(
        name="conv_0_weight", values=np.ones((tensor_shapes[1], 16, 1, 1), dtype=np.float32)
    )
    conv_0_output = gs.Variable(name="conv_0_out")  # , dtype=np.float32, shape=tensor_shapes)
    input_1 = gs.Variable(name="input_1", dtype=np.float32, shape=tensor_shapes)
    custom_output = gs.Variable(name="custom_skip_ln_0_out")  # , dtype=np.float32)
    conv_1_weight = gs.Constant(
        name="conv_1_weight", values=np.ones((16, 16, 1, 1), dtype=np.float32)
    )
    output = gs.Variable(name="output", dtype=np.float32)  # , shape=tensor_shapes)

    # Create ONNX nodes
    conv_0 = gs.Node(
        name="Conv_0",
        op="Conv",
        inputs=[input_0, conv_0_weight],
        outputs=[conv_0_output],
        attrs={
            "kernel_shape": [1, 1],
            "strides": [1, 1],
            "pads": [0, 0, 0, 0],
            "group": tensor_shapes[0],
        },
    )
    custom_op_0 = gs.Node(
        name="custom_skip_ln_0",
        op="CustomSkipLayerNormPluginDynamic",
        inputs=[conv_0_output, input_1],
        outputs=[custom_output],
        attrs={
            "plugin_version": "2",
            "ld": tensor_shapes[-1],
            "beta": np.zeros(tensor_shapes[-1]),
            "gamma": np.ones(tensor_shapes[-1]),
            "type_id": 0,  # 0: float32, 1: float16, 2: int8
        },
    )
    conv_1 = gs.Node(
        name="Conv_1",
        op="Conv",
        inputs=[custom_output, conv_1_weight],
        outputs=[output],
        attrs={
            "kernel_shape": [1, 1],
            "strides": [1, 1],
            "pads": [0, 0, 0, 0],
            "group": tensor_shapes[0],
        },
    )

    # Create ONNX graph and export model
    graph = gs.Graph(
        nodes=[conv_0, custom_op_0, conv_1],
        inputs=[input_0, input_1],
        outputs=[output],
        opset=opset_version,
    )

    model = gs.export_onnx(graph)
    model.ir_version = 10

    return model


def test_trt_plugin_quantization(tmp_path):
    model = _create_test_model_trt()
    with open(os.path.join(tmp_path, "model_with_trt_plugin.onnx"), "w") as f:
        onnx.save_model(model, f.name)

        # Check that the model contains TRT custom op
        _, has_custom_op, custom_ops, _, _ = load_onnx_model(f.name)
        assert has_custom_op and custom_ops == {"CustomSkipLayerNormPluginDynamic"}

        # Quantize model
        quantize(f.name, calibration_eps=["trt", "cuda:0", "cpu"])

        # Output model should be produced in the same tmp_path
        output_onnx_path = f.name.replace(".onnx", ".quant.onnx")

        # Check that quantized explicit model is generated
        assert os.path.isfile(output_onnx_path)

        # Load the output model and check QDQ node placements
        graph = gs.import_onnx(onnx.load(output_onnx_path))

        # Check that the default quantization happened successfully: Conv layer should be quantized
        quantizable_nodes = [n for n in graph.nodes if n.op == "Conv"]
        assert assert_nodes_are_quantized(quantizable_nodes)


def test_trt_plugin_autocast(tmp_path):
    model = _create_test_model_trt()
    with open(os.path.join(tmp_path, "model_with_trt_plugin_autocast.onnx"), "w") as f:
        onnx.save_model(model, f.name)

        # Check that the model contains TRT custom op
        graph_sanitizer = GraphSanitizer(model)
        graph_sanitizer.sanitize()
        assert graph_sanitizer.custom_ops == {"CustomSkipLayerNormPluginDynamic"}

        # Convert model to FP16
        model_fp16 = convert_to_mixed_precision(f.name, providers=["trt", "cpu"])

        # Check that the default conversion happened successfully: all tensors are FP16
        assert _assert_tensors_are_fp16(model_fp16)
