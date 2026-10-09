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

import numpy as np
import onnx
import onnxruntime as ort
import pytest

import modelopt.onnx.quantization as moq


@pytest.mark.parametrize("high_precision_dtype", ["fp32", "fp16"])
@pytest.mark.parametrize(("target_dla", "dq_only"), [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize(
    ("op_type", "scalar_shape", "constant_node"),
    [("Mul", [], False), ("Mul", [1, 1, 1, 1], False), ("Div", [1], True), ("Pad", [], True)],
)
def test_dla_preserves_scalar_operator_constants(
    tmp_path, target_dla, op_type, scalar_shape, constant_node, dq_only, high_precision_dtype
):
    """Preserve scalar values and shared quantized weights in Q/DQ and DQ-only exports."""
    data = np.linspace(-1, 1, 8, dtype=np.float32).reshape(1, 1, 2, 4)
    value = np.full(scalar_shape, 0.0 if op_type == "Pad" else 2.0, dtype=np.float32)
    scalar = onnx.numpy_helper.from_array(value, "scalar")
    # The singleton 1x1 Conv weight is shared with Mul in one case. Bypassing
    # the scalar operand must leave the Conv's shared INT8 weight path intact.
    shared_weight = scalar_shape == [1, 1, 1, 1]
    weight_name = "scalar" if shared_weight else "weight"
    initializers = [] if constant_node else [scalar]
    if not shared_weight:
        initializers.append(
            onnx.numpy_helper.from_array(np.full((1, 1, 1, 1), 2.0, np.float32), "weight")
        )
    nodes = []
    if constant_node:
        nodes.append(
            onnx.helper.make_node("Constant", [], ["scalar"], name="scalar_constant", value=scalar)
        )
    nodes.append(
        onnx.helper.make_node("Conv", ["input", weight_name], ["conv_output"], name="conv")
    )
    inputs = ["conv_output", "scalar"]
    output_shape = list(data.shape)
    if op_type == "Pad":
        initializers.append(
            onnx.numpy_helper.from_array(np.array([0, 0, 0, 0, 0, 0, 2, 0], np.int64), "pads")
        )
        inputs = ["conv_output", "pads", "scalar"]
        output_shape[2] += 2
    nodes.append(onnx.helper.make_node(op_type, inputs, ["output"], name="scalar_operator"))
    model = onnx.helper.make_model(
        onnx.helper.make_graph(
            nodes,
            "scalar_constants",
            [onnx.helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, list(data.shape))],
            [onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, output_shape)],
            initializers,
        ),
        opset_imports=[onnx.helper.make_opsetid("", 19)],
        ir_version=10,
    )
    source, output = tmp_path / "source.onnx", tmp_path / "quantized.onnx"
    onnx.save(model, source)
    moq.quantize(
        str(source),
        output_path=str(output),
        quantize_mode="int8",
        high_precision_dtype=high_precision_dtype,
        calibration_data={"input": data},
        calibration_eps=["cpu"],
        target_dla=target_dla,
        dq_only=dq_only,
        enable_shared_constants_duplication=not shared_weight,
        op_types_to_quantize=["Conv", op_type],
    )
    quantized = onnx.load(output)
    onnx.checker.check_model(quantized, full_check=True)
    producers = {name: node for node in quantized.graph.node for name in node.output}
    constants = {
        tensor.name: onnx.numpy_helper.to_array(tensor) for tensor in quantized.graph.initializer
    }
    for node in quantized.graph.node:
        if node.op_type == "Constant":
            constants[node.output[0]] = onnx.numpy_helper.to_array(
                next(attribute.t for attribute in node.attribute if attribute.name == "value")
            )
    operation = next(node for node in quantized.graph.node if node.name == "scalar_operator")
    scalar_input = operation.input[2 if op_type == "Pad" else 1]
    if target_dla:
        assert scalar_input in constants, "DLA scalar operands must stay compile-time constants"
        scalar_dtype = np.float16 if high_precision_dtype == "fp16" else np.float32
        assert constants[scalar_input].dtype == scalar_dtype
        np.testing.assert_array_equal(constants[scalar_input], value.astype(scalar_dtype))
        conv = next(node for node in quantized.graph.node if node.op_type == "Conv")
        weight_dq = producers[conv.input[1]]
        assert weight_dq.op_type == "DequantizeLinear"
        if dq_only:
            assert constants[weight_dq.input[0]].dtype == np.int8
        else:
            assert producers[weight_dq.input[0]].op_type == "QuantizeLinear"
        assert producers[operation.input[0]].op_type == "DequantizeLinear"
    else:
        assert producers[scalar_input].op_type == "DequantizeLinear"
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(output), sess_options=options, providers=["CPUExecutionProvider"]
    )
    actual = session.run(None, {"input": data})[0]
    expected = (
        data * 4
        if op_type == "Mul"
        else data
        if op_type == "Div"
        else np.pad(data * 2, ((0, 0), (0, 0), (0, 2), (0, 0)))
    )
    np.testing.assert_allclose(actual, expected, atol=0.04, rtol=0.02)
