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
import onnx_graphsurgeon as gs
import onnxruntime as ort
import pytest

import modelopt.onnx.quantization as moq


def _constant(name, values):
    return gs.Constant(name, np.asarray(values, dtype=np.float32))


def _conv(graph, name, tensor):
    weight = _constant(f"{name}_weight", np.eye(2).reshape(2, 2, 1, 1))
    return graph.layer(op="Conv", name=name, inputs=[tensor, weight], outputs=[name])[0]


def _batchnorm(graph, tensor):
    parameters = [
        _constant("gamma", [0.5, 1.5]),
        _constant("beta", [0.25, -0.5]),
        _constant("mean", [0.1, -0.2]),
        _constant("variance", [1.0, 0.25]),
    ]
    return graph.layer(
        op="BatchNormalization", name="scale", inputs=[tensor, *parameters], outputs=["scale"]
    )[0]


def _quantize(tmp_path, graph, **kwargs):
    """Check exported topology and inference against the floating-point source model."""
    for output in graph.outputs:
        output.dtype = np.float32
        output.shape = [1, 2, 4, 4]
    source, output = tmp_path / "source.onnx", tmp_path / "quantized.onnx"
    model = gs.export_onnx(graph)
    model.ir_version = 10
    onnx.save(model, source)
    data = np.linspace(-1, 1, 32, dtype=np.float32).reshape(1, 2, 4, 4)
    moq.quantize(
        str(source),
        output_path=str(output),
        quantize_mode="int8",
        high_precision_dtype="fp16",
        calibration_data={"input": data},
        calibration_eps=["cpu"],
        **kwargs,
    )
    quantized = onnx.load(output)
    onnx.checker.check_model(quantized, full_check=True)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    results = [
        ort.InferenceSession(
            str(path), sess_options=options, providers=["CPUExecutionProvider"]
        ).run(None, {"input": data})[0]
        for path in (source, output)
    ]
    np.testing.assert_allclose(results[1], results[0], atol=0.04, rtol=0.03)
    return gs.import_onnx(quantized)


def _assert_qdq(tensor):
    """Validate a symmetric INT8 boundary and return its floating-point source."""
    dq = tensor.inputs[0]
    assert dq.op == "DequantizeLinear"
    q = dq.inputs[0].inputs[0]
    assert q.op == "QuantizeLinear"
    for quantized, dequantized in zip(q.inputs[1:], dq.inputs[1:]):
        np.testing.assert_array_equal(quantized.values, dequantized.values)
    assert q.inputs[2].values.dtype == np.int8
    np.testing.assert_array_equal(q.inputs[2].values, 0)
    return q.inputs[0]


@pytest.mark.parametrize(
    "op_type", ["Conv", "LeakyRelu", "BatchNormalization", "MaxPool", "AveragePool", "Add", "Mul"]
)
def test_dla_quantizes_adjacent_layers(tmp_path, op_type):
    """Each unfused operation receives its own quantized activation boundary."""
    data = gs.Variable("input", dtype=np.float32, shape=[1, 2, 4, 4])
    graph = gs.Graph(inputs=[data], opset=19)
    first = _conv(graph, "first", data)
    if op_type == "Conv":
        middle = _conv(graph, "middle", first)
    elif op_type == "BatchNormalization":
        middle = _batchnorm(graph, first)
    else:
        inputs = [first, data] if op_type in {"Add", "Mul"} else [first]
        attrs = {"kernel_shape": [3, 3], "pads": [1, 1, 1, 1]} if "Pool" in op_type else {}
        middle = graph.layer(
            op=op_type, name="middle", inputs=inputs, outputs=["middle"], attrs=attrs
        )[0]
    graph.outputs = [_conv(graph, "last", middle)]
    quantized = _quantize(tmp_path, graph, target_dla=True)
    nodes = {node.name: node for node in quantized.nodes}
    middle_node = nodes["scale" if op_type == "BatchNormalization" else "middle"]
    assert _assert_qdq(middle_node.inputs[0]) is nodes["first"].outputs[0]
    assert _assert_qdq(nodes["last"].inputs[0]) is middle_node.outputs[0]
    if op_type in {"Add", "Mul"}:
        assert _assert_qdq(middle_node.inputs[1]) is _assert_qdq(nodes["first"].inputs[0])
    for node in quantized.nodes:
        if node.op == "Conv":
            _assert_qdq(node.inputs[0])
            assert isinstance(_assert_qdq(node.inputs[1]), gs.Constant)
    # The final Conv has no following Q, so its output remains floating point.
    output = quantized.outputs[0]
    if output.inputs[0].op == "Cast":
        output = output.inputs[0].inputs[0]
    assert output is nodes["last"].outputs[0]
    assert output.dtype == np.float16


@pytest.mark.parametrize("pattern", ["conv_scale", "conv_scale_pool", "conv_relu_conv"])
def test_dla_preserves_selected_fusion_boundaries(tmp_path, pattern):
    """Retain valid fusion boundaries and symmetric parameters in the exported ONNX."""
    data = gs.Variable("input", dtype=np.float32, shape=[1, 2, 4, 4])
    graph = gs.Graph(inputs=[data], opset=19)
    conv = _conv(graph, "conv", data)
    if pattern == "conv_relu_conv":
        fused = graph.layer(op="Relu", name="activation", inputs=[conv], outputs=["activation"])[0]
        output = _conv(graph, "tail", fused)
    else:
        fused = _batchnorm(graph, conv)
        output = fused
        if pattern == "conv_scale_pool":
            output = graph.layer(
                op="MaxPool",
                name="tail",
                inputs=[fused],
                outputs=["tail"],
                attrs={"kernel_shape": [3, 3], "pads": [1, 1, 1, 1]},
            )[0]
    graph.outputs = [output]
    # Select the fusion boundaries explicitly; this validates export, not compiler fusion.
    selected = ["conv"] if pattern == "conv_scale" else ["conv", "tail"]
    quantized = _quantize(tmp_path, graph, target_dla=True, nodes_to_quantize=selected)
    nodes = {node.name: node for node in quantized.nodes}
    fused_node = nodes["activation" if pattern == "conv_relu_conv" else "scale"]
    assert fused_node.inputs[0] is nodes["conv"].outputs[0]
    _assert_qdq(nodes["conv"].inputs[0])
    assert isinstance(_assert_qdq(nodes["conv"].inputs[1]), gs.Constant)
    if pattern != "conv_scale":
        assert _assert_qdq(nodes["tail"].inputs[0]) is fused_node.outputs[0]
    if pattern != "conv_relu_conv":
        assert all(isinstance(tensor, gs.Constant) for tensor in fused_node.inputs[1:])
