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
import pytest

from modelopt.onnx.quantization.qdq_graph import get_parallel_conv_fusion_tensors

TAILS = [
    (),
    ("Relu",),
    ("BatchNormalization",),
    ("BatchNormalization", "Relu"),
    ("Sigmoid", "Mul"),
    ("BatchNormalization", "Sigmoid", "Mul"),
]


def _parallel_conv_graph(conv_type, tail, reverse_mul=False):
    x = gs.Variable("input", dtype=np.float32, shape=[1, 2, 4, 4])
    weights = gs.Constant("weights", np.ones((2, 2, 1, 1), dtype=np.float32))
    bn_params = [
        gs.Constant(name, np.full(2, value, dtype=np.float32))
        for name, value in [("scale", 1), ("bias", 0), ("mean", 0), ("variance", 1)]
    ]
    graph = gs.Graph(inputs=[x], opset=19)
    for branch in range(2):
        current = x
        pre_sigmoid = None
        for step, op in enumerate((conv_type, *tail)):
            inputs = [current]
            attrs = {}
            if op == conv_type:
                inputs.append(weights)
                attrs = {"kernel_shape": [1, 1]}
            elif op == "BatchNormalization":
                inputs.extend(bn_params)
            elif op == "Sigmoid":
                pre_sigmoid = current
            elif op == "Mul":
                inputs.append(pre_sigmoid)
                if reverse_mul:
                    inputs.reverse()
            output = gs.Variable(f"branch_{branch}_{step}", dtype=np.float32)
            graph.nodes.append(gs.Node(op, f"{op}_{branch}", attrs, inputs, [output]))
            current = output
        output = gs.Variable(f"output_{branch}", dtype=np.float32, shape=[1, 2, 4, 4])
        graph.nodes.append(gs.Node("Relu", f"sink_{branch}", inputs=[current], outputs=[output]))
        graph.outputs.append(output)
    return graph


@pytest.mark.parametrize("conv_type", ["Conv", "ConvTranspose"])
@pytest.mark.parametrize("tail", TAILS)
def test_parallel_conv_fusion_preserves_supported_tails(conv_type, tail):
    graph = _parallel_conv_graph(conv_type, tail)
    model = gs.export_onnx(graph)
    onnx.checker.check_model(model)
    tensors = {f"branch_{branch}_{len(tail)}" for branch in range(2)}
    assert get_parallel_conv_fusion_tensors(model, ["sink_0", "sink_1"]) == dict.fromkeys(
        tensors, tensors
    )


@pytest.mark.parametrize("conv_type", ["Conv", "ConvTranspose"])
@pytest.mark.parametrize(
    ("tail", "intermediate"),
    [
        (("Relu",), "Conv"),
        (("BatchNormalization", "Relu"), "Conv"),
        (("BatchNormalization", "Relu"), "BatchNormalization"),
        (("Sigmoid", "Mul"), "Conv"),
        (("Sigmoid", "Mul"), "Sigmoid"),
        (("BatchNormalization", "Sigmoid", "Mul"), "BatchNormalization"),
        (("BatchNormalization", "Sigmoid", "Mul"), "Sigmoid"),
    ],
)
def test_parallel_conv_fusion_skips_extra_consumers(conv_type, tail, intermediate):
    graph = _parallel_conv_graph(conv_type, tail)
    intermediate = conv_type if intermediate == "Conv" else intermediate
    source = next(node for node in graph.nodes if node.name == f"{intermediate}_0")
    side_output = gs.Variable("side_output", dtype=np.float32, shape=[1, 2, 4, 4])
    graph.nodes.append(gs.Node("Identity", "side", inputs=source.outputs, outputs=[side_output]))
    graph.outputs.append(side_output)
    model = gs.export_onnx(graph.toposort())
    onnx.checker.check_model(model)
    assert get_parallel_conv_fusion_tensors(model, ["sink_0", "sink_1"]) == {}


@pytest.mark.parametrize("conv_type", ["Conv", "ConvTranspose"])
@pytest.mark.parametrize("tail", TAILS[-2:])
@pytest.mark.parametrize("reverse_mul", [False, True])
def test_parallel_conv_fusion_validates_silu_inputs(conv_type, tail, reverse_mul):
    graph = _parallel_conv_graph(conv_type, tail, reverse_mul)
    tensors = {f"branch_{branch}_{len(tail)}" for branch in range(2)}
    assert get_parallel_conv_fusion_tensors(
        gs.export_onnx(graph), ["sink_0", "sink_1"]
    ) == dict.fromkeys(tensors, tensors)
    mul = next(node for node in graph.nodes if node.name == "Mul_0")
    mul.inputs[0 if reverse_mul else 1] = graph.inputs[0]
    model = gs.export_onnx(graph)
    onnx.checker.check_model(model)
    assert get_parallel_conv_fusion_tensors(model, ["sink_0", "sink_1"]) == {}
