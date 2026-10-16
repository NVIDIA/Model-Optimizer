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

from modelopt.onnx.quantization.qdq_graph import get_concat_eliminated_tensors


def _concat_model(constant_input: bool) -> onnx.ModelProto:
    """Build ``Concat(cls | x_relu) -> MatMul``, where ``cls`` is an initializer or a second activation."""
    x = gs.Variable("x", dtype=np.float32, shape=[1, 4, 8])
    y = gs.Variable("y", dtype=np.float32, shape=[1, 1, 8])
    out = gs.Variable("out", dtype=np.float32)
    relu_out = gs.Variable("relu_out", dtype=np.float32)
    concat_out = gs.Variable("concat_out", dtype=np.float32)
    nodes = [gs.Node("Relu", "relu", inputs=[x], outputs=[relu_out])]

    if constant_input:
        # What constant folding leaves behind for a ViT class token: the Concat input has no producer.
        cls = gs.Constant("cls", values=np.ones((1, 1, 8), dtype=np.float32))
        inputs = [x]
    else:
        cls = gs.Variable("cls", dtype=np.float32, shape=[1, 1, 8])
        nodes.append(gs.Node("Relu", "relu_cls", inputs=[y], outputs=[cls]))
        inputs = [x, y]
    nodes.append(
        gs.Node("Concat", "concat", inputs=[cls, relu_out], outputs=[concat_out], attrs={"axis": 1})
    )
    weight = gs.Constant("weight", values=np.ones((8, 8), dtype=np.float32))
    nodes.append(gs.Node("MatMul", "matmul", inputs=[concat_out, weight], outputs=[out]))
    graph = gs.Graph(nodes=nodes, inputs=inputs, outputs=[out], opset=17)
    return gs.export_onnx(graph)


def test_concat_elimination_groups_activation_inputs():
    model = _concat_model(constant_input=False)
    groups = get_concat_eliminated_tensors(model, ["concat", "matmul", "relu", "relu_cls"])
    assert groups["concat_out"] == {"cls", "relu_out", "concat_out"}
    assert groups["cls"] == groups["concat_out"]


def test_concat_elimination_skips_constant_inputs():
    """A constant Concat input has no calibration range, so it must not join the shared-scale group."""
    model = _concat_model(constant_input=True)
    groups = get_concat_eliminated_tensors(model, ["concat", "matmul", "relu"])
    assert "cls" not in groups
    assert groups["concat_out"] == {"relu_out", "concat_out"}
    assert all("cls" not in group for group in groups.values())
