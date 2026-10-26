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

import hashlib

import numpy as np
import onnx
import onnx_graphsurgeon as gs
import pytest

from modelopt.onnx.quantization.gs_patching import patch_gs_modules
from modelopt.onnx.quantization.int4 import quantize_rtn
from modelopt.onnx.trt_utils import load_onnx_model
from modelopt.onnx.utils import export_onnx, save_onnx


@pytest.mark.manual
@pytest.mark.timeout(300)
@pytest.mark.parametrize("route", ["load", "rtn", "subgraph"])
def test_export_initializer_over_protobuf_limit(tmp_path, route):
    """Large unquantized weights must survive preprocessing and INT4 graph reconstruction."""
    size = onnx.checker.MAXIMUM_PROTOBUF // 2 + 1
    large = gs.Constant("large", np.broadcast_to(np.float16(1), (size,)))
    x = gs.Variable("x", dtype=np.float16, shape=[1, 2])
    y = gs.Variable("y", dtype=np.float16, shape=[1, 2])
    z = gs.Variable("z", dtype=np.float16, shape=[size])
    graph = gs.Graph(
        nodes=[
            gs.Node(
                "MatMul",
                name="matmul",
                inputs=[x, gs.Constant("w", np.ones((2, 2), np.float16))],
                outputs=[y],
            ),
            gs.Node("Identity", inputs=[large], outputs=[z]),
        ],
        inputs=[x],
        outputs=[y, z],
        opset=21,
    )
    expected = hashlib.sha256(large.values.tobytes()).digest()
    if route == "subgraph":
        # Keep the initializer scoped inside an If branch.
        branch = gs.Graph(nodes=[graph.nodes[1]], outputs=[z], opset=21)
        condition = gs.Variable("condition", dtype=np.bool_, shape=[])
        result = gs.Variable("result", dtype=np.float16, shape=[size])
        graph = gs.Graph(
            nodes=[
                gs.Node(
                    "If",
                    inputs=[condition],
                    outputs=[result],
                    attrs={"then_branch": branch, "else_branch": branch.copy()},
                )
            ],
            inputs=[condition],
            outputs=[result],
            opset=21,
        )
        model = export_onnx(graph)
        initializers = [attr.g.initializer[0] for attr in model.graph.node[0].attribute]
    else:
        # Construct input data without the repeated-field serialization being tested.
        graph.nodes.pop()
        graph.outputs.pop()
        model = gs.export_onnx(graph)
        model.graph.node.add(op_type="Identity", input=["large"], output=["z"])
        model.graph.output.add().CopyFrom(
            onnx.helper.make_tensor_value_info("z", onnx.TensorProto.FLOAT16, [size])
        )
        model.graph.initializer.add(
            name="large", data_type=onnx.TensorProto.FLOAT16, dims=[size]
        ).raw_data = large.values.tobytes()
        model.ir_version = 10
        if route == "load":
            input_path = str(tmp_path / "input.onnx")
            save_onnx(model, input_path, save_as_external_data=True)
            model, _, _, _, external = load_onnx_model(input_path)
            assert external
        else:
            patch_gs_modules()
            model = quantize_rtn(model, block_size=2, dq_only=True)
            assert any(node.op_type == "DequantizeLinear" for node in model.graph.node)
        initializers = [tensor for tensor in model.graph.initializer if tensor.name == "large"]
    assert initializers
    for tensor in initializers:
        assert hashlib.sha256(tensor.raw_data).digest() == expected
    output = str(tmp_path / "output.onnx")
    save_onnx(model, output, save_as_external_data=True)
    onnx.checker.check_model(output)
