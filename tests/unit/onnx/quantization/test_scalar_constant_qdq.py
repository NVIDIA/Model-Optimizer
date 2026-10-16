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

"""Scalar (single-element) constant Q/DQ policy, exercised on CPU.

A per-tensor Q/DQ pair on a rank-0 (0-D) constant breaks TensorRT's ONNX parser
(``convertAxis`` asserts the axis lies in ``[0, nbDims]`` but ``nbDims == 0``).
These tests cover both the end-to-end quantize path (int8 and fp8) and the
``remove_scalar_input_qdq`` graph pass directly.
"""

import os

import numpy as np
import onnx
import onnx_graphsurgeon as gs
import pytest

import modelopt.onnx.quantization as moq
from modelopt.onnx.quantization.qdq_graph import remove_scalar_input_qdq


def _build_matmul_add_relu(onnx_path):
    """MatMul -> Add(rank-0 constant) -> Relu, mirroring the autotune failure."""
    x = onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [4, 4])
    y = onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [4, 4])
    weight = onnx.numpy_helper.from_array(np.random.randn(4, 4).astype(np.float32), name="w")
    # Rank-0 (0-D) constant shared into Add, exactly what the TRT parser rejects.
    one = onnx.numpy_helper.from_array(np.array(1.0, dtype=np.float32), name="one")
    nodes = [
        onnx.helper.make_node("MatMul", ["x", "w"], ["mm"], name="matmul"),
        onnx.helper.make_node("Add", ["mm", "one"], ["added"], name="add"),
        onnx.helper.make_node("Relu", ["added"], ["y"], name="relu"),
    ]
    graph = onnx.helper.make_graph(nodes, "mm_add_relu", [x], [y], initializer=[weight, one])
    model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_operatorsetid("", 17)])
    model.ir_version = 9
    onnx.checker.check_model(model)
    onnx.save(model, onnx_path)


@pytest.mark.parametrize("quantize_mode", ["int8", "fp8"])
def test_scalar_constant_input_is_not_quantized(tmp_path, quantize_mode):
    onnx_path = os.path.join(tmp_path, "model.onnx")
    _build_matmul_add_relu(onnx_path)

    moq.quantize(onnx_path, quantize_mode=quantize_mode, nodes_to_quantize=["add", "matmul"])

    output_path = onnx_path.replace(".onnx", ".quant.onnx")
    assert os.path.isfile(output_path)

    graph = gs.import_onnx(onnx.load(output_path))
    quantized_rank0 = {
        node.inputs[0].name
        for node in graph.nodes
        if node.op == "QuantizeLinear"
        and isinstance(node.inputs[0], gs.Constant)
        and node.inputs[0].values.ndim == 0
    }
    assert not quantized_rank0, f"rank-0 constants were quantized: {sorted(quantized_rank0)}"


def _graph_with_qdq_scalar(op_type, scalar_shape):
    """Build ``op_type(x, DQ(Q(const)))`` with ``const`` of ``scalar_shape``."""
    x = gs.Variable("x", dtype=np.float32, shape=[4, 4])
    const = gs.Constant("const", values=np.full(scalar_shape, 2.0, dtype=np.float32))
    scale = gs.Constant("scale", values=np.array(0.1, dtype=np.float32))
    zero_point = gs.Constant("zp", values=np.array(0, dtype=np.int8))

    quantized = gs.Variable("const_q", dtype=np.int8)
    quantize = gs.Node(
        op="QuantizeLinear", name="q", inputs=[const, scale, zero_point], outputs=[quantized]
    )
    dequantized = gs.Variable("const_dq", dtype=np.float32)
    dequantize = gs.Node(
        op="DequantizeLinear",
        name="dq",
        inputs=[quantized, scale, zero_point],
        outputs=[dequantized],
    )
    out = gs.Variable("y", dtype=np.float32)
    op = gs.Node(op=op_type, name="op", inputs=[x, dequantized], outputs=[out])

    graph = gs.Graph(nodes=[quantize, dequantize, op], inputs=[x], outputs=[out])
    return graph


@pytest.mark.parametrize(
    ("op_type", "scalar_shape", "target_dla", "expect_stripped"),
    [
        # Rank-0 constants are always kept unquantized, for any op.
        ("Add", [], False, True),
        ("Add", [], True, True),
        ("Mul", [], False, True),
        # Size-1 but rank-1: only stripped for DLA Mul/Div/Pad operands.
        ("Mul", [1], False, False),
        ("Mul", [1], True, True),
        # Not scalar: never stripped.
        ("MatMul", [4, 4], True, False),
    ],
)
def test_remove_scalar_input_qdq(op_type, scalar_shape, target_dla, expect_stripped):
    graph = _graph_with_qdq_scalar(op_type, scalar_shape)

    remove_scalar_input_qdq(graph, target_dla=target_dla)

    op = next(node for node in graph.nodes if node.name == "op")
    producers = op.inputs[1].inputs
    stripped = not (producers and producers[0].op == "DequantizeLinear")
    assert stripped == expect_stripped
