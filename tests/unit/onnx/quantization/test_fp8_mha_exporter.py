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

"""Tests for the attention-aware FP8 ONNX graph rewrites in ``FP8QuantExporter``."""

import io

import ml_dtypes
import numpy as np
import onnx
import onnx_graphsurgeon as gs
import pytest
import torch
from torch.onnx import symbolic_helper

from modelopt.onnx.export.fp8_exporter import FP8QuantExporter
from modelopt.torch.quantization.export_onnx import export_fp8_mha


class _FP8MHAFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, key, value, high_precision_dtype):
        return torch.nn.functional.scaled_dot_product_attention(query, key, value)

    @staticmethod
    @symbolic_helper.parse_args("v", "v", "v", "s")
    def symbolic(g, query, key, value, high_precision_dtype):
        return export_fp8_mha(
            g,
            query,
            key,
            value,
            q_quantized_scale=1.0,
            k_quantized_scale=1.0,
            v_quantized_scale=1.0,
            high_precision_flag=high_precision_dtype,
            disable_fp8_mha=False,
        )


class _FP8MHAModule(torch.nn.Module):
    def __init__(self, high_precision_dtype):
        super().__init__()
        self.high_precision_dtype = high_precision_dtype

    def forward(self, query, key, value):
        return _FP8MHAFunction.apply(query, key, value, self.high_precision_dtype)


def _var(name, dtype=np.float32, shape=None):
    return gs.Variable(name, dtype=dtype, shape=shape)


def _qdq(src):
    """Build ``QuantizeLinear → DequantizeLinear`` and return [Q, DQ], dq_out."""
    scale = gs.Constant("scale", np.array(0.1, dtype=np.float32))
    q_out, dq_out = _var("q_out"), _var("dq_out")
    return [
        gs.Node(op="QuantizeLinear", inputs=[src, scale], outputs=[q_out]),
        gs.Node(op="DequantizeLinear", inputs=[q_out, scale], outputs=[dq_out]),
    ], dq_out


def _graph(nodes, inputs, outputs):
    return gs.Graph(nodes=nodes, inputs=inputs, outputs=outputs, opset=19)


def test_move_mul_before_qdq_rewrites_dq_mul_matmul_pattern():
    """``DQ → Mul(const) → MatMul`` collapses to ``Mul → Q → DQ → MatMul``."""
    x, k, y, mul_out = _var("x"), _var("k"), _var("y"), _var("mul_out")
    qdq_nodes, dq_out = _qdq(x)
    mul = gs.Node(
        op="Mul",
        inputs=[dq_out, gs.Constant("c", np.array(0.5, dtype=np.float32))],
        outputs=[mul_out],
    )
    mm = gs.Node(op="MatMul", inputs=[mul_out, k], outputs=[y])
    graph = _graph([*qdq_nodes, mul, mm], [x, k], [y])

    assert FP8QuantExporter._move_mul_before_qdq(graph) == 1
    q = next(n for n in graph.nodes if n.op == "QuantizeLinear")
    assert q.inputs[0].inputs[0].op == "Mul"


def test_move_transpose_before_qdq_rewrites_dq_transpose_matmul_pattern():
    """``DQ → Transpose → MatMul`` collapses to ``Transpose → Q → DQ → MatMul``."""
    k_in, q_in, scores, t_out = _var("k_in"), _var("q_in"), _var("scores"), _var("t_out")
    qdq_nodes, dq_out = _qdq(k_in)
    t = gs.Node(op="Transpose", inputs=[dq_out], outputs=[t_out], attrs={"perm": [0, 2, 1]})
    mm = gs.Node(op="MatMul", inputs=[q_in, t_out], outputs=[scores])
    graph = _graph([*qdq_nodes, t, mm], [k_in, q_in], [scores])

    assert FP8QuantExporter._move_transpose_before_qdq(graph) == 1
    q = next(n for n in graph.nodes if n.op == "QuantizeLinear")
    assert q.inputs[0].inputs[0].op == "Transpose"


@pytest.mark.parametrize(
    ("high_precision_dtype", "numpy_dtype", "onnx_dtype"),
    [
        pytest.param(None, np.float32, onnx.TensorProto.FLOAT, id="legacy"),
        pytest.param("Float", np.float32, onnx.TensorProto.FLOAT, id="float"),
        pytest.param("Half", np.float16, onnx.TensorProto.FLOAT16, id="half"),
        pytest.param("BFloat16", ml_dtypes.bfloat16, onnx.TensorProto.BFLOAT16, id="bfloat16"),
    ],
)
def test_insert_qdq_after_softmax_adds_target_scale_q_dq(
    high_precision_dtype, numpy_dtype, onnx_dtype
):
    """Softmax → MatMul picks up ``Q → DQ`` with the fixed ``1/448`` scale."""
    scores, v, y, sm_out = (
        _var("scores", numpy_dtype, [2, 2]),
        _var("v", numpy_dtype, [2, 2]),
        _var("y", numpy_dtype, [2, 2]),
        _var("sm_out", numpy_dtype, [2, 2]),
    )
    sm = gs.Node(op="Softmax", inputs=[scores], outputs=[sm_out], attrs={"axis": -1})
    mm = gs.Node(op="MatMul", inputs=[sm_out, v], outputs=[y])
    graph = _graph([sm, mm], [scores, v], [y])

    count = (
        FP8QuantExporter._insert_qdq_after_softmax(graph)
        if high_precision_dtype is None
        else FP8QuantExporter._insert_qdq_after_softmax(graph, high_precision_dtype)
    )
    assert count == 1
    q = next(n for n in graph.nodes if n.op == "QuantizeLinear")
    dq = next(n for n in graph.nodes if n.op == "DequantizeLinear")
    expected_scale = np.array(1.0 / 448.0, dtype=numpy_dtype)
    for scale in (q.inputs[1], dq.inputs[1]):
        assert scale.values.dtype == expected_scale.dtype
        np.testing.assert_array_equal(scale.values, expected_scale)
    assert np.dtype(dq.outputs[0].dtype) == np.dtype(numpy_dtype)
    assert mm.inputs[0] is dq.outputs[0]

    converted_model = gs.export_onnx(graph)
    onnx.checker.check_model(converted_model)
    onnx.shape_inference.infer_shapes(converted_model, check_type=True, strict_mode=True)


@pytest.mark.parametrize(
    "rewrite", ["_move_mul_before_qdq", "_move_transpose_before_qdq", "_insert_qdq_after_softmax"]
)
def test_rewrites_skip_when_non_matmul_consumer_exists(rewrite):
    """Every MHA rewrite must skip when the candidate tensor fans out to a non-MatMul branch."""
    x, k, y_mm, y_side, shared = _var("x"), _var("k"), _var("y_mm"), _var("y_side"), _var("shared")

    if rewrite == "_move_mul_before_qdq":
        qdq_nodes, dq_out = _qdq(x)
        producer = gs.Node(
            op="Mul",
            inputs=[dq_out, gs.Constant("c", np.array(0.5, dtype=np.float32))],
            outputs=[shared],
        )
        prelude = [*qdq_nodes, producer]
    elif rewrite == "_move_transpose_before_qdq":
        qdq_nodes, dq_out = _qdq(x)
        producer = gs.Node(
            op="Transpose", inputs=[dq_out], outputs=[shared], attrs={"perm": [1, 0]}
        )
        prelude = [*qdq_nodes, producer]
    else:
        prelude = [gs.Node(op="Softmax", inputs=[x], outputs=[shared], attrs={"axis": -1})]

    graph = _graph(
        [
            *prelude,
            gs.Node(op="MatMul", inputs=[shared, k], outputs=[y_mm]),
            gs.Node(op="Relu", inputs=[shared], outputs=[y_side]),
        ],
        [x, k],
        [y_mm, y_side],
    )
    assert getattr(FP8QuantExporter, rewrite)(graph) == 0


@pytest.mark.parametrize(
    ("torch_dtype", "high_precision_dtype", "onnx_dtype", "expected_accumulation_casts"),
    [
        pytest.param(torch.float32, "Float", onnx.TensorProto.FLOAT, 0, id="float"),
        pytest.param(torch.float16, "Half", onnx.TensorProto.FLOAT16, 4, id="half"),
        pytest.param(torch.bfloat16, "BFloat16", onnx.TensorProto.BFLOAT16, 4, id="bfloat16"),
        pytest.param(torch.float32, None, onnx.TensorProto.FLOAT, 0, id="native-float"),
        pytest.param(torch.bfloat16, None, onnx.TensorProto.BFLOAT16, 4, id="native-bfloat16"),
    ],
)
def test_fp8_mha_symbolic_preserves_accumulation_contract(
    torch_dtype, high_precision_dtype, onnx_dtype, expected_accumulation_casts
):
    """FP8-MHA supports FP32 while retaining 16-bit fusion casts."""
    shape = (1, 1, 2, 4)
    inputs = tuple(torch.ones(shape, dtype=torch_dtype) for _ in range(3))
    buffer = io.BytesIO()
    torch.onnx.export(
        _FP8MHAModule(high_precision_dtype),
        inputs,
        buffer,
        opset_version=20,
        dynamo=False,
    )

    model = onnx.load_model_from_string(buffer.getvalue())
    onnx.checker.check_model(model)
    onnx.shape_inference.infer_shapes(model, check_type=True, strict_mode=True)

    qdq_ops = {"TRT_FP8QuantizeLinear", "TRT_FP8DequantizeLinear"}
    qdq_nodes = [node for node in model.graph.node if node.op_type in qdq_ops]
    assert len(qdq_nodes) == 8

    tensor_dtype = {
        initializer.name: initializer.data_type for initializer in model.graph.initializer
    }
    for node in model.graph.node:
        if node.op_type == "Constant":
            tensor = next((attr.t for attr in node.attribute if attr.name == "value"), None)
            if tensor is not None:
                tensor_dtype[node.output[0]] = tensor.data_type
    assert all(tensor_dtype[node.input[1]] == onnx_dtype for node in qdq_nodes)

    producer_by_output = {
        output: node for node in model.graph.node for output in node.output if output
    }
    accumulation_casts = [
        node
        for node in model.graph.node
        if node.op_type == "Cast"
        and any(attr.name == "to" and attr.i == onnx.TensorProto.FLOAT for attr in node.attribute)
        and producer_by_output.get(node.input[0], onnx.NodeProto()).op_type in qdq_ops
    ]
    assert len(accumulation_casts) == expected_accumulation_casts
    back_casts = [
        node
        for node in model.graph.node
        if node.op_type == "Cast"
        and any(attr.name == "to" and attr.i == onnx_dtype for attr in node.attribute)
        and producer_by_output.get(node.input[0], onnx.NodeProto()).op_type == "MatMul"
    ]
    assert len(back_casts) == (0 if onnx_dtype == onnx.TensorProto.FLOAT else 2)
    assert all(
        value.type.tensor_type.elem_type == onnx_dtype
        for value in [*model.graph.input, *model.graph.output]
    )
