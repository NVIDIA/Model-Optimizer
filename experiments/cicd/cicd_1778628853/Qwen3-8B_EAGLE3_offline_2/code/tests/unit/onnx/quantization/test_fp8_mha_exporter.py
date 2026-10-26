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

import numpy as np
import onnx_graphsurgeon as gs
import pytest

from modelopt.onnx.export.fp8_exporter import FP8QuantExporter


def _var(name):
    return gs.Variable(name, dtype=np.float32)


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


def test_insert_qdq_after_softmax_adds_fixed_scale_q_dq():
    """Softmax → MatMul picks up ``Q → DQ`` with the fixed ``1/448`` scale."""
    scores, v, y, sm_out = _var("scores"), _var("v"), _var("y"), _var("sm_out")
    sm = gs.Node(op="Softmax", inputs=[scores], outputs=[sm_out], attrs={"axis": -1})
    mm = gs.Node(op="MatMul", inputs=[sm_out, v], outputs=[y])
    graph = _graph([sm, mm], [scores, v], [y])

    assert FP8QuantExporter._insert_qdq_after_softmax(graph) == 1
    q = next(n for n in graph.nodes if n.op == "QuantizeLinear")
    assert np.isclose(float(q.inputs[1].values), 1.0 / 448.0)


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
