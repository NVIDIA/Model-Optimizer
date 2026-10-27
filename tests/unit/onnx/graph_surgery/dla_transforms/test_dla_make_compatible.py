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

"""Focused assertions for the public DLA pipeline using small synthetic inputs."""

from __future__ import annotations

import copy
import importlib

import numpy as np
import pytest
from onnx import TensorProto, helper, numpy_helper
from onnx.reference import ReferenceEvaluator

ort = pytest.importorskip("onnxruntime", reason="onnxruntime required")
_dla_pipeline = importlib.import_module("modelopt.onnx.graph_surgery.make_dla_compatible")
from modelopt.onnx.graph_surgery.make_dla_compatible import _transform_make_dla_compatible

_OPSET = 17
_MAX_IR = 9


def _vi(name, dtype, shape):
    """Declare a typed tensor for a checked graph."""
    return helper.make_tensor_value_info(name, dtype, shape)


def _clamp_ir(model):
    """Use the model IR supported by the CPU runtime."""
    model.ir_version = min(model.ir_version, _MAX_IR)


def _run_pipeline(model):
    """Deep-copy and run the DLA pipeline."""
    _clamp_ir(model)
    return _transform_make_dla_compatible(copy.deepcopy(model), verbose=False)


def test_pipeline_reorders_nodes_after_final_unsqueeze(monkeypatch):
    """The final reorder runs after the post-cleanup Unsqueeze transform."""

    def fake_unsqueeze(model):
        """Inject a deliberately unordered fixture for final reorder coverage."""
        graph = model.graph
        del graph.node[:]
        graph.node.extend(
            [
                helper.make_node(
                    "Constant",
                    [],
                    ["bias"],
                    name="const_after_unsqueeze",
                    value=numpy_helper.from_array(np.array([1.0], dtype=np.float32)),
                ),
                helper.make_node("Relu", ["X"], ["relu"], name="relu_after_unsqueeze"),
                helper.make_node("Add", ["relu", "bias"], ["Y"], name="add_after_unsqueeze"),
            ]
        )
        return model

    model = helper.make_model(
        helper.make_graph(
            [helper.make_node("Identity", ["X"], ["Y"], name="identity")],
            "g",
            [_vi("X", TensorProto.FLOAT, [1])],
            [_vi("Y", TensorProto.FLOAT, [1])],
        ),
        opset_imports=[helper.make_opsetid("", _OPSET)],
    )
    monkeypatch.setattr(
        _dla_pipeline,
        "DLA_TRANSFORMS",
        tuple(
            (name, fake_unsqueeze if name == "Unsqueeze" else transform)
            for (name, transform) in _dla_pipeline.DLA_TRANSFORMS
        ),
    )
    result = _run_pipeline(model)
    assert [node.name for node in result.graph.node] == [
        "relu_after_unsqueeze",
        "const_after_unsqueeze",
        "add_after_unsqueeze",
    ]


def test_gather_with_cast_indices_promoted_to_4d():
    """Fold constant dequantization and preserve the dynamically selected row."""
    (seq_len, embed) = (5, 3)
    x_data = np.random.default_rng(42).integers(-128, 127, (seq_len, embed), dtype=np.int8)
    inp = helper.make_model(
        helper.make_graph(
            [
                helper.make_node("DequantizeLinear", ["xq", "xs", "xz"], ["xf"], name="dq"),
                helper.make_node("Cast", ["prev"], ["pc"], to=TensorProto.INT64, name="cast"),
                helper.make_node("Gather", ["xf", "pc"], ["Y"], name="gather"),
            ],
            "g",
            [_vi("prev", TensorProto.FLOAT, [1, 1])],
            [_vi("Y", TensorProto.FLOAT, [1, 1, embed])],
            initializer=[
                numpy_helper.from_array(x_data, "xq"),
                numpy_helper.from_array(np.float32(0.003), "xs"),
                numpy_helper.from_array(np.int8(0), "xz"),
            ],
        ),
        opset_imports=[helper.make_opsetid("", _OPSET)],
    )
    candidate = _run_pipeline(inp)
    gather = next(n for n in candidate.graph.node if n.op_type == "Gather")
    assert next(a.i for a in gather.attribute if a.name == "axis") == 2
    assert all(n.op_type != "DequantizeLinear" for n in candidate.graph.node)
    cast = next(n for n in candidate.graph.node if n.op_type == "Cast")
    assert next(a.i for a in cast.attribute if a.name == "to") == TensorProto.INT32
    actual = ReferenceEvaluator(candidate).run(None, {"prev": np.array([[2]], dtype=np.float32)})[0]
    np.testing.assert_allclose(actual, x_data[[[2]]].astype(np.float32) * np.float32(0.003))


def test_matmul_add_softmax_2d():
    """MatMul [1,64]×[64,32] + Add bias + Softmax converted to Conv path."""
    w = np.random.default_rng(2).standard_normal((4, 3)).astype(np.float32)
    b = np.random.default_rng(3).standard_normal((3,)).astype(np.float32)
    inp = helper.make_model(
        helper.make_graph(
            [
                helper.make_node("MatMul", ["X", "W"], ["mm"], name="matmul"),
                helper.make_node("Add", ["mm", "B"], ["a"], name="add"),
                helper.make_node("Softmax", ["a"], ["Y"], name="sm", axis=-1),
            ],
            "g",
            [_vi("X", TensorProto.FLOAT, [1, 4])],
            [_vi("Y", TensorProto.FLOAT, [1, 3])],
            initializer=[numpy_helper.from_array(w, "W"), numpy_helper.from_array(b, "B")],
        ),
        opset_imports=[helper.make_opsetid("", _OPSET)],
    )
    candidate = _run_pipeline(inp)
    conv = next(n for n in candidate.graph.node if n.op_type == "Conv")
    weights = next(t for t in candidate.graph.initializer if t.name == conv.input[1])
    np.testing.assert_array_equal(numpy_helper.to_array(weights), w.T.reshape(3, 4, 1, 1))
    x = np.arange(4, dtype=np.float32).reshape(1, 4)
    logits = x @ w + b
    expected = np.exp(logits - logits.max(axis=-1, keepdims=True))
    expected /= expected.sum(axis=-1, keepdims=True)
    actual = ReferenceEvaluator(candidate).run(None, {"X": x})[0]
    np.testing.assert_allclose(actual, expected, rtol=1e-06, atol=1e-07)


def test_qdq_conv4d_relu_passthrough():
    """DQ(INT8→FP32) + Conv [1,3,8,8] + Relu — already 4D, should be mostly unchanged."""
    w = np.random.default_rng(4).integers(-128, 127, (16, 3, 3, 3), dtype=np.int8)
    inp = helper.make_model(
        helper.make_graph(
            [
                helper.make_node("DequantizeLinear", ["Wq", "Ws", "Wz"], ["Wf"], name="dq"),
                helper.make_node(
                    "Conv", ["X", "Wf"], ["co"], name="conv", kernel_shape=[3, 3], pads=[1, 1, 1, 1]
                ),
                helper.make_node("Relu", ["co"], ["Y"], name="relu"),
            ],
            "g",
            [_vi("X", TensorProto.FLOAT, [1, 3, 8, 8])],
            [_vi("Y", TensorProto.FLOAT, [1, 16, 8, 8])],
            initializer=[
                numpy_helper.from_array(w, "Wq"),
                numpy_helper.from_array(np.float32(0.01), "Ws"),
                numpy_helper.from_array(np.int8(0), "Wz"),
            ],
        ),
        opset_imports=[helper.make_opsetid("", _OPSET)],
    )
    candidate = _run_pipeline(inp)
    assert [n.op_type for n in candidate.graph.node] == ["DequantizeLinear", "Conv", "Relu"]
    conv = next(n for n in candidate.graph.node if n.op_type == "Conv")
    assert next(list(a.ints) for a in conv.attribute if a.name == "pads") == [1, 1, 1, 1]
    assert candidate.graph.output[0].name == "Y"
    assert [d.dim_value for d in candidate.graph.output[0].type.tensor_type.shape.dim] == [
        1,
        16,
        8,
        8,
    ]


def test_encoder_block_2d():
    """LayerNorm(2D) → MatMul → Add → Softmax → MatMul → Add (simplified attention/FFN)."""
    h = 4
    rng = np.random.default_rng(10)
    w1 = rng.standard_normal((h, h)).astype(np.float32)
    w2 = rng.standard_normal((h, h)).astype(np.float32)
    inp = helper.make_model(
        helper.make_graph(
            [
                helper.make_node(
                    "LayerNormalization",
                    ["X", "ls", "lb"],
                    ["ln"],
                    name="ln",
                    axis=-1,
                    epsilon=1e-05,
                ),
                helper.make_node("MatMul", ["ln", "W1"], ["m1"], name="mm1"),
                helper.make_node("Add", ["m1", "B1"], ["a1"], name="add1"),
                helper.make_node("Softmax", ["a1"], ["sm"], name="sm", axis=-1),
                helper.make_node("MatMul", ["sm", "W2"], ["m2"], name="mm2"),
                helper.make_node("Add", ["m2", "B2"], ["Y"], name="add2"),
            ],
            "g",
            [_vi("X", TensorProto.FLOAT, [1, h])],
            [_vi("Y", TensorProto.FLOAT, [1, h])],
            initializer=[
                numpy_helper.from_array(np.ones(h, dtype=np.float32), "ls"),
                numpy_helper.from_array(np.zeros(h, dtype=np.float32), "lb"),
                numpy_helper.from_array(w1, "W1"),
                numpy_helper.from_array(np.zeros(h, dtype=np.float32), "B1"),
                numpy_helper.from_array(w2, "W2"),
                numpy_helper.from_array(np.zeros(h, dtype=np.float32), "B2"),
            ],
        ),
        opset_imports=[helper.make_opsetid("", _OPSET)],
    )
    candidate = _run_pipeline(inp)
    assert sum(n.op_type == "Conv" for n in candidate.graph.node) == 2
    assert all(n.op_type != "MatMul" for n in candidate.graph.node)
    normalization = next(n for n in candidate.graph.node if n.op_type == "LayerNormalization")
    assert next(a.i for a in normalization.attribute if a.name == "axis") == -1
    assert candidate.graph.output[0].name == "Y"
    assert [d.dim_value for d in candidate.graph.output[0].type.tensor_type.shape.dim] == [1, h]


def test_gather_greater_where_chain():
    """Gather(2D data, dynamic idx) → Greater(threshold) → Where(mask, gathered, zeros).

    Exercises Gather 4D promotion, Greater int→float cast, Where→Mul rewrite.
    Expected: Unsqueeze + Cast + Gather + Reshape + Greater + Cast + Mul + Squeeze.
    """
    data = np.random.default_rng(30).standard_normal((10, 8)).astype(np.float32)
    inp = helper.make_model(
        helper.make_graph(
            [
                helper.make_node("Gather", ["D", "idx"], ["go"], name="gather", axis=0),
                helper.make_node("Greater", ["go", "T"], ["mask"], name="gt"),
                helper.make_node("Where", ["mask", "go", "Z"], ["Y"], name="where"),
            ],
            "g",
            [_vi("idx", TensorProto.INT32, [1])],
            [_vi("Y", TensorProto.FLOAT, [1, 8])],
            initializer=[
                numpy_helper.from_array(data, "D"),
                numpy_helper.from_array(np.float32(0.0), "T"),
                numpy_helper.from_array(np.zeros((1, 8), dtype=np.float32), "Z"),
            ],
        ),
        opset_imports=[helper.make_opsetid("", _OPSET)],
    )
    candidate = _run_pipeline(inp)
    feeds = {"idx": np.array([1], dtype=np.int32)}
    transformed_values = ReferenceEvaluator(candidate).run(None, feeds)
    np.testing.assert_array_equal(transformed_values[0], np.maximum(data[[1]], np.float32(0)))


def test_split_dual_branch_add():
    """Split [1,4,8] axis=1 into [1,2,8]+[1,2,8] → Relu + Sigmoid → Add."""
    inp = helper.make_model(
        helper.make_graph(
            [
                helper.make_node("Split", ["X", "sp"], ["a", "b"], name="split", axis=1),
                helper.make_node("Relu", ["a"], ["ra"], name="relu"),
                helper.make_node("Sigmoid", ["b"], ["sb"], name="sig"),
                helper.make_node("Add", ["ra", "sb"], ["Y"], name="add"),
            ],
            "g",
            [_vi("X", TensorProto.FLOAT, [1, 4, 8])],
            [_vi("Y", TensorProto.FLOAT, [1, 2, 8])],
            initializer=[numpy_helper.from_array(np.array([2, 2], dtype=np.int64), "sp")],
        ),
        opset_imports=[helper.make_opsetid("", _OPSET)],
    )
    candidate = _run_pipeline(inp)
    split = next(n for n in candidate.graph.node if n.op_type == "Split")
    assert next(a.i for a in split.attribute if a.name == "axis") == 2
    assert len(split.output) == 2
    actual = ReferenceEvaluator(candidate).run(None, {"X": np.zeros((1, 4, 8), dtype=np.float32)})[
        0
    ]
    np.testing.assert_array_equal(actual, np.full((1, 2, 8), np.float32(0.5)))


def test_matmul_dynamic_then_5d_reshape_qdq_reduce():
    """MatMul(both graph inputs) → 5D Reshape → Q→DQ(int8) → ReduceSum → Q→DQ → Reshape."""
    scale = np.float32(0.05)
    zp = np.int8(0)
    inp = helper.make_model(
        helper.make_graph(
            [
                helper.make_node("MatMul", ["A", "B"], ["mm"], name="matmul"),
                helper.make_node("Reshape", ["mm", "s5"], ["x5"], name="rs5d"),
                helper.make_node("QuantizeLinear", ["x5", "qs", "qz"], ["xq"], name="q1"),
                helper.make_node("DequantizeLinear", ["xq", "ds", "dz"], ["xdq"], name="dq1"),
                helper.make_node("ReduceSum", ["xdq", "ax"], ["red"], name="rsum", keepdims=0),
                helper.make_node("QuantizeLinear", ["red", "qs2", "qz2"], ["rq"], name="q2"),
                helper.make_node("DequantizeLinear", ["rq", "ds2", "dz2"], ["rdq"], name="dq2"),
                helper.make_node("Reshape", ["rdq", "s4"], ["Y"], name="rs4d"),
            ],
            "g",
            [_vi("A", TensorProto.FLOAT, [1, 4, 8]), _vi("B", TensorProto.FLOAT, [1, 8, 6])],
            [_vi("Y", TensorProto.FLOAT, [1, 4, 3, 1])],
            initializer=[
                numpy_helper.from_array(np.array([1, 4, 2, 3, 1], dtype=np.int64), "s5"),
                numpy_helper.from_array(np.array(scale), "qs"),
                numpy_helper.from_array(np.array(zp), "qz"),
                numpy_helper.from_array(np.array(scale), "ds"),
                numpy_helper.from_array(np.array(zp), "dz"),
                numpy_helper.from_array(np.array([2], dtype=np.int64), "ax"),
                numpy_helper.from_array(np.array(scale), "qs2"),
                numpy_helper.from_array(np.array(zp), "qz2"),
                numpy_helper.from_array(np.array(scale), "ds2"),
                numpy_helper.from_array(np.array(zp), "dz2"),
                numpy_helper.from_array(np.array([1, 4, 3, 1], dtype=np.int64), "s4"),
            ],
        ),
        opset_imports=[helper.make_opsetid("", _OPSET)],
    )
    candidate = _run_pipeline(inp)
    assert sum(n.op_type == "Slice" for n in candidate.graph.node) == 4
    reductions = [n for n in candidate.graph.node if n.op_type == "ReduceSum"]
    assert len(reductions) == 4
    assert all(next(a.i for a in n.attribute if a.name == "keepdims") == 1 for n in reductions)
    feeds = {"A": np.full((1, 4, 8), np.float32(0.1)), "B": np.full((1, 8, 6), np.float32(0.2))}
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    actual = ort.InferenceSession(
        candidate.SerializeToString(), options, providers=["CPUExecutionProvider"]
    ).run(None, feeds)[0]
    np.testing.assert_allclose(actual, np.full((1, 4, 3, 1), np.float32(0.3)), atol=1e-07)


def _count_op(graph, op_type):
    """Count nodes matching an operator type."""
    return sum(1 for n in graph.node if n.op_type == op_type)


def _int8_qdq_relu_model():
    """X(4D) → QuantizeLinear(int8 zp) → DequantizeLinear(int8 zp) → Relu → Y."""
    return helper.make_model(
        helper.make_graph(
            [
                helper.make_node("QuantizeLinear", ["X", "s", "zp"], ["xq"], name="q"),
                helper.make_node("DequantizeLinear", ["xq", "s", "zp"], ["xdq"], name="dq"),
                helper.make_node("Relu", ["xdq"], ["Y"], name="relu"),
            ],
            "g",
            [_vi("X", TensorProto.FLOAT, [1, 3, 8, 8])],
            [_vi("Y", TensorProto.FLOAT, [1, 3, 8, 8])],
            initializer=[
                numpy_helper.from_array(np.float32(0.02), "s"),
                numpy_helper.from_array(np.int8(0), "zp"),
            ],
        ),
        opset_imports=[helper.make_opsetid("", _OPSET)],
    )


def test_qdq_dtypes_default_keeps_int8_pair():
    """With the default dtypes (UINT16/INT16) an INT8 Q/DQ pair is left untouched."""
    model = _int8_qdq_relu_model()
    _clamp_ir(model)
    out = _transform_make_dla_compatible(copy.deepcopy(model), verbose=False)
    assert _count_op(out.graph, "QuantizeLinear") >= 1
    assert _count_op(out.graph, "DequantizeLinear") >= 1


def test_qdq_dtypes_custom_removes_int8_pair():
    """Passing ``qdq_quantized_dtypes=['int8']`` strips the INT8 Q/DQ pair."""
    model = _int8_qdq_relu_model()
    _clamp_ir(model)
    out = _transform_make_dla_compatible(
        copy.deepcopy(model), verbose=False, qdq_quantized_dtypes=["int8"]
    )
    assert _count_op(out.graph, "QuantizeLinear") == 0
    assert _count_op(out.graph, "DequantizeLinear") == 0


def test_qdq_dtypes_threaded_through_run_graph_surgery(tmp_path):
    """The kwarg is forwarded by the dispatcher to the make-dla-compatible surgery."""
    import onnx

    from modelopt.onnx.graph_surgery import run_graph_surgery

    model = _int8_qdq_relu_model()
    _clamp_ir(model)
    in_path = tmp_path / "in.onnx"
    out_path = tmp_path / "out.onnx"
    onnx.save(model, str(in_path))
    result = run_graph_surgery(
        "make-dla-compatible",
        model_path=str(in_path),
        output_path=str(out_path),
        verbose=False,
        qdq_quantized_dtypes=["int8"],
    )
    assert _count_op(result.graph, "QuantizeLinear") == 0
    assert _count_op(result.graph, "DequantizeLinear") == 0
