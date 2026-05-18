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

"""End-to-end tests for :func:`_transform_make_dla_compatible`.

Each test builds an input model mimicking a real subgraph pattern, runs the full
DLA pipeline, and structurally compares the result against a hand-built expected
output graph.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pytest
from onnx import TensorProto, helper, numpy_helper

pytest.importorskip("onnxruntime", reason="onnxruntime required")

from modelopt.onnx.graph_surgery.make_dla_compatible import _transform_make_dla_compatible

sys.path.insert(0, str(Path(__file__).resolve().parent))
from graph_compare import assert_graphs_structurally_equal

_OPSET = 17
_MAX_IR = 9


def _vi(name, dtype, shape):
    return helper.make_tensor_value_info(name, dtype, shape)


def _clamp_ir(model):
    model.ir_version = min(model.ir_version, _MAX_IR)


def _run_pipeline(model):
    """Deep-copy and run the DLA pipeline."""
    _clamp_ir(model)
    return _transform_make_dla_compatible(copy.deepcopy(model), verbose=False)


# ═══════════════════════════════════════════════════════════════════════════════
# Test 1: DequantizeLinear + Cast(int64) + Gather
# ═══════════════════════════════════════════════════════════════════════════════


def test_gather_with_cast_indices_promoted_to_4d():
    """DQ + Cast(int64) + Gather(data=[4332,320]) with float32 indices.

    Expected: Unsqueeze → Cast(INT32) → Squeeze → Gather(axis=2) → Squeeze.
    """
    seq_len, embed = 4332, 320
    x_data = np.random.default_rng(42).integers(-128, 127, (seq_len, embed), dtype=np.int8)

    # Input model
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

    # Expected output
    x_float = (x_data.astype(np.int32) * np.float32(0.003)).astype(np.float32)
    exp = helper.make_model(
        helper.make_graph(
            [
                helper.make_node("Unsqueeze", ["prev", "ua"], ["u4"], name="unsq"),
                helper.make_node("Cast", ["u4"], ["ci"], to=TensorProto.INT32, name="cast"),
                helper.make_node("Squeeze", ["ci", "sa"], ["s1"], name="sqidx"),
                helper.make_node("Gather", ["data4d", "s1"], ["g4d"], axis=2, name="gather"),
                helper.make_node("Squeeze", ["g4d", "oa"], ["Y"], name="sqout"),
            ],
            "e",
            [_vi("prev", TensorProto.FLOAT, [1, 1])],
            [_vi("Y", TensorProto.FLOAT, [1, 1, embed])],
            initializer=[
                numpy_helper.from_array(x_float.reshape(1, 1, seq_len, embed), "data4d"),
                numpy_helper.from_array(np.array([0, 1], dtype=np.int64), "ua"),
                numpy_helper.from_array(np.array([0, 1, 2], dtype=np.int64), "sa"),
                numpy_helper.from_array(np.array([0], dtype=np.int64), "oa"),
            ],
        ),
        opset_imports=[helper.make_opsetid("", _OPSET)],
    )

    assert_graphs_structurally_equal(
        _run_pipeline(inp).graph,
        exp.graph,
        msg="Gather+Cast indices not promoted correctly",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Test 2: MatMul 2D + Add + Softmax → Conv conversion
# ═══════════════════════════════════════════════════════════════════════════════


def test_matmul_add_softmax_2d():
    """MatMul [1,64]×[64,32] + Add bias + Softmax converted to Conv path.

    Expected: Unsqueeze → Transpose → Conv → Transpose → Add → Softmax → Squeeze.
    """
    w = np.random.default_rng(2).standard_normal((64, 32)).astype(np.float32)
    b = np.random.default_rng(3).standard_normal((32,)).astype(np.float32)

    inp = helper.make_model(
        helper.make_graph(
            [
                helper.make_node("MatMul", ["X", "W"], ["mm"], name="matmul"),
                helper.make_node("Add", ["mm", "B"], ["a"], name="add"),
                helper.make_node("Softmax", ["a"], ["Y"], name="sm", axis=-1),
            ],
            "g",
            [_vi("X", TensorProto.FLOAT, [1, 64])],
            [_vi("Y", TensorProto.FLOAT, [1, 32])],
            initializer=[numpy_helper.from_array(w, "W"), numpy_helper.from_array(b, "B")],
        ),
        opset_imports=[helper.make_opsetid("", _OPSET)],
    )

    # Expected: MatMul→Conv with Unsqueeze/Transpose wrapping, Squeeze at end
    # Conv weight: [64,32] transposed to [32,64,1,1] kernel
    w_conv = w.T.reshape(32, 64, 1, 1)
    exp = helper.make_model(
        helper.make_graph(
            [
                helper.make_node("Unsqueeze", ["X", "ua"], ["x4"], name="unsq"),
                helper.make_node("Transpose", ["x4"], ["xt"], perm=[0, 3, 2, 1], name="tr1"),
                helper.make_node("Conv", ["xt", "Wc"], ["co"], name="conv"),
                helper.make_node("Transpose", ["co"], ["to"], perm=[0, 3, 2, 1], name="tr2"),
                helper.make_node("Add", ["to", "B2"], ["ao"], name="add"),
                helper.make_node("Softmax", ["ao"], ["sf"], axis=-1, name="sm"),
                helper.make_node("Squeeze", ["sf", "sa"], ["Y"], name="sq"),
            ],
            "e",
            [_vi("X", TensorProto.FLOAT, [1, 64])],
            [_vi("Y", TensorProto.FLOAT, [1, 32])],
            initializer=[
                numpy_helper.from_array(w_conv, "Wc"),
                numpy_helper.from_array(b, "B"),
                numpy_helper.from_array(b, "B2"),
                numpy_helper.from_array(np.array([0, 1], dtype=np.int64), "ua"),
                numpy_helper.from_array(np.array([0, 1], dtype=np.int64), "sa"),
            ],
        ),
        opset_imports=[helper.make_opsetid("", _OPSET)],
    )

    assert_graphs_structurally_equal(
        _run_pipeline(inp).graph,
        exp.graph,
        msg="MatMul+Add+Softmax 2D not converted correctly",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Test 3: ReduceSum 3D with keepdims=0
# ═══════════════════════════════════════════════════════════════════════════════


def test_reducesum_3d_keepdims0():
    """ReduceSum [2,3,4] axis=2 keepdims=0 → Unsqueeze + ReduceSum(keepdims=1) + Squeeze.

    The pipeline forces keepdims=1 internally and adds a Squeeze to restore shape.
    """
    inp = helper.make_model(
        helper.make_graph(
            [helper.make_node("ReduceSum", ["X", "ax"], ["Y"], name="rs", keepdims=0)],
            "g",
            [_vi("X", TensorProto.FLOAT, [2, 3, 4])],
            [_vi("Y", TensorProto.FLOAT, [2, 3])],
            initializer=[numpy_helper.from_array(np.array([2], dtype=np.int64), "ax")],
        ),
        opset_imports=[helper.make_opsetid("", _OPSET)],
    )

    # Expected: Unsqueeze(X→4D) → ReduceSum(keepdims=1) → Squeeze(→2D)
    exp = helper.make_model(
        helper.make_graph(
            [
                helper.make_node("Unsqueeze", ["X", "ua"], ["x4"], name="unsq"),
                helper.make_node("ReduceSum", ["x4", "ax4"], ["r4"], name="rs", keepdims=1),
                helper.make_node("Squeeze", ["r4", "sa"], ["Y"], name="sq"),
            ],
            "e",
            [_vi("X", TensorProto.FLOAT, [2, 3, 4])],
            [_vi("Y", TensorProto.FLOAT, [2, 3])],
            initializer=[
                numpy_helper.from_array(np.array([0], dtype=np.int64), "ua"),
                numpy_helper.from_array(np.array([3], dtype=np.int64), "ax4"),
                numpy_helper.from_array(np.array([0, 3], dtype=np.int64), "sa"),
            ],
        ),
        opset_imports=[helper.make_opsetid("", _OPSET)],
    )

    assert_graphs_structurally_equal(
        _run_pipeline(inp).graph,
        exp.graph,
        msg="ReduceSum 3D keepdims=0 not promoted correctly",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Test 4: Concat 2D → promoted to 4D with axis shift
# ═══════════════════════════════════════════════════════════════════════════════


def test_concat_2d():
    """Concat([2,4], [2,6], axis=1) → Unsqueeze both → Concat(axis=3) → Squeeze."""
    inp = helper.make_model(
        helper.make_graph(
            [helper.make_node("Concat", ["A", "B"], ["Y"], name="cat", axis=1)],
            "g",
            [_vi("A", TensorProto.FLOAT, [2, 4]), _vi("B", TensorProto.FLOAT, [2, 6])],
            [_vi("Y", TensorProto.FLOAT, [2, 10])],
        ),
        opset_imports=[helper.make_opsetid("", _OPSET)],
    )

    exp = helper.make_model(
        helper.make_graph(
            [
                helper.make_node("Unsqueeze", ["A", "ua"], ["a4"], name="unsqA"),
                helper.make_node("Unsqueeze", ["B", "ub"], ["b4"], name="unsqB"),
                helper.make_node("Concat", ["a4", "b4"], ["c4"], name="cat", axis=3),
                helper.make_node("Squeeze", ["c4", "sa"], ["Y"], name="sq"),
            ],
            "e",
            [_vi("A", TensorProto.FLOAT, [2, 4]), _vi("B", TensorProto.FLOAT, [2, 6])],
            [_vi("Y", TensorProto.FLOAT, [2, 10])],
            initializer=[
                numpy_helper.from_array(np.array([0, 1], dtype=np.int64), "ua"),
                numpy_helper.from_array(np.array([0, 1], dtype=np.int64), "ub"),
                numpy_helper.from_array(np.array([0, 1], dtype=np.int64), "sa"),
            ],
        ),
        opset_imports=[helper.make_opsetid("", _OPSET)],
    )

    assert_graphs_structurally_equal(
        _run_pipeline(inp).graph,
        exp.graph,
        msg="Concat 2D not promoted correctly",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Test 5: QDQ + Conv 4D + Relu — already 4D, minimal changes
# ═══════════════════════════════════════════════════════════════════════════════


def test_qdq_conv4d_relu_passthrough():
    """DQ(INT8→FP32) + Conv [1,3,8,8] + Relu — already 4D, should be mostly unchanged.

    DQ is kept (INT8 Conv weight), Conv and Relu stay as-is.
    """
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

    # Expected: DQ + Conv + Relu (same structure, already 4D)
    exp = helper.make_model(
        helper.make_graph(
            [
                helper.make_node("DequantizeLinear", ["Wq", "Ws", "Wz"], ["Wf"], name="dq"),
                helper.make_node(
                    "Conv", ["X", "Wf"], ["co"], name="conv", kernel_shape=[3, 3], pads=[1, 1, 1, 1]
                ),
                helper.make_node("Relu", ["co"], ["Y"], name="relu"),
            ],
            "e",
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

    assert_graphs_structurally_equal(
        _run_pipeline(inp).graph,
        exp.graph,
        msg="QDQ+Conv4D+Relu should be mostly unchanged",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Test 6: Encoder block — LayerNorm + 2x (MatMul→Add) + Softmax
# ═══════════════════════════════════════════════════════════════════════════════


def test_encoder_block_2d():
    """LayerNorm(2D) → MatMul → Add → Softmax → MatMul → Add (simplified attention/FFN).

    All 2D [1,64] tensors. MatMuls converted to Conv with Transpose wrapping.
    Expected: Unsqueeze → LN → Tr → Conv → Tr → Add → Softmax → Tr → Conv → Tr → Add → Squeeze.
    """
    h = 64
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
                    epsilon=1e-5,
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

    # Expected: 12 nodes — Unsq + LN + (Tr+Conv+Tr+Add) x 2 + Softmax + Sq
    w1c = w1.T.reshape(h, h, 1, 1)
    w2c = w2.T.reshape(h, h, 1, 1)
    exp = helper.make_model(
        helper.make_graph(
            [
                helper.make_node("Unsqueeze", ["X", "ua"], ["x4"], name="unsq"),
                helper.make_node(
                    "LayerNormalization",
                    ["x4", "ls", "lb"],
                    ["ln4"],
                    name="ln",
                    axis=-1,
                    epsilon=1e-5,
                ),
                helper.make_node("Transpose", ["ln4"], ["t1"], perm=[0, 3, 2, 1], name="tr1"),
                helper.make_node("Conv", ["t1", "W1c"], ["c1"], name="conv1"),
                helper.make_node("Transpose", ["c1"], ["t2"], perm=[0, 3, 2, 1], name="tr2"),
                helper.make_node("Add", ["t2", "B1"], ["a1"], name="add1"),
                helper.make_node("Softmax", ["a1"], ["sm4"], axis=-1, name="sm"),
                helper.make_node("Transpose", ["sm4"], ["t3"], perm=[0, 3, 2, 1], name="tr3"),
                helper.make_node("Conv", ["t3", "W2c"], ["c2"], name="conv2"),
                helper.make_node("Transpose", ["c2"], ["t4"], perm=[0, 3, 2, 1], name="tr4"),
                helper.make_node("Add", ["t4", "B2"], ["a2"], name="add2"),
                helper.make_node("Squeeze", ["a2", "sa"], ["Y"], name="sq"),
            ],
            "e",
            [_vi("X", TensorProto.FLOAT, [1, h])],
            [_vi("Y", TensorProto.FLOAT, [1, h])],
            initializer=[
                numpy_helper.from_array(np.ones(h, dtype=np.float32), "ls"),
                numpy_helper.from_array(np.zeros(h, dtype=np.float32), "lb"),
                numpy_helper.from_array(w1c, "W1c"),
                numpy_helper.from_array(w2c, "W2c"),
                numpy_helper.from_array(np.zeros(h, dtype=np.float32), "B1"),
                numpy_helper.from_array(np.zeros(h, dtype=np.float32), "B2"),
                numpy_helper.from_array(np.array([0, 1], dtype=np.int64), "ua"),
                numpy_helper.from_array(np.array([0, 1], dtype=np.int64), "sa"),
            ],
        ),
        opset_imports=[helper.make_opsetid("", _OPSET)],
    )

    assert_graphs_structurally_equal(
        _run_pipeline(inp).graph,
        exp.graph,
        msg="Encoder block 2D not converted correctly",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Test 7: Concat 3D + Flatten + Softmax — multi-branch with shape changes
# ═══════════════════════════════════════════════════════════════════════════════


def test_concat_flatten_softmax_3d():
    """Relu(A) + Relu(B) → Concat(axis=2) → Flatten(axis=1) → Softmax.

    Two 3D branches merged, flattened, and softmaxed. Exercises Concat axis shift,
    Flatten→Reshape conversion, and Softmax on non-4D.
    Expected: 2× Unsqueeze + 2× Relu + Concat(axis=3) + Reshape + Softmax + Squeeze.
    """
    inp = helper.make_model(
        helper.make_graph(
            [
                helper.make_node("Relu", ["A"], ["ra"], name="relu_a"),
                helper.make_node("Relu", ["B"], ["rb"], name="relu_b"),
                helper.make_node("Concat", ["ra", "rb"], ["cat"], name="cat", axis=2),
                helper.make_node("Flatten", ["cat"], ["flat"], name="flat", axis=1),
                helper.make_node("Softmax", ["flat"], ["Y"], name="sm", axis=-1),
            ],
            "g",
            [_vi("A", TensorProto.FLOAT, [2, 3, 4]), _vi("B", TensorProto.FLOAT, [2, 3, 6])],
            [_vi("Y", TensorProto.FLOAT, [2, 30])],
        ),
        opset_imports=[helper.make_opsetid("", _OPSET)],
    )

    exp = helper.make_model(
        helper.make_graph(
            [
                helper.make_node("Unsqueeze", ["A", "ua"], ["a4"], name="unsqA"),
                helper.make_node("Unsqueeze", ["B", "ub"], ["b4"], name="unsqB"),
                helper.make_node("Relu", ["a4"], ["ra4"], name="relu_a"),
                helper.make_node("Relu", ["b4"], ["rb4"], name="relu_b"),
                helper.make_node("Concat", ["ra4", "rb4"], ["cat4"], name="cat", axis=3),
                helper.make_node("Reshape", ["cat4", "rs"], ["flat4"], name="flat"),
                helper.make_node("Softmax", ["flat4"], ["sm4"], axis=-1, name="sm"),
                helper.make_node("Squeeze", ["sm4", "sa"], ["Y"], name="sq"),
            ],
            "e",
            [_vi("A", TensorProto.FLOAT, [2, 3, 4]), _vi("B", TensorProto.FLOAT, [2, 3, 6])],
            [_vi("Y", TensorProto.FLOAT, [2, 30])],
            initializer=[
                numpy_helper.from_array(np.array([0], dtype=np.int64), "ua"),
                numpy_helper.from_array(np.array([0], dtype=np.int64), "ub"),
                numpy_helper.from_array(np.array([1, 1, 2, 30], dtype=np.int64), "rs"),
                numpy_helper.from_array(np.array([0, 1], dtype=np.int64), "sa"),
            ],
        ),
        opset_imports=[helper.make_opsetid("", _OPSET)],
    )

    assert_graphs_structurally_equal(
        _run_pipeline(inp).graph,
        exp.graph,
        msg="Concat+Flatten+Softmax 3D not converted correctly",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Test 8: Gather + Greater + Where — chained op-specific transforms
# ═══════════════════════════════════════════════════════════════════════════════


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

    # Pipeline output: idx is already INT32 [1] so no Unsqueeze/Cast needed.
    # Gather uses idx directly; data is constant-folded to 4D.
    # Greater + Where(zeros) → Greater + Cast(FLOAT) + Mul
    # Squeeze at end to restore [1,8]
    exp = helper.make_model(
        helper.make_graph(
            [
                helper.make_node("Gather", ["D4", "idx"], ["go"], name="gather", axis=2),
                helper.make_node("Greater", ["go", "T"], ["mask"], name="gt"),
                helper.make_node("Cast", ["mask"], ["mf"], to=TensorProto.FLOAT, name="castm"),
                helper.make_node("Mul", ["go", "mf"], ["m4"], name="mul"),
                helper.make_node("Squeeze", ["m4", "sa"], ["Y"], name="sq"),
            ],
            "e",
            [_vi("idx", TensorProto.INT32, [1])],
            [_vi("Y", TensorProto.FLOAT, [1, 8])],
            initializer=[
                numpy_helper.from_array(data.reshape(1, 1, 10, 8), "D4"),
                numpy_helper.from_array(np.float32(0.0), "T"),
                numpy_helper.from_array(np.array([0, 1], dtype=np.int64), "sa"),
            ],
        ),
        opset_imports=[helper.make_opsetid("", _OPSET)],
    )

    assert_graphs_structurally_equal(
        _run_pipeline(inp).graph,
        exp.graph,
        msg="Gather+Greater+Where chain not transformed correctly",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Test 9: Split 3D → dual branch (Relu + Sigmoid) → Add
# ═══════════════════════════════════════════════════════════════════════════════


def test_split_dual_branch_add():
    """Split [1,4,8] axis=1 into [1,2,8]+[1,2,8] → Relu + Sigmoid → Add.

    Exercises Split axis shift, dual-branch processing, element-wise ops.
    Expected: Unsqueeze + Split(axis=2) + Relu + Sigmoid + Add + Squeeze.
    """
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

    exp = helper.make_model(
        helper.make_graph(
            [
                helper.make_node("Unsqueeze", ["X", "ua"], ["x4"], name="unsq"),
                helper.make_node("Split", ["x4", "sp"], ["a4", "b4"], name="split", axis=2),
                helper.make_node("Relu", ["a4"], ["ra4"], name="relu"),
                helper.make_node("Sigmoid", ["b4"], ["sb4"], name="sig"),
                helper.make_node("Add", ["ra4", "sb4"], ["y4"], name="add"),
                helper.make_node("Squeeze", ["y4", "sa"], ["Y"], name="sq"),
            ],
            "e",
            [_vi("X", TensorProto.FLOAT, [1, 4, 8])],
            [_vi("Y", TensorProto.FLOAT, [1, 2, 8])],
            initializer=[
                numpy_helper.from_array(np.array([2, 2], dtype=np.int64), "sp"),
                numpy_helper.from_array(np.array([0], dtype=np.int64), "ua"),
                numpy_helper.from_array(np.array([0], dtype=np.int64), "sa"),
            ],
        ),
        opset_imports=[helper.make_opsetid("", _OPSET)],
    )

    assert_graphs_structurally_equal(
        _run_pipeline(inp).graph,
        exp.graph,
        msg="Split+Relu+Sigmoid+Add 3D not converted correctly",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Test 10: MatMul(dynamic) → 5D Reshape → QDQ(int8) → ReduceSum → QDQ → Reshape
# ═══════════════════════════════════════════════════════════════════════════════


def test_matmul_dynamic_then_5d_reshape_qdq_reduce():
    """MatMul(both graph inputs) → 5D Reshape → Q→DQ(int8) → ReduceSum → Q→DQ → Reshape.

    Exercises the full chain:
    - MatMul with dynamic inputs: fallback path (Unsqueeze both → MatMul → no Squeeze
      because output feeds Reshape directly)
    - 5d_reshape_to_4d: decomposes 5D Reshape+Reduce into Slice-per-branch + Reduce + Concat
    - QDQ pairs with int8 are KEPT (not removed)
    - QDQ duplicated per-branch
    - graph_cleanup collapses the final reshape chain to Identity

    Input: A[1,4,8] × B[1,8,6] = [1,4,6] → Reshape[1,4,2,3,1]
           → Q→DQ → ReduceSum(axis=2,keepdims=0) → [1,4,3,1]
           → Q→DQ → Reshape[1,4,3,1]
    """
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

    # Expected: 24 nodes
    # Unsqueeze(A) + Unsqueeze(B) + MatMul + Reshape(merge)
    # + 4x (Slice + Q + DQ + ReduceSum) + Concat + Q + DQ + Identity
    num_branches = 4  # 5D shape [1,4,2,3,1], axis=2, split_ax=1, num_splits=shape[1]=4
    exp_nodes = [
        helper.make_node("Unsqueeze", ["A", "ua"], ["a4"], name="unsqA"),
        helper.make_node("Unsqueeze", ["B", "ub"], ["b4"], name="unsqB"),
        helper.make_node("MatMul", ["a4", "b4"], ["mm4"], name="matmul"),
        helper.make_node("Reshape", ["mm4", "msh"], ["merged"], name="merge"),
    ]
    for i in range(num_branches):
        exp_nodes.append(
            helper.make_node(
                "Slice", ["merged", f"st{i}", f"en{i}", "slax"], [f"sl{i}"], name=f"slice{i}"
            )
        )
        exp_nodes.append(
            helper.make_node("QuantizeLinear", [f"sl{i}", "qs", "qz"], [f"q{i}"], name=f"q{i}")
        )
        exp_nodes.append(
            helper.make_node("DequantizeLinear", [f"q{i}", "ds", "dz"], [f"dq{i}"], name=f"dq{i}")
        )
        exp_nodes.append(
            helper.make_node("ReduceSum", [f"dq{i}", "rax"], [f"r{i}"], name=f"rs{i}", keepdims=1)
        )
    exp_nodes.append(
        helper.make_node(
            "Concat", [f"r{i}" for i in range(num_branches)], ["cat"], name="cat", axis=1
        )
    )
    exp_nodes.append(
        helper.make_node("QuantizeLinear", ["cat", "qs2", "qz2"], ["rq"], name="q_out")
    )
    exp_nodes.append(
        helper.make_node("DequantizeLinear", ["rq", "ds2", "dz2"], ["rdq"], name="dq_out")
    )
    exp_nodes.append(helper.make_node("Identity", ["rdq"], ["Y"], name="id"))

    # chunk_size = 2*3*1 = 6, merged = [1, 4*6, 1] wait...
    # Actually: 5D [1,4,2,3,1], reduce axis=2 → split_ax = 2-1 = 1
    # merged = [1, 4*2, 3, 1] = [1, 8, 3, 1]
    # num_splits = shape[1] = 4, chunk = shape[2] = 2
    # So slices along axis 1, each chunk = 2 wide → starts 0,2,4,6
    exp = helper.make_model(
        helper.make_graph(
            exp_nodes,
            "e",
            [_vi("A", TensorProto.FLOAT, [1, 4, 8]), _vi("B", TensorProto.FLOAT, [1, 8, 6])],
            [_vi("Y", TensorProto.FLOAT, [1, 4, 3, 1])],
            initializer=[
                numpy_helper.from_array(np.array([0], dtype=np.int64), "ua"),
                numpy_helper.from_array(np.array([0], dtype=np.int64), "ub"),
                numpy_helper.from_array(np.array([1, 8, 3, 1], dtype=np.int64), "msh"),
                numpy_helper.from_array(np.array([1], dtype=np.int64), "slax"),
                *[
                    numpy_helper.from_array(np.array([i * 2], dtype=np.int64), f"st{i}")
                    for i in range(num_branches)
                ],
                *[
                    numpy_helper.from_array(np.array([(i + 1) * 2], dtype=np.int64), f"en{i}")
                    for i in range(num_branches)
                ],
                numpy_helper.from_array(np.array(scale), "qs"),
                numpy_helper.from_array(np.array(zp), "qz"),
                numpy_helper.from_array(np.array(scale), "ds"),
                numpy_helper.from_array(np.array(zp), "dz"),
                numpy_helper.from_array(np.array([-1], dtype=np.int64), "rax"),
                numpy_helper.from_array(np.array(scale), "qs2"),
                numpy_helper.from_array(np.array(zp), "qz2"),
                numpy_helper.from_array(np.array(scale), "ds2"),
                numpy_helper.from_array(np.array(zp), "dz2"),
            ],
        ),
        opset_imports=[helper.make_opsetid("", _OPSET)],
    )

    assert_graphs_structurally_equal(
        _run_pipeline(inp).graph,
        exp.graph,
        msg="MatMul(dynamic) + 5D Reshape + QDQ + ReduceSum chain not transformed correctly",
    )
