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

"""Check documented operator lowerings through real graph execution."""

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper, numpy_helper
from onnx.reference import ReferenceEvaluator
from onnxscript import ir

from modelopt.onnx.graph_surgery.dla_transforms import stages
from modelopt.onnx.graph_surgery.dla_transforms.context import DlaGraph
from modelopt.onnx.graph_surgery.dla_transforms.operators import (
    INDEXING_RULES,
    LOGICAL_RULES,
    MATRIX_RULES,
    RECURRENT_RULES,
    weight_eligible,
)


def _apply_rules(model, classes):
    graph, views = DlaGraph(model), {}
    return graph.apply_patterns([cls.rule(graph, views) for cls in classes])


@pytest.mark.parametrize(
    ("op_type", "index_values", "runtime"),
    [
        ("Gather", np.array(1, np.int64), False),
        ("Gather", np.array([2, 0], np.int64), True),
        ("Gather", np.array([[2, 0]], np.int64), False),
        ("GatherElements", np.array([[2, 0], [1, 2]], np.int64), False),
        ("GatherElements", np.array([[2, 0], [1, 2]], np.int64), True),
    ],
)
def test_private_index_views(op_type, index_values, runtime):
    """Both rules fire without changing an exposed INT64 index owner's rank/type."""
    index_info = helper.make_tensor_value_info("indices", TensorProto.INT64, index_values.shape)
    inputs = [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3])]
    inits = []
    if runtime:
        inputs.append(index_info)
    else:
        inits.append(numpy_helper.from_array(index_values, "indices"))
    model = helper.make_model(
        helper.make_graph(
            [helper.make_node(op_type, ["x", "indices"], ["y"], axis=1)],
            "indices",
            inputs,
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, None), index_info],
            inits,
        ),
        opset_imports=[helper.make_opsetid("", 21)],
    )
    original = model.SerializeToString()
    working = ir.serde.deserialize_model(onnx.shape_inference.infer_shapes(model, strict_mode=True))
    assert _apply_rules(working, INDEXING_RULES) == 1
    transformed = onnx.shape_inference.infer_shapes(
        ir.serde.serialize_model(working), strict_mode=True
    )
    onnx.checker.check_model(transformed, full_check=True)
    assert transformed.graph.output[1].type.tensor_type.elem_type == TensorProto.INT64
    assert model.SerializeToString() == original
    feeds = {"x": np.arange(6, dtype=np.float32).reshape(2, 3)}
    if runtime:
        feeds["indices"] = index_values
    for expected, actual in zip(
        ReferenceEvaluator(model).run(None, feeds),
        ReferenceEvaluator(transformed).run(None, feeds),
        strict=True,
    ):
        np.testing.assert_array_equal(actual, expected)


def _run_matrix(model):
    working = ir.serde.deserialize_model(onnx.shape_inference.infer_shapes(model, strict_mode=True))
    count = _apply_rules(working, MATRIX_RULES)
    assert all(
        value.shape is not None and value.type is not None
        for node in working.graph
        for value in node.outputs
    )
    ir.passes.common.RemoveUnusedNodesPass()(working)
    ir.passes.common.TopologicalSortPass()(working)
    result = onnx.shape_inference.infer_shapes(ir.serde.serialize_model(working), strict_mode=True)
    onnx.checker.check_model(result, full_check=True)
    return result, count


@pytest.mark.parametrize(
    ("op_type", "shape", "weight_shape", "attrs", "bias"),
    [
        ("MatMul", [3, 4], [4, 5], {}, False),
        ("MatMul", [2, 3, 4], [1, 4, 5], {}, False),
        ("MatMul", [1, 2, 3, 4], [1, 1, 4, 5], {}, False),
        ("Gemm", [3, 4], [5, 4], {"transB": 1}, False),
        ("Gemm", [4, 3], [4, 5], {"transA": 1}, True),
    ],
)
def test_fixed_matrix_lowering(op_type, shape, weight_shape, attrs, bias):
    """Each matrix rule fires and preserves native public ranks and values."""
    weights = (np.arange(np.prod(weight_shape)).reshape(weight_shape) % 7 - 3).astype(np.float32)
    inits = [numpy_helper.from_array(weights, "weight")]
    operands = ["x", "weight"]
    if bias:
        operands.append("bias")
        inits.append(numpy_helper.from_array(np.arange(5, dtype=np.float32) / 4, "bias"))
    model = helper.make_model(
        helper.make_graph(
            [helper.make_node(op_type, operands, ["y"], **attrs)],
            "matrix",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, shape)],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, None)],
            inits,
        ),
        opset_imports=[helper.make_opsetid("", 21)],
    )
    before = model.SerializeToString()
    transformed, count = _run_matrix(model)
    assert count == 1
    assert [n.op_type for n in transformed.graph.node].count("Conv") == 1
    data = {"x": (np.arange(np.prod(shape)).reshape(shape) % 5 - 2).astype(np.float32) / 8}
    np.testing.assert_array_equal(
        ReferenceEvaluator(model).run(None, data)[0],
        ReferenceEvaluator(transformed).run(None, data)[0],
    )
    assert model.SerializeToString() == before


@pytest.mark.parametrize(
    ("kind", "shape", "weight_shape", "dtype", "attrs", "bias", "runtime"),
    [
        ("MatMul", [3], [3, 4], np.float32, {}, None, False),
        ("MatMul", [2, 1, 2, 3], [3, 4], np.float32, {}, None, False),
        ("MatMul", [2, 3], [3, 4], np.int32, {}, None, False),
        ("MatMul", [2, 2, 3], [2, 3, 4], np.float32, {}, None, False),
        ("MatMul", [2, 3], [3, 4], np.float32, {}, None, True),
        ("Gemm", [2, 3], [3, 4], np.float32, {"alpha": 0.5}, None, False),
        ("Gemm", [2, 3], [3, 4], np.float32, {}, [], False),
    ],
)
def test_unsupported_matrix_forms_remain_native(
    kind, shape, weight_shape, dtype, attrs, bias, runtime
):
    """Valid dynamic-weight, broadcast and coefficient forms retain their exact computation."""
    x = (np.arange(np.prod(shape)).reshape(shape) % 5).astype(dtype)
    weights = (np.arange(np.prod(weight_shape)).reshape(weight_shape) % 7).astype(dtype)
    inputs = [helper.make_tensor_value_info("x", helper.np_dtype_to_tensor_dtype(x.dtype), shape)]
    initializers, feeds = [], {"x": x}
    if runtime:
        inputs.append(helper.make_tensor_value_info("weight", TensorProto.FLOAT, weight_shape))
        feeds["weight"] = weights
    else:
        initializers.append(numpy_helper.from_array(weights, "weight"))
    operands = ["x", "weight"]
    if bias is not None:
        operands.append("bias")
        initializers.append(numpy_helper.from_array(np.ones(bias, dtype), "bias"))
    original = _model(
        [helper.make_node(kind, operands, ["y"], **attrs)],
        inputs,
        [helper.make_tensor_value_info("y", helper.np_dtype_to_tensor_dtype(x.dtype), None)],
        initializers,
    )
    result, count = _run_matrix(original)
    assert count == 0
    assert [node.op_type for node in result.graph.node] == [kind]
    np.testing.assert_array_equal(
        ReferenceEvaluator(result).run(None, feeds)[0],
        ReferenceEvaluator(original).run(None, feeds)[0],
    )


@pytest.mark.parametrize("missing_activation", [False, True])
def test_matrix_preflight_refuses_incomplete_ir_without_mutation(missing_activation):
    """The read-only eligibility API refuses missing operands and nonmatrix Gemm weights."""
    source = _model(
        [helper.make_node("Gemm", ["" if missing_activation else "x", "weight"], ["y"])],
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 4])],
        [
            numpy_helper.from_array(
                np.ones([3, 4] if missing_activation else [1, 3, 4], np.float32), "weight"
            )
        ],
    )
    working = ir.serde.deserialize_model(source)
    before = ir.serde.serialize_model(working).SerializeToString()
    assert not weight_eligible(working.graph[0], working.graph.initializers["weight"])
    assert ir.serde.serialize_model(working).SerializeToString() == before


@pytest.mark.parametrize("per_axis", [False, True])
def test_shared_quantized_weight_keeps_orientation_and_owner(per_axis):
    """Mixed matrix orientations get distinct views while an exposed DQ stays native."""
    scale = np.full(4, 0.25, np.float32) if per_axis else np.array(0.25, np.float32)
    q = (np.arange(16).reshape(4, 4) % 7 - 3).astype(np.int8)
    model = helper.make_model(
        helper.make_graph(
            [
                helper.make_node("DequantizeLinear", ["q", "scale", "zp"], ["weight"]),
                helper.make_node("MatMul", ["x", "weight"], ["y"]),
                helper.make_node("Gemm", ["x", "weight"], ["z"], transB=1),
            ],
            "shared_weights",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [3, 4])],
            [
                helper.make_tensor_value_info(n, TensorProto.FLOAT, s)
                for n, s in [("y", [3, 4]), ("z", [3, 4]), ("weight", [4, 4])]
            ],
            [
                numpy_helper.from_array(a, n)
                for n, a in [
                    ("q", q),
                    ("scale", scale),
                    ("zp", np.zeros_like(scale, dtype=np.int8)),
                ]
            ],
        ),
        opset_imports=[helper.make_opsetid("", 21)],
    )
    transformed, count = _run_matrix(model)
    assert count == 2
    assert len([n for n in transformed.graph.node if n.op_type == "DequantizeLinear"]) == 3
    np.testing.assert_array_equal(
        numpy_helper.to_array(next(t for t in transformed.graph.initializer if t.name == "q")), q
    )
    feeds = {"x": np.arange(12, dtype=np.float32).reshape(3, 4) / 8}
    for expected, actual in zip(
        ReferenceEvaluator(model).run(None, feeds),
        ReferenceEvaluator(transformed).run(None, feeds),
        strict=True,
    ):
        np.testing.assert_array_equal(actual, expected)


def test_blocked_quantized_matrix_keeps_native_weight_chain():
    """A valid blocked DQ matrix remains native until its kernel layout is supported."""
    original = _model(
        [
            helper.make_node(
                "DequantizeLinear", ["q", "scale", "zero"], ["weight"], axis=1, block_size=2
            ),
            helper.make_node("MatMul", ["x", "weight"], ["y"]),
        ],
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 4])],
        [
            numpy_helper.from_array(array, name)
            for name, array in [
                ("q", np.arange(12, dtype=np.int8).reshape(3, 4)),
                ("scale", np.full((3, 2), 0.125, np.float32)),
                ("zero", np.zeros((3, 2), np.int8)),
            ]
        ],
    )
    result, count = _run_matrix(original)
    assert count == 0
    assert result.graph.node == original.graph.node
    assert result.graph.initializer == original.graph.initializer


def _model(nodes, inputs, outputs, initializers=()):
    return helper.make_model(
        helper.make_graph(nodes, "logical", inputs, outputs, initializers),
        opset_imports=[helper.make_opsetid("", 21)],
    )


def _run_logical(model, *, infer=True):
    working = ir.serde.deserialize_model(
        onnx.shape_inference.infer_shapes(model, strict_mode=True) if infer else model
    )
    count = _apply_rules(working, LOGICAL_RULES)
    ir.passes.common.RemoveUnusedNodesPass()(working)
    ir.passes.common.TopologicalSortPass()(working)
    result = onnx.shape_inference.infer_shapes(ir.serde.serialize_model(working), strict_mode=True)
    onnx.checker.check_model(result, full_check=True)
    return result, count


@pytest.mark.parametrize("zero_index", [0, 1])
@pytest.mark.parametrize("static_mask", [False, True])
def test_zero_branch_where(zero_index, static_mask):
    """Both zero-branch orientations lower with fixed or runtime BOOL masks."""
    condition = np.array([[True, False, True], [False, True, False]])
    inputs = [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3])]
    inits = [numpy_helper.from_array(np.array(0, np.float32), "zero")]
    condition_info = helper.make_tensor_value_info("condition", TensorProto.BOOL, [2, 3])
    if static_mask:
        inits.append(numpy_helper.from_array(condition, "condition"))
    else:
        inputs.append(condition_info)
    branches = ["zero", "x"] if zero_index == 0 else ["x", "zero"]
    model = _model(
        [helper.make_node("Where", ["condition", *branches], ["y"])],
        inputs,
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 3])],
        inits,
    )
    transformed, count = _run_logical(model)
    assert count == 1 and not any(n.op_type == "Where" for n in transformed.graph.node)
    feeds = {"x": np.arange(6, dtype=np.float32).reshape(2, 3) - 2}
    if not static_mask:
        feeds["condition"] = condition
    np.testing.assert_array_equal(
        ReferenceEvaluator(transformed).run(None, feeds)[0],
        ReferenceEvaluator(model).run(None, feeds)[0],
    )


def test_binary_not_arithmetic():
    """A private Cast/Not region becomes float arithmetic before its numeric consumer."""
    model = _model(
        [
            helper.make_node("Cast", ["x"], ["boolean"], to=TensorProto.BOOL),
            helper.make_node("Not", ["boolean"], ["negated"]),
            helper.make_node("Cast", ["negated"], ["y"], to=TensorProto.FLOAT),
        ],
        [helper.make_tensor_value_info("x", TensorProto.BOOL, [2, 2])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 2])],
    )
    transformed, count = _run_logical(model)
    assert count == 1 and not any(n.op_type == "Not" for n in transformed.graph.node)
    feeds = {"x": np.array([[True, False], [False, True]])}
    np.testing.assert_array_equal(
        ReferenceEvaluator(transformed).run(None, feeds)[0],
        ReferenceEvaluator(model).run(None, feeds)[0],
    )


@pytest.mark.parametrize(
    ("dtype", "zero_shape", "zero"),
    [
        (np.float32, [], 1),
        (np.float16, [], 0),
        (np.float32, [2, 3], 0),
    ],
)
def test_where_preserves_nonzero_precision_and_broadcast_forms(dtype, zero_shape, zero):
    """Only the documented finite FP32 nonexpanding zero branch is lowered."""
    model = _model(
        [helper.make_node("Where", ["condition", "x", "branch"], ["y"])],
        [
            helper.make_tensor_value_info(
                "x", helper.np_dtype_to_tensor_dtype(np.dtype(dtype)), [1, 3]
            ),
            helper.make_tensor_value_info("condition", TensorProto.BOOL, [1, 3]),
        ],
        [
            helper.make_tensor_value_info(
                "y",
                helper.np_dtype_to_tensor_dtype(np.dtype(dtype)),
                [2, 3] if zero_shape else [1, 3],
            )
        ],
        [numpy_helper.from_array(np.full(zero_shape, zero, dtype), "branch")],
    )
    result, count = _run_logical(model)
    assert count == 0
    feeds = {"x": np.array([[1, 2, 3]], dtype), "condition": np.array([[True, False, True]])}
    np.testing.assert_array_equal(
        ReferenceEvaluator(result).run(None, feeds)[0],
        ReferenceEvaluator(model).run(None, feeds)[0],
    )


@pytest.mark.parametrize("public", [False, True])
def test_not_retains_boolean_consumers(public):
    """Public BOOL results and boolean arithmetic must retain Not semantics."""
    nodes = [
        helper.make_node("Cast", ["x"], ["boolean"], to=TensorProto.BOOL),
        helper.make_node("Not", ["boolean"], ["negated"]),
    ]
    if not public:
        nodes.append(helper.make_node("And", ["negated", "x"], ["y"]))
    source = _model(
        nodes,
        [helper.make_tensor_value_info("x", TensorProto.BOOL, [2])],
        [helper.make_tensor_value_info("negated" if public else "y", TensorProto.BOOL, [2])],
    )
    result, count = _run_logical(source)
    assert count == 0
    feeds = {"x": np.array([False, True])}
    np.testing.assert_array_equal(
        ReferenceEvaluator(result).run(None, feeds)[0],
        ReferenceEvaluator(source).run(None, feeds)[0],
    )


@pytest.mark.parametrize(
    ("threshold", "dtype"), [(0, np.int64), (2**24, np.int64), (0, np.float64)]
)
def test_greater_threshold_ordering(threshold, dtype):
    """Safe integer thresholds lower; rounding-sensitive thresholds stay integer."""
    model = _model(
        [helper.make_node("Greater", ["x", "threshold"], ["y"])],
        [helper.make_tensor_value_info("x", helper.np_dtype_to_tensor_dtype(np.dtype(dtype)), [3])],
        [helper.make_tensor_value_info("y", TensorProto.BOOL, [3])],
        [numpy_helper.from_array(np.array(threshold, dtype), "threshold")],
    )
    transformed, count = _run_logical(model)
    assert count == (1 if threshold == 0 and dtype != np.float64 else 0)
    feeds = {"x": np.array([threshold - 1, threshold, threshold + 1], dtype)}
    np.testing.assert_array_equal(
        ReferenceEvaluator(transformed).run(None, feeds)[0],
        ReferenceEvaluator(model).run(None, feeds)[0],
    )


def test_greater_normalizes_mixed_intermediate_operands():
    """A partially converted IR comparison keeps its FLOAT owner and converts only INT16."""
    source = _model(
        [helper.make_node("Greater", ["floating", "integer"], ["y"])],
        [
            helper.make_tensor_value_info("floating", TensorProto.FLOAT, [3]),
            helper.make_tensor_value_info("integer", TensorProto.INT16, [3]),
        ],
        [
            helper.make_tensor_value_info("y", TensorProto.BOOL, [3]),
            helper.make_tensor_value_info("floating", TensorProto.FLOAT, [3]),
        ],
    )
    result, count = _run_logical(source, infer=False)
    assert count == 1
    casts = [node for node in result.graph.node if node.op_type == "Cast"]
    assert len(casts) == 1 and list(casts[0].input) == ["integer"]
    comparison = next(node for node in result.graph.node if node.op_type == "Greater")
    assert comparison.input[0] == "floating"
    feeds = {
        "floating": np.array([-32768.5, 1.5, 32767], np.float32),
        "integer": np.array([-32768, 1, 32767], np.int16),
    }
    actual = ReferenceEvaluator(result).run(None, feeds)
    np.testing.assert_array_equal(actual[0], feeds["floating"] > feeds["integer"])
    np.testing.assert_array_equal(actual[1], feeds["floating"])


@pytest.mark.parametrize(
    ("direction", "runtime_state", "outputs", "public_states", "known_output_shape"),
    [
        ("forward", False, 3, True, True),
        ("forward", False, 1, True, True),
        ("forward", False, 3, False, True),
        ("forward", False, 2, True, True),
        ("forward", False, 1, True, False),
        ("reverse", True, 3, True, True),
        ("bidirectional", False, 3, True, True),
    ],
)
def test_lstm_unrolling(direction, runtime_state, outputs, public_states, known_output_shape):
    directions, hidden, features, steps = (2 if direction == "bidirectional" else 1), 2, 3, 3
    names, shapes = (
        ["Y", "Y_h", "Y_c"][:outputs],
        [[steps, directions, 1, hidden], [directions, 1, hidden], [directions, 1, hidden]][
            :outputs
        ],
    )
    parameters = {
        "W": ((np.arange(directions * 4 * hidden * features) % 9 - 4) / 32)
        .astype(np.float32)
        .reshape(directions, 4 * hidden, features),
        "R": ((np.arange(directions * 4 * hidden * hidden) % 7 - 3) / 32)
        .astype(np.float32)
        .reshape(directions, 4 * hidden, hidden),
        "B": ((np.arange(directions * 8 * hidden) % 5 - 2) / 64)
        .astype(np.float32)
        .reshape(directions, 8 * hidden),
        "H": np.full([directions, 1, hidden], 0.125, np.float32),
        "C": np.full([directions, 1, hidden], -0.25, np.float32),
    }
    inputs = [helper.make_tensor_value_info("X", TensorProto.FLOAT, [steps, 1, features])]
    feeds = {"X": np.arange(steps * features, dtype=np.float32).reshape(steps, 1, features) / 16}
    if runtime_state:
        for name in ("H", "C"):
            inputs.append(
                helper.make_tensor_value_info(name, TensorProto.FLOAT, [directions, 1, hidden])
            )
            feeds[name] = parameters.pop(name)
    graph = helper.make_graph(
        [
            helper.make_node(
                "LSTM",
                ["X", "W", "R", "B", "", "H", "C"],
                names,
                hidden_size=hidden,
                direction=direction,
            ),
        ],
        "lstm",
        inputs,
        [
            helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)
            for index, (name, shape) in enumerate(zip(names, shapes, strict=True))
            if public_states or index == 0
        ],
        [numpy_helper.from_array(value, name) for name, value in parameters.items()],
    )
    if directions == 2:
        graph.node.extend(
            [
                helper.make_node("Transpose", ["Y"], ["moved"], perm=[0, 2, 1, 3]),
                helper.make_node("Reshape", ["moved", "sequence_shape"], ["sequence_view"]),
                helper.make_node("Transpose", ["Y"], ["moved_again"], perm=[0, 2, 1, 3]),
                helper.make_node("Reshape", ["moved_again", "sequence_shape"], ["second_view"]),
                helper.make_node("Reshape", ["X", "sequence_shape"], ["input_view"]),
                helper.make_node("Identity", ["Y"], ["direct_sequence"]),
                helper.make_node("Transpose", ["Y"], ["reordered"], perm=[1, 0, 2, 3]),
                helper.make_node("Transpose", ["Y"], ["shared_moved"], perm=[0, 2, 1, 3]),
                helper.make_node("Identity", ["shared_moved"], ["direct_moved"]),
                helper.make_node("Reshape", ["shared_moved", "four_shape"], ["four_view"]),
            ]
        )
        graph.initializer.append(
            numpy_helper.from_array(np.array([0, 0, -1], np.int64), "sequence_shape")
        )
        graph.initializer.append(
            numpy_helper.from_array(
                np.array([steps, 1, directions, hidden], np.int64), "four_shape"
            )
        )
        graph.output.append(
            helper.make_tensor_value_info(
                "sequence_view", TensorProto.FLOAT, [steps, 1, directions * hidden]
            )
        )
        graph.output.append(
            helper.make_tensor_value_info("input_view", TensorProto.FLOAT, [steps, 1, features])
        )
        graph.output.extend(
            helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)
            for name, shape in [
                ("second_view", [steps, 1, directions * hidden]),
                ("direct_sequence", [steps, directions, 1, hidden]),
                ("reordered", [directions, steps, 1, hidden]),
                ("direct_moved", [steps, 1, directions, hidden]),
                ("four_view", [steps, 1, directions, hidden]),
            ]
        )
    source = onnx.shape_inference.infer_shapes(
        helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)], ir_version=10)
    )
    model = ir.serde.deserialize_model(source)
    if not known_output_shape:
        model.graph[0].outputs[0].shape = None
    assert _apply_rules(model, RECURRENT_RULES) == (3 if directions == 2 else 1)
    ir.passes.common.RemoveUnusedNodesPass()(model)
    ir.passes.common.TopologicalSortPass()(model)
    result = onnx.shape_inference.infer_shapes(ir.serde.serialize_model(model), strict_mode=True)
    onnx.checker.check_model(result, full_check=True)
    # ONNX 1.21's reference LSTM does not implement Y_c or bidirectional execution.
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    expected = ort.InferenceSession(
        source.SerializeToString(), options, providers=["CPUExecutionProvider"]
    ).run(None, feeds)
    actual = ReferenceEvaluator(result).run(None, feeds)
    for before, after in zip(expected, actual, strict=True):
        np.testing.assert_allclose(before, after, rtol=2e-6, atol=2e-7)
    assert all(node.op_type != "LSTM" for node in result.graph.node)
    if not public_states:
        assert sum(node.op_type == "Squeeze" for node in result.graph.node) == 0
    assert any(node.op_type == "Conv" for node in result.graph.node)
    if directions == 2:
        consumer = next(
            node
            for node in model.graph
            if node.op_type == "Transpose" and node.inputs[0].name == "Y"
        )
        assert consumer.attributes.get_ints("perm") == (0, 1, 2, 3)
        stages.layout(DlaGraph(model))
        promoted = onnx.shape_inference.infer_shapes(
            ir.serde.serialize_model(model), strict_mode=True
        )
        onnx.checker.check_model(promoted, full_check=True)
        for before, after in zip(
            expected, ReferenceEvaluator(promoted).run(None, feeds), strict=True
        ):
            np.testing.assert_allclose(before, after, rtol=2e-6, atol=2e-7)
        targets = {
            value.name
            for value in promoted.graph.initializer
            if np.array_equal(numpy_helper.to_array(value), np.array([1, steps, 1, -1], np.int64))
        }
        assert len(targets) == 1
    assert _apply_rules(model, RECURRENT_RULES) == 0
