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
import pytest
from onnx import TensorProto, helper, numpy_helper
from onnx.reference import ReferenceEvaluator
from onnxscript import ir

from modelopt.onnx.graph_surgery.dla_transforms.context import DlaGraph
from modelopt.onnx.graph_surgery.dla_transforms.operators import (
    INDEXING_RULES,
    MATRIX_RULES,
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
