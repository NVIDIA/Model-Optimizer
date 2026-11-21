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

"""Behavioral tests for the complete DLA preprocessing phase."""

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper
from onnx.reference import ReferenceEvaluator
from onnxscript import ir

from modelopt.onnx.graph_surgery.dla_transforms.context import DlaGraph
from modelopt.onnx.graph_surgery.dla_transforms.preprocessing.passes import AnalyzeFp32Casts
from modelopt.onnx.graph_surgery.dla_transforms.stages import preprocessing


def _model(nodes, inputs, outputs, initializers=()):
    """Create an inferred top-level test graph at a supported operator version."""
    model = helper.make_model(
        helper.make_graph(nodes, "preprocessing", inputs, outputs, initializers),
        opset_imports=[helper.make_opsetid("", 21)],
        ir_version=10,
    )
    return onnx.shape_inference.infer_shapes(model, strict_mode=True, check_type=True)


def _apply(model, **options):
    """Exercise the real SDK rule execution and verify the exported graph."""
    working = ir.serde.deserialize_model(model)
    count = preprocessing(DlaGraph(working), **options)
    result = ir.serde.serialize_model(working)
    onnx.checker.check_model(result, full_check=True)
    return result, count


@pytest.mark.parametrize(
    ("storage", "remove_all", "producer"),
    [
        (np.int16, False, False),
        (np.uint8, False, False),
        (np.uint8, True, False),
        (np.int16, False, True),
    ],
)
def test_qdq_chains_and_shared_quantizer(storage, remove_all, producer):
    nodes = [
        helper.make_node("QuantizeLinear", ["x", "scale", "zero"], ["q1"]),
        helper.make_node("DequantizeLinear", ["q1", "scale", "zero"], ["d1"]),
        helper.make_node("QuantizeLinear", ["d1", "scale", "zero"], ["q2"]),
        helper.make_node("DequantizeLinear", ["q2", "scale", "zero"], ["y"]),
    ]
    if producer:
        nodes[0].input[0] = "activated"
        nodes.insert(0, helper.make_node("Relu", ["x"], ["activated"]))
    model = _model(
        nodes,
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [3])],
        [
            helper.make_tensor_value_info("y", TensorProto.FLOAT, [3]),
            helper.make_tensor_value_info(
                "q1", helper.np_dtype_to_tensor_dtype(np.dtype(storage)), [3]
            ),
        ],
        [
            numpy_helper.from_array(np.asarray(0.5, np.float32), "scale"),
            numpy_helper.from_array(np.asarray(0, storage), "zero"),
        ],
    )
    result, count = _apply(model, remove_all_qdq=remove_all)
    selected = storage == np.int16 or remove_all
    assert count == (2 if selected else 0)
    assert sum(node.op_type == "DequantizeLinear" for node in result.graph.node) == (
        0 if selected else 2
    )
    assert sum(node.op_type == "QuantizeLinear" for node in result.graph.node) == (
        1 if selected else 2
    )
    assert sum(node.op_type == "Identity" for node in result.graph.node) == (
        1 if selected and not producer else 0
    )
    feeds = {"x": np.asarray([0.5, 1, 1.5], np.float32)}
    for original, transformed in zip(
        ReferenceEvaluator(model).run(None, feeds),
        ReferenceEvaluator(result).run(None, feeds),
        strict=True,
    ):
        np.testing.assert_array_equal(original, transformed)


@pytest.mark.parametrize(
    ("attribute", "value", "dtype", "shape"),
    [
        ("value_float", 1.25, TensorProto.FLOAT, []),
        ("value_ints", [2, 3], TensorProto.INT64, [2]),
        ("value", numpy_helper.from_array(np.asarray([1, -2], np.int8)), TensorProto.INT8, [2]),
    ],
)
def test_constant_initializer_preserves_type_and_rank(attribute, value, dtype, shape):
    model = _model(
        [helper.make_node("Constant", [], ["y"], **{attribute: value})],
        [],
        [helper.make_tensor_value_info("y", dtype, shape)],
    )
    result, count = _apply(model)
    assert count == 1
    assert not result.graph.node
    assert result.graph.initializer[0].data_type == dtype
    assert list(result.graph.initializer[0].dims) == shape
    np.testing.assert_array_equal(
        ReferenceEvaluator(model).run(None, {})[0], ReferenceEvaluator(result).run(None, {})[0]
    )


@pytest.mark.parametrize("fixed_operand", [False, True])
def test_cast_promotion_respects_connected_operand_types(fixed_operand):
    nodes = [helper.make_node("Cast", ["x"], ["cast"], to=TensorProto.INT32)]
    nodes.append(
        helper.make_node("Add", ["cast", "fixed"], ["y"])
        if fixed_operand
        else helper.make_node("Abs", ["cast"], ["y"])
    )
    model = _model(
        nodes,
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2])],
        [helper.make_tensor_value_info("y", TensorProto.INT32, [2])],
        [numpy_helper.from_array(np.asarray([1, 2], np.int32), "fixed")] if fixed_operand else [],
    )
    result, count = _apply(model)
    assert count == (0 if fixed_operand else 1)
    expected = TensorProto.INT32 if fixed_operand else TensorProto.FLOAT
    assert result.graph.node[0].attribute[0].i == expected
    assert result.graph.output[0].type.tensor_type.elem_type == expected
    feeds = {"x": np.asarray([-2, 3], np.float32)}
    np.testing.assert_array_equal(
        ReferenceEvaluator(model).run(None, feeds)[0],
        ReferenceEvaluator(result).run(None, feeds)[0],
    )


def test_cast_preserves_shape_controls_and_topk_indices():
    model = _model(
        [
            helper.make_node("Cast", ["shape_data"], ["shape"], to=TensorProto.INT64),
            helper.make_node("Reshape", ["x", "shape"], ["reshaped"]),
            helper.make_node("TopK", ["x", "k"], ["values", "indices"], axis=1),
            helper.make_node("Cast", ["indices"], ["y"], to=TensorProto.INT32),
        ],
        [
            helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3]),
            helper.make_tensor_value_info("shape_data", TensorProto.FLOAT, [2]),
        ],
        [
            helper.make_tensor_value_info("y", TensorProto.INT32, [1, 1]),
            helper.make_tensor_value_info("reshaped", TensorProto.FLOAT, [3, 1]),
        ],
        [numpy_helper.from_array(np.asarray([1], np.int64), "k")],
    )
    result, count = _apply(model)
    assert count == 0
    feeds = {"x": np.asarray([[2, 1, 3]], np.float32), "shape_data": np.asarray([3, 1], np.float32)}
    for original, transformed in zip(
        ReferenceEvaluator(model).run(None, feeds),
        ReferenceEvaluator(result).run(None, feeds),
        strict=True,
    ):
        np.testing.assert_array_equal(original, transformed)


@pytest.mark.parametrize("per_axis", [False, True])
def test_static_dequantization_folds_shared_values_and_preserves_matrix_ownership(per_axis):
    scale = np.asarray([0.5, 0.25, 0.125] if per_axis else 0.5, np.float32)
    zeros = np.zeros(scale.shape, np.int8)
    model = _model(
        [
            helper.make_node("DequantizeLinear", ["weight", "scale", "zero"], ["dq"], axis=1),
            helper.make_node("Add", ["dq", "dq"], ["sum"]),
            helper.make_node(
                "DequantizeLinear", ["matrix", "scale", "zero"], ["matrix_dq"], axis=1
            ),
            helper.make_node("MatMul", ["x", "matrix_dq"], ["y"]),
        ],
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2])],
        [
            helper.make_tensor_value_info("sum", TensorProto.FLOAT, [2, 3]),
            helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3]),
        ],
        [
            numpy_helper.from_array(np.arange(6, dtype=np.int8).reshape(2, 3), "weight"),
            numpy_helper.from_array(np.arange(6, dtype=np.int8).reshape(2, 3), "matrix"),
            numpy_helper.from_array(scale, "scale"),
            numpy_helper.from_array(zeros, "zero"),
        ],
    )
    result, count = _apply(model)
    assert count == 1
    assert sum(node.op_type == "DequantizeLinear" for node in result.graph.node) == 1
    addition = next(node for node in result.graph.node if node.op_type == "Add")
    assert addition.input[0] == addition.input[1]
    feeds = {"x": np.asarray([[1, 2]], np.float32)}
    for original, transformed in zip(
        ReferenceEvaluator(model).run(None, feeds),
        ReferenceEvaluator(result).run(None, feeds),
        strict=True,
    ):
        np.testing.assert_array_equal(original, transformed)


def test_static_int32_dequantization_without_zero_point_preserves_public_output():
    model = _model(
        [helper.make_node("DequantizeLinear", ["data", "scale"], ["y"])],
        [],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [3])],
        [
            numpy_helper.from_array(np.asarray([-16, 3, 7], np.int32), "data"),
            numpy_helper.from_array(np.asarray(0.125, np.float32), "scale"),
        ],
    )
    result, count = _apply(model)
    assert count == 1
    assert result.graph.output[0].name == "y"
    assert all(node.op_type != "DequantizeLinear" for node in result.graph.node)
    np.testing.assert_array_equal(
        ReferenceEvaluator(model).run(None, {})[0], ReferenceEvaluator(result).run(None, {})[0]
    )


def test_selected_qdq_preserves_different_floating_endpoint_types():
    """Removing quantization still converts a FLOAT16 source to its FLOAT endpoint."""
    model = _model(
        [
            helper.make_node("QuantizeLinear", ["x", "q_scale", "zero"], ["q"]),
            helper.make_node("DequantizeLinear", ["q", "dq_scale", "zero"], ["y"]),
        ],
        [helper.make_tensor_value_info("x", TensorProto.FLOAT16, [3])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [3])],
        [
            numpy_helper.from_array(np.array(0.5, np.float16), "q_scale"),
            numpy_helper.from_array(np.array(0.5, np.float32), "dq_scale"),
            numpy_helper.from_array(np.array(0, np.int16), "zero"),
        ],
    )
    result, count = _apply(model)
    assert count == 1
    assert [node.op_type for node in result.graph.node] == ["Cast"]
    assert result.graph.output[0] == model.graph.output[0]
    feeds = {"x": np.array([0.5, 1, 1.5], np.float16)}
    np.testing.assert_array_equal(
        ReferenceEvaluator(model).run(None, feeds)[0],
        ReferenceEvaluator(result).run(None, feeds)[0],
    )


def test_cast_analysis_blocks_native_and_control_flow_regions():
    """Unsupported schema/control-flow regions cannot propagate a nearby FLOAT promotion."""
    branch = helper.make_graph(
        [helper.make_node("Identity", ["flag"], ["branch_y"])],
        "branch",
        [],
        [helper.make_tensor_value_info("branch_y", TensorProto.FLOAT, [])],
    )
    model = helper.make_model(
        helper.make_graph(
            [
                helper.make_node("Cast", ["x"], ["approved"], to=TensorProto.INT32),
                helper.make_node("Abs", ["approved"], ["arithmetic"]),
                helper.make_node("Cast", ["x"], ["native_input"], to=TensorProto.INT32),
                helper.make_node("Passthrough", ["native_input"], ["native"], domain="custom"),
                helper.make_node("Cast", ["flag"], ["condition"], to=TensorProto.BOOL),
                helper.make_node(
                    "If", ["condition"], ["choice"], then_branch=branch, else_branch=branch
                ),
                helper.make_node("Resize", ["x", "", "scales"], ["resized"]),
            ],
            "native_regions",
            [
                helper.make_tensor_value_info("x", TensorProto.FLOAT, [2]),
                helper.make_tensor_value_info("flag", TensorProto.FLOAT, []),
            ],
            [
                helper.make_tensor_value_info("arithmetic", TensorProto.INT32, [2]),
                helper.make_tensor_value_info("native", TensorProto.INT32, [2]),
                helper.make_tensor_value_info("choice", TensorProto.FLOAT, []),
                helper.make_tensor_value_info("resized", TensorProto.FLOAT, [4]),
            ],
            [numpy_helper.from_array(np.array([2], np.float32), "scales")],
        ),
        opset_imports=[helper.make_opsetid("", 21), helper.make_opsetid("custom", 1)],
        ir_version=10,
    )
    onnx.checker.check_model(model)
    working = ir.serde.deserialize_model(onnx.shape_inference.infer_shapes(model))
    before = ir.serde.serialize_model(working).SerializeToString()
    analysis = AnalyzeFp32Casts()
    assert not DlaGraph(working).apply_pass(analysis).modified
    assert {value.name for value in analysis.candidates} == {"approved"}
    assert {value.name for value in analysis.promoted} == {"approved", "arithmetic"}
    assert ir.serde.serialize_model(working).SerializeToString() == before
