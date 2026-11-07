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

"""Check variadic axis passes and coordinated native/public layout boundaries."""

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper
from onnx.reference import ReferenceEvaluator
from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference
from onnxscript import ir

from modelopt.onnx.graph_surgery.dla_transforms import stages
from modelopt.onnx.graph_surgery.dla_transforms.common import static_shape
from modelopt.onnx.graph_surgery.dla_transforms.context import DlaGraph
from modelopt.onnx.graph_surgery.dla_transforms.layout.passes import (
    LayoutBoundaryCommit,
    SplitAxisLayout,
)


@pytest.mark.parametrize("output_count", [2, 4])
@pytest.mark.parametrize(
    ("shape", "axis", "mapped_axis"),
    [((2, 3), 1, 3), ((1, 1, 3, 2, 3), 2, 1), ((2, 3, 1, 1, 4), -4, -3)],
)
def test_split_axis_layout(output_count, shape, axis, mapped_axis):
    """Split remaps positive/negative axes through promotion and first-singleton collapse."""
    names = [f"y{i}" for i in range(output_count)]
    output_shape = list(shape)
    shape = list(shape)
    shape[axis] *= output_count
    source = helper.make_model(
        helper.make_graph(
            [helper.make_node("Split", ["x", "sizes"], names, axis=axis)],
            "split_layout",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, shape)],
            [
                helper.make_tensor_value_info(name, TensorProto.FLOAT, output_shape)
                for name in names
            ],
            [numpy_helper.from_array(np.full(output_count, 3, np.int64), "sizes")],
        ),
        opset_imports=[helper.make_opsetid("", 21)],
        ir_version=10,
    )
    context = DlaGraph(ir.serde.deserialize_model(source))
    context.begin_layout()
    transform = SplitAxisLayout(context)
    assert context.apply_pass(transform).modified
    assert transform.transformed_nodes == 1
    assert not context.apply_pass(transform).modified
    context.apply_pass(LayoutBoundaryCommit(context))
    transformed = ir.serde.serialize_model(context.model)
    context.end_layout()
    onnx.checker.check_model(transformed, full_check=True)
    split = next(node for node in transformed.graph.node if node.op_type == "Split")
    assert len(split.output) == output_count
    assert next(attr.i for attr in split.attribute if attr.name == "axis") == mapped_axis
    feeds = {"x": np.arange(np.prod(shape), dtype=np.float32).reshape(shape)}
    for expected, actual in zip(
        ReferenceEvaluator(source).run(None, feeds),
        ReferenceEvaluator(transformed).run(None, feeds),
        strict=True,
    ):
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("public_intermediate", [False, True])
def test_scalar_gather_before_fixed_reshape(public_intermediate):
    """A private scalar Gather may keep its vector view until a fixed Reshape restores layout."""
    outputs = [helper.make_tensor_value_info("y", TensorProto.FLOAT, [4, 2])]
    if public_intermediate:
        outputs.append(helper.make_tensor_value_info("gathered", TensorProto.FLOAT, [4, 2]))
    model = helper.make_model(
        helper.make_graph(
            [
                helper.make_node("Gather", ["x", "index"], ["gathered"], axis=1),
                helper.make_node("Reshape", ["gathered", "shape"], ["y"]),
            ],
            "gather_restore",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [4, 1, 2])],
            outputs,
            [
                numpy_helper.from_array(np.array(0, np.int64), "index"),
                numpy_helper.from_array(np.array([0, -1], np.int64), "shape"),
            ],
        ),
        opset_imports=[helper.make_opsetid("", 21)],
        ir_version=10,
    )
    if public_intermediate:
        with pytest.raises(ValueError):
            _lower_layout(model)
        return
    transformed = _lower_layout(model)
    onnx.checker.check_model(transformed, full_check=True)
    x = np.arange(8, dtype=np.float32).reshape(4, 1, 2)
    np.testing.assert_array_equal(
        ReferenceEvaluator(model).run(None, {"x": x})[0],
        ReferenceEvaluator(transformed).run(None, {"x": x})[0],
    )
    typed = ir.serde.deserialize_model(transformed)
    gathered = next(node for node in typed.graph if node.op_type == "Gather")
    assert static_shape(gathered.outputs[0]) == (1, 4, 1, 2)


@pytest.mark.parametrize(
    ("kind", "shared_input", "data_output", "computed_control"),
    [
        ("Split", False, False, False),
        ("Split", True, False, False),
        ("Split", True, True, False),
        ("Split", True, True, True),
        ("TopK", False, False, False),
        ("TopK", True, True, False),
    ],
)
def test_control_only_views_and_collapsed_public_boundaries(
    kind, shared_input, data_output, computed_control
):
    """Mixed data/control outputs keep their native shape and vector control operands."""
    arrays = {"squeeze_axes": np.array([1], np.int64), "y_4d": np.array(0.25, np.float32)}
    controls = np.array([1, 1] if kind == "Split" else [1], np.int64)
    selection_inputs = ["x" if shared_input else "control_x"]
    selection_attrs = {"axis": 0}
    prefix, suffix = ([], [])
    outputs = []
    if computed_control:
        arrays.update(control_a=controls, control_b=np.zeros_like(controls))
        prefix.append(helper.make_node("Add", ["control_a", "control_b"], ["selection_control"]))
        suffix.append(
            helper.make_node("Add", ["selection_control", "control_b"], ["control_result"])
        )
        outputs.append(
            helper.make_tensor_value_info("control_result", TensorProto.INT64, controls.shape)
        )
        selection_inputs.append("selection_control")
    elif kind == "TopK":
        arrays["selection_control"] = controls
        selection_inputs.append("selection_control")
    else:
        selection_attrs["num_outputs"] = 2
    if data_output:
        suffix.append(helper.make_node("Relu", ["left"], ["selected_data"]))
        outputs.append(helper.make_tensor_value_info("selected_data", TensorProto.FLOAT, [1, 3]))
    model = helper.make_model(
        helper.make_graph(
            [
                *prefix,
                helper.make_node("Flatten", ["x"], ["flat"]),
                helper.make_node("Shape", ["flat"], ["flat_shape"]),
                helper.make_node(kind, selection_inputs, ["left", "right"], **selection_attrs),
                helper.make_node("Shape", ["left"], ["left_shape"]),
                helper.make_node("Shape", ["right"], ["right_shape"]),
                helper.make_node("Add", ["x", "y_4d"], ["data_y"]),
                helper.make_node("Squeeze", ["volume", "squeeze_axes"], ["squeezed"]),
                helper.make_node("Add", ["volume", "y_4d"], ["y"]),
                *suffix,
            ],
            "native_controls_and_boundaries",
            [
                helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3]),
                helper.make_tensor_value_info("control_x", TensorProto.FLOAT, [2, 3]),
                helper.make_tensor_value_info("volume", TensorProto.FLOAT, [2, 1, 3, 2, 2]),
            ],
            outputs
            + [
                helper.make_tensor_value_info(name, TensorProto.INT64, [2])
                for name in ["flat_shape", "left_shape", "right_shape"]
            ]
            + [
                helper.make_tensor_value_info("data_y", TensorProto.FLOAT, [2, 3]),
                helper.make_tensor_value_info("squeezed", TensorProto.FLOAT, [2, 3, 2, 2]),
                helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 1, 3, 2, 2]),
            ],
            [numpy_helper.from_array(data, name) for name, data in arrays.items()],
        ),
        opset_imports=[helper.make_opsetid("", 21)],
        ir_version=10,
    )
    transformed = _lower_layout(model)
    assert transformed.graph.input == model.graph.input
    assert transformed.graph.output == model.graph.output
    assert len({value.name for value in transformed.graph.initializer}) == len(
        transformed.graph.initializer
    )
    feeds = {
        "x": np.arange(6, dtype=np.float32).reshape(2, 3) / 8 + 0.25,
        "control_x": np.arange(6, dtype=np.float32).reshape(2, 3),
        "volume": np.arange(24, dtype=np.float32).reshape(2, 1, 3, 2, 2) / 8,
    }
    for expected, actual in zip(
        ReferenceEvaluator(model).run(None, feeds),
        ReferenceEvaluator(transformed).run(None, feeds),
        strict=True,
    ):
        np.testing.assert_allclose(actual, expected, rtol=1e-06)


def test_shared_native_control_views_and_canonical_native_output():
    """Native Shape consumers reuse one restored view while a 4D native op stays valid."""
    model = helper.make_model(
        helper.make_graph(
            [
                helper.make_node("Add", ["x", "bias"], ["middle"]),
                helper.make_node("Relu", ["middle"], ["y"]),
                helper.make_node("Shape", ["middle"], ["shape_a"]),
                helper.make_node("Shape", ["middle"], ["shape_b"]),
                helper.make_node("CastLike", ["native_x", "bias"], ["norm"]),
                helper.make_node("Add", ["norm", "bias"], ["native_y"]),
            ],
            "shared_native_controls",
            [
                helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3]),
                helper.make_tensor_value_info("native_x", TensorProto.FLOAT, [1, 1, 2, 3]),
            ],
            [
                helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 3]),
                helper.make_tensor_value_info("shape_a", TensorProto.INT64, [2]),
                helper.make_tensor_value_info("shape_b", TensorProto.INT64, [2]),
                helper.make_tensor_value_info("native_y", TensorProto.FLOAT, [1, 1, 2, 3]),
            ],
            [numpy_helper.from_array(np.array(0.25, np.float32), "bias")],
        ),
        opset_imports=[helper.make_opsetid("", 21)],
        ir_version=10,
    )
    transformed = _lower_layout(model)
    shapes = [node for node in transformed.graph.node if node.op_type == "Shape"]
    assert len(shapes) == 2 and shapes[0].input[0] == shapes[1].input[0]
    data = np.arange(6, dtype=np.float32).reshape(2, 3) / 8
    feeds = {"x": data, "native_x": data.reshape(1, 1, 2, 3)}
    for expected, actual in zip(
        ReferenceEvaluator(model).run(None, feeds),
        ReferenceEvaluator(transformed).run(None, feeds),
        strict=True,
    ):
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("kind", ["Split", "TopK"])
@pytest.mark.parametrize("axis", [0, -5])
def test_selected_singleton_axis_preserves_downstream_softmax(kind, axis):
    """A new leading singleton changes the planned collapse, requiring a real output view."""
    shape = [2, 3, 2, 1, 2]
    operands, attributes, initializers = (["x"], {"axis": axis}, [])
    if kind == "Split":
        attributes["num_outputs"] = 2
    else:
        operands.append("k")
        initializers.append(numpy_helper.from_array(np.array([1], np.int64), "k"))
    model = helper.make_model(
        helper.make_graph(
            [
                helper.make_node(kind, operands, ["selected", "other"], **attributes),
                helper.make_node("Softmax", ["selected"], ["y"], axis=1),
                helper.make_node("Shape", ["other"], ["native_shape"]),
            ],
            "selected_singleton_axis",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, shape)],
            [
                helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3, 2, 1, 2]),
                helper.make_tensor_value_info("native_shape", TensorProto.INT64, [5]),
            ],
            initializers,
        ),
        opset_imports=[helper.make_opsetid("", 21)],
        ir_version=10,
    )
    transformed = _lower_layout(model)
    assert transformed.graph.input == model.graph.input
    assert transformed.graph.output == model.graph.output
    working = ir.serde.deserialize_model(transformed)
    selection = next(node for node in working.graph if node.op_type == kind)
    assert static_shape(selection.inputs[0]) == (2, 3, 2, 2)
    assert [static_shape(value) for value in selection.outputs] == [(1, 3, 2, 2)] * 2
    feeds = {"x": np.arange(24, dtype=np.float32).reshape(shape) / 8 - 1}
    for expected, actual in zip(
        ReferenceEvaluator(model).run(None, feeds),
        ReferenceEvaluator(transformed).run(None, feeds),
        strict=True,
    ):
        np.testing.assert_allclose(actual, expected, rtol=1e-06, atol=1e-07)


def _lower_layout(model):
    """Execute layout directly before the public entry point is introduced."""
    typed = SymbolicShapeInference.infer_shapes(model, auto_merge=True)
    context = DlaGraph(ir.serde.deserialize_model(typed))
    stages.layout(context)
    result = ir.serde.serialize_model(context.model)
    onnx.checker.check_model(result, full_check=True)
    return result
