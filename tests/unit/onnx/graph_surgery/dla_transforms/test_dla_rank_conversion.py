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

"""Numerical checks for independently reviewable rank and boundary passes."""

import copy

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper, numpy_helper

from modelopt.onnx.graph_surgery.dla_transforms.dla_model_boundaries import (
    _apply_non4d_model_inputs,
    _apply_non4d_model_outputs,
)
from modelopt.onnx.graph_surgery.dla_transforms.dla_rank_conversion import (
    _apply_argmax,
    _apply_flatten,
    _apply_non4d_expand,
    _apply_non4d_initializers,
    _apply_non4d_lpnorm,
    _apply_non4d_reshape,
    _apply_non4d_slice_axis,
    _apply_non4d_tile,
    _apply_non4d_transpose,
    _apply_update_axis,
)
from modelopt.onnx.graph_surgery.make_dla_compatible import (
    DLA_TRANSFORMS,
    _transform_make_dla_compatible,
)


@pytest.mark.parametrize("operation", ["Transpose", "Reshape", "Tile", "Expand"])
def test_rank_conversion_preserves_operator_values(operation):
    """Execute native operators across explicitly deferred boundary and rank stages."""
    shape = [1, 3] if operation == "Expand" else [2, 3]
    inputs, initializers, attributes = ["x"], [], {}
    if operation == "Transpose":
        attributes["perm"] = [1, 0]
        output_shape = [3, 2]
        transform = _apply_non4d_transpose
    else:
        values = {"Reshape": [3, 2], "Tile": [2, 1], "Expand": [2, 3]}[operation]
        output_shape = {"Reshape": [3, 2], "Tile": [4, 3], "Expand": [2, 3]}[operation]
        inputs.append("parameters")
        initializers.append(numpy_helper.from_array(np.array(values, dtype=np.int64), "parameters"))
        transform = {
            "Reshape": _apply_non4d_reshape,
            "Tile": _apply_non4d_tile,
            "Expand": _apply_non4d_expand,
        }[operation]
    model = helper.make_model(
        helper.make_graph(
            [helper.make_node(operation, inputs, ["y"], name="operator", **attributes)],
            "rank",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, shape)],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, output_shape)],
            initializers,
        ),
        opset_imports=[helper.make_opsetid("", 17)],
    )
    model.ir_version = 9
    transformed = copy.deepcopy(model)
    for apply in (_apply_non4d_model_inputs, _apply_non4d_model_outputs, transform):
        options = {"defer_rank_conversion": True} if apply is _apply_non4d_model_outputs else {}
        transformed = apply(transformed, **options)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    data = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    expected = ort.InferenceSession(
        model.SerializeToString(), options, providers=["CPUExecutionProvider"]
    ).run(None, {"x": data})
    actual = ort.InferenceSession(
        transformed.SerializeToString(), options, providers=["CPUExecutionProvider"]
    ).run(None, {"x": data})
    np.testing.assert_array_equal(actual[0], expected[0])


@pytest.mark.parametrize("operation", ["Flatten", "ArgMax"])
@pytest.mark.parametrize("tail", [False, True])
def test_registered_rank_reducers_restore_native_public_outputs(operation, tail):
    """Allow deferred output restoration when its actual later conversion stage is installed."""
    shape = [1, 48] if operation == "Flatten" else [1, 4, 4]
    attributes = {"axis": 1} if operation == "Flatten" else {"axis": 1, "keepdims": 0}
    nodes = [helper.make_node(operation, ["x"], ["middle" if tail else "y"], **attributes)]
    if tail:
        nodes.append(helper.make_node("Identity", ["middle"], ["y"]))
    model = helper.make_model(
        helper.make_graph(
            nodes,
            "registered_rank_reducer",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 4, 4])],
            [
                helper.make_tensor_value_info(
                    "y", TensorProto.FLOAT if operation == "Flatten" else TensorProto.INT64, shape
                )
            ],
        ),
        opset_imports=[helper.make_opsetid("", 17)],
    )
    model.ir_version = 9
    feeds = {"x": np.arange(48, dtype=np.float32).reshape(1, 3, 4, 4)}
    expected = _execute(model, feeds)[0]
    converted = _transform_make_dla_compatible(copy.deepcopy(model), verbose=False)
    np.testing.assert_array_equal(_execute(converted, feeds)[0], expected)


@pytest.mark.parametrize("explicit", [False, True])
@pytest.mark.parametrize("batch", [1, 2])
def test_symbolic_native_transpose_keeps_its_permutation(explicit, batch):
    """Keep rank-two permutation semantics when symbolic graph inputs remain native."""
    model = helper.make_model(
        helper.make_graph(
            [helper.make_node("Transpose", ["x"], ["y"], **({"perm": [1, 0]} if explicit else {}))],
            "native_transpose",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, ["N", 3])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [3, "N"])],
        ),
        opset_imports=[helper.make_opsetid("", 17)],
    )
    model.ir_version = 9
    feeds = {"x": np.arange(batch * 3, dtype=np.float32).reshape(batch, 3)}
    expected = _execute(model, feeds)[0]
    converted = _transform_make_dla_compatible(copy.deepcopy(model), verbose=False)
    np.testing.assert_array_equal(_execute(converted, feeds)[0], expected)


@pytest.mark.parametrize(
    "operation", ["Softmax", "LogSoftmax", "LpNormalization", "ArgMax", "Slice", "Transpose"]
)
@pytest.mark.parametrize("negative", [False, True])
@pytest.mark.parametrize("tail", [False, True])
def test_symbolic_native_axis_chain_preserves_concrete_intermediate_rank(operation, negative, tail):
    """Retain native axes after symbolic data acquires concrete Reshape metadata."""
    axis = -1 if negative else 1
    controls = [numpy_helper.from_array(np.asarray([2, 3], np.int64), "target")]
    inputs = ["middle"]
    attributes = {"axis": axis}
    output = [2, 3]
    dtype = TensorProto.FLOAT
    if operation == "ArgMax":
        attributes["keepdims"] = 0
        output, dtype = [2], TensorProto.INT64
    elif operation == "Transpose":
        attributes = {} if negative else {"perm": [1, 0]}
        output = [3, 2]
    elif operation == "Slice":
        attributes = {}
        output = [2, 2]
        for name, values in [("starts", [0]), ("ends", [2]), ("axes", [axis])]:
            inputs.append(name)
            controls.append(numpy_helper.from_array(np.asarray(values, np.int64), name))
    nodes = [
        helper.make_node("Reshape", ["x", "target"], ["middle"]),
        helper.make_node(operation, inputs, ["result" if tail else "y"], **attributes),
    ]
    if tail:
        nodes.append(helper.make_node("Identity", ["result"], ["y"]))
    model = helper.make_model(
        helper.make_graph(
            nodes,
            "native_axis_chain",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, ["N", 3])],
            [helper.make_tensor_value_info("y", dtype, output)],
            controls,
            value_info=[helper.make_tensor_value_info("middle", TensorProto.FLOAT, [2, 3])],
        ),
        opset_imports=[helper.make_opsetid("", 17)],
    )
    model.ir_version = 9
    feeds = {"x": np.arange(1, 7, dtype=np.float32).reshape(2, 3)}
    expected = _execute(model, feeds)[0]
    converted = _transform_make_dla_compatible(copy.deepcopy(model), verbose=False)
    np.testing.assert_array_equal(_execute(converted, feeds)[0], expected)


def test_converted_reshape_target_preserves_skipped_native_owner():
    """Give the converted Reshape a private target when a native 4D owner is skipped."""
    model = helper.make_model(
        helper.make_graph(
            [
                helper.make_node("Reshape", ["a", "target"], ["ay"]),
                helper.make_node("Reshape", ["b", "target"], ["by"]),
            ],
            "shared_target",
            [
                helper.make_tensor_value_info("a", TensorProto.FLOAT, [2, 3]),
                helper.make_tensor_value_info("b", TensorProto.FLOAT, [5, 6, 1, 1]),
            ],
            [
                helper.make_tensor_value_info("ay", TensorProto.FLOAT, [2, 1, 1, 3]),
                helper.make_tensor_value_info("by", TensorProto.FLOAT, [5, 1, 1, 6]),
            ],
            [numpy_helper.from_array(np.array([0, 1, 1, -1], np.int64), "target")],
        ),
        opset_imports=[helper.make_opsetid("", 17)],
    )
    model.ir_version = 9
    feeds = {
        "a": np.arange(6, dtype=np.float32).reshape(2, 3),
        "b": np.arange(30, dtype=np.float32).reshape(5, 6, 1, 1),
    }
    expected = ort.InferenceSession(model.SerializeToString()).run(None, feeds)
    converted = _apply_non4d_reshape(copy.deepcopy(model))
    assert converted.graph.node[0].input[1] != converted.graph.node[1].input[1]
    onnx.checker.check_model(converted, full_check=True)
    for actual, reference in zip(
        ort.InferenceSession(converted.SerializeToString()).run(None, feeds), expected
    ):
        np.testing.assert_array_equal(actual, reference)


@pytest.mark.parametrize("shape", [[4], [3, 4], [2, 3, 4]])
def test_initializer_rank_conversion_preserves_overridable_defaults(shape):
    """Preserve a runtime input's rank and default payload while preparing fixed weights."""
    default = np.ones(shape, dtype=np.float32)
    model = helper.make_model(
        helper.make_graph(
            [helper.make_node("Add", ["x", "weight"], ["y"])],
            "runtime_default",
            [
                helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)
                for name in ("x", "weight")
            ],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, shape)],
            [numpy_helper.from_array(default, "weight")],
        ),
        opset_imports=[helper.make_opsetid("", 17)],
    )
    model.ir_version = 9
    transformed = _apply_non4d_initializers(copy.deepcopy(model))
    assert transformed.graph.initializer[0] == model.graph.initializer[0]
    onnx.checker.check_model(transformed, full_check=True)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    original = ort.InferenceSession(
        model.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )
    converted = ort.InferenceSession(
        transformed.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )
    data = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    for feeds in ({"x": data}, {"x": data, "weight": -default}):
        np.testing.assert_array_equal(converted.run(None, feeds)[0], original.run(None, feeds)[0])


def _execute(model, feeds):
    """Fully check and execute the small native-rank contract fixtures."""
    onnx.checker.check_model(model, full_check=True)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(
        model.SerializeToString(), options, providers=["CPUExecutionProvider"]
    ).run(None, feeds)


@pytest.mark.parametrize("target", [[0, -1], [0, 0, -1], [0, -1, 0]])
@pytest.mark.parametrize("collision", [False, True])
def test_reshape_resolves_copied_dimensions_across_output_aliases(target, collision):
    """Resolve copied dimensions even when output adapters allocate a suffixed private name."""
    shape = [2, 3, 4]
    output_shape = [2, 12] if len(target) == 2 else shape
    model = helper.make_model(
        helper.make_graph(
            [helper.make_node("Reshape", ["x", "target"], ["y"])],
            "copied_shape",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, shape)],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, output_shape)],
            [numpy_helper.from_array(np.asarray(target, np.int64), "target")],
        ),
        opset_imports=[helper.make_opsetid("", 17)],
    )
    model.ir_version = 9
    feeds = {"x": np.arange(24, dtype=np.float32).reshape(shape)}
    if collision:
        model.graph.input.append(helper.make_tensor_value_info("other", TensorProto.FLOAT, [1]))
        model.graph.node.append(helper.make_node("Identity", ["other"], ["y_squeeze_input"]))
        model.graph.output.append(
            helper.make_tensor_value_info("y_squeeze_input", TensorProto.FLOAT, [1])
        )
        feeds["other"] = np.asarray([17], np.float32)
    expected = _execute(model, feeds)
    transformed = _transform_make_dla_compatible(copy.deepcopy(model), verbose=False)
    for actual, reference in zip(_execute(transformed, feeds), expected):
        np.testing.assert_array_equal(actual, reference)


@pytest.mark.parametrize("axis", [0, 1, 2, -1])
@pytest.mark.parametrize("value_info", [False, True])
def test_registered_axis_update_reads_native_initializer_rank(axis, value_info):
    """Keep Concat axes correct with fixed inputs that lack optional value metadata."""
    shape = [2, 3, 4]
    output_shape = shape.copy()
    output_shape[axis] *= 2
    values = np.arange(24, dtype=np.float32).reshape(shape)
    model = helper.make_model(
        helper.make_graph(
            [helper.make_node("Concat", ["first", "second"], ["y"], axis=axis)],
            "initializer_axis",
            [],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, output_shape)],
            [numpy_helper.from_array(values, "first"), numpy_helper.from_array(-values, "second")],
        ),
        opset_imports=[helper.make_opsetid("", 17)],
    )
    model.ir_version = 9
    if value_info:
        model.graph.value_info.extend(
            helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)
            for name in ["first", "second"]
        )
    expected = _execute(model, {})[0]
    transformed = _apply_non4d_model_outputs(copy.deepcopy(model), defer_rank_conversion=True)
    for name, transform in DLA_TRANSFORMS:
        if name in {"Non4dInitializers", "UpdateAxis"}:
            transformed = transform(transformed)
    transformed.graph.ClearField("value_info")
    np.testing.assert_array_equal(_execute(transformed, {})[0], expected)


def _collapsed_normalization_model(
    shape, axis, operation="Softmax", opset=17, target=None, data_shape=None
):
    """Build a native 5D normalization between a rank-expanding and rank-restoring Reshape."""
    collapsed = shape.copy()
    collapsed.remove(1)
    model = helper.make_model(
        helper.make_graph(
            [
                helper.make_node("Reshape", ["x", "native_shape"], ["native"]),
                helper.make_node(
                    operation,
                    ["native"],
                    ["normalized"],
                    **({} if axis is None else {"axis": axis}),
                ),
                helper.make_node("Reshape", ["normalized", "output_shape"], ["y"]),
            ],
            "collapsed_normalization",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, data_shape or collapsed)],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, collapsed)],
            [
                numpy_helper.from_array(np.asarray(target or shape, np.int64), "native_shape"),
                numpy_helper.from_array(np.asarray(collapsed, np.int64), "output_shape"),
            ],
            value_info=[
                helper.make_tensor_value_info(n, TensorProto.FLOAT, shape)
                for n in ["native", "normalized"]
            ],
        ),
        opset_imports=[helper.make_opsetid("", opset)],
    )
    model.ir_version = 9
    onnx.checker.check_model(model, full_check=True)
    return model


@pytest.mark.parametrize(
    ("shape", "axis"),
    [
        ([2, 3, 4, 5, 1], 1),
        ([2, 3, 4, 5, 1], -4),
        ([2, 3, 1, 4, 5], 3),
        ([2, 3, 1, 4, 5], -2),
        ([1, 2, 3, 4, 5], 2),
        ([1, 2, 3, 4, 5], -3),
    ],
)
def test_five_dimensional_axis_tracks_the_removed_unary_position(shape, axis):
    """Execute the registered rank collapse and its downstream native axis mapping."""
    model = _collapsed_normalization_model(shape, axis)
    input_shape = [d.dim_value for d in model.graph.input[0].type.tensor_type.shape.dim]
    feeds = {"x": np.random.default_rng(20).standard_normal(input_shape).astype(np.float32)}
    expected = _execute(model, feeds)[0]
    transformed = _transform_make_dla_compatible(copy.deepcopy(model), verbose=False)
    np.testing.assert_allclose(_execute(transformed, feeds)[0], expected, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize(
    ("shape", "axis"), [([1, 2, 3, 4, 5], 0), ([1, 2, 3, 4, 5], -5), ([2, 3, 4, 5, 6], 2)]
)
def test_invalid_five_dimensional_axis_refuses_before_earlier_axis_mutation(shape, axis):
    """Validate every axis mapping before changing any previously supported operation."""
    model = helper.make_model(
        helper.make_graph(
            [
                helper.make_node("Softmax", ["first"], ["first_out"], axis=1),
                helper.make_node("Softmax", ["second"], ["second_out"], axis=axis),
            ],
            "axis_refusal",
            [
                helper.make_tensor_value_info("first", TensorProto.FLOAT, [2, 3, 4]),
                helper.make_tensor_value_info("second", TensorProto.FLOAT, shape),
            ],
            [
                helper.make_tensor_value_info("first_out", TensorProto.FLOAT, [2, 3, 4]),
                helper.make_tensor_value_info("second_out", TensorProto.FLOAT, shape),
            ],
        ),
        opset_imports=[helper.make_opsetid("", 17)],
    )
    model.ir_version = 9
    onnx.checker.check_model(model, full_check=True)
    before = model.SerializeToString()
    with pytest.raises(ValueError, match="5D axis"):
        _apply_update_axis(model)
    assert model.SerializeToString() == before


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_axis_update_recovers_native_rank_when_boundary_name_is_allocated(axis):
    """Preserve axis semantics when an unrelated public tensor occupies the adapter hint."""
    shape = [2, 3, 4]
    model = helper.make_model(
        helper.make_graph(
            [
                helper.make_node("Softmax", ["x"], ["y"], axis=axis),
                helper.make_node("Identity", ["other"], ["x_unsqueeze_input"]),
            ],
            "allocated_input",
            [
                helper.make_tensor_value_info("x", TensorProto.FLOAT, shape),
                helper.make_tensor_value_info("other", TensorProto.FLOAT, [1]),
            ],
            [
                helper.make_tensor_value_info("y", TensorProto.FLOAT, shape),
                helper.make_tensor_value_info("x_unsqueeze_input", TensorProto.FLOAT, [1]),
            ],
        ),
        opset_imports=[helper.make_opsetid("", 17)],
    )
    model.ir_version = 9
    feeds = {
        "x": np.random.default_rng(22).standard_normal(shape).astype(np.float32),
        "other": np.asarray([17], np.float32),
    }
    expected = _execute(model, feeds)
    transformed = _transform_make_dla_compatible(copy.deepcopy(model), verbose=False)
    for actual, reference in zip(_execute(transformed, feeds), expected):
        np.testing.assert_allclose(actual, reference, rtol=1e-6, atol=1e-7)


def _structural_control_model(operation, *, runtime=False, public=False):
    """Create fixed shape controls with optional runtime or public owners."""
    values = [1, 1] if operation == "Tile" else [2, 3]
    nodes = [helper.make_node(operation, ["x", "target"], ["y"])]
    inputs = [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3])]
    outputs = [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 3])]
    initializers = [numpy_helper.from_array(np.asarray(values, np.int64), "target")]
    if runtime:
        inputs.append(helper.make_tensor_value_info("target", TensorProto.INT64, [2]))
        nodes.insert(0, helper.make_node(operation, ["x", "earlier"], ["first"]))
        initializers.append(numpy_helper.from_array(np.asarray(values, np.int64), "earlier"))
    if public:
        collision = "target_" + operation.lower() + "_shape"
        initializers.append(numpy_helper.from_array(np.asarray([17], np.int64), collision))
        outputs.extend(
            [
                helper.make_tensor_value_info("target", TensorProto.INT64, [2]),
                helper.make_tensor_value_info(collision, TensorProto.INT64, [1]),
            ]
        )
    model = helper.make_model(
        helper.make_graph(nodes, "structural_control", inputs, outputs, initializers),
        opset_imports=[helper.make_opsetid("", 17)],
    )
    model.ir_version = 9
    onnx.checker.check_model(model, full_check=True)
    return model


@pytest.mark.parametrize("operation", ["Reshape", "Expand", "Tile"])
def test_runtime_shape_defaults_refuse_before_any_structural_control_mutation(operation):
    """Validate runtime shape controls before changing an earlier fixed-target node."""
    model = _structural_control_model(operation, runtime=True)
    before = model.SerializeToString()
    transform = {
        "Reshape": _apply_non4d_reshape,
        "Expand": _apply_non4d_expand,
        "Tile": _apply_non4d_tile,
    }[operation]
    with pytest.raises(ValueError, match="fixed"):
        transform(model)
    assert model.SerializeToString() == before


@pytest.mark.parametrize("operation", ["Reshape", "Expand", "Tile"])
def test_private_shape_views_preserve_public_controls_and_existing_names(operation):
    """Preserve public shape values when an allocated control hint is already occupied."""
    model = _structural_control_model(operation, public=True)
    feeds = {"x": np.arange(6, dtype=np.float32).reshape(2, 3)}
    expected = _execute(model, feeds)
    transformed = _transform_make_dla_compatible(copy.deepcopy(model), verbose=False)
    for actual, reference in zip(_execute(transformed, feeds), expected):
        np.testing.assert_array_equal(actual, reference)


def test_argmax_keeps_untouched_constant_payload_in_place():
    """Preserve an unrelated Constant payload while preparing ArgMax."""
    model = helper.make_model(
        helper.make_graph(
            [
                helper.make_node(
                    "Constant",
                    [],
                    ["constant"],
                    value=numpy_helper.from_array(np.arange(8, dtype=np.float32)),
                ),
                helper.make_node("ArgMax", ["x"], ["y"], axis=3, keepdims=1),
            ],
            "argmax_payload",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, 2, 3])],
            [
                helper.make_tensor_value_info("y", TensorProto.INT64, [1, 1, 2, 1]),
                helper.make_tensor_value_info("constant", TensorProto.FLOAT, [8]),
            ],
        ),
        opset_imports=[helper.make_opsetid("", 17)],
    )
    model.ir_version = 9
    feeds = {"x": np.arange(6, dtype=np.float32).reshape(1, 1, 2, 3)}
    expected = _execute(model, feeds)
    constant = model.graph.node[0]
    transformed = _apply_argmax(model)
    assert transformed.graph.node[0] is constant
    for actual, reference in zip(_execute(transformed, feeds), expected):
        np.testing.assert_array_equal(actual, reference)


@pytest.mark.parametrize("collision", [False, True])
@pytest.mark.parametrize("legacy_ir", [False, True])
def test_unnamed_argmax_outputs_preserve_distinct_shapes_and_public_aliases(collision, legacy_ir):
    """Restore two unnamed reductions while respecting an occupied output-adapter hint."""
    model = helper.make_model(
        helper.make_graph(
            [
                helper.make_node("ArgMax", ["x"], ["first"], axis=1, keepdims=0),
                helper.make_node("ArgMax", ["x"], ["second"], axis=2, keepdims=0),
            ],
            "argmax_alias",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3, 4])],
            [
                helper.make_tensor_value_info("first", TensorProto.INT64, [2, 4]),
                helper.make_tensor_value_info("second", TensorProto.INT64, [2, 3]),
            ],
        ),
        opset_imports=[helper.make_opsetid("", 17)],
    )
    model.ir_version = 9
    if collision:
        model.graph.initializer.append(
            numpy_helper.from_array(np.asarray([17], np.int64), "first_squeeze_input")
        )
        model.graph.output.append(
            helper.make_tensor_value_info("first_squeeze_input", TensorProto.INT64, [1])
        )
    if legacy_ir:
        model.ir_version = 3
        model.graph.input.extend(
            helper.make_tensor_value_info(t.name, t.data_type, list(t.dims))
            for t in model.graph.initializer
        )
    feeds = {"x": np.random.default_rng(23).standard_normal((2, 3, 4)).astype(np.float32)}
    expected = _execute(model, feeds)
    transformed = _transform_make_dla_compatible(copy.deepcopy(model), verbose=False)
    for actual, reference in zip(_execute(transformed, feeds), expected):
        np.testing.assert_array_equal(actual, reference)


def test_late_unknown_argmax_shape_refuses_without_changing_an_earlier_node():
    """Validate all native ArgMax dimensions before allocating or mutating adapters."""
    model = helper.make_model(
        helper.make_graph(
            [
                helper.make_node("ArgMax", ["first"], ["first_out"], axis=1, keepdims=0),
                helper.make_node("ArgMax", ["second"], ["second_out"], axis=1, keepdims=0),
            ],
            "argmax_refusal",
            [
                helper.make_tensor_value_info("first", TensorProto.FLOAT, [2, 3, 4]),
                helper.make_tensor_value_info(
                    "second", TensorProto.FLOAT, [None, None, None, None]
                ),
            ],
            [
                helper.make_tensor_value_info("first_out", TensorProto.INT64, [2, 4]),
                helper.make_tensor_value_info("second_out", TensorProto.INT64, [None, None, None]),
            ],
        ),
        opset_imports=[helper.make_opsetid("", 17)],
    )
    model.ir_version = 9
    onnx.checker.check_model(model, full_check=True)
    before = model.SerializeToString()
    with pytest.raises(ValueError, match="static"):
        _apply_argmax(model)
    assert model.SerializeToString() == before


@pytest.mark.parametrize("shape", [[2, 3], [2, 3, 4]])
def test_default_transpose_preserves_native_reversed_axes(shape):
    """Execute the implicit permutation after leading singleton promotion."""
    model = helper.make_model(
        helper.make_graph(
            [helper.make_node("Transpose", ["x"], ["y"])],
            "default_transpose",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, shape)],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, list(reversed(shape)))],
        ),
        opset_imports=[helper.make_opsetid("", 17)],
    )
    model.ir_version = 9
    feeds = {"x": np.arange(np.prod(shape), dtype=np.float32).reshape(shape)}
    expected = _execute(model, feeds)
    transformed = _transform_make_dla_compatible(copy.deepcopy(model), verbose=False)
    np.testing.assert_array_equal(_execute(transformed, feeds)[0], expected[0])


def test_unnamed_flatten_nodes_keep_distinct_batch_shapes_and_public_controls():
    """Allocate private targets for unrelated unnamed Flatten operations."""
    model = helper.make_model(
        helper.make_graph(
            [
                helper.make_node("Flatten", ["x"], ["first"]),
                helper.make_node("Flatten", ["z"], ["second"]),
            ],
            "flatten_names",
            [
                helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3, 4]),
                helper.make_tensor_value_info("z", TensorProto.FLOAT, [3, 2, 4]),
            ],
            [
                helper.make_tensor_value_info("first", TensorProto.FLOAT, [2, 12]),
                helper.make_tensor_value_info("second", TensorProto.FLOAT, [3, 8]),
                helper.make_tensor_value_info("_reshape_shape", TensorProto.INT64, [1]),
            ],
            [numpy_helper.from_array(np.asarray([17], np.int64), "_reshape_shape")],
        ),
        opset_imports=[helper.make_opsetid("", 17)],
    )
    model.ir_version = 9
    feeds = {
        "x": np.arange(24, dtype=np.float32).reshape(2, 3, 4),
        "z": np.arange(24, dtype=np.float32).reshape(3, 2, 4),
    }
    expected = _execute(model, feeds)
    transformed = _transform_make_dla_compatible(copy.deepcopy(model), verbose=False)
    for actual, reference in zip(_execute(transformed, feeds), expected):
        np.testing.assert_array_equal(actual, reference)


@pytest.mark.parametrize("operation", ["Flatten", "Transpose"])
def test_late_invalid_layout_refuses_before_earlier_node_mutation(operation):
    """Plan every layout before mutating earlier valid operators."""
    shape = [None, 3, 4] if operation == "Flatten" else [2, 3, 4, 5, 6]
    first_attrs = {} if operation == "Flatten" else {"perm": [1, 0]}
    second_attrs = {} if operation == "Flatten" else {"perm": [4, 3, 2, 1, 0]}
    model = helper.make_model(
        helper.make_graph(
            [
                helper.make_node(operation, ["x"], ["first"], **first_attrs),
                helper.make_node(operation, ["z"], ["second"], **second_attrs),
            ],
            "layout_refusal",
            [
                helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3]),
                helper.make_tensor_value_info("z", TensorProto.FLOAT, shape),
            ],
            [
                helper.make_tensor_value_info(
                    "first", TensorProto.FLOAT, [2, 3] if operation == "Flatten" else [3, 2]
                ),
                helper.make_tensor_value_info(
                    "second",
                    TensorProto.FLOAT,
                    [None, 12] if operation == "Flatten" else list(reversed(shape)),
                ),
            ],
        ),
        opset_imports=[helper.make_opsetid("", 17)],
    )
    model.ir_version = 9
    onnx.checker.check_model(model, full_check=True)
    before = model.SerializeToString()
    with pytest.raises(ValueError, match="known"):
        {"Flatten": _apply_flatten, "Transpose": _apply_non4d_transpose}[operation](model)
    assert model.SerializeToString() == before


@pytest.mark.parametrize("operation", ["Softmax", "LogSoftmax"])
def test_implicit_legacy_axis_tracks_removed_leading_unary_dimension(operation):
    """Read the imported default axis after an actual registered Reshape collapse."""
    model = _collapsed_normalization_model([1, 2, 3, 4, 5], None, operation, 11)
    feeds = {"x": np.random.default_rng(25).standard_normal((2, 3, 4, 5)).astype(np.float32)}
    expected = _execute(model, feeds)[0]
    transformed = _transform_make_dla_compatible(copy.deepcopy(model), verbose=False)
    np.testing.assert_allclose(_execute(transformed, feeds)[0], expected, rtol=1e-6, atol=1e-7)


def test_implicit_axis_on_removed_dimension_refuses_without_mutation():
    """Reject an omitted default axis when its unary dimension would disappear."""
    model = helper.make_model(
        helper.make_graph(
            [helper.make_node("Softmax", ["x"], ["y"])],
            "implicit_refusal",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3, 4, 5, 1])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 3, 4, 5, 1])],
        ),
        opset_imports=[helper.make_opsetid("", 17)],
    )
    before = model.SerializeToString()
    with pytest.raises(ValueError, match="5D axis"):
        _apply_update_axis(model)
    assert model.SerializeToString() == before


@pytest.mark.parametrize("owner", ["function", "capture"])
@pytest.mark.parametrize("shared", [False, True])
def test_opaque_initializer_owners_preserve_rank_or_refuse_shared_layout(owner, shared):
    """Protect fixed native weights owned by an opaque function or nested scope."""
    shape = [2, 3, 4]
    initializers = [numpy_helper.from_array(np.ones(shape, np.float32), "weight")]
    functions = []
    opsets = [helper.make_opsetid("", 17)]
    if owner == "function":
        functions.append(
            helper.make_function(
                "opaque",
                "Layout",
                ["x"],
                ["y"],
                [helper.make_node("Identity", ["x"], ["y"])],
                [helper.make_opsetid("", 17)],
            )
        )
        nodes = [helper.make_node("Layout", ["weight"], ["y"], domain="opaque")]
        opsets.append(helper.make_opsetid("opaque", 1))
    else:
        branch = helper.make_graph(
            [helper.make_node("Identity", ["weight"], ["out"])],
            "branch",
            [],
            [helper.make_tensor_value_info("out", TensorProto.FLOAT, shape)],
        )
        initializers.append(numpy_helper.from_array(np.asarray(True), "condition"))
        nodes = [
            helper.make_node("If", ["condition"], ["y"], then_branch=branch, else_branch=branch)
        ]
    outputs = [helper.make_tensor_value_info("y", TensorProto.FLOAT, shape)]
    if shared:
        nodes.insert(0, helper.make_node("Identity", ["earlier"], ["early"]))
        nodes.append(helper.make_node("Add", ["weight", "weight"], ["sum"]))
        initializers.insert(0, numpy_helper.from_array(np.ones(shape, np.float32), "earlier"))
        outputs.append(helper.make_tensor_value_info("sum", TensorProto.FLOAT, shape))
    model = helper.make_model(
        helper.make_graph(nodes, "opaque_weight", [], outputs, initializers),
        opset_imports=opsets,
        functions=functions,
    )
    model.ir_version = 9
    onnx.checker.check_model(model, full_check=True)
    before = model.SerializeToString()
    if shared:
        with pytest.raises(ValueError, match="opaque or captured"):
            _apply_non4d_initializers(model)
    else:
        _apply_non4d_initializers(model)
    assert model.SerializeToString() == before


@pytest.mark.parametrize(
    "operation", ["Flatten", "Transpose", "LpNormalization", "Softmax", "Slice"]
)
def test_rank_passes_leave_model_local_operators_unchanged(operation):
    """Skip opaque local functions whose names match standard layout operators."""
    function = helper.make_function(
        "opaque",
        operation,
        ["x"],
        ["y"],
        [helper.make_node("Identity", ["x"], ["y"])],
        [helper.make_opsetid("", 17)],
    )
    model = helper.make_model(
        helper.make_graph(
            [helper.make_node(operation, ["x"], ["y"], domain="opaque")],
            "opaque_rank",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3, 4])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 3, 4])],
        ),
        opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("opaque", 1)],
        functions=[function],
    )
    model.ir_version = 9
    onnx.checker.check_model(model, full_check=True)
    before = model.SerializeToString()
    for transform in [
        _apply_flatten,
        _apply_non4d_transpose,
        _apply_non4d_lpnorm,
        _apply_update_axis,
        _apply_non4d_slice_axis,
    ]:
        transform(model)
    assert model.SerializeToString() == before


@pytest.mark.parametrize("runtime", [False, True])
def test_shared_slice_axes_preserve_private_views_and_transactional_runtime_refusal(runtime):
    """Reuse fixed axis views and reject later runtime axes before changing earlier nodes."""
    nodes = [
        helper.make_node("Slice", ["x", "starts", "ends", "axes"], ["first"]),
        helper.make_node(
            "Slice", ["x", "starts", "ends", "runtime_axes" if runtime else "axes"], ["second"]
        ),
    ]
    initializers = [
        numpy_helper.from_array(np.asarray(v, np.int64), n)
        for n, v in [("starts", [0]), ("ends", [2]), ("axes", [1]), ("axes_axes_only", [17])]
    ]
    inputs = [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3, 4])]
    if runtime:
        inputs.append(helper.make_tensor_value_info("runtime_axes", TensorProto.INT64, [1]))
        initializers.append(numpy_helper.from_array(np.asarray([1], np.int64), "runtime_axes"))
    model = helper.make_model(
        helper.make_graph(
            nodes,
            "slice_controls",
            inputs,
            [
                helper.make_tensor_value_info(n, TensorProto.FLOAT, [2, 2, 4])
                for n in ["first", "second"]
            ]
            + [helper.make_tensor_value_info("axes_axes_only", TensorProto.INT64, [1])],
            initializers,
        ),
        opset_imports=[helper.make_opsetid("", 17)],
    )
    model.ir_version = 9
    onnx.checker.check_model(model, full_check=True)
    if runtime:
        before = model.SerializeToString()
        with pytest.raises(ValueError, match="fixed"):
            _apply_non4d_slice_axis(model)
        assert model.SerializeToString() == before
    else:
        feeds = {"x": np.arange(24, dtype=np.float32).reshape(2, 3, 4)}
        expected = _execute(model, feeds)
        transformed = _transform_make_dla_compatible(copy.deepcopy(model), verbose=False)
        slices = [n for n in transformed.graph.node if n.op_type == "Slice"]
        assert slices[0].input[3] == slices[1].input[3]
        for actual, reference in zip(_execute(transformed, feeds), expected):
            np.testing.assert_array_equal(actual, reference)


def test_implicit_split_axis_preserves_rank_three_values():
    """Map the standard default Split axis during leading singleton promotion."""
    shape = [2, 3, 4]
    model = helper.make_model(
        helper.make_graph(
            [helper.make_node("Split", ["x"], ["first", "second"])],
            "split_axis",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, shape)],
            [
                helper.make_tensor_value_info(n, TensorProto.FLOAT, [1, 3, 4])
                for n in ["first", "second"]
            ],
        ),
        opset_imports=[helper.make_opsetid("", 13)],
    )
    model.ir_version = 9
    feeds = {"x": np.arange(24, dtype=np.float32).reshape(shape)}
    expected = _execute(model, feeds)
    transformed = _transform_make_dla_compatible(copy.deepcopy(model), verbose=False)
    for actual, reference in zip(_execute(transformed, feeds), expected):
        np.testing.assert_array_equal(actual, reference)


@pytest.mark.parametrize(
    ("rank", "empty", "runtime", "fixed_data"),
    [
        (2, False, False, False),
        (3, True, False, False),
        (3, False, True, False),
        (3, True, True, False),
        (3, False, False, True),
    ],
)
def test_implicit_slice_axes_preserve_registered_data_layouts(rank, empty, runtime, fixed_data):
    """Derive implicit axes from the fixed starts length before promoting data."""
    shape = [2, 3] if rank == 2 else [2, 3, 4]
    output_shape = [2, 2] if rank == 2 else [2, 2, 4]
    names = ["x", "starts", "ends", ""] if empty else ["x", "starts", "ends"]
    initializers = [numpy_helper.from_array(np.asarray([2, 3], np.int64), "ends")]
    inputs = [] if fixed_data else [helper.make_tensor_value_info("x", TensorProto.FLOAT, shape)]
    data = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    feeds = {} if fixed_data else {"x": data}
    if fixed_data:
        initializers.append(numpy_helper.from_array(data, "x"))
    if runtime:
        inputs.append(helper.make_tensor_value_info("starts", TensorProto.INT64, [2]))
        feeds["starts"] = np.asarray([0, 1], np.int64)
    else:
        initializers.append(numpy_helper.from_array(np.asarray([0, 1], np.int64), "starts"))
    model = helper.make_model(
        helper.make_graph(
            [helper.make_node("Slice", names, ["y"])],
            "implicit_slice",
            inputs,
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, output_shape)],
            initializers,
        ),
        opset_imports=[helper.make_opsetid("", 17)],
    )
    model.ir_version = 9
    onnx.checker.check_model(model, full_check=True)
    expected = _execute(model, feeds)[0]
    transformed = _transform_make_dla_compatible(copy.deepcopy(model), verbose=False)
    np.testing.assert_array_equal(_execute(transformed, feeds)[0], expected)


def test_unknown_implicit_slice_length_refuses_before_earlier_mutation():
    """Reject an unknown runtime starts length before allocating an earlier fixed axis view."""
    model = helper.make_model(
        helper.make_graph(
            [
                helper.make_node("Slice", ["x", "fixed", "ends"], ["first"]),
                helper.make_node("Slice", ["x", "dynamic", "ends"], ["second"]),
            ],
            "unknown_starts",
            [
                helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3]),
                helper.make_tensor_value_info("dynamic", TensorProto.INT64, [None]),
            ],
            [
                helper.make_tensor_value_info(n, TensorProto.FLOAT, [None, None])
                for n in ["first", "second"]
            ],
            [
                numpy_helper.from_array(np.asarray([0, 0], np.int64), "fixed"),
                numpy_helper.from_array(np.asarray([2, 3], np.int64), "ends"),
            ],
        ),
        opset_imports=[helper.make_opsetid("", 17)],
    )
    model.ir_version = 9
    before = model.SerializeToString()
    with pytest.raises(ValueError, match="known starts length"):
        _apply_non4d_slice_axis(model)
    assert model.SerializeToString() == before


@pytest.mark.parametrize("producer", ["input", "Unsqueeze"])
def test_native_five_dimensional_axes_refuse_without_a_collapsed_reshape(producer):
    """Keep earlier axes unchanged when a later native input never lost a dimension."""
    shape = [1, 2, 3, 4, 5]
    nodes = [helper.make_node("Softmax", ["first"], ["first_out"], axis=1)]
    inputs = [helper.make_tensor_value_info("first", TensorProto.FLOAT, [2, 3, 4])]
    initializers = []
    if producer == "input":
        inputs.append(helper.make_tensor_value_info("native", TensorProto.FLOAT, shape))
    else:
        inputs.append(helper.make_tensor_value_info("x", TensorProto.FLOAT, shape[1:]))
        initializers.append(numpy_helper.from_array(np.asarray([0], np.int64), "axes"))
        nodes.append(helper.make_node("Unsqueeze", ["x", "axes"], ["native"]))
    nodes.append(helper.make_node("Softmax", ["native"], ["y"], axis=2))
    model = helper.make_model(
        helper.make_graph(
            nodes,
            "native_five",
            inputs,
            [
                helper.make_tensor_value_info("first_out", TensorProto.FLOAT, [2, 3, 4]),
                helper.make_tensor_value_info("y", TensorProto.FLOAT, shape),
            ],
            initializers,
            value_info=[helper.make_tensor_value_info("native", TensorProto.FLOAT, shape)],
        ),
        opset_imports=[helper.make_opsetid("", 17)],
    )
    model.ir_version = 9
    onnx.checker.check_model(model, full_check=True)
    before = model.SerializeToString()
    with pytest.raises(ValueError, match="collapsed Reshape"):
        _apply_update_axis(model)
    assert model.SerializeToString() == before


@pytest.mark.parametrize(
    ("shape", "target", "data_shape", "axis"),
    [
        ([1, 2, 1, 3, 4], [-1, 2, 1, 3, 4], [1, 2, 3, 4], 1),
        ([1, 2, 4, 3, 5], [1, 2, 0, 3, 5], [2, 3, 4, 5], 2),
    ],
)
def test_five_dimensional_axis_tracks_the_actually_removed_target_dimension(
    shape, target, data_shape, axis
):
    """Respect inferred and copied dimensions when locating the singleton actually removed."""
    model = _collapsed_normalization_model(shape, axis, target=target, data_shape=data_shape)
    feeds = {"x": np.random.default_rng(29).standard_normal(data_shape).astype(np.float32)}
    expected = _execute(model, feeds)[0]
    transformed = _transform_make_dla_compatible(copy.deepcopy(model), verbose=False)
    np.testing.assert_allclose(_execute(transformed, feeds)[0], expected, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("operation", ["ArgMax", "LpNormalization"])
@pytest.mark.parametrize("axis", [2, -3])
@pytest.mark.parametrize("collapse", [False, True])
def test_specialized_axes_follow_only_actually_collapsed_reshapes(operation, axis, collapse):
    """Remap specialized axes after a singleton collapse while keeping untouched native 5D axes."""
    shape = [1, 2, 3, 4, 5]
    data_shape = shape[1:] if collapse else shape
    reduced = [1, 2, 4, 5]
    output_shape = reduced if operation == "ArgMax" else data_shape
    inits = []
    nodes = []
    source = "x"
    if collapse:
        inits.append(numpy_helper.from_array(np.array(shape, np.int64), "native_shape"))
        nodes.append(helper.make_node("Reshape", ["x", "native_shape"], ["native"]))
        source = "native"
    nodes.append(
        helper.make_node(
            operation,
            [source],
            ["result" if collapse else "y"],
            axis=axis,
            **({"keepdims": 0} if operation == "ArgMax" else {}),
        )
    )
    if collapse:
        inits.append(numpy_helper.from_array(np.array(output_shape, np.int64), "output_shape"))
        nodes.append(helper.make_node("Reshape", ["result", "output_shape"], ["y"]))
    model = helper.make_model(
        helper.make_graph(
            nodes,
            "specialized_axes",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, data_shape)],
            [
                helper.make_tensor_value_info(
                    "y",
                    TensorProto.INT64 if operation == "ArgMax" else TensorProto.FLOAT,
                    output_shape,
                )
            ],
            inits,
        ),
        opset_imports=[helper.make_opsetid("", 17)],
    )
    model.ir_version = 9
    model = onnx.shape_inference.infer_shapes(model)
    feeds = {"x": np.random.default_rng(73).standard_normal(data_shape).astype(np.float32)}
    expected = _execute(model, feeds)[0]
    converted = copy.deepcopy(model)
    if collapse:
        converted = _transform_make_dla_compatible(converted, verbose=False)
    else:
        converted = (_apply_argmax if operation == "ArgMax" else _apply_non4d_lpnorm)(converted)
    np.testing.assert_allclose(_execute(converted, feeds)[0], expected, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("shape", [[2, 3], [2, 3, 4]])
def test_public_initializer_outputs_keep_their_native_dimensions(shape):
    """Preserve fixed public values that have no internal layout-changing consumer."""
    data = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    model = helper.make_model(
        helper.make_graph(
            [],
            "public_weight",
            [],
            [helper.make_tensor_value_info("weight", TensorProto.FLOAT, shape)],
            [numpy_helper.from_array(data, "weight")],
        ),
        opset_imports=[helper.make_opsetid("", 17)],
    )
    model.ir_version = 9
    expected = _execute(model, {})[0]
    converted = _transform_make_dla_compatible(copy.deepcopy(model), verbose=False)
    np.testing.assert_array_equal(_execute(converted, {})[0], expected)


@pytest.mark.parametrize("target", [[0, 1, 1, -1], [1, 0, 1, -1]])
def test_four_dimensional_reshape_targets_resolve_native_copied_dimensions(target):
    """Resolve copied positions before input promotion shifts native dimensions."""
    output_shape = [2, 1, 1, 3] if target[0] == 0 else [1, 3, 1, 2]
    model = helper.make_model(
        helper.make_graph(
            [helper.make_node("Reshape", ["x", "target"], ["y"])],
            "copied_four",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, output_shape)],
            [numpy_helper.from_array(np.asarray(target, np.int64), "target")],
        ),
        opset_imports=[helper.make_opsetid("", 17)],
    )
    model.ir_version = 9
    feeds = {"x": np.arange(6, dtype=np.float32).reshape(2, 3)}
    expected = _execute(model, feeds)[0]
    converted = _transform_make_dla_compatible(copy.deepcopy(model), verbose=False)
    np.testing.assert_array_equal(_execute(converted, feeds)[0], expected)


def test_runtime_four_dimensional_reshape_target_preserves_runtime_overrides():
    """Leave runtime targets native when neither input nor output rank needs promotion."""
    model = helper.make_model(
        helper.make_graph(
            [helper.make_node("Reshape", ["x", "target"], ["y"])],
            "runtime_four",
            [
                helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, 2, 3]),
                helper.make_tensor_value_info("target", TensorProto.INT64, [4]),
            ],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [None, None, None, None])],
            [numpy_helper.from_array(np.asarray([1, 1, 2, 3], np.int64), "target")],
        ),
        opset_imports=[helper.make_opsetid("", 17)],
    )
    model.ir_version = 9
    feeds = {
        "x": np.arange(6, dtype=np.float32).reshape(1, 1, 2, 3),
        "target": np.asarray([1, 2, 1, 3], np.int64),
    }
    expected = _execute(model, feeds)[0]
    converted = _transform_make_dla_compatible(copy.deepcopy(model), verbose=False)
    np.testing.assert_array_equal(_execute(converted, feeds)[0], expected)


@pytest.mark.parametrize(
    ("producer", "copied"),
    [("Reshape", False), ("Reshape", True), ("Squeeze", False), ("Unsqueeze", False)],
)
def test_native_index_views_survive_rank_preparation(producer, copied):
    """Retain scalar and vector control ranks after upstream structural nodes are promoted."""
    source_shape = [] if producer == "Unsqueeze" else ([3, 1] if copied else [1, 3])
    count = 1 if producer == "Unsqueeze" else 3
    values = (
        np.asarray(2, np.int64)
        if not source_shape
        else np.arange(3, dtype=np.int64).reshape(source_shape)
    )
    controls = np.asarray([0] if producer != "Reshape" or copied else [-1], np.int64)
    nodes = [
        helper.make_node(producer, ["source", "control"], ["indices"]),
        helper.make_node("Gather", ["x", "indices"], ["y"], axis=0),
    ]
    model = helper.make_model(
        helper.make_graph(
            nodes,
            "native_index_views",
            [
                helper.make_tensor_value_info("x", TensorProto.FLOAT, [5, 4]),
                helper.make_tensor_value_info("source", TensorProto.INT64, source_shape),
            ],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [count, 4])],
            [numpy_helper.from_array(controls, "control")],
            value_info=[helper.make_tensor_value_info("indices", TensorProto.INT64, [count])],
        ),
        opset_imports=[helper.make_opsetid("", 17)],
    )
    model.ir_version = 9
    onnx.checker.check_model(model, full_check=True)
    feeds = {"x": np.arange(20, dtype=np.float32).reshape(5, 4), "source": values}
    expected = _execute(model, feeds)[0]
    converted = _transform_make_dla_compatible(copy.deepcopy(model), verbose=False)
    np.testing.assert_array_equal(_execute(converted, feeds)[0], expected)


@pytest.mark.parametrize("operation", ["Tile", "Expand", "Reshape"])
@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("static_output", [False, True])
@pytest.mark.parametrize("tail", [False, True])
def test_symbolic_structural_controls_keep_native_data_rank(operation, batch, static_output, tail):
    """Retain symbolic native layouts, including fixed public outputs and unary readers."""
    target = {
        "Tile": [0, 1] if static_output else [2, 1],
        "Expand": [2, 3] if static_output else [1, 3],
        "Reshape": [3, 1] if static_output else [3, -1],
    }[operation]
    if static_output and operation == "Reshape" and batch != 1:
        target = [3, 2]
    output_shape = (
        {"Tile": [0, 3], "Expand": [2, 3], "Reshape": [3, batch]}[operation]
        if static_output
        else ([3, "N"] if operation == "Reshape" else ["M", 3])
    )
    nodes = [helper.make_node(operation, ["x", "control"], ["middle" if tail else "y"])]
    if tail:
        nodes.append(helper.make_node("Identity", ["middle"], ["y"]))
    model = helper.make_model(
        helper.make_graph(
            nodes,
            "symbolic_control",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, ["N", 3])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, output_shape)],
            [numpy_helper.from_array(np.array(target, np.int64), "control")],
        ),
        opset_imports=[helper.make_opsetid("", 17)],
    )
    model.ir_version = 9
    feeds = {"x": np.arange(batch * 3, dtype=np.float32).reshape(batch, 3)}
    expected = _execute(model, feeds)[0]
    original_control = model.graph.initializer[0].SerializeToString()
    converted = _transform_make_dla_compatible(copy.deepcopy(model), verbose=False)
    np.testing.assert_array_equal(_execute(converted, feeds)[0], expected)
    assert converted.graph.initializer[0].SerializeToString() == original_control


@pytest.mark.parametrize("operation", ["Tile", "Expand", "Reshape"])
def test_symbolic_native_control_chain_preserves_concrete_intermediate_rank(operation):
    """Keep native controls after a symbolic input acquires concrete intermediate metadata."""
    control = {"Tile": [1, 2], "Expand": [2, 3], "Reshape": [3, 2]}[operation]
    output = {"Tile": [2, 6], "Expand": [2, 3], "Reshape": [3, 2]}[operation]
    model = helper.make_model(
        helper.make_graph(
            [
                helper.make_node("Reshape", ["x", "first_shape"], ["middle"]),
                helper.make_node(operation, ["middle", "control"], ["y"]),
            ],
            "native_control_chain",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, ["N", 3])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, output)],
            [
                numpy_helper.from_array(np.array([2, 3], np.int64), "first_shape"),
                numpy_helper.from_array(np.array(control, np.int64), "control"),
            ],
            value_info=[helper.make_tensor_value_info("middle", TensorProto.FLOAT, [2, 3])],
        ),
        opset_imports=[helper.make_opsetid("", 17)],
    )
    model.ir_version = 9
    feeds = {"x": np.arange(6, dtype=np.float32).reshape(2, 3)}
    expected = _execute(model, feeds)[0]
    converted = _transform_make_dla_compatible(copy.deepcopy(model), verbose=False)
    np.testing.assert_array_equal(_execute(converted, feeds)[0], expected)
    for actual, original in zip(converted.graph.initializer, model.graph.initializer):
        assert actual.SerializeToString() == original.SerializeToString()


@pytest.mark.parametrize("operation", ["Tile", "Reshape"])
def test_runtime_slice_extents_retain_promoted_control_layouts(operation):
    """Follow already-promoted data through unknown Slice extents before adapting controls."""
    values = [1, 2, 1] if operation == "Tile" else [4, -1]
    model = helper.make_model(
        helper.make_graph(
            [
                helper.make_node("Slice", ["x", "starts", "ends"], ["middle"]),
                helper.make_node(operation, ["middle", "control"], ["y"]),
            ],
            "promoted_slice_control",
            [
                helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3, 4]),
                helper.make_tensor_value_info("starts", TensorProto.INT64, [2]),
            ],
            [
                helper.make_tensor_value_info(
                    "y", TensorProto.FLOAT, ["A", "B", 4] if operation == "Tile" else ["A", "B"]
                )
            ],
            [
                numpy_helper.from_array(np.array([2, 3], np.int64), "ends"),
                numpy_helper.from_array(np.array(values, np.int64), "control"),
            ],
        ),
        opset_imports=[helper.make_opsetid("", 17)],
    )
    model.ir_version = 9
    feeds = {
        "x": np.arange(24, dtype=np.float32).reshape(2, 3, 4),
        "starts": np.array([0, 1], np.int64),
    }
    expected = _execute(model, feeds)[0]
    converted = _transform_make_dla_compatible(copy.deepcopy(model), verbose=False)
    np.testing.assert_array_equal(_execute(converted, feeds)[0], expected)


@pytest.mark.parametrize(
    "operation", ["Softmax", "LogSoftmax", "LpNormalization", "Transpose", "Slice"]
)
def test_runtime_slice_extents_retain_promoted_axes(operation):
    """Map native axes using known rank when runtime Slice extents remain symbolic."""
    inputs, attributes = ["middle"], {"axis": 1}
    if operation == "Transpose":
        attributes = {"perm": [1, 2, 0]}
    elif operation == "Slice":
        inputs += ["starts", "ends", "axes"]
        attributes = {}
    model = helper.make_model(
        helper.make_graph(
            [
                helper.make_node("Slice", ["x", "starts", "ends"], ["middle"]),
                helper.make_node(operation, inputs, ["y"], **attributes),
            ],
            "promoted_slice_axes",
            [
                helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3, 4]),
                helper.make_tensor_value_info("starts", TensorProto.INT64, [2]),
            ],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, ["A", "B", "C"])],
            [
                numpy_helper.from_array(np.array([2, 3], np.int64), "ends"),
                numpy_helper.from_array(np.array([1, 2], np.int64), "axes"),
            ],
        ),
        opset_imports=[helper.make_opsetid("", 17)],
        ir_version=9,
    )
    feeds = {
        "x": np.arange(24, dtype=np.float32).reshape(2, 3, 4) / 16,
        "starts": np.array([0, 1], np.int64),
    }
    expected = _execute(model, feeds)[0]
    converted = _transform_make_dla_compatible(copy.deepcopy(model), verbose=False)
    np.testing.assert_allclose(_execute(converted, feeds)[0], expected, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("initializer", [False, True])
@pytest.mark.parametrize("tail", [False, True])
def test_rank_one_tile_data_and_repeats_remain_native(initializer, tail):
    """Execute rank-one Tile for graph inputs and fixed data without padding repeats."""
    data = np.array([1, 2, 3], np.float32)
    nodes = [helper.make_node("Tile", ["x", "repeats"], ["tiled" if tail else "y"])]
    if tail:
        nodes.append(helper.make_node("Relu", ["tiled"], ["y"]))
    model = helper.make_model(
        helper.make_graph(
            nodes,
            "native_tile",
            [] if initializer else [helper.make_tensor_value_info("x", TensorProto.FLOAT, [3])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [6])],
            [numpy_helper.from_array(np.array([2], np.int64), "repeats")]
            + ([numpy_helper.from_array(data, "x")] if initializer else []),
        ),
        opset_imports=[helper.make_opsetid("", 17)],
        ir_version=9,
    )
    feeds = {} if initializer else {"x": data}
    expected = _execute(model, feeds)[0]
    converted = _transform_make_dla_compatible(copy.deepcopy(model), verbose=False)
    np.testing.assert_array_equal(_execute(converted, feeds)[0], expected)


@pytest.mark.parametrize("operation", ["Tile", "Expand", "Reshape"])
def test_unknown_structural_data_rank_refuses_before_mutation(operation):
    """Refuse unresolved intermediate ranks before allocating any control view."""
    values = [2, 1] if operation == "Tile" else [2, 3]
    model = helper.make_model(
        helper.make_graph(
            [
                helper.make_node("Identity", ["x"], ["hidden"]),
                helper.make_node(operation, ["hidden", "control"], ["y"]),
            ],
            "unknown_control_rank",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3])],
            [
                helper.make_tensor_value_info(
                    "y", TensorProto.FLOAT, [4, 3] if operation == "Tile" else [2, 3]
                )
            ],
            [numpy_helper.from_array(np.array(values, np.int64), "control")],
        ),
        opset_imports=[helper.make_opsetid("", 17)],
    )
    model.ir_version = 9
    _execute(model, {"x": np.arange(6, dtype=np.float32).reshape(2, 3)})
    before = model.SerializeToString()
    callback = {
        "Tile": _apply_non4d_tile,
        "Expand": _apply_non4d_expand,
        "Reshape": _apply_non4d_reshape,
    }[operation]
    with pytest.raises(ValueError, match="input rank"):
        callback(model)
    assert model.SerializeToString() == before


@pytest.mark.parametrize("batch", [1, 2])
def test_symbolic_input_output_remains_native(batch):
    """Leave symbolic input ranks intact instead of restoring an unpromoted output."""
    model = helper.make_model(
        helper.make_graph(
            [helper.make_node("Softmax", ["x"], ["y"], axis=-1)],
            "symbolic_native",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, ["N", 3, 4])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, ["N", 3, 4])],
        ),
        opset_imports=[helper.make_opsetid("", 17)],
    )
    model.ir_version = 9
    feeds = {"x": np.arange(batch * 12, dtype=np.float32).reshape(batch, 3, 4)}
    expected = _execute(model, feeds)
    transformed = _transform_make_dla_compatible(copy.deepcopy(model), verbose=False)
    np.testing.assert_allclose(_execute(transformed, feeds)[0], expected[0])


@pytest.mark.parametrize("operation", ["ArgMin", "Hardmax", "LayerNormalization"])
def test_axis_update_preserves_native_four_dimensional_operators(operation):
    """Allow axis-bearing operators that already consume their final rank."""
    shape = [2, 3, 4, 5]
    inputs = ["x"]
    initializers = []
    output_shape = [2, 1, 4, 5] if operation == "ArgMin" else shape
    if operation == "LayerNormalization":
        inputs.append("scale")
        initializers.append(numpy_helper.from_array(np.ones(shape[1:], np.float32), "scale"))
    model = helper.make_model(
        helper.make_graph(
            [helper.make_node(operation, inputs, ["y"], axis=1)],
            "native_four_dimensional",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, shape)],
            [
                helper.make_tensor_value_info(
                    "y",
                    TensorProto.INT64 if operation == "ArgMin" else TensorProto.FLOAT,
                    output_shape,
                )
            ],
            initializers,
        ),
        opset_imports=[helper.make_opsetid("", 17)],
    )
    model.ir_version = 9
    feeds = {"x": np.arange(120, dtype=np.float32).reshape(shape)}
    expected = _execute(model, feeds)
    transformed = _apply_update_axis(copy.deepcopy(model))
    assert transformed == model
    np.testing.assert_array_equal(_execute(transformed, feeds)[0], expected[0])


@pytest.mark.parametrize("operation", ["Reshape", "Expand", "Tile", "Slice", "Add"])
def test_legacy_initializer_controls_remain_fixed(operation):
    """Execute IR3 initializer declarations through boundary and rank conversion."""
    shape = [2, 3]
    controls = {
        "Reshape": [3, 2],
        "Expand": [2, 3],
        "Tile": [2, 1],
        "Slice": [0],
        "Add": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
    }
    output_shape = {"Reshape": [3, 2], "Tile": [4, 3], "Slice": [1, 3]}.get(operation, shape)
    inputs = ["x", "control"]
    initializers = [
        numpy_helper.from_array(
            np.asarray(controls[operation], np.float32 if operation == "Add" else np.int64),
            "control",
        )
    ]
    if operation == "Slice":
        for name, values in (("ends", [1]), ("axes", [0])):
            initializers.append(numpy_helper.from_array(np.asarray(values, np.int64), name))
            inputs.append(name)
    model = helper.make_model(
        helper.make_graph(
            [helper.make_node(operation, inputs, ["y"])],
            "legacy_controls",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, shape)]
            + [
                helper.make_tensor_value_info(t.name, t.data_type, list(t.dims))
                for t in initializers
            ],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, output_shape)],
            initializers,
        ),
        opset_imports=[helper.make_opsetid("", 17)],
    )
    model.ir_version = 3
    feeds = {"x": np.arange(6, dtype=np.float32).reshape(shape)}
    expected = _execute(model, feeds)
    transformed = copy.deepcopy(model)
    transforms = (
        _apply_non4d_model_inputs,
        _apply_non4d_model_outputs,
        {
            "Reshape": _apply_non4d_reshape,
            "Expand": _apply_non4d_expand,
            "Tile": _apply_non4d_tile,
            "Slice": _apply_non4d_slice_axis,
            "Add": _apply_non4d_initializers,
        }[operation],
    )
    for apply in transforms:
        transformed = apply(
            transformed,
            **({"defer_rank_conversion": True} if apply is _apply_non4d_model_outputs else {}),
        )
    if operation == "Add":
        assert list(transformed.graph.initializer[0].dims) == [1, 1, 2, 3]
    np.testing.assert_array_equal(_execute(transformed, feeds)[0], expected[0])


@pytest.mark.parametrize("operation", ["QuantizeLinear", "DequantizeLinear"])
@pytest.mark.parametrize("shape", [[2, 3], [2, 3, 4]])
@pytest.mark.parametrize("per_channel", [False, True, "singleton"])
def test_registered_implicit_quantizer_axis_preserves_channel_values(operation, shape, per_channel):
    """Map every vector scale's channel axis, including a single-channel vector."""
    shape = shape.copy()
    if per_channel == "singleton":
        shape[1] = 1
    scales = np.asarray(
        [0.25, 0.5, 1.0] if per_channel is True else [0.5] if per_channel else 0.5, np.float32
    )
    zeros = np.asarray([0, 1, 2] if per_channel is True else [0] if per_channel else 0, np.uint8)
    input_type = TensorProto.FLOAT if operation == "QuantizeLinear" else TensorProto.UINT8
    output_type = TensorProto.UINT8 if operation == "QuantizeLinear" else TensorProto.FLOAT
    model = helper.make_model(
        helper.make_graph(
            [helper.make_node(operation, ["x", "scale", "zero"], ["y"])],
            "implicit_quantizer",
            [helper.make_tensor_value_info("x", input_type, shape)],
            [helper.make_tensor_value_info("y", output_type, shape)],
            [numpy_helper.from_array(scales, "scale"), numpy_helper.from_array(zeros, "zero")],
        ),
        opset_imports=[helper.make_opsetid("", 17)],
    )
    model.ir_version = 9
    data = (
        np.arange(np.prod(shape))
        .reshape(shape)
        .astype(np.float32 if input_type == TensorProto.FLOAT else np.uint8)
    )
    feeds = {"x": data}
    expected = _execute(model, feeds)[0]
    transformed = _transform_make_dla_compatible(copy.deepcopy(model), verbose=False)
    quantizer = next(node for node in transformed.graph.node if node.op_type == operation)
    if per_channel:
        assert next(attr.i for attr in quantizer.attribute if attr.name == "axis") == 5 - len(shape)
    else:
        assert all(attr.name != "axis" for attr in quantizer.attribute)
    np.testing.assert_array_equal(_execute(transformed, feeds)[0], expected)


@pytest.mark.parametrize("operation", ["QuantizeLinear", "DequantizeLinear"])
def test_legacy_quantizer_schema_keeps_implicit_scalar_axis(operation):
    """Keep pre-axis quantizer schemas byte-identical during axis preparation."""
    input_type = TensorProto.FLOAT if operation == "QuantizeLinear" else TensorProto.UINT8
    output_type = TensorProto.UINT8 if operation == "QuantizeLinear" else TensorProto.FLOAT
    model = helper.make_model(
        helper.make_graph(
            [helper.make_node(operation, ["x", "scale", "zero"], ["y"])],
            "legacy_quantizer",
            [helper.make_tensor_value_info("x", input_type, [2, 3])],
            [helper.make_tensor_value_info("y", output_type, [2, 3])],
            [
                numpy_helper.from_array(np.asarray(0.5, np.float32), "scale"),
                numpy_helper.from_array(np.asarray(0, np.uint8), "zero"),
            ],
        ),
        opset_imports=[helper.make_opsetid("", 10)],
    )
    model.ir_version = 9
    original = model.SerializeToString()
    _apply_update_axis(model)
    assert model.SerializeToString() == original
    feeds = {
        "x": np.arange(6)
        .reshape(2, 3)
        .astype(np.float32 if input_type == TensorProto.FLOAT else np.uint8)
    }
    _execute(model, feeds)


@pytest.mark.parametrize("operation", ["Expand", "Tile"])
@pytest.mark.parametrize("source", ["input", "default", "shape", "symbolic"])
def test_native_four_dimensional_runtime_controls_need_no_padding(operation, source):
    """Execute native rank-four runtime layouts without freezing their controls."""
    shape = [1, 1, 2, 3]
    first = [2, 1, 2, 3] if operation == "Expand" else [1, 1, 2, 1]
    second = [3, 1, 2, 3] if operation == "Expand" else [2, 1, 1, 1]
    declared = ["batch", 1, 2, 3] if source == "symbolic" else shape
    inputs = [helper.make_tensor_value_info("x", TensorProto.FLOAT, declared)]
    nodes = []
    initializers = []
    if source == "shape":
        nodes.append(helper.make_node("Shape", ["x"], ["control"]))
    else:
        inputs.append(helper.make_tensor_value_info("control", TensorProto.INT64, [4]))
    if source == "default":
        initializers.append(numpy_helper.from_array(np.asarray(first, np.int64), "control"))
    nodes.append(helper.make_node(operation, ["x", "control"], ["y"]))
    model = helper.make_model(
        helper.make_graph(
            nodes,
            "native_runtime_controls",
            inputs,
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [None] * 4)],
            initializers,
        ),
        opset_imports=[helper.make_opsetid("", 17)],
    )
    model.ir_version = 9
    data = np.arange(6, dtype=np.float32).reshape(shape)
    feeds = (
        [{"x": data}]
        if source in {"shape", "default"}
        else [{"x": data, "control": np.asarray(first, np.int64)}]
    )
    if source != "shape":
        feeds.append({"x": data, "control": np.asarray(second, np.int64)})
    expected = [_execute(model, values)[0] for values in feeds]
    if source == "shape":
        transformed = _transform_make_dla_compatible(copy.deepcopy(model), verbose=False)
    else:
        # ORT symbolic inference cannot resolve unconstrained runtime Expand controls.
        transformed = dict(DLA_TRANSFORMS)["Non4D" + operation](copy.deepcopy(model))
        assert transformed.SerializeToString() == model.SerializeToString()
    if source == "shape":
        assert any(node.op_type == "Shape" for node in transformed.graph.node)
    else:
        assert any(value.name == "control" for value in transformed.graph.input)
    if initializers:
        assert (
            next(t for t in transformed.graph.initializer if t.name == "control")
            == model.graph.initializer[0]
        )
    for values, reference in zip(feeds, expected):
        np.testing.assert_array_equal(_execute(transformed, values)[0], reference)
