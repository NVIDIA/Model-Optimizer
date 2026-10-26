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

"""Boundary promotion refuses unsupported exposed adapters before changing the graph."""

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper
from onnx.reference import ReferenceEvaluator

import modelopt.onnx.graph_surgery.make_dla_compatible as dla_pipeline
from modelopt.onnx.graph_surgery.dla_transforms.dla_model_boundaries import (
    _apply_fix_instancenorm_channels,
    _apply_non4d_model_inputs,
    _apply_non4d_model_outputs,
    _apply_remove_intermediary_squeeze_unsqueeze,
)
from modelopt.onnx.graph_surgery.make_dla_compatible import _transform_make_dla_compatible


def test_input_promotion_refusal_leaves_graph_unchanged():
    """Refuse removing a public input adapter before altering any earlier consumer."""
    model = helper.make_model(
        helper.make_graph(
            [
                helper.make_node("Identity", ["x"], ["first"]),
                helper.make_node("Unsqueeze", ["x", "axes"], ["y"], name="exposed"),
            ],
            "boundary",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1, 2, 3])],
            [numpy_helper.from_array(np.array([0, 1], dtype=np.int64), "axes")],
        ),
        opset_imports=[helper.make_opsetid("", 21)],
    )
    before = model.SerializeToString()
    with pytest.raises(ValueError):
        _apply_non4d_model_inputs(model)
    assert model.SerializeToString() == before


def _checked_run(model, feeds):
    """Check every generated interface and execute through the independent reference runtime."""
    onnx.checker.check_model(model, full_check=True)
    return ReferenceEvaluator(model).run(None, feeds)


def _make_model(nodes, inputs, outputs, initializers=()):
    """Build a small model with explicit public interface shapes."""
    return helper.make_model(
        helper.make_graph(nodes, "boundaries", inputs, outputs, initializer=initializers),
        opset_imports=[helper.make_opsetid("", 21)],
    )


def test_unnamed_input_adapters_keep_distinct_targets():
    """Promote two unnamed trailing-axis adapters without reusing their different targets."""
    inputs = [
        helper.make_tensor_value_info("a", TensorProto.FLOAT, [2, 3]),
        helper.make_tensor_value_info("b", TensorProto.FLOAT, [3, 4]),
    ]
    outputs = [
        helper.make_tensor_value_info("a_out", TensorProto.FLOAT, [2, 3, 1]),
        helper.make_tensor_value_info("b_out", TensorProto.FLOAT, [3, 4, 1]),
    ]
    model = _make_model(
        [
            helper.make_node("Unsqueeze", ["a", "axes"], ["a_out"]),
            helper.make_node("Unsqueeze", ["b", "axes"], ["b_out"]),
        ],
        inputs,
        outputs,
        [numpy_helper.from_array(np.asarray([-1], np.int64), "axes")],
    )
    feeds = {
        "a": np.arange(6, dtype=np.float32).reshape(2, 3),
        "b": np.arange(12, dtype=np.float32).reshape(3, 4),
    }
    expected = _checked_run(model, feeds)
    _apply_non4d_model_inputs(model)
    for actual, reference in zip(_checked_run(model, feeds), expected):
        np.testing.assert_array_equal(actual, reference)


@pytest.mark.parametrize("width", [1, 4])
@pytest.mark.parametrize("axis", [0, -2])
def test_ordered_pipeline_preserves_vector_output(width, axis):
    """Keep vector outputs, including a singleton vector, through the actual registered sequence."""
    model = _make_model(
        [
            helper.make_node("MatMul", ["x", "w"], ["matrix"]),
            helper.make_node("Squeeze", ["matrix", "axes"], ["y"]),
        ],
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [width])],
        [
            numpy_helper.from_array(np.arange(3 * width, dtype=np.float32).reshape(3, width), "w"),
            numpy_helper.from_array(np.asarray([axis], np.int64), "axes"),
        ],
    )
    feeds = {"x": np.asarray([[1, 2, 3]], np.float32)}
    expected = _checked_run(model, feeds)[0]
    transformed = _transform_make_dla_compatible(model, verbose=False)
    actual = _checked_run(transformed, feeds)[0]
    assert actual.shape == (width,)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("axes", [[1], [-2], None])
def test_ordered_pipeline_preserves_matrix_squeeze_axes(axes):
    """Restore matrix outputs with positive, negative or omitted source squeeze axes."""
    controls = [] if axes is None else [numpy_helper.from_array(np.asarray(axes, np.int64), "axes")]
    squeeze_inputs = ["matrix"] if axes is None else ["matrix", "axes"]
    model = _make_model(
        [
            helper.make_node("MatMul", ["x", "w"], ["matrix"]),
            helper.make_node("Squeeze", squeeze_inputs, ["y"]),
        ],
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 1, 3])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 4])],
        [numpy_helper.from_array(np.arange(12, dtype=np.float32).reshape(3, 4), "w"), *controls],
    )
    feeds = {"x": np.arange(6, dtype=np.float32).reshape(2, 1, 3)}
    expected = _checked_run(model, feeds)[0]
    transformed = _transform_make_dla_compatible(model, verbose=False)
    np.testing.assert_array_equal(_checked_run(transformed, feeds)[0], expected)


def test_standalone_intermediary_rank_change_preserves_concat_inputs():
    """Retain an internal rank-changing squeeze whose downstream consumer expects rank three."""
    model = _make_model(
        [
            helper.make_node("Identity", ["x"], ["before"]),
            helper.make_node("Squeeze", ["before", "axes"], ["squeezed"]),
            helper.make_node("Concat", ["squeezed", "b"], ["y"], axis=0),
        ],
        [
            helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 4, 4]),
            helper.make_tensor_value_info("b", TensorProto.FLOAT, [2, 4, 4]),
        ],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [5, 4, 4])],
        [numpy_helper.from_array(np.asarray([0], np.int64), "axes")],
    )
    model = onnx.shape_inference.infer_shapes(model)
    feeds = {
        "x": np.arange(48, dtype=np.float32).reshape(1, 3, 4, 4),
        "b": np.ones((2, 4, 4), np.float32),
    }
    expected = _checked_run(model, feeds)[0]
    _apply_remove_intermediary_squeeze_unsqueeze(model)
    np.testing.assert_array_equal(_checked_run(model, feeds)[0], expected)


def test_standalone_empty_axes_adapter_can_be_removed():
    """Retain cleanup of a genuine no-op adapter in standalone mode."""
    model = _make_model(
        [
            helper.make_node("Identity", ["x"], ["before"]),
            helper.make_node("Unsqueeze", ["before", "axes"], ["adapted"]),
            helper.make_node("Identity", ["adapted"], ["y"]),
        ],
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, 3, 4])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2, 3, 4])],
        [numpy_helper.from_array(np.asarray([], np.int64), "axes")],
    )
    model = onnx.shape_inference.infer_shapes(model)
    feeds = {"x": np.arange(24, dtype=np.float32).reshape(1, 2, 3, 4)}
    expected = _checked_run(model, feeds)[0]
    _apply_remove_intermediary_squeeze_unsqueeze(model)
    assert not any(node.op_type == "Unsqueeze" for node in model.graph.node)
    np.testing.assert_array_equal(_checked_run(model, feeds)[0], expected)


@pytest.mark.parametrize("batch", [1, 2])
def test_internal_normalization_boundary_restores_trailing_singleton(batch):
    """Restore native normalization output shapes after preserving its channel dimension."""
    model = _make_model(
        [
            helper.make_node(
                "InstanceNormalization", ["x", "scale", "bias"], ["y"], name="normalize"
            )
        ],
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [batch, 3, 4])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [batch, 3, 4])],
        [
            numpy_helper.from_array(np.ones(3, np.float32), "scale"),
            numpy_helper.from_array(np.zeros(3, np.float32), "bias"),
        ],
    )
    feeds = {"x": np.arange(batch * 12, dtype=np.float32).reshape(batch, 3, 4)}
    expected = _checked_run(model, feeds)[0]
    _apply_non4d_model_outputs(model, defer_rank_conversion=True)
    _apply_fix_instancenorm_channels(model)
    np.testing.assert_allclose(_checked_run(model, feeds)[0], expected, rtol=1e-6, atol=1e-6)


def test_unnamed_normalizations_keep_distinct_targets():
    """Give unnamed normalization inputs with different shapes independent reshape targets."""
    model = _make_model(
        [
            helper.make_node("InstanceNormalization", ["a", "scale", "bias"], ["a_out"]),
            helper.make_node("InstanceNormalization", ["b", "scale", "bias"], ["b_out"]),
        ],
        [
            helper.make_tensor_value_info("a", TensorProto.FLOAT, [2, 3, 4]),
            helper.make_tensor_value_info("b", TensorProto.FLOAT, [3, 3, 5]),
        ],
        [
            helper.make_tensor_value_info("a_out", TensorProto.FLOAT, [2, 3, 4]),
            helper.make_tensor_value_info("b_out", TensorProto.FLOAT, [3, 3, 5]),
        ],
        [
            numpy_helper.from_array(np.ones(3, np.float32), "scale"),
            numpy_helper.from_array(np.zeros(3, np.float32), "bias"),
        ],
    )
    feeds = {
        "a": np.arange(24, dtype=np.float32).reshape(2, 3, 4),
        "b": np.arange(45, dtype=np.float32).reshape(3, 3, 5),
    }
    expected = _checked_run(model, feeds)
    _apply_non4d_model_outputs(model, defer_rank_conversion=True)
    _apply_fix_instancenorm_channels(model)
    for actual, reference in zip(_checked_run(model, feeds), expected):
        np.testing.assert_allclose(actual, reference, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize(
    "transform",
    [
        _apply_non4d_model_inputs,
        _apply_remove_intermediary_squeeze_unsqueeze,
        _apply_fix_instancenorm_channels,
    ],
)
def test_untouched_constant_payload_stays_in_place(transform):
    """Preserve the original protobuf payload object when no node replacement touches it."""
    model = _make_model(
        [
            helper.make_node(
                "Constant",
                [],
                ["constant"],
                value=numpy_helper.from_array(np.arange(8, dtype=np.float32)),
            ),
            helper.make_node("Identity", ["x"], ["y"]),
        ],
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, 3, 4])],
        [
            helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2, 3, 4]),
            helper.make_tensor_value_info("constant", TensorProto.FLOAT, [8]),
        ],
    )
    constant = model.graph.node[0]
    feeds = {"x": np.arange(24, dtype=np.float32).reshape(1, 2, 3, 4)}
    expected = _checked_run(model, feeds)
    transform(model)
    assert model.graph.node[0] is constant
    for actual, reference in zip(_checked_run(model, feeds), expected):
        np.testing.assert_array_equal(actual, reference)


def test_input_adapter_runtime_axes_refuses_before_mutation():
    """Refuse baking an overridable axes default before applying earlier promotions."""
    model = _make_model(
        [
            helper.make_node("Identity", ["a"], ["a_out"]),
            helper.make_node("Unsqueeze", ["x", "axes"], ["y"]),
        ],
        [
            helper.make_tensor_value_info("a", TensorProto.FLOAT, [2, 3]),
            helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3]),
            helper.make_tensor_value_info("axes", TensorProto.INT64, [1]),
        ],
        [
            helper.make_tensor_value_info("a_out", TensorProto.FLOAT, [2, 3]),
            helper.make_tensor_value_info("y", TensorProto.FLOAT, ["a", "b", "c"]),
        ],
        [numpy_helper.from_array(np.asarray([-1], np.int64), "axes")],
    )
    onnx.checker.check_model(model, full_check=True)
    before = model.SerializeToString()
    with pytest.raises(ValueError, match="fixed"):
        _apply_non4d_model_inputs(model)
    assert model.SerializeToString() == before


@pytest.mark.parametrize("transform", [_apply_non4d_model_inputs, _apply_non4d_model_outputs])
def test_legacy_boundary_adapter_schema_refuses_before_mutation(transform):
    """Refuse input-form axes adapters on a legacy operator schema."""
    model = _make_model(
        [helper.make_node("Neg", ["x"], ["y"])],
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 3])],
    )
    model.opset_import[0].version = 11
    onnx.checker.check_model(model, full_check=True)
    before = model.SerializeToString()
    with pytest.raises(ValueError, match="13"):
        options = {"defer_rank_conversion": True} if transform is _apply_non4d_model_outputs else {}
        transform(model, **options)
    assert model.SerializeToString() == before


def test_legacy_ir_boundary_adapters_register_initializer_inputs():
    """Keep both new boundary-axis initializers legal in an IR3 graph."""
    model = _make_model(
        [helper.make_node("Neg", ["x"], ["y"])],
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 3])],
    )
    model.ir_version = 3
    feeds = {"x": np.arange(6, dtype=np.float32).reshape(2, 3)}
    expected = _checked_run(model, feeds)[0]
    _apply_non4d_model_inputs(model)
    _apply_non4d_model_outputs(model, defer_rank_conversion=True)
    np.testing.assert_array_equal(_checked_run(model, feeds)[0], expected)


def _capture_branch(name, source, shape):
    """Make a branch that reads an outer-scope value."""
    return helper.make_graph(
        [helper.make_node("Identity", [source], [name])],
        name,
        [],
        [helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)],
    )


def test_input_promotion_captured_adapter_refuses_before_mutation():
    """Retain a captured adapter output instead of leaving its lexical read unbound."""
    model = _make_model(
        [
            helper.make_node("Unsqueeze", ["x", "axes"], ["captured"]),
            helper.make_node(
                "If",
                ["condition"],
                ["y"],
                then_branch=_capture_branch("then", "captured", [1, 1, 2, 3]),
                else_branch=_capture_branch("else", "captured", [1, 1, 2, 3]),
            ),
        ],
        [
            helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3]),
            helper.make_tensor_value_info("condition", TensorProto.BOOL, []),
        ],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1, 2, 3])],
        [numpy_helper.from_array(np.asarray([0, 1], np.int64), "axes")],
    )
    onnx.checker.check_model(model, full_check=True)
    before = model.SerializeToString()
    with pytest.raises(ValueError, match="captured"):
        _apply_non4d_model_inputs(model)
    assert model.SerializeToString() == before


@pytest.mark.parametrize("rank_converted", [False, True])
def test_intermediary_cleanup_preserves_captured_aliases(rank_converted):
    """Keep the producer name captured by an If when a no-op adapter could be removed."""
    model = _make_model(
        [
            helper.make_node("Identity", ["x"], ["captured"]),
            helper.make_node("Unsqueeze", ["captured", "axes"], ["adapted"]),
            helper.make_node(
                "If",
                ["condition"],
                ["chosen"],
                then_branch=_capture_branch("then", "captured", [1, 2, 3, 4]),
                else_branch=_capture_branch("else", "captured", [1, 2, 3, 4]),
            ),
            helper.make_node("Add", ["chosen", "adapted"], ["y"]),
        ],
        [
            helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, 3, 4]),
            helper.make_tensor_value_info("condition", TensorProto.BOOL, []),
        ],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2, 3, 4])],
        [numpy_helper.from_array(np.asarray([], np.int64), "axes")],
    )
    model = onnx.shape_inference.infer_shapes(model)
    feeds = {"x": np.arange(24, dtype=np.float32).reshape(1, 2, 3, 4), "condition": np.bool_(True)}
    expected = _checked_run(model, feeds)[0]
    if rank_converted:
        _apply_remove_intermediary_squeeze_unsqueeze(model, rank_converted=True)
    else:
        _apply_remove_intermediary_squeeze_unsqueeze(model)
    np.testing.assert_array_equal(_checked_run(model, feeds)[0], expected)


def test_output_boundary_orders_a_captured_public_value_before_its_reader():
    """Make a restored public value available before an existing lexical consumer."""
    model = _make_model(
        [
            helper.make_node("Neg", ["x"], ["y"]),
            helper.make_node(
                "If",
                ["condition"],
                ["unused"],
                then_branch=_capture_branch("then", "y", [2, 3]),
                else_branch=_capture_branch("else", "y", [2, 3]),
            ),
        ],
        [
            helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3]),
            helper.make_tensor_value_info("condition", TensorProto.BOOL, []),
        ],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 3])],
    )
    feeds = {"x": np.arange(6, dtype=np.float32).reshape(2, 3), "condition": np.bool_(True)}
    expected = _checked_run(model, feeds)[0]
    _apply_non4d_model_inputs(model)
    _apply_non4d_model_outputs(model, defer_rank_conversion=True)
    np.testing.assert_array_equal(_checked_run(model, feeds)[0], expected)


@pytest.mark.parametrize(
    ("axes", "output_shape"),
    [([2, 0], [1, 2, 1, 3]), ([-2, -1], [2, 3, 1, 1]), ([2, -4], [1, 2, 1, 3])],
)
def test_input_adapters_resolve_unsqueeze_axes_against_output_rank(axes, output_shape):
    """Preserve unordered positive, negative and mixed Unsqueeze axis semantics."""
    model = _make_model(
        [helper.make_node("Unsqueeze", ["x", "axes"], ["y"])],
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, output_shape)],
        [numpy_helper.from_array(np.asarray(axes, np.int64), "axes")],
    )
    feeds = {"x": np.arange(6, dtype=np.float32).reshape(2, 3)}
    expected = _checked_run(model, feeds)[0]
    transformed = _apply_non4d_model_inputs(model)
    np.testing.assert_array_equal(_checked_run(transformed, feeds)[0], expected)


@pytest.mark.parametrize("batch", [1, 2, 3])
def test_registered_normalization_preserves_native_channel_axis(batch):
    """Keep direct rank-three normalization inputs on the channel-preserving route."""
    shape = [batch, 3, 4]
    model = _make_model(
        [helper.make_node("InstanceNormalization", ["x", "scale", "bias"], ["y"])],
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, shape)],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, shape)],
        [
            numpy_helper.from_array(np.asarray([1, 2, 3], np.float32), "scale"),
            numpy_helper.from_array(np.asarray([-1, 0, 1], np.float32), "bias"),
        ],
    )
    feeds = {"x": np.random.default_rng(19).standard_normal(shape).astype(np.float32)}
    expected = _checked_run(model, feeds)[0]
    transformed = _transform_make_dla_compatible(model, verbose=False)
    np.testing.assert_allclose(_checked_run(transformed, feeds)[0], expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("entrypoint", ["standalone", "pipeline"])
def test_native_reduction_output_refuses_before_adding_rank_adapters(entrypoint, monkeypatch):
    """Refuse a native producer when no rank-conversion stage can prepare its output."""
    model = _make_model(
        [helper.make_node("ReduceMean", ["x"], ["y"], axes=[2, 3], keepdims=0)],
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 4, 4])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3])],
    )
    before = model.SerializeToString()
    if entrypoint == "standalone":
        with pytest.raises(ValueError, match="rank"):
            _apply_non4d_model_outputs(model)
    else:
        monkeypatch.setattr(
            dla_pipeline,
            "DLA_TRANSFORMS",
            tuple(
                (name, transform)
                for name, transform in dla_pipeline.DLA_TRANSFORMS
                if name != "StandaloneReduceOperations"
            ),
        )
        with pytest.raises(NotImplementedError, match="reductions"):
            _transform_make_dla_compatible(model, verbose=False)
    assert model.SerializeToString() == before


@pytest.mark.parametrize("axes", [[0, 2], [0, -3]])
def test_output_adapter_refuses_native_rank_five_squeeze_before_mutation(axes):
    """Avoid remapping axes through an unspecified five-dimensional rank collapse."""
    model = _make_model(
        [helper.make_node("Squeeze", ["x", "axes"], ["y"])],
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, 1, 3, 4])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 3, 4])],
        [numpy_helper.from_array(np.asarray(axes, np.int64), "axes")],
    )
    before = model.SerializeToString()
    with pytest.raises(ValueError, match="rank"):
        _apply_non4d_model_outputs(model, defer_rank_conversion=True)
    assert model.SerializeToString() == before
