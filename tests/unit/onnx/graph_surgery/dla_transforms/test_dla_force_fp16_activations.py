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

"""Focused FP16 payload, boundary, and DLA projection assertions on small fixtures."""

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import TensorProto as T
from onnx import helper as h
from onnx import numpy_helper as nh
from onnx.reference import ReferenceEvaluator

from modelopt.onnx.graph_surgery import make_dla_compatible
from modelopt.onnx.graph_surgery.dla_transforms.dla_force_fp16_activations import (
    _apply_force_fp16_activations,
    _apply_fp16_lpnorm,
)
from modelopt.onnx.graph_surgery.dla_transforms.dla_rank_conversion import _apply_non4d_reshape
from modelopt.onnx.graph_surgery.make_dla_compatible import _transform_make_dla_compatible


def _value(name, shape, dtype=T.FLOAT):
    """Declare a typed tensor for a checked precision fixture."""
    return h.make_tensor_value_info(name, dtype, shape)


def _model(nodes, inputs, outputs, initializers, opset=19):
    """Build a model with inferred intermediate types for a precision fixture."""
    model = h.make_model(
        h.make_graph(nodes, "fp16", inputs, outputs, initializers),
        opset_imports=[h.make_opsetid("", opset)],
    )
    model.ir_version = 9
    return onnx.shape_inference.infer_shapes(model)


@pytest.mark.parametrize("enabled", [False, True])
def test_public_option_preserves_boundaries_and_selects_weight_precision(tmp_path, enabled):
    """Keep public FLOAT interfaces while selecting internal weight precision."""
    values = np.array([1.0003, 0.12345], np.float32)
    model = _model(
        [h.make_node("Mul", ["x", "w"], ["y"])],
        [_value("x", [1, 1, 1, 2])],
        [_value("y", [1, 1, 1, 2])],
        [nh.from_array(values.reshape(1, 1, 1, 2), "w")],
    )
    source, output = tmp_path / "source.onnx", tmp_path / "output.onnx"
    onnx.save(model, source)
    result = make_dla_compatible(
        str(source),
        str(output),
        verbose=False,
        use_external_data=False,
        force_fp16_activations=enabled,
    )
    assert result.graph.input[0].type.tensor_type.elem_type == T.FLOAT
    assert result.graph.output[0].name == "y"
    assert result.graph.output[0].type.tensor_type.elem_type == T.FLOAT
    assert result.graph.initializer[0].data_type == (T.FLOAT16 if enabled else T.FLOAT)
    data = np.array([[[[0.3333, 2.0003]]]], np.float32)
    dtype = np.float16 if enabled else np.float32
    expected = (data.astype(dtype) * values.astype(dtype)).astype(np.float32)
    actual = ReferenceEvaluator(onnx.load(output)).run(None, {"x": data})[0]
    np.testing.assert_array_equal(actual, expected)


def test_initializer_rounding_preserves_subnormals_signed_zero_and_nan():
    """Retain the original converter rounding, signed zero and NaN behavior."""
    values = np.array([0, -0.0, 2**-24, 2**-25, 2**-14, 65504, 65520, 0, 0], np.float32)
    values.view(np.uint32)[-2:] = [0x7F812345, 0xFF812345]
    model = _model([], [], [_value("weight", [len(values)])], [nh.from_array(values, "weight")])
    result = _apply_force_fp16_activations(model)
    actual = nh.to_array(result.graph.initializer[0]).view(np.uint16)
    np.testing.assert_array_equal(actual[:-2], [0, 0x8000, 1, 0, 0x400, 0x7BFF, 0x7C00])
    assert np.isnan(actual[-2:].view(np.float16)).all()
    assert result.graph.node[0].op_type == "Cast"
    assert result.graph.node[0].output[0] == "weight"


def test_boundary_adapters_and_direct_input_consumers_get_their_own_casts():
    """Give boundary adapters and direct input uses their own typed views."""
    model = _model(
        [
            h.make_node("Unsqueeze", ["x", "axes"], ["x4"]),
            h.make_node("Relu", ["x4"], ["r4"]),
            h.make_node("Squeeze", ["r4", "axes"], ["y"]),
            h.make_node("Neg", ["x"], ["direct"]),
        ],
        [_value("x", [1, 2])],
        [_value("y", [1, 2]), _value("direct", [1, 2])],
        [nh.from_array(np.array([0, 1], np.int64), "axes")],
    )
    result = _apply_force_fp16_activations(model)
    producers = {name: n for n in result.graph.node for name in n.output}
    relu = next(n for n in result.graph.node if n.op_type == "Relu")
    assert producers[relu.input[0]].op_type == "Cast"
    assert producers[producers[relu.input[0]].input[0]].op_type == "Unsqueeze"
    squeeze = producers["y"]
    assert squeeze.op_type == "Squeeze"
    assert producers[squeeze.input[0]].op_type == "Cast"
    actual = ReferenceEvaluator(result).run(None, {"x": np.array([[-2, 3]], np.float32)})
    np.testing.assert_array_equal(actual[0], [[0, 3]])
    np.testing.assert_array_equal(actual[1], [[2, -3]])


@pytest.mark.parametrize("operation", ["Resize", "QuantizeLinear"])
def test_schema_specific_float_parameters_and_qdq_opset(operation):
    """Keep required FLOAT controls and migrate FP16 quantizer schemas."""
    if operation == "Resize":
        nodes = [
            h.make_node("Mul", ["x", "weight"], ["weighted"]),
            h.make_node("Resize", ["weighted", "roi", "scales"], ["y"]),
        ]
        initializers = [
            nh.from_array(np.full((1, 1, 1, 2), 0.5, np.float32), "weight"),
            nh.from_array(np.array([], np.float32), "roi"),
            nh.from_array(np.array([1, 1, 1, 2], np.float32), "scales"),
        ]
        output_shape = [1, 1, 1, 4]
    else:
        nodes = [
            h.make_node("QuantizeLinear", ["x", "scale", "zero"], ["q"]),
            h.make_node("DequantizeLinear", ["q", "scale", "zero"], ["y"]),
        ]
        initializers = [nh.from_array(np.float32(0.1), "scale"), nh.from_array(np.int8(0), "zero")]
        output_shape = [1, 1, 1, 2]
    model = _model(
        nodes, [_value("x", [1, 1, 1, 2])], [_value("y", output_shape)], initializers, opset=17
    )
    result = _apply_force_fp16_activations(model)
    result = onnx.shape_inference.infer_shapes(result, check_type=True)
    onnx.checker.check_model(result)
    types = {t.name: t.data_type for t in result.graph.initializer}
    if operation == "Resize":
        assert types == {"weight": T.FLOAT16, "roi": T.FLOAT, "scales": T.FLOAT}
        assert result.opset_import[0].version == 17
    else:
        assert types == {"scale": T.FLOAT16, "zero": T.INT8}
        assert result.opset_import[0].version == 19


def test_scaled_l2_normalization_keeps_zero_slices_finite():
    """Stabilize zero-vector normalization before applying FP16 precision."""
    model = _model(
        [
            h.make_node("LpNormalization", ["x"], ["norm"], p=2, axis=-1),
            h.make_node("Mul", ["norm", "gamma"], ["y"]),
        ],
        [_value("x", [1, 1, 2, 2])],
        [_value("y", [1, 1, 2, 2])],
        [nh.from_array(np.ones((1, 1, 1, 2), np.float32), "gamma")],
    )
    result = _apply_force_fp16_activations(_apply_fp16_lpnorm(model))
    assert not any(n.op_type == "LpNormalization" for n in result.graph.node)
    assert any(n.op_type == "Max" for n in result.graph.node)
    data = np.array([[[[0, 0], [3, 4]]]], np.float32)
    actual = ReferenceEvaluator(result).run(None, {"x": data})[0]
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, [[[[0, 0], [0.6, 0.8]]]], atol=3e-4)


def _scaled_collapsed_lpnorm_model(shape, axis, prefix=False):
    """Build checked native normalization around a real singleton Reshape collapse."""
    collapsed = shape.copy()
    collapsed.remove(1)
    nodes = [
        h.make_node("Reshape", ["x", "native_shape"], ["native"]),
        h.make_node("LpNormalization", ["native"], ["norm"], axis=axis, p=2),
        h.make_node("Mul", ["norm", "gamma"], ["scaled"]),
        h.make_node("Reshape", ["scaled", "output_shape"], ["y"]),
    ]
    outputs = [_value("y", collapsed)]
    initializers = [
        nh.from_array(np.asarray(shape, np.int64), "native_shape"),
        nh.from_array(np.asarray(collapsed, np.int64), "output_shape"),
        nh.from_array(np.ones(shape[-1], np.float32), "gamma"),
    ]
    if prefix:
        nodes = [
            h.make_node("LpNormalization", ["x"], ["first_norm"], axis=-1, p=2),
            h.make_node("Mul", ["first_norm", "first_gamma"], ["first"]),
            *nodes,
        ]
        outputs.insert(0, _value("first", collapsed))
        initializers.append(nh.from_array(np.ones(collapsed[-1], np.float32), "first_gamma"))
    return _model(nodes, [_value("x", collapsed)], outputs, initializers, opset=17)


@pytest.mark.parametrize("shape", [[1, 2, 3, 4, 5], [2, 3, 1, 4, 5]])
@pytest.mark.parametrize("axis", [-1, 4])
def test_fp16_scaled_lpnorm_uses_actual_collapsed_axis(shape, axis):
    """Use the mapped last axis after the registered five-dimensional collapse."""
    model = _scaled_collapsed_lpnorm_model(shape, axis)
    collapsed = shape.copy()
    collapsed.remove(1)
    feeds = {"x": np.linspace(0.25, 2, np.prod(collapsed), dtype=np.float32).reshape(collapsed)}
    onnx.checker.check_model(model, full_check=True)
    expected = ReferenceEvaluator(model).run(None, feeds)[0]
    converted = _transform_make_dla_compatible(model, verbose=False, force_fp16_activations=True)
    onnx.checker.check_model(converted, full_check=True)
    actual = ReferenceEvaluator(converted).run(None, feeds)[0]
    np.testing.assert_allclose(actual, expected, rtol=2e-3, atol=3e-4)


@pytest.mark.parametrize("axis", [-1, 4])
@pytest.mark.parametrize("enabled", [False, True])
def test_scaled_lpnorm_refuses_normalizing_over_the_removed_axis(axis, enabled):
    """Give FP16 and FP32 the same refusal for a normalization axis removed by collapse."""
    model = _scaled_collapsed_lpnorm_model([2, 3, 4, 5, 1], axis)
    feeds = {"x": np.ones([2, 3, 4, 5], np.float32)}
    onnx.checker.check_model(model, full_check=True)
    np.testing.assert_array_equal(ReferenceEvaluator(model).run(None, feeds)[0], feeds["x"])
    with pytest.raises(ValueError, match="cannot map axis"):
        _transform_make_dla_compatible(model, verbose=False, force_fp16_activations=enabled)


@pytest.mark.parametrize("axis", [-1, 4])
def test_fp16_lpnorm_validates_late_collapsed_axis_before_decomposition(axis):
    """Refuse unsupported normalization before mutating an earlier eligible FP16 branch."""
    model = _scaled_collapsed_lpnorm_model([2, 3, 4, 5, 1], axis, prefix=True)
    onnx.checker.check_model(model, full_check=True)
    ReferenceEvaluator(model).run(None, {"x": np.ones([2, 3, 4, 5], np.float32)})
    _apply_non4d_reshape(model)
    before = model.SerializeToString()
    with pytest.raises(ValueError, match="cannot map axis"):
        _apply_fp16_lpnorm(model)
    assert model.SerializeToString() == before


def test_terminal_projection_splits_channels_and_preserves_values():
    """Bound private projection channels while preserving real output values."""
    channels = 8193
    model = _model(
        [
            h.make_node("Conv", ["x", "weight"], ["conv"]),
            h.make_node("Transpose", ["conv"], ["transposed"], perm=[0, 3, 2, 1]),
            h.make_node("Concat", ["transposed"], ["concat1"], axis=0),
            h.make_node("ReduceSum", ["concat1", "axes"], ["reduced"], keepdims=1),
            h.make_node("Concat", ["reduced"], ["concat2"], axis=2),
            h.make_node("Squeeze", ["concat2", "axes"], ["y"]),
        ],
        [_value("x", [1, 1, 1, 1])],
        [_value("y", [1, 1, channels])],
        [
            nh.from_array(np.full((channels, 1, 1, 1), np.float32(0.25)), "weight"),
            nh.from_array(np.array([0], np.int64), "axes"),
        ],
    )
    result = _apply_force_fp16_activations(model)
    assert sum(n.op_type == "Conv" for n in result.graph.node) == 2
    weights = [t for t in result.graph.initializer if t.data_type == T.FLOAT16]
    assert [list(t.dims) for t in weights] == [[8192, 1, 1, 1], [1, 1, 1, 1]]
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    actual = ort.InferenceSession(
        result.SerializeToString(), options, providers=["CPUExecutionProvider"]
    ).run(None, {"x": np.array([[[[2]]]], np.float32)})[0]
    np.testing.assert_array_equal(actual, np.full((1, 1, channels), np.float32(0.5)))


def test_excluded_initializers_are_restored_when_conversion_fails():
    """Retain excluded weight payloads when the original converter rejects an included tensor."""
    roi = nh.from_array(np.array([], np.float32), "roi")
    scales = nh.from_array(np.array([1, 1, 1, 2], np.float32), "scales")
    invalid = T(name="invalid", data_type=T.FLOAT, dims=[2], float_data=[1])
    model = h.make_model(
        h.make_graph(
            [h.make_node("Resize", ["x", "roi", "scales"], ["y"])],
            "conversion_failure",
            [_value("x", [1, 1, 1, 2])],
            [_value("y", [1, 1, 1, 4])],
            [roi, invalid, scales],
        )
    )
    expected = {value.name: value.SerializeToString() for value in (roi, scales)}
    with pytest.raises(ValueError):
        _apply_force_fp16_activations(model)
    actual = {value.name: value.SerializeToString() for value in model.graph.initializer}
    assert all(actual[name] == payload for name, payload in expected.items())


def _checked_cpu_outputs(model, feeds):
    """Execute a fully checked graph with bounded CPU runtime threading."""
    onnx.checker.check_model(model, full_check=True)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(
        model.SerializeToString(), options, providers=["CPUExecutionProvider"]
    ).run(None, feeds)


def test_fp16_runtime_initializer_default_retains_fp32_public_contract():
    """Keep FP32 default-input declarations and accept FP32 runtime overrides."""
    model = _model(
        [h.make_node("Mul", ["x", "w"], ["y"])],
        [_value("x", [2]), _value("w", [2])],
        [_value("y", [2])],
        [nh.from_array(np.asarray([0.5, 2], np.float32), "w")],
    )
    before = model.graph.initializer[0].SerializeToString()
    cases = [
        {"x": np.asarray([2, 4], np.float32)},
        {"x": np.asarray([2, 4], np.float32), "w": np.asarray([2, 0.25], np.float32)},
    ]
    expected = [_checked_cpu_outputs(model, feeds)[0] for feeds in cases]
    changed = _apply_force_fp16_activations(model)
    assert (
        next(value for value in changed.graph.input if value.name == "w").type.tensor_type.elem_type
        == T.FLOAT
    )
    assert (
        next(value for value in changed.graph.initializer if value.name == "w").SerializeToString()
        == before
    )
    for feeds, wanted in zip(cases, expected):
        np.testing.assert_array_equal(_checked_cpu_outputs(changed, feeds)[0], wanted)


def test_fp16_resize_parameter_shared_with_arithmetic_uses_private_precision_view():
    """Keep required FP32 scales while providing a separate FP16 arithmetic operand."""
    model = _model(
        [
            h.make_node("Mul", ["x", "scales"], ["weighted"]),
            h.make_node("Resize", ["weighted", "roi", "scales"], ["y"]),
        ],
        [_value("x", [1, 1, 1, 4])],
        [_value("y", [1, 1, 1, 8])],
        [
            nh.from_array(np.asarray([1, 1, 1, 2], np.float32), "scales"),
            nh.from_array(np.asarray([], np.float32), "roi"),
        ],
    )
    feeds = {"x": np.asarray([[[[2, 4, 6, 8]]]], np.float32)}
    expected = _checked_cpu_outputs(model, feeds)[0]
    changed = _apply_force_fp16_activations(model)
    np.testing.assert_array_equal(_checked_cpu_outputs(changed, feeds)[0], expected)


def test_fp16_conversion_preserves_unrelated_constant_object():
    """Avoid copying an unrelated integer Constant while inserting floating boundaries."""
    constant = h.make_node(
        "Constant", [], ["integer_payload"], value=nh.from_array(np.arange(64, dtype=np.uint8))
    )
    model = _model(
        [constant, h.make_node("Relu", ["x"], ["y"])], [_value("x", [2])], [_value("y", [2])], []
    )
    constant = model.graph.node[0]
    before = constant.SerializeToString()
    changed = _apply_force_fp16_activations(model)
    assert next(node for node in changed.graph.node if node.op_type == "Constant") is constant
    assert constant.SerializeToString() == before
    _checked_cpu_outputs(changed, {"x": np.asarray([-1, 2], np.float32)})


def test_fp16_malformed_late_weight_refuses_before_any_mutation():
    """Validate every floating payload before converting an earlier fixed initializer."""
    model = h.make_model(
        h.make_graph(
            [h.make_node("Mul", ["x", "valid"], ["y"])],
            "invalid_late_weight",
            [_value("x", [2])],
            [_value("y", [2])],
            [
                nh.from_array(np.asarray([1, 2], np.float32), "valid"),
                T(name="invalid", data_type=T.FLOAT, dims=[2], float_data=[1]),
            ],
        ),
        opset_imports=[h.make_opsetid("", 19)],
    )
    model.ir_version = 9
    before = model.SerializeToString()
    with pytest.raises(ValueError):
        _apply_force_fp16_activations(model)
    assert model.SerializeToString() == before


def test_fp16_qdq_upgrade_keeps_legacy_reduce_axes_valid():
    """Migrate schema-sensitive reduction axes when enabling FP16 quantizers."""
    model = _model(
        [
            h.make_node("ReduceMean", ["x"], ["mean"], axes=[1], keepdims=1),
            h.make_node("QuantizeLinear", ["mean", "scale", "zero"], ["q"]),
            h.make_node("DequantizeLinear", ["q", "scale", "zero"], ["y"]),
        ],
        [_value("x", [2, 2])],
        [_value("y", [2, 1])],
        [nh.from_array(np.float32(0.1), "scale"), nh.from_array(np.int8(0), "zero")],
        opset=17,
    )
    onnx.checker.check_model(model, full_check=True)
    feeds = {"x": np.asarray([[0.2, 0.2], [0.4, 0.4]], np.float32)}
    expected = _checked_cpu_outputs(model, feeds)[0]
    changed = _apply_force_fp16_activations(model)
    onnx.checker.check_model(changed, full_check=True)
    np.testing.assert_allclose(
        ReferenceEvaluator(changed).run(None, feeds)[0], expected, rtol=2e-3, atol=1e-4
    )


def test_fp16_lpnorm_keeps_opaque_function_native():
    """Do not interpret an opaque LpNormalization name as standard normalization."""
    model = _model(
        [
            h.make_node("LpNormalization", ["x"], ["norm"], p=2, axis=-1),
            h.make_node("Mul", ["norm", "gamma"], ["y"]),
        ],
        [_value("x", [1, 1, 1, 2])],
        [_value("y", [1, 1, 1, 2])],
        [nh.from_array(np.ones([1, 1, 1, 2], np.float32), "gamma")],
    )
    model.graph.node[0].domain = "opaque"
    model.functions.append(
        h.make_function(
            "opaque",
            "LpNormalization",
            ["x"],
            ["y"],
            [h.make_node("Neg", ["x"], ["y"])],
            [h.make_opsetid("", 19)],
            attributes=["p", "axis"],
        )
    )
    model.opset_import.append(h.make_opsetid("opaque", 1))
    before = model.SerializeToString()
    changed = _apply_fp16_lpnorm(model)
    assert changed.SerializeToString() == before
    _checked_cpu_outputs(changed, {"x": np.asarray([[[[3, 4]]]], np.float32)})


def test_fp16_boundary_cast_names_are_allocated():
    """Keep generated Cast node names distinct from existing nodes and tensor names."""
    model = _model(
        [
            h.make_node(
                "Constant",
                [],
                ["integer"],
                name="x_to_fp16",
                value=nh.from_array(np.asarray([1], np.int8)),
            ),
            h.make_node("Relu", ["x"], ["y"]),
        ],
        [_value("x", [2])],
        [_value("y", [2])],
        [],
    )
    changed = _apply_force_fp16_activations(model)
    names = [node.name for node in changed.graph.node if node.name]
    assert len(names) == len(set(names))
    _checked_cpu_outputs(changed, {"x": np.asarray([-1, 2], np.float32)})


def test_fp16_lpnorm_keeps_unrelated_constant_object():
    """Retain a large-payload owner while replacing a private normalization node."""
    model = _model(
        [
            h.make_node(
                "Constant", [], ["integer"], value=nh.from_array(np.arange(64, dtype=np.uint8))
            ),
            h.make_node("LpNormalization", ["x"], ["norm"], p=2, axis=-1),
            h.make_node("Mul", ["norm", "gamma"], ["y"]),
        ],
        [_value("x", [1, 1, 1, 2])],
        [_value("y", [1, 1, 1, 2])],
        [nh.from_array(np.ones([1, 1, 1, 2], np.float32), "gamma")],
    )
    constant = model.graph.node[0]
    changed = _apply_fp16_lpnorm(model)
    assert next(node for node in changed.graph.node if node.op_type == "Constant") is constant
    onnx.checker.check_model(changed, full_check=True)


def test_fp16_keeps_opaque_float_function_type_contract():
    """Retain a local function's FP32 region and cast only its downstream standard use."""
    model = _model(
        [h.make_node("Identity", ["x"], ["middle"]), h.make_node("Relu", ["middle"], ["y"])],
        [_value("x", [2])],
        [_value("y", [2])],
        [],
    )
    model.graph.node[0].domain = "opaque"
    model.functions.append(
        h.make_function(
            "opaque",
            "Identity",
            ["x"],
            ["out"],
            [
                h.make_node("Constant", [], ["bias"], value=nh.from_array(np.float32(0.25))),
                h.make_node("Add", ["x", "bias"], ["out"]),
            ],
            [h.make_opsetid("", 19)],
        )
    )
    model.opset_import.append(h.make_opsetid("opaque", 1))
    feeds = {"x": np.asarray([0.5, -1], np.float32)}
    expected = _checked_cpu_outputs(model, feeds)[0]
    changed = _apply_force_fp16_activations(model)
    np.testing.assert_array_equal(_checked_cpu_outputs(changed, feeds)[0], expected)


def test_fp16_retains_captured_values_and_control_flow_types():
    """Keep a captured FP32 result available to unchanged nested graph bodies."""
    branch = h.make_graph(
        [h.make_node("Identity", ["middle"], ["branch_value"])],
        "capture",
        [],
        [_value("branch_value", [2])],
    )
    model = _model(
        [
            h.make_node("Relu", ["x"], ["middle"]),
            h.make_node("Neg", ["middle"], ["y"]),
            h.make_node("If", ["condition"], ["captured"], then_branch=branch, else_branch=branch),
        ],
        [_value("x", [2]), _value("condition", [], T.BOOL)],
        [_value("y", [2]), _value("captured", [2])],
        [],
    )
    feeds = {"x": np.asarray([1, -2], np.float32), "condition": np.asarray(True)}
    expected = _checked_cpu_outputs(model, feeds)
    changed = _apply_force_fp16_activations(model)
    for actual, wanted in zip(_checked_cpu_outputs(changed, feeds), expected):
        np.testing.assert_array_equal(actual, wanted)


def test_fp16_float_constant_keeps_valid_consumer_types():
    """Give an unfurled FLOAT Constant a private FP16 data view."""
    model = _model(
        [
            h.make_node(
                "Constant", [], ["bias"], value=nh.from_array(np.asarray([0.5, 0.25], np.float32))
            ),
            h.make_node("Add", ["x", "bias"], ["y"]),
        ],
        [_value("x", [2])],
        [_value("y", [2])],
        [],
    )
    feeds = {"x": np.asarray([1, 2], np.float32)}
    expected = _checked_cpu_outputs(model, feeds)[0]
    changed = _apply_force_fp16_activations(model)
    np.testing.assert_array_equal(_checked_cpu_outputs(changed, feeds)[0], expected)


def _terminal_model(channels=8193, *, groups=1, quantized=False, batch=1):
    """Build the exact terminal projection pattern with independently checkable values."""
    data_channels = groups
    weights = nh.from_array(
        np.ones((channels, 1, 1, 1), np.uint8)
        if quantized
        else np.full((channels, 1, 1, 1), 0.25, np.float32),
        "weight",
    )
    nodes = []
    initializers = [weights, nh.from_array(np.asarray([0], np.int64), "axes")]
    if quantized:
        initializers.extend(
            [nh.from_array(np.float32(0.25), "scale"), nh.from_array(np.uint8(0), "zero")]
        )
        nodes.append(h.make_node("DequantizeLinear", ["weight", "scale", "zero"], ["dequantized"]))
    nodes.extend(
        [
            h.make_node(
                "Conv", ["x", "dequantized" if quantized else "weight"], ["conv"], group=groups
            ),
            h.make_node("Transpose", ["conv"], ["transposed"], perm=[0, 3, 2, 1]),
            h.make_node("Concat", ["transposed"], ["concat1"], axis=0),
            h.make_node("ReduceSum", ["concat1", "axes"], ["reduced"], keepdims=1),
            h.make_node("Concat", ["reduced"], ["concat2"], axis=2),
            h.make_node("Squeeze", ["concat2", "axes"], ["y"]),
        ]
    )
    return _model(
        nodes,
        [_value("x", [batch, data_channels, 1, 1])],
        [_value("y", [1, 1, channels])],
        initializers,
    )


@pytest.mark.parametrize("quantized", [False, True])
def test_terminal_projection_preserves_nontrivial_batch_reduction(quantized):
    """Retain batch summation instead of replacing it with singleton projection chunks."""
    model = _terminal_model(batch=2, quantized=quantized)
    feeds = {"x": np.asarray([2, 6], np.float32).reshape(2, 1, 1, 1)}
    expected = _checked_cpu_outputs(model, feeds)[0]
    converted = _apply_force_fp16_activations(model)
    assert sum(node.op_type == "Conv" for node in converted.graph.node) == 1
    np.testing.assert_array_equal(_checked_cpu_outputs(converted, feeds)[0], expected)


@pytest.mark.parametrize(
    "owner", ["grouped", "runtime_weight", "public_axes", "legacy", "constant"]
)
def test_terminal_projection_preserves_native_owners_and_group_layout(owner):
    """Chunk only private fixed projections and preserve control owners and grouped channels."""
    grouped = owner == "grouped"
    model = _terminal_model(
        channels=8194 if grouped else 8193,
        groups=2 if grouped else 1,
        quantized=owner == "runtime_weight",
    )
    if owner == "runtime_weight":
        model.graph.input.append(_value("weight", [8193, 1, 1, 1], T.UINT8))
    if owner == "public_axes":
        model.graph.output.append(_value("axes", [1], T.INT64))
    if owner == "legacy":
        model.ir_version = 3
        model.graph.input.extend(
            _value(t.name, list(t.dims), t.data_type) for t in model.graph.initializer
        )
    constant = None
    if owner == "constant":
        model.graph.node.insert(
            0,
            h.make_node(
                "Constant", [], ["integer"], value=nh.from_array(np.arange(64, dtype=np.uint8))
            ),
        )
        constant = model.graph.node[0]
    data = (
        np.asarray([[[[2]], [[8]]]], np.float32) if grouped else np.asarray([[[[2]]]], np.float32)
    )
    feeds = {"x": data}
    onnx.checker.check_model(model, full_check=True)
    expected = ReferenceEvaluator(model).run(None, feeds)
    override = (
        {"x": data, "weight": np.full((8193, 1, 1, 1), 2, np.uint8)}
        if owner == "runtime_weight"
        else None
    )
    override_expected = (
        ReferenceEvaluator(model).run(None, override) if override is not None else None
    )
    changed = _apply_force_fp16_activations(model)
    if owner in {"grouped", "runtime_weight", "public_axes"}:
        assert sum(node.op_type == "Conv" for node in changed.graph.node) == 1
    if owner == "legacy":
        assert not {"weight", "axes"} & {value.name for value in changed.graph.input}
    if constant is not None:
        assert next(node for node in changed.graph.node if node.op_type == "Constant") is constant
    onnx.checker.check_model(changed, full_check=True)
    for actual, wanted in zip(ReferenceEvaluator(changed).run(None, feeds), expected):
        np.testing.assert_array_equal(actual, wanted)
    if override is not None:
        for actual, wanted in zip(
            ReferenceEvaluator(changed).run(None, override), override_expected
        ):
            np.testing.assert_array_equal(actual, wanted)


def test_fp16_public_legacy_initializer_uses_private_declaration():
    """Rename the legacy initializer declaration together with its private FP16 payload."""
    model = _model(
        [],
        [_value("weight", [2])],
        [_value("weight", [2])],
        [nh.from_array(np.asarray([1, 2], np.float32), "weight")],
    )
    model.ir_version = 3
    changed = _apply_force_fp16_activations(model)
    onnx.checker.check_model(changed, full_check=True)
    np.testing.assert_array_equal(
        _checked_cpu_outputs(changed, {})[0], np.asarray([1, 2], np.float32)
    )


@pytest.mark.parametrize("legacy", [False, True])
def test_fp16_qdq_upgrade_migrates_nested_reduce_axes(legacy):
    """Migrate axes in nested graphs that inherit the upgraded main operator set."""
    branch = h.make_graph(
        [h.make_node("ReduceMean", ["x"], ["branch_value"], axes=[1], keepdims=1)],
        "nested_reduce",
        [],
        [_value("branch_value", [2, 1])],
    )
    model = _model(
        [
            h.make_node("QuantizeLinear", ["x", "scale", "zero"], ["q"]),
            h.make_node("DequantizeLinear", ["q", "scale", "zero"], ["y"]),
            h.make_node("If", ["condition"], ["reduced"], then_branch=branch, else_branch=branch),
        ],
        [_value("x", [2, 2]), _value("condition", [], T.BOOL)],
        [_value("y", [2, 2]), _value("reduced", [2, 1])],
        [nh.from_array(np.float32(0.1), "scale"), nh.from_array(np.int8(0), "zero")],
        opset=17,
    )
    if legacy:
        model.ir_version = 3
        model.graph.input.extend(
            _value(t.name, list(t.dims), t.data_type) for t in model.graph.initializer
        )
    feeds = {"x": np.asarray([[0.2, 0.2], [0.4, 0.4]], np.float32), "condition": np.asarray(True)}
    expected = _checked_cpu_outputs(model, feeds)
    changed = _apply_force_fp16_activations(model)
    onnx.checker.check_model(changed, full_check=True)
    for got, wanted in zip(ReferenceEvaluator(changed).run(None, feeds), expected):
        np.testing.assert_allclose(got, wanted, rtol=2e-3, atol=1e-4)


def test_fp16_legacy_qdq_schema_refuses_before_mutation():
    """Refuse legacy operator-set migration that needs adapters beyond the supported range."""
    model = _model(
        [
            h.make_node("Clip", ["x"], ["clipped"], min=0.0, max=1.0),
            h.make_node("QuantizeLinear", ["clipped", "scale", "zero"], ["y"]),
        ],
        [_value("x", [2])],
        [_value("y", [2], T.INT8)],
        [nh.from_array(np.float32(0.1), "scale"), nh.from_array(np.int8(0), "zero")],
        opset=10,
    )
    onnx.checker.check_model(model, full_check=True)
    before = model.SerializeToString()
    with pytest.raises(ValueError, match="opset"):
        _apply_force_fp16_activations(model)
    assert model.SerializeToString() == before


@pytest.mark.parametrize("operation", ["Range", "NonMaxSuppression"])
def test_fp16_preserves_operator_schemas_without_half_inputs(operation):
    """Keep mandatory FP32 schema bindings valid instead of forcing unsupported HALF inputs."""
    if operation == "Range":
        model = _model(
            [h.make_node("Range", ["start", "limit", "delta"], ["y"])],
            [_value("start", [])],
            [_value("y", [4])],
            [nh.from_array(np.float32(4), "limit"), nh.from_array(np.float32(1), "delta")],
        )
        feeds = {"start": np.asarray(0, np.float32)}
    else:
        model = _model(
            [h.make_node("NonMaxSuppression", ["boxes", "scores", "count", "iou"], ["y"])],
            [_value("boxes", [1, 2, 4]), _value("scores", [1, 1, 2])],
            [_value("y", [None, 3], T.INT64)],
            [nh.from_array(np.int64(1), "count"), nh.from_array(np.float32(0.5), "iou")],
        )
        feeds = {
            "boxes": np.asarray([[[0, 0, 1, 1], [2, 2, 3, 3]]], np.float32),
            "scores": np.asarray([[[0.8, 0.5]]], np.float32),
        }
    expected = _checked_cpu_outputs(model, feeds)[0]
    changed = _apply_force_fp16_activations(model)
    np.testing.assert_array_equal(_checked_cpu_outputs(changed, feeds)[0], expected)


@pytest.mark.parametrize("operation", ["ConstantOfShape", "RandomNormal", "EyeLike"])
def test_fp16_preserves_explicit_generated_float_types(operation):
    """Keep generated FLOAT values typed correctly and provide private HALF consumer views."""
    inputs = []
    initializers = []
    feeds = {}
    shape = [2, 2]
    if operation == "ConstantOfShape":
        initializers = [nh.from_array(np.asarray(shape, np.int64), "shape")]
        generated = h.make_node(
            operation, ["shape"], ["generated"], value=nh.from_array(np.asarray([0.25], np.float32))
        )
    elif operation == "RandomNormal":
        generated = h.make_node(operation, [], ["generated"], shape=shape, dtype=T.FLOAT, seed=42.0)
    else:
        inputs = [_value("x", shape)]
        feeds = {"x": np.zeros(shape, np.float32)}
        generated = h.make_node(operation, ["x"], ["generated"], dtype=T.FLOAT)
    model = _model(
        [generated, h.make_node("Relu", ["generated"], ["y"])],
        inputs,
        [_value("y", shape)],
        initializers,
    )
    expected_cpu = _checked_cpu_outputs(model, feeds)[0]
    expected = ReferenceEvaluator(model).run(None, feeds)[0].astype(np.float16).astype(np.float32)
    changed = _apply_force_fp16_activations(model)
    onnx.checker.check_model(changed, full_check=True)
    np.testing.assert_array_equal(ReferenceEvaluator(changed).run(None, feeds)[0], expected)
    # CPU fallback can retain FP32 arithmetic even when the graph requests HALF.
    np.testing.assert_allclose(
        _checked_cpu_outputs(changed, feeds)[0], expected_cpu, rtol=1e-3, atol=1e-3
    )


def test_fp16_lpnorm_allocates_support_node_names():
    """Reserve normalization support-node names across existing graph owners."""
    model = _model(
        [
            h.make_node(
                "Constant",
                [],
                ["integer"],
                name="normalize_square",
                value=nh.from_array(np.asarray([1], np.int8)),
            ),
            h.make_node("LpNormalization", ["x"], ["norm"], name="normalize", p=2, axis=-1),
            h.make_node("Mul", ["norm", "gamma"], ["y"]),
        ],
        [_value("x", [1, 1, 1, 2])],
        [_value("y", [1, 1, 1, 2])],
        [nh.from_array(np.ones([1, 1, 1, 2], np.float32), "gamma")],
    )
    changed = _apply_fp16_lpnorm(model)
    names = [node.name for node in changed.graph.node if node.name]
    assert len(names) == len(set(names))
    onnx.checker.check_model(changed, full_check=True)


@pytest.mark.parametrize("dtype", [T.UINT8, T.FLOAT])
def test_fp16_skips_conversion_for_preserved_runtime_defaults_only(dtype):
    """Skip weight conversion when the model contains only preserved defaults."""
    numpy_dtype = np.uint8 if dtype == T.UINT8 else np.float32
    model = _model(
        [h.make_node("Cast", ["weight"], ["y"], to=T.FLOAT)],
        [_value("weight", [2], dtype)],
        [_value("y", [2])],
        [nh.from_array(np.asarray([1, 2], numpy_dtype), "weight")],
    )
    weight = model.graph.initializer[0]
    changed = _apply_force_fp16_activations(model)
    assert next(t for t in changed.graph.initializer if t.name == "weight") is weight
    np.testing.assert_array_equal(
        _checked_cpu_outputs(changed, {})[0], np.asarray([1, 2], np.float32)
    )
    np.testing.assert_array_equal(
        _checked_cpu_outputs(changed, {"weight": np.asarray([2, 3], numpy_dtype)})[0],
        np.asarray([2, 3], np.float32),
    )


@pytest.mark.parametrize("operation", ["QuantizeLinear", "DequantizeLinear"])
def test_fp16_supported_contrib_quantizers_keep_half_interface(operation):
    """Retain the supported contrib quantizers' HALF scales without treating them as opaque."""
    initializers = [nh.from_array(np.float32(0.5), "scale"), nh.from_array(np.int8(0), "zero")]
    if operation == "QuantizeLinear":
        nodes = [h.make_node(operation, ["x", "scale", "zero"], ["y"])]
        inputs = [_value("x", [2])]
        outputs = [_value("y", [2], T.INT8)]
        feeds = {"x": np.asarray([-1, 2], np.float32)}
    else:
        initializers.append(nh.from_array(np.asarray([-2, 4], np.int8), "quantized"))
        nodes = [
            h.make_node(operation, ["quantized", "scale", "zero"], ["middle"]),
            h.make_node("Relu", ["middle"], ["y"]),
        ]
        inputs = []
        outputs = [_value("y", [2])]
        feeds = {}
    model = _model(nodes, inputs, outputs, initializers, opset=17)
    model.graph.node[0].domain = "com.microsoft"
    model.opset_import.append(h.make_opsetid("com.microsoft", 1))
    expected = _checked_cpu_outputs(model, feeds)[0]
    changed = _apply_force_fp16_activations(model)
    assert next(t for t in changed.graph.initializer if t.name == "scale").data_type == T.FLOAT16
    assert next(op for op in changed.opset_import if not op.domain).version == 17
    np.testing.assert_array_equal(_checked_cpu_outputs(changed, feeds)[0], expected)


def test_terminal_projection_chunks_supported_contrib_dequantization():
    """Keep the supported contrib dequantizer on each private bounded weight chunk."""
    model = _terminal_model(quantized=True)
    model.graph.node[0].domain = "com.microsoft"
    model.opset_import.append(h.make_opsetid("com.microsoft", 1))
    feeds = {"x": np.asarray([[[[2]]]], np.float32)}
    expected = _checked_cpu_outputs(model, feeds)[0]
    changed = _apply_force_fp16_activations(model)
    assert sum(node.op_type == "Conv" for node in changed.graph.node) == 2
    np.testing.assert_array_equal(_checked_cpu_outputs(changed, feeds)[0], expected)


def test_fp16_supported_contrib_gelu_keeps_upstream_half_precision():
    """Keep a supported contrib activation and its standard normalization in HALF."""
    model = h.make_model(
        h.make_graph(
            [
                h.make_node("LayerNormalization", ["x", "scale", "bias"], ["normalized"], axis=-1),
                h.make_node("Gelu", ["normalized"], ["y"], domain="com.microsoft"),
            ],
            "contrib_gelu",
            [_value("x", [2, 2])],
            [_value("y", [2, 2])],
            [
                nh.from_array(np.ones(2, np.float32), "scale"),
                nh.from_array(np.zeros(2, np.float32), "bias"),
            ],
        ),
        opset_imports=[h.make_opsetid("", 17), h.make_opsetid("com.microsoft", 1)],
    )
    model.ir_version = 9
    model = onnx.shape_inference.infer_shapes(model)
    feeds = {"x": np.asarray([[1, 3], [2, 4]], np.float32)}
    expected = _checked_cpu_outputs(model, feeds)[0]
    changed = _apply_force_fp16_activations(model)
    assert all(t.data_type == T.FLOAT16 for t in changed.graph.initializer)
    assert next(op for op in changed.opset_import if not op.domain).version == 17
    np.testing.assert_allclose(
        _checked_cpu_outputs(changed, feeds)[0], expected, rtol=2e-3, atol=1e-4
    )
