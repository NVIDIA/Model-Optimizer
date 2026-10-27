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

from importlib import import_module

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, inliner, numpy_helper
from onnx.reference import ReferenceEvaluator

from modelopt.onnx.graph_surgery.dla_transforms.dla_model_boundaries import (
    _apply_fix_instancenorm_channels,
    _apply_non4d_model_inputs,
    _apply_non4d_model_outputs,
    _apply_remove_intermediary_squeeze_unsqueeze,
)
from modelopt.onnx.graph_surgery.make_dla_compatible import _transform_make_dla_compatible


@pytest.mark.parametrize("identity", ["custom", "local", "overload"])
@pytest.mark.parametrize(
    ("operation", "entrypoint"),
    [
        ("Squeeze", "pipeline"),
        ("Unsqueeze", "pipeline"),
        ("Concat", "pipeline"),
        ("Squeeze", "cleanup"),
        ("Unsqueeze", "cleanup"),
    ],
)
def test_operator_lookalikes_keep_function_semantics(identity, operation, entrypoint):
    """Execute opaque lookalikes without rewriting their builtin-named calls or controls."""
    domain = "opaque.boundary" if identity == "custom" else ""
    overload = "custom" if identity == "overload" else ""
    inputs = ["x"] if operation == "Concat" else ["x", "axes"]
    function = helper.make_function(
        domain,
        operation,
        inputs,
        ["y"],
        [helper.make_node("Neg", ["x"], ["y"])],
        [helper.make_opsetid("", 21)],
        attributes=["axis"] if operation == "Concat" else [],
    )
    function.overload = overload
    opaque = helper.make_node(
        operation,
        ["middle" if entrypoint == "cleanup" else "x", *inputs[1:]],
        ["hidden" if entrypoint == "cleanup" else "y"],
        name="opaque",
        domain=domain,
        **({"axis": 1} if operation == "Concat" else {}),
    )
    opaque.overload = overload
    nodes = (
        [
            helper.make_node("Identity", ["x"], ["middle"]),
            opaque,
            helper.make_node("Identity", ["hidden"], ["y"]),
        ]
        if entrypoint == "cleanup"
        else [opaque]
    )
    model = _make_model(
        nodes,
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 3])],
        [] if operation == "Concat" else [numpy_helper.from_array(np.array([], np.int64), "axes")],
    )
    model.functions.append(function)
    if domain:
        model.opset_import.append(helper.make_opsetid(domain, 1))
    if entrypoint == "cleanup":
        model.graph.value_info.extend(
            helper.make_tensor_value_info(name, TensorProto.FLOAT, [2, 3])
            for name in ("middle", "hidden")
        )
    before = next(n for n in model.graph.node if n.name == "opaque").SerializeToString()
    feeds = {"x": np.arange(6, dtype=np.float32).reshape(2, 3)}
    expected = _checked_run(model, feeds)[0]
    converted = (
        _transform_make_dla_compatible(model, verbose=False)
        if entrypoint == "pipeline"
        else _apply_remove_intermediary_squeeze_unsqueeze(model, rank_converted=True)
    )
    assert next(n for n in converted.graph.node if n.name == "opaque").SerializeToString() == before
    np.testing.assert_array_equal(_checked_run(converted, feeds)[0], expected)


@pytest.mark.parametrize("operation", ["Reshape", "Expand", "Tile"])
@pytest.mark.parametrize("tail", [False, True])
def test_symbolic_native_controls_do_not_gain_output_adapters(operation, tail):
    """Keep static native outputs when symbolic data was not promoted, including unary tails."""
    target = {"Reshape": [3, 2], "Expand": [2, 3], "Tile": [0, 1]}[operation]
    output_shape = {"Reshape": [3, 2], "Expand": [2, 3], "Tile": [0, 3]}[operation]
    nodes = [helper.make_node(operation, ["x", "control"], ["middle" if tail else "y"])]
    if tail:
        nodes.append(helper.make_node("Identity", ["middle"], ["y"]))
    model = _make_model(
        nodes,
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, ["N", 3])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, output_shape)],
        [numpy_helper.from_array(np.array(target, np.int64), "control")],
    )
    feeds = {"x": np.arange(6, dtype=np.float32).reshape(2, 3)}
    expected = _checked_run(model, feeds)[0]
    converted = _transform_make_dla_compatible(model, verbose=False)
    np.testing.assert_array_equal(_checked_run(converted, feeds)[0], expected)


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


@pytest.mark.parametrize("identity", ["custom", "local", "overload"])
@pytest.mark.parametrize("tail", [False, True])
def test_generic_opaque_transpose_keeps_native_shared_input(identity, tail):
    """Preserve opaque function layouts while a separate standard branch promotes the same input."""
    domain = "opaque.boundary" if identity == "custom" else ""
    operation = "OpaqueTranspose" if identity == "custom" else "Abs"
    overload = "transpose" if identity == "overload" else ""
    function = helper.make_function(
        domain,
        operation,
        ["data"],
        ["result"],
        [helper.make_node("Transpose", ["data"], ["result"], perm=[1, 0])],
        [helper.make_opsetid("", 21)],
        overload=overload,
    )
    opaque = helper.make_node(
        operation, ["x"], ["middle" if tail else "y"], domain=domain, overload=overload
    )
    nodes = [helper.make_node("Relu", ["x"], ["first"]), opaque]
    if tail:
        nodes.append(helper.make_node("Relu", ["middle"], ["y"]))
    model = _make_model(
        nodes,
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 2])],
        [
            helper.make_tensor_value_info("first", TensorProto.FLOAT, [2, 2]),
            helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 2]),
        ],
    )
    model.functions.append(function)
    if domain:
        model.opset_import.append(helper.make_opsetid(domain, 1))
    feeds = {"x": np.arange(-1, 3, dtype=np.float32).reshape(2, 2)}
    onnx.checker.check_model(model, full_check=True)
    expected = _checked_run(inliner.inline_local_functions(model), feeds)
    np.testing.assert_array_equal(
        expected[1], np.maximum(feeds["x"].T, 0) if tail else feeds["x"].T
    )
    original = opaque.SerializeToString()
    converted = _transform_make_dla_compatible(model, verbose=False)
    call = next(
        node
        for node in converted.graph.node
        if node.op_type == operation and node.domain == domain and node.overload == overload
    )
    assert call.SerializeToString() == original
    for actual, reference in zip(
        _checked_run(inliner.inline_local_functions(converted), feeds), expected
    ):
        np.testing.assert_array_equal(actual, reference)


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


@pytest.mark.parametrize("operation", ["Softmax", "LogSoftmax"])
@pytest.mark.parametrize("shape", [[2, 3], [2, 3, 4]])
@pytest.mark.parametrize("axis", [0, 1, -1, None])
@pytest.mark.parametrize("opset", [13, 17])
def test_boundary_softmax_preserves_native_axes(operation, shape, axis, opset):
    """Preserve explicit and schema-default normalization axes across input promotion."""
    model = _make_model(
        [helper.make_node(operation, ["x"], ["y"], **({} if axis is None else {"axis": axis}))],
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, shape)],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, shape)],
    )
    model.opset_import[0].version = opset
    feeds = {"x": np.arange(np.prod(shape), dtype=np.float32).reshape(shape) / 10}
    expected = _checked_run(model, feeds)[0]
    converted = _transform_make_dla_compatible(model, verbose=False)
    np.testing.assert_array_equal(_checked_run(converted, feeds)[0], expected)


@pytest.mark.parametrize("axis", [1, -1])
def test_boundary_layer_normalization_preserves_native_axis(axis):
    """Preserve LayerNormalization reductions after adding leading singleton dimensions."""
    model = _make_model(
        [helper.make_node("LayerNormalization", ["x", "scale"], ["y"], axis=axis)],
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 3])],
        [numpy_helper.from_array(np.ones(3, np.float32), "scale")],
    )
    feeds = {"x": np.arange(6, dtype=np.float32).reshape(2, 3)}
    expected = _checked_run(model, feeds)[0]
    converted = _transform_make_dla_compatible(model, verbose=False)
    np.testing.assert_allclose(_checked_run(converted, feeds)[0], expected, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize(
    ("operation", "axis"),
    [
        ("Softmax", -1),
        ("Softmax", 1),
        ("LogSoftmax", None),
        ("Transpose", None),
        ("Transpose", 1),
        ("Conv", None),
        ("opaque", None),
        ("Shape", None),
        ("InstanceNormalization", None),
    ],
)
@pytest.mark.parametrize("unary", [False, True])
def test_normalization_preserves_rank_sensitive_and_opaque_consumers(operation, axis, unary):
    """Execute native and promoted consumers alongside a shared public normalization output."""
    shape = [2, 3, 4]
    nodes = [helper.make_node("InstanceNormalization", ["x", "scale", "bias"], ["normalized"])]
    current = "normalized"
    if unary:
        nodes.append(helper.make_node("Relu", [current], ["activated"]))
        current = "activated"
    controls = [
        numpy_helper.from_array(np.ones(3, np.float32), "scale"),
        numpy_helper.from_array(np.zeros(3, np.float32), "bias"),
    ]
    output_shape, dtype, functions = shape, TensorProto.FLOAT, []
    if operation in {"Softmax", "LogSoftmax"}:
        nodes.append(
            helper.make_node(
                operation, [current], ["y"], **({"axis": axis} if axis is not None else {})
            )
        )
    elif operation == "Transpose":
        nodes.append(
            helper.make_node(
                operation, [current], ["y"], **({"perm": [0, 2, 1]} if axis is not None else {})
            )
        )
        output_shape = [2, 4, 3] if axis is not None else [4, 3, 2]
    elif operation == "Conv":
        controls.append(
            numpy_helper.from_array(np.arange(18, dtype=np.float32).reshape(2, 3, 3) / 18, "kernel")
        )
        nodes.append(helper.make_node("Conv", [current, "kernel"], ["y"]))
        output_shape = [2, 2, 2]
    elif operation == "opaque":
        functions.append(
            helper.make_function(
                "opaque.norm",
                "NativeLayout",
                ["value"],
                ["result"],
                [helper.make_node("Transpose", ["value"], ["result"], perm=[0, 2, 1])],
                [helper.make_opsetid("", 21)],
            )
        )
        nodes.append(helper.make_node("NativeLayout", [current], ["y"], domain="opaque.norm"))
        output_shape = [2, 4, 3]
    elif operation == "InstanceNormalization":
        nodes.append(helper.make_node(operation, [current, "scale", "bias"], ["y"]))
    else:
        nodes.append(helper.make_node("Shape", [current], ["y"]))
        output_shape, dtype = [3], TensorProto.INT64
    outputs = [
        helper.make_tensor_value_info("y", dtype, output_shape),
        helper.make_tensor_value_info("normalized", TensorProto.FLOAT, shape),
    ]
    if operation == "Transpose" and axis is None:
        for name in ("normalized_squeeze_input", "normalized_squeeze"):
            controls.append(numpy_helper.from_array(np.array(1, np.int64), name))
            outputs.append(helper.make_tensor_value_info(name, TensorProto.INT64, []))
    if operation in {"Conv", "opaque", "Shape"}:
        nodes.append(helper.make_node("Softmax", [current], ["side"], axis=-1))
        outputs.append(helper.make_tensor_value_info("side", TensorProto.FLOAT, shape))
    model = _make_model(
        nodes, [helper.make_tensor_value_info("x", TensorProto.FLOAT, shape)], outputs, controls
    )
    model.functions.extend(functions)
    if functions:
        model.opset_import.append(helper.make_opsetid("opaque.norm", 1))
    feeds = {"x": np.linspace(-1, 2, 24, dtype=np.float32).reshape(shape)}
    expected = _checked_run(inliner.inline_local_functions(model), feeds)
    converted = _transform_make_dla_compatible(model, verbose=False)
    if operation == "Conv":
        assert list(
            next(tensor for tensor in converted.graph.initializer if tensor.name == "kernel").dims
        ) == [2, 3, 3]
    for actual, reference in zip(
        _checked_run(inliner.inline_local_functions(converted), feeds), expected
    ):
        np.testing.assert_allclose(actual, reference, rtol=1e-5, atol=1e-6)


def test_normalization_refuses_unprepared_consumers_before_mutation():
    """Refuse a rank-sensitive consumer before widening any eligible normalization node."""
    shape = [2, 3, 4]
    model = _make_model(
        [
            helper.make_node("InstanceNormalization", ["x", "scale", "bias"], ["normalized"]),
            helper.make_node("Hardmax", ["normalized"], ["y"], axis=1),
        ],
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, shape)],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, shape)],
        [
            numpy_helper.from_array(np.ones(3, np.float32), "scale"),
            numpy_helper.from_array(np.zeros(3, np.float32), "bias"),
        ],
    )
    _checked_run(model, {"x": np.arange(24, dtype=np.float32).reshape(shape)})
    before = model.SerializeToString()
    with pytest.raises(ValueError, match="Unsupported normalization consumer"):
        _apply_fix_instancenorm_channels(model)
    assert model.SerializeToString() == before


@pytest.mark.parametrize("batch", [1, 2, 3])
@pytest.mark.parametrize("operation", ["Add", "Mul"])
@pytest.mark.parametrize("tail", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
def test_normalization_binary_branch_preserves_broadcast_layout(batch, operation, tail, reverse):
    """Keep shared normalized public values and binary residual inputs in compatible layouts."""
    shape = [batch, 3, 4]
    normalized = "tail" if tail else "normalized"
    nodes = [
        helper.make_node(
            "InstanceNormalization", ["x", "scale", "bias"], ["normalized"], epsilon=1e-4
        )
    ]
    if tail:
        nodes.append(helper.make_node("Tanh", ["normalized"], ["tail"]))
    nodes.append(
        helper.make_node(operation, [normalized, "x"] if reverse else ["x", normalized], ["y"])
    )
    model = _make_model(
        nodes,
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, shape)],
        [
            helper.make_tensor_value_info("y", TensorProto.FLOAT, shape),
            helper.make_tensor_value_info(normalized, TensorProto.FLOAT, shape),
        ],
        [
            numpy_helper.from_array(np.asarray([0.5, 1, 1.5], np.float32), "scale"),
            numpy_helper.from_array(np.asarray([0.1, -0.2, 0.3], np.float32), "bias"),
        ],
    )
    feeds = {"x": np.arange(np.prod(shape), dtype=np.float32).reshape(shape) / 8}
    expected = _checked_run(model, feeds)
    converted = _transform_make_dla_compatible(model, verbose=False)
    for actual, reference in zip(_checked_run(converted, feeds), expected):
        np.testing.assert_allclose(actual, reference, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("width", [8, 12])
@pytest.mark.parametrize("axis", [-1, 4])
@pytest.mark.parametrize("quantized", [False, True])
@pytest.mark.parametrize("preparation", ["direct", "intermediate", "constant"])
def test_early_registered_reshape_reduction_is_not_rejected(
    width, axis, quantized, preparation, monkeypatch
):
    """Allow reductions consumed by actual earlier reshape-pattern stages without a native reducer stage."""
    pipeline = import_module(_transform_make_dla_compatible.__module__)
    monkeypatch.setattr(
        pipeline,
        "DLA_TRANSFORMS",
        tuple(
            (name, transform)
            for name, transform in pipeline.DLA_TRANSFORMS
            if name != "StandaloneReduceOperations"
        ),
    )
    shape = [1, 1, 2, width]
    controls = [
        numpy_helper.from_array(np.asarray([1, 1, 2, width // 4, 4], np.int64), "target"),
        numpy_helper.from_array(np.asarray([axis], np.int64), "axes"),
    ]
    nodes = [helper.make_node("Reshape", ["x", "target"], ["expanded"])]
    current = "expanded"
    if quantized:
        controls += [
            numpy_helper.from_array(np.float32(0.125), "scale"),
            numpy_helper.from_array(np.int8(0), "zero"),
        ]
        nodes += [
            helper.make_node("QuantizeLinear", [current, "scale", "zero"], ["q"]),
            helper.make_node("DequantizeLinear", ["q", "scale", "zero"], ["dq"]),
        ]
        current = "dq"
    nodes.append(helper.make_node("ReduceSum", [current, "axes"], ["y"], keepdims=0))
    if preparation == "intermediate":
        nodes[0].input[0] = "hidden"
        nodes.insert(0, helper.make_node("Identity", ["x"], ["hidden"]))
    elif preparation == "constant":
        nodes = [
            helper.make_node("Constant", [], [tensor.name], value=tensor) for tensor in controls
        ] + nodes
        controls = []
    model = _make_model(
        nodes,
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, shape)],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1, 2, width // 4])],
        controls,
    )
    feeds = {"x": np.arange(np.prod(shape), dtype=np.float32).reshape(shape) / 8}
    expected = _checked_run(model, feeds)[0]
    converted = _transform_make_dla_compatible(model, verbose=False, qdq_quantized_dtypes=[])
    np.testing.assert_array_equal(_checked_run(converted, feeds)[0], expected)


@pytest.mark.parametrize(
    "operation", ["ReduceSum", "Gather", "Slice", "Split", "ArgMax", "Transpose"]
)
def test_input_promotion_requires_registered_axis_conversion(operation, monkeypatch):
    """Reject valid native operators before boundary mutation when their rank stage is absent."""
    pipeline = import_module(_transform_make_dla_compatible.__module__)
    monkeypatch.setattr(
        pipeline,
        "DLA_TRANSFORMS",
        (
            ("ConstantsToInitializers", pipeline._apply_constants_to_initializers),
            ("ShapeInference", pipeline.infer_shapes),
            ("Non4dModelInputs", _apply_non4d_model_inputs),
            ("Non4dModelOutputs", _apply_non4d_model_outputs),
        ),
    )
    inputs, attributes, output_shape = ["x"], {}, [2, 1]
    controls = []
    dtype = TensorProto.FLOAT
    if operation == "Transpose":
        attributes, output_shape = {"perm": [1, 0]}, [3, 2]
    elif operation == "ArgMax":
        attributes, dtype = {"axis": 1, "keepdims": 1}, TensorProto.INT64
    else:
        values = {"ReduceSum": [1], "Gather": [0], "Split": [3], "Slice": [0]}[operation]
        controls.append(numpy_helper.from_array(np.asarray(values, np.int64), "control"))
        inputs.append("control")
        if operation == "ReduceSum":
            attributes = {"keepdims": 1}
        elif operation == "Gather":
            attributes, output_shape = {"axis": 0}, [1, 3]
        elif operation == "Split":
            attributes, output_shape = {"axis": 1}, [2, 3]
        else:
            controls += [
                numpy_helper.from_array(np.asarray([1], np.int64), "end"),
                numpy_helper.from_array(np.asarray([1], np.int64), "axes"),
            ]
            inputs += ["end", "axes"]
    model = _make_model(
        [helper.make_node(operation, inputs, ["y"], **attributes)],
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3])],
        [helper.make_tensor_value_info("y", dtype, output_shape)],
        controls,
    )
    _checked_run(model, {"x": np.arange(6, dtype=np.float32).reshape(2, 3)})
    before = model.SerializeToString()
    with pytest.raises(ValueError, match="Input rank conversion"):
        _transform_make_dla_compatible(model, verbose=False)
    assert model.SerializeToString() == before


@pytest.mark.parametrize("tail", [False, True])
def test_ineligible_gemm_keeps_native_boundary_and_weights(tail):
    """Retain native matrix Gemm semantics when convolution lowering declines its coefficients."""
    nodes = [helper.make_node("Gemm", ["x", "w", "b"], ["hidden" if tail else "y"], alpha=0.5)]
    if tail:
        nodes.append(helper.make_node("Relu", ["hidden"], ["y"]))
    model = _make_model(
        nodes,
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 4])],
        [
            numpy_helper.from_array(np.arange(12, dtype=np.float32).reshape(3, 4), "w"),
            numpy_helper.from_array(np.ones(4, np.float32), "b"),
        ],
    )
    feeds = {"x": np.arange(6, dtype=np.float32).reshape(2, 3)}
    expected = _checked_run(model, feeds)[0]
    converted = _transform_make_dla_compatible(model, verbose=False)
    np.testing.assert_array_equal(_checked_run(converted, feeds)[0], expected)
    assert [list(tensor.dims) for tensor in converted.graph.initializer] == [[3, 4], [4]]


@pytest.mark.parametrize("source", ["initializer", "constant", "runtime"])
@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("tail", [False, True])
def test_scalar_gather_requires_registered_rank_conversion(source, axis, tail, monkeypatch):
    """Refuse unprepared scalar Gather before the actual pipeline changes caller state."""
    pipeline = import_module(_transform_make_dla_compatible.__module__)
    monkeypatch.setattr(
        pipeline,
        "DLA_TRANSFORMS",
        tuple((name, transform) for name, transform in pipeline.DLA_TRANSFORMS if name != "Gather"),
    )
    shape = [2, 3, 4, 4] if axis == 0 else [1, 3, 4, 4]
    output_shape = shape[:axis] + shape[axis + 1 :]
    nodes = [helper.make_node("Gather", ["x", "indices"], ["hidden" if tail else "y"], axis=axis)]
    if tail:
        nodes.append(helper.make_node("Identity", ["hidden"], ["y"]))
    tensor = numpy_helper.from_array(np.int64(1), "indices")
    inputs = [helper.make_tensor_value_info("x", TensorProto.FLOAT, shape)]
    if source == "constant":
        nodes.insert(0, helper.make_node("Constant", [], ["indices"], value=tensor))
    elif source == "runtime":
        inputs.append(helper.make_tensor_value_info("indices", TensorProto.INT64, []))
    model = _make_model(
        nodes,
        inputs,
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, output_shape)],
        [tensor] if source == "initializer" else [],
    )
    feeds = {"x": np.arange(np.prod(shape), dtype=np.float32).reshape(shape)}
    if source == "runtime":
        feeds["indices"] = np.int64(1)
    _checked_run(model, feeds)
    before = model.SerializeToString()
    with pytest.raises(ValueError, match=r"Output rank conversion.*Gather"):
        _transform_make_dla_compatible(model, verbose=False)
    assert model.SerializeToString() == before


@pytest.mark.parametrize("indices_shape", [[2], ["K"]])
def test_native_vector_gather_keeps_rank_four_output(indices_shape, monkeypatch):
    """Keep native rank-four Gather when only its index-vector length is symbolic."""
    pipeline = import_module(_transform_make_dla_compatible.__module__)
    monkeypatch.setattr(
        pipeline,
        "DLA_TRANSFORMS",
        tuple((name, transform) for name, transform in pipeline.DLA_TRANSFORMS if name != "Gather"),
    )
    shape = [1, 3, 4, 4]
    model = _make_model(
        [helper.make_node("Gather", ["x", "indices"], ["y"], axis=1)],
        [
            helper.make_tensor_value_info("x", TensorProto.FLOAT, shape),
            helper.make_tensor_value_info("indices", TensorProto.INT64, indices_shape),
        ],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, indices_shape[0], 4, 4])],
    )
    feeds = {
        "x": np.arange(48, dtype=np.float32).reshape(shape),
        "indices": np.array([0, 2], np.int64),
    }
    expected = _checked_run(model, feeds)[0]
    converted = _transform_make_dla_compatible(model, verbose=False)
    np.testing.assert_array_equal(_checked_run(converted, feeds)[0], expected)


@pytest.mark.parametrize("shape", [[3], [1, 3, 4, 4]])
@pytest.mark.parametrize("tail", [False, True])
def test_shape_outputs_keep_native_vector_rank(shape, tail):
    """Retain native Shape vectors, including a one-element vector through unary tails."""
    nodes = [helper.make_node("Shape", ["x"], ["hidden" if tail else "y"])]
    if tail:
        nodes.append(helper.make_node("Identity", ["hidden"], ["y"]))
    model = _make_model(
        nodes,
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, shape)],
        [helper.make_tensor_value_info("y", TensorProto.INT64, [len(shape)])],
    )
    feeds = {"x": np.ones(shape, np.float32)}
    expected = _checked_run(model, feeds)[0]
    converted = _transform_make_dla_compatible(model, verbose=False)
    actual = _checked_run(converted, feeds)[0]
    assert actual.shape == (len(shape),)
    np.testing.assert_array_equal(actual, expected)


def test_early_pattern_does_not_allow_an_unmatched_native_reduction(monkeypatch):
    """Refuse a native reducer before mutation even when another reducer has a valid early lowering."""
    pipeline = import_module(_transform_make_dla_compatible.__module__)
    monkeypatch.setattr(
        pipeline,
        "DLA_TRANSFORMS",
        tuple(
            (name, transform)
            for name, transform in pipeline.DLA_TRANSFORMS
            if name != "StandaloneReduceOperations"
        ),
    )
    shape = [1, 1, 2, 12]
    model = _make_model(
        [
            helper.make_node("Reshape", ["x", "target"], ["expanded"]),
            helper.make_node("ReduceSum", ["expanded", "axes"], ["y"], keepdims=0),
            helper.make_node("ReduceSum", ["x", "axes"], ["native"], keepdims=0),
        ],
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, shape)],
        [
            helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1, 2, 3]),
            helper.make_tensor_value_info("native", TensorProto.FLOAT, [1, 1, 2]),
        ],
        [
            numpy_helper.from_array(np.asarray([1, 1, 2, 3, 4], np.int64), "target"),
            numpy_helper.from_array(np.asarray([-1], np.int64), "axes"),
        ],
    )
    _checked_run(model, {"x": np.ones(shape, np.float32)})
    before = model.SerializeToString()
    with pytest.raises(NotImplementedError, match="reductions"):
        _transform_make_dla_compatible(model, verbose=False)
    assert model.SerializeToString() == before


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


@pytest.mark.parametrize("axis", [0, 1, -1])
def test_registered_concat_preserves_promoted_input_axes(axis):
    """Shift Concat axes once after consistently promoting every native operand."""
    shape = [2, 3]
    result_shape = shape.copy()
    result_shape[axis] *= 2
    model = _make_model(
        [helper.make_node("Concat", ["a", "b"], ["y"], axis=axis)],
        [helper.make_tensor_value_info(name, TensorProto.FLOAT, shape) for name in ("a", "b")],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, result_shape)],
    )
    feeds = {"a": np.arange(6, dtype=np.float32).reshape(shape), "b": np.ones(shape, np.float32)}
    expected = _checked_run(model, feeds)[0]
    changed = _transform_make_dla_compatible(model, verbose=False)
    np.testing.assert_array_equal(_checked_run(changed, feeds)[0], expected)


@pytest.mark.parametrize("axis", [0, -3])
def test_normalization_public_squeeze_keeps_original_axes(axis):
    """Retarget original batch squeezing together with the generated spatial singleton."""
    model = _make_model(
        [
            helper.make_node("InstanceNormalization", ["x", "scale", "bias"], ["normalized"]),
            helper.make_node("Squeeze", ["normalized", "axes"], ["y"]),
        ],
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 4])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [3, 4])],
        [
            numpy_helper.from_array(np.ones(3, np.float32), "scale"),
            numpy_helper.from_array(np.zeros(3, np.float32), "bias"),
            numpy_helper.from_array(np.array([axis], np.int64), "axes"),
        ],
    )
    feeds = {"x": np.arange(12, dtype=np.float32).reshape(1, 3, 4)}
    expected = _checked_run(model, feeds)[0]
    changed = _transform_make_dla_compatible(model, verbose=False)
    np.testing.assert_allclose(_checked_run(changed, feeds)[0], expected, rtol=1e-6, atol=1e-7)


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
    before = model.SerializeToString()
    with pytest.raises(ValueError, match="Concat"):
        _apply_non4d_model_inputs(model)
    assert model.SerializeToString() == before


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
@pytest.mark.parametrize("activations", [(), ("Relu",), ("Relu", "Tanh")])
def test_registered_normalization_preserves_native_channel_axis(batch, activations):
    """Keep direct rank-three normalization inputs on the channel-preserving route."""
    shape = [batch, 3, 4]
    output = "normalized" if activations else "y"
    nodes = [helper.make_node("InstanceNormalization", ["x", "scale", "bias"], [output])]
    for index, operation in enumerate(activations):
        following = "y" if index == len(activations) - 1 else f"activation_{index}"
        nodes.append(helper.make_node(operation, [output], [following]))
        output = following
    model = _make_model(
        nodes,
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
        dla_pipeline = import_module(_transform_make_dla_compatible.__module__)
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


@pytest.mark.parametrize("shape", [[2, 3], [2, 3, 4]])
@pytest.mark.parametrize("tail", [False, True])
@pytest.mark.parametrize("entrypoint", ["deferred", "pipeline"])
def test_native_reshape_output_requires_registered_rank_stage(shape, tail, entrypoint, monkeypatch):
    """Refuse low-rank Reshape restoration before changing public boundary state."""
    nodes = [helper.make_node("Reshape", ["x", "target"], ["native" if tail else "y"])]
    if tail:
        nodes.append(helper.make_node("Relu", ["native"], ["y"]))
    model = _make_model(
        nodes,
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1] * (4 - len(shape)) + shape)],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, shape)],
        [numpy_helper.from_array(np.asarray(shape, np.int64), "target")],
    )
    _checked_run(model, {"x": np.ones([1] * (4 - len(shape)) + shape, np.float32)})
    before = model.SerializeToString()
    if entrypoint == "pipeline":
        pipeline = import_module(_transform_make_dla_compatible.__module__)
        monkeypatch.setattr(
            pipeline,
            "DLA_TRANSFORMS",
            tuple(
                (name, callback)
                for name, callback in pipeline.DLA_TRANSFORMS
                if name != "Non4DReshape"
            ),
        )
    with pytest.raises(ValueError, match="rank"):
        if entrypoint == "pipeline":
            _transform_make_dla_compatible(model, verbose=False)
        else:
            _apply_non4d_model_outputs(model, defer_rank_conversion=True)
    assert model.SerializeToString() == before


@pytest.mark.parametrize("shape", [[1], [3]])
@pytest.mark.parametrize("tail", [False, True])
def test_native_vector_unary_outputs_keep_their_rank(shape, tail):
    """Leave rank-one unary paths native when boundary inputs are not widened."""
    nodes = [helper.make_node("Relu", ["x"], ["middle" if tail else "y"])]
    if tail:
        nodes.append(helper.make_node("Neg", ["middle"], ["y"]))
    model = _make_model(
        nodes,
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, shape)],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, shape)],
    )
    feeds = {"x": np.arange(np.prod(shape), dtype=np.float32).reshape(shape)}
    expected = _checked_run(model, feeds)[0]
    converted = _transform_make_dla_compatible(model, verbose=False)
    np.testing.assert_array_equal(_checked_run(converted, feeds)[0], expected)


def _rank_sensitive_adapter_model(operation, unary, metadata):
    """Build an eligible prefix before a rank-sensitive branch to expose partial rewiring."""
    nodes = [
        helper.make_node("Relu", ["x"], ["a"]),
        helper.make_node("Squeeze", ["a", "axes"], ["b"]),
        helper.make_node("Relu", ["b"], ["early"]),
        helper.make_node("Relu", ["x"], ["r"]),
        helper.make_node("Squeeze", ["r", "axes"], ["s"]),
    ]
    if unary:
        nodes.append(helper.make_node("Relu", ["s"], ["native"]))
    nodes.append(
        helper.make_node(
            operation,
            ["native" if unary else "s"],
            ["y"],
            **({"perm": [1, 2, 0]} if operation == "Transpose" else {"axis": 1}),
        )
    )
    model = _make_model(
        nodes,
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 4, 4])],
        [
            helper.make_tensor_value_info("early", TensorProto.FLOAT, [3, 4, 4]),
            helper.make_tensor_value_info(
                "y", TensorProto.FLOAT, [4, 4, 3] if operation == "Transpose" else [3, 4, 4]
            ),
        ],
        [numpy_helper.from_array(np.array([0], np.int64), "axes")],
    )
    if metadata:
        model = onnx.shape_inference.infer_shapes(model)
    feeds = {"x": np.linspace(0.1, 4, 48, dtype=np.float32).reshape(1, 3, 4, 4)}
    return model, feeds


@pytest.mark.parametrize("operation", ["Softmax", "LogSoftmax", "Transpose"])
@pytest.mark.parametrize("unary", [False, True])
@pytest.mark.parametrize(
    ("entrypoint", "metadata"), [("stage", True), ("pipeline", True), ("pipeline", False)]
)
def test_rank_changing_adapter_requires_actual_consumer_preparation(
    operation, unary, entrypoint, metadata, monkeypatch
):
    """Refuse unsupported reader ranks before changing an earlier eligible adapter."""
    model, feeds = _rank_sensitive_adapter_model(operation, unary, metadata)
    _checked_run(model, feeds)
    pipeline = import_module(_transform_make_dla_compatible.__module__)
    unavailable = "Non4DTranspose" if operation == "Transpose" else "UpdateAxis"
    monkeypatch.setattr(
        pipeline,
        "DLA_TRANSFORMS",
        tuple(
            (name, callback) for name, callback in pipeline.DLA_TRANSFORMS if name != unavailable
        ),
    )
    before = model.SerializeToString()
    with pytest.raises(ValueError, match="rank"):
        if entrypoint == "stage":
            _apply_remove_intermediary_squeeze_unsqueeze(model, rank_converted=True)
        else:
            _transform_make_dla_compatible(model, verbose=False)
    assert model.SerializeToString() == before


@pytest.mark.parametrize("operation", ["Softmax", "LogSoftmax", "Transpose"])
@pytest.mark.parametrize("unary", [False, True])
def test_rank_changing_adapter_uses_registered_consumer_preparation(operation, unary):
    """Execute preserved adapter ownership through the matching later axis stage."""
    pipeline = import_module(_transform_make_dla_compatible.__module__)
    stage = "Non4DTranspose" if operation == "Transpose" else "UpdateAxis"
    if not any(name == stage for name, _ in pipeline.DLA_TRANSFORMS):
        pytest.skip("The matching rank stage is introduced by a later series part")
    model, feeds = _rank_sensitive_adapter_model(operation, unary, True)
    expected = _checked_run(model, feeds)
    converted = _transform_make_dla_compatible(model, verbose=False)
    for actual, reference in zip(_checked_run(converted, feeds), expected):
        np.testing.assert_allclose(actual, reference, rtol=1e-5, atol=1e-6)


def test_output_preflight_does_not_copy_for_missing_intermediate_squeeze_metadata(monkeypatch):
    """Avoid a prerequisite model copy when normal shape inference can prepare Squeeze."""
    model = _make_model(
        [
            helper.make_node("Relu", ["x"], ["hidden"]),
            helper.make_node("Squeeze", ["hidden"], ["y"]),
        ],
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, 2, 3])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 3])],
    )
    feeds = {"x": np.arange(6, dtype=np.float32).reshape(1, 1, 2, 3)}
    expected = _checked_run(model, feeds)[0]
    pipeline = import_module(_transform_make_dla_compatible.__module__)
    original = pipeline.copy.deepcopy
    copies = []

    def record_copy(value, *args, **kwargs):
        """Observe copies of the caller without changing the real fallback behavior."""
        if value is model:
            copies.append(True)
        return original(value, *args, **kwargs)

    monkeypatch.setattr(pipeline.copy, "deepcopy", record_copy)
    converted = _transform_make_dla_compatible(model, verbose=False)
    np.testing.assert_array_equal(_checked_run(converted, feeds)[0], expected)
    assert copies == []


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


@pytest.mark.parametrize("operation", ["Flatten", "ArgMax", "ArgMin", "ReduceProd", "ReduceL2"])
@pytest.mark.parametrize("tail", [False, True])
@pytest.mark.parametrize("entrypoint", ["deferred", "pipeline"])
def test_native_rank_reducing_outputs_require_registered_conversion(
    operation, tail, entrypoint, monkeypatch
):
    """Refuse native rank reducers before adding planned adapters to earlier valid outputs."""
    attributes = (
        {"axis": 1}
        if operation == "Flatten"
        else (
            {"axis": 1, "keepdims": 0}
            if operation.startswith("Arg")
            else {"axes": [2, 3], "keepdims": 0}
        )
    )
    shape = (
        [1, 48]
        if operation == "Flatten"
        else ([1, 4, 4] if operation.startswith("Arg") else [1, 3])
    )
    nodes = [
        helper.make_node("Relu", ["x"], ["first"]),
        helper.make_node(operation, ["z"], ["middle" if tail else "y"], **attributes),
    ]
    if tail:
        nodes.append(helper.make_node("Identity", ["middle"], ["y"]))
    model = _make_model(
        nodes,
        [
            helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3]),
            helper.make_tensor_value_info("z", TensorProto.FLOAT, [1, 3, 4, 4]),
        ],
        [
            helper.make_tensor_value_info("first", TensorProto.FLOAT, [2, 3]),
            helper.make_tensor_value_info(
                "y", TensorProto.INT64 if operation.startswith("Arg") else TensorProto.FLOAT, shape
            ),
        ],
    )
    model.opset_import[0].version = 17
    feeds = {
        "x": np.arange(6, dtype=np.float32).reshape(2, 3),
        "z": np.linspace(0.1, 0.9, 48, dtype=np.float32).reshape(1, 3, 4, 4),
    }
    _checked_run(model, feeds)
    before = model.SerializeToString()
    if entrypoint == "pipeline":
        pipeline = import_module(_transform_make_dla_compatible.__module__)
        monkeypatch.setattr(
            pipeline,
            "DLA_TRANSFORMS",
            tuple(
                (name, transform)
                for name, transform in pipeline.DLA_TRANSFORMS
                if name != operation
            ),
        )
        with pytest.raises(ValueError, match=r"rank.*conversion"):
            _transform_make_dla_compatible(model, verbose=False)
    else:
        with pytest.raises(ValueError, match=r"rank.*conversion"):
            _apply_non4d_model_outputs(model, defer_rank_conversion=True)
    assert model.SerializeToString() == before
