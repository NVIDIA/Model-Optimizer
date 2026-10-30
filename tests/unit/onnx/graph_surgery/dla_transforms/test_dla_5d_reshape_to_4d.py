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

# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Unit tests for :mod:`modelopt.onnx.graph_surgery.dla_transforms.dla_5d_reshape_to_4d`.

"""Tests for 5D Reshape to 4D graph surgery (reduce and transpose paths)."""

from __future__ import annotations

import copy

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper
from onnx.reference import ReferenceEvaluator

from modelopt.onnx.graph_surgery.dla_transforms.dla_5d_reshape_to_4d import _apply_5d_reshape_to_4d
from modelopt.onnx.graph_surgery.utils.graph_utils import infer_shapes

ort = pytest.importorskip("onnxruntime", reason="onnxruntime required for numerical parity checks")

# ReduceMin with axes-as-input requires opset 18+.
# Transpose / NoOp models use opset 13 (axes as attribute) — IR version clamped separately.
_REDUCE_OPSET = 18
_GENERIC_OPSET = 17
_MAX_IR_VERSION_FOR_ORT = 9


def _init(name: str, arr: np.ndarray) -> onnx.TensorProto:
    """Create a named initializer from an array."""
    return numpy_helper.from_array(arr, name=name)


def _clamp_ir(model: onnx.ModelProto) -> None:
    """Use an IR version supported by the test runtime."""
    model.ir_version = min(model.ir_version, _MAX_IR_VERSION_FOR_ORT)


def _run_ort(model: onnx.ModelProto, feeds: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Execute all graph outputs with the CPU runtime."""
    m = copy.deepcopy(model)
    _clamp_ir(m)
    sess = ort.InferenceSession(
        m.SerializeToString(),
        providers=["CPUExecutionProvider"],
    )
    out = {}
    for o in m.graph.output:
        out[o.name] = sess.run([o.name], feeds)[0]
    return out


def _make_reduce_min_axis4_model() -> onnx.ModelProto:
    """X [6,4,4,16] -> Reshape(2,4,12,4,4) -> ReduceMin(axis=4) -> [2,4,12,4]."""
    inits = [
        _init("shape5", np.array([2, 4, 12, 4, 4], dtype=np.int64)),
        _init("axes_reduce", np.array([4], dtype=np.int64)),
    ]
    nodes = [
        helper.make_node("Reshape", ["X", "shape5"], ["T"], name="reshape_5d"),
        helper.make_node(
            "ReduceMin",
            ["T", "axes_reduce"],
            ["Y"],
            name="reduce_min",
            keepdims=0,
        ),
    ]
    inputs = [helper.make_tensor_value_info("X", TensorProto.FLOAT, [6, 4, 4, 16])]
    outputs = [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 4, 12, 4])]
    graph = helper.make_graph(nodes, "reduce5d", inputs, outputs, initializer=inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", _REDUCE_OPSET)])
    onnx.checker.check_model(model)
    return model


def _make_reduce_with_clip_model() -> onnx.ModelProto:
    """Reshape 5D -> Clip -> ReduceMin(axis=4)."""
    inits = [
        _init("shape5", np.array([2, 4, 12, 4, 4], dtype=np.int64)),
        _init("axes_reduce", np.array([4], dtype=np.int64)),
        _init("clip_min", np.array(-1e6, dtype=np.float32).reshape(1)),
        _init("clip_max", np.array(1e6, dtype=np.float32).reshape(1)),
    ]
    nodes = [
        helper.make_node("Reshape", ["X", "shape5"], ["T"], name="reshape_5d"),
        helper.make_node("Clip", ["T", "clip_min", "clip_max"], ["Tc"], name="clip_mid"),
        helper.make_node(
            "ReduceMin",
            ["Tc", "axes_reduce"],
            ["Y"],
            name="reduce_min",
            keepdims=0,
        ),
    ]
    inputs = [helper.make_tensor_value_info("X", TensorProto.FLOAT, [6, 4, 4, 16])]
    outputs = [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 4, 12, 4])]
    graph = helper.make_graph(nodes, "reduce_clip", inputs, outputs, initializer=inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", _REDUCE_OPSET)])
    onnx.checker.check_model(model)
    return model


def _make_transpose_reshape_model() -> onnx.ModelProto:
    """X [6,4,4,16] -> Reshape(2,4,12,4,4) -> Transpose(perm[0]==0) -> Reshape(2,12,16,4)."""
    inits = [
        _init("shape5", np.array([2, 4, 12, 4, 4], dtype=np.int64)),
        _init("shape_out", np.array([2, 12, 16, 4], dtype=np.int64)),
    ]
    nodes = [
        helper.make_node("Reshape", ["X", "shape5"], ["T5"], name="reshape_5d"),
        helper.make_node(
            "Transpose",
            ["T5"],
            ["Tv"],
            name="transpose_mid",
            perm=[0, 2, 1, 4, 3],
        ),
        helper.make_node("Reshape", ["Tv", "shape_out"], ["Y"], name="reshape_out"),
    ]
    inputs = [helper.make_tensor_value_info("X", TensorProto.FLOAT, [6, 4, 4, 16])]
    outputs = [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 12, 16, 4])]
    graph = helper.make_graph(nodes, "tr5d", inputs, outputs, initializer=inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", _GENERIC_OPSET)])
    onnx.checker.check_model(model)
    return model


@pytest.mark.parametrize("captured", ["T", "clip", "quantized", "dequantized", "Y"])
@pytest.mark.parametrize("skip_inference", [False, True])
def test_nested_capture_retains_native_producer(captured, skip_inference):
    """Refuse removed middleware captures while allowing the recreated terminal output."""
    model = _make_reduce_min_axis4_model()
    model.graph.initializer.extend(
        [_init("scale", np.array(0.25, np.float32)), _init("zero", np.array(128, np.uint8))]
    )
    model.graph.node[1].input[0] = "dequantized"
    model.graph.node.insert(1, helper.make_node("Clip", ["T"], ["clip"]))
    model.graph.node.insert(
        2, helper.make_node("QuantizeLinear", ["clip", "scale", "zero"], ["quantized"])
    )
    model.graph.node.insert(
        3, helper.make_node("DequantizeLinear", ["quantized", "scale", "zero"], ["dequantized"])
    )
    dtype = TensorProto.UINT8 if captured == "quantized" else TensorProto.FLOAT
    shape = [2, 4, 12, 4] if captured == "Y" else [2, 4, 12, 4, 4]
    branch = helper.make_graph(
        [helper.make_node("Identity", [captured], ["view"])],
        "capturing_branch",
        [],
        [helper.make_tensor_value_info("view", dtype, shape)],
    )
    model.graph.initializer.append(_init("condition", np.array(True)))
    model.graph.node.append(
        helper.make_node("If", ["condition"], ["captured"], then_branch=branch, else_branch=branch)
    )
    model.graph.output.append(helper.make_tensor_value_info("captured", dtype, shape))
    onnx.checker.check_model(model, full_check=True)
    feeds = {"X": np.arange(1536, dtype=np.float32).reshape(6, 4, 4, 16) / 128}
    expected = _run_ort(model, feeds)
    before = model.SerializeToString()
    if captured != "Y":
        with pytest.raises(ValueError, match="captur"):
            _apply_5d_reshape_to_4d(model, skip_symbolic_shape_inference=skip_inference)
        assert model.SerializeToString() == before
    else:
        _apply_5d_reshape_to_4d(model, skip_symbolic_shape_inference=skip_inference)
        onnx.checker.check_model(model, full_check=True)
        for name, actual in _run_ort(model, feeds).items():
            np.testing.assert_array_equal(actual, expected[name])


def _make_reduce_axis0_model() -> onnx.ModelProto:
    """Invalid for this pass: ReduceMin axis 0 on 5D."""
    inits = [
        _init("shape5", np.array([2, 4, 12, 4, 4], dtype=np.int64)),
        _init("axes_reduce", np.array([0], dtype=np.int64)),
    ]
    nodes = [
        helper.make_node("Reshape", ["X", "shape5"], ["T"], name="reshape_5d"),
        helper.make_node(
            "ReduceMin",
            ["T", "axes_reduce"],
            ["Y"],
            name="reduce_min",
            keepdims=0,
        ),
    ]
    inputs = [helper.make_tensor_value_info("X", TensorProto.FLOAT, [6, 4, 4, 16])]
    outputs = [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [4, 12, 4, 4])]
    graph = helper.make_graph(nodes, "bad_axis", inputs, outputs, initializer=inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", _REDUCE_OPSET)])
    onnx.checker.check_model(model)
    return model


def _make_only_4d_reshape_model() -> onnx.ModelProto:
    """Reshape to 4D target — not a 5D candidate."""
    inits = [_init("shape4", np.array([24, 4, 16], dtype=np.int64))]
    nodes = [helper.make_node("Reshape", ["X", "shape4"], ["Y"], name="reshape_4d")]
    inputs = [helper.make_tensor_value_info("X", TensorProto.FLOAT, [6, 4, 4, 16])]
    outputs = [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [24, 4, 16])]
    graph = helper.make_graph(nodes, "r4", inputs, outputs, initializer=inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", _GENERIC_OPSET)])
    onnx.checker.check_model(model)
    return model


class TestDla5dReshapeTo4dReduce:
    @pytest.mark.parametrize("unsupported", ["axis", "shared"])
    def test_unsafe_qdq_middleware_refuses_before_mutation(self, unsupported):
        """Refuse middleware whose branching or axes cannot be preserved."""
        model = _make_reduce_min_axis4_model()
        reshape, reduce = model.graph.node
        attrs = {"axis": 4} if unsupported == "axis" else {}
        quantize = helper.make_node(
            "QuantizeLinear", [reshape.output[0], "scale", "zero"], ["q"], **attrs
        )
        dequantize = helper.make_node("DequantizeLinear", ["q", "scale", "zero"], ["dq"])
        reduce.input[0] = "dq"
        nodes = [reshape, quantize, dequantize, reduce]
        if unsupported == "shared":
            nodes.append(helper.make_node("Identity", ["q"], ["other"]))
        del model.graph.node[:]
        model.graph.node.extend(nodes)
        model.graph.initializer.extend(
            [
                numpy_helper.from_array(np.float32(0.1), "scale"),
                numpy_helper.from_array(np.uint8(0), "zero"),
            ]
        )
        before = model.SerializeToString()
        with pytest.raises(ValueError, match=r"per-tensor|unshared"):
            _apply_5d_reshape_to_4d(model, skip_symbolic_shape_inference=True)
        assert model.SerializeToString() == before

    def test_reduce_min_axis4_ort_matches_reference(self):
        """Preserve reduction values while generating slices and negative axes."""
        orig = _make_reduce_min_axis4_model()
        x = np.random.default_rng(0).standard_normal((6, 4, 4, 16)).astype(np.float32)
        y_ref = _run_ort(orig, {"X": x})["Y"]

        rewritten = copy.deepcopy(orig)
        rewritten = _apply_5d_reshape_to_4d(rewritten)
        onnx.checker.check_model(rewritten)
        y_new = _run_ort(rewritten, {"X": x})["Y"]

        assert y_ref.shape == y_new.shape == (2, 4, 12, 4)
        np.testing.assert_allclose(y_ref, y_new, rtol=0, atol=0)

        # Verify Split is NOT present — replaced by Slice nodes
        assert not any(n.op_type == "Split" for n in rewritten.graph.node), (
            "Split must be replaced by Slice nodes"
        )
        slice_nodes = [n for n in rewritten.graph.node if n.op_type == "Slice"]
        assert len(slice_nodes) == 4, (
            f"Expected 4 Slice nodes (one per chunk), got {len(slice_nodes)}"
        )
        initializers = {init.name: init for init in rewritten.graph.initializer}
        for node in rewritten.graph.node:
            if node.op_type == "ReduceMin":
                assert numpy_helper.to_array(initializers[node.input[1]]).tolist() == [-1]

    def test_reduce_with_clip_ort_matches_reference(self):
        """Preserve clipped reduction values with one Clip per branch."""
        orig = _make_reduce_with_clip_model()
        x = np.random.default_rng(1).standard_normal((6, 4, 4, 16)).astype(np.float32)
        y_ref = _run_ort(orig, {"X": x})["Y"]

        rewritten = copy.deepcopy(orig)
        rewritten = _apply_5d_reshape_to_4d(rewritten)
        onnx.checker.check_model(rewritten)
        y_new = _run_ort(rewritten, {"X": x})["Y"]

        assert y_ref.shape == y_new.shape
        np.testing.assert_allclose(y_ref, y_new, rtol=0, atol=0)

        assert sum(node.op_type == "Clip" for node in rewritten.graph.node) == 4

    def test_reduce_axis_zero_raises(self):
        """Refuse the unsupported batch-axis reduction."""
        orig = _make_reduce_axis0_model()
        with pytest.raises(ValueError, match=r"reduce axis must be 1\.\.4"):
            _apply_5d_reshape_to_4d(infer_shapes(copy.deepcopy(orig)))


@pytest.mark.parametrize("control", ["shape5", "shape_out", "axes_reduce"])
@pytest.mark.parametrize("skip_inference", [False, True])
def test_overridable_structure_refuses_before_mutation(control, skip_inference):
    """Preserve runtime shape and axis overrides rather than baking in initializer defaults."""
    model = (
        _make_reduce_min_axis4_model()
        if control == "axes_reduce"
        else _make_transpose_reshape_model()
    )
    if control != "axes_reduce":
        dims = model.graph.input[0].type.tensor_type.shape.dim
        dims[0].dim_value, dims[1].dim_value = 2, 12
    initializer = next(tensor for tensor in model.graph.initializer if tensor.name == control)
    model.graph.input.append(
        helper.make_tensor_value_info(control, TensorProto.INT64, list(initializer.dims))
    )
    overrides = {"shape5": [2, 12, 4, 4, 4], "shape_out": [2, 16, 12, 4], "axes_reduce": [3]}
    if control == "shape_out":
        for index, dim in enumerate(model.graph.output[0].type.tensor_type.shape.dim):
            dim.ClearField("dim_value")
            dim.dim_param = f"output_{index}"
    onnx.checker.check_model(model, full_check=True)
    data = np.arange(1536, dtype=np.float32).reshape(
        [dim.dim_value for dim in model.graph.input[0].type.tensor_type.shape.dim]
    )
    reference = ReferenceEvaluator(model)
    expected = reference.run(
        None, {"X": data, control: np.asarray(overrides[control], dtype=np.int64)}
    )[0]
    default = reference.run(None, {"X": data})[0]
    assert not np.array_equal(default, expected)
    before = model.SerializeToString()
    with pytest.raises(ValueError, match="initializer"):
        _apply_5d_reshape_to_4d(model, skip_symbolic_shape_inference=skip_inference)
    assert model.SerializeToString() == before


@pytest.mark.parametrize("branch", ["transpose", "reduce"])
@pytest.mark.parametrize("runtime", ["reference", "ort"])
@pytest.mark.parametrize("control_shape", [[], [1]])
def test_qdq_control_rank_when_lowering_changes_default_axis_length(branch, runtime, control_shape):
    """Lower scalar QDQ controls and refuse per-axis singleton controls without mutation."""
    nodes = [
        helper.make_node("Reshape", ["X", "shape5"], ["reshaped"], name="reshape5"),
        helper.make_node(
            "QuantizeLinear", ["reshaped", "scale", "zero"], ["quantized"], name="quantize"
        ),
        helper.make_node(
            "DequantizeLinear", ["quantized", "scale", "zero"], ["dequantized"], name="dequantize"
        ),
    ]
    initializers = [
        _init("shape5", np.asarray([2, 1, 3, 4, 5], np.int64)),
        _init("scale", np.asarray(0.25, np.float32).reshape(control_shape)),
        _init("zero", np.asarray(7, np.uint8).reshape(control_shape)),
    ]
    if branch == "transpose":
        nodes.extend(
            [
                helper.make_node(
                    "Transpose",
                    ["dequantized"],
                    ["transposed"],
                    name="transpose",
                    perm=[0, 1, 3, 4, 2],
                ),
                helper.make_node("Reshape", ["transposed", "shape_out"], ["Y"], name="reshape_out"),
            ]
        )
        initializers.append(_init("shape_out", np.asarray([2, 1, 12, 5], np.int64)))
        output_shape = [2, 1, 12, 5]
    else:
        nodes.append(
            helper.make_node("ReduceMin", ["dequantized", "axes"], ["Y"], name="reduce", keepdims=0)
        )
        initializers.append(_init("axes", np.asarray([2], np.int64)))
        output_shape = [2, 1, 4, 5]
    model = helper.make_model(
        helper.make_graph(
            nodes,
            "single_element",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 1, 3, 20])],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, output_shape)],
            initializer=initializers,
        ),
        opset_imports=[helper.make_opsetid("", 19)],
    )
    onnx.checker.check_model(model, full_check=True)
    feeds = {"X": np.arange(120, dtype=np.float32).reshape(2, 1, 3, 20) / 16 - 2}
    if control_shape:
        before = model.SerializeToString()
        with pytest.raises(ValueError, match="per-tensor"):
            _apply_5d_reshape_to_4d(model, skip_symbolic_shape_inference=True)
        assert model.SerializeToString() == before
        return
    transformed = _apply_5d_reshape_to_4d(copy.deepcopy(model), skip_symbolic_shape_inference=True)
    onnx.checker.check_model(transformed, full_check=True)
    assert sum(node.op_type == "Slice" for node in transformed.graph.node) == (
        2 if branch == "transpose" else 1
    )
    if runtime == "ort":
        np.testing.assert_array_equal(
            _run_ort(transformed, feeds)["Y"], _run_ort(model, feeds)["Y"]
        )
    else:
        np.testing.assert_array_equal(
            ReferenceEvaluator(transformed).run(None, feeds)[0],
            ReferenceEvaluator(model).run(None, feeds)[0],
        )


class TestDla5dReshapeTo4dTranspose:
    @pytest.mark.parametrize(
        ("opset", "skip_inference"), [(5, True), (7, False), (9, False), (9, True)]
    )
    def test_unsupported_slice_schema_refuses_before_mutation(self, opset, skip_inference):
        """Refuse versions without input-based Slice and restore inferred metadata."""
        model = _make_transpose_reshape_model()
        model.opset_import[0].version = opset
        dims = model.graph.input[0].type.tensor_type.shape.dim
        dims[0].dim_value, dims[1].dim_value = 2, 12
        onnx.checker.check_model(model, full_check=True)
        before = model.SerializeToString()
        with pytest.raises(ValueError, match="opset"):
            _apply_5d_reshape_to_4d(model, skip_symbolic_shape_inference=skip_inference)
        assert model.SerializeToString() == before

    def test_attribute_reshape_is_not_an_input_shape_candidate(self):
        """Leave valid legacy attribute-form reshapes intact instead of indexing input 1."""
        model = _make_transpose_reshape_model()
        model.opset_import[0].version = 1
        targets = {
            init.name: numpy_helper.to_array(init).tolist() for init in model.graph.initializer
        }
        for node in model.graph.node:
            if node.op_type == "Reshape":
                node.attribute.append(helper.make_attribute("shape", targets[node.input[1]]))
                del node.input[1:]
        del model.graph.initializer[:]
        onnx.checker.check_model(model, full_check=True)
        before = model.SerializeToString()
        _apply_5d_reshape_to_4d(model, skip_symbolic_shape_inference=True)
        assert model.SerializeToString() == before

    @pytest.mark.parametrize("custom_op", ["Reshape", "Clip", "DequantizeLinear"])
    def test_custom_functions_are_not_lowered_as_builtin_ops(self, custom_op):
        """Preserve custom candidates and refuse custom followers before mutation."""
        model = _make_transpose_reshape_model()
        model.opset_import[0].version = 21
        model.opset_import.append(helper.make_opsetid("custom", 1))
        dims = model.graph.input[0].type.tensor_type.shape.dim
        dims[0].dim_value, dims[1].dim_value = 2, 12
        if custom_op == "Reshape":
            model.graph.node[0].domain = "custom"
            inputs = ["data", "shape"]
            body = [
                helper.make_node("Reshape", inputs, ["reshaped"]),
                helper.make_node("Neg", ["reshaped"], ["result"]),
            ]
        else:
            model.graph.node[1].input[0] = "middle"
            inputs = ["data", "scale", "zero"]
            if custom_op == "Clip":
                nodes = [
                    helper.make_node("Clip", ["T5", "scale", "zero"], ["middle"], domain="custom")
                ]
                body = [helper.make_node("Identity", ["data"], ["result"])]
                zero = np.array(0.5, np.float32)
            else:
                nodes = [
                    helper.make_node("QuantizeLinear", ["T5", "scale", "zero"], ["q"]),
                    helper.make_node(
                        "DequantizeLinear", ["q", "scale", "zero"], ["middle"], domain="custom"
                    ),
                ]
                body = [helper.make_node("Cast", ["data"], ["result"], to=TensorProto.FLOAT)]
                zero = np.array(0, np.uint8)
            for index, node in enumerate(nodes, start=1):
                model.graph.node.insert(index, node)
            model.graph.initializer.extend(
                [
                    _init("scale", np.array(0.25, np.float32)),
                    _init("zero", zero),
                ]
            )
        model.functions.append(
            helper.make_function(
                "custom", custom_op, inputs, ["result"], body, [helper.make_opsetid("", 21)]
            )
        )
        onnx.checker.check_model(model, full_check=True)
        feeds = {"X": np.random.default_rng(17).standard_normal((2, 12, 4, 16)).astype(np.float32)}
        expected = ReferenceEvaluator(model).run(None, feeds)
        before = model.SerializeToString()
        if custom_op == "Reshape":
            rewritten = _apply_5d_reshape_to_4d(model, skip_symbolic_shape_inference=True)
            for actual, reference in zip(ReferenceEvaluator(rewritten).run(None, feeds), expected):
                np.testing.assert_array_equal(actual, reference)
        else:
            with pytest.raises(ValueError):
                _apply_5d_reshape_to_4d(model, skip_symbolic_shape_inference=True)
        assert model.SerializeToString() == before

    @pytest.mark.parametrize(
        "controls", ["initializer", "input", "symbolic", "overridable", "dequantizer", "zero_only"]
    )
    def test_implicit_per_axis_qdq_refuses_before_mutation(self, controls):
        """Refuse per-axis controls even when both QDQ nodes omit their axis."""
        model = _make_transpose_reshape_model()
        model.opset_import[0].version = 21
        dims = model.graph.input[0].type.tensor_type.shape.dim
        dims[0].dim_value, dims[1].dim_value = 2, 12
        nodes = list(model.graph.node)
        nodes[1].input[0] = "dequantized"
        scale = np.array([0.125, 0.25, 0.5, 1], np.float32)
        zero = np.zeros(4, np.uint8)
        feeds = {"X": np.random.default_rng(13).standard_normal((2, 12, 4, 16)).astype(np.float32)}
        leading = []
        if controls in {"input", "symbolic", "overridable"}:
            shape = [4] if controls == "input" else ["channels"]
            model.graph.input.extend(
                [
                    helper.make_tensor_value_info("scale", TensorProto.FLOAT, shape),
                    helper.make_tensor_value_info("zero", TensorProto.UINT8, shape),
                ]
            )
            feeds.update(scale=scale, zero=zero)
            if controls == "overridable":
                model.graph.initializer.extend(
                    [
                        _init("scale", np.array([0.25], np.float32)),
                        _init("zero", np.array([0], np.uint8)),
                    ]
                )
        else:
            model.graph.initializer.append(_init("zero", zero))
            if controls == "zero_only":
                model.graph.input.append(
                    helper.make_tensor_value_info("scale_input", TensorProto.FLOAT, [4])
                )
                leading.append(
                    helper.make_node("Identity", ["scale_input"], ["scale"], name="scale_producer")
                )
                feeds["scale_input"] = scale
            else:
                model.graph.initializer.append(_init("scale", scale))
        q_controls = ["scale", "zero"]
        if controls == "dequantizer":
            q_controls = ["scalar_scale", "scalar_zero"]
            model.graph.initializer.extend(
                [
                    _init("scalar_scale", np.array(0.25, np.float32)),
                    _init("scalar_zero", np.array(0, np.uint8)),
                ]
            )
        quantize = helper.make_node("QuantizeLinear", ["T5", *q_controls], ["q"], name="quantize")
        dequantize = helper.make_node(
            "DequantizeLinear", ["q", "scale", "zero"], ["dequantized"], name="dequantize"
        )
        del model.graph.node[:]
        model.graph.node.extend([*leading, nodes[0], quantize, dequantize, *nodes[1:]])
        onnx.checker.check_model(model, full_check=True)
        ReferenceEvaluator(model).run(None, feeds)
        before = model.SerializeToString()
        with pytest.raises(ValueError, match="per-tensor"):
            _apply_5d_reshape_to_4d(model, skip_symbolic_shape_inference=True)
        assert model.SerializeToString() == before

    @pytest.mark.parametrize("path", ["transpose", "reduce"])
    @pytest.mark.parametrize("middleware", ["Clip", "QDQ"])
    @pytest.mark.parametrize("skip_inference", [False, True])
    def test_late_dynamic_controls_preserve_values(self, path, middleware, skip_inference):
        """Keep runtime controls ordered and refuse QDQ controls whose rank is unproven."""
        model = (
            _make_transpose_reshape_model()
            if path == "transpose"
            else _make_reduce_min_axis4_model()
        )
        model.opset_import[0].version = 21
        if path == "transpose":
            dims = model.graph.input[0].type.tensor_type.shape.dim
            dims[0].dim_value, dims[1].dim_value = 2, 12
        nodes = list(model.graph.node)
        data = nodes[0].output[0]
        control = helper.make_node("Identity", ["bound"], ["control"], name="late_control")
        if middleware == "Clip":
            middle = [helper.make_node("Clip", [data, "control"], ["middle"], name="clip")]
            bound = np.float32(-0.5)
        else:
            middle = [
                helper.make_node(
                    "QuantizeLinear", [data, "control", "zero"], ["q"], name="quantize"
                ),
                helper.make_node(
                    "DequantizeLinear", ["q", "control", "zero"], ["middle"], name="dequantize"
                ),
            ]
            model.graph.initializer.append(_init("zero", np.array(0, np.uint8)))
            bound = np.float32(0.25)
        nodes[1].input[0] = "middle"
        del model.graph.node[:]
        model.graph.node.extend([nodes[0], control, *middle, *nodes[1:]])
        model.graph.input.append(helper.make_tensor_value_info("bound", TensorProto.FLOAT, []))
        onnx.checker.check_model(model, full_check=True)
        shape = [dim.dim_value for dim in model.graph.input[0].type.tensor_type.shape.dim]
        feeds = {
            "X": np.random.default_rng(11).standard_normal(shape).astype(np.float32),
            "bound": bound,
        }
        expected = ReferenceEvaluator(model).run(None, feeds)
        if middleware == "QDQ" and skip_inference:
            before = model.SerializeToString()
            with pytest.raises(ValueError, match="per-tensor"):
                _apply_5d_reshape_to_4d(model, skip_symbolic_shape_inference=True)
            assert model.SerializeToString() == before
            return
        rewritten = _apply_5d_reshape_to_4d(model, skip_symbolic_shape_inference=skip_inference)
        onnx.checker.check_model(rewritten, full_check=True)
        for actual, reference in zip(ReferenceEvaluator(rewritten).run(None, feeds), expected):
            np.testing.assert_array_equal(actual, reference)

    @pytest.mark.parametrize("path", ["transpose", "reduce"])
    def test_generated_names_preserve_existing_controls_and_branches(self, path):
        """Reserve generated names without retargeting existing data or QDQ controls."""
        transpose = path == "transpose"
        model = _make_transpose_reshape_model() if transpose else _make_reduce_with_clip_model()
        model.opset_import[0].version = 21
        prefix = "tr" if transpose else "rs"
        if transpose:
            dims = model.graph.input[0].type.tensor_type.shape.dim
            dims[0].dim_value, dims[1].dim_value = 2, 12
            clip = helper.make_node(
                "Clip", ["T5", "clip_min", "clip_max"], ["clipped"], name="clip"
            )
            model.graph.node.insert(1, clip)
            model.graph.initializer.extend(
                [
                    _init("clip_min", np.array(-1, np.float32)),
                    _init("clip_max", np.array(1, np.float32)),
                ]
            )
        clip_output = model.graph.node[1].output[0]
        model.graph.node[2].input[0] = "dequantized"
        scale_name = "u5d_tr_r1_0" if transpose else "u5d_rs_slax_0"
        suffix = "tr0_0a" if transpose else "rs0_0"
        zero_name = f"_u5d_{suffix}_q1_1"
        model.graph.node.insert(
            2,
            helper.make_node(
                "QuantizeLinear", [clip_output, scale_name, zero_name], ["quantized"], name="q"
            ),
        )
        model.graph.node.insert(
            3,
            helper.make_node(
                "DequantizeLinear", ["quantized", scale_name, zero_name], ["dequantized"], name="dq"
            ),
        )
        model.graph.initializer.extend(
            [
                _init(scale_name, np.array(0.25, np.float32)),
                _init(zero_name, np.array(0, np.uint8)),
            ]
        )
        if transpose:
            model.graph.value_info.extend(
                [
                    helper.make_tensor_value_info(scale_name, TensorProto.FLOAT, []),
                    helper.make_tensor_value_info(zero_name, TensorProto.UINT8, []),
                ]
            )
        renames = {
            "X": "u5d_tr_r4_0" if transpose else "u5d_rs_msh_0",
            "shape5": "u5d_tr_slice_axes_0" if transpose else "u5d_rs_ax4_0",
            "clip_min": f"_u5d_{suffix}_c0_0",
        }
        for value in [*model.graph.input, *model.graph.initializer]:
            value.name = renames.get(value.name, value.name)
        for node in model.graph.node:
            for index, name in enumerate(node.input):
                node.input[index] = renames.get(name, name)
        branch = "u5d_tr_r4o_0_0" if transpose else "u5d_rs_sp_0_0"
        model.graph.node.append(
            helper.make_node("Identity", [renames["X"]], [branch], name=f"u5d_{prefix}_slice_0_0")
        )
        branch_info = copy.deepcopy(model.graph.input[0])
        branch_info.name = branch
        model.graph.output.append(branch_info)
        onnx.checker.check_model(model, full_check=True)
        controls = {value.name: value.SerializeToString() for value in model.graph.initializer}
        shape = [dim.dim_value for dim in model.graph.input[0].type.tensor_type.shape.dim]
        values = np.random.default_rng(9).standard_normal(shape).astype(np.float32)
        feeds = {renames["X"]: values}
        expected = ReferenceEvaluator(model).run(None, feeds)
        rewritten = _apply_5d_reshape_to_4d(model, skip_symbolic_shape_inference=True)
        onnx.checker.check_model(rewritten, full_check=True)
        assert len({init.name for init in rewritten.graph.initializer}) == len(
            rewritten.graph.initializer
        )
        for init in rewritten.graph.initializer:
            if init.name in controls:
                assert init.SerializeToString() == controls[init.name]
        for actual, reference in zip(ReferenceEvaluator(rewritten).run(None, feeds), expected):
            np.testing.assert_array_equal(actual, reference)

    @pytest.mark.parametrize("shape", [[0, -1], [0, 0, 4, -1], [0, 0, 0, -1]])
    def test_final_reshape_zero_dimensions_preserve_values(self, shape):
        """Resolve copied dimensions against the original five-dimensional input."""
        model = _make_transpose_reshape_model()
        dims = model.graph.input[0].type.tensor_type.shape.dim
        dims[0].dim_value, dims[1].dim_value = 2, 12
        model.graph.initializer[1].CopyFrom(_init("shape_out", np.array(shape, np.int64)))
        resolved = [2, 12, 4, 4, 4]
        target = [resolved[i] if dim == 0 else dim for i, dim in enumerate(shape)]
        target = np.empty((2, 12, 4, 16)).reshape(target).shape
        model.graph.output[0].CopyFrom(
            helper.make_tensor_value_info("Y", TensorProto.FLOAT, target)
        )
        values = np.random.default_rng(7).standard_normal((2, 12, 4, 16)).astype(np.float32)
        expected = _run_ort(model, {"X": values})["Y"]
        rewritten = _apply_5d_reshape_to_4d(model, skip_symbolic_shape_inference=True)
        onnx.checker.check_model(rewritten)
        np.testing.assert_array_equal(_run_ort(rewritten, {"X": values})["Y"], expected)

    @pytest.mark.parametrize("unsupported", ["dynamic", "overlap", "late_failure"])
    @pytest.mark.parametrize("skip_inference", [True, False])
    def test_unsupported_patterns_leave_model_unchanged(self, unsupported, skip_inference):
        """Refusal preserves nodes, initializers and inferred metadata, even on retry."""
        model = _make_transpose_reshape_model()
        dims = model.graph.input[0].type.tensor_type.shape.dim
        dims[0].dim_value, dims[1].dim_value = 2, 12
        if unsupported == "dynamic":
            del model.graph.initializer[1]
            model.graph.input.append(
                helper.make_tensor_value_info("shape_out", TensorProto.INT64, [4])
            )
        elif unsupported == "overlap":
            model.graph.initializer[1].CopyFrom(
                _init("shape_out", np.array([2, 12, 4, 4, 4], np.int64))
            )
            model.graph.initializer.append(_init("axes", np.array([4], np.int64)))
            model.graph.node[-1].output[0] = "second_5d"
            model.graph.node.append(
                helper.make_node("ReduceMin", ["second_5d", "axes"], ["Y"], keepdims=0)
            )
            model.opset_import[0].version = _REDUCE_OPSET
            model.graph.output[0].CopyFrom(
                helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 12, 4, 4])
            )
        else:
            model.graph.initializer.append(
                _init("late_shape", np.array([2, 4, 12, 4, 4], np.int64))
            )
            model.graph.node.extend(
                [
                    helper.make_node(
                        "Reshape", ["X", "late_shape"], ["late_5d"], name="late_reshape"
                    ),
                    helper.make_node("Identity", ["late_5d"], ["late_output"], name="unsupported"),
                ]
            )
            model.graph.output.append(
                helper.make_tensor_value_info("late_output", TensorProto.FLOAT, [2, 4, 12, 4, 4])
            )
        onnx.checker.check_model(model)
        before = model.SerializeToString()
        for _ in range(2):
            with pytest.raises(ValueError):
                _apply_5d_reshape_to_4d(model, skip_symbolic_shape_inference=skip_inference)
            assert model.SerializeToString() == before

    @pytest.mark.parametrize(
        ("legacy_clip", "marked_node"),
        [(False, None), (True, None), (False, "transpose_mid"), (False, "reshape_out")],
    )
    def test_transpose_pipeline_rewritten(self, legacy_clip, marked_node):
        """The supported leading-slice layout preserves values and output shape."""
        orig = _make_transpose_reshape_model()
        if marked_node is not None:
            next(
                node for node in orig.graph.node if node.name == marked_node
            ).name += "_transformed"
        dims = orig.graph.input[0].type.tensor_type.shape.dim
        dims[0].dim_value = 2
        dims[1].dim_value = 12
        if legacy_clip:
            orig.opset_import[0].version = 10
            orig.graph.node[1].input[0] = "clipped"
            orig.graph.node.insert(
                1, helper.make_node("Clip", ["T5"], ["clipped"], name="clip", min=-0.5, max=0.5)
            )
        onnx.checker.check_model(orig, full_check=True)
        values = np.random.default_rng(5).standard_normal((2, 12, 4, 16)).astype(np.float32)
        expected = _run_ort(orig, {"X": values})["Y"]
        rewritten = _apply_5d_reshape_to_4d(infer_shapes(copy.deepcopy(orig)))
        actual = _run_ort(rewritten, {"X": values})["Y"]
        np.testing.assert_array_equal(actual, expected)

        # 5D Reshape must be removed
        assert not any(n.name == "reshape_5d" for n in rewritten.graph.node), (
            "5D Reshape must be removed by the transform"
        )
        # Slice nodes inserted by decomposition
        assert any(n.op_type == "Slice" for n in rewritten.graph.node), (
            "Slice nodes must be inserted"
        )
        # Transpose preserved (one per slice)
        assert any(n.op_type == "Transpose" for n in rewritten.graph.node), (
            "Transpose nodes must remain"
        )
        # Concat stitches results
        assert any(n.op_type == "Concat" for n in rewritten.graph.node), (
            "Concat node must be present"
        )

    def test_incompatible_leading_slice_layout_refuses_before_mutation(self):
        """Refuse a leading dimension that does not match the slice count."""
        model = infer_shapes(_make_transpose_reshape_model())
        before = model.SerializeToString()
        with pytest.raises(ValueError, match="leading input dimension"):
            _apply_5d_reshape_to_4d(model)
        assert model.SerializeToString() == before


class TestDla5dReshapeTo4dNoOp:
    def test_no_5d_reshape_leaves_graph_unchanged(self):
        """Leave unrelated reshape nodes and their values unchanged."""
        orig = _make_only_4d_reshape_model()
        x = np.random.default_rng(4).standard_normal((6, 4, 4, 16)).astype(np.float32)
        y_ref = _run_ort(orig, {"X": x})["Y"]

        before = [n.SerializeToString() for n in orig.graph.node]
        rewritten = copy.deepcopy(orig)
        rewritten = infer_shapes(rewritten)
        rewritten = _apply_5d_reshape_to_4d(rewritten)
        after = [n.SerializeToString() for n in rewritten.graph.node]

        y_new = _run_ort(rewritten, {"X": x})["Y"]

        assert before == after
        assert y_ref.shape == y_new.shape
        np.testing.assert_allclose(y_ref, y_new, rtol=0, atol=0)
