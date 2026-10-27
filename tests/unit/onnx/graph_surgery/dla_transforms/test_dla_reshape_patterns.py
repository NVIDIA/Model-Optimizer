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

"""Execute and validate the ordered DLA reshape pattern stages."""

import copy

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper
from onnx.reference import ReferenceEvaluator

from modelopt.onnx.graph_surgery.dla_transforms.dla_reshape_patterns import (
    _apply_reshape_qdq_reduce_sum,
    _apply_reshape_reduce_sum,
    _apply_reshape_transpose_reshape,
)

ort = pytest.importorskip("onnxruntime", reason="ORT required for numerical stage checks")


def _make_model(
    pattern,
    *,
    changed_controls=False,
    per_axis=False,
    zeros=None,
    collision=False,
    late_control=False,
):
    """Build valid transpose or fast/fallback reduction inputs with optional QDQ controls."""
    transpose = pattern == "transpose"
    quantized = pattern == "qdq"
    input_shape = [2, 3, 4, 5] if transpose else [2, 1, 3, 12 if pattern == "fallback" else 8]
    shape5 = [2, 1, 3, 4, 5] if transpose else [2, 1, 3, 3 if pattern == "fallback" else 2, 4]
    if zeros == "first":
        shape5 = [2, 1, 0, 3, 5]
    initializers = [numpy_helper.from_array(np.asarray(shape5, np.int64), name="shape5")]
    nodes = [helper.make_node("Reshape", ["X", "shape5"], ["reshaped"], name="reshape")]
    current = "reshaped"
    if quantized:
        scale = np.asarray([0.25] * 4 if per_axis else 0.25, np.float32)
        zero = np.asarray([2] * 4 if per_axis else 2, np.uint8)
        initializers.extend(
            [
                numpy_helper.from_array(scale, name="q_scale"),
                numpy_helper.from_array(zero, name="q_zero"),
            ]
        )
        if changed_controls:
            initializers.extend(
                [
                    numpy_helper.from_array(scale * 2, name="dq_scale"),
                    numpy_helper.from_array(zero + 2, name="dq_zero"),
                ]
            )
        q_scale = "q_scale"
        if late_control:
            nodes.append(
                helper.make_node("Identity", [q_scale], ["runtime_scale"], name="scale_producer")
            )
            q_scale = "runtime_scale"
        attrs = {"axis": 4} if per_axis else {}
        nodes.extend(
            [
                helper.make_node(
                    "QuantizeLinear",
                    [current, q_scale, "q_zero"],
                    ["quantized"],
                    name="quantize",
                    **attrs,
                ),
                helper.make_node(
                    "DequantizeLinear",
                    [
                        "quantized",
                        "dq_scale" if changed_controls else q_scale,
                        "dq_zero" if changed_controls else "q_zero",
                    ],
                    ["dequantized"],
                    name="dequantize",
                    **attrs,
                ),
            ]
        )
        current = "dequantized"
    if transpose:
        target = [2, -1, 0, 4] if zeros == "final" else [2, 3, 5, 4]
        output_shape = [2, 15, 1, 4] if zeros == "final" else [2, 3, 5, 4]
        initializers.append(numpy_helper.from_array(np.asarray(target, np.int64), name="target"))
        nodes.extend(
            [
                helper.make_node(
                    "Transpose", [current], ["transposed"], name="transpose", perm=[0, 2, 1, 4, 3]
                ),
                helper.make_node("Reshape", ["transposed", "target"], ["Y"], name="final"),
            ]
        )
    else:
        output_shape = [2, 1, 3, shape5[-2]]
        initializers.append(numpy_helper.from_array(np.asarray([-1], np.int64), name="axes"))
        nodes.append(
            helper.make_node("ReduceSum", [current, "axes"], ["Y"], name="reduce", keepdims=0)
        )
    outputs = [helper.make_tensor_value_info("Y", TensorProto.FLOAT, output_shape)]
    if collision:
        name = "reshape_qdq_reduce_0_0_slice"
        nodes.insert(0, helper.make_node("Identity", ["X"], [name], name="existing_output"))
        outputs.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, input_shape))
    model = helper.make_model(
        helper.make_graph(
            nodes,
            pattern,
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, input_shape)],
            outputs,
            initializer=initializers,
            value_info=[
                helper.make_tensor_value_info("runtime_scale", TensorProto.FLOAT, list(scale.shape))
            ]
            if quantized and late_control
            else [],
        ),
        opset_imports=[helper.make_opsetid("", 21)],
    )
    model.ir_version = 10
    onnx.checker.check_model(model, full_check=True)
    feeds = {"X": (np.arange(np.prod(input_shape), dtype=np.float32).reshape(input_shape) - 17) / 8}
    return model, feeds


def _execute(model, feeds, runtime):
    """Execute all graph outputs with the selected independent engine."""
    if runtime == "reference":
        return ReferenceEvaluator(model).run(None, feeds)
    session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    return session.run(None, feeds)


def _transform(pattern):
    """Select the registered stage for each synthetic pattern."""
    return {
        "transpose": _apply_reshape_transpose_reshape,
        "qdq": _apply_reshape_qdq_reduce_sum,
    }.get(pattern, _apply_reshape_reduce_sum)


@pytest.mark.parametrize("pattern", ["plain", "qdq", "fallback", "transpose"])
@pytest.mark.parametrize("runtime", ["reference", "ort"])
def test_registered_reshape_paths_execute(pattern, runtime):
    """Run fast, QDQ, fallback and nontrivial transpose stages through real engines."""
    model, feeds = _make_model(pattern)
    expected = _execute(model, feeds, runtime)
    transformed = _transform(pattern)(copy.deepcopy(model))
    onnx.checker.check_model(transformed, full_check=True)
    assert not any(
        node.op_type == "Reshape" and node.input[1] == "shape5" for node in transformed.graph.node
    )
    if pattern == "qdq":
        assert sum(node.op_type == "QuantizeLinear" for node in transformed.graph.node) == 2
        assert sum(node.op_type == "DequantizeLinear" for node in transformed.graph.node) == 2
    for actual, reference in zip(_execute(transformed, feeds, runtime), expected):
        np.testing.assert_array_equal(actual, reference)


@pytest.mark.parametrize("runtime", ["reference", "ort"])
def test_fast_qdq_keeps_each_nodes_controls(runtime):
    """Preserve different quantizer and dequantizer controls in every branch."""
    model, feeds = _make_model("qdq", changed_controls=True)
    expected = _execute(model, feeds, runtime)
    transformed = _apply_reshape_qdq_reduce_sum(copy.deepcopy(model))
    onnx.checker.check_model(transformed, full_check=True)
    for actual, reference in zip(_execute(transformed, feeds, runtime), expected):
        np.testing.assert_array_equal(actual, reference)


def test_fast_per_axis_qdq_refuses_before_mutation():
    """Refuse rank-sensitive per-axis middleware before adding branch initializers."""
    model, feeds = _make_model("qdq", per_axis=True)
    ReferenceEvaluator(model).run(None, feeds)
    before = model.SerializeToString()
    with pytest.raises(ValueError, match="per-tensor"):
        _apply_reshape_qdq_reduce_sum(model)
    assert model.SerializeToString() == before


@pytest.mark.parametrize("zeros", ["first", "final"])
def test_transpose_resolves_copied_dimensions(zeros):
    """Resolve copied dimensions against their original input before changing rank."""
    model, feeds = _make_model("transpose", zeros=zeros)
    expected = ReferenceEvaluator(model).run(None, feeds)
    transformed = _apply_reshape_transpose_reshape(copy.deepcopy(model))
    onnx.checker.check_model(transformed, full_check=True)
    for actual, reference in zip(ReferenceEvaluator(transformed).run(None, feeds), expected):
        np.testing.assert_array_equal(actual, reference)


@pytest.mark.parametrize("option", ["collision", "late_control"])
def test_qdq_rewrite_preserves_names_and_control_order(option):
    """Retain unrelated public tensors and controls produced after the original reshape."""
    model, feeds = _make_model("qdq", **{option: True})
    expected = ReferenceEvaluator(model).run(None, feeds)
    transformed = _apply_reshape_qdq_reduce_sum(copy.deepcopy(model))
    onnx.checker.check_model(transformed, full_check=True)
    for actual, reference in zip(ReferenceEvaluator(transformed).run(None, feeds), expected):
        np.testing.assert_array_equal(actual, reference)


def test_later_refusal_restores_earlier_planning_initializers():
    """Leave the whole model unchanged when an unsupported second pattern is found."""
    first, _ = _make_model("qdq")
    second, _ = _make_model("qdq", per_axis=True)
    first = onnx.compose.add_prefix(first, "first_")
    second = onnx.compose.add_prefix(second, "second_")
    first.graph.node.extend(second.graph.node)
    first.graph.initializer.extend(second.graph.initializer)
    first.graph.input.extend(second.graph.input)
    first.graph.output.extend(second.graph.output)
    onnx.checker.check_model(first, full_check=True)
    before = first.SerializeToString()
    with pytest.raises(ValueError, match="per-tensor"):
        _apply_reshape_qdq_reduce_sum(first)
    assert first.SerializeToString() == before


def _attach_if_capture(model, tensor, shape, dtype=TensorProto.FLOAT):
    """Attach a checked control-flow output that captures the requested outer tensor."""
    branch = helper.make_graph(
        [helper.make_node("Identity", [tensor], ["branch_output"])],
        "branch",
        [],
        [helper.make_tensor_value_info("branch_output", dtype, shape)],
    )
    model.graph.input.append(helper.make_tensor_value_info("condition", TensorProto.BOOL, []))
    model.graph.node.append(
        helper.make_node(
            "If",
            ["condition"],
            ["capture_output"],
            name="capture",
            then_branch=branch,
            else_branch=copy.deepcopy(branch),
        )
    )
    model.graph.output.append(helper.make_tensor_value_info("capture_output", dtype, shape))
    onnx.checker.check_model(model, full_check=True)


@pytest.mark.parametrize(
    ("pattern", "tensor"),
    [
        ("plain", "reshaped"),
        ("fallback", "reshaped"),
        ("qdq", "reshaped"),
        ("qdq", "quantized"),
        ("qdq", "dequantized"),
        ("transpose", "reshaped"),
        ("transpose", "transposed"),
    ],
)
def test_reshape_refuses_captured_intermediates(pattern, tensor):
    """Reject removal of captured intermediate names and restore all planned initializers."""
    model, feeds = _make_model(pattern)
    shape = (
        [2, 1, 3, 4, 5]
        if pattern == "transpose"
        else [2, 1, 3, 3 if pattern == "fallback" else 2, 4]
    )
    if tensor == "transposed":
        shape = [2, 3, 1, 5, 4]
    _attach_if_capture(
        model, tensor, shape, TensorProto.UINT8 if tensor == "quantized" else TensorProto.FLOAT
    )
    feeds["condition"] = np.asarray(True)
    ReferenceEvaluator(model).run(None, feeds)
    before = model.SerializeToString()
    with pytest.raises(ValueError, match="captur"):
        _transform(pattern)(model)
    assert model.SerializeToString() == before


@pytest.mark.parametrize("pattern", ["plain", "fallback", "qdq", "transpose"])
def test_reshape_preserves_captured_terminal_outputs(pattern):
    """Allow a captured terminal output when the replacement recreates the same name."""
    model, feeds = _make_model(pattern)
    shape = [dim.dim_value for dim in model.graph.output[0].type.tensor_type.shape.dim]
    _attach_if_capture(model, "Y", shape)
    expected = {
        condition: ReferenceEvaluator(model).run(
            None, {**feeds, "condition": np.asarray(condition)}
        )
        for condition in [False, True]
    }
    transformed = _transform(pattern)(copy.deepcopy(model))
    onnx.checker.check_model(transformed, full_check=True)
    for condition, references in expected.items():
        for actual, reference in zip(
            ReferenceEvaluator(transformed).run(
                None, {**feeds, "condition": np.asarray(condition)}
            ),
            references,
        ):
            np.testing.assert_array_equal(actual, reference)


@pytest.mark.parametrize("chained", [False, True])
def test_terminal_five_dimensional_target_is_not_an_overlap(chained):
    """Distinguish an unmatched terminal 5D target from two genuinely overlapping triples."""
    model, feeds = _make_model("transpose")
    target = next(tensor for tensor in model.graph.initializer if tensor.name == "target")
    target.CopyFrom(numpy_helper.from_array(np.asarray([2, 3, 1, 5, 4], np.int64), name="target"))
    model.graph.output[0].CopyFrom(
        helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 3, 1, 5, 4])
    )
    if chained:
        model.graph.node.extend(
            [
                helper.make_node("Transpose", ["Y"], ["again"], name="again", perm=[0, 2, 1, 4, 3]),
                helper.make_node("Reshape", ["again", "last_target"], ["Z"], name="last"),
            ]
        )
        model.graph.initializer.append(
            numpy_helper.from_array(np.asarray([2, 3, 5, 4], np.int64), name="last_target")
        )
        model.graph.output[0].CopyFrom(
            helper.make_tensor_value_info("Z", TensorProto.FLOAT, [2, 3, 5, 4])
        )
    onnx.checker.check_model(model, full_check=True)
    expected = ReferenceEvaluator(model).run(None, feeds)
    before = model.SerializeToString()
    if chained:
        with pytest.raises(ValueError, match="overlap"):
            _apply_reshape_transpose_reshape(model)
        assert model.SerializeToString() == before
    else:
        transformed = _apply_reshape_transpose_reshape(model)
        onnx.checker.check_model(transformed, full_check=True)
        np.testing.assert_array_equal(
            ReferenceEvaluator(transformed).run(None, feeds)[0], expected[0]
        )
