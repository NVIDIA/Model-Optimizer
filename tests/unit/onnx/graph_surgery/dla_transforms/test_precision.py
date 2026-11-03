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

"""Execute FP16 preparation and bounded projection rules."""

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper, numpy_helper
from onnxscript import ir

from modelopt.onnx.graph_surgery.dla_transforms.context import DlaGraph
from modelopt.onnx.graph_surgery.dla_transforms.patterns.rules import TerminalProjection
from modelopt.onnx.graph_surgery.dla_transforms.stages import precision


def _model(nodes, inputs, outputs, initializers, opset=21):
    """Create a fully inferred graph suitable for the final precision phase."""
    model = helper.make_model(
        helper.make_graph(nodes, "precision", inputs, outputs, initializers),
        opset_imports=[helper.make_opsetid("", opset)],
        ir_version=10,
    )
    return onnx.shape_inference.infer_shapes(model, strict_mode=True, check_type=True)


def _finish(working):
    """Export and fully check the real rewritten IR graph."""
    result = ir.serde.serialize_model(working)
    onnx.checker.check_model(result, full_check=True)
    return result


def _run(model, feeds):
    """Use one ORT CPU thread for the small numerical behavior fixtures."""
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    return ort.InferenceSession(
        model.SerializeToString(), options, providers=["CPUExecutionProvider"]
    ).run(None, feeds)


def test_fp16_boundaries_keep_public_io_and_share_input_cast():
    model = _model(
        [
            helper.make_node("Add", ["x", "bias"], ["sum"]),
            helper.make_node("Mul", ["sum", "x"], ["y"]),
        ],
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, 2, 2])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1, 2, 2])],
        [numpy_helper.from_array(np.asarray(0.5, np.float32), "bias")],
    )
    working = ir.serde.deserialize_model(model)
    assert precision(DlaGraph(working)) == 3
    result = _finish(working)
    assert result.graph.input[0].name == "x"
    assert result.graph.output[0].name == "y"
    assert result.graph.initializer[0].data_type == TensorProto.FLOAT16
    assert sum(node.op_type == "Cast" for node in result.graph.node) == 2
    feeds = {"x": np.asarray([0, 0.5, 1, 2], np.float32).reshape(1, 1, 2, 2)}
    np.testing.assert_array_equal(_run(model, feeds)[0], _run(result, feeds)[0])


@pytest.mark.parametrize("kind", ["mixed", "scalar", "empty", "noncontiguous"])
def test_fp16_initializer_range_preserves_special_values_and_zero_masking(kind):
    values = np.asarray(
        [
            -1e9,
            1e9,
            -65504,
            65504,
            -0.0,
            0.0,
            -0.5,
            0.5,
            2**-24,
            2**-25,
            -(2**-25),
            np.nan,
            -np.inf,
            np.inf,
        ],
        np.float32,
    ).reshape(1, 1, 2, 7)
    expected = np.asarray(
        [
            -65504,
            65504,
            -65504,
            65504,
            -0.0,
            0.0,
            -0.5,
            0.5,
            2**-24,
            0.0,
            -0.0,
            np.nan,
            -np.inf,
            np.inf,
        ],
        np.float16,
    ).reshape(values.shape)
    mask = np.ones_like(values)
    mask.reshape(-1)[:2] = 0
    if kind == "scalar":
        values, expected, mask = values[0, 0, 0, 0], expected[0, 0, 0, 0], np.float32(0)
    elif kind == "empty":
        values, expected, mask = values[..., :0], expected[..., :0], mask[..., :0]
    elif kind == "noncontiguous":
        values, expected, mask = (array.swapaxes(-1, -2) for array in (values, expected, mask))
        assert not values.flags.c_contiguous
    model = _model(
        [helper.make_node("Mul", ["mask", "fill"], ["y"])],
        [helper.make_tensor_value_info("mask", TensorProto.FLOAT, list(values.shape))],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, list(values.shape))],
        [numpy_helper.from_array(np.asarray(values), "fill")],
    )
    working = ir.serde.deserialize_model(model)
    working.graph.initializers["fill"].const_value = ir.tensor(np.asarray(values))
    original = values.copy()
    assert precision(DlaGraph(working)) == 3
    result = _finish(working)
    converted = numpy_helper.to_array(result.graph.initializer[0])
    assert converted.dtype == np.float16
    np.testing.assert_array_equal(converted, expected)
    zeros = expected == 0
    np.testing.assert_array_equal(np.signbit(converted[zeros]), np.signbit(expected[zeros]))
    np.testing.assert_array_equal(values, original)
    actual = _run(result, {"mask": np.asarray(mask)})[0]
    expected_output = expected.astype(np.float32) * mask
    np.testing.assert_array_equal(actual, expected_output)


def test_fp16_preserves_schema_fp32_region_and_casts_its_shared_reader():
    model = _model(
        [
            helper.make_node("Add", ["x", "offset"], ["boxes"]),
            helper.make_node("NonMaxSuppression", ["boxes", "scores", "maximum"], ["selected"]),
            helper.make_node("Mul", ["boxes", "gain"], ["y"]),
        ],
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, 4])],
        [
            helper.make_tensor_value_info("selected", TensorProto.INT64, [None, 3]),
            helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2, 4]),
        ],
        [
            numpy_helper.from_array(np.asarray(0.5, np.float32), "offset"),
            numpy_helper.from_array(np.asarray([[[1, 0.5]]], np.float32), "scores"),
            numpy_helper.from_array(np.asarray(2, np.int64), "maximum"),
            numpy_helper.from_array(np.asarray(2, np.float32), "gain"),
        ],
    )
    working = ir.serde.deserialize_model(model)
    assert precision(DlaGraph(working)) == 3
    result = _finish(working)
    dtypes = {value.name: value.data_type for value in result.graph.initializer}
    assert dtypes["offset"] == dtypes["scores"] == TensorProto.FLOAT
    assert dtypes["gain"] == TensorProto.FLOAT16
    add = next(node for node in result.graph.node if node.op_type == "Add")
    assert add.input[0] == "x"
    feeds = {"x": np.asarray([[[0, 0, 1, 1], [2, 2, 3, 3]]], np.float32)}
    for original, transformed in zip(_run(model, feeds), _run(result, feeds), strict=True):
        np.testing.assert_array_equal(original, transformed)


@pytest.mark.parametrize("axes", [[0, 1], [0]])
def test_fp16_promoted_input_reuses_its_existing_cast(axes):
    model = _model(
        [
            helper.make_node("Unsqueeze", ["x", "axes"], ["promoted"]),
            helper.make_node("Cast", ["promoted"], ["explicit"], to=TensorProto.FLOAT),
            helper.make_node("Add", ["explicit", "bias"], ["sum"]),
            helper.make_node("Squeeze", ["sum", "axes"], ["y"]),
        ],
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 3])],
        [
            numpy_helper.from_array(np.asarray(axes, np.int64), "axes"),
            numpy_helper.from_array(np.asarray(0.5, np.float32), "bias"),
        ],
    )
    working = ir.serde.deserialize_model(model)
    assert precision(DlaGraph(working)) == (3 if len(axes) == 2 else 4)
    result = _finish(working)
    uses = {name for node in result.graph.node for name in node.input}
    casts = [node for node in result.graph.node if node.op_type == "Cast"]
    assert len(casts) == (2 if len(axes) == 2 else 3)
    assert all(node.output[0] in uses for node in casts)
    feeds = {"x": np.arange(6, dtype=np.float32).reshape(2, 3)}
    np.testing.assert_array_equal(_run(model, feeds)[0], _run(result, feeds)[0])


@pytest.mark.parametrize(
    ("kind", "shared", "channels"),
    [
        ("fp16", False, 8195),
        ("scalar_dq", False, 8195),
        ("scalar_dq", True, 8195),
        ("identity", False, 8195),
        ("fp16", False, 16),
        ("per_axis_dq", False, 8195),
    ],
)
def test_terminal_projection_rule_bounds_channels_and_preserves_values(kind, shared, channels):
    quantized = kind in {"scalar_dq", "per_axis_dq"}
    rewritten = kind in {"fp16", "scalar_dq"} and not shared and channels > 8192
    weight = np.ones((channels, 1, 1, 1), np.uint8 if quantized else np.float16)
    nodes = []
    initializers = [
        numpy_helper.from_array(weight, "weight"),
        numpy_helper.from_array(np.asarray([0], np.int64), "axes"),
    ]
    if quantized:
        nodes.append(
            helper.make_node("DequantizeLinear", ["weight", "scale", "zero"], ["dq"], axis=0)
        )
        initializers.extend(
            [
                numpy_helper.from_array(
                    np.full(channels, 0.5, np.float16)
                    if kind == "per_axis_dq"
                    else np.asarray(0.5, np.float16),
                    "scale",
                ),
                numpy_helper.from_array(
                    np.zeros(channels, np.uint8)
                    if kind == "per_axis_dq"
                    else np.asarray(0, np.uint8),
                    "zero",
                ),
            ]
        )
    if kind == "identity":
        nodes.append(helper.make_node("Identity", ["weight"], ["viewed_weight"]))
    kernel = "dq" if quantized else "viewed_weight" if kind == "identity" else "weight"
    nodes.extend(
        [
            helper.make_node("Conv", ["x", kernel], ["conv"]),
            helper.make_node("Transpose", ["conv"], ["transposed"], perm=[0, 3, 2, 1]),
            helper.make_node("Concat", ["transposed"], ["first"], axis=0),
            helper.make_node("ReduceSum", ["first", "axes"], ["reduced"], keepdims=1),
            helper.make_node("Concat", ["reduced"], ["last"], axis=0),
            helper.make_node("Cast", ["last"], ["fp32"], to=TensorProto.FLOAT),
            helper.make_node("Squeeze", ["fp32", "axes"], ["y"]),
        ]
    )
    outputs = [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1, channels])]
    if shared:
        nodes.append(helper.make_node("Identity", ["dq"], ["shared"]))
        outputs.append(
            helper.make_tensor_value_info("shared", TensorProto.FLOAT16, [channels, 1, 1, 1])
        )
    model = _model(
        nodes,
        [helper.make_tensor_value_info("x", TensorProto.FLOAT16, [1, 1, 1, 1])],
        outputs,
        initializers,
    )
    working = ir.serde.deserialize_model(model)
    graph = DlaGraph(working)
    assert graph.apply_pattern(TerminalProjection.rule(graph)) == int(rewritten)
    ir.passes.common.RemoveUnusedNodesPass()(working)
    result = _finish(working)
    assert sum(node.op_type == "Conv" for node in result.graph.node) == (2 if rewritten else 1)
    assert max(value.dims[0] for value in result.graph.initializer if len(value.dims) == 4) == (
        8192 if rewritten else channels
    )
    feeds = {"x": np.asarray([[[[2]]]], np.float16)}
    for original, transformed in zip(_run(model, feeds), _run(result, feeds), strict=True):
        np.testing.assert_array_equal(original, transformed)


@pytest.mark.parametrize("kind", ["unknown", "overload", "sequence"])
def test_fp16_preserves_native_operation_regions_and_adapts_shared_arithmetic(kind):
    """Opaque and sequence operations retain FLOAT while a shared arithmetic view narrows."""
    tensor = ir.TensorType(ir.DataType.FLOAT)
    shape = ir.Shape([1, 1, 2, 2])
    source = ir.Value(name="x", shape=shape, type=tensor)
    constant = ir.Value(
        name="bias",
        shape=ir.Shape([]),
        type=tensor,
        const_value=ir.tensor(np.array(0.5, np.float32)),
    )
    summed = ir.Value(name="summed", shape=shape, type=tensor)
    addition = ir.Node("", "Add", [source, constant], outputs=[summed])
    native_output = ir.Value(
        name="native",
        type=ir.SequenceType(tensor) if kind == "sequence" else tensor,
        shape=None if kind == "sequence" else shape,
    )
    native = ir.Node(
        "custom" if kind == "unknown" else "",
        "Opaque"
        if kind == "unknown"
        else "SequenceConstruct"
        if kind == "sequence"
        else "Identity",
        [summed],
        outputs=[native_output],
        overload="native" if kind == "overload" else "",
    )
    result = ir.Value(name="y", shape=shape, type=tensor)
    arithmetic = ir.Node("", "Mul", [summed, constant], outputs=[result])
    working = ir.Model(
        ir.Graph(
            [source],
            [native_output, result],
            nodes=[addition, native, arithmetic],
            initializers=[constant],
            opset_imports={"": 21, "custom": 1},
            name="native_regions",
        ),
        ir_version=10,
    )
    precision(DlaGraph(working))
    assert native.inputs[0] is summed
    assert source.dtype == summed.dtype == constant.dtype == ir.DataType.FLOAT
    assert arithmetic.inputs[0].producer().op_type == "Cast"
    assert arithmetic.inputs[0].dtype == arithmetic.outputs[0].dtype == ir.DataType.FLOAT16
    assert working.graph.outputs[1].dtype == ir.DataType.FLOAT
    if kind == "sequence":
        serialized = _finish(working)
        feeds = {"x": np.arange(4, dtype=np.float32).reshape(1, 1, 2, 2)}
        sequence, numeric = _run(serialized, feeds)
        np.testing.assert_array_equal(sequence[0], feeds["x"] + 0.5)
        np.testing.assert_array_equal(numeric, (feeds["x"] + 0.5) * 0.5)
