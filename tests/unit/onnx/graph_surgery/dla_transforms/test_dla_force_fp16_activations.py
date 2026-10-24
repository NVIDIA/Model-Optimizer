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


def _value(name, shape, dtype=T.FLOAT):
    return h.make_tensor_value_info(name, dtype, shape)


def _model(nodes, inputs, outputs, initializers, opset=19):
    model = h.make_model(
        h.make_graph(nodes, "fp16", inputs, outputs, initializers),
        opset_imports=[h.make_opsetid("", opset)],
    )
    model.ir_version = 9
    return onnx.shape_inference.infer_shapes(model)


@pytest.mark.parametrize("enabled", [False, True])
def test_public_option_preserves_boundaries_and_selects_weight_precision(tmp_path, enabled):
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


def test_terminal_projection_splits_channels_and_preserves_values():
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
