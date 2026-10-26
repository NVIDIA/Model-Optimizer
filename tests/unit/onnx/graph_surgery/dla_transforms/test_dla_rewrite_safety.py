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

"""Model-level safety contracts spanning the ordered DLA graph rewrites."""

import copy

import numpy as np
import onnx
import pytest
from onnx import TensorProto as T
from onnx import helper as h
from onnx import numpy_helper as nh
from onnx.reference import ReferenceEvaluator

from modelopt.onnx import graph_surgery
from modelopt.onnx.graph_surgery.dla_transforms.dla_convert_ops_to_4d import (
    _apply_convert_ops_to_4d,
)
from modelopt.onnx.graph_surgery.dla_transforms.dla_graph_cleanup import (
    _remove_consecutive_casts,
    _remove_unused_initializers,
)
from modelopt.onnx.graph_surgery.dla_transforms.dla_greater import _apply_greater
from modelopt.onnx.graph_surgery.dla_transforms.dla_indexing import _apply_gather
from modelopt.onnx.graph_surgery.dla_transforms.dla_matmul_to_transpose_conv_transpose import (
    _apply_matmul_to_transpose_conv_transpose,
)
from modelopt.onnx.graph_surgery.dla_transforms.dla_model_boundaries import (
    _apply_non4d_model_inputs,
    _apply_non4d_model_outputs,
)
from modelopt.onnx.graph_surgery.dla_transforms.dla_not import _apply_not
from modelopt.onnx.graph_surgery.dla_transforms.dla_rank_conversion import (
    _apply_argmax,
    _apply_non4d_lpnorm,
    _apply_update_axis,
)
from modelopt.onnx.graph_surgery.dla_transforms.dla_remove_deqlin import _apply_remove_deqlin
from modelopt.onnx.graph_surgery.dla_transforms.dla_remove_qdq import _remove_qdq_from_model
from modelopt.onnx.graph_surgery.dla_transforms.dla_remove_reshapes import _apply_remove_reshapes
from modelopt.onnx.graph_surgery.dla_transforms.dla_topk import _apply_topk
from modelopt.onnx.graph_surgery.dla_transforms.dla_where import _apply_where
from modelopt.onnx.graph_surgery.make_dla_compatible import _transform_make_dla_compatible
from modelopt.onnx.graph_surgery.utils.graph_utils import GraphCache


def _vi(name, dtype=T.FLOAT, shape=(2, 3)):
    return h.make_tensor_value_info(name, dtype, shape)


def _model(nodes, inputs, outputs, initializers=(), value_info=()):
    return h.make_model(
        h.make_graph(nodes, "safety", inputs, outputs, initializers, value_info=value_info),
        opset_imports=[h.make_opsetid("", 21)],
    )


def _same_values(model, transform, feeds):
    model = onnx.shape_inference.infer_shapes(model)
    expected = ReferenceEvaluator(model).run(None, feeds)
    candidate = transform(copy.deepcopy(model))
    onnx.checker.check_model(candidate)
    actual = ReferenceEvaluator(candidate).run(None, feeds)
    for left, right in zip(expected, actual, strict=True):
        np.testing.assert_allclose(right, left, rtol=1e-6, atol=1e-7)
    return candidate


@pytest.mark.parametrize("static", [False, True])
def test_shared_where_mask_keeps_both_polarities(static):
    mask = np.array([[True, False, True]])
    initializers = [nh.from_array(np.float32(0), "zero")]
    inputs = [_vi("x")]
    feeds = {"x": np.arange(6, dtype=np.float32).reshape(2, 3)}
    if static:
        initializers.append(nh.from_array(mask, "mask"))
    else:
        inputs.append(_vi("mask", T.BOOL, [1, 3]))
        feeds["mask"] = mask
    model = _model(
        [
            h.make_node("Where", ["mask", "zero", "x"], ["a"]),
            h.make_node("Where", ["mask", "x", "zero"], ["b"]),
        ],
        inputs,
        [_vi("a"), _vi("b")],
        initializers,
    )
    _same_values(model, _apply_where, feeds)


def test_shared_integer_comparisons_have_one_cast_producer():
    model = _model(
        [h.make_node("Greater", ["x", "x"], ["a"]), h.make_node("Greater", ["x", "z"], ["b"])],
        [_vi("x", T.INT32)],
        [_vi("a", T.BOOL), _vi("b", T.BOOL)],
        [nh.from_array(np.int32(2), "z")],
    )
    _same_values(model, _apply_greater, {"x": np.arange(6, dtype=np.int32).reshape(2, 3)})


def test_quantized_graph_output_and_non_dq_consumer_survive():
    model = _model(
        [
            h.make_node("QuantizeLinear", ["x", "s", "z"], ["q"]),
            h.make_node("DequantizeLinear", ["q", "s", "z"], ["y"]),
            h.make_node("Identity", ["q"], ["other"]),
        ],
        [_vi("x")],
        [_vi("q", T.INT16), _vi("y"), _vi("other", T.INT16)],
        [nh.from_array(np.float32(1), "s"), nh.from_array(np.int16(0), "z")],
    )
    model = onnx.shape_inference.infer_shapes(model)
    _remove_qdq_from_model(
        model,
        keep_clip_after_inputs=False,
        max_chained_qdq_pairs=5,
        quant_types_to_remove=frozenset({T.INT16}),
    )
    onnx.checker.check_model(model)
    assert any(node.op_type == "QuantizeLinear" for node in model.graph.node)


def test_shared_quantized_weight_with_different_scales_keeps_outputs():
    model = _model(
        [
            h.make_node("DequantizeLinear", ["w", "s1", "z"], ["a"]),
            h.make_node("DequantizeLinear", ["w", "s2", "z"], ["b"]),
        ],
        [],
        [_vi("a", shape=[2]), _vi("b", shape=[2])],
        [
            nh.from_array(np.array([2, 4], dtype=np.uint8), "w"),
            nh.from_array(np.float32(0.5), "s1"),
            nh.from_array(np.float32(0.25), "s2"),
            nh.from_array(np.uint8(0), "z"),
        ],
    )
    _same_values(model, _apply_remove_deqlin, {})


def test_blocked_dequantization_is_left_intact():
    model = _model(
        [h.make_node("DequantizeLinear", ["w", "s", "z"], ["y"], axis=1, block_size=2)],
        [],
        [_vi("y", shape=[2, 4])],
        [
            nh.from_array(np.arange(8, dtype=np.uint8).reshape(2, 4), "w"),
            nh.from_array(np.ones((2, 2), dtype=np.float32), "s"),
            nh.from_array(np.zeros((2, 2), dtype=np.uint8), "z"),
        ],
    )
    before = model.SerializeToString()
    _apply_remove_deqlin(model)
    assert model.SerializeToString() == before


def test_lossy_cast_roundtrip_keeps_its_rounding():
    model = _model(
        [
            h.make_node("Cast", ["x"], ["half"], to=T.FLOAT16),
            h.make_node("Cast", ["half"], ["y"], to=T.FLOAT),
        ],
        [_vi("x")],
        [_vi("y")],
    )
    _same_values(
        model,
        lambda m: _remove_consecutive_casts(m, GraphCache(m.graph)),
        {"x": np.full((2, 3), 1.0003, dtype=np.float32)},
    )


def test_symbolic_reshape_is_preserved():
    model = _model(
        [h.make_node("Reshape", ["x", "shape"], ["y"])],
        [_vi("x", shape=["N", 6])],
        [_vi("y", shape=["N", 2, 3])],
        [nh.from_array(np.array([0, 2, 3], dtype=np.int64), "shape")],
    )
    _same_values(
        model, _apply_remove_reshapes, {"x": np.arange(12, dtype=np.float32).reshape(2, 6)}
    )


def test_reshape_chain_keeps_exposed_intermediate():
    model = _model(
        [h.make_node("Reshape", ["x", "s1"], ["a"]), h.make_node("Reshape", ["a", "s2"], ["b"])],
        [_vi("x")],
        [_vi("a", shape=[3, 2]), _vi("b", shape=[6])],
        [
            nh.from_array(np.array([3, 2], dtype=np.int64), "s1"),
            nh.from_array(np.array([6], dtype=np.int64), "s2"),
        ],
    )
    _same_values(model, _apply_remove_reshapes, {"x": np.arange(6, dtype=np.float32).reshape(2, 3)})


@pytest.mark.parametrize("axis", [None, -2, 0, 1])
@pytest.mark.parametrize("operation", ["ArgMax", "Softmax", "LpNormalization", "Gather"])
def test_rank_promotion_preserves_axis_semantics(operation, axis):
    attrs = {} if axis is None else {"axis": axis}
    inputs = ["x"]
    initializers = []
    dtype = T.FLOAT
    shape = [2, 3]
    if operation == "ArgMax":
        dtype = T.INT64
        selected = 0 if axis is None else axis % 2
        shape = [1, 3] if selected == 0 else [2, 1]
    elif operation == "Gather":
        inputs.append("indices")
        initializers.append(nh.from_array(np.array([1], dtype=np.int64), "indices"))
        shape = [1, 3] if axis is None or axis % 2 == 0 else [2, 1]
    model = _model(
        [h.make_node(operation, inputs, ["y"], **attrs)],
        [_vi("x")],
        [_vi("y", dtype, shape)],
        initializers,
    )

    def promote(candidate):
        """Declare deferred boundary ranks before applying the operation-specific stage."""
        if operation == "Gather":
            candidate = _apply_gather(candidate)
        candidate = _apply_non4d_model_inputs(candidate)
        candidate = _apply_non4d_model_outputs(candidate, defer_rank_conversion=True)
        if operation == "ArgMax":
            candidate = _apply_argmax(candidate)
        elif operation == "LpNormalization":
            candidate = _apply_non4d_lpnorm(candidate)
        elif operation == "Softmax":
            candidate = _apply_update_axis(candidate)
        return candidate

    _same_values(
        model,
        promote,
        {"x": np.array([[1, 5, 3], [4, 2, 6]], dtype=np.float32)},
    )


def test_not_preserves_nonbinary_and_shared_boolean_uses():
    model = _model(
        [
            h.make_node("Cast", ["x"], ["mask"], to=T.BOOL),
            h.make_node("Not", ["mask"], ["y"]),
            h.make_node("Identity", ["mask"], ["other"]),
        ],
        [_vi("x")],
        [_vi("y", T.BOOL), _vi("other", T.BOOL)],
    )
    _same_values(model, _apply_not, {"x": np.array([[0, -3, 0.5], [1, -1, 0]], dtype=np.float32)})


@pytest.mark.parametrize("change", ["alpha", "bias"])
def test_unsupported_gemm_refuses_before_mutation(change):
    attrs = {"alpha": 2.0} if change == "alpha" else {}
    bias = np.ones((2, 3) if change == "bias" else (3,), dtype=np.float32)
    model = _model(
        [h.make_node("Gemm", ["x", "w", "b"], ["y"], **attrs)],
        [_vi("x")],
        [_vi("y")],
        [nh.from_array(np.eye(3, dtype=np.float32), "w"), nh.from_array(bias, "b")],
    )
    before = model.SerializeToString()
    with pytest.raises(ValueError):
        _apply_matmul_to_transpose_conv_transpose(model)
    assert model.SerializeToString() == before


@pytest.mark.parametrize("deferred", [False, True])
def test_broadcast_matmul_restores_output_rank(deferred):
    model = _model(
        [h.make_node("MatMul", ["x", "w"], ["y"])],
        [_vi("x")],
        [_vi("y", shape=[1, 2, 3])],
        [nh.from_array(np.eye(3, dtype=np.float32)[None], "w")],
    )
    _same_values(
        model,
        (lambda model: _transform_make_dla_compatible(model, verbose=False))
        if deferred
        else _apply_matmul_to_transpose_conv_transpose,
        {"x": np.arange(6, dtype=np.float32).reshape(2, 3)},
    )


@pytest.mark.parametrize("axes", [None, [-1]])
def test_legacy_reduce_restores_scalar_or_reduced_shape(axes):
    inputs = ["x"]
    initializers = []
    if axes is not None:
        inputs.append("axes")
        initializers.append(nh.from_array(np.array(axes, dtype=np.int64), "axes"))
    model = _model(
        [h.make_node("ReduceSum", inputs, ["y"], keepdims=0)],
        [_vi("x")],
        [_vi("y", shape=[] if axes is None else [2])],
        initializers,
    )
    _same_values(
        model, _apply_convert_ops_to_4d, {"x": np.arange(6, dtype=np.float32).reshape(2, 3)}
    )


def test_legacy_rank_dependent_op_is_left_intact():
    model = _model(
        [h.make_node("Einsum", ["x"], ["y"], equation="ij->ji")],
        [_vi("x")],
        [_vi("y", shape=[3, 2])],
    )
    _same_values(
        model, _apply_convert_ops_to_4d, {"x": np.arange(6, dtype=np.float32).reshape(2, 3)}
    )


def test_dispatcher_keeps_existing_return_and_keyword_contract(monkeypatch):
    calls = []

    def original(**kwargs):
        calls.append(kwargs)
        return {"converted": 3}

    monkeypatch.setitem(graph_surgery._SURGERY_REGISTRY, "convert-bf16", original)
    result = graph_surgery.run_graph_surgery(
        "convert-bf16", "input.onnx", "output.onnx", external_data=False, verbose=False
    )
    assert result == {"converted": 3}
    assert calls == [
        {
            "model_path": "input.onnx",
            "output_path": "output.onnx",
            "external_data": False,
            "verbose": False,
        }
    ]


def test_control_flow_capture_keeps_initializer():
    branch = h.make_graph(
        [h.make_node("Identity", ["w"], ["v"])], "branch", [], [_vi("v", shape=[2])]
    )
    model = _model(
        [h.make_node("If", ["condition"], ["y"], then_branch=branch, else_branch=branch)],
        [_vi("condition", T.BOOL, [])],
        [_vi("y", shape=[2])],
        [nh.from_array(np.array([1, 2], dtype=np.float32), "w")],
    )
    _same_values(model, _remove_unused_initializers, {"condition": np.array(True)})


def test_rank_converted_topk_chain_preserves_gathered_values():
    model = _model(
        [
            h.make_node("TopK", ["scores", "k"], ["values", "indices"], axis=3),
            h.make_node("Cast", ["indices"], ["cast1"], to=T.INT64),
            h.make_node("Reshape", ["cast1", "shape"], ["reshaped"]),
            h.make_node("Tile", ["reshaped", "repeats"], ["tiled"]),
            h.make_node("Cast", ["tiled"], ["cast2"], to=T.INT32),
            h.make_node("GatherElements", ["data", "cast2"], ["y"], axis=2),
        ],
        [_vi("scores", shape=[1, 1, 1, 4]), _vi("data", shape=[1, 1, 4, 3])],
        [_vi("y", shape=[1, 1, 2, 3])],
        [
            nh.from_array(np.array([2], dtype=np.int64), "k"),
            nh.from_array(np.array([1, 1, 2, 1], dtype=np.int64), "shape"),
            nh.from_array(np.array([1, 1, 1, 3], dtype=np.int64), "repeats"),
        ],
    )
    candidate = _same_values(
        model,
        lambda m: _apply_topk(m, rank_converted=True),
        {
            "scores": np.array([[[[1, 4, 2, 3]]]], dtype=np.float32),
            "data": np.arange(12, dtype=np.float32).reshape(1, 1, 4, 3),
        },
    )
    assert not any(node.op_type in {"Tile", "GatherElements"} for node in candidate.graph.node)
