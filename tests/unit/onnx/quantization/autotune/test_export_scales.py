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

"""Tests for the pseudo-random activation scales used in autotune candidates."""

import numpy as np
import onnx_graphsurgeon as gs
import pytest
from onnx import TensorProto, helper, numpy_helper

import modelopt.onnx.quantization.autotune.export_utils as export_utils
from modelopt.onnx.quantization.autotune.common import Config
from modelopt.onnx.quantization.autotune.export_utils import (
    _ACTIVATION_SCALE_RANGE,
    build_activation_scales,
    build_scale_group_map,
    export_qdq_onnx,
)
from modelopt.onnx.quantization.autotune.insertion_points import ResolvedInsertionPoint


def _weight(name, shape, seed):
    rng = np.random.RandomState(seed)
    return numpy_helper.from_array((rng.randn(*shape) * 0.1).astype(np.float32), name)


@pytest.fixture
def model():
    """Conv -> Relu -> MaxPool -> Conv, with the Relu output also feeding a Concat."""
    nodes = [
        helper.make_node("Conv", ["x", "w1"], ["c1"], name="conv1", pads=[1, 1, 1, 1]),
        helper.make_node("Relu", ["c1"], ["r1"], name="relu1"),
        helper.make_node(
            "MaxPool", ["r1"], ["p1"], name="pool1", kernel_shape=[2, 2], strides=[2, 2]
        ),
        helper.make_node("Conv", ["p1", "w2"], ["c2"], name="conv2", pads=[1, 1, 1, 1]),
        helper.make_node("Relu", ["c2"], ["r2"], name="relu2"),
        helper.make_node("Concat", ["p1", "r2"], ["cat"], name="concat1", axis=1),
        helper.make_node("Conv", ["cat", "w3"], ["out"], name="conv3", pads=[1, 1, 1, 1]),
    ]
    inits = [
        _weight("w1", (8, 4, 3, 3), 0),
        _weight("w2", (8, 8, 3, 3), 1),
        _weight("w3", (4, 16, 3, 3), 2),
    ]
    graph = helper.make_graph(
        nodes,
        "scales",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4, 16, 16])],
        [helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 4, 8, 8])],
        initializer=inits,
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])


@pytest.fixture
def rearrange_model():
    """Conv1 output feeds Conv3 directly and Conv2 through Reshape -> Transpose."""
    nodes = [
        helper.make_node("Conv", ["x", "w1"], ["a"], name="conv1"),
        helper.make_node("Conv", ["a", "w3"], ["out1"], name="conv3"),
        helper.make_node("Reshape", ["a", "shape"], ["b"], name="reshape1"),
        helper.make_node("Transpose", ["b"], ["c"], name="transpose1", perm=[0, 1, 3, 2]),
        helper.make_node("Conv", ["c", "w2"], ["out2"], name="conv2"),
    ]
    inits = [
        _weight("w1", (8, 4, 1, 1), 0),
        _weight("w2", (4, 8, 1, 1), 1),
        _weight("w3", (4, 8, 1, 1), 2),
        numpy_helper.from_array(np.array([1, 8, 16, 1], dtype=np.int64), "shape"),
    ]
    graph = helper.make_graph(
        nodes,
        "rearrange",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4, 4, 4])],
        [
            helper.make_tensor_value_info("out1", TensorProto.FLOAT, [1, 4, 4, 4]),
            helper.make_tensor_value_info("out2", TensorProto.FLOAT, [1, 4, 1, 16]),
        ],
        initializer=inits,
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])


def _points(model, specs):
    graph = gs.import_onnx(model)
    graph.toposort()
    index = {node.name: i for i, node in enumerate(graph.nodes)}
    return {
        ResolvedInsertionPoint(tensor_name=tensor, node_index=index[node], input_index=inp)
        for node, inp, tensor in specs
    }


def _activation_scales(model):
    inits = {i.name: i for i in model.graph.initializer}
    return {
        node.input[0]: float(np.asarray(numpy_helper.to_array(inits[node.input[1]])).flatten()[0])
        for node in model.graph.node
        if node.op_type == "QuantizeLinear" and node.input[0] not in inits
    }


def test_groups_get_their_own_scale(model):
    """Tensors share a scale exactly when they are in one group."""
    graph = gs.import_onnx(model)
    graph.toposort()
    quantized = {"x", "c1", "r1", "p1", "c2", "r2", "cat", "out"}
    groups = build_scale_group_map(graph, quantized)
    scales = build_activation_scales(quantized, groups)
    low, high = _ACTIVATION_SCALE_RANGE
    assert all(low <= value <= high for value in scales.values())
    for a in quantized:
        for b in quantized:
            same_group = groups.get(a, a) == groups.get(b, b)
            assert (scales[a] == scales[b]) == same_group


@pytest.mark.parametrize(("bfloat16", "count"), [(False, 3000), (True, 1500)])
def test_distinct_groups_never_share_a_scale(bfloat16, count):
    """Hashed scales of unrelated groups collide once rounded to 16 bits; they must not."""
    scales = build_activation_scales({f"t{i}" for i in range(count)}, {}, bfloat16=bfloat16)
    values = np.array(list(scales.values()), dtype=np.float32)
    low, high = _ACTIVATION_SCALE_RANGE
    assert len(set(values.tolist())) == count
    assert np.all((values >= low * 0.99) & (values <= high))
    if bfloat16:
        assert np.all(values.view(np.uint32) & 0xFFFF == 0)
    else:
        assert np.array_equal(values.astype(np.float16).astype(np.float32), values)


def test_fp8_scales_stay_distinct_after_conversion():
    """int8_to_fp8 rescales by 448/127 in float16, which merges some adjacent grid values."""
    scales = build_activation_scales({f"t{i}" for i in range(3000)}, {}, fp8=True)
    stored = np.array(list(scales.values()), dtype=np.float16)
    converted = ((stored * 448.0) / 127.0).astype(np.float16)
    assert len(set(converted.tolist())) == len(scales)


@pytest.mark.parametrize("needs_fp8_conversion", [False, True])
def test_fp8_export_checks_scales_after_conversion(model, monkeypatch, needs_fp8_conversion):
    """export_qdq_onnx inserts INT8 Q/DQ before converting to FP8; the intent must reach the scales."""
    calls = []
    build = export_utils.build_activation_scales

    def spy(*args, **kwargs):
        calls.append(kwargs)
        return build(*args, **kwargs)

    monkeypatch.setattr(export_utils, "build_activation_scales", spy)
    points = _points(model, [("conv2", 0, "p1"), ("conv3", 0, "cat")])
    config = Config(default_quant_type="fp8" if needs_fp8_conversion else "int8")
    export_qdq_onnx(model, points, config, needs_fp8_conversion=needs_fp8_conversion)
    assert [call["fp8"] for call in calls] == [needs_fp8_conversion]


def test_more_groups_than_scales_stay_in_range():
    scales = build_activation_scales({f"t{i}" for i in range(2000)}, {}, bfloat16=True)
    low, high = _ACTIVATION_SCALE_RANGE
    assert all(low * 0.99 <= value <= high for value in scales.values())


def test_scale_is_stable_across_candidates(model):
    """A tensor keeps its scale when unrelated insertion points change, absent hash collisions."""
    common = ("conv3", 0, "cat")
    a = _activation_scales(
        export_qdq_onnx(model, _points(model, [common, ("conv1", 0, "x")]), Config())
    )
    b = _activation_scales(export_qdq_onnx(model, _points(model, [common]), Config()))
    assert a["cat"] == b["cat"]


def test_concat_does_not_share_with_unquantized_inputs(model):
    graph = gs.import_onnx(model)
    graph.toposort()
    assert build_scale_group_map(graph, {"cat"}) == {"cat": "cat"}
    assert build_scale_group_map(graph, {"p1", "r2"}) == {"p1": "p1", "r2": "r2"}


def test_maxpool_does_not_share(model):
    """MaxPool changes the value distribution, so the two sides are calibrated apart."""
    scales = _activation_scales(
        export_qdq_onnx(model, _points(model, [("pool1", 0, "r1"), ("conv2", 0, "p1")]), Config())
    )
    assert scales["r1"] != scales["p1"]


def test_concat_shares_scale_with_its_quantized_inputs(model):
    scales = _activation_scales(
        export_qdq_onnx(model, _points(model, [("conv2", 0, "p1"), ("conv3", 0, "cat")]), Config())
    )
    assert scales["p1"] == scales["cat"]


def test_rearrange_chain_shares_scale_through_unquantized_tensor(rearrange_model):
    """Reshape -> Transpose only reorder values, so both ends share a scale without Q/DQ on b."""
    points = _points(rearrange_model, [("conv3", 0, "a"), ("conv2", 0, "c")])
    scales = _activation_scales(export_qdq_onnx(rearrange_model, points, Config()))
    assert set(scales) == {"a", "c"}
    assert scales["a"] == scales["c"]


def test_rearrange_shape_input_is_not_grouped(rearrange_model):
    graph = gs.import_onnx(rearrange_model)
    graph.toposort()
    groups = build_scale_group_map(graph, {"a", "c", "shape"})
    assert groups["a"] == groups["c"]
    assert groups["shape"] != groups["a"]


def test_flag_off_restores_the_uniform_scale(model):
    points = _points(model, [("pool1", 0, "r1"), ("conv2", 0, "p1"), ("conv3", 0, "cat")])
    config = Config(use_varied_activation_scales=False)
    scales = _activation_scales(export_qdq_onnx(model, points, config))
    assert scales
    assert all(value == pytest.approx(config.default_q_scale) for value in scales.values())


def test_weights_keep_the_default_scale(model):
    config = Config()
    points = _points(model, [("conv1", 0, "x"), ("conv1", 1, "w1")])
    exported = export_qdq_onnx(model, points, config)
    inits = {i.name: i for i in exported.graph.initializer}
    weight_scales = [
        float(np.asarray(numpy_helper.to_array(inits[node.input[1]])).flatten()[0])
        for node in exported.graph.node
        if node.op_type == "QuantizeLinear" and node.input[0] in inits
    ]
    assert weight_scales
    assert all(value == pytest.approx(config.default_q_scale) for value in weight_scales)
    activations = _activation_scales(exported).values()
    assert all(value != pytest.approx(config.default_q_scale) for value in activations)
