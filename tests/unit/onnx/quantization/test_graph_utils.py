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

import numpy as np
import onnx_graphsurgeon as gs
import pytest
from onnx import TensorProto, helper

from modelopt.onnx.quantization.graph_utils import (
    _exclude_matmuls_by_shape_inference,
    _get_inp_b_k_dim,
    find_nodes_from_convs_to_exclude,
)


def _make_conv_graph(output_channels, input_channels, kernel_shape=(3, 3), name="Conv_0"):
    """Build a minimal graph with a single Conv node."""
    spatial = [32, 32]
    inp = gs.Variable(name="input", dtype=np.float32, shape=[1, input_channels, *spatial])
    out = gs.Variable(name="output", dtype=np.float32)

    weight_shape = (output_channels, input_channels, *kernel_shape)
    weight = gs.Constant(name="weight", values=np.ones(weight_shape, dtype=np.float32))

    conv = gs.Node(
        name=name,
        op="Conv",
        inputs=[inp, weight],
        outputs=[out],
        attrs={"kernel_shape": list(kernel_shape)},
    )

    return gs.Graph(nodes=[conv], inputs=[inp], outputs=[out], opset=13)


@pytest.mark.parametrize(
    ("oc", "ic", "expected_excluded"),
    [
        (16, 64, True),
        (64, 16, True),
        (8, 8, True),
        (16, 16, True),
        (17, 64, False),
        (64, 17, False),
        (17, 17, False),
        (32, 32, False),
        (64, 64, False),
    ],
)
def test_fp8_small_channel_conv_exclusion(oc, ic, expected_excluded):
    """FP8 mode should exclude Conv nodes with OC or IC <= 16."""
    graph = _make_conv_graph(output_channels=oc, input_channels=ic)
    excluded = find_nodes_from_convs_to_exclude(graph, quantize_mode="fp8")
    if expected_excluded:
        assert "Conv_0" in excluded
    else:
        assert "Conv_0" not in excluded


def test_fp8_small_channel_exclusion_does_not_affect_int8():
    """The small-channel FP8 exclusion should not apply in int8 mode."""
    # OC=8 would be excluded in FP8 (see oc=8, ic=8 case above), but not in int8.
    graph = _make_conv_graph(output_channels=8, input_channels=64, kernel_shape=(3, 3))
    excluded = find_nodes_from_convs_to_exclude(graph, quantize_mode="int8")
    assert "Conv_0" not in excluded


@pytest.mark.parametrize(
    ("oc", "ic"),
    [
        (15, 64),
        (64, 15),
        (1, 1),
    ],
)
def test_fp8_channels_below_16_excluded_by_general_check(oc, ic):
    """Channels strictly < 16 are excluded by the general channel check, not the FP8 check."""
    graph = _make_conv_graph(output_channels=oc, input_channels=ic, kernel_shape=(3, 3))
    excluded = find_nodes_from_convs_to_exclude(graph, quantize_mode="fp8")
    assert "Conv_0" in excluded


def _make_matmul_model(m, k, n, name="MatMul_0", inp_b_constant=True):
    """Build a minimal ONNX model with a single MatMul: [M, K] x [K, N] -> [M, N]."""
    inp_a = helper.make_tensor_value_info("A", TensorProto.FLOAT, [m, k])
    out = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [m, n])

    if inp_b_constant:
        b_init = helper.make_tensor("B", TensorProto.FLOAT, [k, n], np.ones(k * n).tolist())
        matmul = helper.make_node("MatMul", ["A", "B"], ["Y"], name=name)
        graph = helper.make_graph([matmul], "test", [inp_a], [out], initializer=[b_init])
    else:
        inp_b = helper.make_tensor_value_info("B", TensorProto.FLOAT, [k, n])
        matmul = helper.make_node("MatMul", ["A", "B"], ["Y"], name=name)
        graph = helper.make_graph([matmul], "test", [inp_a, inp_b], [out])

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    return model


def _get_matmul_nodes(model):
    """Import an ONNX model and return its MatMul gs.Nodes."""
    graph = gs.import_onnx(model)
    return [n for n in graph.nodes if n.op == "MatMul"]


def test_get_inp_b_k_dim_constant():
    """K dimension should be read from the Constant weight shape."""
    model = _make_matmul_model(m=32, k=8, n=64)
    nodes = _get_matmul_nodes(model)
    assert _get_inp_b_k_dim(nodes[0]) == 8


def test_get_inp_b_k_dim_variable_with_output_map():
    """K dimension should be read from output_map for Variable inputs."""
    model = _make_matmul_model(m=32, k=10, n=64, inp_b_constant=False)
    nodes = _get_matmul_nodes(model)
    output_map = {"B": np.zeros((10, 64))}
    assert _get_inp_b_k_dim(nodes[0], output_map=output_map) == 10


def test_get_inp_b_k_dim_returns_none_when_unknown():
    """Should return None if K cannot be determined."""
    model = _make_matmul_model(m=32, k=8, n=64, inp_b_constant=False)
    nodes = _get_matmul_nodes(model)
    assert _get_inp_b_k_dim(nodes[0]) is None


@pytest.mark.parametrize(
    ("m", "k", "n", "expected_excluded"),
    [
        (32, 64, 8, True),
        (32, 64, 15, True),
        (32, 8, 64, True),
        (32, 15, 64, True),
        (32, 8, 8, True),
        (32, 64, 16, False),
        (32, 16, 64, False),
        (32, 64, 64, False),
        (32, 32, 32, False),
    ],
)
def test_matmul_small_gemm_exclusion(m, k, n, expected_excluded):
    """MatMuls with N or K < 16 should be excluded by shape inference."""
    model = _make_matmul_model(m=m, k=k, n=n)
    nodes = _get_matmul_nodes(model)
    calibration_shapes = {"A": [m, k]}
    excluded = _exclude_matmuls_by_shape_inference(model, nodes, calibration_shapes)
    if expected_excluded:
        assert "MatMul_0" in excluded
    else:
        assert "MatMul_0" not in excluded


def test_matmul_gemv_excluded():
    """MatMul with N=1 (GEMV) should be excluded regardless of other dims."""
    model = _make_matmul_model(m=32, k=64, n=1)
    nodes = _get_matmul_nodes(model)
    calibration_shapes = {"A": [32, 64]}
    excluded = _exclude_matmuls_by_shape_inference(model, nodes, calibration_shapes)
    assert "MatMul_0" in excluded


def test_matmul_large_dims_not_excluded():
    """MatMul with all large dims should not be excluded."""
    model = _make_matmul_model(m=128, k=256, n=64)
    nodes = _get_matmul_nodes(model)
    calibration_shapes = {"A": [128, 256]}
    excluded = _exclude_matmuls_by_shape_inference(model, nodes, calibration_shapes)
    assert "MatMul_0" not in excluded
