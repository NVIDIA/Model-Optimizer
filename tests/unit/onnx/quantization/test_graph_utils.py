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

from unittest import mock

import numpy as np
import onnx_graphsurgeon as gs
import pytest
from onnx import TensorProto, helper

from modelopt.onnx.quantization.graph_utils import (
    _exclude_matmuls_by_inference,
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


def _get_nodes_by_op(model, op):
    """Import an ONNX model and return its gs.Nodes whose op matches ``op``."""
    graph = gs.import_onnx(model)
    return [n for n in graph.nodes if n.op == op]


def test_get_inp_b_k_dim_constant():
    """K dimension should be read from the Constant weight shape."""
    model = _make_matmul_model(m=32, k=8, n=64)
    nodes = _get_nodes_by_op(model, "MatMul")
    assert _get_inp_b_k_dim(nodes[0]) == 8


def test_get_inp_b_k_dim_variable_with_output_map():
    """K dimension should be read from output_map for Variable inputs."""
    model = _make_matmul_model(m=32, k=10, n=64, inp_b_constant=False)
    nodes = _get_nodes_by_op(model, "MatMul")
    output_map = {"B": np.zeros((10, 64))}
    assert _get_inp_b_k_dim(nodes[0], output_map=output_map) == 10


def test_get_inp_b_k_dim_returns_none_when_unknown():
    """Should return None if K cannot be determined."""
    model = _make_matmul_model(m=32, k=8, n=64, inp_b_constant=False)
    nodes = _get_nodes_by_op(model, "MatMul")
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
    nodes = _get_nodes_by_op(model, "MatMul")
    calibration_shapes = {"A": [m, k]}
    excluded = _exclude_matmuls_by_shape_inference(model, nodes, calibration_shapes)
    if expected_excluded:
        assert "MatMul_0" in excluded
    else:
        assert "MatMul_0" not in excluded


def test_matmul_gemv_excluded():
    """MatMul with N=1 (GEMV) should be excluded regardless of other dims."""
    model = _make_matmul_model(m=32, k=64, n=1)
    nodes = _get_nodes_by_op(model, "MatMul")
    calibration_shapes = {"A": [32, 64]}
    excluded = _exclude_matmuls_by_shape_inference(model, nodes, calibration_shapes)
    assert "MatMul_0" in excluded


def test_matmul_gemv_variable_b_excluded():
    """All-Variable MatMul with N=1 should be excluded via the all-Variable GEMV branch."""
    # inp_b_constant=False makes B a graph input (Variable) so the all-Variable path is taken.
    model = _make_matmul_model(m=32, k=64, n=1, inp_b_constant=False)
    nodes = _get_nodes_by_op(model, "MatMul")
    calibration_shapes = {"A": [32, 64], "B": [64, 1]}
    excluded = _exclude_matmuls_by_shape_inference(model, nodes, calibration_shapes)
    assert "MatMul_0" in excluded


def test_matmul_large_dims_not_excluded():
    """MatMul with all large dims should not be excluded."""
    model = _make_matmul_model(m=128, k=256, n=64)
    nodes = _get_nodes_by_op(model, "MatMul")
    calibration_shapes = {"A": [128, 256]}
    excluded = _exclude_matmuls_by_shape_inference(model, nodes, calibration_shapes)
    assert "MatMul_0" not in excluded


def _make_gemm_model(m, k, n, trans_b, name="Gemm_0"):
    """Build a minimal ONNX model with a single Gemm node and a constant B.

    If trans_b is 1, B has shape [N, K] (K is last axis).
    Otherwise B has shape [K, N].
    """
    inp_a = helper.make_tensor_value_info("A", TensorProto.FLOAT, [m, k])
    out = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [m, n])

    b_shape = [n, k] if trans_b else [k, n]
    b_init = helper.make_tensor(
        "B", TensorProto.FLOAT, b_shape, np.ones(b_shape[0] * b_shape[1]).tolist()
    )
    gemm = helper.make_node("Gemm", ["A", "B"], ["Y"], name=name, transB=trans_b)
    graph = helper.make_graph([gemm], "test", [inp_a], [out], initializer=[b_init])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    return model


@pytest.mark.parametrize("trans_b", [0, 1])
def test_get_inp_b_k_dim_gemm_transb_constant(trans_b):
    """Gemm should honor transB when deriving K from a Constant B."""
    model = _make_gemm_model(m=32, k=10, n=64, trans_b=trans_b)
    nodes = _get_nodes_by_op(model, "Gemm")
    assert _get_inp_b_k_dim(nodes[0]) == 10


@pytest.mark.parametrize("trans_b", [0, 1])
def test_get_inp_b_k_dim_gemm_transb_output_map(trans_b):
    """Gemm should honor transB when deriving K from an output_map."""
    # Build with a Variable B so the node's input is not a Constant.
    inp_a = helper.make_tensor_value_info("A", TensorProto.FLOAT, [32, 10])
    inp_b = helper.make_tensor_value_info("B", TensorProto.FLOAT, [64, 10] if trans_b else [10, 64])
    out = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [32, 64])
    gemm = helper.make_node("Gemm", ["A", "B"], ["Y"], name="Gemm_0", transB=trans_b)
    graph = helper.make_graph([gemm], "test", [inp_a, inp_b], [out])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    nodes = _get_nodes_by_op(model, "Gemm")

    b_runtime_shape = (64, 10) if trans_b else (10, 64)
    output_map = {"B": np.zeros(b_runtime_shape)}
    assert _get_inp_b_k_dim(nodes[0], output_map=output_map) == 10


def test_gemm_small_k_excluded_with_transb():
    """Gemm with transB=1 and small K should be excluded (regression: prior code read N)."""
    # N=64 is large; K=8 is small. With transB=1, B=[N,K]=[64,8], K axis is -1.
    # If _get_inp_b_k_dim ignored transB it would read 64 (N) and not exclude.
    model = _make_gemm_model(m=32, k=8, n=64, trans_b=1)
    nodes = _get_nodes_by_op(model, "Gemm")
    calibration_shapes = {"A": [32, 8]}
    excluded = _exclude_matmuls_by_shape_inference(model, nodes, calibration_shapes)
    assert "Gemm_0" in excluded


def test_gemm_large_dims_not_excluded_with_transb():
    """Gemm with transB=1 and all large dims should NOT be excluded."""
    model = _make_gemm_model(m=32, k=64, n=64, trans_b=1)
    nodes = _get_nodes_by_op(model, "Gemm")
    calibration_shapes = {"A": [32, 64]}
    excluded = _exclude_matmuls_by_shape_inference(model, nodes, calibration_shapes)
    assert "Gemm_0" not in excluded


def _make_matmul_model_graph_input_b(m, k, n, name="MatMul_0"):
    """MatMul where B is a graph input (its shape lives in model.graph.input only)."""
    inp_a = helper.make_tensor_value_info("A", TensorProto.FLOAT, [m, k])
    inp_b = helper.make_tensor_value_info("B", TensorProto.FLOAT, [k, n])
    out = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [m, n])
    matmul = helper.make_node("MatMul", ["A", "B"], ["Y"], name=name)
    graph = helper.make_graph([matmul], "test", [inp_a, inp_b], [out])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])


def test_matmul_small_k_graph_input_b_excluded():
    """Small-K MatMul whose B is a graph input should still be excluded.

    Regression: previous value_info_map only covered model.graph.value_info/output,
    missing graph inputs, so K was undetectable and the MatMul wasn't excluded.
    """
    model = _make_matmul_model_graph_input_b(m=32, k=8, n=64)
    nodes = _get_nodes_by_op(model, "MatMul")
    calibration_shapes = {"A": [32, 8], "B": [8, 64]}
    excluded = _exclude_matmuls_by_shape_inference(model, nodes, calibration_shapes)
    assert "MatMul_0" in excluded


@pytest.mark.parametrize(
    ("k", "n", "expected_excluded"),
    [
        (8, 64, True),
        (64, 8, True),
        (64, 64, False),
    ],
)
def test_exclude_matmuls_by_inference_runtime_path(k, n, expected_excluded):
    """Exercise the runtime-inference path with B as a graph input (read from output_map)."""
    m = 32
    model = _make_matmul_model_graph_input_b(m=m, k=k, n=n)
    nodes = _get_nodes_by_op(model, "MatMul")

    # Mock get_extended_model_outputs to return a synthetic output_map so we don't
    # need an actual ORT session.
    fake_output_map = {
        "Y": np.zeros((m, n), dtype=np.float32),
        "B": np.zeros((k, n), dtype=np.float32),
    }
    with mock.patch(
        "modelopt.onnx.quantization.graph_utils.get_extended_model_outputs",
        return_value=fake_output_map,
    ):
        excluded = _exclude_matmuls_by_inference(
            onnx_path="unused.onnx",
            model=model,
            matmul_nodes=nodes,
            use_external_data_format=False,
            intermediate_generated_files=[],
            calibration_data_reader=None,
            calibration_eps=["cpu"],
        )
    if expected_excluded:
        assert "MatMul_0" in excluded
    else:
        assert "MatMul_0" not in excluded


def test_exclude_matmuls_by_inference_gemv_variable_b():
    """All-Variable GEMV (N=1) should be excluded via the runtime-inference all-Variable path."""
    m, k, n = 32, 64, 1
    model = _make_matmul_model_graph_input_b(m=m, k=k, n=n)
    nodes = _get_nodes_by_op(model, "MatMul")
    fake_output_map = {
        "Y": np.zeros((m, n), dtype=np.float32),
        "B": np.zeros((k, n), dtype=np.float32),
    }
    with mock.patch(
        "modelopt.onnx.quantization.graph_utils.get_extended_model_outputs",
        return_value=fake_output_map,
    ):
        excluded = _exclude_matmuls_by_inference(
            onnx_path="unused.onnx",
            model=model,
            matmul_nodes=nodes,
            use_external_data_format=False,
            intermediate_generated_files=[],
            calibration_data_reader=None,
            calibration_eps=["cpu"],
        )
    assert "MatMul_0" in excluded


def test_exclude_matmuls_by_inference_gemv_constant_b():
    """Constant-B GEMV (N=1) should be excluded via the runtime-inference elif path."""
    m, k, n = 32, 64, 1
    model = _make_matmul_model(m=m, k=k, n=n, inp_b_constant=True)
    nodes = _get_nodes_by_op(model, "MatMul")
    # B is a Constant (initializer) so only the matmul output is added to graph outputs.
    fake_output_map = {"Y": np.zeros((m, n), dtype=np.float32)}
    with mock.patch(
        "modelopt.onnx.quantization.graph_utils.get_extended_model_outputs",
        return_value=fake_output_map,
    ):
        excluded = _exclude_matmuls_by_inference(
            onnx_path="unused.onnx",
            model=model,
            matmul_nodes=nodes,
            use_external_data_format=False,
            intermediate_generated_files=[],
            calibration_data_reader=None,
            calibration_eps=["cpu"],
        )
    assert "MatMul_0" in excluded


def test_exclude_matmuls_by_inference_dedupes_added_outputs():
    """Two MatMuls sharing the same Variable B must not create duplicate graph outputs."""
    # Build two MatMuls sharing B as a graph input.
    m, k, n = 32, 8, 64
    inp_a1 = helper.make_tensor_value_info("A1", TensorProto.FLOAT, [m, k])
    inp_a2 = helper.make_tensor_value_info("A2", TensorProto.FLOAT, [m, k])
    inp_b = helper.make_tensor_value_info("B", TensorProto.FLOAT, [k, n])
    out1 = helper.make_tensor_value_info("Y1", TensorProto.FLOAT, [m, n])
    out2 = helper.make_tensor_value_info("Y2", TensorProto.FLOAT, [m, n])
    mm1 = helper.make_node("MatMul", ["A1", "B"], ["Y1"], name="MatMul_0")
    mm2 = helper.make_node("MatMul", ["A2", "B"], ["Y2"], name="MatMul_1")
    graph = helper.make_graph([mm1, mm2], "test", [inp_a1, inp_a2, inp_b], [out1, out2])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    nodes = _get_nodes_by_op(model, "MatMul")

    fake_output_map = {
        "Y1": np.zeros((m, n), dtype=np.float32),
        "Y2": np.zeros((m, n), dtype=np.float32),
        "B": np.zeros((k, n), dtype=np.float32),
    }
    with mock.patch(
        "modelopt.onnx.quantization.graph_utils.get_extended_model_outputs",
        return_value=fake_output_map,
    ):
        excluded = _exclude_matmuls_by_inference(
            onnx_path="unused.onnx",
            model=model,
            matmul_nodes=nodes,
            use_external_data_format=False,
            intermediate_generated_files=[],
            calibration_data_reader=None,
            calibration_eps=["cpu"],
        )
    output_names = [o.name for o in model.graph.output]
    # B should appear only once in the graph outputs.
    assert output_names.count("B") == 1
    # Both MatMuls should be excluded (small K).
    assert "MatMul_0" in excluded
    assert "MatMul_1" in excluded
