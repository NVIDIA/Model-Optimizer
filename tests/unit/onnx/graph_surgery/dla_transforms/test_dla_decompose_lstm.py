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

# Unit tests for :mod:`modelopt.onnx.graph_surgery.dla_transforms.dla_decompose_lstm`.

"""ORT CPU parity tests for LSTM decomposition (forward / reverse / bidirectional)."""

from __future__ import annotations

import copy

import numpy as np
import onnx
import pytest
from onnx import AttributeProto, TensorProto, helper, numpy_helper
from onnx.compose import add_prefix

from modelopt.onnx.graph_surgery.dla_transforms.dla_decompose_lstm import (
    _apply_decompose_lstm,
    _fold_lstm_constants,
)
from modelopt.onnx.graph_surgery.utils.graph_utils import infer_shapes

ort = pytest.importorskip("onnxruntime", reason="onnxruntime required for ORT parity checks")


@pytest.mark.parametrize(
    ("state", "kernel", "expected"),
    [([16777216, 1, -16777216], [1, 1, 1], 1), ([0, 0, 0], [-1, -1, -1], 0)],
)
def test_constant_initial_state_convolution_matches_cpp_double_accumulation(
    state, kernel, expected
):
    """Constant initial state convolution matches cpp double accumulation."""
    values = np.array(state, dtype=np.float32).reshape(1, 3, 1, 1)
    weights = np.array(kernel, dtype=np.float32).reshape(1, 3, 1, 1)
    graph = helper.make_graph(
        [],
        "constant_state",
        [],
        [],
        initializer=[
            numpy_helper.from_array(values, "state"),
            numpy_helper.from_array(weights, "weights"),
        ],
    )
    node = helper.make_node("Conv", ["state", "weights"], ["folded"])
    assert _fold_lstm_constants(graph, [node], {"state", "weights"}) == []
    tensors = {tensor.name: numpy_helper.to_array(tensor) for tensor in graph.initializer}
    assert tensors["folded"].tobytes() == np.array([expected], dtype=np.float32).tobytes()
    np.testing.assert_array_equal(tensors["state"], values)


# Some onnxruntime builds reject very new ONNX IR versions from recent onnx packages.
_MAX_IR_VERSION_FOR_ORT = 9
_OPSET_VERSION = 17

_T = 4
_N = 5
_D_IN = 3
_H = 16


def _init(name: str, arr: np.ndarray) -> onnx.TensorProto:
    """Init."""
    return numpy_helper.from_array(arr, name=name)


def _vi(name: str, elem_type: int, shape: list[int | str]) -> onnx.ValueInfoProto:
    """Vi."""
    return helper.make_tensor_value_info(name, elem_type, shape)


def _clamp_ir_version(model: onnx.ModelProto) -> None:
    """Clamp ir version."""
    model.ir_version = min(model.ir_version, _MAX_IR_VERSION_FOR_ORT)


def _make_lstm_model(direction: str, batch: int = _N) -> onnx.ModelProto:
    """Single-LSTM graph: X input, W/R/B initializers, Y + Y_h + Y_c outputs."""
    assert direction in ("forward", "reverse", "bidirectional")
    num_dir = 2 if direction == "bidirectional" else 1

    seeds = {"forward": 42, "reverse": 43, "bidirectional": 44}
    rng = np.random.default_rng(seeds[direction])

    w = rng.standard_normal((num_dir, 4 * _H, _D_IN)).astype(np.float32)
    r = rng.standard_normal((num_dir, 4 * _H, _H)).astype(np.float32)
    b = rng.standard_normal((num_dir, 8 * _H)).astype(np.float32)

    inits = [_init("W", w), _init("R", r), _init("B", b)]
    nodes = [
        helper.make_node(
            "LSTM",
            ["X", "W", "R", "B"],
            ["Y", "Y_h", "Y_c"],
            name="lstm0",
            hidden_size=_H,
            direction=direction,
        )
    ]
    inputs = [_vi("X", TensorProto.FLOAT, [_T, batch, _D_IN])]
    outputs = [
        _vi("Y", TensorProto.FLOAT, [_T, num_dir, batch, _H]),
        _vi("Y_h", TensorProto.FLOAT, [num_dir, batch, _H]),
        _vi("Y_c", TensorProto.FLOAT, [num_dir, batch, _H]),
    ]
    graph = helper.make_graph(nodes, f"lstm_{direction}", inputs, outputs, initializer=inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", _OPSET_VERSION)])
    _clamp_ir_version(model)
    onnx.checker.check_model(model)
    return model


def _single_lstm_node(model: onnx.ModelProto) -> onnx.NodeProto:
    """Single lstm node."""
    return next(n for n in model.graph.node if n.op_type == "LSTM")


def test_downstream_transpose_keeps_its_nontrivial_permutation():
    """Downstream transpose keeps its nontrivial permutation."""
    model = _make_lstm_model("bidirectional")
    model.graph.node.append(
        helper.make_node("Transpose", ["Y"], ["permuted"], perm=[0, 2, 1, 3], name="downstream")
    )
    model.graph.output[0].CopyFrom(_vi("permuted", TensorProto.FLOAT, [_T, _N, 2, _H]))
    feeds = {"X": np.random.default_rng(8).standard_normal((_T, _N, _D_IN)).astype(np.float32)}
    expected = _run_ort(model, feeds)
    candidate = onnx.shape_inference.infer_shapes(copy.deepcopy(model))
    _apply_decompose_lstm(candidate)
    actual = _run_ort(candidate, feeds)
    for name in expected:
        np.testing.assert_allclose(actual[name], expected[name], rtol=1e-5, atol=1e-5)


def _expect_unsupported_lstm(model: onnx.ModelProto, expected: str) -> None:
    """Expect unsupported lstm."""
    model = infer_shapes(copy.deepcopy(model))
    with pytest.raises(ValueError, match=expected):
        _apply_decompose_lstm(model)


def _run_ort(model: onnx.ModelProto, feeds: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Run ort."""
    _clamp_ir_version(model)
    sess = ort.InferenceSession(
        model.SerializeToString(),
        providers=["CPUExecutionProvider"],
    )
    out: dict[str, np.ndarray] = {}
    for o in model.graph.output:
        out[o.name] = sess.run([o.name], feeds)[0]
    return out


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity."""
    a64 = np.asarray(a, dtype=np.float64).ravel()
    b64 = np.asarray(b, dtype=np.float64).ravel()
    na = float(np.linalg.norm(a64))
    nb = float(np.linalg.norm(b64))
    if na == 0.0 and nb == 0.0:
        return 1.0
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a64, b64) / (na * nb))


def _assert_outputs_match(ref: dict[str, np.ndarray], got: dict[str, np.ndarray]) -> None:
    """Assert outputs match."""
    assert ref.keys() == got.keys()
    for name in ref:
        a, b = ref[name], got[name]
        assert a.shape == b.shape, f"{name}: shape {a.shape} vs {b.shape}"
        cos = _cosine_similarity(a, b)
        assert cos >= 1.0 - 1e-5, f"{name}: cosine similarity {cos} expected ~1.0"
        np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-4, err_msg=f"{name}: value mismatch")


@pytest.mark.parametrize("batch", [1, _N])
@pytest.mark.parametrize("direction", ["forward", "reverse", "bidirectional"])
def test_lstm_decompose_ort_cpu_parity(direction: str, batch: int) -> None:
    """Decomposed graph matches reference LSTM on CPU EP (cosine ~1, tight allclose)."""
    model = _make_lstm_model(direction, batch)
    rng = np.random.default_rng(7)
    x = rng.standard_normal((_T, batch, _D_IN)).astype(np.float32)
    feeds = {"X": x}

    ref = _run_ort(model, feeds)

    dec = copy.deepcopy(model)
    dec = infer_shapes(dec)
    dec = _apply_decompose_lstm(dec)
    _clamp_ir_version(dec)
    onnx.checker.check_model(dec)

    got = _run_ort(dec, feeds)
    _assert_outputs_match(ref, got)


@pytest.mark.parametrize(
    ("attr_name", "attr_value", "expected"),
    [
        ("layout", 1, "layout=1"),
        ("clip", 0.5, "clip=0.5"),
        ("input_forget", 1, "input_forget=1"),
    ],
)
def test_lstm_decompose_rejects_unsupported_scalar_attrs(attr_name, attr_value, expected) -> None:
    """Unsupported scalar LSTM attributes must fail before decomposition."""
    model = _make_lstm_model("forward")
    node = _single_lstm_node(model)
    if isinstance(attr_value, float):
        node.attribute.append(helper.make_attribute(attr_name, attr_value))
    else:
        node.attribute.append(helper.make_attribute(attr_name, int(attr_value)))

    _expect_unsupported_lstm(model, expected)


def test_lstm_decompose_rejects_peephole_input() -> None:
    """Peephole input P is not represented by this decomposition."""
    model = _make_lstm_model("forward")
    node = _single_lstm_node(model)
    p = np.zeros((1, 3 * _H), dtype=np.float32)
    model.graph.initializer.append(_init("P", p))
    del node.input[:]
    node.input.extend(["X", "W", "R", "B", "", "", "", "P"])

    _expect_unsupported_lstm(model, "peephole P")


def test_lstm_decompose_rejects_sequence_lens_input() -> None:
    """sequence_lens changes sequence semantics and is not handled here."""
    model = _make_lstm_model("forward")
    node = _single_lstm_node(model)
    seq_lens = np.array([_T] * _N, dtype=np.int32)
    model.graph.initializer.append(_init("seq_lens", seq_lens))
    del node.input[:]
    node.input.extend(["X", "W", "R", "B", "seq_lens"])

    _expect_unsupported_lstm(model, "sequence_lens")


def test_lstm_decompose_rejects_custom_activations() -> None:
    """Only the default Sigmoid/Tanh/Tanh LSTM activation set is supported."""
    model = _make_lstm_model("forward")
    node = _single_lstm_node(model)
    node.attribute.append(helper.make_attribute("activations", ["Relu", "Tanh", "Tanh"]))

    _expect_unsupported_lstm(model, "activations")


def test_lstm_decompose_rejects_activation_params() -> None:
    """Custom activation parameters imply non-default activation semantics."""
    model = _make_lstm_model("forward")
    node = _single_lstm_node(model)
    node.attribute.append(helper.make_attribute("activation_alpha", [0.1]))

    _expect_unsupported_lstm(model, "activation_alpha")


@pytest.mark.parametrize("direction", ["forward", "reverse", "bidirectional"])
def test_optional_bias_omission_preserves_all_outputs(direction):
    """Execute all three directions without an optional bias tensor."""
    model = _make_lstm_model(direction)
    del model.graph.node[0].input[3:]
    model.graph.initializer.remove(next(t for t in model.graph.initializer if t.name == "B"))
    feeds = {"X": np.random.default_rng(2).standard_normal((_T, _N, _D_IN)).astype(np.float32)}
    expected = _run_ort(model, feeds)
    transformed = _apply_decompose_lstm(onnx.shape_inference.infer_shapes(copy.deepcopy(model)))
    onnx.checker.check_model(transformed, full_check=True)
    _assert_outputs_match(expected, _run_ort(transformed, feeds))


@pytest.mark.parametrize("name", ["W", "R", "B"])
@pytest.mark.parametrize("axis", [None, 1, -1])
def test_shared_dq_weight_preserves_source_layout_and_axes(name, axis):
    """Preserve a public quantized source and DQ view while decomposing its LSTM use."""
    model = _make_lstm_model("forward")
    weight = next(t for t in model.graph.initializer if t.name == name)
    values = numpy_helper.to_array(weight)
    quantized = np.clip(np.rint(values * 16), -128, 127).astype(np.int8)
    dims = list(quantized.shape)
    weight.CopyFrom(_init(name + "_q", quantized))
    controls = np.float32(0.0625) if axis is None else np.full(dims[axis], 0.0625, np.float32)
    zeros = np.int8(0) if axis is None else np.zeros(dims[axis], np.int8)
    model.graph.initializer.extend([_init(name + "_scale", controls), _init(name + "_zero", zeros)])
    dq = helper.make_node(
        "DequantizeLinear", [name + "_q", name + "_scale", name + "_zero"], [name], name="weight_dq"
    )
    if axis is not None:
        dq.attribute.append(helper.make_attribute("axis", axis))
    model.graph.node.insert(0, dq)
    model.graph.output.extend(
        [_vi(name + "_q", TensorProto.INT8, dims), _vi(name, TensorProto.FLOAT, dims)]
    )
    feeds = {"X": np.random.default_rng(3).standard_normal((_T, _N, _D_IN)).astype(np.float32)}
    expected = _run_ort(model, feeds)
    transformed = _apply_decompose_lstm(onnx.shape_inference.infer_shapes(copy.deepcopy(model)))
    onnx.checker.check_model(transformed, full_check=True)
    _assert_outputs_match(expected, _run_ort(transformed, feeds))


@pytest.mark.parametrize("name", ["W", "R", "B"])
def test_public_fixed_lstm_inputs_keep_their_original_view(name):
    """Retain fixed weights or bias that are also graph outputs."""
    model = _make_lstm_model("forward")
    weight = next(t for t in model.graph.initializer if t.name == name)
    model.graph.output.append(_vi(name, TensorProto.FLOAT, list(weight.dims)))
    feeds = {"X": np.random.default_rng(4).standard_normal((_T, _N, _D_IN)).astype(np.float32)}
    expected = _run_ort(model, feeds)
    transformed = _apply_decompose_lstm(onnx.shape_inference.infer_shapes(copy.deepcopy(model)))
    onnx.checker.check_model(transformed, full_check=True)
    _assert_outputs_match(expected, _run_ort(transformed, feeds))


@pytest.mark.parametrize("name", ["W", "R", "B"])
def test_overridable_lstm_inputs_use_runtime_values(name):
    """Preserve graph-input initializer defaults as runtime LSTM controls."""
    model = _make_lstm_model("forward")
    weight = next(t for t in model.graph.initializer if t.name == name)
    model.graph.input.append(_vi(name, TensorProto.FLOAT, list(weight.dims)))
    feeds = {
        "X": np.random.default_rng(5).standard_normal((_T, _N, _D_IN)).astype(np.float32),
        name: -numpy_helper.to_array(weight),
    }
    expected = _run_ort(model, feeds)
    transformed = _apply_decompose_lstm(onnx.shape_inference.infer_shapes(copy.deepcopy(model)))
    onnx.checker.check_model(transformed, full_check=True)
    _assert_outputs_match(expected, _run_ort(transformed, feeds))


@pytest.mark.parametrize("outputs", [["", "Y_h", "Y_c"], ["Y"], ["", "Y_h"]])
def test_optional_lstm_outputs_remain_valid(outputs):
    """Support omitted sequence and state output slots without introducing empty names."""
    model = _make_lstm_model("forward")
    del model.graph.node[0].output[:]
    model.graph.node[0].output.extend(outputs)
    retained = [value for value in model.graph.output if value.name in outputs]
    del model.graph.output[:]
    model.graph.output.extend(retained)
    feeds = {"X": np.random.default_rng(6).standard_normal((_T, _N, _D_IN)).astype(np.float32)}
    expected = _run_ort(model, feeds)
    transformed = _apply_decompose_lstm(onnx.shape_inference.infer_shapes(copy.deepcopy(model)))
    onnx.checker.check_model(transformed, full_check=True)
    _assert_outputs_match(expected, _run_ort(transformed, feeds))


def test_explicit_zero_clip_refuses_before_mutation():
    """Reject a zero clipping threshold rather than interpreting it as omitted clipping."""
    model = _make_lstm_model("forward")
    model.graph.node[0].attribute.append(helper.make_attribute("clip", 0.0))
    onnx.checker.check_model(model, full_check=True)
    before = model.SerializeToString()
    with pytest.raises(ValueError, match="clip"):
        _apply_decompose_lstm(model)
    assert model.SerializeToString() == before


def test_custom_lstm_function_is_preserved():
    """Leave a model-local operation whose name happens to be LSTM unchanged."""
    function = helper.make_function(
        "custom",
        "LSTM",
        ["X"],
        ["Y"],
        [helper.make_node("Neg", ["X"], ["Y"])],
        [helper.make_opsetid("", 17)],
    )
    model = helper.make_model(
        helper.make_graph(
            [helper.make_node("LSTM", ["X"], ["Y"], domain="custom")],
            "local_lstm",
            [_vi("X", TensorProto.FLOAT, [2, 3])],
            [_vi("Y", TensorProto.FLOAT, [2, 3])],
        ),
        functions=[function],
        opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("custom", 1)],
    )
    _clamp_ir_version(model)
    onnx.checker.check_model(model, full_check=True)
    before = model.SerializeToString()
    _apply_decompose_lstm(model)
    assert model.SerializeToString() == before


@pytest.mark.parametrize("kind", ["node", "initializer"])
def test_generated_names_preserve_existing_public_values(kind):
    """Allocate private names without overwriting an unrelated public graph value."""
    model = _make_lstm_model("forward")
    name = "forward_conv_x_w_lstm0" if kind == "node" else "split_z_iofc_init_lstm0"
    if kind == "node":
        model.graph.node.insert(0, helper.make_node("Identity", ["X"], [name], name="existing"))
        model.graph.output.append(_vi(name, TensorProto.FLOAT, [_T, _N, _D_IN]))
    else:
        model.graph.initializer.append(_init(name, np.asarray([97], np.int64)))
        model.graph.output.append(_vi(name, TensorProto.INT64, [1]))
    feeds = {"X": np.random.default_rng(9).standard_normal((_T, _N, _D_IN)).astype(np.float32)}
    expected = _run_ort(model, feeds)
    transformed = _apply_decompose_lstm(onnx.shape_inference.infer_shapes(copy.deepcopy(model)))
    onnx.checker.check_model(transformed, full_check=True)
    _assert_outputs_match(expected, _run_ort(transformed, feeds))


def test_legacy_ir_registers_all_new_lstm_initializers():
    """Keep the legacy initializer-input contract after standalone decomposition."""
    model = _make_lstm_model("forward")
    model.ir_version = 3
    for tensor in model.graph.initializer:
        model.graph.input.append(_vi(tensor.name, tensor.data_type, list(tensor.dims)))
    onnx.checker.check_model(model, full_check=True)
    feeds = {"X": np.random.default_rng(10).standard_normal((_T, _N, _D_IN)).astype(np.float32)}
    expected = _run_ort(model, feeds)
    transformed = _apply_decompose_lstm(onnx.shape_inference.infer_shapes(copy.deepcopy(model)))
    onnx.checker.check_model(transformed, full_check=True)
    _assert_outputs_match(expected, _run_ort(transformed, feeds))


def test_buffered_constant_state_convolution_keeps_sequential_rounding():
    """Retain established double accumulation across multiple bounded row buffers."""
    rng = np.random.default_rng(11)
    weights = rng.standard_normal((513, 1024)).astype(np.float32)
    state = rng.standard_normal(1024).astype(np.float32)
    terms = np.concatenate(
        [np.zeros((513, 1)), weights.astype(np.float64) * state.astype(np.float64)], axis=1
    )
    expected = np.cumsum(terms, axis=1)[:, -1].astype(np.float32)
    graph = helper.make_graph(
        [],
        "buffered_state",
        [],
        [],
        initializer=[
            _init("state", state.reshape(1, 1024, 1, 1)),
            _init("weights", weights.reshape(513, 1024, 1, 1)),
        ],
    )
    assert (
        _fold_lstm_constants(
            graph,
            [helper.make_node("Conv", ["state", "weights"], ["folded"])],
            {"state", "weights"},
        )
        == []
    )
    actual = numpy_helper.to_array(next(t for t in graph.initializer if t.name == "folded"))
    assert actual.tobytes() == expected.tobytes()


@pytest.mark.parametrize("role", ["X", "W", "R", "B"])
def test_dequantized_inputs_without_value_info_preserve_numerics(role):
    """Resolve the native LSTM contract without relying on a prior inference pass."""
    model = _make_lstm_model("forward")
    rng = np.random.default_rng(12)
    values = (
        rng.standard_normal((_T, _N, _D_IN)).astype(np.float32)
        if role == "X"
        else numpy_helper.to_array(next(t for t in model.graph.initializer if t.name == role))
    )
    quantized = np.clip(np.rint(values * 16), -128, 127).astype(np.int8)
    feeds = {"X": rng.standard_normal((_T, _N, _D_IN)).astype(np.float32)}
    if role == "X":
        model.graph.input[0].CopyFrom(_vi("X_q", TensorProto.INT8, list(quantized.shape)))
        feeds = {"X_q": quantized}
    else:
        weight = next(t for t in model.graph.initializer if t.name == role)
        weight.CopyFrom(_init(role + "_q", quantized))
    model.graph.initializer.extend(
        [_init(role + "_scale", np.float32(0.0625)), _init(role + "_zero", np.int8(0))]
    )
    model.graph.node.insert(
        0,
        helper.make_node(
            "DequantizeLinear", [role + "_q", role + "_scale", role + "_zero"], [role]
        ),
    )
    onnx.checker.check_model(model, full_check=True)
    expected = _run_ort(model, feeds)
    transformed = _apply_decompose_lstm(copy.deepcopy(model))
    onnx.checker.check_model(transformed, full_check=True)
    _assert_outputs_match(expected, _run_ort(transformed, feeds))


@pytest.mark.parametrize("constant_axes", [False, True])
def test_unsqueezed_weight_without_value_info_reads_existing_axes(constant_axes):
    """Look up existing axes by their actual initializer or Constant tensor name."""
    model = _make_lstm_model("forward")
    weight = next(t for t in model.graph.initializer if t.name == "W")
    weight.CopyFrom(_init("W_flat", numpy_helper.to_array(weight).reshape(4 * _H, _D_IN)))
    axes = _init("weight_axes", np.asarray([0], np.int64))
    if constant_axes:
        model.graph.node.insert(0, helper.make_node("Constant", [], ["weight_axes"], value=axes))
    else:
        model.graph.initializer.append(axes)
    model.graph.node.insert(
        int(constant_axes), helper.make_node("Unsqueeze", ["W_flat", "weight_axes"], ["W"])
    )
    feeds = {"X": np.random.default_rng(13).standard_normal((_T, _N, _D_IN)).astype(np.float32)}
    onnx.checker.check_model(model, full_check=True)
    expected = _run_ort(model, feeds)
    transformed = _apply_decompose_lstm(copy.deepcopy(model))
    onnx.checker.check_model(transformed, full_check=True)
    _assert_outputs_match(expected, _run_ort(transformed, feeds))


@pytest.mark.parametrize("axes", [[0, 1], [-3, -2]])
def test_unsqueezed_activation_without_value_info_handles_multiple_axes(axes):
    """Resolve positive and negative multiple axes against the Unsqueeze output rank."""
    model = _make_lstm_model("forward")
    model.graph.input[0].CopyFrom(_vi("X_flat", TensorProto.FLOAT, [_D_IN]))
    model.graph.initializer.append(_init("activation_axes", np.asarray(axes, np.int64)))
    model.graph.node.insert(0, helper.make_node("Unsqueeze", ["X_flat", "activation_axes"], ["X"]))
    model.graph.output[0].CopyFrom(_vi("Y", TensorProto.FLOAT, [1, 1, 1, _H]))
    for output in model.graph.output[1:]:
        output.CopyFrom(_vi(output.name, TensorProto.FLOAT, [1, 1, _H]))
    feeds = {"X_flat": np.random.default_rng(14).standard_normal(_D_IN).astype(np.float32)}
    onnx.checker.check_model(model, full_check=True)
    expected = _run_ort(model, feeds)
    transformed = _apply_decompose_lstm(copy.deepcopy(model))
    onnx.checker.check_model(transformed, full_check=True)
    _assert_outputs_match(expected, _run_ort(transformed, feeds))


@pytest.mark.parametrize(
    ("role", "index"), [("X", 0), ("W", 1), ("R", 2), ("B", 3), ("initial_h", 5), ("initial_c", 6)]
)
def test_runtime_input_transpose_preserves_existing_tensor_names(role, index):
    """Allocate runtime transpose outputs without redefining an unrelated public tensor."""
    model = _make_lstm_model("forward")
    rng = np.random.default_rng(16)
    feeds = {"X": rng.standard_normal((_T, _N, _D_IN)).astype(np.float32)}
    if role in {"W", "R", "B"}:
        tensor = next(tensor for tensor in model.graph.initializer if tensor.name == role)
        shape = list(tensor.dims)
        model.graph.input.append(_vi(role, TensorProto.FLOAT, shape))
        feeds[role] = -numpy_helper.to_array(tensor)
    elif role in {"initial_h", "initial_c"}:
        shape = [1, _N, _H]
        model.graph.input.append(_vi(role, TensorProto.FLOAT, shape))
        node = model.graph.node[0]
        while len(node.input) <= index:
            node.input.append("")
        node.input[index] = role
        feeds[role] = rng.standard_normal(shape).astype(np.float32)
    else:
        shape = [_T, _N, _D_IN]
    existing = f"{role}_lstm4d_lstm0_{index}_transposed"
    model.graph.node.insert(
        0, helper.make_node("Identity", [role], [existing], name="existing_view")
    )
    model.graph.output.append(_vi(existing, TensorProto.FLOAT, shape))
    onnx.checker.check_model(model, full_check=True)
    expected = _run_ort(model, feeds)
    transformed = _apply_decompose_lstm(onnx.shape_inference.infer_shapes(copy.deepcopy(model)))
    onnx.checker.check_model(transformed, full_check=True)
    _assert_outputs_match(expected, _run_ort(transformed, feeds))


def _make_lstm_followed_by_copied_reshape(target, view="private"):
    """Create a valid unary-dimension transpose followed by a potentially shared target."""
    model = _make_lstm_model("forward", 2)
    model.graph.initializer.append(_init("shape_target", np.asarray(target, np.int64)))
    model.graph.node.extend(
        [
            helper.make_node(
                "Transpose", ["Y"], ["layout"], perm=[0, 2, 1, 3], name="layout_change"
            ),
            helper.make_node(
                "Reshape", ["layout", "shape_target"], ["reshaped"], name="restore_shape"
            ),
        ]
    )
    model.graph.output[0].CopyFrom(_vi("reshaped", TensorProto.FLOAT, None))
    feeds = {"X": np.random.default_rng(17).standard_normal((_T, 2, _D_IN)).astype(np.float32)}
    if view == "public":
        model.graph.output.append(_vi("shape_target", TensorProto.INT64, [3]))
    elif view == "overridable":
        model.graph.input.append(_vi("shape_target", TensorProto.INT64, [3]))
    elif view == "captured":
        model.graph.input.append(_vi("condition", TensorProto.BOOL, []))
        feeds["condition"] = np.array(True)
        branch = helper.make_graph(
            [helper.make_node("Identity", ["layout"], ["branch_output"])],
            "layout_reader",
            [],
            [_vi("branch_output", TensorProto.FLOAT, [_T, 2, 1, _H])],
        )
        model.graph.node.append(
            helper.make_node(
                "If", ["condition"], ["captured_output"], then_branch=branch, else_branch=branch
            )
        )
        model.graph.output.append(_vi("captured_output", TensorProto.FLOAT, [_T, 2, 1, _H]))
    elif view == "shared":
        model.graph.input.append(_vi("other", TensorProto.FLOAT, [_T, 3, _H]))
        feeds["other"] = np.arange(_T * 3 * _H, dtype=np.float32).reshape(_T, 3, _H)
        model.graph.node.append(
            helper.make_node("Reshape", ["other", "shape_target"], ["other_out"])
        )
        model.graph.output.append(_vi("other_out", TensorProto.FLOAT, [_T, 3, _H]))
    model = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(model, full_check=True)
    return model, feeds


@pytest.mark.parametrize("target", [[0, 0, _H], [0, -1, 0], [-1, 0, 0], [_T, 2, _H]])
@pytest.mark.parametrize("defer", [False, True])
def test_deferred_transpose_keeps_copied_reshape_dimensions(target, defer):
    """Preserve copied dimensions when moving unary axes through a deferred reshape."""
    model, feeds = _make_lstm_followed_by_copied_reshape(target)
    expected = _run_ort(model, feeds)
    transformed = _apply_decompose_lstm(copy.deepcopy(model), defer_rank_conversion=defer)
    onnx.checker.check_model(transformed, full_check=True)
    _assert_outputs_match(expected, _run_ort(transformed, feeds))


@pytest.mark.parametrize("view", ["public", "shared", "overridable", "captured"])
def test_deferred_copied_reshape_preserves_target_owners(view):
    """Preserve public, shared and runtime structural controls while changing a private layout."""
    model, feeds = _make_lstm_followed_by_copied_reshape([0, 0, _H], view)
    expected = _run_ort(model, feeds)
    transformed = _apply_decompose_lstm(copy.deepcopy(model), defer_rank_conversion=True)
    onnx.checker.check_model(transformed, full_check=True)
    _assert_outputs_match(expected, _run_ort(transformed, feeds))


@pytest.mark.parametrize("operator", ["Transpose", "Reshape"])
def test_deferred_layout_gate_preserves_model_local_followers(operator):
    """Preserve opaque follower functions whose names match standard layout operators."""
    model, feeds = _make_lstm_followed_by_copied_reshape([_T, 2, _H])
    node = next(node for node in model.graph.node if node.op_type == operator)
    node.domain = "custom"
    if operator == "Transpose":
        inner = helper.make_node("Transpose", ["X"], ["moved"])
        inner.attribute.append(
            AttributeProto(name="perm", type=AttributeProto.INTS, ref_attr_name="perm")
        )
        function = helper.make_function(
            "custom",
            operator,
            ["X"],
            ["Y"],
            [inner, helper.make_node("Softmax", ["moved"], ["Y"], axis=1)],
            [helper.make_opsetid("", 17)],
            attributes=["perm"],
        )
    else:
        function = helper.make_function(
            "custom",
            operator,
            ["X", "shape"],
            ["Y"],
            [
                helper.make_node("Softmax", ["X"], ["probabilities"], axis=1),
                helper.make_node("Reshape", ["probabilities", "shape"], ["Y"]),
            ],
            [helper.make_opsetid("", 17)],
        )
    model.functions.append(function)
    model.opset_import.append(helper.make_opsetid("custom", 1))
    onnx.checker.check_model(model, full_check=True)
    expected = _run_ort(model, feeds)
    transformed = _apply_decompose_lstm(copy.deepcopy(model), defer_rank_conversion=True)
    onnx.checker.check_model(transformed, full_check=True)
    _assert_outputs_match(expected, _run_ort(transformed, feeds))


@pytest.mark.parametrize(
    ("direction", "actual_directions"), [("forward", 2), ("reverse", 2), ("bidirectional", 1)]
)
@pytest.mark.parametrize("late", [False, True])
def test_direction_count_refusal_precedes_all_lstm_mutation(direction, actual_directions, late):
    """Refuse a direction-count mismatch before rewriting this or an earlier LSTM."""
    model = _make_lstm_model("bidirectional" if actual_directions == 2 else "forward")
    node = model.graph.node[0]
    next(
        attribute for attribute in node.attribute if attribute.name == "direction"
    ).s = direction.encode()
    if late:
        first = add_prefix(_make_lstm_model("forward"), "first/")
        model.graph.node.insert(0, first.graph.node[0])
        model.graph.initializer.extend(first.graph.initializer)
        model.graph.input.extend(first.graph.input)
        model.graph.output.extend(first.graph.output)
    onnx.checker.check_model(model)
    before = model.SerializeToString()
    with pytest.raises(ValueError, match="num_directions"):
        _apply_decompose_lstm(model)
    assert model.SerializeToString() == before
