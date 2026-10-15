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
# Unit tests for :mod:`modelopt.onnx.graph_surgery.dla_transforms.dla_decompose_lstm`.

"""ORT CPU parity tests for LSTM decomposition (forward / reverse / bidirectional).

Each parametrized run writes two ONNX files under
``scratch_space/lstm_decompose_test_models/`` (base + decomposed for that direction).
The decomposed file is passed through :func:`onnx.shape_inference.infer_shapes` before save
(when inference succeeds).
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from modelopt.onnx.graph_surgery.dla_transforms.dla_decompose_lstm import (
    _apply_decompose_lstm,
    _fold_lstm_constants,
)

pytest.importorskip("onnxruntime", reason="onnxruntime required for ORT parity checks")

_REPO_ROOT = Path(__file__).resolve().parents[5]
# Written on each parametrized run (six files total: three base + three decomposed).
_SAVED_MODELS_DIR = _REPO_ROOT / "scratch_space" / "lstm_decompose_test_models"


def _save_lstm_test_pair(
    base: onnx.ModelProto, transformed: onnx.ModelProto, direction: str
) -> None:
    _SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    base_path = _SAVED_MODELS_DIR / f"lstm_base_{direction}.onnx"
    dec_path = _SAVED_MODELS_DIR / f"lstm_decomposed_{direction}.onnx"
    onnx.save(base, str(base_path))
    try:
        from onnxruntime.tools.symbolic_shape_infer import (
            SymbolicShapeInference,  # type: ignore[import-untyped]
        )

        inferred = SymbolicShapeInference.infer_shapes(transformed, auto_merge_symbolic_dims=True)
        transformed_to_save = inferred if inferred is not None else transformed
    except Exception:
        transformed_to_save = transformed
    onnx.save(transformed_to_save, str(dec_path))
    print(
        f"[test_dla_decompose_lstm] saved {base_path.name} and {dec_path.name} under {_SAVED_MODELS_DIR}"
    )


@pytest.mark.parametrize(
    ("state", "kernel", "expected"),
    [([16777216, 1, -16777216], [1, 1, 1], 1), ([0, 0, 0], [-1, -1, -1], 0)],
)
def test_constant_initial_state_convolution_matches_cpp_double_accumulation(
    state, kernel, expected
):
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
    return numpy_helper.from_array(arr, name=name)


def _vi(name: str, elem_type: int, shape: list[int | str]) -> onnx.ValueInfoProto:
    return helper.make_tensor_value_info(name, elem_type, shape)


def _clamp_ir_version(model: onnx.ModelProto) -> None:
    model.ir_version = min(model.ir_version, _MAX_IR_VERSION_FOR_ORT)


def _lstm_input_shape_summary(model: onnx.ModelProto, input_name: str) -> str | tuple:
    """Return a printable shape for an LSTM input (initializer, graph input, or value_info)."""
    graph = model.graph
    for init in graph.initializer:
        if init.name == input_name:
            return tuple(numpy_helper.to_array(init).shape)
    for vi in list(graph.input) + list(graph.value_info):
        if vi.name != input_name:
            continue
        dims: list[int | str] = []
        for d in vi.type.tensor_type.shape.dim:
            if d.dim_value:
                dims.append(int(d.dim_value))
            elif d.dim_param:
                dims.append(d.dim_param)
            else:
                dims.append("?")
        return tuple(dims)
    return f"<unknown tensor {input_name!r}>"


def _print_lstm_input_shapes(model: onnx.ModelProto, *, direction: str) -> None:
    graph = model.graph
    lstm_nodes = [n for n in graph.node if n.op_type == "LSTM"]
    if not lstm_nodes:
        print(f"[test_dla_decompose_lstm] direction={direction}: no LSTM node in graph")
        return
    node = lstm_nodes[0]
    print(f"[test_dla_decompose_lstm] direction={direction} LSTM name={node.name!r} input slots:")
    for idx, inp in enumerate(node.input):
        if not inp:
            print(f"  [{idx}] (empty optional)")
            continue
        sh = _lstm_input_shape_summary(model, inp)
        print(f"  [{idx}] {inp!r} shape={sh}")


def _make_lstm_model(direction: str) -> onnx.ModelProto:
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
    inputs = [_vi("X", TensorProto.FLOAT, [_T, _N, _D_IN])]
    outputs = [
        _vi("Y", TensorProto.FLOAT, [_T, num_dir, _N, _H]),
        _vi("Y_h", TensorProto.FLOAT, [num_dir, _N, _H]),
        _vi("Y_c", TensorProto.FLOAT, [num_dir, _N, _H]),
    ]
    graph = helper.make_graph(nodes, f"lstm_{direction}", inputs, outputs, initializer=inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", _OPSET_VERSION)])
    _clamp_ir_version(model)
    onnx.checker.check_model(model)
    return model


def _single_lstm_node(model: onnx.ModelProto) -> onnx.NodeProto:
    return next(n for n in model.graph.node if n.op_type == "LSTM")


def test_downstream_transpose_keeps_its_nontrivial_permutation():
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
    from modelopt.onnx.graph_surgery.utils.graph_utils import infer_shapes

    model = infer_shapes(copy.deepcopy(model))
    with pytest.raises(ValueError, match=expected):
        _apply_decompose_lstm(model)


def _run_ort(model: onnx.ModelProto, feeds: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    import onnxruntime as ort  # type: ignore[import-not-found]

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
    assert ref.keys() == got.keys()
    for name in ref:
        a, b = ref[name], got[name]
        assert a.shape == b.shape, f"{name}: shape {a.shape} vs {b.shape}"
        cos = _cosine_similarity(a, b)
        assert cos >= 1.0 - 1e-5, f"{name}: cosine similarity {cos} expected ~1.0"
        np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-4, err_msg=f"{name}: value mismatch")


@pytest.mark.parametrize("direction", ["forward", "reverse", "bidirectional"])
def test_lstm_decompose_ort_cpu_parity(direction: str) -> None:
    """Decomposed graph matches reference LSTM on CPU EP (cosine ~1, tight allclose)."""
    model = _make_lstm_model(direction)
    _print_lstm_input_shapes(model, direction=direction)
    rng = np.random.default_rng(7)
    x = rng.standard_normal((_T, _N, _D_IN)).astype(np.float32)
    feeds = {"X": x}

    ref = _run_ort(model, feeds)

    dec = copy.deepcopy(model)
    from modelopt.onnx.graph_surgery.utils.graph_utils import infer_shapes

    dec = infer_shapes(dec)
    dec = _apply_decompose_lstm(dec)
    _clamp_ir_version(dec)
    onnx.checker.check_model(dec)

    got = _run_ort(dec, feeds)
    _save_lstm_test_pair(model, dec, direction)
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
