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
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

pytest.importorskip("onnxruntime", reason="onnxruntime required for ORT parity checks")

_REPO_ROOT = Path(__file__).resolve().parents[3]
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


def _load_apply_decompose_lstm():
    """Import LSTM pass without loading ``graph_surgery/__init__.py`` (onnxscript, registry)."""
    if "modelopt.onnx.graph_surgery.dla_transforms.dla_decompose_lstm" in sys.modules:
        return sys.modules[
            "modelopt.onnx.graph_surgery.dla_transforms.dla_decompose_lstm"
        ]._apply_decompose_lstm

    import modelopt.onnx  # noqa: F401 — package root; quantization may warn if optional deps skew

    gs = "modelopt.onnx.graph_surgery"
    dt = "modelopt.onnx.graph_surgery.dla_transforms"
    ut = "modelopt.onnx.graph_surgery.utils"

    if gs not in sys.modules:
        m = types.ModuleType(gs)
        m.__path__ = [str(_REPO_ROOT / "modelopt/onnx/graph_surgery")]
        sys.modules[gs] = m
    if dt not in sys.modules:
        m = types.ModuleType(dt)
        m.__path__ = [str(_REPO_ROOT / "modelopt/onnx/graph_surgery/dla_transforms")]
        sys.modules[dt] = m
    if ut not in sys.modules:
        m = types.ModuleType(ut)
        m.__path__ = [str(_REPO_ROOT / "modelopt/onnx/graph_surgery/utils")]
        sys.modules[ut] = m

    dec_path = _REPO_ROOT / "modelopt/onnx/graph_surgery/dla_transforms/dla_decompose_lstm.py"
    dec_name = "modelopt.onnx.graph_surgery.dla_transforms.dla_decompose_lstm"
    spec_dec = importlib.util.spec_from_file_location(dec_name, dec_path)
    mod_dec = importlib.util.module_from_spec(spec_dec)
    sys.modules[dec_name] = mod_dec
    assert spec_dec.loader is not None
    spec_dec.loader.exec_module(mod_dec)
    return mod_dec._apply_decompose_lstm


_apply_decompose_lstm = _load_apply_decompose_lstm()

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
    from modelopt.onnx.graph_surgery.dla_transforms._common import infer_shapes

    dec = infer_shapes(dec)
    dec = _apply_decompose_lstm(dec)
    _clamp_ir_version(dec)
    onnx.checker.check_model(dec)

    got = _run_ort(dec, feeds)
    _save_lstm_test_pair(model, dec, direction)
    _assert_outputs_match(ref, got)
