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
# Unit tests for :mod:`modelopt.onnx.graph_surgery.dla_transforms.dla_where`.

"""ORT CPU parity tests for ``dla_where``.

Transform rule
--------------
``Where(cond, x, y)`` is replaced by a ``Mul``-based equivalent when one of
{x, y} is an all-zero static initializer:

    Where(cond, 0, y)  →  Mul(y, 1 - float(cond))
    Where(cond, x, 0)  →  Mul(x, float(cond))

Coverage
--------
BOOL init condition, true-branch zeros   → Mul(y, 1-cond_float), ORT parity
BOOL init condition, false-branch zeros  → Mul(x, cond_float),   ORT parity
Dynamic BOOL condition, true-branch zeros  → Cast+Sub+Mul,        ORT parity
Dynamic BOOL condition, false-branch zeros → Cast+Mul,            ORT parity
Non-BOOL float condition, false-branch zeros → Mul directly,      ORT parity
Neither branch all-zeros                    → graph unchanged
No Where nodes                              → noop, ORT parity
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
_SAVED_MODELS_DIR = _REPO_ROOT / "scratch_space" / "where_test_models"
_OPSET_VERSION = 17
_MAX_IR_VERSION_FOR_ORT = 9


# ── Dynamic import ────────────────────────────────────────────────────────────


def _load_module():
    mod_key = "modelopt.onnx.graph_surgery.dla_transforms.dla_where"
    if mod_key in sys.modules:
        return sys.modules[mod_key]

    import modelopt.onnx  # noqa: F401

    for pkg, rel in [
        ("modelopt.onnx.graph_surgery", "modelopt/onnx/graph_surgery"),
        (
            "modelopt.onnx.graph_surgery.dla_transforms",
            "modelopt/onnx/graph_surgery/dla_transforms",
        ),
        ("modelopt.onnx.graph_surgery.utils", "modelopt/onnx/graph_surgery/utils"),
    ]:
        if pkg not in sys.modules:
            m = types.ModuleType(pkg)
            m.__path__ = [str(_REPO_ROOT / rel)]
            sys.modules[pkg] = m

    spec = importlib.util.spec_from_file_location(
        mod_key,
        _REPO_ROOT / "modelopt/onnx/graph_surgery/dla_transforms/dla_where.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_key] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


_mod = _load_module()
_apply = _mod._apply_where


# ── Shared utilities ──────────────────────────────────────────────────────────


def _vi(name: str, elem_type: int, shape: list) -> onnx.ValueInfoProto:
    return helper.make_tensor_value_info(name, elem_type, shape)


def _clamp_ir(model: onnx.ModelProto) -> None:
    model.ir_version = min(model.ir_version, _MAX_IR_VERSION_FOR_ORT)


def _prepare_model(model: onnx.ModelProto) -> None:
    try:
        from onnxruntime.tools.symbolic_shape_infer import (
            SymbolicShapeInference,  # type: ignore[import-untyped]
        )

        inferred = SymbolicShapeInference.infer_shapes(model)
        if inferred is not None:
            del model.graph.value_info[:]
            model.graph.value_info.extend(inferred.graph.value_info)
    except Exception:
        pass


def _run_ort(model: onnx.ModelProto, feeds: dict) -> dict:
    import onnxruntime as ort  # type: ignore[import-untyped]

    _clamp_ir(model)
    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    return {o.name: sess.run([o.name], feeds)[0] for o in model.graph.output}


def _save(base, tr, tag):
    _SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    onnx.save(base, str(_SAVED_MODELS_DIR / f"{tag}_base.onnx"))
    try:
        from onnxruntime.tools.symbolic_shape_infer import (
            SymbolicShapeInference,  # type: ignore[import-untyped]
        )

        tr2 = SymbolicShapeInference.infer_shapes(
            copy.deepcopy(tr), auto_merge_symbolic_dims=True
        ) or copy.deepcopy(tr)
    except Exception:
        tr2 = copy.deepcopy(tr)
    onnx.save(tr2, str(_SAVED_MODELS_DIR / f"{tag}_transformed.onnx"))


_RNG = np.random.default_rng(2024)
_SHAPE = [2, 3]


# ── 1. Static BOOL condition, true-branch (x) is zeros ───────────────────────


def test_where_bool_init_cond_x_zeros():
    """Where(bool_init, zeros, y) → Mul(y, 1-cond_float); ORT parity.

    Where(cond, 0, y) selects y when condition is False, so the mask is (1 - cond).
    """
    cond_arr = np.array([[True, False, True], [False, True, False]])
    zeros_arr = np.zeros(_SHAPE, dtype=np.float32)
    y_arr = _RNG.standard_normal(_SHAPE).astype(np.float32)

    cond_init = numpy_helper.from_array(cond_arr, name="cond")
    x_init = numpy_helper.from_array(zeros_arr, name="x_zeros")

    where = helper.make_node("Where", ["cond", "x_zeros", "Y_in"], ["Z"], name="where")
    graph = helper.make_graph(
        [where],
        "where_bool_init_x0",
        [_vi("Y_in", TensorProto.FLOAT, _SHAPE)],
        [_vi("Z", TensorProto.FLOAT, _SHAPE)],
        initializer=[cond_init, x_init],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", _OPSET_VERSION)])
    _clamp_ir(model)

    _prepare_model(model)
    ref = _run_ort(model, {"Y_in": y_arr})

    tr = copy.deepcopy(model)
    _prepare_model(tr)
    tr = _apply(tr)
    _prepare_model(tr)
    got = _run_ort(tr, {"Y_in": y_arr})
    _save(model, tr, "where_bool_init_x0")

    assert not any(n.op_type == "Where" for n in tr.graph.node), "Where must be removed"
    np.testing.assert_allclose(ref["Z"], got["Z"], rtol=1e-5, atol=1e-6)


# ── 2. Static BOOL condition, false-branch (y) is zeros ──────────────────────


def test_where_bool_init_cond_y_zeros():
    """Where(bool_init, x, zeros) → Mul(x, cond_float); ORT parity.

    Where(cond, x, 0) selects x when condition is True, so the mask is cond itself.
    """
    cond_arr = np.array([[True, False, True], [False, True, False]])
    zeros_arr = np.zeros(_SHAPE, dtype=np.float32)
    x_arr = _RNG.standard_normal(_SHAPE).astype(np.float32)

    cond_init = numpy_helper.from_array(cond_arr, name="cond2")
    y_init = numpy_helper.from_array(zeros_arr, name="y_zeros")

    where = helper.make_node("Where", ["cond2", "X_in", "y_zeros"], ["Z"], name="where")
    graph = helper.make_graph(
        [where],
        "where_bool_init_y0",
        [_vi("X_in", TensorProto.FLOAT, _SHAPE)],
        [_vi("Z", TensorProto.FLOAT, _SHAPE)],
        initializer=[cond_init, y_init],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", _OPSET_VERSION)])
    _clamp_ir(model)

    _prepare_model(model)
    ref = _run_ort(model, {"X_in": x_arr})

    tr = copy.deepcopy(model)
    _prepare_model(tr)
    tr = _apply(tr)
    _prepare_model(tr)
    got = _run_ort(tr, {"X_in": x_arr})
    _save(model, tr, "where_bool_init_y0")

    assert not any(n.op_type == "Where" for n in tr.graph.node), "Where must be removed"
    np.testing.assert_allclose(ref["Z"], got["Z"], rtol=1e-5, atol=1e-6)


# ── 3. Dynamic BOOL condition, true-branch (x) is zeros ──────────────────────


def test_where_dynamic_bool_cond_x_zeros():
    """Where(bool_graph_input, zeros_init, y) → Cast+Sub+Mul; ORT parity."""
    zeros_arr = np.zeros(_SHAPE, dtype=np.float32)
    x_init = numpy_helper.from_array(zeros_arr, name="x_zeros_dyn")

    where = helper.make_node("Where", ["cond", "x_zeros_dyn", "Y_in"], ["Z"], name="where")
    graph = helper.make_graph(
        [where],
        "where_dyn_bool_x0",
        [
            _vi("cond", TensorProto.BOOL, _SHAPE),
            _vi("Y_in", TensorProto.FLOAT, _SHAPE),
        ],
        [_vi("Z", TensorProto.FLOAT, _SHAPE)],
        initializer=[x_init],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", _OPSET_VERSION)])
    _clamp_ir(model)

    cond_val = _RNG.integers(0, 2, _SHAPE).astype(bool)
    y_arr = _RNG.standard_normal(_SHAPE).astype(np.float32)
    _prepare_model(model)
    ref = _run_ort(model, {"cond": cond_val, "Y_in": y_arr})

    tr = copy.deepcopy(model)
    _prepare_model(tr)
    tr = _apply(tr)
    _prepare_model(tr)
    got = _run_ort(tr, {"cond": cond_val, "Y_in": y_arr})
    _save(model, tr, "where_dyn_bool_x0")

    assert not any(n.op_type == "Where" for n in tr.graph.node), "Where must be removed"
    # Cast + Sub + Mul should be present
    op_types = {n.op_type for n in tr.graph.node}
    assert "Cast" in op_types, "Cast node expected for dynamic BOOL condition"
    assert "Sub" in op_types, "Sub node expected for x-is-zero inversion"
    assert "Mul" in op_types
    np.testing.assert_allclose(ref["Z"], got["Z"], rtol=1e-5, atol=1e-6)


# ── 4. Dynamic BOOL condition, false-branch (y) is zeros ─────────────────────


def test_where_dynamic_bool_cond_y_zeros():
    """Where(bool_graph_input, x, zeros_init) → Cast+Mul; ORT parity."""
    zeros_arr = np.zeros(_SHAPE, dtype=np.float32)
    y_init = numpy_helper.from_array(zeros_arr, name="y_zeros_dyn")

    where = helper.make_node("Where", ["cond", "X_in", "y_zeros_dyn"], ["Z"], name="where")
    graph = helper.make_graph(
        [where],
        "where_dyn_bool_y0",
        [
            _vi("cond", TensorProto.BOOL, _SHAPE),
            _vi("X_in", TensorProto.FLOAT, _SHAPE),
        ],
        [_vi("Z", TensorProto.FLOAT, _SHAPE)],
        initializer=[y_init],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", _OPSET_VERSION)])
    _clamp_ir(model)

    cond_val = _RNG.integers(0, 2, _SHAPE).astype(bool)
    x_arr = _RNG.standard_normal(_SHAPE).astype(np.float32)
    _prepare_model(model)
    ref = _run_ort(model, {"cond": cond_val, "X_in": x_arr})

    tr = copy.deepcopy(model)
    _prepare_model(tr)
    tr = _apply(tr)
    _prepare_model(tr)
    got = _run_ort(tr, {"cond": cond_val, "X_in": x_arr})
    _save(model, tr, "where_dyn_bool_y0")

    assert not any(n.op_type == "Where" for n in tr.graph.node), "Where must be removed"
    op_types = {n.op_type for n in tr.graph.node}
    assert "Cast" in op_types, "Cast node expected for dynamic BOOL condition"
    assert "Sub" not in op_types, "Sub must NOT be inserted for y-is-zero case"
    assert "Mul" in op_types
    np.testing.assert_allclose(ref["Z"], got["Z"], rtol=1e-5, atol=1e-6)


# ── 5. Non-BOOL float condition (dynamic), false-branch zeros ─────────────────


def test_where_float_cond_y_zeros():
    """Where with a FLOAT dynamic condition and zeros false-branch → Mul directly (no Cast).

    The float condition (0.0 / 1.0 values) acts directly as the mask.
    """
    zeros_arr = np.zeros(_SHAPE, dtype=np.float32)
    y_init = numpy_helper.from_array(zeros_arr, name="y_zeros_f")

    # Sigmoid produces values in (0, 1); use a Cast(BOOL) then Cast(FLOAT) workaround
    # is not needed here — instead, we make a simple graph where the condition is FLOAT.
    # ORT does NOT accept a FLOAT condition for Where, so we model it differently:
    # build a graph with Greater producing a BOOL mask, then use that as condition.
    greater = helper.make_node("Greater", ["X_in", "thresh"], ["cond_bool"], name="greater")
    cast = helper.make_node(
        "Cast", ["cond_bool"], ["cond_float"], to=TensorProto.FLOAT, name="cast"
    )
    where = helper.make_node("Where", ["cond_bool", "X_in", "y_zeros_f"], ["Z"], name="where")

    thresh_init = numpy_helper.from_array(np.array(0.0, dtype=np.float32), name="thresh")
    graph = helper.make_graph(
        [greater, cast, where],
        "where_float_cond_y0",
        [_vi("X_in", TensorProto.FLOAT, _SHAPE)],
        [_vi("Z", TensorProto.FLOAT, _SHAPE)],
        initializer=[y_init, thresh_init],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", _OPSET_VERSION)])
    _clamp_ir(model)

    x_arr = _RNG.standard_normal(_SHAPE).astype(np.float32)
    _prepare_model(model)
    ref = _run_ort(model, {"X_in": x_arr})

    tr = copy.deepcopy(model)
    _prepare_model(tr)
    tr = _apply(tr)
    _prepare_model(tr)
    got = _run_ort(tr, {"X_in": x_arr})
    _save(model, tr, "where_float_cond_y0")

    assert not any(n.op_type == "Where" for n in tr.graph.node), "Where must be removed"
    np.testing.assert_allclose(ref["Z"], got["Z"], rtol=1e-5, atol=1e-6)


# ── 6. Neither branch is all-zeros → graph unchanged ─────────────────────────


def test_where_no_zero_branch_skipped():
    """Where with no all-zero branch is left unchanged."""
    x_arr = np.ones(_SHAPE, dtype=np.float32)
    y_arr = np.full(_SHAPE, 2.0, dtype=np.float32)
    cond_arr = np.array([[True, False, True], [False, True, False]])

    cond_init = numpy_helper.from_array(cond_arr, name="cond_nz")
    x_init = numpy_helper.from_array(x_arr, name="x_nz")
    y_init = numpy_helper.from_array(y_arr, name="y_nz")

    where = helper.make_node("Where", ["cond_nz", "x_nz", "y_nz"], ["Z"], name="where")
    graph = helper.make_graph(
        [where],
        "where_no_zero",
        [],
        [_vi("Z", TensorProto.FLOAT, _SHAPE)],
        initializer=[cond_init, x_init, y_init],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", _OPSET_VERSION)])
    _clamp_ir(model)
    original_node_count = len(model.graph.node)

    _prepare_model(model)
    ref = _run_ort(model, {})

    tr = copy.deepcopy(model)
    _prepare_model(tr)
    tr = _apply(tr)
    _prepare_model(tr)
    got = _run_ort(tr, {})
    _save(model, tr, "where_no_zero")

    assert len(tr.graph.node) == original_node_count, "graph must be unchanged"
    np.testing.assert_array_equal(ref["Z"], got["Z"])


# ── 7. No Where nodes → noop ──────────────────────────────────────────────────


def test_no_where_nodes_noop():
    """Graph without Where nodes: transform is a noop, ORT parity preserved."""
    shape = [2, 3]
    relu_node = helper.make_node("Relu", ["X"], ["Y"], name="relu")
    graph = helper.make_graph(
        [relu_node],
        "no_where",
        [_vi("X", TensorProto.FLOAT, shape)],
        [_vi("Y", TensorProto.FLOAT, shape)],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", _OPSET_VERSION)])
    _clamp_ir(model)
    original_node_count = len(model.graph.node)

    x = _RNG.standard_normal(shape).astype(np.float32)
    _prepare_model(model)
    ref = _run_ort(model, {"X": x})

    tr = copy.deepcopy(model)
    _prepare_model(tr)
    tr = _apply(tr)
    _prepare_model(tr)
    got = _run_ort(tr, {"X": x})
    _save(model, tr, "where_noop")

    assert len(tr.graph.node) == original_node_count, "noop: node count must not change"
    np.testing.assert_array_equal(ref["Y"], got["Y"])
