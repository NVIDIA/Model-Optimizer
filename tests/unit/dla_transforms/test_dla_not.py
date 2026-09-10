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
# Unit tests for :mod:`modelopt.onnx.graph_surgery.dla_transforms.dla_not`.

"""ORT CPU parity tests for ``dla_not``.

Transform rule
--------------
``Cast(to=BOOL) → Not`` is rewritten to
``Cast(to=FLOAT) → Clip(0, 1) → Sub(1 - clipped)``.

Only ``Not`` nodes whose input is produced by a ``Cast`` node are touched;
all other ``Not`` nodes (e.g. ``Equal → Not``, graph-input → Not) are left in place.

Coverage
--------
INT32 input → Cast(BOOL) → Not  → transformed, ORT parity
Non-Cast producer (Equal → Not)  → skipped, graph unchanged
Graph-input (BOOL) → Not         → skipped, graph unchanged
Two Cast→Not chains              → both transformed, unique initializer names, ORT parity
Graph with no Not nodes          → noop, ORT parity
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
from onnx import TensorProto, helper

pytest.importorskip("onnxruntime", reason="onnxruntime required for ORT parity checks")

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SAVED_MODELS_DIR = _REPO_ROOT / "scratch_space" / "not_test_models"
_OPSET_VERSION = 17
_MAX_IR_VERSION_FOR_ORT = 9


# ── Dynamic import ────────────────────────────────────────────────────────────


def _load_module():
    mod_key = "modelopt.onnx.graph_surgery.dla_transforms.dla_not"
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
        _REPO_ROOT / "modelopt/onnx/graph_surgery/dla_transforms/dla_not.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_key] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


_mod = _load_module()
_apply = _mod._apply_not


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


# ── 1. INT32 → Cast(BOOL) → Not → transformed ────────────────────────────────


def test_not_cast_to_bool_transformed():
    """Cast(to=BOOL) → Not: Cast redirected to FLOAT, Not replaced by Clip+Sub; ORT parity.

    Input uses only 0 / 1 values so that Cast(INT32→FLOAT) and Cast(INT32→BOOL)
    produce numerically consistent boolean representations.
    """
    shape = [2, 3]
    cast_node = helper.make_node("Cast", ["X"], ["X_bool"], to=TensorProto.BOOL, name="cast")
    not_node = helper.make_node("Not", ["X_bool"], ["Y"], name="not")
    graph = helper.make_graph(
        [cast_node, not_node],
        "not_cast_bool",
        [_vi("X", TensorProto.INT32, shape)],
        [_vi("Y", TensorProto.BOOL, shape)],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", _OPSET_VERSION)])
    _clamp_ir(model)

    # Only 0/1 so Not(Cast(x, BOOL)) == 1 - cast(x, FLOAT) for these values.
    x = np.array([[0, 1, 0], [1, 1, 0]], dtype=np.int32)
    _prepare_model(model)
    ref = _run_ort(model, {"X": x})

    tr = copy.deepcopy(model)
    _prepare_model(tr)
    tr = _apply(tr)
    _prepare_model(tr)
    got = _run_ort(tr, {"X": x})
    _save(model, tr, "not_cast_bool")

    # Transformed output is FLOAT; compare against bool cast to float.
    np.testing.assert_array_equal(ref["Y"].astype(np.float32), got["Y"])

    # Cast node must now produce FLOAT, Not node must be gone.
    cast_nodes = [n for n in tr.graph.node if n.op_type == "Cast"]
    assert len(cast_nodes) == 1
    for attr in cast_nodes[0].attribute:
        if attr.name == "to":
            assert attr.i == TensorProto.FLOAT, "Cast must target FLOAT after transform"
    assert not any(n.op_type == "Not" for n in tr.graph.node), "Not node must be removed"


# ── 2. Equal → Not: non-Cast producer → skipped ──────────────────────────────


def test_not_non_cast_producer_skipped():
    """Not whose input comes from Equal (not Cast) → graph unchanged."""
    shape = [3]
    eq_node = helper.make_node("Equal", ["X", "Z"], ["X_bool"], name="eq")
    not_node = helper.make_node("Not", ["X_bool"], ["Y"], name="not")
    graph = helper.make_graph(
        [eq_node, not_node],
        "not_equal_producer",
        [_vi("X", TensorProto.FLOAT, shape), _vi("Z", TensorProto.FLOAT, shape)],
        [_vi("Y", TensorProto.BOOL, shape)],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", _OPSET_VERSION)])
    _clamp_ir(model)
    original_node_count = len(model.graph.node)

    x = _RNG.standard_normal(shape).astype(np.float32)
    z = _RNG.standard_normal(shape).astype(np.float32)
    _prepare_model(model)
    ref = _run_ort(model, {"X": x, "Z": z})

    tr = copy.deepcopy(model)
    _prepare_model(tr)
    tr = _apply(tr)
    _prepare_model(tr)
    got = _run_ort(tr, {"X": x, "Z": z})
    _save(model, tr, "not_equal_producer")

    assert len(tr.graph.node) == original_node_count, (
        "graph must be unchanged for non-Cast producer"
    )
    np.testing.assert_array_equal(ref["Y"], got["Y"])


# ── 3. BOOL graph input → Not: no producer → skipped ─────────────────────────


def test_not_graph_input_skipped():
    """Not whose operand is a BOOL graph input (no producing node) → graph unchanged."""
    shape = [2, 3]
    not_node = helper.make_node("Not", ["X"], ["Y"], name="not")
    graph = helper.make_graph(
        [not_node],
        "not_graph_input",
        [_vi("X", TensorProto.BOOL, shape)],
        [_vi("Y", TensorProto.BOOL, shape)],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", _OPSET_VERSION)])
    _clamp_ir(model)
    original_node_count = len(model.graph.node)

    x = _RNG.integers(0, 2, shape).astype(bool)
    _prepare_model(model)
    ref = _run_ort(model, {"X": x})

    tr = copy.deepcopy(model)
    _prepare_model(tr)
    tr = _apply(tr)
    _prepare_model(tr)
    got = _run_ort(tr, {"X": x})
    _save(model, tr, "not_graph_input")

    assert len(tr.graph.node) == original_node_count, "graph must be unchanged for graph-input Not"
    np.testing.assert_array_equal(ref["Y"], got["Y"])


# ── 4. Two Cast→Not chains → both transformed, unique initializer names ───────


def test_not_multiple_not_nodes():
    """Two independent Cast(BOOL)→Not chains are both transformed with unique initializer names."""
    shape = [2]
    cast_a = helper.make_node("Cast", ["A"], ["A_bool"], to=TensorProto.BOOL, name="cast_a")
    not_a = helper.make_node("Not", ["A_bool"], ["Y_A"], name="not_a")
    cast_b = helper.make_node("Cast", ["B"], ["B_bool"], to=TensorProto.BOOL, name="cast_b")
    not_b = helper.make_node("Not", ["B_bool"], ["Y_B"], name="not_b")
    graph = helper.make_graph(
        [cast_a, not_a, cast_b, not_b],
        "not_multi",
        [_vi("A", TensorProto.INT32, shape), _vi("B", TensorProto.INT32, shape)],
        [_vi("Y_A", TensorProto.BOOL, shape), _vi("Y_B", TensorProto.BOOL, shape)],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", _OPSET_VERSION)])
    _clamp_ir(model)

    a = np.array([0, 1], dtype=np.int32)
    b = np.array([1, 0], dtype=np.int32)
    _prepare_model(model)
    ref = _run_ort(model, {"A": a, "B": b})

    tr = copy.deepcopy(model)
    _prepare_model(tr)
    tr = _apply(tr)
    _prepare_model(tr)
    got = _run_ort(tr, {"A": a, "B": b})
    _save(model, tr, "not_multi")

    np.testing.assert_array_equal(ref["Y_A"].astype(np.float32), got["Y_A"])
    np.testing.assert_array_equal(ref["Y_B"].astype(np.float32), got["Y_B"])

    # Both Not nodes must be gone; no initializer name collisions.
    assert not any(n.op_type == "Not" for n in tr.graph.node), "all Not nodes must be removed"
    init_names = [init.name for init in tr.graph.initializer]
    assert len(init_names) == len(set(init_names)), "initializer names must be unique"


# ── 5. No Not nodes → noop ────────────────────────────────────────────────────


def test_no_not_nodes_noop():
    """Graph with no Not nodes: transform is a noop, ORT parity preserved."""
    shape = [2, 3]
    relu_node = helper.make_node("Relu", ["X"], ["Y"], name="relu")
    graph = helper.make_graph(
        [relu_node],
        "no_not",
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
    _save(model, tr, "not_noop")

    assert len(tr.graph.node) == original_node_count, "noop: node count must not change"
    np.testing.assert_array_equal(ref["Y"], got["Y"])
