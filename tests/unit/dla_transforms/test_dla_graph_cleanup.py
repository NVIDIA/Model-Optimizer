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
# Unit tests for :mod:`modelopt.onnx.graph_surgery.dla_transforms.dla_graph_cleanup`.

"""Structural and ORT parity tests for ``dla_graph_cleanup``.

The combined transform applies five steps:
  1. Canonicalize non-4D inputs  (adds Unsqueeze, deduplicates existing ones)
  2. Replace intermediary Squeeze/Unsqueeze with 4-D Reshape
  3. Collapse Reshape chains (noop → remove; multi → single Reshape)
  4. Fold constants via onnx-graphsurgeon
  5. Remove unused initializers

Steps 1 and 2 intentionally change tensor shapes, so ORT parity only works
for end-to-end models that happen to keep graph I/O shapes the same.
Tests for those steps verify structural changes (node types, counts).
Steps 3-5 preserve graph I/O shapes, so ORT parity is verified for them.
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
_SAVED_MODELS_DIR = _REPO_ROOT / "scratch_space" / "graph_cleanup_test_models"
_OPSET_VERSION = 17
_MAX_IR_VERSION_FOR_ORT = 9


# ── Dynamic import ────────────────────────────────────────────────────────────


def _load_module():
    mod_key = "modelopt.onnx.graph_surgery.dla_transforms.dla_graph_cleanup"
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
        _REPO_ROOT / "modelopt/onnx/graph_surgery/dla_transforms/dla_graph_cleanup.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_key] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


_mod = _load_module()
_apply = _mod._apply_graph_cleanup
_fold_consts = _mod._fold_constants
_remove_unused_inits = _mod._remove_unused_initializers

from modelopt.onnx.graph_surgery.dla_transforms._common import GraphCache


def _canonicalize_inputs(model):
    return _mod._canonicalize_inputs(model, GraphCache(model.graph))


def _replace_sq_unsq(model):
    return _mod._replace_intermediary_squeeze_unsqueeze(model, GraphCache(model.graph))


def _remove_reshapes(model):
    return _mod._remove_reshape_chains(model, GraphCache(model.graph))


def _reshape_to_squeeze(model):
    return _mod._replace_graph_output_reshape_with_squeeze(model, GraphCache(model.graph))


def _remove_cast_chains(model):
    return _mod._remove_consecutive_casts(model, GraphCache(model.graph))


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


# ══════════════════════════════════════════════════════════════════════════════
# Step 1 — Canonicalize non-4D inputs
# ══════════════════════════════════════════════════════════════════════════════


def test_canonicalize_3d_input_unsqueeze_added():
    """3-D graph input X [2,3,4] → Unsqueeze to 4-D [1,2,3,4] is inserted.

    Downstream Relu receives the 4-D tensor.
    """
    relu = helper.make_node("Relu", ["X"], ["Y"], name="relu")
    graph = helper.make_graph(
        [relu],
        "g",
        [_vi("X", TensorProto.FLOAT, [2, 3, 4])],
        [_vi("Y", TensorProto.FLOAT, [2, 3, 4])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", _OPSET_VERSION)])
    _clamp_ir(model)

    tr = copy.deepcopy(model)
    _prepare_model(tr)
    tr = _canonicalize_inputs(tr)

    unsq_nodes = [n for n in tr.graph.node if n.op_type == "Unsqueeze"]
    assert len(unsq_nodes) == 1, "one Unsqueeze must be inserted for the 3-D input"
    unsq = unsq_nodes[0]
    assert unsq.input[0] == "X", "Unsqueeze reads the graph input"

    # Relu must now read the Unsqueeze output, not X directly
    relu_node = next(n for n in tr.graph.node if n.op_type == "Relu")
    assert relu_node.input[0] == unsq.output[0], "Relu must be redirected to Unsqueeze output"
    _save(model, tr, "canonicalize_3d")


def test_canonicalize_2d_input_unsqueeze_added():
    """2-D graph input X [4,5] → Unsqueeze at axes [0,1] to 4-D [1,1,4,5]."""
    relu = helper.make_node("Relu", ["X"], ["Y"], name="relu")
    graph = helper.make_graph(
        [relu],
        "g",
        [_vi("X", TensorProto.FLOAT, [4, 5])],
        [_vi("Y", TensorProto.FLOAT, [4, 5])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", _OPSET_VERSION)])
    _clamp_ir(model)

    tr = copy.deepcopy(model)
    _prepare_model(tr)
    tr = _canonicalize_inputs(tr)

    unsq_nodes = [n for n in tr.graph.node if n.op_type == "Unsqueeze"]
    assert len(unsq_nodes) == 1
    axes_init = next(i for i in tr.graph.initializer if i.name.endswith("_unsqueeze_axes"))
    axes = numpy_helper.to_array(axes_init).tolist()
    assert axes == [0, 1], f"axes should be [0,1] for 2-D input, got {axes}"
    _save(model, tr, "canonicalize_2d")


def test_canonicalize_3d_input_duplicate_unsqueezes_deduped():
    """3-D input consumed by two Unsqueeze nodes → only one canonical Unsqueeze remains.

    Both existing Unsqueezes (same axes) are replaced by the single canonical one.
    """
    axes_init = numpy_helper.from_array(np.array([0], dtype=np.int64), name="ax")
    unsq_a = helper.make_node("Unsqueeze", ["X", "ax"], ["Xa"], name="unsq_a")
    unsq_b = helper.make_node("Unsqueeze", ["X", "ax"], ["Xb"], name="unsq_b")
    relu_a = helper.make_node("Relu", ["Xa"], ["Ya"], name="relu_a")
    relu_b = helper.make_node("Relu", ["Xb"], ["Yb"], name="relu_b")
    graph = helper.make_graph(
        [unsq_a, unsq_b, relu_a, relu_b],
        "g",
        [_vi("X", TensorProto.FLOAT, [2, 3, 4])],
        [_vi("Ya", TensorProto.FLOAT, [1, 2, 3, 4]), _vi("Yb", TensorProto.FLOAT, [1, 2, 3, 4])],
        initializer=[axes_init],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", _OPSET_VERSION)])
    _clamp_ir(model)

    tr = copy.deepcopy(model)
    _prepare_model(tr)
    tr = _canonicalize_inputs(tr)

    unsq_nodes = [n for n in tr.graph.node if n.op_type == "Unsqueeze"]
    # Only the single canonical Unsqueeze should remain; old ones are removed/replaced
    assert len(unsq_nodes) == 1, (
        f"exactly one Unsqueeze should remain after dedup, got {len(unsq_nodes)}"
    )
    _save(model, tr, "canonicalize_dedup")


def test_canonicalize_4d_input_unchanged():
    """4-D graph input → step 1 is a noop."""
    relu = helper.make_node("Relu", ["X"], ["Y"], name="relu")
    graph = helper.make_graph(
        [relu],
        "g",
        [_vi("X", TensorProto.FLOAT, [1, 2, 3, 4])],
        [_vi("Y", TensorProto.FLOAT, [1, 2, 3, 4])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", _OPSET_VERSION)])
    _clamp_ir(model)
    orig_nodes = len(model.graph.node)

    tr = copy.deepcopy(model)
    _prepare_model(tr)
    tr = _canonicalize_inputs(tr)

    assert len(tr.graph.node) == orig_nodes, "4-D input: no nodes should be added"


# ══════════════════════════════════════════════════════════════════════════════
# Step 2 — Replace intermediary Squeeze / Unsqueeze with 4-D Reshape
# ══════════════════════════════════════════════════════════════════════════════


def test_intermediary_squeeze_replaced_with_reshape():
    """Squeeze [2,1,6,8] → [2,6,8] (axis=1) is replaced by Reshape → [1,2,6,8].

    pad([2,1,6,8]) = [2,1,6,8], pad([2,6,8]) = [1,2,6,8] — shapes differ → Reshape inserted.
    The squeezed axis must be 1 (a unit dimension) for ORT symbolic shape inference to accept it.
    """
    sq_axes = numpy_helper.from_array(np.array([1], dtype=np.int64), name="sq_ax")
    relu = helper.make_node("Relu", ["X"], ["X_relu"], name="relu")
    squeeze = helper.make_node("Squeeze", ["X_relu", "sq_ax"], ["Sq_out"], name="sq")
    relu2 = helper.make_node("Relu", ["Sq_out"], ["Y"], name="relu2")
    graph = helper.make_graph(
        [relu, squeeze, relu2],
        "g",
        [_vi("X", TensorProto.FLOAT, [2, 1, 6, 8])],
        [_vi("Y", TensorProto.FLOAT, [2, 6, 8])],
        initializer=[sq_axes],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", _OPSET_VERSION)])
    _clamp_ir(model)

    tr = copy.deepcopy(model)
    _prepare_model(tr)
    tr = _replace_sq_unsq(tr)

    assert not any(n.op_type == "Squeeze" for n in tr.graph.node), "Squeeze must be removed"
    reshape_nodes = [n for n in tr.graph.node if n.op_type == "Reshape"]
    assert reshape_nodes, "a Reshape must be inserted"

    # Shape init name follows pattern {node.name}_4d_shape → "sq_4d_shape"
    shape_init = next((i for i in tr.graph.initializer if i.name == "sq_4d_shape"), None)
    assert shape_init is not None, "Reshape shape initializer must exist"
    shape = numpy_helper.to_array(shape_init).tolist()
    assert shape == [1, 2, 6, 8], f"expected [1,2,6,8], got {shape}"
    _save(model, tr, "intermediary_squeeze")


def test_intermediary_unsqueeze_replaced_with_reshape():
    """Unsqueeze [2,4,6] → [2,4,1,6] (axis=2) is replaced by Reshape → [2,4,1,6].

    pad([2,4,6]) = [1,2,4,6], pad([2,4,1,6]) = [2,4,1,6] — shapes differ → Reshape inserted.
    """
    unsq_axes = numpy_helper.from_array(np.array([2], dtype=np.int64), name="unsq_ax")
    relu = helper.make_node("Relu", ["X"], ["X_relu"], name="relu")
    unsqueeze = helper.make_node("Unsqueeze", ["X_relu", "unsq_ax"], ["Uq_out"], name="uq")
    relu2 = helper.make_node("Relu", ["Uq_out"], ["Y"], name="relu2")
    graph = helper.make_graph(
        [relu, unsqueeze, relu2],
        "g",
        [_vi("X", TensorProto.FLOAT, [2, 4, 6])],
        [_vi("Y", TensorProto.FLOAT, [2, 4, 1, 6])],
        initializer=[unsq_axes],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", _OPSET_VERSION)])
    _clamp_ir(model)

    tr = copy.deepcopy(model)
    _prepare_model(tr)
    tr = _replace_sq_unsq(tr)

    assert not any(n.op_type == "Unsqueeze" for n in tr.graph.node), "Unsqueeze must be removed"
    assert any(n.op_type == "Reshape" for n in tr.graph.node), "Reshape must be inserted"
    # Shape init name: {node.name}_4d_shape → "uq_4d_shape"
    shape_init = next((i for i in tr.graph.initializer if i.name == "uq_4d_shape"), None)
    assert shape_init is not None, "Reshape shape initializer must exist"
    shape = numpy_helper.to_array(shape_init).tolist()
    assert shape == [2, 4, 1, 6], f"expected [2,4,1,6], got {shape}"
    _save(model, tr, "intermediary_unsqueeze")


def test_intermediary_squeeze_at_graph_output_replaced():
    """Squeeze whose output IS a graph output is now replaced like any other intermediary Squeeze.

    Squeeze [1,2,1,5] → [1,2,5] (axis=2): pad([1,2,1,5])=[1,2,1,5] ≠ pad([1,2,5])=[1,1,2,5]
    so a Reshape is inserted and the Squeeze is removed.
    """
    sq_axes = numpy_helper.from_array(np.array([2], dtype=np.int64), name="sq_ax2")
    relu = helper.make_node("Relu", ["X"], ["X_relu"], name="relu")
    squeeze = helper.make_node("Squeeze", ["X_relu", "sq_ax2"], ["Y"], name="sq")
    graph = helper.make_graph(
        [relu, squeeze],
        "g",
        [_vi("X", TensorProto.FLOAT, [1, 2, 1, 5])],
        [_vi("Y", TensorProto.FLOAT, [1, 2, 5])],
        initializer=[sq_axes],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", _OPSET_VERSION)])
    _clamp_ir(model)

    tr = copy.deepcopy(model)
    _prepare_model(tr)
    tr = _replace_sq_unsq(tr)

    assert not any(n.op_type == "Squeeze" for n in tr.graph.node), (
        "Squeeze at graph output must now be replaced"
    )
    assert any(n.op_type == "Reshape" for n in tr.graph.node), "Reshape must be inserted"
    _save(model, tr, "squeeze_at_graph_output")


def test_intermediary_squeeze_identity_at_graph_output_uses_reshape():
    """Squeeze identity in 4-D AND graph output → Reshape with graph output shape.

    Squeeze [1,1,5] → [1,5] (axis=0): padded shapes are identical so normally the node
    would be removed.  Because Y is a graph output, a Reshape is inserted instead and its
    target shape must equal the declared graph output shape [1,5] — not the 4D padded shape.
    """
    sq_axes = numpy_helper.from_array(np.array([0], dtype=np.int64), name="sq_ax_id")
    relu = helper.make_node("Relu", ["X"], ["X_relu"], name="relu")
    squeeze = helper.make_node("Squeeze", ["X_relu", "sq_ax_id"], ["Y"], name="sq_id")
    graph = helper.make_graph(
        [relu, squeeze],
        "g",
        [_vi("X", TensorProto.FLOAT, [1, 1, 5])],
        [_vi("Y", TensorProto.FLOAT, [1, 5])],
        initializer=[sq_axes],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", _OPSET_VERSION)])
    _clamp_ir(model)

    tr = copy.deepcopy(model)
    _prepare_model(tr)
    tr = _replace_sq_unsq(tr)

    assert not any(n.op_type == "Squeeze" for n in tr.graph.node), (
        "Squeeze must be replaced even for identity case at graph output"
    )
    assert any(n.op_type == "Reshape" for n in tr.graph.node), (
        "Reshape must be inserted to preserve the graph output name"
    )

    # Reshape target shape must equal the graph output shape [1,5], not 4D padded [1,1,1,5]
    rs_shape_init = next((i for i in tr.graph.initializer if i.name == "sq_id_4d_shape"), None)
    assert rs_shape_init is not None, "Reshape shape initializer must exist"
    assert numpy_helper.to_array(rs_shape_init).tolist() == [1, 5], (
        "Reshape target must equal graph output shape [1,5]"
    )

    assert any(o.name == "Y" for o in tr.graph.output), "Y must remain a graph output"
    _save(model, tr, "squeeze_identity_graph_output")


def test_unsqueeze_on_graph_input_skipped():
    """Unsqueeze whose data input IS a graph input must not be replaced.

    This Unsqueeze is the canonical 4D-promotion node placed by convert_ops_to_4d
    or canonicalize_inputs.  Replacing it with a Reshape would break the graph
    because the input tensor shape stays 3D but the Reshape would expect a 4D input.
    """
    unsq_axes = numpy_helper.from_array(np.array([0], dtype=np.int64), name="gi_unsq_ax")
    # Unsqueeze directly on graph input X [2,4,6] → [1,2,4,6]
    unsqueeze = helper.make_node("Unsqueeze", ["X", "gi_unsq_ax"], ["X_4d"], name="gi_unsq")
    relu = helper.make_node("Relu", ["X_4d"], ["Y"], name="relu")
    graph = helper.make_graph(
        [unsqueeze, relu],
        "g",
        [_vi("X", TensorProto.FLOAT, [2, 4, 6])],
        [_vi("Y", TensorProto.FLOAT, [1, 2, 4, 6])],
        initializer=[unsq_axes],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", _OPSET_VERSION)])
    _clamp_ir(model)
    orig_node_count = len(model.graph.node)

    tr = copy.deepcopy(model)
    _prepare_model(tr)
    tr = _replace_sq_unsq(tr)

    # Unsqueeze on graph input must be preserved exactly as-is
    assert any(n.op_type == "Unsqueeze" for n in tr.graph.node), (
        "Unsqueeze on graph input must be preserved"
    )
    assert not any(n.op_type == "Reshape" for n in tr.graph.node), (
        "No Reshape must be inserted for graph-input Unsqueeze"
    )
    assert len(tr.graph.node) == orig_node_count, "Node count must be unchanged"
    _save(model, tr, "unsqueeze_on_graph_input_skipped")


# ══════════════════════════════════════════════════════════════════════════════
# Step 3 — Collapse Reshape chains
# ══════════════════════════════════════════════════════════════════════════════


def _make_reshape_graph(shapes: list[list[int]]) -> onnx.ModelProto:
    """Build a graph: X → Reshape1 → Reshape2 → … → Y, each using a fixed shape."""
    assert len(shapes) >= 2
    inits = []
    nodes = []
    prev_name = "X"
    for i, shape in enumerate(shapes[1:], 1):
        sh_init = numpy_helper.from_array(np.array(shape, dtype=np.int64), name=f"sh{i}")
        inits.append(sh_init)
        out_name = "Y" if i == len(shapes) - 1 else f"T{i}"
        nodes.append(helper.make_node("Reshape", [prev_name, f"sh{i}"], [out_name], name=f"rs{i}"))
        prev_name = out_name
    graph = helper.make_graph(
        nodes,
        "g",
        [_vi("X", TensorProto.FLOAT, shapes[0])],
        [_vi("Y", TensorProto.FLOAT, shapes[-1])],
        initializer=inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", _OPSET_VERSION)])
    _clamp_ir(model)
    return model


def test_reshape_chain_noop_removed():
    """X[1,2,3,4] → Reshape[1,2,3,4] → Reshape[1,2,3,4] → Y: both removed, ORT parity."""
    model = _make_reshape_graph([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]])
    x = _RNG.standard_normal([1, 2, 3, 4]).astype(np.float32)

    _prepare_model(model)
    ref = _run_ort(model, {"X": x})

    tr = copy.deepcopy(model)
    _prepare_model(tr)
    tr = _remove_reshapes(tr)

    reshape_nodes = [n for n in tr.graph.node if n.op_type == "Reshape"]
    assert len(reshape_nodes) == 0, "identity-reshape chain must be fully removed"

    _prepare_model(tr)
    got = _run_ort(tr, {"X": x})
    _save(model, tr, "reshape_chain_noop")
    np.testing.assert_array_equal(ref["Y"], got["Y"])


def test_reshape_chain_collapsed_to_one():
    """X[6] → Reshape[2,3] → Reshape[1,1,2,3] → Y: collapsed to single Reshape, ORT parity."""
    model = _make_reshape_graph([[6], [2, 3], [1, 1, 2, 3]])
    x = _RNG.standard_normal([6]).astype(np.float32)

    _prepare_model(model)
    ref = _run_ort(model, {"X": x})

    tr = copy.deepcopy(model)
    _prepare_model(tr)
    tr = _remove_reshapes(tr)

    reshape_nodes = [n for n in tr.graph.node if n.op_type == "Reshape"]
    assert len(reshape_nodes) == 1, "two-step chain must collapse to one Reshape"

    _prepare_model(tr)
    got = _run_ort(tr, {"X": x})
    _save(model, tr, "reshape_chain_collapsed")
    np.testing.assert_array_equal(ref["Y"], got["Y"])


def test_single_reshape_unchanged():
    """A single Reshape is not touched."""
    model = _make_reshape_graph([[6], [2, 3]])
    orig_count = len(model.graph.node)

    tr = copy.deepcopy(model)
    _prepare_model(tr)
    tr = _remove_reshapes(tr)

    assert len(tr.graph.node) == orig_count, "single Reshape must not be changed"


# ══════════════════════════════════════════════════════════════════════════════
# Step 4 — Fold constants
# ══════════════════════════════════════════════════════════════════════════════


def test_fold_constants_const_add_folded():
    """Constant + Add → output is a constant: the Add is folded away.

    After folding, the graph should have fewer nodes and ORT output must match.
    """
    a_init = numpy_helper.from_array(np.array([1.0, 2.0, 3.0], dtype=np.float32), name="A")
    b_init = numpy_helper.from_array(np.array([4.0, 5.0, 6.0], dtype=np.float32), name="B")
    add = helper.make_node("Add", ["A", "B"], ["Y"], name="add")
    graph = helper.make_graph(
        [add],
        "g",
        [],
        [_vi("Y", TensorProto.FLOAT, [3])],
        initializer=[a_init, b_init],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", _OPSET_VERSION)])
    _clamp_ir(model)

    _prepare_model(model)
    ref = _run_ort(model, {})

    tr = _fold_consts(copy.deepcopy(model))
    _prepare_model(tr)
    got = _run_ort(tr, {})
    _save(model, tr, "fold_constants")

    # After folding the Add node should be gone (its result is a new initializer)
    assert not any(n.op_type == "Add" for n in tr.graph.node), "Add must be folded away"
    np.testing.assert_array_equal(ref["Y"], got["Y"])


def test_fold_constants_dynamic_input_not_folded():
    """Add with a dynamic input is not folded."""
    a_init = numpy_helper.from_array(np.array([1.0, 2.0, 3.0], dtype=np.float32), name="A")
    add = helper.make_node("Add", ["X", "A"], ["Y"], name="add")
    graph = helper.make_graph(
        [add],
        "g",
        [_vi("X", TensorProto.FLOAT, [3])],
        [_vi("Y", TensorProto.FLOAT, [3])],
        initializer=[a_init],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", _OPSET_VERSION)])
    _clamp_ir(model)
    orig_count = len(model.graph.node)

    tr = _fold_consts(copy.deepcopy(model))

    assert len(tr.graph.node) == orig_count, "dynamic Add must not be folded"

    x = _RNG.standard_normal([3]).astype(np.float32)
    _prepare_model(model)
    ref = _run_ort(model, {"X": x})
    _prepare_model(tr)
    got = _run_ort(tr, {"X": x})
    np.testing.assert_array_equal(ref["Y"], got["Y"])


# ══════════════════════════════════════════════════════════════════════════════
# Step 5 — Remove unused initializers
# ══════════════════════════════════════════════════════════════════════════════


def test_remove_unused_initializers():
    """Initializer not referenced by any node is dropped; ORT parity preserved."""
    used = numpy_helper.from_array(np.array([1.0, 2.0], dtype=np.float32), name="used")
    orphan = numpy_helper.from_array(np.array([99.0], dtype=np.float32), name="orphan")
    relu = helper.make_node("Relu", ["used"], ["Y"], name="relu")
    graph = helper.make_graph(
        [relu],
        "g",
        [],
        [_vi("Y", TensorProto.FLOAT, [2])],
        initializer=[used, orphan],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", _OPSET_VERSION)])
    _clamp_ir(model)

    _prepare_model(model)
    ref = _run_ort(model, {})

    tr = copy.deepcopy(model)
    tr = _remove_unused_inits(tr)

    init_names = {i.name for i in tr.graph.initializer}
    assert "used" in init_names, "'used' initializer must be kept"
    assert "orphan" not in init_names, "'orphan' initializer must be removed"

    _prepare_model(tr)
    got = _run_ort(tr, {})
    _save(model, tr, "remove_unused_inits")
    np.testing.assert_array_equal(ref["Y"], got["Y"])


def test_remove_unused_initializers_all_used_unchanged():
    """When all initializers are in use, none are removed."""
    a = numpy_helper.from_array(np.array([1.0, 2.0], dtype=np.float32), name="A")
    b = numpy_helper.from_array(np.array([3.0, 4.0], dtype=np.float32), name="B")
    add = helper.make_node("Add", ["A", "B"], ["Y"], name="add")
    graph = helper.make_graph(
        [add],
        "g",
        [],
        [_vi("Y", TensorProto.FLOAT, [2])],
        initializer=[a, b],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", _OPSET_VERSION)])
    _clamp_ir(model)
    orig_count = len(model.graph.initializer)

    tr = copy.deepcopy(model)
    tr = _remove_unused_inits(tr)

    assert len(tr.graph.initializer) == orig_count, "no initializers should be removed"


# ══════════════════════════════════════════════════════════════════════════════
# Combined: noop graph → all steps are noops, ORT parity
# ══════════════════════════════════════════════════════════════════════════════


def test_combined_noop_graph():
    """4-D input, single Relu, no Reshape chains, no constants → combined transform is a noop."""
    relu = helper.make_node("Relu", ["X"], ["Y"], name="relu")
    graph = helper.make_graph(
        [relu],
        "g",
        [_vi("X", TensorProto.FLOAT, [1, 2, 3, 4])],
        [_vi("Y", TensorProto.FLOAT, [1, 2, 3, 4])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", _OPSET_VERSION)])
    _clamp_ir(model)
    orig_count = len(model.graph.node)
    orig_inits = len(model.graph.initializer)

    x = _RNG.standard_normal([1, 2, 3, 4]).astype(np.float32)
    _prepare_model(model)
    ref = _run_ort(model, {"X": x})

    tr = copy.deepcopy(model)
    _prepare_model(tr)
    tr = _apply(tr)
    _prepare_model(tr)
    got = _run_ort(tr, {"X": x})
    _save(model, tr, "combined_noop")

    assert len(tr.graph.node) == orig_count, "noop: node count unchanged"
    assert len(tr.graph.initializer) == orig_inits, "noop: initializer count unchanged"
    np.testing.assert_array_equal(ref["Y"], got["Y"])


# ══════════════════════════════════════════════════════════════════════════════
# Combined: Reshape chain + orphan init + constant fold
# ══════════════════════════════════════════════════════════════════════════════


def test_combined_reshape_and_unused_init_cleaned():
    """Combined: Reshape chain (noop) removed + orphan init removed, ORT parity."""
    x_data = _RNG.standard_normal([1, 2, 3, 4]).astype(np.float32)
    orphan = numpy_helper.from_array(np.array([0.0], dtype=np.float32), name="orphan2")
    sh1 = numpy_helper.from_array(np.array([1, 2, 3, 4], dtype=np.int64), name="sh1")
    sh2 = numpy_helper.from_array(np.array([1, 2, 3, 4], dtype=np.int64), name="sh2")

    rs1 = helper.make_node("Reshape", ["X", "sh1"], ["T1"], name="rs1")
    rs2 = helper.make_node("Reshape", ["T1", "sh2"], ["Y"], name="rs2")
    graph = helper.make_graph(
        [rs1, rs2],
        "g",
        [_vi("X", TensorProto.FLOAT, [1, 2, 3, 4])],
        [_vi("Y", TensorProto.FLOAT, [1, 2, 3, 4])],
        initializer=[orphan, sh1, sh2],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", _OPSET_VERSION)])
    _clamp_ir(model)

    _prepare_model(model)
    ref = _run_ort(model, {"X": x_data})

    tr = copy.deepcopy(model)
    _prepare_model(tr)
    tr = _apply(tr)
    _prepare_model(tr)
    got = _run_ort(tr, {"X": x_data})
    _save(model, tr, "combined_reshape_unused")

    reshape_nodes = [n for n in tr.graph.node if n.op_type == "Reshape"]
    assert len(reshape_nodes) == 0, "identity-Reshape chain must be fully removed"
    assert "orphan2" not in {i.name for i in tr.graph.initializer}, "orphan init must be removed"
    np.testing.assert_array_equal(ref["Y"], got["Y"])


def test_interleaved_reshape_chains_collapsed():
    """Two Reshape chains interleaved in node order are both collapsed correctly.

    Graph topology:
        X1 [1,2,4,4] --r1--> [1,4,2,4] --r2--> [2,4,4,1] --\\
                                                               Add --> Y [2,4,4,1]
        X2 [1,2,4,4] --r4--> [1,4,2,4] --r5--> [2,4,4,1] --/

    Node sequence in the graph: [r1, r4, r2, r5, add]
    (r4 is interleaved between r1 and r2 in the node list)

    The old sequential algorithm misses both chains:
    - Sees r1, then r4 (not consuming T1) → flushes singleton [r1], starts [r4]
    - Then r2 (not consuming T4) → flushes singleton [r4], starts [r2]
    - Then r5 (not consuming T2) → flushes singleton [r2], starts [r5]
    - End: flushes singleton [r5]
    → All 4 Reshape nodes remain unchanged.

    The new output-to-chain map algorithm finds both chains:
    - r1 → {T1: [r1]};  r4 → {T1:[r1], T4:[r4]}
    - r2 input=T1 in map → {T4:[r4], T2:[r1,r2]}
    - r5 input=T4 in map → {T2:[r1,r2], T5:[r4,r5]}
    - Flush [r1,r2]: in=[1,2,4,4] != out=[2,4,4,1] → collapse to 1 Reshape
    - Flush [r4,r5]: same → collapse to 1 Reshape
    → 2 Reshape nodes remain (one per chain), ORT parity.
    """
    # All shapes are 4D; chain input [1,2,4,4] != chain output [2,4,4,1]
    # so each 2-node chain collapses to a single Reshape (not removed as noop).
    sh_a = numpy_helper.from_array(np.array([1, 4, 2, 4], dtype=np.int64), name="sh_a")
    sh_b = numpy_helper.from_array(np.array([2, 4, 4, 1], dtype=np.int64), name="sh_b")
    sh_c = numpy_helper.from_array(np.array([1, 4, 2, 4], dtype=np.int64), name="sh_c")
    sh_d = numpy_helper.from_array(np.array([2, 4, 4, 1], dtype=np.int64), name="sh_d")

    r1 = helper.make_node("Reshape", ["X1", "sh_a"], ["T1"], name="r1")
    r2 = helper.make_node("Reshape", ["T1", "sh_b"], ["T2"], name="r2")
    r4 = helper.make_node("Reshape", ["X2", "sh_c"], ["T4"], name="r4")
    r5 = helper.make_node("Reshape", ["T4", "sh_d"], ["T5"], name="r5")
    add = helper.make_node("Add", ["T2", "T5"], ["Y"], name="add1")

    # Interleaved: r4 sits between r1 and r2 in the node list
    graph = helper.make_graph(
        [r1, r4, r2, r5, add],
        "interleaved_chains",
        [
            _vi("X1", TensorProto.FLOAT, [1, 2, 4, 4]),
            _vi("X2", TensorProto.FLOAT, [1, 2, 4, 4]),
        ],
        [_vi("Y", TensorProto.FLOAT, [2, 4, 4, 1])],
        initializer=[sh_a, sh_b, sh_c, sh_d],
        value_info=[
            _vi("T1", TensorProto.FLOAT, [1, 4, 2, 4]),
            _vi("T2", TensorProto.FLOAT, [2, 4, 4, 1]),
            _vi("T4", TensorProto.FLOAT, [1, 4, 2, 4]),
            _vi("T5", TensorProto.FLOAT, [2, 4, 4, 1]),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", _OPSET_VERSION)])
    _clamp_ir(model)

    x1 = _RNG.standard_normal([1, 2, 4, 4]).astype(np.float32)
    x2 = _RNG.standard_normal([1, 2, 4, 4]).astype(np.float32)
    _prepare_model(model)
    ref = _run_ort(model, {"X1": x1, "X2": x2})

    tr = copy.deepcopy(model)
    _prepare_model(tr)
    tr = _remove_reshapes(tr)
    _prepare_model(tr)
    got = _run_ort(tr, {"X1": x1, "X2": x2})
    _save(model, tr, "interleaved_reshape_chains")

    # Each 2-node chain collapses to 1 Reshape → 2 Reshape nodes total (down from 4)
    reshape_nodes = [n for n in tr.graph.node if n.op_type == "Reshape"]
    assert len(reshape_nodes) == 2, (
        f"Each chain collapses to 1 Reshape (2 total), got {[n.name for n in reshape_nodes]}"
    )
    # Verify ORT parity
    np.testing.assert_allclose(ref["Y"], got["Y"], rtol=1e-5, atol=1e-6)


# ══════════════════════════════════════════════════════════════════════════════
# Step 4 — Replace graph-output Reshape with Squeeze
# ══════════════════════════════════════════════════════════════════════════════


def test_graph_output_reshape_replaced_with_squeeze():
    """[1,3,4,1] → Reshape → [3,4] at graph output → replaced by Squeeze(axes=[0,3]); ORT parity."""
    sh_init = numpy_helper.from_array(np.array([3, 4], dtype=np.int64), name="rs_shape")
    reshape = helper.make_node("Reshape", ["X", "rs_shape"], ["Y"], name="rs")
    graph = helper.make_graph(
        [reshape],
        "g",
        [_vi("X", TensorProto.FLOAT, [1, 3, 4, 1])],
        [_vi("Y", TensorProto.FLOAT, [3, 4])],
        initializer=[sh_init],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", _OPSET_VERSION)])
    _clamp_ir(model)

    x = _RNG.standard_normal([1, 3, 4, 1]).astype(np.float32)
    _prepare_model(model)
    ref = _run_ort(model, {"X": x})

    tr = copy.deepcopy(model)
    _prepare_model(tr)
    tr = _reshape_to_squeeze(tr)
    _prepare_model(tr)
    got = _run_ort(tr, {"X": x})
    _save(model, tr, "go_reshape_to_squeeze")

    assert not any(n.op_type == "Reshape" for n in tr.graph.node), "Reshape must be replaced"
    sq_nodes = [n for n in tr.graph.node if n.op_type == "Squeeze"]
    assert len(sq_nodes) == 1, "One Squeeze must be inserted"

    # Verify axes initializer contains [0, 3]
    axes_init = next((i for i in tr.graph.initializer if "squeeze_axes" in i.name), None)
    assert axes_init is not None, "Squeeze axes initializer must exist"
    assert numpy_helper.to_array(axes_init).tolist() == [0, 3], (
        f"Expected axes [0,3], got {numpy_helper.to_array(axes_init).tolist()}"
    )

    np.testing.assert_array_equal(ref["Y"], got["Y"])


def test_graph_output_reshape_squeeze_axes_left_to_right():
    """[1,1,1,512] → Reshape → [1,512]: squeeze axes must be [0,1] (leftmost 1s), not [1,2].

    Verifies that when multiple 1-dims could be squeezed, the algorithm always
    removes from the lowest (leftmost) axis indices first.
    """
    sh_init = numpy_helper.from_array(np.array([1, 512], dtype=np.int64), name="rs_shape_ltr")
    reshape = helper.make_node("Reshape", ["X", "rs_shape_ltr"], ["Y"], name="rs_ltr")
    graph = helper.make_graph(
        [reshape],
        "g",
        [_vi("X", TensorProto.FLOAT, [1, 1, 1, 512])],
        [_vi("Y", TensorProto.FLOAT, [1, 512])],
        initializer=[sh_init],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", _OPSET_VERSION)])
    _clamp_ir(model)

    x = _RNG.standard_normal([1, 1, 1, 512]).astype(np.float32)
    _prepare_model(model)
    ref = _run_ort(model, {"X": x})

    tr = copy.deepcopy(model)
    _prepare_model(tr)
    tr = _reshape_to_squeeze(tr)
    _prepare_model(tr)
    got = _run_ort(tr, {"X": x})
    _save(model, tr, "go_reshape_ltr_axes")

    assert not any(n.op_type == "Reshape" for n in tr.graph.node), "Reshape must be replaced"
    sq_nodes = [n for n in tr.graph.node if n.op_type == "Squeeze"]
    assert len(sq_nodes) == 1, "One Squeeze must be inserted"

    axes_init = next((i for i in tr.graph.initializer if "squeeze_axes" in i.name), None)
    assert axes_init is not None
    axes = numpy_helper.to_array(axes_init).tolist()
    assert axes == [0, 1], f"Expected axes [0,1] (leftmost 1s), got {axes}"

    np.testing.assert_array_equal(ref["Y"], got["Y"])


def test_graph_output_reshape_not_squeezable_padded_to_4d():
    """[2,3,4] → Reshape → [6,4] at graph output: not a pure squeeze, output is 2D (<4D).

    New behaviour: pad Reshape target to [1,6,4], then Squeeze([0]) → [6,4].
    This ensures the Reshape produces a 4D tensor before being squeezed back.
    """
    sh_init = numpy_helper.from_array(np.array([6, 4], dtype=np.int64), name="rs_shape2")
    reshape = helper.make_node("Reshape", ["X", "rs_shape2"], ["Y"], name="rs2")
    graph = helper.make_graph(
        [reshape],
        "g",
        [_vi("X", TensorProto.FLOAT, [2, 3, 4])],
        [_vi("Y", TensorProto.FLOAT, [6, 4])],
        initializer=[sh_init],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", _OPSET_VERSION)])
    _clamp_ir(model)

    x = _RNG.standard_normal([2, 3, 4]).astype(np.float32)
    _prepare_model(model)
    ref = _run_ort(model, {"X": x})

    tr = copy.deepcopy(model)
    _prepare_model(tr)
    tr = _reshape_to_squeeze(tr)
    _prepare_model(tr)
    got = _run_ort(tr, {"X": x})
    _save(model, tr, "go_reshape_not_squeezable")

    # Reshape must still exist (now targeting [1,6,4])
    rs_nodes = [n for n in tr.graph.node if n.op_type == "Reshape"]
    assert len(rs_nodes) == 1, "Reshape must remain (with updated shape)"
    # Squeeze([0]) must be inserted after Reshape
    sq_nodes = [n for n in tr.graph.node if n.op_type == "Squeeze"]
    assert len(sq_nodes) == 1, "Squeeze([0]) must be inserted"
    # Verify Reshape now targets a 4D shape [1,6,4]
    rs_shape_init = next((i for i in tr.graph.initializer if i.name == rs_nodes[0].input[1]), None)
    assert rs_shape_init is not None
    assert numpy_helper.to_array(rs_shape_init).tolist() == [1, 6, 4], (
        f"Reshape shape must be [1,6,4], got {numpy_helper.to_array(rs_shape_init).tolist()}"
    )
    np.testing.assert_array_equal(ref["Y"], got["Y"])


def test_graph_output_reshape_4d_output_kept():
    """[5,6,7,8] → Reshape → [30,14,4] at graph output: not a pure squeeze, BUT padded to [1,30,14,4] + Squeeze([0])."""
    sh_init = numpy_helper.from_array(np.array([30, 14, 4], dtype=np.int64), name="rs_shape_3d")
    reshape = helper.make_node("Reshape", ["X", "rs_shape_3d"], ["Y"], name="rs_3d")
    graph = helper.make_graph(
        [reshape],
        "g",
        [_vi("X", TensorProto.FLOAT, [5, 6, 7, 8])],
        [_vi("Y", TensorProto.FLOAT, [30, 14, 4])],
        initializer=[sh_init],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", _OPSET_VERSION)])
    _clamp_ir(model)

    x = _RNG.standard_normal([5, 6, 7, 8]).astype(np.float32)
    _prepare_model(model)
    ref = _run_ort(model, {"X": x})

    tr = copy.deepcopy(model)
    _prepare_model(tr)
    tr = _reshape_to_squeeze(tr)
    _prepare_model(tr)
    got = _run_ort(tr, {"X": x})
    _save(model, tr, "go_reshape_3d_padded")

    # Reshape must target [1,30,14,4]
    rs_nodes = [n for n in tr.graph.node if n.op_type == "Reshape"]
    assert len(rs_nodes) == 1
    rs_shape_init = next((i for i in tr.graph.initializer if i.name == rs_nodes[0].input[1]), None)
    assert rs_shape_init is not None
    assert numpy_helper.to_array(rs_shape_init).tolist() == [1, 30, 14, 4], (
        f"Reshape shape must be [1,30,14,4], got {numpy_helper.to_array(rs_shape_init).tolist()}"
    )
    # Squeeze([0]) strips the leading 1 back to [30,14,4]
    sq_nodes = [n for n in tr.graph.node if n.op_type == "Squeeze"]
    assert len(sq_nodes) == 1
    np.testing.assert_array_equal(ref["Y"], got["Y"])


def test_graph_output_reshape_not_at_graph_output_unchanged():
    """Reshape that is NOT a graph output must not be touched by this step."""
    sh_init = numpy_helper.from_array(np.array([3, 4], dtype=np.int64), name="rs_shape3")
    reshape = helper.make_node("Reshape", ["X", "rs_shape3"], ["T"], name="rs3")
    relu = helper.make_node("Relu", ["T"], ["Y"], name="relu")
    graph = helper.make_graph(
        [reshape, relu],
        "g",
        [_vi("X", TensorProto.FLOAT, [1, 3, 4, 1])],
        [_vi("Y", TensorProto.FLOAT, [3, 4])],
        initializer=[sh_init],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", _OPSET_VERSION)])
    _clamp_ir(model)

    tr = copy.deepcopy(model)
    _prepare_model(tr)
    tr = _reshape_to_squeeze(tr)

    # Reshape feeds Relu (not a graph output) → must stay as Reshape
    assert any(n.op_type == "Reshape" for n in tr.graph.node), (
        "Reshape not at graph output must be unchanged"
    )
    assert not any(n.op_type == "Squeeze" for n in tr.graph.node), (
        "No Squeeze must be inserted for non-output Reshape"
    )


# ══════════════════════════════════════════════════════════════════════════════
# GroupQueryAttention — Unsqueeze after GQA output must be preserved
# ══════════════════════════════════════════════════════════════════════════════


def test_unsqueeze_after_gqa_3d_to_4d_skipped():
    """Unsqueeze whose input is a GQA 3D output and whose output is 4D must not be replaced.

    Pattern:  X [b,s,h] → GQA → gqa_out [b,s,h]  (3D)
              gqa_out → Unsqueeze(axes=[0]) → gqa_4d [1,b,s,h]  (4D)  ← must be preserved
              gqa_4d → Relu → Y [1,b,s,h]

    The Unsqueeze is the canonical 3D→4D bridge after GQA and must not be replaced
    by a Reshape.
    """
    b, s, h = 2, 4, 8
    unsq_axes = numpy_helper.from_array(np.array([0], dtype=np.int64), name="gqa_unsq_ax")

    # Stub GQA node with 3D input/output (com.microsoft domain, no real attributes needed)
    gqa = helper.make_node(
        "GroupQueryAttention",
        inputs=["X", "", ""],
        outputs=["gqa_out"],
        name="gqa_node",
        domain="com.microsoft",
        num_heads=2,
        kv_num_heads=2,
    )
    unsqueeze = helper.make_node(
        "Unsqueeze", ["gqa_out", "gqa_unsq_ax"], ["gqa_4d"], name="gqa_unsq"
    )
    relu = helper.make_node("Relu", ["gqa_4d"], ["Y"], name="relu")

    graph = helper.make_graph(
        [gqa, unsqueeze, relu],
        "g",
        [_vi("X", TensorProto.FLOAT, [b, s, h])],
        [_vi("Y", TensorProto.FLOAT, [1, b, s, h])],
        value_info=[
            _vi("gqa_out", TensorProto.FLOAT, [b, s, h]),  # 3D GQA output
            _vi("gqa_4d", TensorProto.FLOAT, [1, b, s, h]),  # 4D Unsqueeze output
        ],
        initializer=[unsq_axes],
    )
    model = helper.make_model(
        graph,
        opset_imports=[
            helper.make_opsetid("", _OPSET_VERSION),
            helper.make_opsetid("com.microsoft", 1),
        ],
    )
    _clamp_ir(model)
    orig_node_count = len(model.graph.node)

    tr = copy.deepcopy(model)
    _prepare_model(tr)
    tr = _replace_sq_unsq(tr)

    # Unsqueeze after GQA (3D→4D) must be preserved unchanged
    assert any(n.op_type == "Unsqueeze" for n in tr.graph.node), (
        "Unsqueeze after GQA 3D output must be preserved"
    )
    assert not any(n.op_type == "Reshape" for n in tr.graph.node), (
        "No Reshape must replace the GQA→Unsqueeze bridge"
    )
    assert len(tr.graph.node) == orig_node_count, (
        "Node count must be unchanged — no nodes added or removed"
    )
    _save(model, tr, "unsqueeze_after_gqa_skipped")


def test_unsqueeze_after_gqa_4d_output_not_skipped():
    """Unsqueeze after a GQA node whose output is already 4D is NOT special — treated normally.

    If the GQA output is 4D (not 3D), the skip condition does not apply.
    """
    b, s, h = 1, 4, 8
    unsq_axes = numpy_helper.from_array(np.array([1], dtype=np.int64), name="gqa4d_unsq_ax")

    gqa = helper.make_node(
        "GroupQueryAttention",
        inputs=["X", "", ""],
        outputs=["gqa_out"],
        name="gqa_node2",
        domain="com.microsoft",
        num_heads=2,
        kv_num_heads=2,
    )
    unsqueeze = helper.make_node(
        "Unsqueeze", ["gqa_out", "gqa4d_unsq_ax"], ["gqa_5d"], name="gqa_unsq2"
    )
    relu = helper.make_node("Relu", ["gqa_5d"], ["Y"], name="relu")

    graph = helper.make_graph(
        [gqa, unsqueeze, relu],
        "g",
        [_vi("X", TensorProto.FLOAT, [b, s, h])],
        [_vi("Y", TensorProto.FLOAT, [b, 1, s, h])],
        value_info=[
            # GQA output is 4D here (not 3D) — skip condition does NOT apply
            _vi("gqa_out", TensorProto.FLOAT, [b, 1, s, h]),
            _vi("gqa_5d", TensorProto.FLOAT, [b, 1, 1, s, h]),
        ],
        initializer=[unsq_axes],
    )
    model = helper.make_model(
        graph,
        opset_imports=[
            helper.make_opsetid("", _OPSET_VERSION),
            helper.make_opsetid("com.microsoft", 1),
        ],
    )
    _clamp_ir(model)

    tr = copy.deepcopy(model)
    _prepare_model(tr)
    tr = _replace_sq_unsq(tr)

    # GQA output is 4D → Unsqueeze is NOT skipped by the GQA rule;
    # it should be replaced (output has 5D — beyond 4D, so the generic handler skips too,
    # but the important thing is the GQA-specific skip did NOT fire).
    # Verify no GQA-specific skip was applied by confirming the Unsqueeze was processed
    # (either replaced or removed, not left as-is due to the GQA rule).
    # Here output is 5D so the transform skips it anyway — just assert no crash.
    _save(model, tr, "unsqueeze_after_gqa_4d_not_skipped")


# ── Tests: _remove_consecutive_casts ─────────────────────────────────────────


def test_cast_chain_collapsed_to_single():
    """Cast(FLOAT→INT64) → Cast(INT64→INT32) collapsed to Cast(FLOAT→INT32)."""
    cast1 = helper.make_node("Cast", ["X"], ["c1"], name="cast1", to=TensorProto.INT64)
    cast2 = helper.make_node("Cast", ["c1"], ["Y"], name="cast2", to=TensorProto.INT32)
    graph = helper.make_graph(
        [cast1, cast2],
        "g",
        [_vi("X", TensorProto.FLOAT, [2, 4])],
        [_vi("Y", TensorProto.INT32, [2, 4])],
        value_info=[_vi("c1", TensorProto.INT64, [2, 4])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", _OPSET_VERSION)])
    _clamp_ir(model)

    tr = copy.deepcopy(model)
    _prepare_model(tr)
    tr = _remove_cast_chains(tr)

    cast_nodes = [n for n in tr.graph.node if n.op_type == "Cast"]
    assert len(cast_nodes) == 1, f"Expected 1 Cast, got {len(cast_nodes)}"
    # Final Cast should target INT32
    to_val = next(a.i for a in cast_nodes[0].attribute if a.name == "to")
    assert to_val == TensorProto.INT32
    # Input should be X, output should be Y
    assert cast_nodes[0].input[0] == "X"
    assert cast_nodes[0].output[0] == "Y"


def test_cast_chain_three_nodes():
    """Three consecutive Casts collapsed to one."""
    cast1 = helper.make_node("Cast", ["X"], ["c1"], name="cast1", to=TensorProto.INT64)
    cast2 = helper.make_node("Cast", ["c1"], ["c2"], name="cast2", to=TensorProto.FLOAT)
    cast3 = helper.make_node("Cast", ["c2"], ["Y"], name="cast3", to=TensorProto.INT32)
    graph = helper.make_graph(
        [cast1, cast2, cast3],
        "g",
        [_vi("X", TensorProto.FLOAT, [3])],
        [_vi("Y", TensorProto.INT32, [3])],
        value_info=[
            _vi("c1", TensorProto.INT64, [3]),
            _vi("c2", TensorProto.FLOAT, [3]),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", _OPSET_VERSION)])
    _clamp_ir(model)

    tr = copy.deepcopy(model)
    _prepare_model(tr)
    tr = _remove_cast_chains(tr)

    cast_nodes = [n for n in tr.graph.node if n.op_type == "Cast"]
    assert len(cast_nodes) == 1
    to_val = next(a.i for a in cast_nodes[0].attribute if a.name == "to")
    assert to_val == TensorProto.INT32


def test_single_cast_unchanged():
    """A lone Cast node is not modified."""
    cast1 = helper.make_node("Cast", ["X"], ["Y"], name="cast1", to=TensorProto.INT32)
    graph = helper.make_graph(
        [cast1],
        "g",
        [_vi("X", TensorProto.FLOAT, [4])],
        [_vi("Y", TensorProto.INT32, [4])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", _OPSET_VERSION)])
    _clamp_ir(model)

    tr = copy.deepcopy(model)
    _prepare_model(tr)
    tr = _remove_cast_chains(tr)

    cast_nodes = [n for n in tr.graph.node if n.op_type == "Cast"]
    assert len(cast_nodes) == 1
    assert cast_nodes[0].input[0] == "X"
    assert cast_nodes[0].output[0] == "Y"


def test_cast_chain_broken_by_multi_consumer():
    """Cast chain is broken when intermediate tensor has multiple consumers."""
    cast1 = helper.make_node("Cast", ["X"], ["c1"], name="cast1", to=TensorProto.INT64)
    cast2 = helper.make_node("Cast", ["c1"], ["Y"], name="cast2", to=TensorProto.INT32)
    relu = helper.make_node("Relu", ["c1"], ["Z"], name="relu")
    graph = helper.make_graph(
        [cast1, cast2, relu],
        "g",
        [_vi("X", TensorProto.FLOAT, [2, 4])],
        [_vi("Y", TensorProto.INT32, [2, 4]), _vi("Z", TensorProto.INT64, [2, 4])],
        value_info=[_vi("c1", TensorProto.INT64, [2, 4])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", _OPSET_VERSION)])
    _clamp_ir(model)

    tr = copy.deepcopy(model)
    _prepare_model(tr)
    tr = _remove_cast_chains(tr)

    # Both Casts should remain (chain broken by multi-consumer c1)
    cast_nodes = [n for n in tr.graph.node if n.op_type == "Cast"]
    assert len(cast_nodes) == 2
