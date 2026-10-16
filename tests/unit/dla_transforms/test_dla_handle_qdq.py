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
# Unit tests for :mod:`modelopt.onnx.graph_surgery.dla_transforms.dla_handle_qdq`.

"""ORT CPU parity tests for ``dla_handle_qdq``.

Transform rule
--------------
Every ``QuantizeLinear`` and ``DequantizeLinear`` node whose data input is
non-4D gets wrapped independently:

    data [*orig_shape]
    → Unsqueeze  → data_4d  [1,…,1, *orig_shape]
    → Q / DQ     → out_4d   [1,…,1, *orig_shape]
    → Squeeze    → out      [*orig_shape]

Scale / zero-point inputs are untouched.  The ``axis`` attribute is shifted
by ``4 - orig_rank``.  Nodes whose data input is already 4-D are skipped.

Coverage
--------
QuantizeLinear  3-D input (dynamic) → wrapped, axis adjusted, ORT parity
DequantizeLinear 3-D input (static init) → wrapped, axis adjusted, ORT parity
DequantizeLinear 2-D input (dynamic) → wrapped, ORT parity
Q already 4-D → skipped (no Unsqueeze/Squeeze inserted)
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
_SAVED_MODELS_DIR = _REPO_ROOT / "scratch_space" / "handle_qdq_test_models"
_OPSET_VERSION = 21  # INT4/UINT4 require ≥ opset 21
_MAX_IR_VERSION_FOR_ORT = 9


# ── Dynamic import ────────────────────────────────────────────────────────────


def _load_module():
    mod_key = "modelopt.onnx.graph_surgery.dla_transforms.dla_handle_qdq"
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
        _REPO_ROOT / "modelopt/onnx/graph_surgery/dla_transforms/dla_handle_qdq.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_key] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


_mod = _load_module()
_apply = _mod._apply_handle_qdq


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


def _assert_wrapped(tr, op_type: str, orig_name: str) -> None:
    """Check Unsqueeze→op→Squeeze chain is present and op has the right name."""
    assert any(n.op_type == "Unsqueeze" for n in tr.graph.node), (
        f"Unsqueeze must be inserted before {op_type}"
    )
    assert any(n.op_type == "Squeeze" for n in tr.graph.node), (
        f"Squeeze must be inserted after {op_type}"
    )
    assert any(n.op_type == op_type for n in tr.graph.node), f"{op_type} node must still be present"


_RNG = np.random.default_rng(2024)


# ── 1. QuantizeLinear — 3-D dynamic input, per-channel axis ──────────────────


def test_quantize_linear_3d_dynamic_wrapped():
    """Q [2,4,8] axis=1 → Unsqueeze→Q(axis=2)→Squeeze; output shape and values match base."""
    shape = [2, 4, 8]
    scale = numpy_helper.from_array(np.full((4,), 0.1, dtype=np.float32), name="scale")
    zp = numpy_helper.from_array(np.zeros((4,), dtype=np.int8), name="zp")

    q = helper.make_node("QuantizeLinear", ["X", "scale", "zp"], ["Y"], name="q", axis=1)
    graph = helper.make_graph(
        [q],
        "g",
        [_vi("X", TensorProto.FLOAT, shape)],
        [_vi("Y", TensorProto.INT8, shape)],
        initializer=[scale, zp],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", _OPSET_VERSION)])
    _clamp_ir(model)

    x = _RNG.standard_normal(shape).astype(np.float32)
    _prepare_model(model)
    ref = _run_ort(model, {"X": x})

    tr = copy.deepcopy(model)
    _prepare_model(tr)
    tr = _apply(tr)
    _prepare_model(tr)
    got = _run_ort(tr, {"X": x})
    _save(model, tr, "q_3d_dynamic")

    _assert_wrapped(tr, "QuantizeLinear", "q")

    # axis must have been shifted from 1 to 2 (4-3=1 delta)
    q_node = next(n for n in tr.graph.node if n.op_type == "QuantizeLinear")
    axis_attr = next((a for a in q_node.attribute if a.name == "axis"), None)
    assert axis_attr is not None and axis_attr.i == 2, (
        f"axis must be shifted to 2, got {axis_attr.i if axis_attr else 'None'}"
    )

    # scale and zp must be untouched (still 1-D)
    s_init = next(i for i in tr.graph.initializer if i.name == "scale")
    z_init = next(i for i in tr.graph.initializer if i.name == "zp")
    assert list(s_init.dims) == [4], f"scale must stay 1-D [4], got {list(s_init.dims)}"
    assert list(z_init.dims) == [4], f"zp must stay 1-D [4], got {list(z_init.dims)}"

    np.testing.assert_array_equal(ref["Y"], got["Y"])


# ── 2. DequantizeLinear — 3-D static initializer data input, per-channel ─────


def test_dequantize_linear_3d_static_init_wrapped():
    """DQ with static int8 initializer [2,4,8] axis=1 → wrapped; float output matches base."""
    shape = [2, 4, 8]
    data = numpy_helper.from_array((_RNG.integers(-5, 6, shape)).astype(np.int8), name="data")
    scale = numpy_helper.from_array(np.full((4,), 0.1, dtype=np.float32), name="dq_scale")
    zp = numpy_helper.from_array(np.zeros((4,), dtype=np.int8), name="dq_zp")

    dq = helper.make_node(
        "DequantizeLinear", ["data", "dq_scale", "dq_zp"], ["Y"], name="dq", axis=1
    )
    graph = helper.make_graph(
        [dq],
        "g",
        [],
        [_vi("Y", TensorProto.FLOAT, shape)],
        initializer=[data, scale, zp],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", _OPSET_VERSION)])
    _clamp_ir(model)

    _prepare_model(model)
    ref = _run_ort(model, {})

    tr = copy.deepcopy(model)
    _prepare_model(tr)
    tr = _apply(tr)
    _prepare_model(tr)
    got = _run_ort(tr, {})
    _save(model, tr, "dq_3d_static")

    _assert_wrapped(tr, "DequantizeLinear", "dq")

    dq_node = next(n for n in tr.graph.node if n.op_type == "DequantizeLinear")
    axis_attr = next((a for a in dq_node.attribute if a.name == "axis"), None)
    assert axis_attr is not None and axis_attr.i == 2, (
        f"axis must be shifted to 2, got {axis_attr.i if axis_attr else 'None'}"
    )

    np.testing.assert_allclose(ref["Y"], got["Y"], rtol=1e-5, atol=1e-6)


# ── 3. DequantizeLinear — 2-D dynamic input ───────────────────────────────────


def test_dequantize_linear_2d_dynamic_wrapped():
    """DQ [4,8] dynamic input (per-tensor, no axis) → Unsqueeze→DQ→Squeeze; ORT parity."""
    shape = [4, 8]
    scale = numpy_helper.from_array(np.array(0.1, dtype=np.float32), name="s2")
    zp = numpy_helper.from_array(np.array(0, dtype=np.int8), name="z2")

    dq = helper.make_node("DequantizeLinear", ["X", "s2", "z2"], ["Y"], name="dq2")
    graph = helper.make_graph(
        [dq],
        "g",
        [_vi("X", TensorProto.INT8, shape)],
        [_vi("Y", TensorProto.FLOAT, shape)],
        initializer=[scale, zp],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", _OPSET_VERSION)])
    _clamp_ir(model)

    x = _RNG.integers(-5, 6, shape).astype(np.int8)
    _prepare_model(model)
    ref = _run_ort(model, {"X": x})

    tr = copy.deepcopy(model)
    _prepare_model(tr)
    tr = _apply(tr)
    _prepare_model(tr)
    got = _run_ort(tr, {"X": x})
    _save(model, tr, "dq_2d_dynamic")

    _assert_wrapped(tr, "DequantizeLinear", "dq2")
    np.testing.assert_allclose(ref["Y"], got["Y"], rtol=1e-5, atol=1e-6)


# ── 4. QuantizeLinear already 4-D → skipped ──────────────────────────────────


def test_quantize_linear_4d_skipped():
    """Q with 4-D input is already 4-D → no Unsqueeze/Squeeze inserted, graph unchanged."""
    shape = [1, 2, 4, 8]
    scale = numpy_helper.from_array(np.full((2,), 0.1, dtype=np.float32), name="s4d")
    zp = numpy_helper.from_array(np.zeros((2,), dtype=np.int8), name="z4d")

    q = helper.make_node("QuantizeLinear", ["X", "s4d", "z4d"], ["Y"], name="q4d", axis=1)
    graph = helper.make_graph(
        [q],
        "g",
        [_vi("X", TensorProto.FLOAT, shape)],
        [_vi("Y", TensorProto.INT8, shape)],
        initializer=[scale, zp],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", _OPSET_VERSION)])
    _clamp_ir(model)
    orig_count = len(model.graph.node)

    x = _RNG.standard_normal(shape).astype(np.float32)
    _prepare_model(model)
    ref = _run_ort(model, {"X": x})

    tr = copy.deepcopy(model)
    _prepare_model(tr)
    tr = _apply(tr)
    _prepare_model(tr)
    got = _run_ort(tr, {"X": x})
    _save(model, tr, "q_4d_skipped")

    assert len(tr.graph.node) == orig_count, "4-D input: no nodes should be added"
    assert not any(n.op_type == "Unsqueeze" for n in tr.graph.node)
    assert not any(n.op_type == "Squeeze" for n in tr.graph.node)
    np.testing.assert_array_equal(ref["Y"], got["Y"])
