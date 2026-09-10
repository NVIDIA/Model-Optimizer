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
# Unit tests for :mod:`modelopt.onnx.graph_surgery.dla_transforms.dla_greater`.

"""ORT CPU parity tests for ``dla_greater``.

Transform rule
--------------
For every ``Greater`` node, any initializer input whose dtype is INT32 or
INT64 is replaced by a new float32 initializer (name ``{original}_cast``).
Float32 / other-dtype initializers and dynamic (non-initializer) inputs are
left untouched.

Coverage
--------
INT32 initializer threshold  → cast to float32, ORT parity
INT64 initializer threshold  → cast to float32, ORT parity
float32 initializer          → unchanged (no cast)
both inputs dynamic          → graph unchanged, ORT parity
both inputs INT32 initializers → both cast to float32
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
_SAVED_MODELS_DIR = _REPO_ROOT / "scratch_space" / "greater_test_models"
_OPSET_VERSION = 17
_MAX_IR_VERSION_FOR_ORT = 9


# ── Dynamic import ────────────────────────────────────────────────────────────


def _load_module():
    mod_key = "modelopt.onnx.graph_surgery.dla_transforms.dla_greater"
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
        _REPO_ROOT / "modelopt/onnx/graph_surgery/dla_transforms/dla_greater.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_key] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


_mod = _load_module()
_apply = _mod._apply_greater


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


def _init_dtype(model: onnx.ModelProto, name: str) -> int | None:
    for init in model.graph.initializer:
        if init.name == name:
            return init.data_type
    return None


_RNG = np.random.default_rng(2024)


# ── 1. INT32 threshold initializer → cast to float32 ─────────────────────────


def test_greater_int32_init_converted():
    """Greater(X_int32, int32_threshold) → threshold becomes float32; ORT parity.

    Both inputs are INT32 so the base model is valid for ORT.
    After the transform the threshold becomes float32; X's declared type is updated
    to float32 in the test (reflecting that upstream transforms will produce float32).
    """
    shape = [3, 4]
    threshold = np.array(5, dtype=np.int32)
    thresh_init = numpy_helper.from_array(threshold, name="thresh")

    greater = helper.make_node("Greater", ["X", "thresh"], ["Y"], name="greater")
    graph = helper.make_graph(
        [greater],
        "greater_i32",
        [_vi("X", TensorProto.INT32, shape)],
        [_vi("Y", TensorProto.BOOL, shape)],
        initializer=[thresh_init],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", _OPSET_VERSION)])
    _clamp_ir(model)

    x = (_RNG.standard_normal(shape) * 10).astype(np.int32)
    _prepare_model(model)
    ref = _run_ort(model, {"X": x})

    tr = copy.deepcopy(model)
    _prepare_model(tr)
    tr = _apply(tr)
    _prepare_model(tr)
    got = _run_ort(tr, {"X": x})
    _save(model, tr, "greater_i32")

    assert _init_dtype(tr, "thresh_cast") == TensorProto.FLOAT, "threshold must become float32"
    # original INT32 init is left for _remove_unused_initializers to clean up
    np.testing.assert_array_equal(ref["Y"], got["Y"])


# ── 2. INT64 threshold initializer → cast to float32 ─────────────────────────


def test_greater_int64_init_converted():
    """Greater(X_int64, int64_threshold) → threshold becomes float32; ORT parity."""
    shape = [2, 6]
    threshold = np.array(-3, dtype=np.int64)
    thresh_init = numpy_helper.from_array(threshold, name="thresh64")

    greater = helper.make_node("Greater", ["X", "thresh64"], ["Y"], name="greater")
    graph = helper.make_graph(
        [greater],
        "greater_i64",
        [_vi("X", TensorProto.INT64, shape)],
        [_vi("Y", TensorProto.BOOL, shape)],
        initializer=[thresh_init],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", _OPSET_VERSION)])
    _clamp_ir(model)

    x = (_RNG.standard_normal(shape) * 5).astype(np.int64)
    _prepare_model(model)
    ref = _run_ort(model, {"X": x})

    tr = copy.deepcopy(model)
    _prepare_model(tr)
    tr = _apply(tr)
    _prepare_model(tr)
    got = _run_ort(tr, {"X": x})
    _save(model, tr, "greater_i64")

    assert _init_dtype(tr, "thresh64_cast") == TensorProto.FLOAT
    np.testing.assert_array_equal(ref["Y"], got["Y"])


# ── 3. float32 initializer → unchanged ───────────────────────────────────────


def test_greater_float32_init_unchanged():
    """Greater(X_float32, float32_threshold) — already float32, no change at all."""
    shape = [2, 4]
    threshold = np.array(0.5, dtype=np.float32)
    thresh_init = numpy_helper.from_array(threshold, name="thresh_f32")

    greater = helper.make_node("Greater", ["X", "thresh_f32"], ["Y"], name="greater")
    graph = helper.make_graph(
        [greater],
        "greater_f32",
        [_vi("X", TensorProto.FLOAT, shape)],
        [_vi("Y", TensorProto.BOOL, shape)],
        initializer=[thresh_init],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", _OPSET_VERSION)])
    _clamp_ir(model)
    original_node_count = len(model.graph.node)
    original_init_count = len(model.graph.initializer)

    x = _RNG.standard_normal(shape).astype(np.float32)
    _prepare_model(model)
    ref = _run_ort(model, {"X": x})

    tr = copy.deepcopy(model)
    _prepare_model(tr)
    tr = _apply(tr)
    _prepare_model(tr)
    got = _run_ort(tr, {"X": x})
    _save(model, tr, "greater_f32")

    # Both inputs already float32 — no Cast inserted, no new initializer, node inputs unchanged
    assert len(tr.graph.node) == original_node_count
    assert len(tr.graph.initializer) == original_init_count
    greater_node = next(n for n in tr.graph.node if n.op_type == "Greater")
    assert greater_node.input[1] == "thresh_f32"
    np.testing.assert_array_equal(ref["Y"], got["Y"])


# ── 4. Both inputs dynamic → graph unchanged ─────────────────────────────────


def test_greater_dynamic_float32_inputs_noop():
    """Greater(X_float32, Z_float32) — both already float32, no Cast inserted."""
    shape = [3, 4]
    greater = helper.make_node("Greater", ["X", "Z"], ["Y"], name="greater")
    graph = helper.make_graph(
        [greater],
        "greater_dynamic",
        [_vi("X", TensorProto.FLOAT, shape), _vi("Z", TensorProto.FLOAT, shape)],
        [_vi("Y", TensorProto.BOOL, shape)],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", _OPSET_VERSION)])
    _clamp_ir(model)
    original_node_count = len(model.graph.node)

    x = _RNG.standard_normal(shape).astype(np.float32) * 5
    z = _RNG.standard_normal(shape).astype(np.float32) * 5
    _prepare_model(model)
    ref = _run_ort(model, {"X": x, "Z": z})

    tr = copy.deepcopy(model)
    _prepare_model(tr)
    tr = _apply(tr)
    _prepare_model(tr)
    got = _run_ort(tr, {"X": x, "Z": z})
    _save(model, tr, "greater_dynamic")

    # Both inputs are float32 — shape inference informs transform to skip them
    assert len(tr.graph.node) == original_node_count, (
        "no Cast nodes should be inserted for float32 inputs"
    )
    np.testing.assert_array_equal(ref["Y"], got["Y"])


# ── 5. Both inputs are INT32 initializers → both cast ────────────────────────


def test_greater_both_inputs_int32_converted():
    """Greater(int32_A, int32_B) — both initializers cast to float32."""
    a_init = numpy_helper.from_array(np.array([1, 2, 3, 4], dtype=np.int32), name="A")
    b_init = numpy_helper.from_array(np.array([4, 3, 2, 1], dtype=np.int32), name="B")

    greater = helper.make_node("Greater", ["A", "B"], ["Y"], name="greater")
    graph = helper.make_graph(
        [greater],
        "greater_both_int",
        [],
        [_vi("Y", TensorProto.BOOL, [4])],
        initializer=[a_init, b_init],
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
    _save(model, tr, "greater_both_int")

    assert _init_dtype(tr, "A_cast") == TensorProto.FLOAT, "A must be cast to float32"
    assert _init_dtype(tr, "B_cast") == TensorProto.FLOAT, "B must be cast to float32"
    np.testing.assert_array_equal(ref["Y"], got["Y"])
