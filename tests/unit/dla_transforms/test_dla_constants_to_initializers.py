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
# Unit tests for :mod:`modelopt.onnx.graph_surgery.dla_transforms.dla_constants_to_initializers`.

"""ORT CPU parity tests for ``constants_to_initializers`` / ``_constants_to_initializers_from_model``.

Transform rule
--------------
Every ``Constant`` node is removed and replaced by a graph initializer carrying
the same tensor data.  Downstream consumers are unaffected because they
reference the tensor by name, which is preserved.

Supported Constant payload attributes
--------------------------------------
``value``          — dense TensorProto (any dtype / any shape)
``value_float``    — scalar float32
``value_floats``   — 1-D float32 list
``value_int``      — scalar int64
``value_ints``     — 1-D int64 list

Coverage
--------
value (float32 tensor)              → node removed, initializer added, ORT parity
value_float (scalar)                → node removed, initializer added, ORT parity
value_floats (1-D list)             → node removed, initializer added, ORT parity
value_int  (scalar)                 → node removed, initializer added
value_ints (1-D list)               → node removed, initializer added
no Constant nodes                   → graph unchanged, ORT output unchanged
multiple Constants                  → all converted in one pass, ORT parity
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
_SAVED_MODELS_DIR = _REPO_ROOT / "scratch_space" / "constants_to_initializers_test_models"
_OPSET_VERSION = 17
_MAX_IR_VERSION_FOR_ORT = 9


# ── Dynamic import ────────────────────────────────────────────────────────────


def _load_module():
    mod_key = "modelopt.onnx.graph_surgery.dla_transforms.dla_constants_to_initializers"
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
        _REPO_ROOT / "modelopt/onnx/graph_surgery/dla_transforms/dla_constants_to_initializers.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_key] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


_mod = _load_module()
_apply = _mod._constants_to_initializers_from_model


# ── Shared utilities ──────────────────────────────────────────────────────────


def _vi(name: str, elem_type: int, shape: list) -> onnx.ValueInfoProto:
    return helper.make_tensor_value_info(name, elem_type, shape)


def _clamp_ir(model: onnx.ModelProto) -> None:
    model.ir_version = min(model.ir_version, _MAX_IR_VERSION_FOR_ORT)


def _prepare_model(model: onnx.ModelProto) -> None:
    """Run shape inference in-place to populate value_info."""
    try:
        from onnxruntime.tools.symbolic_shape_infer import (
            SymbolicShapeInference,  # type: ignore[import-not-found]
        )

        inferred = SymbolicShapeInference.infer_shapes(model)
        if inferred is not None:
            del model.graph.value_info[:]
            model.graph.value_info.extend(inferred.graph.value_info)
    except Exception:
        pass


def _run_ort(model: onnx.ModelProto, feeds: dict) -> dict:
    import onnxruntime as ort  # type: ignore[import-not-found]

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


def _initializer_names(model: onnx.ModelProto) -> set[str]:
    return {i.name for i in model.graph.initializer}


def _constant_node_names(model: onnx.ModelProto) -> set[str]:
    return {n.output[0] for n in model.graph.node if n.op_type == "Constant"}


_RNG = np.random.default_rng(2024)


# ── 1. value attribute (dense float32 tensor) ────────────────────────────────


def test_constant_value_tensor_converted():
    """Constant with 'value' (TensorProto) replaced by initializer; ORT output unchanged."""
    shape = [2, 4]
    const_data = _RNG.standard_normal(shape).astype(np.float32)
    const_init = numpy_helper.from_array(const_data, name="const_w")
    const_node = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["const_w"],
        name="c_w",
        value=const_init,
    )
    # Add const_w to X element-wise
    add_node = helper.make_node("Add", ["X", "const_w"], ["Y"], name="add")

    graph = helper.make_graph(
        [const_node, add_node],
        "const_value",
        [_vi("X", TensorProto.FLOAT, shape)],
        [_vi("Y", TensorProto.FLOAT, shape)],
        value_info=[_vi("const_w", TensorProto.FLOAT, shape)],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", _OPSET_VERSION)])
    _clamp_ir(model)

    x = _RNG.standard_normal(shape).astype(np.float32)
    ref = _run_ort(model, {"X": x})

    tr = copy.deepcopy(model)
    _prepare_model(tr)
    cnt = _apply(tr, verbose=False)
    got = _run_ort(tr, {"X": x})
    _save(model, tr, "const_value_tensor")

    assert cnt == 1
    assert "const_w" in _initializer_names(tr)
    assert "const_w" not in _constant_node_names(tr)
    np.testing.assert_allclose(ref["Y"], got["Y"], rtol=1e-5, atol=1e-5)


# ── 2. value_float attribute (scalar float32) ────────────────────────────────


def test_constant_value_float_converted():
    """Constant with 'value_float' attribute converted and ORT output matches."""
    shape = [3, 4]
    scalar = 2.5
    const_node = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["scalar_c"],
        name="c_f",
        value_float=scalar,
    )
    mul_node = helper.make_node("Mul", ["X", "scalar_c"], ["Y"], name="mul")

    graph = helper.make_graph(
        [const_node, mul_node],
        "const_float",
        [_vi("X", TensorProto.FLOAT, shape)],
        [_vi("Y", TensorProto.FLOAT, shape)],
        value_info=[_vi("scalar_c", TensorProto.FLOAT, [])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", _OPSET_VERSION)])
    _clamp_ir(model)

    x = _RNG.standard_normal(shape).astype(np.float32)
    ref = _run_ort(model, {"X": x})

    tr = copy.deepcopy(model)
    _prepare_model(tr)
    cnt = _apply(tr, verbose=False)
    got = _run_ort(tr, {"X": x})
    _save(model, tr, "const_value_float")

    assert cnt == 1
    assert "scalar_c" in _initializer_names(tr)
    assert "scalar_c" not in _constant_node_names(tr)
    np.testing.assert_allclose(ref["Y"], got["Y"], rtol=1e-5, atol=1e-5)


# ── 3. value_floats attribute (1-D float32 list) ──────────────────────────────


def test_constant_value_floats_converted():
    """Constant with 'value_floats' (list) converted; ORT output matches."""
    floats = [1.0, 2.0, 3.0, 4.0]
    shape_x = [2, 4]
    const_node = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["bias"],
        name="c_fs",
        value_floats=floats,
    )
    add_node = helper.make_node("Add", ["X", "bias"], ["Y"], name="add")

    graph = helper.make_graph(
        [const_node, add_node],
        "const_floats",
        [_vi("X", TensorProto.FLOAT, shape_x)],
        [_vi("Y", TensorProto.FLOAT, shape_x)],
        value_info=[_vi("bias", TensorProto.FLOAT, [len(floats)])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", _OPSET_VERSION)])
    _clamp_ir(model)

    x = _RNG.standard_normal(shape_x).astype(np.float32)
    ref = _run_ort(model, {"X": x})

    tr = copy.deepcopy(model)
    _prepare_model(tr)
    cnt = _apply(tr, verbose=False)
    got = _run_ort(tr, {"X": x})
    _save(model, tr, "const_value_floats")

    assert cnt == 1
    assert "bias" in _initializer_names(tr)
    assert "bias" not in _constant_node_names(tr)
    np.testing.assert_allclose(ref["Y"], got["Y"], rtol=1e-5, atol=1e-5)


# ── 4. value_int attribute (scalar int64) ────────────────────────────────────


def test_constant_value_int_converted():
    """Constant with 'value_int' converted to initializer (int64 scalar).

    Uses Gather with a scalar index, which is the natural consumer of a
    scalar int64 constant (ReduceSum opset-13 requires a 1-D axes tensor).
    """
    shape = [4, 8]
    const_node = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["idx"],
        name="c_i",
        value_int=1,
    )
    # Gather selects row 1 from X → output shape [8]
    gather_node = helper.make_node("Gather", ["X", "idx"], ["Y"], name="gather", axis=0)

    graph = helper.make_graph(
        [const_node, gather_node],
        "const_int",
        [_vi("X", TensorProto.FLOAT, shape)],
        [_vi("Y", TensorProto.FLOAT, [shape[1]])],
        value_info=[_vi("idx", TensorProto.INT64, [])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", _OPSET_VERSION)])
    _clamp_ir(model)

    x = _RNG.standard_normal(shape).astype(np.float32)
    ref = _run_ort(model, {"X": x})

    tr = copy.deepcopy(model)
    _prepare_model(tr)
    cnt = _apply(tr, verbose=False)
    got = _run_ort(tr, {"X": x})
    _save(model, tr, "const_value_int")

    assert cnt == 1
    assert "idx" in _initializer_names(tr)
    assert "idx" not in _constant_node_names(tr)
    np.testing.assert_allclose(ref["Y"], got["Y"], rtol=1e-5, atol=1e-5)


# ── 5. value_ints attribute (1-D int64 list) ──────────────────────────────────


def test_constant_value_ints_converted():
    """Constant with 'value_ints' (shape tensor) converted; Reshape ORT parity."""
    shape_in = [2, 4, 8]
    target_shape = [2, 32]
    const_node = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["new_shape"],
        name="c_is",
        value_ints=target_shape,
    )
    reshape_node = helper.make_node("Reshape", ["X", "new_shape"], ["Y"], name="reshape")

    graph = helper.make_graph(
        [const_node, reshape_node],
        "const_ints",
        [_vi("X", TensorProto.FLOAT, shape_in)],
        [_vi("Y", TensorProto.FLOAT, target_shape)],
        value_info=[_vi("new_shape", TensorProto.INT64, [len(target_shape)])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", _OPSET_VERSION)])
    _clamp_ir(model)

    x = _RNG.standard_normal(shape_in).astype(np.float32)
    ref = _run_ort(model, {"X": x})

    tr = copy.deepcopy(model)
    _prepare_model(tr)
    cnt = _apply(tr, verbose=False)
    got = _run_ort(tr, {"X": x})
    _save(model, tr, "const_value_ints")

    assert cnt == 1
    assert "new_shape" in _initializer_names(tr)
    assert "new_shape" not in _constant_node_names(tr)
    np.testing.assert_allclose(ref["Y"], got["Y"], rtol=1e-5, atol=1e-5)


# ── 6. No Constant nodes: graph unchanged ────────────────────────────────────


def test_no_constants_noop():
    """Model with no Constant nodes: graph and ORT output unchanged."""
    shape = [2, 4]
    w_init = numpy_helper.from_array(_RNG.standard_normal(shape).astype(np.float32), name="W")
    add_node = helper.make_node("Add", ["X", "W"], ["Y"], name="add")

    graph = helper.make_graph(
        [add_node],
        "no_const",
        [_vi("X", TensorProto.FLOAT, shape)],
        [_vi("Y", TensorProto.FLOAT, shape)],
        initializer=[w_init],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", _OPSET_VERSION)])
    _clamp_ir(model)
    original_node_count = len(model.graph.node)

    x = _RNG.standard_normal(shape).astype(np.float32)
    ref = _run_ort(model, {"X": x})

    tr = copy.deepcopy(model)
    _prepare_model(tr)
    cnt = _apply(tr, verbose=False)
    got = _run_ort(tr, {"X": x})
    _save(model, tr, "no_constants")

    assert cnt == 0
    assert len(tr.graph.node) == original_node_count
    np.testing.assert_array_equal(ref["Y"], got["Y"])


# ── 7. Multiple Constants in one graph ───────────────────────────────────────


def test_multiple_constants_all_converted():
    """All Constant nodes in the graph are converted in a single pass; ORT parity."""
    shape = [3, 4]
    w_data = _RNG.standard_normal(shape).astype(np.float32)
    b_data = _RNG.standard_normal([shape[1]]).astype(np.float32)

    w_const = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["W"],
        name="c_w",
        value=numpy_helper.from_array(w_data, name="W"),
    )
    b_const = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["B"],
        name="c_b",
        value=numpy_helper.from_array(b_data, name="B"),
    )
    matmul = helper.make_node("MatMul", ["X", "W"], ["mm_out"], name="mm")
    add = helper.make_node("Add", ["mm_out", "B"], ["Y"], name="add")

    graph = helper.make_graph(
        [w_const, b_const, matmul, add],
        "multi_const",
        [_vi("X", TensorProto.FLOAT, [shape[0], shape[0]])],
        [_vi("Y", TensorProto.FLOAT, [shape[0], shape[1]])],
        value_info=[
            _vi("W", TensorProto.FLOAT, shape),
            _vi("B", TensorProto.FLOAT, [shape[1]]),
            _vi("mm_out", TensorProto.FLOAT, [shape[0], shape[1]]),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", _OPSET_VERSION)])
    _clamp_ir(model)

    x = _RNG.standard_normal([shape[0], shape[0]]).astype(np.float32)
    ref = _run_ort(model, {"X": x})

    tr = copy.deepcopy(model)
    _prepare_model(tr)
    cnt = _apply(tr, verbose=False)
    got = _run_ort(tr, {"X": x})
    _save(model, tr, "multi_constants")

    assert cnt == 2
    assert "W" in _initializer_names(tr)
    assert "B" in _initializer_names(tr)
    assert not any(n.op_type == "Constant" for n in tr.graph.node)
    np.testing.assert_allclose(ref["Y"], got["Y"], rtol=1e-5, atol=1e-5)
