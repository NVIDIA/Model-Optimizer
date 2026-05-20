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
# Unit tests for :mod:`modelopt.onnx.graph_surgery.dla_transforms.dla_cast_to_fp32`.

"""ORT CPU parity tests for ``dla_cast_to_fp32``.

Transform rule
--------------
Every Cast node whose output is NOT consumed by a dtype-sensitive op has its
``to`` attribute changed to FLOAT (float32).

Skip conditions (Cast node left untouched):
  - Output feeds ``input[1]`` (indices) of a Gather node  → must stay INT32
  - Output feeds ``input[1]`` (indices) of a GatherElements node → must stay INT32
  - Output feeds any input of a GroupQueryAttention node  → must stay FP16

Coverage
--------
Cast INT64→FLOAT redirected to FLOAT, ORT parity
Cast INT32→FLOAT redirected to FLOAT, ORT parity
Cast already targeting FLOAT → unchanged (cnt == 0)
Cast → Gather indices: skipped (output keeps INT32/INT64)
Cast → GatherElements indices: skipped
Cast → GroupQueryAttention input: skipped
Multiple Casts in one graph: eligible ones converted, protected ones skipped
No Cast nodes: graph unchanged, ORT output unchanged
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
_SAVED_MODELS_DIR = _REPO_ROOT / "scratch_space" / "cast_to_fp32_test_models"
_OPSET_VERSION = 17
_MAX_IR_VERSION_FOR_ORT = 9


# ── Dynamic import ────────────────────────────────────────────────────────────


def _load_module():
    mod_key = "modelopt.onnx.graph_surgery.dla_transforms.dla_cast_to_fp32"
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
        _REPO_ROOT / "modelopt/onnx/graph_surgery/dla_transforms/dla_cast_to_fp32.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_key] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


_mod = _load_module()
_apply = _mod._apply_cast_to_fp32


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


def _cast_to_attr(model: onnx.ModelProto, cast_name: str) -> int | None:
    """Return the ``to`` attribute value of a Cast node by name."""
    for node in model.graph.node:
        if node.op_type == "Cast" and node.name == cast_name:
            for attr in node.attribute:
                if attr.name == "to":
                    return int(attr.i)
    return None


_RNG = np.random.default_rng(2024)


# ── 1. Cast INT64 → redirected to FLOAT, ORT parity ─────────────────────────


def test_cast_int64_redirected_to_float():
    """Cast(to=INT64) on a float input is redirected to FLOAT; ORT output matches."""
    shape = [2, 4]
    cast_node = helper.make_node(
        "Cast",
        inputs=["X"],
        outputs=["Y"],
        name="cast_i64",
        to=TensorProto.INT64,
    )
    graph = helper.make_graph(
        [cast_node],
        "cast_int64",
        [_vi("X", TensorProto.FLOAT, shape)],
        [_vi("Y", TensorProto.FLOAT, shape)],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", _OPSET_VERSION)])
    _clamp_ir(model)

    x = _RNG.standard_normal(shape).astype(np.float32)
    # After redirect: Cast(to=FLOAT) is identity for float input
    ref = x.copy()

    tr = copy.deepcopy(model)
    _prepare_model(tr)
    tr = _apply(tr)
    _prepare_model(tr)
    got = _run_ort(tr, {"X": x})
    _save(model, tr, "cast_int64")

    assert _cast_to_attr(tr, "cast_i64") == TensorProto.FLOAT
    np.testing.assert_allclose(got["Y"], ref, rtol=1e-5, atol=1e-5)


# ── 2. Cast INT32 → redirected to FLOAT, ORT parity ─────────────────────────


def test_cast_int32_redirected_to_float():
    """Cast(to=INT32) redirected to FLOAT."""
    shape = [3, 8]
    cast_node = helper.make_node(
        "Cast",
        inputs=["X"],
        outputs=["Y"],
        name="cast_i32",
        to=TensorProto.INT32,
    )
    graph = helper.make_graph(
        [cast_node],
        "cast_int32",
        [_vi("X", TensorProto.FLOAT, shape)],
        [_vi("Y", TensorProto.FLOAT, shape)],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", _OPSET_VERSION)])
    _clamp_ir(model)

    x = _RNG.standard_normal(shape).astype(np.float32)
    ref = x.copy()

    tr = copy.deepcopy(model)
    _prepare_model(tr)
    tr = _apply(tr)
    _prepare_model(tr)
    got = _run_ort(tr, {"X": x})
    _save(model, tr, "cast_int32")

    assert _cast_to_attr(tr, "cast_i32") == TensorProto.FLOAT
    np.testing.assert_allclose(got["Y"], ref, rtol=1e-5, atol=1e-5)


# ── 3. Cast already FLOAT → unchanged (cnt == 0) ─────────────────────────────


def test_cast_already_float_unchanged():
    """Cast(to=FLOAT) is already float32 — attribute must not be touched."""
    shape = [2, 4]
    cast_node = helper.make_node(
        "Cast",
        inputs=["X"],
        outputs=["Y"],
        name="cast_fp32",
        to=TensorProto.FLOAT,
    )
    graph = helper.make_graph(
        [cast_node],
        "cast_fp32_graph",
        [_vi("X", TensorProto.FLOAT, shape)],
        [_vi("Y", TensorProto.FLOAT, shape)],
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

    assert _cast_to_attr(tr, "cast_fp32") == TensorProto.FLOAT
    np.testing.assert_array_equal(ref["Y"], got["Y"])


# ── 4. Cast → Gather indices: skipped ────────────────────────────────────────


def test_cast_to_gather_indices_skipped():
    """Cast feeding Gather input[1] (indices) must NOT be redirected to FLOAT."""
    data_shape = [4, 8]
    # Cast INT64 → INT32 for the indices, then feed into Gather
    cast_node = helper.make_node(
        "Cast",
        inputs=["idx_i64"],
        outputs=["idx_i32"],
        name="cast_idx",
        to=TensorProto.INT32,
    )
    gather_node = helper.make_node(
        "Gather",
        inputs=["X", "idx_i32"],
        outputs=["Y"],
        name="gather",
        axis=0,
    )
    graph = helper.make_graph(
        [cast_node, gather_node],
        "cast_gather",
        [
            _vi("X", TensorProto.FLOAT, data_shape),
            _vi("idx_i64", TensorProto.INT64, []),
        ],
        [_vi("Y", TensorProto.FLOAT, [data_shape[1]])],
        value_info=[_vi("idx_i32", TensorProto.INT32, [])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", _OPSET_VERSION)])
    _clamp_ir(model)

    tr = copy.deepcopy(model)
    _prepare_model(tr)
    tr = _apply(tr)
    _save(model, tr, "cast_gather_skip")

    # Cast must still target INT32 — it was NOT redirected
    assert _cast_to_attr(tr, "cast_idx") == TensorProto.INT32, (
        "Cast feeding Gather indices must keep INT32"
    )


# ── 5. Cast → GatherElements indices: skipped ────────────────────────────────


def test_cast_to_gatherelements_indices_skipped():
    """Cast feeding GatherElements input[1] (indices) must NOT be redirected."""
    shape = [3, 4]
    cast_node = helper.make_node(
        "Cast",
        inputs=["idx_i64"],
        outputs=["idx_i32"],
        name="cast_geidx",
        to=TensorProto.INT32,
    )
    ge_node = helper.make_node(
        "GatherElements",
        inputs=["X", "idx_i32"],
        outputs=["Y"],
        name="gather_elements",
        axis=0,
    )
    graph = helper.make_graph(
        [cast_node, ge_node],
        "cast_gatherelements",
        [
            _vi("X", TensorProto.FLOAT, shape),
            _vi("idx_i64", TensorProto.INT64, shape),
        ],
        [_vi("Y", TensorProto.FLOAT, shape)],
        value_info=[_vi("idx_i32", TensorProto.INT32, shape)],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", _OPSET_VERSION)])
    _clamp_ir(model)

    tr = copy.deepcopy(model)
    _prepare_model(tr)
    tr = _apply(tr)
    _save(model, tr, "cast_gatherelements_skip")

    assert _cast_to_attr(tr, "cast_geidx") == TensorProto.INT32, (
        "Cast feeding GatherElements indices must keep INT32"
    )


# ── 6. Cast → GQA input: skipped ─────────────────────────────────────────────


def test_cast_to_gqa_input_skipped():
    """Cast feeding any GroupQueryAttention input must NOT be redirected."""
    shape = [2, 4, 8]
    # Cast FLOAT → FLOAT16 for GQA input
    cast_node = helper.make_node(
        "Cast",
        inputs=["X"],
        outputs=["X_fp16"],
        name="cast_gqa",
        to=TensorProto.FLOAT16,
    )
    # Minimal GQA node stub — ORT symbolic inference requires at least 3 inputs
    gqa_node = helper.make_node(
        "GroupQueryAttention",
        inputs=["X_fp16", "", ""],
        outputs=["Y"],
        name="gqa",
        domain="com.microsoft",
        num_heads=2,
        kv_num_heads=2,
    )
    graph = helper.make_graph(
        [cast_node, gqa_node],
        "cast_gqa_graph",
        [_vi("X", TensorProto.FLOAT, shape)],
        [_vi("Y", TensorProto.FLOAT16, shape)],
        value_info=[_vi("X_fp16", TensorProto.FLOAT16, shape)],
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
    tr = _apply(tr)

    assert _cast_to_attr(tr, "cast_gqa") == TensorProto.FLOAT16, (
        "Cast feeding GQA input must keep FLOAT16"
    )


# ── 7. Mixed graph: eligible Cast converted, protected Cast skipped ───────────


def test_mixed_graph_eligible_converted_protected_skipped():
    """Graph with one eligible and one protected Cast; ORT parity on transformed model.

    Structure
    ---------
    Cast(eligible, INT64) → Reshape(same shape) → X_reshaped  [graph output 1]
    Cast(protected, INT32) → Gather(X, axis=0)  → Y_gather    [graph output 2]

    Reshape accepts any dtype so no type-op compatibility issue.
    X_i64 is intentionally NOT added to value_info: after the transform its
    dtype changes INT64→FLOAT and a stale INT64 declaration would cause ORT
    to reject the transformed model.  Omitting it lets ORT infer the type.

    ORT is run only on the transformed model; numpy reference is used for
    parity (the base model would output INT64 for X_reshaped, the transformed
    model outputs FLOAT — they are numerically the same for integer-valued x).
    """
    data_shape = [4, 8]
    shape_init = numpy_helper.from_array(np.array(data_shape, dtype=np.int64), name="reshape_shape")

    # ── Path 1: eligible cast → Reshape (dtype-agnostic) ─────────────────────
    cast_eligible = helper.make_node(
        "Cast",
        inputs=["X"],
        outputs=["X_i64"],
        name="cast_eligible",
        to=TensorProto.INT64,
    )
    reshape_node = helper.make_node(
        "Reshape",
        inputs=["X_i64", "reshape_shape"],
        outputs=["X_reshaped"],
        name="reshape_node",
    )

    # ── Path 2: protected cast → Gather ──────────────────────────────────────
    cast_protected = helper.make_node(
        "Cast",
        inputs=["idx_i64"],
        outputs=["idx_i32"],
        name="cast_protected",
        to=TensorProto.INT32,
    )
    gather_node = helper.make_node(
        "Gather",
        inputs=["X", "idx_i32"],
        outputs=["Y_gather"],
        name="gather",
        axis=0,
    )

    graph = helper.make_graph(
        [cast_eligible, reshape_node, cast_protected, gather_node],
        "mixed_graph",
        [
            _vi("X", TensorProto.FLOAT, data_shape),
            _vi("idx_i64", TensorProto.INT64, []),
        ],
        [
            _vi("X_reshaped", TensorProto.FLOAT, data_shape),
            _vi("Y_gather", TensorProto.FLOAT, [data_shape[1]]),
        ],
        initializer=[shape_init],
        # X_i64 intentionally omitted from value_info — its dtype changes after
        # the transform (INT64 → FLOAT) and a stale entry would cause ORT type errors.
        value_info=[_vi("idx_i32", TensorProto.INT32, [])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", _OPSET_VERSION)])
    _clamp_ir(model)

    # Use integer-valued input so INT64 cast is lossless and numpy ref is exact.
    x = np.arange(np.prod(data_shape), dtype=np.float32).reshape(data_shape)

    # Numpy reference for the TRANSFORMED model:
    #   cast_eligible → FLOAT (identity), Reshape (no-op) → x
    #   Gather row 1 → x[1, :]
    reshaped_ref = x.reshape(data_shape)
    y_gather_ref = x[1, :]

    tr = copy.deepcopy(model)
    _prepare_model(tr)
    tr = _apply(tr)
    _prepare_model(tr)
    got = _run_ort(tr, {"X": x, "idx_i64": np.array(1, dtype=np.int64)})
    _save(model, tr, "mixed_cast")

    # ── Attribute checks ──────────────────────────────────────────────────────
    assert _cast_to_attr(tr, "cast_eligible") == TensorProto.FLOAT, "eligible must become FLOAT"
    assert _cast_to_attr(tr, "cast_protected") == TensorProto.INT32, "protected must stay INT32"

    # ── ORT parity (transformed model vs numpy) ───────────────────────────────
    np.testing.assert_allclose(got["X_reshaped"], reshaped_ref, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(got["Y_gather"], y_gather_ref, rtol=1e-5, atol=1e-5)


# ── 8. No Cast nodes: graph and ORT output unchanged ─────────────────────────


def test_no_cast_nodes_noop():
    """Graph with no Cast nodes: graph structure and ORT output unchanged."""
    shape = [2, 4]
    relu = helper.make_node("Relu", ["X"], ["Y"], name="relu")
    graph = helper.make_graph(
        [relu],
        "no_cast",
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

    assert len(tr.graph.node) == original_node_count
    np.testing.assert_array_equal(ref["Y"], got["Y"])
