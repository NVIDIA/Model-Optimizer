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
# Unit tests for :mod:`modelopt.onnx.graph_surgery.dla_transforms.dla_convert_ops_to_4d`.

"""ORT CPU parity tests for ``dla_convert_ops_to_4d``.

Each test:
  1. Builds a minimal ONNX model with a non-4D op.
  2. Runs ORT to get reference outputs.
  3. Applies ``_apply_convert_ops_to_4d`` (via deep-copy).
  4. Runs ORT on the transformed model.
  5. Asserts cosine similarity ≥ 1-1e-5 and allclose(rtol=1e-4, atol=1e-4).

Models saved to ``scratch_space/convert_ops_to_4d_test_models/`` for inspection.

Coverage
--------
Reshape (2D→4D, 3D→4D)
Transpose (2D, 3D)
Slice (2D data)
ArgMax (2D keepdims=0, 3D keepdims=1)
ReduceSum (2D keepdims=0, 3D keepdims=1)
Gather (2D data with static indices)
GatherElements (2D data + 2D static indices)
Expand (3D input with static shape)
Tile (3D input with static repeats)
LpNormalization (2D)
Softmax (2D)
LogSoftmax (2D)
Concat (two 2D inputs)
Split (3D input)
InstanceNormalization (3D: [N,C,D])
Relu — generic handler (2D)
4D input — should not be transformed
MatMul — in _OP_SKIP, must not be transformed
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
_SAVED_MODELS_DIR = _REPO_ROOT / "scratch_space" / "convert_ops_to_4d_test_models"
_OPSET_VERSION = 17
_MAX_IR_VERSION_FOR_ORT = 9


# ── Dynamic import (avoids loading graph_surgery/__init__.py with heavy deps) ──────────────────────


def _load_apply_convert_ops_to_4d():
    mod_key = "modelopt.onnx.graph_surgery.dla_transforms.dla_convert_ops_to_4d"
    if mod_key in sys.modules:
        return sys.modules[mod_key]._apply_convert_ops_to_4d

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
        _REPO_ROOT / "modelopt/onnx/graph_surgery/dla_transforms/dla_convert_ops_to_4d.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_key] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod._apply_convert_ops_to_4d


_apply_convert_ops_to_4d = _load_apply_convert_ops_to_4d()


# ── Shared utilities ────────────────────────────────────────────────────────────────────────────────


def _init(name: str, arr: np.ndarray) -> onnx.TensorProto:
    return numpy_helper.from_array(arr, name=name)


def _vi(name: str, elem_type: int, shape: list) -> onnx.ValueInfoProto:
    return helper.make_tensor_value_info(name, elem_type, shape)


def _clamp_ir(model: onnx.ModelProto) -> None:
    model.ir_version = min(model.ir_version, _MAX_IR_VERSION_FOR_ORT)


def _make_model(nodes, name, inputs, outputs, inits=None):
    graph = helper.make_graph(nodes, name, inputs, outputs, initializer=inits or [])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", _OPSET_VERSION)])
    _clamp_ir(model)
    try:
        from onnxruntime.tools.symbolic_shape_infer import (
            SymbolicShapeInference,  # type: ignore[import-untyped]
        )

        model = SymbolicShapeInference.infer_shapes(model, auto_merge_symbolic_dims=True) or model
    except Exception:
        pass
    onnx.checker.check_model(model)
    return model


def _run_ort(model: onnx.ModelProto, feeds: dict) -> dict:
    import onnxruntime as ort  # type: ignore[import-not-found]

    _clamp_ir(model)
    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    return {o.name: sess.run([o.name], feeds)[0] for o in model.graph.output}


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a64, b64 = a.astype(np.float64).ravel(), b.astype(np.float64).ravel()
    na, nb = float(np.linalg.norm(a64)), float(np.linalg.norm(b64))
    if na == 0.0 and nb == 0.0:
        return 1.0
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a64, b64) / (na * nb))


def _assert_match(ref: dict, got: dict) -> None:
    assert ref.keys() == got.keys()
    for name in ref:
        a, b = ref[name], got[name]
        assert a.shape == b.shape, f"{name}: shape {a.shape} vs {b.shape}"
        if a.dtype.kind == "f":
            cos = _cosine_similarity(a, b)
            assert cos >= 1.0 - 1e-5, f"{name}: cosine {cos:.8f}"
            np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-4, err_msg=name)


def _assert_values_match(ref: dict, got: dict) -> None:
    """Element-wise comparison ignoring shape (for ops whose output rank changes)."""
    assert ref.keys() == got.keys()
    for name in ref:
        a, b = ref[name].ravel(), got[name].ravel()
        assert a.shape == b.shape, f"{name}: element count {a.shape} vs {b.shape}"
        if a.dtype.kind == "f":
            cos = _cosine_similarity(a, b)
            assert cos >= 1.0 - 1e-5, f"{name}: cosine {cos:.8f}"
            np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-4, err_msg=name)


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


def _apply_and_run(model, feeds):
    ref = _run_ort(model, feeds)
    tr = copy.deepcopy(model)
    tr = _apply_convert_ops_to_4d(tr)
    _clamp_ir(tr)
    got = _run_ort(tr, feeds)
    return ref, got, tr


_RNG = np.random.default_rng(2024)


# ── Reshape ─────────────────────────────────────────────────────────────────────────────────────────


def test_reshape_2d_to_4d():
    """Reshape [8,16] → [16,8]: shape init promoted [1,1,16,8]; output squeezed back to [16,8]."""
    shape = _init("shape", np.array([16, 8], dtype=np.int64))
    model = _make_model(
        [helper.make_node("Reshape", ["X", "shape"], ["Y"], name="rs")],
        "rs2d",
        [_vi("X", TensorProto.FLOAT, [8, 16])],
        [_vi("Y", TensorProto.FLOAT, [16, 8])],
        [shape],
    )
    feeds = {"X": _RNG.standard_normal((8, 16)).astype(np.float32)}
    ref, got, tr = _apply_and_run(model, feeds)
    _save(model, tr, "reshape_2d")
    _assert_match(ref, got)


def test_reshape_3d_to_4d():
    """Reshape [3,8,16] → [3,128]: shape init promoted to [1,3,128]; output squeezed to [3,128]."""
    shape = _init("shape", np.array([3, 128], dtype=np.int64))
    model = _make_model(
        [helper.make_node("Reshape", ["X", "shape"], ["Y"], name="rs")],
        "rs3d",
        [_vi("X", TensorProto.FLOAT, [3, 8, 16])],
        [_vi("Y", TensorProto.FLOAT, [3, 128])],
        [shape],
    )
    feeds = {"X": _RNG.standard_normal((3, 8, 16)).astype(np.float32)}
    ref, got, tr = _apply_and_run(model, feeds)
    _save(model, tr, "reshape_3d")
    _assert_match(ref, got)


# ── Transpose ───────────────────────────────────────────────────────────────────────────────────────


def test_transpose_2d():
    """Transpose [8,16] perm=[1,0] → [16,8]: perm adjusted to [0,1,3,2]."""
    model = _make_model(
        [helper.make_node("Transpose", ["X"], ["Y"], name="tr", perm=[1, 0])],
        "tr2d",
        [_vi("X", TensorProto.FLOAT, [8, 16])],
        [_vi("Y", TensorProto.FLOAT, [16, 8])],
    )
    feeds = {"X": _RNG.standard_normal((8, 16)).astype(np.float32)}
    ref, got, tr = _apply_and_run(model, feeds)
    _save(model, tr, "transpose_2d")
    _assert_match(ref, got)
    # Verify perm was adjusted
    tr_node = next(
        n
        for n in tr.graph.node
        if n.op_type == "Transpose"
        and any(a.name == "perm" and list(a.ints) == [0, 1, 3, 2] for a in n.attribute)
    )
    assert tr_node is not None, "Expected perm [0,1,3,2] in transformed Transpose"


def test_transpose_3d():
    """Transpose [3,8,16] perm=[2,0,1] → [16,3,8]: perm adjusted to [0,3,1,2]."""
    model = _make_model(
        [helper.make_node("Transpose", ["X"], ["Y"], name="tr", perm=[2, 0, 1])],
        "tr3d",
        [_vi("X", TensorProto.FLOAT, [3, 8, 16])],
        [_vi("Y", TensorProto.FLOAT, [16, 3, 8])],
    )
    feeds = {"X": _RNG.standard_normal((3, 8, 16)).astype(np.float32)}
    ref, got, tr = _apply_and_run(model, feeds)
    _save(model, tr, "transpose_3d")
    _assert_match(ref, got)


# ── Slice ───────────────────────────────────────────────────────────────────────────────────────────


def test_slice_2d():
    """Slice data[8,16] along axis=0 (starts=2,ends=6) → [4,16]; axis updated to 2."""
    starts = _init("starts", np.array([2], dtype=np.int64))
    ends = _init("ends", np.array([6], dtype=np.int64))
    axes = _init("axes", np.array([0], dtype=np.int64))
    model = _make_model(
        [helper.make_node("Slice", ["X", "starts", "ends", "axes"], ["Y"], name="sl")],
        "sl2d",
        [_vi("X", TensorProto.FLOAT, [8, 16])],
        [_vi("Y", TensorProto.FLOAT, [4, 16])],
        [starts, ends, axes],
    )
    feeds = {"X": _RNG.standard_normal((8, 16)).astype(np.float32)}
    ref, got, tr = _apply_and_run(model, feeds)
    _save(model, tr, "slice_2d")
    _assert_match(ref, got)


# ── ArgMax ──────────────────────────────────────────────────────────────────────────────────────────


def test_argmax_2d_keepdims0():
    """ArgMax [8,16] axis=1 keepdims=0 → [8]; axis adjusted to 3, keepdims forced=1, Squeeze added."""
    model = _make_model(
        [helper.make_node("ArgMax", ["X"], ["Y"], name="am", axis=1, keepdims=0)],
        "am2d",
        [_vi("X", TensorProto.FLOAT, [8, 16])],
        [_vi("Y", TensorProto.INT64, [8])],
    )
    feeds = {"X": _RNG.standard_normal((8, 16)).astype(np.float32)}
    ref, got, tr = _apply_and_run(model, feeds)
    _save(model, tr, "argmax_2d_kd0")
    _assert_match(ref, got)
    # keepdims must be forced to 1 on the transformed node
    am_node = next(n for n in tr.graph.node if n.op_type == "ArgMax")
    kd = next((int(a.i) for a in am_node.attribute if a.name == "keepdims"), 1)
    assert kd == 1, f"ArgMax keepdims must be 1 after transform, got {kd}"
    # a Squeeze node must be inserted to restore the original [8] shape
    assert any(n.op_type == "Squeeze" for n in tr.graph.node), (
        "Squeeze must be added for keepdims=0"
    )


def test_argmax_3d_keepdims1():
    """ArgMax [3,8,16] axis=2 keepdims=1 → [3,8,1]; axis adjusted to 3, keepdims unchanged."""
    model = _make_model(
        [helper.make_node("ArgMax", ["X"], ["Y"], name="am", axis=2, keepdims=1)],
        "am3d",
        [_vi("X", TensorProto.FLOAT, [3, 8, 16])],
        [_vi("Y", TensorProto.INT64, [3, 8, 1])],
    )
    feeds = {"X": _RNG.standard_normal((3, 8, 16)).astype(np.float32)}
    ref, got, tr = _apply_and_run(model, feeds)
    _save(model, tr, "argmax_3d_kd1")
    _assert_match(ref, got)
    # keepdims must remain 1 on the transformed node
    am_node = next(n for n in tr.graph.node if n.op_type == "ArgMax")
    kd = next((int(a.i) for a in am_node.attribute if a.name == "keepdims"), 1)
    assert kd == 1, f"ArgMax keepdims must be 1 after transform, got {kd}"


# ── ReduceSum ───────────────────────────────────────────────────────────────────────────────────────


def test_reducesum_2d_keepdims0():
    """ReduceSum [8,16] axes=[1] keepdims=0 → [8]; axis adjusted to 3, Squeeze covers adjusted axis."""
    axes_init = _init("axes_rd", np.array([1], dtype=np.int64))
    model = _make_model(
        [helper.make_node("ReduceSum", ["X", "axes_rd"], ["Y"], name="rd", keepdims=0)],
        "rd2d_kd0",
        [_vi("X", TensorProto.FLOAT, [8, 16])],
        [_vi("Y", TensorProto.FLOAT, [8])],
        [axes_init],
    )
    feeds = {"X": _RNG.standard_normal((8, 16)).astype(np.float32)}
    ref, got, tr = _apply_and_run(model, feeds)
    _save(model, tr, "reducesum_2d_kd0")
    _assert_match(ref, got)
    # keepdims must be forced to 1 on the transformed node
    rd_node = next(n for n in tr.graph.node if n.op_type == "ReduceSum")
    kd = next((int(a.i) for a in rd_node.attribute if a.name == "keepdims"), 1)
    assert kd == 1, f"ReduceSum keepdims must be 1 after transform, got {kd}"
    # a Squeeze node must be inserted to restore the original [8] shape
    assert any(n.op_type == "Squeeze" for n in tr.graph.node), (
        "Squeeze must be added for keepdims=0"
    )


def test_reducesum_3d_keepdims1():
    """ReduceSum [3,8,16] axes=[1] keepdims=1 → [3,1,16]; axis adjusted to 2, leading Squeeze only."""
    axes_init = _init("axes_rd3", np.array([1], dtype=np.int64))
    model = _make_model(
        [helper.make_node("ReduceSum", ["X", "axes_rd3"], ["Y"], name="rd", keepdims=1)],
        "rd3d_kd1",
        [_vi("X", TensorProto.FLOAT, [3, 8, 16])],
        [_vi("Y", TensorProto.FLOAT, [3, 1, 16])],
        [axes_init],
    )
    feeds = {"X": _RNG.standard_normal((3, 8, 16)).astype(np.float32)}
    ref, got, tr = _apply_and_run(model, feeds)
    _save(model, tr, "reducesum_3d_kd1")
    _assert_match(ref, got)
    # keepdims must remain 1 on the transformed node
    rd_node = next(n for n in tr.graph.node if n.op_type == "ReduceSum")
    kd = next((int(a.i) for a in rd_node.attribute if a.name == "keepdims"), 1)
    assert kd == 1, f"ReduceSum keepdims must be 1 after transform, got {kd}"


def test_reducesum_4d_keepdims0():
    """ReduceSum [1,8,1,16] axes=[3] keepdims=0 → [1,8,1]; keepdims forced to 1, Squeeze added for axis 3."""
    axes_init = _init("axes_rd4", np.array([3], dtype=np.int64))
    model = _make_model(
        [helper.make_node("ReduceSum", ["X", "axes_rd4"], ["Y"], name="rd", keepdims=0)],
        "rd4d_kd0",
        [_vi("X", TensorProto.FLOAT, [1, 8, 1, 16])],
        [_vi("Y", TensorProto.FLOAT, [1, 8, 1])],
        [axes_init],
    )
    feeds = {"X": _RNG.standard_normal((1, 8, 1, 16)).astype(np.float32)}
    ref, got, tr = _apply_and_run(model, feeds)
    _save(model, tr, "reducesum_4d_kd0")
    _assert_match(ref, got)
    # keepdims must be forced to 1 even for 4D input
    rd_node = next(n for n in tr.graph.node if n.op_type == "ReduceSum")
    kd = next((int(a.i) for a in rd_node.attribute if a.name == "keepdims"), 1)
    assert kd == 1, f"ReduceSum keepdims must be 1 after transform, got {kd}"
    # Squeeze must be added to remove the reduced axis (axis 3)
    assert any(n.op_type == "Squeeze" for n in tr.graph.node), (
        "Squeeze must be added for keepdims=0"
    )


def test_argmax_4d_keepdims0():
    """ArgMax [1,8,1,16] axis=3 keepdims=0 → [1,8,1]; keepdims forced to 1, Squeeze added for axis 3."""
    model = _make_model(
        [helper.make_node("ArgMax", ["X"], ["Y"], name="am", axis=3, keepdims=0)],
        "am4d_kd0",
        [_vi("X", TensorProto.FLOAT, [1, 8, 1, 16])],
        [_vi("Y", TensorProto.INT64, [1, 8, 1])],
    )
    feeds = {"X": _RNG.standard_normal((1, 8, 1, 16)).astype(np.float32)}
    ref, got, tr = _apply_and_run(model, feeds)
    _save(model, tr, "argmax_4d_kd0")
    _assert_match(ref, got)
    # keepdims must be forced to 1 even for 4D input
    am_node = next(n for n in tr.graph.node if n.op_type == "ArgMax")
    kd = next((int(a.i) for a in am_node.attribute if a.name == "keepdims"), 1)
    assert kd == 1, f"ArgMax keepdims must be 1 after transform, got {kd}"
    # Squeeze must be added to remove the reduced axis (axis 3)
    assert any(n.op_type == "Squeeze" for n in tr.graph.node), (
        "Squeeze must be added for keepdims=0"
    )


# ── Gather ──────────────────────────────────────────────────────────────────────────────────────────


def test_gather_2d_data():
    """Gather data[8,16] axis=0 with 1D indices[2] → [2,16]; data unsqueezed to 4D, output squeezed back."""
    # Use 1D indices (shape [2]) → output shape [2,16]
    idx = _init("idx", np.array([1, 3], dtype=np.int32))
    model = _make_model(
        [helper.make_node("Gather", ["X", "idx"], ["Y"], name="ga", axis=0)],
        "ga2d",
        [_vi("X", TensorProto.FLOAT, [8, 16])],
        [_vi("Y", TensorProto.FLOAT, [2, 16])],
        [idx],
    )
    feeds = {"X": _RNG.standard_normal((8, 16)).astype(np.float32)}
    ref, got, tr = _apply_and_run(model, feeds)
    _save(model, tr, "gather_2d")
    _assert_match(ref, got)
    assert any(n.op_type == "Squeeze" for n in tr.graph.node), "Squeeze must restore output shape"


def test_gather_3d_axis1_scalar_index():
    """Gather [1900,1,2] axis=1 scalar-index=0 → [1900,2].

    The extra unary dim in the 4D Gather output is NOT at the leading position
    ([1,1900,1,2] after unsqueeze → axis shifts to 2 → output [1,1900,2])
    so the squeeze axes must be [0,?] determined by _find_squeeze_axes, not just [0,1].
    """
    # scalar index 0 → output drops axis=1 (size 1) → [1900,2]
    idx = _init("idx3d", np.array(0, dtype=np.int32))
    model = _make_model(
        [helper.make_node("Gather", ["X", "idx3d"], ["Y"], name="ga3d", axis=1)],
        "ga3d_ax1",
        [_vi("X", TensorProto.FLOAT, [1900, 1, 2])],
        [_vi("Y", TensorProto.FLOAT, [1900, 2])],
        [idx],
    )
    feeds = {"X": _RNG.standard_normal((1900, 1, 2)).astype(np.float32)}
    ref, got, tr = _apply_and_run(model, feeds)
    _save(model, tr, "gather_3d_ax1_scalar")

    # Output shape must be [1900,2] — not [1,2] or [1900,1,2]
    assert got["Y"].shape == (1900, 2), f"Expected (1900,2), got {got['Y'].shape}"
    _assert_match(ref, got)
    assert any(n.op_type == "Squeeze" for n in tr.graph.node), "Squeeze must restore output shape"


# ── GatherElements ──────────────────────────────────────────────────────────────────────────────────


def test_gatherelements_2d():
    """GatherElements data[4,8] axis=1 indices[4,8] → [4,8]; both expanded to 4D."""
    # indices: gather last 4 of 8 cols for each row
    indices_arr = np.tile(np.arange(4, 8, dtype=np.int32), (4, 1))  # [4,4]
    # shape must match data: [4,8]
    indices_arr = np.concatenate([indices_arr, indices_arr], axis=1)  # [4,8]
    idx = _init("idx_ge", indices_arr)
    model = _make_model(
        [helper.make_node("GatherElements", ["X", "idx_ge"], ["Y"], name="ge", axis=1)],
        "ge2d",
        [_vi("X", TensorProto.FLOAT, [4, 8])],
        [_vi("Y", TensorProto.FLOAT, [4, 8])],
        [idx],
    )
    feeds = {"X": _RNG.standard_normal((4, 8)).astype(np.float32)}
    ref, got, tr = _apply_and_run(model, feeds)
    _save(model, tr, "gatherelements_2d")
    _assert_match(ref, got)
    assert any(n.op_type == "Squeeze" for n in tr.graph.node), "Squeeze must restore output shape"


# ── Expand ──────────────────────────────────────────────────────────────────────────────────────────


def test_expand_3d():
    """Expand [3,1,16] to [3,8,16]: shape init updated from [3,8,16] → [1,3,8,16]."""
    target = _init("target_sh", np.array([3, 8, 16], dtype=np.int64))
    model = _make_model(
        [helper.make_node("Expand", ["X", "target_sh"], ["Y"], name="ex")],
        "ex3d",
        [_vi("X", TensorProto.FLOAT, [3, 1, 16])],
        [_vi("Y", TensorProto.FLOAT, [3, 8, 16])],
        [target],
    )
    feeds = {"X": _RNG.standard_normal((3, 1, 16)).astype(np.float32)}
    ref, got, tr = _apply_and_run(model, feeds)
    _save(model, tr, "expand_3d")
    _assert_match(ref, got)


# ── Tile ────────────────────────────────────────────────────────────────────────────────────────────


def test_tile_3d():
    """Tile [3,8,16] repeats=[1,2,1] → [3,16,16]: repeats updated from [1,2,1] → [1,1,2,1]."""
    repeats = _init("repeats", np.array([1, 2, 1], dtype=np.int64))
    model = _make_model(
        [helper.make_node("Tile", ["X", "repeats"], ["Y"], name="tl")],
        "tl3d",
        [_vi("X", TensorProto.FLOAT, [3, 8, 16])],
        [_vi("Y", TensorProto.FLOAT, [3, 16, 16])],
        [repeats],
    )
    feeds = {"X": _RNG.standard_normal((3, 8, 16)).astype(np.float32)}
    ref, got, tr = _apply_and_run(model, feeds)
    _save(model, tr, "tile_3d")
    _assert_match(ref, got)


# ── LpNormalization ─────────────────────────────────────────────────────────────────────────────────


def test_lpnorm_2d():
    """LpNormalization [8,16] axis set to -1; input unsqueezed, output squeezed."""
    model = _make_model(
        [helper.make_node("LpNormalization", ["X"], ["Y"], name="lpn", axis=1, p=2)],
        "lpn2d",
        [_vi("X", TensorProto.FLOAT, [8, 16])],
        [_vi("Y", TensorProto.FLOAT, [8, 16])],
    )
    feeds = {"X": _RNG.standard_normal((8, 16)).astype(np.float32)}
    ref, got, tr = _apply_and_run(model, feeds)
    _save(model, tr, "lpnorm_2d")
    _assert_match(ref, got)


# ── Softmax ─────────────────────────────────────────────────────────────────────────────────────────


def test_softmax_2d():
    """Softmax [8,16] axis=1 → [8,16]; axis adjusted to 3."""
    model = _make_model(
        [helper.make_node("Softmax", ["X"], ["Y"], name="sm", axis=1)],
        "sm2d",
        [_vi("X", TensorProto.FLOAT, [8, 16])],
        [_vi("Y", TensorProto.FLOAT, [8, 16])],
    )
    feeds = {"X": _RNG.standard_normal((8, 16)).astype(np.float32)}
    ref, got, tr = _apply_and_run(model, feeds)
    _save(model, tr, "softmax_2d")
    _assert_match(ref, got)
    # Verify axis update
    sm_node = next(n for n in tr.graph.node if n.op_type == "Softmax")
    axis_val = next(a.i for a in sm_node.attribute if a.name == "axis")
    assert axis_val == 3, f"Expected axis=3 after 2D→4D promotion, got {axis_val}"


# ── LogSoftmax ──────────────────────────────────────────────────────────────────────────────────────


def test_logsoftmax_3d():
    """LogSoftmax [3,8,16] axis=2 → [3,8,16]; axis adjusted to 3."""
    model = _make_model(
        [helper.make_node("LogSoftmax", ["X"], ["Y"], name="lsm", axis=2)],
        "lsm3d",
        [_vi("X", TensorProto.FLOAT, [3, 8, 16])],
        [_vi("Y", TensorProto.FLOAT, [3, 8, 16])],
    )
    feeds = {"X": _RNG.standard_normal((3, 8, 16)).astype(np.float32)}
    ref, got, tr = _apply_and_run(model, feeds)
    _save(model, tr, "logsoftmax_3d")
    _assert_match(ref, got)


# ── Concat ──────────────────────────────────────────────────────────────────────────────────────────


def test_concat_2d():
    """Concat two [8,16] tensors along axis=1 → [8,32]; axis adjusted to 3."""
    model = _make_model(
        [helper.make_node("Concat", ["A", "B"], ["Y"], name="ct", axis=1)],
        "ct2d",
        [_vi("A", TensorProto.FLOAT, [8, 16]), _vi("B", TensorProto.FLOAT, [8, 16])],
        [_vi("Y", TensorProto.FLOAT, [8, 32])],
    )
    feeds = {
        "A": _RNG.standard_normal((8, 16)).astype(np.float32),
        "B": _RNG.standard_normal((8, 16)).astype(np.float32),
    }
    ref, got, tr = _apply_and_run(model, feeds)
    _save(model, tr, "concat_2d")
    _assert_match(ref, got)


# ── Split ───────────────────────────────────────────────────────────────────────────────────────────


def test_split_3d():
    """Split [3,8,16] along axis=2 into two [3,8,8] tensors; axis adjusted to 3."""
    split_sizes = _init("split_sizes", np.array([8, 8], dtype=np.int64))
    model = _make_model(
        [helper.make_node("Split", ["X", "split_sizes"], ["Y1", "Y2"], name="sp", axis=2)],
        "sp3d",
        [_vi("X", TensorProto.FLOAT, [3, 8, 16])],
        [_vi("Y1", TensorProto.FLOAT, [3, 8, 8]), _vi("Y2", TensorProto.FLOAT, [3, 8, 8])],
        [split_sizes],
    )
    feeds = {"X": _RNG.standard_normal((3, 8, 16)).astype(np.float32)}
    ref, got, tr = _apply_and_run(model, feeds)
    _save(model, tr, "split_3d")
    _assert_match(ref, got)


# ── InstanceNormalization ───────────────────────────────────────────────────────────────────────────


def test_instancenorm_3d():
    """InstanceNorm [N=2,C=8,D=16]: trailing Unsqueeze(axis=3) before; Squeeze(axis=3) after."""
    n, c, d = 2, 8, 16
    scale = _init("in_scale", np.ones((c,), dtype=np.float32))
    bias = _init("in_bias", np.zeros((c,), dtype=np.float32))
    model = _make_model(
        [helper.make_node("InstanceNormalization", ["X", "in_scale", "in_bias"], ["Y"], name="in")],
        "in3d",
        [_vi("X", TensorProto.FLOAT, [n, c, d])],
        [_vi("Y", TensorProto.FLOAT, [n, c, d])],
        [scale, bias],
    )
    feeds = {"X": _RNG.standard_normal((n, c, d)).astype(np.float32)}
    ref, got, tr = _apply_and_run(model, feeds)
    _save(model, tr, "instancenorm_3d")
    _assert_match(ref, got)
    # Verify Unsqueeze and Squeeze for trailing dim were added
    assert any(n.op_type == "Unsqueeze" for n in tr.graph.node), "Unsqueeze expected"
    assert any(n.op_type == "Squeeze" for n in tr.graph.node), "Squeeze expected"


# ── Generic handler (Relu) ──────────────────────────────────────────────────────────────────────────


def test_generic_relu_2d():
    """Relu [8,16] → [8,16]: generic handler unsqueezes input, squeezes output."""
    model = _make_model(
        [helper.make_node("Relu", ["X"], ["Y"], name="rl")],
        "rl2d",
        [_vi("X", TensorProto.FLOAT, [8, 16])],
        [_vi("Y", TensorProto.FLOAT, [8, 16])],
    )
    feeds = {"X": _RNG.standard_normal((8, 16)).astype(np.float32)}
    ref, got, tr = _apply_and_run(model, feeds)
    _save(model, tr, "relu_2d")
    _assert_match(ref, got)
    assert any(n.op_type == "Unsqueeze" for n in tr.graph.node), "Unsqueeze expected"
    assert any(n.op_type == "Squeeze" for n in tr.graph.node), "Squeeze expected"


# ── 4D input — no transform ─────────────────────────────────────────────────────────────────────────


def test_4d_input_no_change():
    """Softmax with 4D input [1,3,8,16]: no Unsqueeze/Squeeze should be added."""
    model = _make_model(
        [helper.make_node("Softmax", ["X"], ["Y"], name="sm4d", axis=3)],
        "sm4d",
        [_vi("X", TensorProto.FLOAT, [1, 3, 8, 16])],
        [_vi("Y", TensorProto.FLOAT, [1, 3, 8, 16])],
    )
    feeds = {"X": _RNG.standard_normal((1, 3, 8, 16)).astype(np.float32)}
    ref, got, tr = _apply_and_run(model, feeds)
    _save(model, tr, "softmax_4d_noop")
    _assert_match(ref, got)
    assert not any(n.op_type in ("Unsqueeze", "Squeeze") for n in tr.graph.node), (
        "No Unsqueeze/Squeeze should be added for 4D input"
    )


# ── MatMul — in _OP_SKIP ────────────────────────────────────────────────────────────────────────────


def test_matmul_skipped():
    """MatMul in _OP_SKIP: Relu (2D) is transformed but MatMul is left unchanged."""
    w = _RNG.standard_normal((16, 12)).astype(np.float32)
    model = _make_model(
        [
            helper.make_node("Relu", ["X"], ["X_relu"], name="rl"),
            helper.make_node("MatMul", ["X_relu", "W"], ["Y"], name="mm"),
        ],
        "mm_skip",
        [_vi("X", TensorProto.FLOAT, [8, 16])],
        [_vi("Y", TensorProto.FLOAT, [8, 12])],
        [_init("W", w)],
    )
    feeds = {"X": _RNG.standard_normal((8, 16)).astype(np.float32)}
    ref, got, tr = _apply_and_run(model, feeds)
    _save(model, tr, "matmul_skipped")
    _assert_match(ref, got)
    mm_nodes = [n for n in tr.graph.node if n.op_type == "MatMul"]
    assert len(mm_nodes) == 1, "MatMul must remain unchanged"
    assert any(n.op_type == "Unsqueeze" for n in tr.graph.node), "Relu should have been wrapped"


# ── 5D Transpose collapse ───────────────────────────────────────────────────────────────────────────


def test_transpose_5d_collapse():
    """Transpose [2,1,3,4,5] perm=[0,2,1,3,4] collapses to 4D with ORT parity.

    Algorithm:
      current_shape = [orig_shape[p] for p in perm] = [2,3,1,4,5]
      idx_to_remove = 2  (leftmost unary dim in the output)
      removed_orig_dim = perm[2] = 1
      new_perm_raw = [0,2,3,4] → shift down values >=1 → [0,1,2,3]

    Transform: Squeeze(input, axis=1) → 4D → Transpose(perm=[0,1,2,3]) →
               Unsqueeze(axis=2) → restores 5D output [2,3,1,4,5].
    ORT parity holds because the Unsqueeze restores the original output shape.
    """
    perm = [0, 2, 1, 3, 4]
    input_shape = [2, 1, 3, 4, 5]
    output_shape_perm = [input_shape[p] for p in perm]  # [2,3,1,4,5]
    model = _make_model(
        [helper.make_node("Transpose", ["X"], ["Y"], name="tr5d", perm=perm)],
        "tr5d",
        [_vi("X", TensorProto.FLOAT, input_shape)],
        [_vi("Y", TensorProto.FLOAT, output_shape_perm)],
    )
    feeds = {"X": _RNG.standard_normal(input_shape).astype(np.float32)}
    ref, got, tr = _apply_and_run(model, feeds)
    _save(model, tr, "transpose_5d")
    _assert_match(ref, got)

    tr_nodes = list(tr.graph.node)
    # Squeeze must be added to remove unary input dim, Unsqueeze to restore output shape
    assert any(n.op_type == "Squeeze" for n in tr_nodes), (
        "Squeeze must be added to collapse unary input dim"
    )
    assert any(n.op_type == "Unsqueeze" for n in tr_nodes), (
        "Unsqueeze must be added to restore 5D output shape"
    )
    # Transpose must have the collapsed 4-element perm [0,1,2,3]
    tr_node = next((n for n in tr_nodes if n.op_type == "Transpose"), None)
    assert tr_node is not None, "Transpose must still exist after collapse"
    perm_attr = next((a for a in tr_node.attribute if a.name == "perm"), None)
    assert perm_attr is not None, "Transpose must have a perm attribute"
    assert list(perm_attr.ints) == [0, 1, 2, 3], (
        f"Expected collapsed perm [0,1,2,3], got {list(perm_attr.ints)}"
    )


# ── Graph-output squeeze split ────────────────────────────────────────────────
#
# When a node's output is BOTH a graph output AND an input to other nodes,
# _split_graph_output_squeezes creates two separate Squeeze nodes:
#   • Squeeze → <name>      (preserves the graph-output binding)
#   • Squeeze → <name>_sq   (for internal compute consumers)
# A value_info entry is added for <name>_sq; consumers are rewired.
# ─────────────────────────────────────────────────────────────────────────────


def test_graph_output_split_single_internal_consumer():
    """Y is a graph output AND consumed by Relu → two Squeeze nodes, ORT parity.

    Model:  X [1,3,4] → Add(X,X) → Y [1,3,4]  ← graph output
                         Y        → Relu(Y) → Z [1,3,4]  ← graph output
    """
    shape = [1, 3, 4]
    model = _make_model(
        [
            helper.make_node("Add", ["X", "X"], ["Y"], name="add"),
            helper.make_node("Relu", ["Y"], ["Z"], name="relu"),
        ],
        "g",
        [_vi("X", TensorProto.FLOAT, shape)],
        [_vi("Y", TensorProto.FLOAT, shape), _vi("Z", TensorProto.FLOAT, shape)],
    )
    feeds = {"X": _RNG.standard_normal(shape).astype(np.float32)}
    ref = _run_ort(model, feeds)

    tr = copy.deepcopy(model)
    tr = _apply_convert_ops_to_4d(tr)
    _clamp_ir(tr)
    got = _run_ort(tr, feeds)
    _save(model, tr, "graph_out_split_single")

    # Y must still be a graph output
    go_names = {o.name for o in tr.graph.output}
    assert "Y" in go_names, "Y must remain a graph output"

    # Exactly one Squeeze producing Y (graph output binding)
    sq_y = [n for n in tr.graph.node if n.op_type == "Squeeze" and n.output[0] == "Y"]
    assert len(sq_y) == 1, f"Expected 1 Squeeze → Y, got {len(sq_y)}"

    # Exactly one Squeeze producing Y_sq (internal consumers)
    sq_y_sq = [n for n in tr.graph.node if n.op_type == "Squeeze" and n.output[0] == "Y_sq"]
    assert len(sq_y_sq) == 1, f"Expected 1 Squeeze → Y_sq, got {len(sq_y_sq)}"

    # Both squeezes must share the same 4D input (Add's promoted output)
    assert sq_y[0].input[0] == sq_y_sq[0].input[0], "Both squeezes must consume the same 4D tensor"

    # value_info must be registered for Y_sq
    vi_names = {vi.name for vi in tr.graph.value_info}
    assert "Y_sq" in vi_names, "value_info must be added for Y_sq"

    # No internal compute node may directly consume Y
    for node in tr.graph.node:
        for inp in node.input:
            assert inp != "Y", f"Internal node {node.name!r} must not consume Y directly; use Y_sq"

    _assert_match(ref, got)


def test_graph_output_only_no_split():
    """Y is a graph output only (no internal consumer) → single Squeeze, no Y_sq created."""
    shape = [1, 3, 4]
    model = _make_model(
        [helper.make_node("Add", ["X", "X"], ["Y"], name="add")],
        "g",
        [_vi("X", TensorProto.FLOAT, shape)],
        [_vi("Y", TensorProto.FLOAT, shape)],
    )
    feeds = {"X": _RNG.standard_normal(shape).astype(np.float32)}
    ref = _run_ort(model, feeds)

    tr = copy.deepcopy(model)
    tr = _apply_convert_ops_to_4d(tr)
    _clamp_ir(tr)
    got = _run_ort(tr, feeds)
    _save(model, tr, "graph_out_only_no_split")

    sq_y = [n for n in tr.graph.node if n.op_type == "Squeeze" and n.output[0] == "Y"]
    assert len(sq_y) == 1, "Exactly one Squeeze → Y when Y has no internal consumers"

    sq_y_sq = [n for n in tr.graph.node if n.op_type == "Squeeze" and n.output[0] == "Y_sq"]
    assert not sq_y_sq, "No Y_sq Squeeze must be created when Y is only a graph output"

    _assert_match(ref, got)


def test_graph_output_split_multiple_internal_consumers():
    """Y is a graph output AND consumed by two independent nodes → single Y_sq shared, ORT parity.

    Model:  X [1,3,4] → Add(X,X) → Y (graph output)
                         Y → Relu(Y) → Z1 (graph output)
                         Y → Abs(Y)  → Z2 (graph output)
    """
    shape = [1, 3, 4]
    model = _make_model(
        [
            helper.make_node("Add", ["X", "X"], ["Y"], name="add"),
            helper.make_node("Relu", ["Y"], ["Z1"], name="relu"),
            helper.make_node("Abs", ["Y"], ["Z2"], name="absn"),
        ],
        "g",
        [_vi("X", TensorProto.FLOAT, shape)],
        [
            _vi("Y", TensorProto.FLOAT, shape),
            _vi("Z1", TensorProto.FLOAT, shape),
            _vi("Z2", TensorProto.FLOAT, shape),
        ],
    )
    feeds = {"X": _RNG.standard_normal(shape).astype(np.float32)}
    ref = _run_ort(model, feeds)

    tr = copy.deepcopy(model)
    tr = _apply_convert_ops_to_4d(tr)
    _clamp_ir(tr)
    got = _run_ort(tr, feeds)
    _save(model, tr, "graph_out_split_multi_consumers")

    # Y must remain a graph output; Y_sq for internal nodes
    go_names = {o.name for o in tr.graph.output}
    assert "Y" in go_names

    sq_y_sq = [n for n in tr.graph.node if n.op_type == "Squeeze" and n.output[0] == "Y_sq"]
    assert sq_y_sq, "Y_sq Squeeze must exist for internal consumers"

    # No compute node may directly consume Y
    for node in tr.graph.node:
        for inp in node.input:
            assert inp != "Y", f"Node {node.name!r} must not consume Y directly after split"

    _assert_match(ref, got)


# ── LayerNormalization ────────────────────────────────────────────────────────


def test_layernorm_3d_data_input_promoted():
    """LayerNorm [2,4,8] → data unsqueezed to 4D; scale/bias kept 1D; output squeezed back; ORT parity."""
    n, c, d = 2, 4, 8
    scale = numpy_helper.from_array(np.ones(d, dtype=np.float32), name="scale")
    bias = numpy_helper.from_array(np.zeros(d, dtype=np.float32), name="bias")
    model = _make_model(
        [helper.make_node("LayerNormalization", ["X", "scale", "bias"], ["Y"], name="ln", axis=-1)],
        "ln3d",
        [_vi("X", TensorProto.FLOAT, [n, c, d])],
        [_vi("Y", TensorProto.FLOAT, [n, c, d])],
        [scale, bias],
    )
    feeds = {"X": _RNG.standard_normal((n, c, d)).astype(np.float32)}
    ref = _run_ort(model, feeds)

    tr = copy.deepcopy(model)
    tr = _apply_convert_ops_to_4d(tr)
    _clamp_ir(tr)
    got = _run_ort(tr, feeds)
    _save(model, tr, "layernorm_3d")

    ln_node = next(n for n in tr.graph.node if n.op_type == "LayerNormalization")

    # Data input must be 4D (Unsqueeze inserted before LN)
    assert any(n.op_type == "Unsqueeze" for n in tr.graph.node), (
        "Unsqueeze must be inserted to promote data input to 4D"
    )
    unsq_out = next(n for n in tr.graph.node if n.op_type == "Unsqueeze").output[0]
    assert ln_node.input[0] == unsq_out, "LN must read the 4D unsqueezed tensor"

    # Scale and bias must be unchanged (still the original 1D initializer names)
    assert ln_node.input[1] == "scale", "scale must remain unchanged"
    assert ln_node.input[2] == "bias", "bias must remain unchanged"
    scale_init = next(i for i in tr.graph.initializer if i.name == "scale")
    bias_init = next(i for i in tr.graph.initializer if i.name == "bias")
    assert list(scale_init.dims) == [d], f"scale must stay 1D [{d}], got {list(scale_init.dims)}"
    assert list(bias_init.dims) == [d], f"bias must stay 1D [{d}], got {list(bias_init.dims)}"

    # Squeeze inserted after LN to restore original rank
    assert any(n.op_type == "Squeeze" for n in tr.graph.node), (
        "Squeeze must be inserted to restore output rank"
    )

    _assert_match(ref, got)


def test_layernorm_2d_data_input_promoted():
    """LayerNorm [4,8] → data unsqueezed to 4D; scale/bias kept 1D; output squeezed back; ORT parity."""
    c, d = 4, 8
    scale = numpy_helper.from_array(np.ones(d, dtype=np.float32), name="scale2d")
    bias = numpy_helper.from_array(np.zeros(d, dtype=np.float32), name="bias2d")
    model = _make_model(
        [
            helper.make_node(
                "LayerNormalization", ["X", "scale2d", "bias2d"], ["Y"], name="ln2d", axis=-1
            )
        ],
        "ln2d",
        [_vi("X", TensorProto.FLOAT, [c, d])],
        [_vi("Y", TensorProto.FLOAT, [c, d])],
        [scale, bias],
    )
    feeds = {"X": _RNG.standard_normal((c, d)).astype(np.float32)}
    ref = _run_ort(model, feeds)

    tr = copy.deepcopy(model)
    tr = _apply_convert_ops_to_4d(tr)
    _clamp_ir(tr)
    got = _run_ort(tr, feeds)
    _save(model, tr, "layernorm_2d")

    ln_node = next(n for n in tr.graph.node if n.op_type == "LayerNormalization")

    # scale and bias must remain 1D
    scale_init = next(i for i in tr.graph.initializer if i.name == "scale2d")
    bias_init = next(i for i in tr.graph.initializer if i.name == "bias2d")
    assert list(scale_init.dims) == [d]
    assert list(bias_init.dims) == [d]
    assert ln_node.input[1] == "scale2d"
    assert ln_node.input[2] == "bias2d"

    _assert_match(ref, got)


# ── Flatten (handled inside convert_ops_to_4d via _handle_flatten) ───────────
#
# Flatten(X, axis=a) → Unsqueeze(to 4D) → Reshape([1,1,outer,inner]) → Squeeze([outer,inner])
# Input already 4D: Unsqueeze omitted.
# ─────────────────────────────────────────────────────────────────────────────


def _assert_flatten_chain(tr, *, expect_unsqueeze: bool) -> None:
    op_types = [n.op_type for n in tr.graph.node]
    assert "Flatten" not in op_types, "Flatten must be removed"
    assert "Reshape" in op_types, "Reshape must be present"
    assert "Squeeze" in op_types, "Squeeze must be present"
    if expect_unsqueeze:
        assert "Unsqueeze" in op_types, "Unsqueeze must be present for non-4D input"
    else:
        assert "Unsqueeze" not in op_types, "Unsqueeze must NOT be inserted for 4D input"


def test_flatten_3d_axis2():
    """Flatten [1,3,4] axis=2: outer=3, inner=4 → output [3,4]; ORT parity."""
    model = _make_model(
        [helper.make_node("Flatten", ["X"], ["Y"], name="fl", axis=2)],
        "fl3d_ax2",
        [_vi("X", TensorProto.FLOAT, [1, 3, 4])],
        [_vi("Y", TensorProto.FLOAT, [3, 4])],
    )
    feeds = {"X": _RNG.standard_normal((1, 3, 4)).astype(np.float32)}
    ref, got, tr = _apply_and_run(model, feeds)
    _save(model, tr, "flatten_3d_ax2")

    assert got["Y"].shape == (3, 4), f"Expected (3,4), got {got['Y'].shape}"
    _assert_flatten_chain(tr, expect_unsqueeze=True)
    np.testing.assert_allclose(ref["Y"], got["Y"], rtol=1e-5, atol=1e-6)


def test_flatten_2d_axis0():
    """Flatten [8,16] axis=0: outer=1, inner=128 → output [1,128]; ORT parity."""
    model = _make_model(
        [helper.make_node("Flatten", ["X"], ["Y"], name="fl", axis=0)],
        "fl2d_ax0",
        [_vi("X", TensorProto.FLOAT, [8, 16])],
        [_vi("Y", TensorProto.FLOAT, [1, 128])],
    )
    feeds = {"X": _RNG.standard_normal((8, 16)).astype(np.float32)}
    ref, got, tr = _apply_and_run(model, feeds)
    _save(model, tr, "flatten_2d_ax0")

    assert got["Y"].shape == (1, 128), f"Expected (1,128), got {got['Y'].shape}"
    _assert_flatten_chain(tr, expect_unsqueeze=True)
    np.testing.assert_allclose(ref["Y"], got["Y"], rtol=1e-5, atol=1e-6)


def test_flatten_2d_axis1():
    """Flatten [8,16] axis=1: outer=8, inner=16 → output [8,16]; ORT parity."""
    model = _make_model(
        [helper.make_node("Flatten", ["X"], ["Y"], name="fl", axis=1)],
        "fl2d_ax1",
        [_vi("X", TensorProto.FLOAT, [8, 16])],
        [_vi("Y", TensorProto.FLOAT, [8, 16])],
    )
    feeds = {"X": _RNG.standard_normal((8, 16)).astype(np.float32)}
    ref, got, tr = _apply_and_run(model, feeds)
    _save(model, tr, "flatten_2d_ax1")

    assert got["Y"].shape == (8, 16), f"Expected (8,16), got {got['Y'].shape}"
    _assert_flatten_chain(tr, expect_unsqueeze=True)
    np.testing.assert_allclose(ref["Y"], got["Y"], rtol=1e-5, atol=1e-6)


def test_flatten_3d_axis1():
    """Flatten [3,8,16] axis=1: outer=3, inner=128 → output [3,128]; ORT parity."""
    model = _make_model(
        [helper.make_node("Flatten", ["X"], ["Y"], name="fl", axis=1)],
        "fl3d_ax1",
        [_vi("X", TensorProto.FLOAT, [3, 8, 16])],
        [_vi("Y", TensorProto.FLOAT, [3, 128])],
    )
    feeds = {"X": _RNG.standard_normal((3, 8, 16)).astype(np.float32)}
    ref, got, tr = _apply_and_run(model, feeds)
    _save(model, tr, "flatten_3d_ax1")

    assert got["Y"].shape == (3, 128), f"Expected (3,128), got {got['Y'].shape}"
    _assert_flatten_chain(tr, expect_unsqueeze=True)
    np.testing.assert_allclose(ref["Y"], got["Y"], rtol=1e-5, atol=1e-6)


def test_flatten_3d_neg_axis():
    """Flatten [3,8,16] axis=-1 (→2): outer=24, inner=16 → output [24,16]; ORT parity."""
    model = _make_model(
        [helper.make_node("Flatten", ["X"], ["Y"], name="fl", axis=-1)],
        "fl3d_neg_ax",
        [_vi("X", TensorProto.FLOAT, [3, 8, 16])],
        [_vi("Y", TensorProto.FLOAT, [24, 16])],
    )
    feeds = {"X": _RNG.standard_normal((3, 8, 16)).astype(np.float32)}
    ref, got, tr = _apply_and_run(model, feeds)
    _save(model, tr, "flatten_3d_neg_ax")

    assert got["Y"].shape == (24, 16), f"Expected (24,16), got {got['Y'].shape}"
    _assert_flatten_chain(tr, expect_unsqueeze=True)
    np.testing.assert_allclose(ref["Y"], got["Y"], rtol=1e-5, atol=1e-6)


def test_flatten_4d_axis1_no_unsqueeze():
    """Flatten [1,3,8,16] axis=1: already 4D; no Unsqueeze inserted; output [1,384]; ORT parity."""
    model = _make_model(
        [helper.make_node("Flatten", ["X"], ["Y"], name="fl", axis=1)],
        "fl4d_ax1",
        [_vi("X", TensorProto.FLOAT, [1, 3, 8, 16])],
        [_vi("Y", TensorProto.FLOAT, [1, 384])],
    )
    feeds = {"X": _RNG.standard_normal((1, 3, 8, 16)).astype(np.float32)}
    ref, got, tr = _apply_and_run(model, feeds)
    _save(model, tr, "flatten_4d_ax1")

    assert got["Y"].shape == (1, 384), f"Expected (1,384), got {got['Y'].shape}"
    _assert_flatten_chain(tr, expect_unsqueeze=False)
    np.testing.assert_allclose(ref["Y"], got["Y"], rtol=1e-5, atol=1e-6)
