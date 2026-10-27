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
from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper
from onnx.reference import ReferenceEvaluator

from modelopt.onnx.graph_surgery.dla_transforms.dla_convert_ops_to_4d import (
    _apply_convert_ops_to_4d,
)

ort = pytest.importorskip("onnxruntime", reason="onnxruntime required for ORT parity checks")

_REPO_ROOT = Path(__file__).resolve().parents[5]
_OPSET_VERSION = 17
_MAX_IR_VERSION_FOR_ORT = 9


# ── Shared utilities ────────────────────────────────────────────────────────────────────────────────


def _init(name: str, arr: np.ndarray) -> onnx.TensorProto:
    """Build a named fixed tensor for a synthetic graph."""
    return numpy_helper.from_array(arr, name=name)


def _vi(name: str, elem_type: int, shape: list) -> onnx.ValueInfoProto:
    """Declare tensor metadata for a synthetic graph."""
    return helper.make_tensor_value_info(name, elem_type, shape)


def _clamp_ir(model: onnx.ModelProto) -> None:
    """Use the model IR supported by the installed CPU runtime."""
    model.ir_version = min(model.ir_version, _MAX_IR_VERSION_FOR_ORT)


def _make_model(nodes, name, inputs, outputs, inits=None, *, infer_symbolic_shapes=True):
    """Build and check a model with optional symbolic shape inference."""
    graph = helper.make_graph(nodes, name, inputs, outputs, initializer=inits or [])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", _OPSET_VERSION)])
    _clamp_ir(model)
    if infer_symbolic_shapes:
        try:
            # Shape inference is optional in partial runtime installations.
            from onnxruntime.tools.symbolic_shape_infer import (
                SymbolicShapeInference,  # type: ignore[import-untyped]
            )

        except ImportError:
            pass
        else:
            model = SymbolicShapeInference.infer_shapes(model, auto_merge=True) or model
    onnx.checker.check_model(model)
    return model


def _run_ort(model: onnx.ModelProto, feeds: dict) -> dict:
    """Execute all public outputs on the CPU runtime."""
    _clamp_ir(model)
    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    return {o.name: sess.run([o.name], feeds)[0] for o in model.graph.output}


@pytest.mark.parametrize("operation", ["Concat", "TopK"])
def test_axis_operator_with_unknown_data_rank_stays_native(operation):
    """Missing intermediate data metadata must not select a control rank or partially promote data."""
    nodes = [helper.make_node("Identity", ["x"], ["hidden"])]
    inputs = [_vi("x", TensorProto.FLOAT, [2, 3])]
    feeds = {"x": np.arange(6, dtype=np.float32).reshape(2, 3)}
    if operation == "Concat":
        inputs.append(_vi("other", TensorProto.FLOAT, [2, 3]))
        feeds["other"] = -feeds["x"]
        nodes.append(helper.make_node("Concat", ["other", "hidden"], ["y"], axis=1))
        outputs = [_vi("y", TensorProto.FLOAT, [2, 6])]
    else:
        inputs.append(_vi("k", TensorProto.INT64, [1]))
        feeds["k"] = np.array([2], np.int64)
        nodes.append(helper.make_node("TopK", ["hidden", "k"], ["y", "indices"], axis=1))
        outputs = [_vi("y", TensorProto.FLOAT, [2, 2]), _vi("indices", TensorProto.INT64, [2, 2])]
    model = _make_model(nodes, "unknown_axis_data", inputs, outputs, infer_symbolic_shapes=False)
    expected = _run_ort(model, feeds)
    converted = _apply_convert_ops_to_4d(copy.deepcopy(model))
    target = next(n for n in converted.graph.node if n.op_type == operation)
    assert target.SerializeToString() == model.graph.node[-1].SerializeToString()
    onnx.checker.check_model(converted, full_check=True)
    for name, actual in _run_ort(converted, feeds).items():
        np.testing.assert_array_equal(actual, expected[name])


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compare finite tensor directions in double precision."""
    a64, b64 = a.astype(np.float64).ravel(), b.astype(np.float64).ravel()
    na, nb = float(np.linalg.norm(a64)), float(np.linalg.norm(b64))
    if na == 0.0 and nb == 0.0:
        return 1.0
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a64, b64) / (na * nb))


def _assert_match(ref: dict, got: dict) -> None:
    """Check public output shapes and numerical parity."""
    assert ref.keys() == got.keys()
    for name in ref:
        a, b = ref[name], got[name]
        assert a.shape == b.shape, f"{name}: shape {a.shape} vs {b.shape}"
        if a.dtype.kind == "f":
            cos = _cosine_similarity(a, b)
            assert cos >= 1.0 - 1e-5, f"{name}: cosine {cos:.8f}"
            np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-4, err_msg=name)
        else:
            np.testing.assert_array_equal(a, b, err_msg=name)


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
        else:
            np.testing.assert_array_equal(a, b, err_msg=name)


def _apply_and_run(model, feeds):
    """Execute native and converted graphs for parity checks."""
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
    """Check the expected structural Flatten conversion chain."""
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

    assert got["Y"].shape == (1, 384), f"Expected (1,384), got {got['Y'].shape}"
    _assert_flatten_chain(tr, expect_unsqueeze=False)
    np.testing.assert_allclose(ref["Y"], got["Y"], rtol=1e-5, atol=1e-6)


def test_five_dimensional_transpose_without_singleton_is_unchanged():
    """Skip a valid permutation whose native layout cannot collapse to four dimensions."""
    shape = [2, 3, 4, 5, 6]
    model = _make_model(
        [helper.make_node("Transpose", ["X"], ["Y"], perm=[4, 3, 2, 1, 0])],
        "no_unary",
        [_vi("X", TensorProto.FLOAT, shape)],
        [_vi("Y", TensorProto.FLOAT, shape[::-1])],
    )
    before = model.SerializeToString()
    converted = _apply_convert_ops_to_4d(model)
    assert converted.SerializeToString() == before
    feeds = {"X": np.arange(np.prod(shape), dtype=np.float32).reshape(shape)}
    np.testing.assert_array_equal(_run_ort(converted, feeds)["Y"], np.transpose(feeds["X"]))


@pytest.mark.parametrize("target", [[0, -1], [0, 0, -1]])
def test_unresolved_reshape_copied_dimensions_preserve_intermediate_layout(target):
    """Leave copied-zero targets native when intermediate shape metadata is absent."""
    output_shape = [2, 12] if len(target) == 2 else [2, 3, 4]
    model = helper.make_model(
        helper.make_graph(
            [
                helper.make_node("Identity", ["X"], ["hidden_input"]),
                helper.make_node("Reshape", ["hidden_input", "target"], ["hidden_output"]),
                helper.make_node("Identity", ["hidden_output"], ["Y"]),
            ],
            "missing_metadata",
            [_vi("X", TensorProto.FLOAT, [2, 3, 4])],
            [_vi("Y", TensorProto.FLOAT, output_shape)],
            [_init("target", np.asarray(target, np.int64))],
        ),
        opset_imports=[helper.make_opsetid("", 17)],
    )
    _clamp_ir(model)
    onnx.checker.check_model(model, full_check=True)
    before = model.SerializeToString()
    converted = _apply_convert_ops_to_4d(model)
    assert converted.SerializeToString() == before
    feeds = {"X": np.arange(24, dtype=np.float32).reshape(2, 3, 4)}
    np.testing.assert_array_equal(_run_ort(converted, feeds)["Y"], feeds["X"].reshape(output_shape))


@pytest.mark.parametrize("owner", ["private", "public", "peer", "capture"])
def test_gather_cast_preserves_other_owners_and_updates_private_metadata(owner):
    """Retarget only exclusive standard Cast outputs and preserve shared integer tensors."""
    nodes = [
        helper.make_node("Cast", ["source"], ["indices"], to=TensorProto.INT64),
        helper.make_node("Gather", ["X", "indices"], ["Y"], axis=2),
    ]
    outputs = [_vi("Y", TensorProto.FLOAT, [1, 1, 2, 4])]
    inits = []
    if owner == "public":
        outputs.append(_vi("indices", TensorProto.INT64, [2]))
    elif owner == "peer":
        nodes.append(helper.make_node("Identity", ["indices"], ["other"]))
        outputs.append(_vi("other", TensorProto.INT64, [2]))
    elif owner == "capture":
        branch = helper.make_graph(
            [helper.make_node("Identity", ["indices"], ["out"])],
            "branch",
            [],
            [_vi("out", TensorProto.INT64, [2])],
        )
        inits.append(_init("condition", np.asarray(True)))
        nodes.append(
            helper.make_node("If", ["condition"], ["other"], then_branch=branch, else_branch=branch)
        )
        outputs.append(_vi("other", TensorProto.INT64, [2]))
    model = helper.make_model(
        helper.make_graph(
            nodes,
            "cast_owner",
            [_vi("X", TensorProto.FLOAT, [1, 1, 5, 4]), _vi("source", TensorProto.FLOAT, [None])],
            outputs,
            inits,
        ),
        opset_imports=[helper.make_opsetid("", 17)],
    )
    _clamp_ir(model)
    model.graph.value_info.extend([_vi("indices", TensorProto.INT64, [2])])
    onnx.checker.check_model(model, full_check=True)
    feeds = {
        "X": np.arange(20, dtype=np.float32).reshape(1, 1, 5, 4),
        "source": np.asarray([1, 3], np.float32),
    }
    expected = _run_ort(model, feeds)
    converted = _apply_convert_ops_to_4d(copy.deepcopy(model))
    onnx.checker.check_model(converted, full_check=True)
    _assert_match(expected, _run_ort(converted, feeds))
    cast = next(n for n in converted.graph.node if n.op_type == "Cast" and n.output[0] == "indices")
    expected_type = TensorProto.INT32 if owner == "private" else TensorProto.INT64
    assert next(a.i for a in cast.attribute if a.name == "to") == expected_type
    assert (
        next(
            v for v in converted.graph.value_info if v.name == "indices"
        ).type.tensor_type.elem_type
        == expected_type
    )


@pytest.mark.parametrize("axis", [None, 0, -1])
def test_fixed_dequantization_restores_public_and_convolution_bias_ranks(axis):
    """Preserve native DQ outputs needed by public values and rank-one Conv biases."""
    scale = np.asarray(0.25, np.float32) if axis is None else np.asarray([0.25, 0.5], np.float32)
    model = _make_model(
        [
            helper.make_node(
                "DequantizeLinear",
                ["quantized", "scale", "zero"],
                ["bias"],
                **({} if axis is None else {"axis": axis}),
            ),
            helper.make_node("Conv", ["X", "weight", "bias"], ["Y"]),
        ],
        "dq_bias",
        [_vi("X", TensorProto.FLOAT, [1, 1, 2, 2])],
        [_vi("bias", TensorProto.FLOAT, [2]), _vi("Y", TensorProto.FLOAT, [1, 2, 2, 2])],
        [
            _init("quantized", np.asarray([2, 4], np.int8)),
            _init("scale", scale),
            _init("zero", np.zeros(scale.shape, np.int8)),
            _init("weight", np.ones((2, 1, 1, 1), np.float32)),
        ],
    )
    feeds = {"X": np.arange(4, dtype=np.float32).reshape(1, 1, 2, 2)}
    expected = _run_ort(model, feeds)
    converted = _apply_convert_ops_to_4d(copy.deepcopy(model))
    onnx.checker.check_model(converted, full_check=True)
    _assert_match(expected, _run_ort(converted, feeds))
    assert (
        next(n for n in converted.graph.node if n.op_type == "DequantizeLinear").output[0] != "bias"
    )


def test_blocked_dequantization_layout_is_unchanged():
    """Leave blocked quantization native because its scale layout is rank-sensitive."""
    model = helper.make_model(
        helper.make_graph(
            [
                helper.make_node(
                    "DequantizeLinear", ["data", "scale", "zero"], ["Y"], axis=1, block_size=2
                )
            ],
            "blocked",
            [],
            [_vi("Y", TensorProto.FLOAT, [2, 4])],
            [
                _init("data", np.arange(8, dtype=np.int8).reshape(2, 4)),
                _init("scale", np.full((2, 2), 0.25, np.float32)),
                _init("zero", np.zeros((2, 2), np.int8)),
            ],
        ),
        opset_imports=[helper.make_opsetid("", 21)],
    )
    _clamp_ir(model)
    onnx.checker.check_model(model, full_check=True)
    before = model.SerializeToString()
    assert _apply_convert_ops_to_4d(model).SerializeToString() == before


@pytest.mark.parametrize("storage", ["raw", "typed"])
def test_fixed_weight_promotion_preserves_payload_representation_and_metadata(storage):
    """Change only dimension metadata when promoting a fixed elementwise weight."""
    data = np.arange(6, dtype=np.float32).reshape(2, 3)
    tensor = (
        _init("weight", data)
        if storage == "raw"
        else helper.make_tensor("weight", TensorProto.FLOAT, [2, 3], data.reshape(-1))
    )
    tensor.doc_string = "Preserve caller metadata"
    model = _make_model(
        [helper.make_node("Add", ["X", "weight"], ["Y"])],
        "weight_metadata",
        [_vi("X", TensorProto.FLOAT, [2, 3])],
        [_vi("Y", TensorProto.FLOAT, [2, 3])],
        [tensor],
    )
    expected = copy.deepcopy(model.graph.initializer[0])
    expected.ClearField("dims")
    converted = _apply_convert_ops_to_4d(model)
    actual = copy.deepcopy(next(t for t in converted.graph.initializer if t.name == "weight"))
    actual.ClearField("dims")
    assert actual.SerializeToString() == expected.SerializeToString()
    np.testing.assert_array_equal(_run_ort(converted, {"X": data})["Y"], data * 2)


@pytest.mark.parametrize(
    ("operation", "mixed"), [("Relu", False), ("Add", False), ("Add", True), ("Mul", True)]
)
def test_scalar_inputs_preserve_native_values_with_optional_tensor_broadcast(operation, mixed):
    """Choose a nonscalar data layout and leave entirely scalar operations native."""
    inputs = [_vi("scalar", TensorProto.FLOAT, [])]
    node_inputs = ["scalar"]
    feeds = {"scalar": np.asarray(2, np.float32)}
    shape = [2, 3] if mixed else []
    if operation != "Relu":
        inputs.append(_vi("other", TensorProto.FLOAT, shape))
        node_inputs.append("other")
        feeds["other"] = (
            np.arange(6, dtype=np.float32).reshape(shape) if mixed else np.asarray(3, np.float32)
        )
    model = _make_model(
        [helper.make_node(operation, node_inputs, ["Y"])],
        "scalar_layout",
        inputs,
        [_vi("Y", TensorProto.FLOAT, shape)],
    )
    expected = _run_ort(model, feeds)
    converted = _apply_convert_ops_to_4d(copy.deepcopy(model))
    onnx.checker.check_model(converted, full_check=True)
    _assert_match(expected, _run_ort(converted, feeds))


@pytest.mark.parametrize(
    ("shape", "dtype", "opset"), [([], np.float32, 17), ([2, 3], np.float16, 19)]
)
def test_fixed_dequantization_preserves_scalar_and_half_precision_outputs(shape, dtype, opset):
    """Restore native DQ ranks and keep the scale-selected output dtype."""
    data = np.asarray(4, np.int8) if not shape else np.arange(6, dtype=np.int8).reshape(shape)
    model = helper.make_model(
        helper.make_graph(
            [helper.make_node("DequantizeLinear", ["data", "scale", "zero"], ["Y"])],
            "dq_output_types",
            [],
            [_vi("Y", TensorProto.FLOAT if dtype == np.float32 else TensorProto.FLOAT16, shape)],
            [
                _init("data", data),
                _init("scale", np.asarray(0.25, dtype)),
                _init("zero", np.asarray(0, np.int8)),
            ],
        ),
        opset_imports=[helper.make_opsetid("", opset)],
    )
    _clamp_ir(model)
    expected = _run_ort(model, {})
    converted = _apply_convert_ops_to_4d(copy.deepcopy(model))
    onnx.checker.check_model(converted, full_check=True)
    _assert_match(expected, _run_ort(converted, {}))


@pytest.mark.parametrize("storage", ["typed", "external"])
def test_gather_elements_fixed_index_storage_is_preserved(tmp_path, storage):
    """Use runtime index views without decoding unloaded sidecars or replacing tensor metadata."""
    indices = np.array([[0, 2, 1], [1, 0, 2]], np.int64)
    tensor = (
        _init("indices", indices)
        if storage == "external"
        else helper.make_tensor("indices", TensorProto.INT64, [2, 3], indices.reshape(-1).tolist())
    )
    tensor.doc_string = "native index owner"
    model = _make_model(
        [helper.make_node("GatherElements", ["x", "indices"], ["y"], axis=1)],
        "index_storage",
        [_vi("x", TensorProto.FLOAT, [2, 3])],
        [_vi("y", TensorProto.FLOAT, [2, 3])],
        [tensor],
    )
    native = tmp_path / "native.onnx"
    if storage == "external":
        onnx.save_model(
            model,
            str(native),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location="indices.bin",
            size_threshold=0,
        )
    else:
        onnx.save_model(model, str(native))
    feeds = {"x": np.arange(6, dtype=np.float32).reshape(2, 3)}
    expected = ort.InferenceSession(str(native), providers=["CPUExecutionProvider"]).run(
        None, feeds
    )[0]
    unloaded = onnx.load(str(native), load_external_data=False)
    original = unloaded.graph.initializer[0].SerializeToString()
    converted = _apply_convert_ops_to_4d(unloaded)
    assert (
        next(t for t in converted.graph.initializer if t.name == "indices").SerializeToString()
        == original
    )
    target = tmp_path / "converted.onnx"
    onnx.save_model(converted, str(target))
    onnx.checker.check_model(str(target), full_check=True)
    actual = ort.InferenceSession(str(target), providers=["CPUExecutionProvider"]).run(None, feeds)[
        0
    ]
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("operation", ["Gather", "Reshape", "Slice", "ReduceSum", "Expand", "Tile"])
@pytest.mark.parametrize("decoy", [False, True])
def test_unloaded_external_controls_preserve_model_directory(
    tmp_path, monkeypatch, operation, decoy
):
    """Never resolve unloaded control payloads against the process working directory."""
    specifications = {
        "Gather": (["indices"], {"indices": [2, 0]}, [2, 2], {"axis": 1}),
        "Reshape": (["shape"], {"shape": [3, 2]}, [3, 2], {}),
        "Slice": (
            ["starts", "ends", "axes"],
            {"starts": [0], "ends": [2], "axes": [1]},
            [2, 2],
            {},
        ),
        "ReduceSum": (["axes"], {"axes": [1]}, [2, 1], {"keepdims": 1}),
        "Expand": (["shape"], {"shape": [2, 3]}, [2, 3], {}),
        "Tile": (["repeats"], {"repeats": [1, 2]}, [2, 6], {}),
    }
    inputs, controls, output, attributes = specifications[operation]
    model = _make_model(
        [helper.make_node(operation, ["x", *inputs], ["y"], **attributes)],
        "external_controls",
        [_vi("x", TensorProto.FLOAT, [2, 3])],
        [_vi("y", TensorProto.FLOAT, output)],
        [_init(name, np.asarray(values, np.int64)) for name, values in controls.items()],
    )
    native = tmp_path / "native.onnx"
    onnx.save_model(
        model,
        str(native),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="controls.bin",
        size_threshold=0,
    )
    payload = (tmp_path / "controls.bin").read_bytes()
    feeds = {"x": np.arange(6, dtype=np.float32).reshape(2, 3)}
    expected = _run_ort(onnx.load(str(native)), feeds)["y"]
    unloaded = onnx.load(str(native), load_external_data=False)
    originals = {tensor.name: tensor.SerializeToString() for tensor in unloaded.graph.initializer}
    working = tmp_path / "unrelated"
    working.mkdir()
    if decoy:
        (working / "controls.bin").write_bytes(bytes(len(payload)))
    monkeypatch.chdir(working)
    converted = _apply_convert_ops_to_4d(unloaded)
    assert {
        tensor.name: tensor.SerializeToString()
        for tensor in converted.graph.initializer
        if tensor.name in originals
    } == originals
    if operation == "Gather":
        assert any(node.op_type == "Cast" for node in converted.graph.node)
    else:
        assert len(converted.graph.node) == 1
    target = tmp_path / "converted.onnx"
    onnx.save_model(converted, str(target))
    loaded = onnx.load(str(target))
    onnx.checker.check_model(loaded, full_check=True)
    actual = _run_ort(loaded, feeds)["y"]
    np.testing.assert_array_equal(actual, expected)
    assert (tmp_path / "controls.bin").read_bytes() == payload


def test_external_weight_promotion_preserves_sidecar_bytes_and_relative_metadata(tmp_path):
    """Promote unloaded external weights without reading or rewriting their payload sidecar."""
    weight = np.arange(6, dtype=np.float32).reshape(2, 3)
    model = _make_model(
        [helper.make_node("Add", ["X", "weight"], ["Y"])],
        "external_weight",
        [_vi("X", TensorProto.FLOAT, [2, 3])],
        [_vi("Y", TensorProto.FLOAT, [2, 3])],
        [_init("weight", weight)],
    )
    native = tmp_path / "native.onnx"
    onnx.save_model(
        model,
        str(native),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="weights.bin",
        size_threshold=0,
    )
    payload = (tmp_path / "weights.bin").read_bytes()
    feeds = {"X": weight}
    expected = ort.InferenceSession(str(native), providers=["CPUExecutionProvider"]).run(
        None, feeds
    )[0]
    unloaded = onnx.load(str(native), load_external_data=False)
    converted = _apply_convert_ops_to_4d(unloaded)
    target = tmp_path / "converted.onnx"
    onnx.save_model(converted, str(target))
    onnx.checker.check_model(str(target), full_check=True)
    actual = ort.InferenceSession(str(target), providers=["CPUExecutionProvider"]).run(None, feeds)[
        0
    ]
    np.testing.assert_array_equal(actual, expected)
    assert (tmp_path / "weights.bin").read_bytes() == payload
    tensor = next(t for t in converted.graph.initializer if t.name == "weight")
    assert tensor.data_location == TensorProto.EXTERNAL and not tensor.HasField("raw_data")
    assert {entry.key: entry.value for entry in tensor.external_data}["location"] == "weights.bin"


@pytest.mark.parametrize("keepdims", [0, 1])
def test_scalar_reduction_preserves_native_rank(keepdims):
    """Promote a scalar reduction internally and restore its zero-dimensional output."""
    model = _make_model(
        [helper.make_node("ReduceSum", ["X"], ["Y"], keepdims=keepdims)],
        "scalar_reduce",
        [_vi("X", TensorProto.FLOAT, [])],
        [_vi("Y", TensorProto.FLOAT, [])],
    )
    feeds = {"X": np.asarray(3, np.float32)}
    expected = _run_ort(model, feeds)
    converted = _apply_convert_ops_to_4d(copy.deepcopy(model))
    onnx.checker.check_model(converted, full_check=True)
    _assert_match(expected, _run_ort(converted, feeds))


@pytest.mark.parametrize("axes", [[0], [0, 1]])
def test_runtime_reduction_axes_remain_native(axes):
    """Leave unsupported runtime axes unchanged instead of reducing all promoted dimensions."""
    # The optional symbolic inferencer cannot handle runtime axes with keepdims=0.
    model = helper.make_model(
        helper.make_graph(
            [helper.make_node("ReduceSum", ["X", "axes"], ["Y"], keepdims=0)],
            "runtime_reduce",
            [_vi("X", TensorProto.FLOAT, [1, 1, 1]), _vi("axes", TensorProto.INT64, [len(axes)])],
            [_vi("Y", TensorProto.FLOAT, [1] * (3 - len(axes)))],
        ),
        opset_imports=[helper.make_opsetid("", 17)],
    )
    _clamp_ir(model)
    feeds = {"X": np.asarray([[[3]]], np.float32), "axes": np.asarray(axes, np.int64)}
    expected = _run_ort(model, feeds)
    before = model.SerializeToString()
    converted = _apply_convert_ops_to_4d(model)
    assert converted.SerializeToString() == before
    _assert_match(expected, _run_ort(converted, feeds))


@pytest.mark.parametrize("owner", ["default", "capture"])
def test_fixed_dequantization_preserves_runtime_and_captured_data_owners(owner):
    """Keep native quantized layouts owned by callers or nested graphs."""
    data = np.arange(6, dtype=np.int8).reshape(2, 3)
    inputs = [_vi("data", TensorProto.INT8, [2, 3])] if owner == "default" else []
    nodes = [helper.make_node("DequantizeLinear", ["data", "scale", "zero"], ["Y"])]
    outputs = [_vi("Y", TensorProto.FLOAT, [2, 3])]
    inits = [
        _init("data", data),
        _init("scale", np.asarray(0.25, np.float32)),
        _init("zero", np.asarray(0, np.int8)),
    ]
    if owner == "capture":
        branch = helper.make_graph(
            [helper.make_node("Identity", ["data"], ["out"])],
            "branch",
            [],
            [_vi("out", TensorProto.INT8, [2, 3])],
        )
        inits.append(_init("condition", np.asarray(True)))
        nodes.append(
            helper.make_node("If", ["condition"], ["view"], then_branch=branch, else_branch=branch)
        )
        outputs.append(_vi("view", TensorProto.INT8, [2, 3]))
    model = _make_model(nodes, "dq_owners", inputs, outputs, inits)
    feeds = {"data": data + np.int8(10)} if owner == "default" else {}
    expected = _run_ort(model, feeds)
    before = model.SerializeToString()
    converted = _apply_convert_ops_to_4d(model)
    assert converted.SerializeToString() == before
    _assert_match(expected, _run_ort(converted, feeds))


@pytest.mark.parametrize("operation", ["Transpose", "Gather", "DequantizeLinear"])
def test_standalone_converter_preserves_opaque_layout_functions(operation):
    """Leave opaque functions intact even when their names match supported handlers."""
    attrs = {"perm": [1, 0]} if operation == "Transpose" else {}
    inputs = [] if operation == "DequantizeLinear" else [_vi("X", TensorProto.FLOAT, [2, 3])]
    inits = [_init("X", np.arange(6, dtype=np.float32).reshape(2, 3))] if not inputs else []
    function = helper.make_function(
        "opaque",
        operation,
        ["x"],
        ["y"],
        [helper.make_node("Neg", ["x"], ["y"])],
        [helper.make_opsetid("", 17)],
        attributes=list(attrs),
    )
    model = helper.make_model(
        helper.make_graph(
            [helper.make_node(operation, ["X"], ["Y"], domain="opaque", **attrs)],
            "opaque_layout",
            inputs,
            [_vi("Y", TensorProto.FLOAT, [2, 3])],
            inits,
        ),
        opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("opaque", 1)],
        functions=[function],
    )
    _clamp_ir(model)
    model = onnx.shape_inference.infer_shapes(model)
    feeds = {"X": np.arange(6, dtype=np.float32).reshape(2, 3)} if inputs else {}
    expected = ReferenceEvaluator(model).run(None, feeds)
    before = model.SerializeToString()
    converted = _apply_convert_ops_to_4d(model)
    assert converted.SerializeToString() == before
    onnx.checker.check_model(converted, full_check=True)
    np.testing.assert_array_equal(ReferenceEvaluator(converted).run(None, feeds)[0], expected[0])


def test_gather_does_not_retarget_an_opaque_cast_producer():
    """Prepare a standard private Cast while preserving the opaque producer's attributes."""
    function = helper.make_function(
        "opaque",
        "Cast",
        ["x"],
        ["y"],
        [helper.make_node("Neg", ["x"], ["y"])],
        [helper.make_opsetid("", 17)],
        attributes=["to"],
    )
    model = helper.make_model(
        helper.make_graph(
            [
                helper.make_node(
                    "Cast", ["source"], ["indices"], domain="opaque", to=TensorProto.INT64
                ),
                helper.make_node("Gather", ["X", "indices"], ["Y"], axis=2),
            ],
            "opaque_cast",
            [_vi("source", TensorProto.INT64, [2]), _vi("X", TensorProto.FLOAT, [1, 1, 5, 4])],
            [_vi("Y", TensorProto.FLOAT, [1, 1, 2, 4]), _vi("indices", TensorProto.INT64, [2])],
        ),
        opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("opaque", 1)],
        functions=[function],
    )
    _clamp_ir(model)
    model = onnx.shape_inference.infer_shapes(model)
    feeds = {
        "source": np.asarray([1, 3], np.int64),
        "X": np.arange(20, dtype=np.float32).reshape(1, 1, 5, 4),
    }
    expected = ReferenceEvaluator(model).run(None, feeds)
    producer = model.graph.node[0].SerializeToString()
    converted = _apply_convert_ops_to_4d(model)
    assert converted.graph.node[0].SerializeToString() == producer
    onnx.checker.check_model(converted, full_check=True)
    for actual, reference in zip(ReferenceEvaluator(converted).run(None, feeds), expected):
        np.testing.assert_array_equal(actual, reference)


@pytest.mark.parametrize("omit_mean", [False, True])
def test_layernorm_restores_optional_statistics_and_shared_public_views(omit_mean):
    """Restore every optional normalization output and its internal consumers to native rank."""
    names = ["Y", "" if omit_mean else "Mean", "Inv"]
    statistic = "Inv" if omit_mean else "Mean"
    outputs = [
        _vi("Y", TensorProto.FLOAT, [2, 3, 4]),
        _vi("Inv", TensorProto.FLOAT, [2, 1, 1]),
        _vi("view", TensorProto.FLOAT, [2, 1, 1]),
    ]
    if not omit_mean:
        outputs.insert(1, _vi("Mean", TensorProto.FLOAT, [2, 1, 1]))
    model = _make_model(
        [
            helper.make_node("LayerNormalization", ["X", "scale", "bias"], names, axis=1),
            helper.make_node("Add", [statistic, "zero"], ["view"]),
        ],
        "layernorm_statistics",
        [_vi("X", TensorProto.FLOAT, [2, 3, 4])],
        outputs,
        [
            _init("scale", np.ones((3, 4), np.float32)),
            _init("bias", np.zeros((3, 4), np.float32)),
            _init("zero", np.asarray(0, np.float32)),
        ],
        # The optional inferencer cannot handle an omitted middle output.
        infer_symbolic_shapes=not omit_mean,
    )
    feeds = {"X": np.random.default_rng(31).standard_normal((2, 3, 4)).astype(np.float32)}
    expected = _run_ort(model, feeds)
    converted = _apply_convert_ops_to_4d(copy.deepcopy(model))
    onnx.checker.check_model(converted, full_check=True)
    _assert_match(expected, _run_ort(converted, feeds))


def test_generated_rank_adapters_preserve_existing_control_value_and_node_names():
    """Allocate around simultaneous control, tensor and node-name collisions."""
    model = _make_model(
        [
            helper.make_node("Relu", ["X"], ["Y"], name="relu"),
            helper.make_node("Identity", ["other"], ["relu_unsq4d_0"], name="relu_sq_node_1"),
        ],
        "adapter_names",
        [_vi("X", TensorProto.FLOAT, [2, 3]), _vi("other", TensorProto.FLOAT, [1])],
        [
            _vi("Y", TensorProto.FLOAT, [2, 3]),
            _vi("relu_unsq4d_0", TensorProto.FLOAT, [1]),
            _vi("relu_unsq_axes_0", TensorProto.INT64, [1]),
        ],
        [_init("relu_unsq_axes_0", np.asarray([99], np.int64))],
    )
    feeds = {
        "X": np.arange(6, dtype=np.float32).reshape(2, 3) - 3,
        "other": np.asarray([17], np.float32),
    }
    expected = _run_ort(model, feeds)
    converted = _apply_convert_ops_to_4d(copy.deepcopy(model))
    onnx.checker.check_model(converted, full_check=True)
    _assert_match(expected, _run_ort(converted, feeds))


@pytest.mark.parametrize("collision", [False, True])
def test_shared_squeeze_preserves_partial_rank_metadata_and_existing_aliases(collision):
    """Copy the native partial shape while allocating private consumer aliases."""
    nodes = [
        helper.make_node("Squeeze", ["X", "axes"], ["Y"], name="sq"),
        helper.make_node("Identity", ["Y"], ["Z"]),
    ]
    inputs = [_vi("X", TensorProto.FLOAT, [1, None, 3, None])]
    outputs = [
        _vi("Y", TensorProto.FLOAT, [None, 3, None]),
        _vi("Z", TensorProto.FLOAT, [None, 3, None]),
    ]
    feeds = {"X": np.arange(24, dtype=np.float32).reshape(1, 2, 3, 4)}
    if collision:
        nodes.append(helper.make_node("Identity", ["other"], ["Y_sq"], name="sq_internal"))
        inputs.append(_vi("other", TensorProto.FLOAT, [1]))
        outputs.append(_vi("Y_sq", TensorProto.FLOAT, [1]))
        feeds["other"] = np.asarray([17], np.float32)
    model = helper.make_model(
        helper.make_graph(
            nodes, "partial_squeeze", inputs, outputs, [_init("axes", np.asarray([0], np.int64))]
        ),
        opset_imports=[helper.make_opsetid("", 17)],
    )
    _clamp_ir(model)
    expected = _run_ort(model, feeds)
    converted = _apply_convert_ops_to_4d(copy.deepcopy(model))
    onnx.checker.check_model(converted, full_check=True)
    _assert_match(expected, _run_ort(converted, feeds))
    alias = next(
        n.input[0] for n in converted.graph.node if n.op_type == "Identity" and n.output[0] == "Z"
    )
    metadata = copy.deepcopy(next(v for v in converted.graph.value_info if v.name == alias))
    metadata.name = "Y"
    assert metadata.SerializeToString() == model.graph.output[0].SerializeToString()


def test_opaque_squeeze_with_shared_public_output_is_not_split_as_a_standard_operator():
    """Keep opaque function semantics intact when its public result also feeds internal nodes."""
    function = helper.make_function(
        "opaque",
        "Squeeze",
        ["x"],
        ["y"],
        [helper.make_node("Neg", ["x"], ["y"])],
        [helper.make_opsetid("", 17)],
    )
    model = helper.make_model(
        helper.make_graph(
            [
                helper.make_node("Squeeze", ["X"], ["Y"], domain="opaque"),
                helper.make_node("Relu", ["Y"], ["Z"]),
            ],
            "opaque_shared_squeeze",
            [_vi("X", TensorProto.FLOAT, [1, 2, 3])],
            [_vi("Y", TensorProto.FLOAT, [1, 2, 3]), _vi("Z", TensorProto.FLOAT, [1, 2, 3])],
        ),
        opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("opaque", 1)],
        functions=[function],
    )
    _clamp_ir(model)
    model = onnx.shape_inference.infer_shapes(model)
    feeds = {"X": np.arange(6, dtype=np.float32).reshape(1, 2, 3) - 3}
    expected = ReferenceEvaluator(model).run(None, feeds)
    opaque = model.graph.node[0].SerializeToString()
    converted = _apply_convert_ops_to_4d(model)
    assert converted.graph.node[0].SerializeToString() == opaque
    onnx.checker.check_model(converted, full_check=True)
    for actual, reference in zip(ReferenceEvaluator(converted).run(None, feeds), expected):
        np.testing.assert_array_equal(actual, reference)


@pytest.mark.parametrize("transpose", [False, True])
def test_five_dimensional_reshape_restores_native_public_and_downstream_layouts(transpose):
    """Restore the original five-dimensional contract after computing with a collapsed target."""
    shape = [1, 2, 3, 4, 5]
    nodes = [helper.make_node("Reshape", ["X", "target"], ["middle" if transpose else "Y"])]
    output_shape = shape
    if transpose:
        nodes.append(helper.make_node("Transpose", ["middle"], ["Y"], perm=[1, 2, 3, 4, 0]))
        output_shape = [2, 3, 4, 5, 1]
    model = _make_model(
        nodes,
        "reshape_five",
        [_vi("X", TensorProto.FLOAT, [2, 3, 4, 5])],
        [_vi("Y", TensorProto.FLOAT, output_shape)],
        [_init("target", np.asarray(shape, np.int64))],
    )
    feeds = {"X": np.arange(120, dtype=np.float32).reshape(2, 3, 4, 5)}
    expected = _run_ort(model, feeds)
    converted = _apply_convert_ops_to_4d(copy.deepcopy(model))
    onnx.checker.check_model(converted, full_check=True)
    _assert_match(expected, _run_ort(converted, feeds))


@pytest.mark.parametrize(
    ("operation", "control"), [("Slice", [0]), ("Slice", [1]), ("Expand", [2, 3]), ("Tile", [2, 1])]
)
def test_runtime_structural_controls_preserve_native_intermediate_layouts(operation, control):
    """Skip unsupported runtime controls even when their intermediate metadata is absent."""
    shape = [1, 3] if operation == "Expand" else [2, 3]
    inits = []
    if operation == "Slice":
        names = ["X", "starts", "ends", "control"]
        inits = [
            _init("starts", np.asarray([1], np.int64)),
            _init("ends", np.asarray([2], np.int64)),
        ]
    else:
        names = ["X", "control"]
    model = helper.make_model(
        helper.make_graph(
            [
                helper.make_node(operation, names, ["hidden"]),
                helper.make_node("Identity", ["hidden"], ["Y"]),
            ],
            "runtime_structural",
            [_vi("X", TensorProto.FLOAT, shape), _vi("control", TensorProto.INT64, [len(control)])],
            [_vi("Y", TensorProto.FLOAT, [None, None])],
            inits,
        ),
        opset_imports=[helper.make_opsetid("", 17)],
    )
    _clamp_ir(model)
    feeds = {
        "X": np.arange(np.prod(shape), dtype=np.float32).reshape(shape),
        "control": np.asarray(control, np.int64),
    }
    expected = _run_ort(model, feeds)
    before = model.SerializeToString()
    converted = _apply_convert_ops_to_4d(model)
    assert converted.SerializeToString() == before
    _assert_match(expected, _run_ort(converted, feeds))


def test_fixed_expand_derives_output_rank_without_intermediate_metadata():
    """Restore the broadcast output rank from fixed control values and the native input shape."""
    model = helper.make_model(
        helper.make_graph(
            [
                helper.make_node("Expand", ["X", "target"], ["hidden"]),
                helper.make_node("Identity", ["hidden"], ["Y"]),
            ],
            "expand_without_metadata",
            [_vi("X", TensorProto.FLOAT, [1, 3])],
            [_vi("Y", TensorProto.FLOAT, [2, 2, 3])],
            [_init("target", np.asarray([2, 2, 3], np.int64))],
        ),
        opset_imports=[helper.make_opsetid("", 17)],
    )
    _clamp_ir(model)
    feeds = {"X": np.arange(3, dtype=np.float32).reshape(1, 3)}
    expected = _run_ort(model, feeds)
    converted = _apply_convert_ops_to_4d(copy.deepcopy(model))
    onnx.checker.check_model(converted, full_check=True)
    _assert_match(expected, _run_ort(converted, feeds))


@pytest.mark.parametrize("operation", ["Relu", "Softmax"])
def test_legacy_opset_without_input_axes_adapters_stays_native(operation):
    """Preserve legacy schemas instead of inserting newer Squeeze/Unsqueeze inputs."""
    model = helper.make_model(
        helper.make_graph(
            [helper.make_node(operation, ["x"], ["y"])],
            "legacy_adapter_schema",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 3])],
        ),
        opset_imports=[helper.make_opsetid("", 11)],
    )
    model.ir_version = 9
    original = model.SerializeToString()
    _apply_convert_ops_to_4d(model)
    assert model.SerializeToString() == original
    onnx.checker.check_model(model, full_check=True)
    data = np.arange(6, dtype=np.float32).reshape(2, 3)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    output = ort.InferenceSession(
        model.SerializeToString(), options, providers=["CPUExecutionProvider"]
    ).run(None, {"x": data})[0]
    assert output.shape == data.shape


@pytest.mark.parametrize("fixed", [False, True])
@pytest.mark.parametrize(
    ("shape", "index_shape", "axis", "supported"),
    [
        ([8, 16], [2, 1], 0, False),
        ([5, 4, 3], [1, 2], 1, False),
        ([2, 3, 4], [1, 1, 2], 1, False),
        ([8, 16], [1, 2], 0, True),
        ([2, 3], [0], 0, True),
    ],
)
def test_gather_requires_exact_native_output_restoration(
    fixed, shape, index_shape, axis, supported
):
    """Flatten only layouts whose public native output can be restored exactly."""
    indices = np.arange(np.prod(index_shape), dtype=np.int64).reshape(index_shape) % shape[axis]
    inputs = [_vi("x", TensorProto.FLOAT, shape)]
    initializers = []
    feeds = {"x": np.arange(np.prod(shape), dtype=np.float32).reshape(shape)}
    if fixed:
        initializers.append(_init("indices", indices))
    else:
        inputs.append(_vi("indices", TensorProto.INT64, index_shape))
        feeds["indices"] = indices
    native = shape[:axis] + index_shape + shape[axis + 1 :]
    model = _make_model(
        [helper.make_node("Gather", ["x", "indices"], ["y"], axis=axis)],
        "gather_layout",
        inputs,
        [_vi("y", TensorProto.FLOAT, native)],
        initializers,
    )
    original = model.SerializeToString()
    expected = _run_ort(copy.deepcopy(model), feeds)
    converted = _apply_convert_ops_to_4d(copy.deepcopy(model))
    if not supported:
        assert converted.SerializeToString() == original
    else:
        assert any(node.op_type == "Unsqueeze" for node in converted.graph.node)
    onnx.checker.check_model(converted, full_check=True)
    _assert_match(expected, _run_ort(converted, feeds))


@pytest.mark.parametrize("operation", ["Add", "Mul", "Where"])
@pytest.mark.parametrize("fixed", [False, True])
def test_generic_broadcast_output_without_intermediate_metadata(operation, fixed):
    """Restore the maximum broadcast rank including fixed data branches."""
    inputs = [_vi("a", TensorProto.FLOAT, [3])]
    initializers = []
    feeds = {"a": np.asarray([1, 2, 3], np.float32)}
    matrix = np.arange(6, dtype=np.float32).reshape(2, 3)
    if fixed:
        initializers.append(_init("b", matrix))
    else:
        inputs.append(_vi("b", TensorProto.FLOAT, [2, 3]))
        feeds["b"] = matrix
    operands = ["a", "b"]
    if operation == "Where":
        inputs.append(_vi("condition", TensorProto.BOOL, []))
        feeds["condition"] = np.asarray(False)
        operands.insert(0, "condition")
    model = helper.make_model(
        helper.make_graph(
            [
                helper.make_node(operation, operands, ["hidden"]),
                helper.make_node("Relu", ["hidden"], ["y"]),
            ],
            "broadcast_without_metadata",
            inputs,
            [_vi("y", TensorProto.FLOAT, [2, 3])],
            initializers,
        ),
        opset_imports=[helper.make_opsetid("", 17)],
    )
    _clamp_ir(model)
    assert not model.graph.value_info
    expected = _run_ort(copy.deepcopy(model), feeds)
    converted = _apply_convert_ops_to_4d(copy.deepcopy(model))
    onnx.checker.check_model(converted, full_check=True)
    _assert_match(expected, _run_ort(converted, feeds))


@pytest.mark.parametrize("owner", ["peer", "public", "capture", "two_gathers", "default", "legacy"])
def test_gather_preserves_four_dimensional_initializer_owners(owner):
    """Keep the original index type, shape, and runtime override semantics for every owner."""
    indices = np.asarray([0, 3], np.int64).reshape(1, 1, 1, 2)
    nodes = [helper.make_node("Gather", ["x", "indices"], ["y"], axis=0)]
    inputs = [_vi("x", TensorProto.FLOAT, [8])]
    outputs = [_vi("y", TensorProto.FLOAT, [1, 1, 1, 2])]
    if owner == "peer":
        nodes.append(helper.make_node("Identity", ["indices"], ["peer"]))
        outputs.append(_vi("peer", TensorProto.INT64, [1, 1, 1, 2]))
    elif owner == "public":
        outputs.append(_vi("indices", TensorProto.INT64, [1, 1, 1, 2]))
    elif owner == "two_gathers":
        nodes.append(helper.make_node("Gather", ["x", "indices"], ["z"], axis=0))
        outputs.append(_vi("z", TensorProto.FLOAT, [1, 1, 1, 2]))
    elif owner == "capture":
        branch = helper.make_graph(
            [helper.make_node("Identity", ["indices"], ["branch_result"])],
            "capture",
            [],
            [_vi("branch_result", TensorProto.INT64, [1, 1, 1, 2])],
        )
        nodes.append(
            helper.make_node(
                "If", ["condition"], ["captured"], then_branch=branch, else_branch=branch
            )
        )
        inputs.append(_vi("condition", TensorProto.BOOL, []))
        outputs.append(_vi("captured", TensorProto.INT64, [1, 1, 1, 2]))
    if owner in {"default", "legacy"}:
        inputs.append(_vi("indices", TensorProto.INT64, [1, 1, 1, 2]))
    model = _make_model(
        nodes,
        "owned_gather_indices",
        inputs,
        outputs,
        [_init("indices", indices)],
        infer_symbolic_shapes=False,
    )
    if owner == "legacy":
        model.ir_version = 3
    original = model.graph.initializer[0].SerializeToString()
    feeds = {"x": np.arange(8, dtype=np.float32)}
    if owner == "capture":
        feeds["condition"] = np.asarray(True)
    cases = [feeds]
    if owner == "default":
        cases.append({**feeds, "indices": np.asarray([7, 5], np.int64).reshape(1, 1, 1, 2)})
    expected = [_run_ort(copy.deepcopy(model), case) for case in cases]
    converted = _apply_convert_ops_to_4d(copy.deepcopy(model))
    assert (
        next(t for t in converted.graph.initializer if t.name == "indices").SerializeToString()
        == original
    )
    onnx.checker.check_model(converted, full_check=True)
    for case, reference in zip(cases, expected):
        _assert_match(reference, _run_ort(converted, case))
    if owner == "two_gathers":
        gathers = [node for node in converted.graph.node if node.op_type == "Gather"]
        assert gathers[0].input[1] == gathers[1].input[1]


@pytest.mark.parametrize("missing", ["indices", "output", "both", "unequal_extents"])
@pytest.mark.parametrize("fixed", [False, True])
def test_gather_elements_restores_native_rank_without_intermediate_metadata(missing, fixed):
    """Derive both adapter ranks from the schema, including unequal non-axis extents."""
    data_shape, index_shape, axis = [2, 3], [2, 3], 1
    if missing == "unequal_extents":
        data_shape, index_shape, axis = [2, 1, 4], [1, 2, 4], 1
    indices = np.zeros(index_shape, np.int64)
    nodes = []
    inputs = [_vi("x", TensorProto.FLOAT, data_shape)]
    initializers = []
    if fixed:
        initializers.append(_init("indices", indices))
    else:
        inputs.append(_vi("indices", TensorProto.INT64, index_shape))
    index_name = "indices"
    if missing in {"indices", "both"}:
        nodes.append(helper.make_node("Identity", ["indices"], ["hidden_indices"]))
        index_name = "hidden_indices"
    result_name = "hidden_result" if missing in {"output", "both", "unequal_extents"} else "y"
    nodes.append(helper.make_node("GatherElements", ["x", index_name], [result_name], axis=axis))
    if result_name != "y":
        nodes.append(helper.make_node("Identity", [result_name], ["y"]))
    model = _make_model(
        nodes,
        "missing_gather_elements_metadata",
        inputs,
        [_vi("y", TensorProto.FLOAT, index_shape)],
        initializers,
        infer_symbolic_shapes=False,
    )
    feeds = {"x": np.arange(np.prod(data_shape), dtype=np.float32).reshape(data_shape)}
    if not fixed:
        feeds["indices"] = indices
    expected = _run_ort(copy.deepcopy(model), feeds)
    converted = _apply_convert_ops_to_4d(copy.deepcopy(model))
    onnx.checker.check_model(converted, full_check=True)
    _assert_match(expected, _run_ort(converted, feeds))


def test_expand_with_unknown_native_input_and_output_stays_unchanged():
    """Leave a fixed Expand native when neither operand rank nor result rank is known."""
    model = _make_model(
        [
            helper.make_node("Identity", ["x"], ["hidden_input"]),
            helper.make_node("Expand", ["hidden_input", "target"], ["hidden_output"]),
            helper.make_node("Identity", ["hidden_output"], ["y"]),
        ],
        "unknown_expand_rank",
        [_vi("x", TensorProto.FLOAT, [1, 3])],
        [_vi("y", TensorProto.FLOAT, [2, 3])],
        [_init("target", np.asarray([2, 3], np.int64))],
        infer_symbolic_shapes=False,
    )
    before = model.SerializeToString()
    feeds = {"x": np.arange(3, dtype=np.float32).reshape(1, 3)}
    expected = _run_ort(copy.deepcopy(model), feeds)
    converted = _apply_convert_ops_to_4d(copy.deepcopy(model))
    assert converted.SerializeToString() == before
    onnx.checker.check_model(converted, full_check=True)
    _assert_match(expected, _run_ort(converted, feeds))
