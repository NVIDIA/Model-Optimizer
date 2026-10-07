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
# Unit tests for :mod:`modelopt.onnx.graph_surgery.dla_transforms.dla_fix_instancenorm_channel_mismatch`.

"""ORT CPU parity tests for ``dla_fix_instancenorm_channel_mismatch``.

Transform rule
--------------
For 3D input ``[N, c,D]``:

  ``[N,C,D] → Reshape[N,C,D,1] → InstanceNorm → Reshape[1,N,C,D] → Squeeze(axis=0) → [N,C,D]``

Numerical correctness holds because InstanceNorm normalises each (N,C) slice over
its spatial dims; ``[N,C,D,1]`` has the same spatial product as ``[N,C,D]``.

For 4D input the transform is a no-op (node unchanged, no extra nodes inserted).
For >4D input a ``ValueError`` is raised.

Coverage
--------
3D input [2,4,6]     → output shape (2,4,6)   with ORT parity
4D input [2,4,8,6]   → no-op (0 nodes inserted, output unchanged)
5D input [2,4,8,6,3] → raises ValueError
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
_SAVED_MODELS_DIR = _REPO_ROOT / "scratch_space" / "instancenorm_channel_mismatch_test_models"
_OPSET_VERSION = 17
_MAX_IR_VERSION_FOR_ORT = 9


# ── Dynamic import ────────────────────────────────────────────────────────────


def _load_apply_fix_instancenorm():
    mod_key = "modelopt.onnx.graph_surgery.dla_transforms.dla_fix_instancenorm_channel_mismatch"
    if mod_key in sys.modules:
        return sys.modules[mod_key]._apply_fix_instancenorm_channel_mismatch

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
        _REPO_ROOT
        / "modelopt/onnx/graph_surgery/dla_transforms/dla_fix_instancenorm_channel_mismatch.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_key] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod._apply_fix_instancenorm_channel_mismatch


_apply_fix_instancenorm = _load_apply_fix_instancenorm()


# ── Shared utilities ──────────────────────────────────────────────────────────


def _vi(name: str, elem_type: int, shape: list) -> onnx.ValueInfoProto:
    return helper.make_tensor_value_info(name, elem_type, shape)


def _clamp_ir(model: onnx.ModelProto) -> None:
    model.ir_version = min(model.ir_version, _MAX_IR_VERSION_FOR_ORT)


def _make_instancenorm_model(input_shape: list, epsilon: float = 1e-5) -> onnx.ModelProto:
    """Build a minimal model with a single InstanceNormalization node."""
    rank = len(input_shape)
    assert rank >= 3, "InstanceNorm requires at least 3D input"
    c = input_shape[1]

    scale_data = np.ones(c, dtype=np.float32)
    bias_data = np.zeros(c, dtype=np.float32)
    scale_init = numpy_helper.from_array(scale_data, name="scale")
    bias_init = numpy_helper.from_array(bias_data, name="bias")

    node = helper.make_node(
        "InstanceNormalization",
        inputs=["X", "scale", "bias"],
        outputs=["Y"],
        name="instnorm",
        epsilon=epsilon,
    )
    graph = helper.make_graph(
        [node],
        "instnorm_graph",
        [_vi("X", TensorProto.FLOAT, input_shape)],
        [_vi("Y", TensorProto.FLOAT, input_shape)],
        initializer=[scale_init, bias_init],
    )
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


def _assert_values_match(ref: dict, got: dict) -> None:
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


_RNG = np.random.default_rng(2024)


# ── 3D input [2,4,6] → output shape (2,4,6) ──────────────────────────────────


def test_instancenorm_3d_parity():
    """3D input [2,4,6]: transform produces same numerics and restores original shape."""
    shape = [2, 4, 6]
    model = _make_instancenorm_model(shape)
    feeds = {"X": _RNG.standard_normal(shape).astype(np.float32)}

    ref = _run_ort(model, feeds)

    tr = copy.deepcopy(model)
    tr = _apply_fix_instancenorm(tr)
    _clamp_ir(tr)
    got = _run_ort(tr, feeds)
    _save(model, tr, "instancenorm_3d")

    # Output shape must be restored to original 3D
    assert got["Y"].shape == tuple(shape), f"Expected {tuple(shape)}, got {got['Y'].shape}"

    # Values must match
    _assert_values_match(ref, got)

    # The original InstanceNorm node must be gone
    assert not any(
        n.op_type == "InstanceNormalization" and n.name == "instnorm" for n in tr.graph.node
    ), "Original InstanceNorm must be replaced"

    # Exactly one new InstanceNorm, two Reshape nodes, and one Squeeze
    in_nodes = [n for n in tr.graph.node if n.op_type == "InstanceNormalization"]
    reshape_nodes = [n for n in tr.graph.node if n.op_type == "Reshape"]
    squeeze_nodes = [n for n in tr.graph.node if n.op_type == "Squeeze"]
    assert len(in_nodes) == 1, f"Expected 1 InstanceNorm, got {len(in_nodes)}"
    assert len(reshape_nodes) == 2, f"Expected 2 Reshape nodes, got {len(reshape_nodes)}"
    assert len(squeeze_nodes) == 1, f"Expected 1 Squeeze node, got {len(squeeze_nodes)}"


# ── 3D input with non-trivial scale/bias ─────────────────────────────────────


def test_instancenorm_3d_nontrivial_scale_bias():
    """3D input [3,8,16] with random scale/bias: parity and shape preserved."""
    shape = [3, 8, 16]
    c = shape[1]
    scale_data = _RNG.standard_normal(c).astype(np.float32) * 2.0 + 1.0
    bias_data = _RNG.standard_normal(c).astype(np.float32) * 0.5

    scale_init = numpy_helper.from_array(scale_data, name="scale")
    bias_init = numpy_helper.from_array(bias_data, name="bias")
    node = helper.make_node(
        "InstanceNormalization",
        inputs=["X", "scale", "bias"],
        outputs=["Y"],
        name="instnorm",
        epsilon=1e-3,
    )
    graph = helper.make_graph(
        [node],
        "instnorm_nontrivial",
        [_vi("X", TensorProto.FLOAT, shape)],
        [_vi("Y", TensorProto.FLOAT, shape)],
        initializer=[scale_init, bias_init],
    )
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

    feeds = {"X": _RNG.standard_normal(shape).astype(np.float32)}
    ref = _run_ort(model, feeds)

    tr = copy.deepcopy(model)
    tr = _apply_fix_instancenorm(tr)
    _clamp_ir(tr)
    got = _run_ort(tr, feeds)
    _save(model, tr, "instancenorm_3d_nontrivial")

    assert got["Y"].shape == tuple(shape)
    _assert_values_match(ref, got)


# ── 4D input: no-op ───────────────────────────────────────────────────────────


def test_instancenorm_4d_noop():
    """4D input [2,4,8,6]: transform must not modify the graph; ORT outputs unchanged."""
    shape = [2, 4, 8, 6]
    model = _make_instancenorm_model(shape)
    original_node_count = len(model.graph.node)
    feeds = {"X": _RNG.standard_normal(shape).astype(np.float32)}

    ref = _run_ort(model, feeds)

    tr = copy.deepcopy(model)
    tr = _apply_fix_instancenorm(tr)
    _clamp_ir(tr)
    got = _run_ort(tr, feeds)
    _save(model, tr, "instancenorm_4d_noop")

    assert len(tr.graph.node) == original_node_count, (
        f"4D model should be unchanged: {original_node_count} → {len(tr.graph.node)} nodes"
    )
    assert any(n.op_type == "InstanceNormalization" for n in tr.graph.node)
    assert got["Y"].shape == tuple(shape)
    _assert_values_match(ref, got)


# ── >4D input: raises ValueError ─────────────────────────────────────────────


def test_instancenorm_5d_raises():
    """5D input must raise ValueError."""
    shape = [2, 4, 8, 6, 3]
    c = shape[1]
    scale_init = numpy_helper.from_array(np.ones(c, dtype=np.float32), name="scale")
    bias_init = numpy_helper.from_array(np.zeros(c, dtype=np.float32), name="bias")
    node = helper.make_node(
        "InstanceNormalization",
        inputs=["X", "scale", "bias"],
        outputs=["Y"],
        name="instnorm",
    )
    graph = helper.make_graph(
        [node],
        "instnorm_5d",
        [_vi("X", TensorProto.FLOAT, shape)],
        [_vi("Y", TensorProto.FLOAT, shape)],
        initializer=[scale_init, bias_init],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", _OPSET_VERSION)])
    _clamp_ir(model)
    try:
        from onnxruntime.tools.symbolic_shape_infer import (
            SymbolicShapeInference,  # type: ignore[import-untyped]
        )

        model = SymbolicShapeInference.infer_shapes(model, auto_merge_symbolic_dims=True) or model
    except Exception:
        pass

    with pytest.raises(ValueError, match="rank.*5.*not supported|not supported"):
        _apply_fix_instancenorm(copy.deepcopy(model))


# ── Multiple InstanceNorm nodes in one graph ──────────────────────────────────


def test_instancenorm_multiple_nodes():
    """Graph with two InstanceNorm nodes (one 3D, one 4D): only 3D one is transformed."""
    c = 4

    scale3 = numpy_helper.from_array(np.ones(c, dtype=np.float32), name="scale3")
    bias3 = numpy_helper.from_array(np.zeros(c, dtype=np.float32), name="bias3")
    scale4 = numpy_helper.from_array(np.ones(c, dtype=np.float32), name="scale4")
    bias4 = numpy_helper.from_array(np.zeros(c, dtype=np.float32), name="bias4")

    node3d = helper.make_node(
        "InstanceNormalization",
        inputs=["X3", "scale3", "bias3"],
        outputs=["Y3"],
        name="instnorm_3d",
    )
    node4d = helper.make_node(
        "InstanceNormalization",
        inputs=["X4", "scale4", "bias4"],
        outputs=["Y4"],
        name="instnorm_4d",
    )
    graph = helper.make_graph(
        [node3d, node4d],
        "multi_instnorm",
        [
            _vi("X3", TensorProto.FLOAT, [2, c, 6]),
            _vi("X4", TensorProto.FLOAT, [2, c, 8, 6]),
        ],
        [
            _vi("Y3", TensorProto.FLOAT, [2, c, 6]),
            _vi("Y4", TensorProto.FLOAT, [2, c, 8, 6]),
        ],
        initializer=[scale3, bias3, scale4, bias4],
    )
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

    feeds = {
        "X3": _RNG.standard_normal([2, c, 6]).astype(np.float32),
        "X4": _RNG.standard_normal([2, c, 8, 6]).astype(np.float32),
    }
    ref = _run_ort(model, feeds)

    tr = copy.deepcopy(model)
    tr = _apply_fix_instancenorm(tr)
    _clamp_ir(tr)
    got = _run_ort(tr, feeds)
    _save(model, tr, "instancenorm_multi")

    # Y3 shape must remain (2,4,6)
    assert got["Y3"].shape == (2, c, 6)
    # Y4 shape must remain (2,4,8,6)
    assert got["Y4"].shape == (2, c, 8, 6)
    _assert_values_match(ref, got)

    # Exactly 2 InstanceNorm nodes remain (1 original 4D + 1 new for 3D)
    in_nodes = [n for n in tr.graph.node if n.op_type == "InstanceNormalization"]
    assert len(in_nodes) == 2, f"Expected 2 InstanceNorm nodes, got {len(in_nodes)}"
