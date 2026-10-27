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
from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper
from onnx.reference import ReferenceEvaluator

from modelopt.onnx.graph_surgery.dla_transforms.dla_fix_instancenorm_channel_mismatch import (
    _apply_fix_instancenorm_channel_mismatch as _apply_fix_instancenorm,
)

pytest.importorskip("onnxruntime", reason="onnxruntime required for ORT parity checks")

_REPO_ROOT = Path(__file__).resolve().parents[5]
_SAVED_MODELS_DIR = _REPO_ROOT / "scratch_space" / "instancenorm_channel_mismatch_test_models"
_OPSET_VERSION = 17
_MAX_IR_VERSION_FOR_ORT = 9


# ── Shared utilities ──────────────────────────────────────────────────────────


def _vi(name: str, elem_type: int, shape: list) -> onnx.ValueInfoProto:
    """Declare a typed native tensor."""
    return helper.make_tensor_value_info(name, elem_type, shape)


def _clamp_ir(model: onnx.ModelProto) -> None:
    """Use a model IR supported by the CPU runtime."""
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
    model = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    onnx.checker.check_model(model)
    return model


def _run_ort(model: onnx.ModelProto, feeds: dict) -> dict:
    """Execute all public outputs on the CPU runtime."""
    import onnxruntime as ort  # type: ignore[import-not-found]

    _clamp_ir(model)
    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    return {o.name: sess.run([o.name], feeds)[0] for o in model.graph.output}


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compare finite tensor directions in double precision."""
    a64, b64 = a.astype(np.float64).ravel(), b.astype(np.float64).ravel()
    na, nb = float(np.linalg.norm(a64)), float(np.linalg.norm(b64))
    if na == 0.0 and nb == 0.0:
        return 1.0
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a64, b64) / (na * nb))


def _assert_values_match(ref: dict, got: dict) -> None:
    """Compare restored values after checking element counts."""
    assert ref.keys() == got.keys()
    for name in ref:
        a, b = ref[name].ravel(), got[name].ravel()
        assert a.shape == b.shape, f"{name}: element count {a.shape} vs {b.shape}"
        if a.dtype.kind == "f":
            cos = _cosine_similarity(a, b)
            assert cos >= 1.0 - 1e-5, f"{name}: cosine {cos:.8f}"
            np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-4, err_msg=name)


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
    model = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    onnx.checker.check_model(model)

    feeds = {"X": _RNG.standard_normal(shape).astype(np.float32)}
    ref = _run_ort(model, feeds)

    tr = copy.deepcopy(model)
    tr = _apply_fix_instancenorm(tr)
    _clamp_ir(tr)
    got = _run_ort(tr, feeds)

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
    model = onnx.shape_inference.infer_shapes(model, strict_mode=True)

    with pytest.raises(ValueError, match=r"rank.*5.*not supported|not supported"):
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
    model = onnx.shape_inference.infer_shapes(model, strict_mode=True)
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

    # Y3 shape must remain (2,4,6)
    assert got["Y3"].shape == (2, c, 6)
    # Y4 shape must remain (2,4,8,6)
    assert got["Y4"].shape == (2, c, 8, 6)
    _assert_values_match(ref, got)

    # Exactly 2 InstanceNorm nodes remain (1 original 4D + 1 new for 3D)
    in_nodes = [n for n in tr.graph.node if n.op_type == "InstanceNormalization"]
    assert len(in_nodes) == 2, f"Expected 2 InstanceNorm nodes, got {len(in_nodes)}"


@pytest.mark.parametrize("dtype", [TensorProto.FLOAT, TensorProto.FLOAT16])
@pytest.mark.parametrize("legacy", [False, True])
def test_instancenorm_preserves_dtype_epsilon_and_allocated_names(dtype, legacy):
    """Keep typed metadata, attributes, legacy declarations and existing name owners."""
    model = _make_instancenorm_model([2, 4, 6], epsilon=0.125)
    numpy_dtype = np.float16 if dtype == TensorProto.FLOAT16 else np.float32
    for tensor in model.graph.initializer:
        tensor.CopyFrom(
            numpy_helper.from_array(numpy_helper.to_array(tensor).astype(numpy_dtype), tensor.name)
        )
    for value in [*model.graph.input, *model.graph.output]:
        value.type.tensor_type.elem_type = dtype
    model.graph.initializer.append(
        numpy_helper.from_array(np.asarray([99], np.int64), "instnorm_pre_shape_0")
    )
    model.graph.node.insert(
        0,
        helper.make_node(
            "Constant",
            [],
            ["X_trail4d_0"],
            name="instnorm_pre_reshape_0",
            value=numpy_helper.from_array(np.asarray([42], np.float32)),
        ),
    )
    constant = model.graph.node[0]
    constant_bytes = constant.SerializeToString()
    if legacy:
        model.ir_version = 3
        model.graph.input.extend(
            _vi(t.name, t.data_type, list(t.dims)) for t in model.graph.initializer
        )
    feed = {"X": np.arange(48, dtype=numpy_dtype).reshape(2, 4, 6)}
    onnx.checker.check_model(model, full_check=True)
    expected = ReferenceEvaluator(model).run(None, feed)[0]
    changed = _apply_fix_instancenorm(model)
    assert next(node for node in changed.graph.node if node.op_type == "Constant") is constant
    assert constant.SerializeToString() == constant_bytes
    onnx.checker.check_model(changed, full_check=True)
    actual = ReferenceEvaluator(changed).run(None, feed)[0]
    assert actual.dtype == numpy_dtype and actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-3)


def test_instancenorm_late_unsupported_rank_leaves_caller_unchanged():
    """Reject a later five-dimensional operation before adapting an earlier valid one."""
    model = _make_instancenorm_model([2, 4, 6])
    model.graph.input.append(_vi("x5", TensorProto.FLOAT, [2, 4, 2, 3, 4]))
    model.graph.output.append(_vi("y5", TensorProto.FLOAT, [2, 4, 2, 3, 4]))
    model.graph.node.append(
        helper.make_node("InstanceNormalization", ["x5", "scale", "bias"], ["y5"])
    )
    before = model.SerializeToString()
    with pytest.raises(ValueError, match="rank"):
        _apply_fix_instancenorm(model)
    assert model.SerializeToString() == before


@pytest.mark.parametrize("version", [11, 12])
def test_instancenorm_legacy_opset_refuses_before_mutation(version):
    """Refuse unsupported axes-input adapters before allocating controls."""
    model = _make_instancenorm_model([2, 4, 6])
    model.opset_import[0].version = version
    before = model.SerializeToString()
    with pytest.raises(ValueError, match="opset"):
        _apply_fix_instancenorm(model)
    assert model.SerializeToString() == before


def test_opaque_instancenorm_function_is_unchanged():
    """Do not replace model-local operators by interpreting their standard-op names."""
    model = _make_instancenorm_model([2, 4, 6])
    model.graph.node[0].domain = "opaque"
    model.functions.append(
        helper.make_function(
            "opaque",
            "InstanceNormalization",
            ["x", "s", "b"],
            ["y"],
            [helper.make_node("Neg", ["x"], ["y"])],
            [helper.make_opsetid("", 17)],
            attributes=["epsilon"],
        )
    )
    model.opset_import.append(helper.make_opsetid("opaque", 1))
    before = model.SerializeToString()
    feeds = {"X": np.arange(48, dtype=np.float32).reshape(2, 4, 6)}
    expected = ReferenceEvaluator(model).run(None, feeds)[0]
    changed = _apply_fix_instancenorm(model)
    assert changed.SerializeToString() == before
    np.testing.assert_array_equal(ReferenceEvaluator(changed).run(None, feeds)[0], expected)


def test_instancenorm_zero_batch_stays_native():
    """Preserve zero dimensions instead of applying copied-zero Reshape semantics."""
    model = _make_instancenorm_model([0, 4, 6])
    before = model.SerializeToString()
    changed = _apply_fix_instancenorm(model)
    assert changed.SerializeToString() == before
    onnx.checker.check_model(changed, full_check=True)
    assert ReferenceEvaluator(changed).run(None, {"X": np.empty((0, 4, 6), np.float32)})[
        0
    ].shape == (0, 4, 6)
