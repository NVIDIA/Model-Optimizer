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
# Unit tests for :mod:`modelopt.onnx.graph_surgery.dla_transforms.dla_matmul_to_transpose_conv_transpose`.

"""ORT CPU parity tests for MatMul/Gemm → Unsqueeze-Transpose-Conv-Transpose-Squeeze.

Test matrix
-----------
Converts:
  2D activation  [M,N] x W_init[N,K]            → Y[M,K]
  3D activation  [B,M,N] x W_init[N,K]           → Y[B,M,K]
  4D activation  [1,B,M,N] x W_init[N,K]         → Y[1,B,M,K]  (no Unsqueeze/Squeeze)
  Gemm transB=1  A[M,N] x W_init[K,N]^T          → Y[M,K]
  Gemm transA=1  A[N,M]^T x W_init[N,K]          → Y[M,K]
  Gemm bias      A[M,N] x W_init[N,K] + B[K]     → Y[M,K]
  DQ weight      A[M,N] x dequant(W_q_int8[N,K]) → Y[M,K]  (init→DQ chain)

Does NOT convert:
  Dynamic weight  A[M,N] x W_graph_input[N,K]    → MatMul left unchanged
  NCHW non-unary  A[2,3,M,N] x W_init[2,3,N,K]  → MatMul left unchanged (4D weight, N/C > 1)

Each test saves a base + transformed ONNX pair under
``scratch_space/matmul_conv_test_models/`` for offline inspection.
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
_SAVED_MODELS_DIR = _REPO_ROOT / "scratch_space" / "matmul_conv_test_models"
_OPSET_VERSION = 17
_MAX_IR_VERSION_FOR_ORT = 9

# Small but non-trivial dims: activation [M,N], weight [N,K], batch B
M, N, K, B = 8, 16, 12, 3


# ── Dynamic import (avoids loading graph_surgery/__init__.py → onnxscript etc.) ──────────────────


def _load_apply_matmul_to_conv():
    mod_key = "modelopt.onnx.graph_surgery.dla_transforms.dla_matmul_to_transpose_conv_transpose"
    if mod_key in sys.modules:
        return sys.modules[mod_key]._apply_matmul_to_transpose_conv_transpose

    import modelopt.onnx  # noqa: F401 — loads package root + logging_config

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
        / "modelopt/onnx/graph_surgery/dla_transforms/dla_matmul_to_transpose_conv_transpose.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_key] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod._apply_matmul_to_transpose_conv_transpose


_apply_matmul_to_conv = _load_apply_matmul_to_conv()


# ── Shared test utilities ─────────────────────────────────────────────────────────────────────────


def _init(name: str, arr: np.ndarray) -> onnx.TensorProto:
    return numpy_helper.from_array(arr, name=name)


def _vi(name: str, elem_type: int, shape: list) -> onnx.ValueInfoProto:
    return helper.make_tensor_value_info(name, elem_type, shape)


def _clamp_ir_version(model: onnx.ModelProto) -> None:
    model.ir_version = min(model.ir_version, _MAX_IR_VERSION_FOR_ORT)


def _run_ort(model: onnx.ModelProto, feeds: dict) -> dict:
    import onnxruntime as ort  # type: ignore[import-not-found]

    _clamp_ir_version(model)
    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    return {o.name: sess.run([o.name], feeds)[0] for o in model.graph.output}


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a64 = np.asarray(a, dtype=np.float64).ravel()
    b64 = np.asarray(b, dtype=np.float64).ravel()
    na, nb = float(np.linalg.norm(a64)), float(np.linalg.norm(b64))
    if na == 0.0 and nb == 0.0:
        return 1.0
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a64, b64) / (na * nb))


def _assert_outputs_match(ref: dict, got: dict) -> None:
    assert ref.keys() == got.keys()
    for name in ref:
        a, b = ref[name], got[name]
        assert a.shape == b.shape, f"{name}: shape {a.shape} vs {b.shape}"
        cos = _cosine_similarity(a, b)
        assert cos >= 1.0 - 1e-5, f"{name}: cosine similarity {cos:.8f} (expected ~1.0)"
        np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-4, err_msg=f"{name}: value mismatch")


def _save_test_pair(base: onnx.ModelProto, transformed: onnx.ModelProto, tag: str) -> None:
    _SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    onnx.save(base, str(_SAVED_MODELS_DIR / f"matmul_base_{tag}.onnx"))
    import contextlib

    with contextlib.suppress(Exception):
        transformed = onnx.shape_inference.infer_shapes(copy.deepcopy(transformed))
    onnx.save(transformed, str(_SAVED_MODELS_DIR / f"matmul_transformed_{tag}.onnx"))


def _make_model(nodes, graph_name, inputs, outputs, initializers=None):
    """Build, shape-infer, and checker-validate a simple ONNX model."""
    graph = helper.make_graph(nodes, graph_name, inputs, outputs, initializer=initializers or [])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", _OPSET_VERSION)])
    _clamp_ir_version(model)
    try:
        from onnxruntime.tools.symbolic_shape_infer import (
            SymbolicShapeInference,  # type: ignore[import-untyped]
        )

        model = SymbolicShapeInference.infer_shapes(model, auto_merge_symbolic_dims=True) or model
    except Exception:
        pass
    onnx.checker.check_model(model)
    return model


_RNG = np.random.default_rng(42)


# ── Model builders ────────────────────────────────────────────────────────────────────────────────


def _make_matmul_2d():
    """MatMul: A[M,N] x W[N,K] → Y[M,K].  W is a float32 initializer."""
    w = _RNG.standard_normal((N, K)).astype(np.float32)
    model = _make_model(
        [helper.make_node("MatMul", ["A", "W"], ["Y"], name="mm")],
        "mm_2d",
        [_vi("A", TensorProto.FLOAT, [M, N])],
        [_vi("Y", TensorProto.FLOAT, [M, K])],
        [_init("W", w)],
    )
    feeds = {"A": _RNG.standard_normal((M, N)).astype(np.float32)}
    return model, feeds


def _make_matmul_3d():
    """MatMul: A[B,M,N] x W[N,K] → Y[B,M,K].  W is a float32 initializer."""
    w = _RNG.standard_normal((N, K)).astype(np.float32)
    model = _make_model(
        [helper.make_node("MatMul", ["A", "W"], ["Y"], name="mm")],
        "mm_3d",
        [_vi("A", TensorProto.FLOAT, [B, M, N])],
        [_vi("Y", TensorProto.FLOAT, [B, M, K])],
        [_init("W", w)],
    )
    feeds = {"A": _RNG.standard_normal((B, M, N)).astype(np.float32)}
    return model, feeds


def _make_matmul_4d():
    """MatMul: A[1,B,M,N] x W[N,K] → Y[1,B,M,K].  W is a float32 initializer."""
    w = _RNG.standard_normal((N, K)).astype(np.float32)
    model = _make_model(
        [helper.make_node("MatMul", ["A", "W"], ["Y"], name="mm")],
        "mm_4d",
        [_vi("A", TensorProto.FLOAT, [1, B, M, N])],
        [_vi("Y", TensorProto.FLOAT, [1, B, M, K])],
        [_init("W", w)],
    )
    feeds = {"A": _RNG.standard_normal((1, B, M, N)).astype(np.float32)}
    return model, feeds


def _make_gemm_transb():
    """Gemm transB=1: A[M,N] x W[K,N]^T → Y[M,K].  W is a float32 initializer."""
    w = _RNG.standard_normal((K, N)).astype(np.float32)
    model = _make_model(
        [helper.make_node("Gemm", ["A", "W"], ["Y"], name="gemm", transB=1)],
        "gemm_transb",
        [_vi("A", TensorProto.FLOAT, [M, N])],
        [_vi("Y", TensorProto.FLOAT, [M, K])],
        [_init("W", w)],
    )
    feeds = {"A": _RNG.standard_normal((M, N)).astype(np.float32)}
    return model, feeds


def _make_gemm_transa():
    """Gemm transA=1: A[N,M]^T x W[N,K] → Y[M,K].  W is a float32 initializer."""
    w = _RNG.standard_normal((N, K)).astype(np.float32)
    model = _make_model(
        [helper.make_node("Gemm", ["A", "W"], ["Y"], name="gemm", transA=1)],
        "gemm_transa",
        [_vi("A", TensorProto.FLOAT, [N, M])],  # A is [N,M]; transA makes it [M,N]
        [_vi("Y", TensorProto.FLOAT, [M, K])],
        [_init("W", w)],
    )
    feeds = {"A": _RNG.standard_normal((N, M)).astype(np.float32)}
    return model, feeds


def _make_gemm_bias():
    """Gemm: A[M,N] x W[N,K] + Bias[K] → Y[M,K].  W and Bias are float32 initializers."""
    w = _RNG.standard_normal((N, K)).astype(np.float32)
    bias = _RNG.standard_normal((K,)).astype(np.float32)
    model = _make_model(
        [helper.make_node("Gemm", ["A", "W", "Bias"], ["Y"], name="gemm")],
        "gemm_bias",
        [_vi("A", TensorProto.FLOAT, [M, N])],
        [_vi("Y", TensorProto.FLOAT, [M, K])],
        [_init("W", w), _init("Bias", bias)],
    )
    feeds = {"A": _RNG.standard_normal((M, N)).astype(np.float32)}
    return model, feeds


def _make_matmul_dq_weight():
    """MatMul: A[M,N] x dequant(W_q[N,K]) → Y[M,K].  Per-tensor int8 weight via DQ chain."""
    w_q = _RNG.integers(-127, 127, (N, K)).astype(np.int8)
    scale = np.float32(0.05)
    zp = np.int8(0)
    nodes = [
        helper.make_node("DequantizeLinear", ["W_q", "W_scale", "W_zp"], ["W_dq"], name="dq_W"),
        helper.make_node("MatMul", ["A", "W_dq"], ["Y"], name="mm"),
    ]
    model = _make_model(
        nodes,
        "mm_dq",
        [_vi("A", TensorProto.FLOAT, [M, N])],
        [_vi("Y", TensorProto.FLOAT, [M, K])],
        [_init("W_q", w_q), _init("W_scale", np.array(scale)), _init("W_zp", np.array(zp))],
    )
    feeds = {"A": _RNG.standard_normal((M, N)).astype(np.float32)}
    return model, feeds


def _make_matmul_dynamic_weight():
    """MatMul: A[M,N] x W[N,K] → Y[M,K].  W is a graph input — should NOT be converted."""
    model = _make_model(
        [helper.make_node("MatMul", ["A", "W"], ["Y"], name="mm")],
        "mm_dyn",
        [_vi("A", TensorProto.FLOAT, [M, N]), _vi("W", TensorProto.FLOAT, [N, K])],
        [_vi("Y", TensorProto.FLOAT, [M, K])],
    )
    feeds = {
        "A": _RNG.standard_normal((M, N)).astype(np.float32),
        "W": _RNG.standard_normal((N, K)).astype(np.float32),
    }
    return model, feeds


def _make_matmul_nchw_non_unary():
    """Batch MatMul: A[2,3,M,N] x W[2,3,N,K] → Y[2,3,M,K].

    W is a 4D initializer with shape[0]=2 (N > 1) — check_to_apply_transpose returns False,
    so the MatMul must remain unchanged.
    """
    c1, c2 = 2, 3
    w = _RNG.standard_normal((c1, c2, N, K)).astype(np.float32)
    model = _make_model(
        [helper.make_node("MatMul", ["A", "W"], ["Y"], name="mm")],
        "mm_nchw",
        [_vi("A", TensorProto.FLOAT, [c1, c2, M, N])],
        [_vi("Y", TensorProto.FLOAT, [c1, c2, M, K])],
        [_init("W", w)],
    )
    feeds = {"A": _RNG.standard_normal((c1, c2, M, N)).astype(np.float32)}
    return model, feeds


# ── Helper: apply transform and return (ref_outputs, got_outputs, transformed_model) ─────────────


def _apply_and_run(model, feeds):
    ref = _run_ort(model, feeds)
    transformed = copy.deepcopy(model)
    transformed = _apply_matmul_to_conv(transformed)
    _clamp_ir_version(transformed)
    got = _run_ort(transformed, feeds)
    return ref, got, transformed


# ── Tests: cases that SHOULD convert ─────────────────────────────────────────────────────────────


def test_matmul_2d_activation_converts():
    """2D activation → Unsqueeze[0,1] + Conv + Squeeze[0,1]; output must match MatMul."""
    model, feeds = _make_matmul_2d()
    ref, got, tr = _apply_and_run(model, feeds)

    assert not any(n.op_type in ("MatMul", "Gemm") for n in tr.graph.node), (
        "MatMul should be replaced"
    )
    assert sum(1 for n in tr.graph.node if n.op_type == "Conv") == 1
    assert any(n.op_type == "Unsqueeze" for n in tr.graph.node), "Unsqueeze expected for 2D input"
    assert any(n.op_type == "Squeeze" for n in tr.graph.node), "Squeeze expected for 2D input"

    _save_test_pair(model, tr, "2d")
    _assert_outputs_match(ref, got)


def test_matmul_3d_activation_converts():
    """3D activation [B,M,N] → Unsqueeze[0] + Conv + Squeeze[0]; output must match MatMul."""
    model, feeds = _make_matmul_3d()
    ref, got, tr = _apply_and_run(model, feeds)

    assert not any(n.op_type in ("MatMul", "Gemm") for n in tr.graph.node)
    assert sum(1 for n in tr.graph.node if n.op_type == "Conv") == 1
    assert any(n.op_type == "Unsqueeze" for n in tr.graph.node), "Unsqueeze expected for 3D input"
    assert any(n.op_type == "Squeeze" for n in tr.graph.node), "Squeeze expected for 3D input"

    _save_test_pair(model, tr, "3d")
    _assert_outputs_match(ref, got)


def test_matmul_4d_activation_converts():
    """4D activation [1,B,M,N] → Conv only, NO Unsqueeze/Squeeze; output must match MatMul."""
    model, feeds = _make_matmul_4d()
    ref, got, tr = _apply_and_run(model, feeds)

    assert not any(n.op_type in ("MatMul", "Gemm") for n in tr.graph.node)
    assert sum(1 for n in tr.graph.node if n.op_type == "Conv") == 1
    assert not any(n.op_type == "Unsqueeze" for n in tr.graph.node), (
        "No Unsqueeze expected for 4D input"
    )
    assert not any(n.op_type == "Squeeze" for n in tr.graph.node), (
        "No Squeeze expected for 4D input"
    )

    _save_test_pair(model, tr, "4d")
    _assert_outputs_match(ref, got)


def test_gemm_transb_converts():
    """Gemm transB=1 (W[K,N] → used without extra transpose): output must match Gemm."""
    model, feeds = _make_gemm_transb()
    ref, got, tr = _apply_and_run(model, feeds)

    assert not any(n.op_type in ("MatMul", "Gemm") for n in tr.graph.node)

    _save_test_pair(model, tr, "gemm_transb")
    _assert_outputs_match(ref, got)


def test_gemm_transa_converts():
    """Gemm transA=1 (A[N,M] used transposed): output must match Gemm."""
    model, feeds = _make_gemm_transa()
    ref, got, tr = _apply_and_run(model, feeds)

    assert not any(n.op_type in ("MatMul", "Gemm") for n in tr.graph.node)

    _save_test_pair(model, tr, "gemm_transa")
    _assert_outputs_match(ref, got)


def test_gemm_with_bias_converts():
    """Gemm with 1D bias: bias forwarded to Conv as third input; output must match Gemm."""
    model, feeds = _make_gemm_bias()
    ref, got, tr = _apply_and_run(model, feeds)

    assert not any(n.op_type in ("MatMul", "Gemm") for n in tr.graph.node)
    conv_nodes = [n for n in tr.graph.node if n.op_type == "Conv"]
    assert len(conv_nodes) == 1
    assert len(conv_nodes[0].input) == 3, "Conv should carry bias as third input"

    _save_test_pair(model, tr, "gemm_bias")
    _assert_outputs_match(ref, got)


def test_matmul_dq_weight_converts():
    """Int8 init→DQ weight: W_q reshaped to [K,N,1,1], DQ node preserved, Conv uses DQ output."""
    model, feeds = _make_matmul_dq_weight()
    ref, got, tr = _apply_and_run(model, feeds)

    assert not any(n.op_type in ("MatMul", "Gemm") for n in tr.graph.node)
    assert any(n.op_type == "DequantizeLinear" for n in tr.graph.node), "DQ node should remain"
    assert sum(1 for n in tr.graph.node if n.op_type == "Conv") == 1

    # Verify W_q was reshaped to 4D [K,N,1,1]
    w_q_init = next((i for i in tr.graph.initializer if i.name == "W_q"), None)
    assert w_q_init is not None
    assert list(w_q_init.dims) == [K, N, 1, 1], (
        f"W_q dims: {list(w_q_init.dims)}, expected [{K},{N},1,1]"
    )

    _save_test_pair(model, tr, "dq_weight")
    _assert_outputs_match(ref, got)


# ── Tests: cases that should NOT convert ─────────────────────────────────────────────────────────


def test_matmul_dynamic_weight_not_converted():
    """Dynamic weight: MatMul remains; both inputs and output promoted to 4D; ORT parity."""
    model, feeds = _make_matmul_dynamic_weight()
    ref, got, tr = _apply_and_run(model, feeds)

    matmul_nodes = [n for n in tr.graph.node if n.op_type == "MatMul"]
    assert len(matmul_nodes) == 1, "MatMul must remain when weight is a graph input"
    assert not any(n.op_type == "Conv" for n in tr.graph.node), "No Conv should be added"
    # Fallback: both 2D inputs promoted → Unsqueeze nodes inserted
    assert any(n.op_type == "Unsqueeze" for n in tr.graph.node), (
        "Unsqueeze expected for 4D promotion of dynamic inputs"
    )
    # Fallback: 2D output squeezed back
    assert any(n.op_type == "Squeeze" for n in tr.graph.node), (
        "Squeeze expected to restore original output rank"
    )

    _save_test_pair(model, tr, "dynamic_weight")
    _assert_outputs_match(ref, got)


def test_matmul_nchw_non_unary_not_converted():
    """Batch MatMul [2,3,M,N] x W[2,3,N,K]: ineligible for Conv; inputs/output promoted to 4D."""
    model, feeds = _make_matmul_nchw_non_unary()
    ref, got, tr = _apply_and_run(model, feeds)

    matmul_nodes = [n for n in tr.graph.node if n.op_type == "MatMul"]
    assert len(matmul_nodes) == 1, "MatMul must remain when weight 4D shape[0] > 1"
    assert not any(n.op_type == "Conv" for n in tr.graph.node), "No Conv should be added"
    # Activation [2,3,M,N] is already 4D — no Unsqueeze needed for it
    # Weight W[2,3,N,K] is already 4D — no init expansion needed
    # Output [2,3,M,K] is already 4D — no Squeeze needed
    assert not any(n.op_type in ("Unsqueeze", "Squeeze") for n in tr.graph.node), (
        "No Unsqueeze/Squeeze expected when all tensors are already 4D"
    )

    _save_test_pair(model, tr, "nchw_non_unary")
    _assert_outputs_match(ref, got)
