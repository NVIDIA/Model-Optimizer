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

"""

from __future__ import annotations

import copy

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper
from onnx.reference import ReferenceEvaluator

from modelopt.onnx.graph_surgery.dla_transforms.dla_matmul_to_transpose_conv_transpose import (
    _apply_matmul_to_transpose_conv_transpose as _apply_matmul_to_conv,
)
from modelopt.onnx.graph_surgery.make_dla_compatible import _transform_make_dla_compatible
from modelopt.onnx.graph_surgery.utils.graph_utils import GraphCache

ort = pytest.importorskip("onnxruntime", reason="onnxruntime required for ORT parity checks")

_OPSET_VERSION = 17
_MAX_IR_VERSION_FOR_ORT = 9

# Small but non-trivial dims: activation [M,N], weight [N,K], batch B
M, N, K, B = 8, 16, 12, 3


# ── Shared test utilities ─────────────────────────────────────────────────────────────────────────


def _init(name: str, arr: np.ndarray) -> onnx.TensorProto:
    """Create a named tensor initializer."""
    return numpy_helper.from_array(arr, name=name)


def _vi(name: str, elem_type: int, shape: list) -> onnx.ValueInfoProto:
    """Create typed shape metadata."""
    return helper.make_tensor_value_info(name, elem_type, shape)


def _clamp_ir_version(model: onnx.ModelProto) -> None:
    """Use an IR version supported by the test runtime."""
    model.ir_version = min(model.ir_version, _MAX_IR_VERSION_FOR_ORT)


def _run_ort(model: onnx.ModelProto, feeds: dict) -> dict:
    """Execute all graph outputs with the optional CPU runtime."""
    _clamp_ir_version(model)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(
        model.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )
    return dict(zip([output.name for output in model.graph.output], sess.run(None, feeds)))


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Measure the directional agreement of numerical outputs."""
    a64 = np.asarray(a, dtype=np.float64).ravel()
    b64 = np.asarray(b, dtype=np.float64).ravel()
    na, nb = float(np.linalg.norm(a64)), float(np.linalg.norm(b64))
    if na == 0.0 and nb == 0.0:
        return 1.0
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a64, b64) / (na * nb))


def _assert_outputs_match(ref: dict, got: dict) -> None:
    """Check output names, shapes and numerical agreement."""
    assert ref.keys() == got.keys()
    for name in ref:
        a, b = ref[name], got[name]
        assert a.shape == b.shape, f"{name}: shape {a.shape} vs {b.shape}"
        cos = _cosine_similarity(a, b)
        assert cos >= 1.0 - 1e-5, f"{name}: cosine similarity {cos:.8f} (expected ~1.0)"
        np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-4, err_msg=f"{name}: value mismatch")


def _make_model(nodes, graph_name, inputs, outputs, initializers=None):
    """Build, shape-infer, and checker-validate a simple ONNX model."""
    graph = helper.make_graph(nodes, graph_name, inputs, outputs, initializer=initializers or [])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", _OPSET_VERSION)])
    _clamp_ir_version(model)
    try:
        # Optional runtime tools may be absent without preventing test collection.
        from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference

        model = SymbolicShapeInference.infer_shapes(model, auto_merge=True) or model
    except ImportError:
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
    """Execute the original and converted graph for comparison."""
    ref = _run_ort(model, feeds)
    transformed = copy.deepcopy(model)
    transformed = _apply_matmul_to_conv(transformed)
    _clamp_ir_version(transformed)
    got = _run_ort(transformed, feeds)
    return ref, got, transformed


# ── Tests: cases that SHOULD convert ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("builder", "adapters"),
    [(_make_matmul_2d, True), (_make_matmul_3d, True), (_make_matmul_4d, False)],
)
def test_matmul_activation_converts(builder, adapters):
    """Convert each supported activation rank with only the required boundary adapters."""
    model, feeds = builder()
    expected, actual, transformed = _apply_and_run(model, feeds)
    assert not any(node.op_type in {"MatMul", "Gemm"} for node in transformed.graph.node)
    assert sum(node.op_type == "Conv" for node in transformed.graph.node) == 1
    for operation in ["Unsqueeze", "Squeeze"]:
        assert any(node.op_type == operation for node in transformed.graph.node) == adapters
    _assert_outputs_match(expected, actual)


@pytest.mark.parametrize("builder", [_make_gemm_transb, _make_gemm_transa])
def test_gemm_transpose_converts(builder):
    """Preserve values with either supported Gemm transpose convention."""
    model, feeds = builder()
    expected, actual, transformed = _apply_and_run(model, feeds)
    assert not any(node.op_type in {"MatMul", "Gemm"} for node in transformed.graph.node)
    _assert_outputs_match(expected, actual)


@pytest.mark.parametrize("trans_b", [False, True])
@pytest.mark.parametrize(
    "form", ["alpha", "beta", "row", "matrix", "scalar", "singleton", "symbolic", "dynamic"]
)
def test_ineligible_gemm_preserves_native_broadcast_and_attributes(form, trans_b):
    """Leave valid unsupported Gemm forms intact, including broadcast biases and dynamic weights."""
    bias_shape = {"row": [1, 4], "matrix": [2, 4], "scalar": [], "singleton": [1]}.get(form, [4])
    bias = np.ones(bias_shape, np.float32)
    weight = np.arange(12, dtype=np.float32).reshape(3, 4)
    if trans_b:
        weight = weight.T.copy()
    attrs = {"transB": int(trans_b)}
    if form in {"alpha", "dynamic"}:
        attrs["alpha"] = 2.0
    if form == "beta":
        attrs["beta"] = 0.5
    inputs = [_vi("A", TensorProto.FLOAT, [2, 3])]
    feeds = {"A": np.arange(6, dtype=np.float32).reshape(2, 3)}
    inits = []
    if form == "dynamic":
        inputs.append(_vi("W", TensorProto.FLOAT, list(weight.shape)))
        feeds["W"] = weight
    else:
        inits.append(_init("W", weight))
    if form == "symbolic":
        inputs.append(_vi("Bias", TensorProto.FLOAT, ["channels"]))
        feeds["Bias"] = bias
    else:
        inits.append(_init("Bias", bias))
    model = helper.make_model(
        helper.make_graph(
            [helper.make_node("Gemm", ["A", "W", "Bias"], ["Y"], **attrs)],
            "native_gemm",
            inputs,
            [_vi("Y", TensorProto.FLOAT, [2, 4])],
            inits,
        ),
        opset_imports=[helper.make_opsetid("", _OPSET_VERSION)],
    )
    _clamp_ir_version(model)
    onnx.checker.check_model(model, full_check=True)
    expected = _run_ort(model, feeds)
    before = model.SerializeToString()
    transformed = _apply_matmul_to_conv(copy.deepcopy(model))
    assert transformed.SerializeToString() == before
    _assert_outputs_match(expected, _run_ort(transformed, feeds))


@pytest.mark.parametrize("operation", ["MatMul", "Gemm"])
@pytest.mark.parametrize("defer", [False, True])
def test_initializer_activation_without_value_info_preserves_values(operation, defer):
    """Fixed activations need not have graph-input or intermediate metadata to remain valid."""
    model = helper.make_model(
        helper.make_graph(
            [helper.make_node(operation, ["A", "W"], ["Y"])],
            "fixed_activation",
            [],
            [_vi("Y", TensorProto.FLOAT, [2, 4])],
            [
                _init("A", np.arange(6, dtype=np.float32).reshape(2, 3)),
                _init("W", np.arange(12, dtype=np.float32).reshape(3, 4)),
            ],
        ),
        opset_imports=[helper.make_opsetid("", _OPSET_VERSION)],
    )
    _clamp_ir_version(model)
    onnx.checker.check_model(model, full_check=True)
    expected = _run_ort(model, {})
    converted = _apply_matmul_to_conv(copy.deepcopy(model), defer_rank_conversion=defer)
    onnx.checker.check_model(converted, full_check=True)
    _assert_outputs_match(expected, _run_ort(converted, {}))


def test_gemm_with_bias_converts():
    """Gemm with 1D bias: bias forwarded to Conv as third input; output must match Gemm."""
    model, feeds = _make_gemm_bias()
    ref, got, tr = _apply_and_run(model, feeds)

    assert not any(n.op_type in ("MatMul", "Gemm") for n in tr.graph.node)
    conv_nodes = [n for n in tr.graph.node if n.op_type == "Conv"]
    assert len(conv_nodes) == 1
    assert len(conv_nodes[0].input) == 3, "Conv should carry bias as third input"

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

    _assert_outputs_match(ref, got)


# ── Tests: cases that should NOT convert ─────────────────────────────────────────────────────────


@pytest.mark.parametrize("defer", [False, True])
def test_matmul_dynamic_weight_not_converted(defer):
    """Keep dynamic-weight MatMul valid with immediate or deferred rank conversion."""
    model, feeds = _make_matmul_dynamic_weight()
    expected = _run_ort(model, feeds)
    transformed = _apply_matmul_to_conv(copy.deepcopy(model), defer_rank_conversion=defer)
    onnx.checker.check_model(transformed, full_check=True)
    assert sum(node.op_type == "MatMul" for node in transformed.graph.node) == 1
    assert not any(node.op_type == "Conv" for node in transformed.graph.node)
    assert any(node.op_type == "Unsqueeze" for node in transformed.graph.node)
    assert any(node.op_type == "Squeeze" for node in transformed.graph.node)
    _assert_outputs_match(expected, _run_ort(transformed, feeds))


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

    _assert_outputs_match(ref, got)


@pytest.mark.parametrize("trans_a", [False, True])
@pytest.mark.parametrize("trans_b", [False, True])
@pytest.mark.parametrize("defer", [False, True])
def test_dynamic_gemm_preserves_two_dimensional_contract(trans_a, trans_b, defer):
    """Leave dynamic Gemm operands and transpose semantics valid in both modes."""
    a_shape = [N, M] if trans_a else [M, N]
    b_shape = [K, N] if trans_b else [N, K]
    model = _make_model(
        [
            helper.make_node(
                "Gemm", ["A", "W"], ["Y"], name="gemm", transA=int(trans_a), transB=int(trans_b)
            )
        ],
        "dynamic_gemm",
        [_vi("A", TensorProto.FLOAT, a_shape), _vi("W", TensorProto.FLOAT, b_shape)],
        [_vi("Y", TensorProto.FLOAT, [M, K])],
    )
    feeds = {
        "A": _RNG.standard_normal(a_shape).astype(np.float32),
        "W": _RNG.standard_normal(b_shape).astype(np.float32),
    }
    expected = _run_ort(model, feeds)
    before = model.SerializeToString()
    _apply_matmul_to_conv(model, defer_rank_conversion=defer)
    onnx.checker.check_model(model, full_check=True)
    assert model.SerializeToString() == before
    _assert_outputs_match(expected, _run_ort(model, feeds))


@pytest.mark.parametrize("rank", [2, 3])
def test_eligible_deferred_conversion_keeps_rank_adapters(rank):
    """Keep eligible Conv rewrites valid before any later rank-promotion stage."""
    model, feeds = (_make_matmul_2d if rank == 2 else _make_matmul_3d)()
    expected = _run_ort(model, feeds)
    transformed = _apply_matmul_to_conv(copy.deepcopy(model), defer_rank_conversion=True)
    cache = GraphCache(transformed.graph)
    for adapter in transformed.graph.node:
        if adapter.op_type == "Unsqueeze":
            shape = cache.get_shape(adapter.output[0])
        elif adapter.op_type == "Squeeze":
            shape = cache.get_shape(adapter.input[0])
        else:
            continue
        assert shape is not None and len(shape) == 4
    onnx.checker.check_model(transformed, full_check=True)
    _assert_outputs_match(expected, _run_ort(transformed, feeds))


def test_internal_pipeline_executes_eligible_two_dimensional_matmul():
    """Run the registered deferred conversion through the actual internal sequence."""
    model, feeds = _make_matmul_2d()
    expected = _run_ort(model, feeds)
    transformed = _transform_make_dla_compatible(copy.deepcopy(model), verbose=False)
    onnx.checker.check_model(transformed, full_check=True)
    _assert_outputs_match(expected, _run_ort(transformed, feeds))


@pytest.mark.parametrize("axis", [None, -1, 1])
@pytest.mark.parametrize("shared", [False, True])
def test_per_axis_dq_weight_layout_is_preserved(axis, shared):
    """Remap implicit and negative quantization axes while preserving shared DQ outputs."""
    model, feeds = _make_matmul_dq_weight()
    for tensor in model.graph.initializer:
        if tensor.name == "W_scale":
            tensor.CopyFrom(_init("W_scale", np.linspace(0.01, 0.02, K, dtype=np.float32)))
        elif tensor.name == "W_zp":
            tensor.CopyFrom(_init("W_zp", np.zeros(K, np.int8)))
    if axis is not None:
        model.graph.node[0].attribute.append(helper.make_attribute("axis", axis))
    if shared:
        model.graph.output.append(_vi("W_dq", TensorProto.FLOAT, [N, K]))
    onnx.checker.check_model(model, full_check=True)
    expected = _run_ort(model, feeds)
    transformed = _apply_matmul_to_conv(copy.deepcopy(model))
    onnx.checker.check_model(transformed, full_check=True)
    _assert_outputs_match(expected, _run_ort(transformed, feeds))


@pytest.mark.parametrize("quantized", [False, True])
def test_public_weight_initializer_keeps_original_layout(quantized):
    """Retain public weight tensors while allocating a Conv kernel for this consumer."""
    model, feeds = (_make_matmul_dq_weight if quantized else _make_matmul_2d)()
    name = "W_q" if quantized else "W"
    model.graph.output.append(
        _vi(name, TensorProto.INT8 if quantized else TensorProto.FLOAT, [N, K])
    )
    onnx.checker.check_model(model, full_check=True)
    expected = _run_ort(model, feeds)
    transformed = _apply_matmul_to_conv(copy.deepcopy(model))
    onnx.checker.check_model(transformed, full_check=True)
    _assert_outputs_match(expected, _run_ort(transformed, feeds))


@pytest.mark.parametrize("quantized", [False, True])
@pytest.mark.parametrize("defer", [False, True])
def test_overridable_weight_initializer_uses_runtime_feed(quantized, defer):
    """Preserve runtime weight overrides instead of converting their initializer defaults."""
    model, feeds = (_make_matmul_dq_weight if quantized else _make_matmul_2d)()
    name = "W_q" if quantized else "W"
    tensor = next(tensor for tensor in model.graph.initializer if tensor.name == name)
    model.graph.input.append(_vi(name, tensor.data_type, list(tensor.dims)))
    feeds[name] = -numpy_helper.to_array(tensor)
    onnx.checker.check_model(model, full_check=True)
    expected = _run_ort(model, feeds)
    transformed = _apply_matmul_to_conv(copy.deepcopy(model), defer_rank_conversion=defer)
    onnx.checker.check_model(transformed, full_check=True)
    assert not any(node.op_type == "Conv" for node in transformed.graph.node)
    _assert_outputs_match(expected, _run_ort(transformed, feeds))


@pytest.mark.parametrize("op_type", ["MatMul", "Gemm"])
def test_custom_matrix_function_is_not_lowered_as_builtin(op_type):
    """Preserve custom matrix functions whose result differs from the standard operation."""
    function = helper.make_function(
        "custom",
        op_type,
        ["A", "W"],
        ["Y"],
        [
            helper.make_node("MatMul", ["A", "W"], ["product"]),
            helper.make_node("Neg", ["product"], ["Y"]),
        ],
        opset_imports=[helper.make_opsetid("", 17)],
    )
    model = helper.make_model(
        helper.make_graph(
            [helper.make_node(op_type, ["A", "W"], ["Y"], name="custom_matrix", domain="custom")],
            "custom_matrix",
            [_vi("A", TensorProto.FLOAT, [M, N])],
            [_vi("Y", TensorProto.FLOAT, [M, K])],
            initializer=[_init("W", _RNG.standard_normal([N, K]).astype(np.float32))],
        ),
        opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("custom", 1)],
        functions=[function],
    )
    _clamp_ir_version(model)
    onnx.checker.check_model(model, full_check=True)
    feeds = {"A": _RNG.standard_normal([M, N]).astype(np.float32)}
    expected = _run_ort(model, feeds)
    before = model.SerializeToString()
    _apply_matmul_to_conv(model)
    assert model.SerializeToString() == before
    _assert_outputs_match(expected, _run_ort(model, feeds))


@pytest.mark.parametrize("collision", ["activation", "cloned-dq"])
def test_generated_matrix_names_preserve_unrelated_outputs(collision):
    """Allocate fresh generated names across activation and cloned DQ branches."""
    model, feeds = (_make_matmul_dq_weight if collision == "cloned-dq" else _make_matmul_2d)()
    if collision == "cloned-dq":
        model.graph.output.append(_vi("W_dq", TensorProto.FLOAT, [N, K]))
    name = "W_dq_conv4d_0" if collision == "cloned-dq" else "mm_tr_before0_0"
    model.graph.node.insert(0, helper.make_node("Identity", ["A"], [name], name="existing_alias"))
    model.graph.output.append(_vi(name, TensorProto.FLOAT, [M, N]))
    onnx.checker.check_model(model, full_check=True)
    expected = _run_ort(model, feeds)
    transformed = _apply_matmul_to_conv(copy.deepcopy(model))
    onnx.checker.check_model(transformed, full_check=True)
    _assert_outputs_match(expected, _run_ort(transformed, feeds))


def test_integer_matmul_remains_a_valid_matmul():
    """Keep integer matrix multiplication on its supported operator and preserve exact values."""
    data = np.arange(6, dtype=np.int64).reshape(2, 3)
    weight = np.arange(12, dtype=np.int64).reshape(3, 4)
    model = _make_model(
        [helper.make_node("MatMul", ["A", "W"], ["Y"], name="integer_mm")],
        "integer_mm",
        [_vi("A", TensorProto.INT64, [2, 3])],
        [_vi("Y", TensorProto.INT64, [2, 4])],
        [_init("W", weight)],
    )
    expected = ReferenceEvaluator(model).run(None, {"A": data})[0]
    transformed = _apply_matmul_to_conv(copy.deepcopy(model))
    onnx.checker.check_model(transformed, full_check=True)
    assert any(node.op_type == "MatMul" for node in transformed.graph.node)
    np.testing.assert_array_equal(
        ReferenceEvaluator(transformed).run(None, {"A": data})[0], expected
    )


@pytest.mark.parametrize("symbolic", [False, True])
def test_batched_weight_fallback_preserves_values_without_decoding(symbolic):
    """Promote batched matrix layouts using ranks even when dimension values are symbolic."""
    weight = _RNG.standard_normal([2, N, K]).astype(np.float32)
    model = _make_model(
        [helper.make_node("MatMul", ["A", "W"], ["Y"], name="batched")],
        "batched_fallback",
        [_vi("A", TensorProto.FLOAT, [2, M, N])],
        [_vi("Y", TensorProto.FLOAT, [2, M, K])],
        [_init("W", weight)],
    )
    if symbolic:
        model.graph.input[0].CopyFrom(_vi("A", TensorProto.FLOAT, [2, "rows", N]))
        model.graph.output[0].CopyFrom(_vi("Y", TensorProto.FLOAT, [2, "rows", K]))
    onnx.checker.check_model(model, full_check=True)
    feeds = {"A": _RNG.standard_normal([2, M, N]).astype(np.float32)}
    expected = _run_ort(model, feeds)
    transformed = _apply_matmul_to_conv(copy.deepcopy(model))
    onnx.checker.check_model(transformed, full_check=True)
    assert any(node.op_type == "MatMul" for node in transformed.graph.node)
    assert (
        next(tensor for tensor in transformed.graph.initializer if tensor.name == "W").raw_data
        == next(tensor for tensor in model.graph.initializer if tensor.name == "W").raw_data
    )
    _assert_outputs_match(expected, _run_ort(transformed, feeds))


@pytest.mark.parametrize("weight_rank", [3, 4])
def test_batched_weight_output_rank_survives_symbolic_metadata(weight_rank):
    """Restore the declared batch rank independently of concrete output dimension values."""
    weight = _RNG.standard_normal([*([1] * (weight_rank - 2)), N, K]).astype(np.float32)
    model = _make_model(
        [helper.make_node("MatMul", ["A", "W"], ["Y"], name="batched")],
        "batched_output",
        [_vi("A", TensorProto.FLOAT, [M, N])],
        [_vi("Y", TensorProto.FLOAT, [*([1] * (weight_rank - 2)), M, K])],
        [_init("W", weight)],
    )
    model.graph.output[0].CopyFrom(
        _vi("Y", TensorProto.FLOAT, [f"dim_{index}" for index in range(weight_rank)])
    )
    onnx.checker.check_model(model, full_check=True)
    feeds = {"A": _RNG.standard_normal([M, N]).astype(np.float32)}
    expected = _run_ort(model, feeds)
    transformed = _apply_matmul_to_conv(copy.deepcopy(model))
    onnx.checker.check_model(transformed, full_check=True)
    _assert_outputs_match(expected, _run_ort(transformed, feeds))


@pytest.mark.parametrize("quantized", [False, True])
def test_captured_weight_layout_is_preserved(quantized):
    """Clone converted weights when nested control flow still reads their original layout."""
    model, feeds = (_make_matmul_dq_weight if quantized else _make_matmul_2d)()
    name = "W_dq" if quantized else "W"
    branch = helper.make_graph(
        [helper.make_node("Identity", [name], ["branch_out"])],
        "branch",
        [],
        [_vi("branch_out", TensorProto.FLOAT, [N, K])],
    )
    model.graph.input.append(_vi("condition", TensorProto.BOOL, []))
    model.graph.node.append(
        helper.make_node(
            "If",
            ["condition"],
            ["captured_weight"],
            name="capture",
            then_branch=branch,
            else_branch=copy.deepcopy(branch),
        )
    )
    model.graph.output.append(_vi("captured_weight", TensorProto.FLOAT, [N, K]))
    feeds["condition"] = np.asarray(True)
    onnx.checker.check_model(model, full_check=True)
    expected = _run_ort(model, feeds)
    transformed = _apply_matmul_to_conv(copy.deepcopy(model))
    onnx.checker.check_model(transformed, full_check=True)
    actual = _run_ort(transformed, feeds)
    np.testing.assert_array_equal(actual["captured_weight"], expected["captured_weight"])
    _assert_outputs_match(expected, actual)


def test_blocked_weight_dequantization_retains_its_layout():
    """Keep blocked scale/data coordinates intact while promoting only the DQ result."""
    model, feeds = _make_matmul_dq_weight()
    model.opset_import[0].version = 21
    model.graph.node[0].attribute.extend(
        [helper.make_attribute("axis", 1), helper.make_attribute("block_size", 4)]
    )
    for tensor in model.graph.initializer:
        if tensor.name == "W_scale":
            tensor.CopyFrom(_init("W_scale", np.full([N, K // 4], 0.05, np.float32)))
        elif tensor.name == "W_zp":
            tensor.CopyFrom(_init("W_zp", np.zeros([N, K // 4], np.int8)))
    onnx.checker.check_model(model, full_check=True)
    expected = ReferenceEvaluator(model).run(None, feeds)
    transformed = _apply_matmul_to_conv(copy.deepcopy(model))
    onnx.checker.check_model(transformed, full_check=True)
    assert any(node.op_type == "MatMul" for node in transformed.graph.node)
    for actual, reference in zip(ReferenceEvaluator(transformed).run(None, feeds), expected):
        np.testing.assert_array_equal(actual, reference)


@pytest.mark.parametrize("domain", ["custom", "com.microsoft"])
def test_opaque_weight_function_is_not_reinterpreted_as_dq(domain):
    """Preserve model-local weight functions with nonpointwise semantics."""
    model, feeds = _make_matmul_dq_weight()
    function = helper.make_function(
        domain,
        "DequantizeLinear",
        ["data", "scale", "zero"],
        ["output"],
        [
            helper.make_node("Cast", ["data"], ["cast"], to=TensorProto.FLOAT),
            helper.make_node("Softmax", ["cast"], ["output"], axis=1),
        ],
        opset_imports=[helper.make_opsetid("", 17)],
    )
    model.functions.append(function)
    model.opset_import.append(helper.make_opsetid(domain, 1))
    model.graph.node[0].domain = domain
    onnx.checker.check_model(model, full_check=True)
    expected = ReferenceEvaluator(model).run(None, feeds)
    transformed = _apply_matmul_to_conv(copy.deepcopy(model))
    onnx.checker.check_model(transformed, full_check=True)
    assert any(node.op_type == "MatMul" for node in transformed.graph.node)
    for actual, reference in zip(ReferenceEvaluator(transformed).run(None, feeds), expected):
        np.testing.assert_allclose(actual, reference, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize(("vector_a", "vector_b"), [(True, False), (False, True), (True, True)])
@pytest.mark.parametrize("defer", [False, True])
def test_vector_matmul_is_refused_or_left_for_later_conversion(vector_a, vector_b, defer):
    """Avoid applying matrix-only rank promotion to vector multiplication semantics."""
    a_shape = [N] if vector_a else [M, N]
    w_shape = [N] if vector_b else [N, K]
    output_shape = [] if vector_a and vector_b else [K] if vector_a else [M]
    model = _make_model(
        [helper.make_node("MatMul", ["A", "W"], ["Y"], name="vector")],
        "vector",
        [_vi("A", TensorProto.FLOAT, a_shape)],
        [_vi("Y", TensorProto.FLOAT, output_shape)],
        [_init("W", _RNG.standard_normal(w_shape).astype(np.float32))],
    )
    onnx.checker.check_model(model, full_check=True)
    feeds = {"A": _RNG.standard_normal(a_shape).astype(np.float32)}
    expected = _run_ort(model, feeds)
    before = model.SerializeToString()
    if defer:
        _apply_matmul_to_conv(model, defer_rank_conversion=True)
        assert model.SerializeToString() == before
        _assert_outputs_match(expected, _run_ort(model, feeds))
    else:
        with pytest.raises(ValueError, match=r"[Uu]nsupported|[Vv]ector"):
            _apply_matmul_to_conv(model)
        assert model.SerializeToString() == before


@pytest.mark.parametrize("empty", ["rows", "outputs"])
def test_empty_matrix_dimensions_remain_valid(empty):
    """Keep empty matrix results on MatMul instead of creating an unsupported Conv kernel."""
    rows = 0 if empty == "rows" else M
    outputs = 0 if empty == "outputs" else K
    weight = _RNG.standard_normal([N, outputs]).astype(np.float32)
    model = _make_model(
        [helper.make_node("MatMul", ["A", "W"], ["Y"], name="empty")],
        "empty",
        [_vi("A", TensorProto.FLOAT, [rows, N])],
        [_vi("Y", TensorProto.FLOAT, [rows, outputs])],
        [_init("W", weight)],
    )
    onnx.checker.check_model(model, full_check=True)
    feeds = {"A": _RNG.standard_normal([rows, N]).astype(np.float32)}
    expected = _run_ort(model, feeds)
    transformed = _apply_matmul_to_conv(copy.deepcopy(model))
    onnx.checker.check_model(transformed, full_check=True)
    assert any(node.op_type == "MatMul" for node in transformed.graph.node)
    _assert_outputs_match(expected, _run_ort(transformed, feeds))


def test_legacy_rank_adapter_schema_refuses_before_mutation():
    """Refuse old Unsqueeze/Squeeze schemas before adding input-form rank adapters."""
    model, feeds = _make_matmul_2d()
    model.opset_import[0].version = 11
    onnx.checker.check_model(model, full_check=True)
    _run_ort(model, feeds)
    before = model.SerializeToString()
    with pytest.raises(ValueError, match="opset"):
        _apply_matmul_to_conv(model)
    assert model.SerializeToString() == before


def test_legacy_ir_registers_new_initializer_inputs():
    """Keep IR3 valid when rank adapters introduce new constant axes tensors."""
    model, feeds = _make_matmul_2d()
    model.ir_version = 3
    for tensor in model.graph.initializer:
        model.graph.input.append(_vi(tensor.name, tensor.data_type, list(tensor.dims)))
    onnx.checker.check_model(model, full_check=True)
    expected = _run_ort(model, feeds)
    transformed = _apply_matmul_to_conv(copy.deepcopy(model))
    onnx.checker.check_model(transformed, full_check=True)
    assert {tensor.name for tensor in transformed.graph.initializer} <= {
        value.name for value in transformed.graph.input
    }
    _assert_outputs_match(expected, _run_ort(transformed, feeds))


@pytest.mark.parametrize("layout", ["symbolic_2d", "symbolic_3d", "batch_4d", "rank_5d"])
@pytest.mark.parametrize("defer", [False, True])
def test_ineligible_activation_layout_keeps_matmul_semantics(layout, defer):
    """Route valid dynamic or unsupported Conv activation layouts through matrix fallback."""
    shape = {
        "symbolic_2d": [M, N],
        "symbolic_3d": [B, M, N],
        "batch_4d": [2, 3, M, N],
        "rank_5d": [2, 1, 3, M, N],
    }[layout]
    metadata = list(shape)
    if layout.startswith("symbolic"):
        metadata[0] = "batch"
    model = _make_model(
        [helper.make_node("MatMul", ["A", "W"], ["Y"], name="matrix")],
        "activation_fallback",
        [_vi("A", TensorProto.FLOAT, metadata)],
        [_vi("Y", TensorProto.FLOAT, [*metadata[:-1], K])],
        [_init("W", _RNG.standard_normal((N, K)).astype(np.float32))],
    )
    feeds = {"A": _RNG.standard_normal(shape).astype(np.float32)}
    expected = _run_ort(model, feeds)
    transformed = _apply_matmul_to_conv(copy.deepcopy(model), defer_rank_conversion=defer)
    assert any(node.op_type == "MatMul" for node in transformed.graph.node)
    onnx.checker.check_model(transformed, full_check=True)
    _assert_outputs_match(expected, _run_ort(transformed, feeds))


@pytest.mark.parametrize("defer", [False, True])
def test_unknown_intermediate_result_rank_is_preserved(defer):
    """Leave a dynamic matrix operation unchanged when its result rank is absent."""
    model, feeds = _make_matmul_dynamic_weight()
    node = next(node for node in model.graph.node if node.op_type == "MatMul")
    node.output[0] = "product"
    model.graph.node.append(helper.make_node("Identity", ["product"], ["Y"]))
    retained = [info for info in model.graph.value_info if info.name != "product"]
    del model.graph.value_info[:]
    model.graph.value_info.extend(retained)
    onnx.checker.check_model(model, full_check=True)
    expected = _run_ort(model, feeds)
    before = model.SerializeToString()
    _apply_matmul_to_conv(model, defer_rank_conversion=defer)
    onnx.checker.check_model(model, full_check=True)
    assert model.SerializeToString() == before
    _assert_outputs_match(expected, _run_ort(model, feeds))


def test_test_fixture_inference_supplies_intermediate_metadata():
    """Exercise the fixture inference rather than silently accepting keyword failures."""
    pytest.importorskip(
        "onnxruntime.tools.symbolic_shape_infer",
        reason="This fixture requires ORT symbolic shape inference",
    )
    model = _make_model(
        [
            helper.make_node("Identity", ["X"], ["middle"]),
            helper.make_node("Identity", ["middle"], ["Y"]),
        ],
        "inferred_fixture",
        [_vi("X", TensorProto.FLOAT, [2, 3])],
        [_vi("Y", TensorProto.FLOAT, [2, 3])],
    )
    assert GraphCache(model.graph).get_shape("middle") == [2, 3]


@pytest.mark.parametrize("per_axis", [False, True])
def test_dq_matrix_rewrite_matches_independent_reference(per_axis):
    """Check quantized-weight rewrites against independent unfused execution."""
    model, feeds = _make_matmul_dq_weight()
    # The reference evaluator implements these INT8 controls starting at opset 19.
    model.opset_import[0].version = 21
    if per_axis:
        for tensor in model.graph.initializer:
            if tensor.name == "W_scale":
                tensor.CopyFrom(_init("W_scale", np.linspace(0.01, 0.02, K, dtype=np.float32)))
            elif tensor.name == "W_zp":
                tensor.CopyFrom(_init("W_zp", np.zeros(K, np.int8)))
    expected = ReferenceEvaluator(model).run(None, feeds)[0]
    transformed = _apply_matmul_to_conv(copy.deepcopy(model))
    onnx.checker.check_model(transformed, full_check=True)
    for actual in [
        ReferenceEvaluator(transformed).run(None, feeds)[0],
        _run_ort(model, feeds)["Y"],
        _run_ort(transformed, feeds)["Y"],
    ]:
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)
