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
# Unit tests for :mod:`modelopt.onnx.graph_surgery.dla_transforms.dla_remove_qdq`.

"""ORT CPU parity tests for ``remove_qdq`` / ``_remove_qdq_from_model``.

Transform rule
--------------
A contiguous QuantizeLinear → DequantizeLinear pair is removed when the
quantizer zero-point element type belongs to ``qdq_quantized_dtypes``
(default: UINT16 and INT16).

After removal the DQ output is rewired directly to the Q input; downstream
nodes continue to see a float32 tensor with no quantisation error.

ORT note: ORT does not reliably execute UINT16 / INT16 QDQ ops across all
opset / runtime versions, so the base model is *not* executed with ORT.
Instead we build numpy references that model the expected post-transform
output (the float computation without any quantisation round-trip) and
compare against ORT on the transformed (float-only) graph.

Coverage
--------
single uint16 Q→DQ          → pair removed, Y = relu(X)
single uint8  Q→DQ          → NOT removed (not in default types)
custom dtype  uint8 remove  → pair removed when uint8 specified
chained uint16+uint16       → both pairs removed (consecutive Q→DQ→Q→DQ)
chained uint16+uint8        → only uint16 pair removed
graph-output Q→DQ           → producer rewired to graph output name
no QDQ nodes                → graph unchanged
int16 Q→DQ                  → removed by default
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
_SAVED_MODELS_DIR = _REPO_ROOT / "scratch_space" / "remove_qdq_test_models"
_OPSET_VERSION = 17
_MAX_IR_VERSION_FOR_ORT = 9


# ── Dynamic import ────────────────────────────────────────────────────────────


def _load_remove_qdq():
    mod_key = "modelopt.onnx.graph_surgery.dla_transforms.dla_remove_qdq"
    if mod_key in sys.modules:
        m = sys.modules[mod_key]
        return m._remove_qdq_from_model, m

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
        _REPO_ROOT / "modelopt/onnx/graph_surgery/dla_transforms/dla_remove_qdq.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_key] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod._remove_qdq_from_model, mod


_remove_qdq_from_model, _remove_qdq_mod = _load_remove_qdq()
_DEFAULT_QUANT_TYPES = _remove_qdq_mod.DEFAULT_QDQ_REMOVE_QUANT_TYPES
_parse_dtype_list = _remove_qdq_mod.parse_qdq_quantized_dtype_list


# ── Shared utilities ──────────────────────────────────────────────────────────


def _vi(name: str, elem_type: int, shape: list) -> onnx.ValueInfoProto:
    return helper.make_tensor_value_info(name, elem_type, shape)


def _clamp_ir(model: onnx.ModelProto) -> None:
    model.ir_version = min(model.ir_version, _MAX_IR_VERSION_FOR_ORT)


def _prepare_model(model: onnx.ModelProto) -> onnx.ModelProto:
    """Run symbolic shape inference in-place to populate value_info for all intermediate tensors."""
    try:
        from onnxruntime.tools.symbolic_shape_infer import (  # type: ignore[import-not-found]
            SymbolicShapeInference,
        )

        inferred = SymbolicShapeInference.infer_shapes(model)
        if inferred is not None:
            del model.graph.value_info[:]
            model.graph.value_info.extend(inferred.graph.value_info)
    except Exception:
        # Fall back to ONNX built-in shape inference
        try:
            inferred = onnx.shape_inference.infer_shapes(model)
            del model.graph.value_info[:]
            model.graph.value_info.extend(inferred.graph.value_info)
        except Exception:
            pass
    return model


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


def _apply(model, quant_types=None, **kw):
    """Run _remove_qdq_from_model with sensible defaults."""
    qt = _DEFAULT_QUANT_TYPES if quant_types is None else _parse_dtype_list(quant_types)
    return _remove_qdq_from_model(
        model,
        keep_clip_after_inputs=kw.get("keep_clip_after_inputs", False),
        max_chained_qdq_pairs=kw.get("max_chained_qdq_pairs", 5),
        quant_types_to_remove=qt,
    )


def _make_qdq_init(name_prefix, scale_val, zp_val, zp_np_dtype):
    scale = numpy_helper.from_array(np.float32(scale_val), name=f"{name_prefix}_scale")
    zp = numpy_helper.from_array(zp_np_dtype(zp_val), name=f"{name_prefix}_zp")
    return scale, zp


def _make_qdq_nodes(name_prefix, input_name, output_name, onnx_zp_type):
    """Return (Q node, DQ node, intermediate quantized tensor name)."""
    q_out = f"{name_prefix}_q_out"
    q = helper.make_node(
        "QuantizeLinear",
        inputs=[input_name, f"{name_prefix}_scale", f"{name_prefix}_zp"],
        outputs=[q_out],
        name=f"{name_prefix}_q",
    )
    dq = helper.make_node(
        "DequantizeLinear",
        inputs=[q_out, f"{name_prefix}_scale", f"{name_prefix}_zp"],
        outputs=[output_name],
        name=f"{name_prefix}_dq",
    )
    return q, dq, q_out


_RNG = np.random.default_rng(2024)


# ── 1. Single uint16 Q→DQ: pair removed, output is relu(X) ───────────────────


def test_remove_single_uint16_qdq():
    """uint16 Q→DQ sandwiching Relu: pair removed; ORT output matches relu(X)."""
    shape = [2, 4, 8]
    scale, zp = _make_qdq_init("u16", 0.01, 0, np.uint16)
    relu_in = helper.make_node("Relu", ["X"], ["relu_out"], name="relu_pre")
    q, dq, q_out = _make_qdq_nodes("u16", "relu_out", "Y", TensorProto.UINT16)

    graph = helper.make_graph(
        [relu_in, q, dq],
        "single_u16",
        [_vi("X", TensorProto.FLOAT, shape)],
        [_vi("Y", TensorProto.FLOAT, shape)],
        initializer=[scale, zp],
        value_info=[
            _vi("relu_out", TensorProto.FLOAT, shape),
            _vi(q_out, TensorProto.UINT16, shape),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", _OPSET_VERSION)])
    _clamp_ir(model)

    x = _RNG.standard_normal(shape).astype(np.float32)
    ref = np.maximum(x, 0.0)  # After removing QDQ: Y = relu(X)

    tr = copy.deepcopy(model)
    _prepare_model(tr)
    pairs_removed = _apply(tr)
    got = _run_ort(tr, {"X": x})
    _save(model, tr, "remove_single_u16")

    assert pairs_removed == 1, f"Expected 1 pair removed, got {pairs_removed}"
    assert got["Y"].shape == tuple(shape)
    np.testing.assert_allclose(got["Y"], ref, rtol=1e-5, atol=1e-5)
    assert not any(n.op_type in ("QuantizeLinear", "DequantizeLinear") for n in tr.graph.node)


# ── 2. Single uint8 Q→DQ: NOT removed (not in default types) ─────────────────


def test_remove_uint8_not_removed_by_default():
    """uint8 Q→DQ must NOT be removed with default quant_types (UINT16/INT16 only)."""
    shape = [3, 8]
    scale, zp = _make_qdq_init("u8", 0.1, 128, np.uint8)
    q, dq, q_out = _make_qdq_nodes("u8", "X", "Y", TensorProto.UINT8)

    graph = helper.make_graph(
        [q, dq],
        "u8_graph",
        [_vi("X", TensorProto.FLOAT, shape)],
        [_vi("Y", TensorProto.FLOAT, shape)],
        initializer=[scale, zp],
        value_info=[
            _vi(q_out, TensorProto.UINT8, shape),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", _OPSET_VERSION)])
    _clamp_ir(model)
    original_count = len(model.graph.node)

    tr = copy.deepcopy(model)
    _prepare_model(tr)
    pairs_removed = _apply(tr)

    assert pairs_removed == 0, f"Expected 0 pairs removed, got {pairs_removed}"
    assert len(tr.graph.node) == original_count
    assert any(n.op_type == "QuantizeLinear" for n in tr.graph.node)
    assert any(n.op_type == "DequantizeLinear" for n in tr.graph.node)


# ── 3. Custom dtype: uint8 explicitly added → removed ────────────────────────


def test_remove_uint8_with_custom_dtype():
    """uint8 Q→DQ removed when uint8 explicitly in qdq_quantized_dtypes."""
    shape = [3, 8]
    scale, zp = _make_qdq_init("u8c", 0.1, 128, np.uint8)
    q, dq, q_out = _make_qdq_nodes("u8c", "X", "Y", TensorProto.UINT8)

    graph = helper.make_graph(
        [q, dq],
        "u8_custom",
        [_vi("X", TensorProto.FLOAT, shape)],
        [_vi("Y", TensorProto.FLOAT, shape)],
        initializer=[scale, zp],
        value_info=[
            _vi(q_out, TensorProto.UINT8, shape),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", _OPSET_VERSION)])
    _clamp_ir(model)

    x = _RNG.standard_normal(shape).astype(np.float32)
    ref = x  # After removing QDQ: Y = X

    tr = copy.deepcopy(model)
    _prepare_model(tr)
    pairs_removed = _apply(tr, quant_types=["uint8"])
    got = _run_ort(tr, {"X": x})
    _save(model, tr, "remove_u8_custom")

    assert pairs_removed == 1
    np.testing.assert_allclose(got["Y"], ref, rtol=1e-5, atol=1e-5)
    assert not any(n.op_type in ("QuantizeLinear", "DequantizeLinear") for n in tr.graph.node)


# ── 4. Chained consecutive uint16 Q→DQ→Q→DQ: both pairs removed ─────────────


def test_remove_chained_uint16_qdq():
    """Two consecutive uint16 Q→DQ pairs in sequence: both removed, Y = X."""
    shape = [4, 8]
    scale1, zp1 = _make_qdq_init("ch1", 0.01, 0, np.uint16)
    scale2, zp2 = _make_qdq_init("ch2", 0.02, 0, np.uint16)

    # X → Q1 → DQ1("mid") → Q2 → DQ2 → Y
    q1, dq1, q1_out = _make_qdq_nodes("ch1", "X", "mid", TensorProto.UINT16)
    q2, dq2, q2_out = _make_qdq_nodes("ch2", "mid", "Y", TensorProto.UINT16)

    graph = helper.make_graph(
        [q1, dq1, q2, dq2],
        "chained_u16",
        [_vi("X", TensorProto.FLOAT, shape)],
        [_vi("Y", TensorProto.FLOAT, shape)],
        initializer=[scale1, zp1, scale2, zp2],
        value_info=[
            _vi(q1_out, TensorProto.UINT16, shape),
            _vi("mid", TensorProto.FLOAT, shape),
            _vi(q2_out, TensorProto.UINT16, shape),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", _OPSET_VERSION)])
    _clamp_ir(model)

    x = _RNG.standard_normal(shape).astype(np.float32)
    ref = x  # After removing both QDQ: Y = X

    tr = copy.deepcopy(model)
    _prepare_model(tr)
    pairs_removed = _apply(tr)
    got = _run_ort(tr, {"X": x})
    _save(model, tr, "remove_chained_u16")

    assert pairs_removed == 2, f"Expected 2 pairs removed, got {pairs_removed}"
    assert got["Y"].shape == tuple(shape)
    np.testing.assert_allclose(got["Y"], ref, rtol=1e-5, atol=1e-5)
    assert not any(n.op_type in ("QuantizeLinear", "DequantizeLinear") for n in tr.graph.node)


# ── 5. Chained uint16+uint8: only uint16 pair removed ────────────────────────


def test_remove_chained_mixed_dtypes():
    """Chain uint16 Q→DQ → Relu → uint8 Q→DQ: only uint16 pair removed."""
    shape = [4, 8]
    scale16, zp16 = _make_qdq_init("m16", 0.01, 0, np.uint16)
    scale8, zp8 = _make_qdq_init("m8", 0.1, 128, np.uint8)

    # X → Q16 → DQ16("after_16") → Relu("relu_out") → Q8 → DQ8 → Y
    q16, dq16, q16_out = _make_qdq_nodes("m16", "X", "after_16", TensorProto.UINT16)
    relu = helper.make_node("Relu", ["after_16"], ["relu_out"], name="relu")
    q8, dq8, q8_out = _make_qdq_nodes("m8", "relu_out", "Y", TensorProto.UINT8)

    graph = helper.make_graph(
        [q16, dq16, relu, q8, dq8],
        "mixed_chain",
        [_vi("X", TensorProto.FLOAT, shape)],
        [_vi("Y", TensorProto.FLOAT, shape)],
        initializer=[scale16, zp16, scale8, zp8],
        value_info=[
            _vi(q16_out, TensorProto.UINT16, shape),
            _vi("after_16", TensorProto.FLOAT, shape),
            _vi("relu_out", TensorProto.FLOAT, shape),
            _vi(q8_out, TensorProto.UINT8, shape),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", _OPSET_VERSION)])
    _clamp_ir(model)

    tr = copy.deepcopy(model)
    _prepare_model(tr)
    pairs_removed = _apply(tr)
    _save(model, tr, "remove_mixed_chain")

    assert pairs_removed == 1, f"Expected 1 pair removed, got {pairs_removed}"

    # uint16 nodes gone, uint8 nodes still present
    q_nodes = [n for n in tr.graph.node if n.op_type == "QuantizeLinear"]
    dq_nodes = [n for n in tr.graph.node if n.op_type == "DequantizeLinear"]
    assert len(q_nodes) == 1 and q_nodes[0].name == "m8_q", "Only uint8 Q node should remain"
    assert len(dq_nodes) == 1 and dq_nodes[0].name == "m8_dq", "Only uint8 DQ node should remain"

    # Relu must now consume X directly (uint16 bypassed)
    relu_node = next(n for n in tr.graph.node if n.op_type == "Relu")
    assert relu_node.input[0] == "X", f"Relu should consume X, got {relu_node.input[0]!r}"


# ── 6. Q→DQ directly to graph output ─────────────────────────────────────────


def test_remove_qdq_to_graph_output():
    """Q→DQ whose DQ output is a graph output: producer output rewired to graph output name."""
    shape = [2, 4]
    scale, zp = _make_qdq_init("go", 0.01, 0, np.uint16)

    relu = helper.make_node("Relu", ["X"], ["relu_out"], name="relu")
    q, dq, q_out = _make_qdq_nodes("go", "relu_out", "Y", TensorProto.UINT16)

    graph = helper.make_graph(
        [relu, q, dq],
        "graph_out",
        [_vi("X", TensorProto.FLOAT, shape)],
        [_vi("Y", TensorProto.FLOAT, shape)],
        initializer=[scale, zp],
        value_info=[
            _vi("relu_out", TensorProto.FLOAT, shape),
            _vi(q_out, TensorProto.UINT16, shape),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", _OPSET_VERSION)])
    _clamp_ir(model)

    x = _RNG.standard_normal(shape).astype(np.float32)
    ref = np.maximum(x, 0.0)  # After removing QDQ: Y = relu(X)

    tr = copy.deepcopy(model)
    _prepare_model(tr)
    pairs_removed = _apply(tr)
    got = _run_ort(tr, {"X": x})
    _save(model, tr, "remove_graph_output")

    assert pairs_removed == 1
    assert got["Y"].shape == tuple(shape)
    np.testing.assert_allclose(got["Y"], ref, rtol=1e-5, atol=1e-5)
    assert not any(n.op_type in ("QuantizeLinear", "DequantizeLinear") for n in tr.graph.node)


# ── 7. No QDQ nodes: graph unchanged ─────────────────────────────────────────


def test_remove_qdq_noop():
    """Model without any QDQ nodes: graph and ORT output unchanged."""
    shape = [2, 4, 8]
    relu = helper.make_node("Relu", ["X"], ["Y"], name="relu")
    graph = helper.make_graph(
        [relu],
        "no_qdq",
        [_vi("X", TensorProto.FLOAT, shape)],
        [_vi("Y", TensorProto.FLOAT, shape)],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", _OPSET_VERSION)])
    _clamp_ir(model)
    original_count = len(model.graph.node)

    x = _RNG.standard_normal(shape).astype(np.float32)
    ref = _run_ort(model, {"X": x})

    tr = copy.deepcopy(model)
    _prepare_model(tr)
    pairs_removed = _apply(tr)
    got = _run_ort(tr, {"X": x})
    _save(model, tr, "remove_noop")

    assert pairs_removed == 0
    assert len(tr.graph.node) == original_count
    np.testing.assert_array_equal(ref["Y"], got["Y"])


# ── 8. int16 Q→DQ: removed by default ────────────────────────────────────────


def test_remove_int16_qdq():
    """int16 Q→DQ is in the default remove types and must be removed."""
    shape = [2, 8]
    scale, zp = _make_qdq_init("i16", 0.02, 0, np.int16)
    q, dq, q_out = _make_qdq_nodes("i16", "X", "Y", TensorProto.INT16)

    graph = helper.make_graph(
        [q, dq],
        "i16_graph",
        [_vi("X", TensorProto.FLOAT, shape)],
        [_vi("Y", TensorProto.FLOAT, shape)],
        initializer=[scale, zp],
        value_info=[
            _vi(q_out, TensorProto.INT16, shape),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", _OPSET_VERSION)])
    _clamp_ir(model)

    x = _RNG.standard_normal(shape).astype(np.float32)
    ref = x  # After removing QDQ: Y = X

    tr = copy.deepcopy(model)
    _prepare_model(tr)
    pairs_removed = _apply(tr)
    got = _run_ort(tr, {"X": x})
    _save(model, tr, "remove_int16")

    assert pairs_removed == 1
    np.testing.assert_allclose(got["Y"], ref, rtol=1e-5, atol=1e-5)
    assert not any(n.op_type in ("QuantizeLinear", "DequantizeLinear") for n in tr.graph.node)
