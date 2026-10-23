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
import importlib

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from modelopt.onnx.graph_surgery.dla_transforms.dla_remove_qdq import _remove_qdq_from_model

ort = pytest.importorskip("onnxruntime", reason="onnxruntime required for ORT parity checks")

_OPSET_VERSION = 21
_MAX_IR_VERSION_FOR_ORT = 9


_onnx_dtypes_mod = importlib.import_module("modelopt.onnx.graph_surgery.utils.graph_utils")
_DEFAULT_QUANT_TYPES = _onnx_dtypes_mod.DEFAULT_QDQ_REMOVE_QUANT_TYPES
_parse_dtype_list = _onnx_dtypes_mod.parse_qdq_quantized_dtype_list


# ── Shared utilities ──────────────────────────────────────────────────────────


def _vi(name: str, elem_type: int, shape: list) -> onnx.ValueInfoProto:
    """Build a typed tensor descriptor for a small test graph."""
    return helper.make_tensor_value_info(name, elem_type, shape)


def _clamp_ir(model: onnx.ModelProto) -> None:
    """Keep the test model compatible with the supported ORT versions."""
    model.ir_version = min(model.ir_version, _MAX_IR_VERSION_FOR_ORT)


def _prepare_model(model: onnx.ModelProto) -> onnx.ModelProto:
    """Refresh intermediate metadata and surface invalid graph or inference errors."""
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    onnx.checker.check_model(inferred, full_check=True)
    del model.graph.value_info[:]
    model.graph.value_info.extend(inferred.graph.value_info)
    return model


def _run_ort(model: onnx.ModelProto, feeds: dict) -> dict:
    """Execute the test model on the CPU and return named outputs."""
    _clamp_ir(model)
    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    return {o.name: sess.run([o.name], feeds)[0] for o in model.graph.output}


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
    """Build a scale and zero point for the selected quantized storage type."""
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

    assert pairs_removed == 1
    np.testing.assert_allclose(got["Y"], ref, rtol=1e-5, atol=1e-5)
    assert not any(n.op_type in ("QuantizeLinear", "DequantizeLinear") for n in tr.graph.node)


@pytest.mark.parametrize("extra_outputs", [[], ["a"], ["a", "b"], ["activation", "a", "b"]])
def test_remove_qdq_fanout_preserves_every_graph_output(extra_outputs):
    """Preserve all output names when dequantizers share one quantizer."""
    shape = [1, 1, 1, 3]
    nodes = [
        helper.make_node("Relu", ["X"], ["activation"]),
        helper.make_node("QuantizeLinear", ["activation", "scale", "zero"], ["q"]),
        helper.make_node("DequantizeLinear", ["q", "scale", "zero"], ["a"]),
        helper.make_node("DequantizeLinear", ["q", "scale", "zero"], ["b"]),
        helper.make_node("Add", ["a", "b"], ["Y"]),
    ]
    model = helper.make_model(
        helper.make_graph(
            nodes,
            "fanout",
            [_vi("X", TensorProto.FLOAT, shape)],
            [_vi(name, TensorProto.FLOAT, shape) for name in [*extra_outputs, "Y"]],
            initializer=[
                numpy_helper.from_array(np.float32(1), "scale"),
                numpy_helper.from_array(np.uint16(0), "zero"),
            ],
            value_info=[_vi("activation", TensorProto.FLOAT, shape)],
        ),
        opset_imports=[helper.make_opsetid("", 21)],
    )
    _clamp_ir(model)
    assert _apply(model) == 2
    onnx.checker.check_model(model)
    data = np.array([-1, 2, 3], dtype=np.float32).reshape(shape)
    results = _run_ort(model, {"X": data})
    for name in extra_outputs:
        np.testing.assert_array_equal(results[name], np.maximum(data, 0))
    np.testing.assert_array_equal(results["Y"], np.maximum(data, 0) * 2)


@pytest.mark.parametrize("empty_zero_point", [False, True])
def test_remove_default_uint8_qdq_without_explicit_zero_point(empty_zero_point):
    """Recognize the default UINT8 zero point when removing a selected QDQ pair."""
    shape = [1, 1, 1, 3]
    model = helper.make_model(
        helper.make_graph(
            [
                helper.make_node(
                    "QuantizeLinear", ["X", "scale"] + ([""] if empty_zero_point else []), ["q"]
                ),
                helper.make_node(
                    "DequantizeLinear", ["q", "scale"] + ([""] if empty_zero_point else []), ["Y"]
                ),
            ],
            "default_zero",
            [_vi("X", TensorProto.FLOAT, shape)],
            [_vi("Y", TensorProto.FLOAT, shape)],
            initializer=[numpy_helper.from_array(np.float32(1), "scale")],
        ),
        opset_imports=[helper.make_opsetid("", 17)],
    )
    _clamp_ir(model)
    assert _apply(model, quant_types=[TensorProto.UINT8]) == 1
    data = np.arange(3, dtype=np.float32).reshape(shape)
    np.testing.assert_array_equal(_run_ort(model, {"X": data})["Y"], data)


@pytest.mark.parametrize("storage_dtype", [TensorProto.UINT16, TensorProto.INT16])
@pytest.mark.parametrize("empty_zero_point", [False, True])
def test_output_dtype_selects_quantizers_without_a_zero_point(storage_dtype, empty_zero_point):
    """Explicit quantized output types control selection when the zero-point slot is absent."""
    extra = [""] if empty_zero_point else []
    shape = [2, 3]
    model = helper.make_model(
        helper.make_graph(
            [
                helper.make_node(
                    "QuantizeLinear", ["X", "scale", *extra], ["q"], output_dtype=storage_dtype
                ),
                helper.make_node("DequantizeLinear", ["q", "scale", *extra], ["Y"]),
            ],
            "explicit_storage",
            [_vi("X", TensorProto.FLOAT, shape)],
            [_vi("Y", TensorProto.FLOAT, shape)],
            initializer=[numpy_helper.from_array(np.float32(0.25), "scale")],
            value_info=[_vi("q", storage_dtype, shape)],
        ),
        opset_imports=[helper.make_opsetid("", 21)],
        ir_version=_MAX_IR_VERSION_FOR_ORT,
    )
    onnx.checker.check_model(model, full_check=True)
    excluded = copy.deepcopy(model)
    before = excluded.SerializeToString()
    assert _apply(excluded, quant_types=["uint8"]) == 0
    assert excluded.SerializeToString() == before
    assert _apply(model) == 1
    onnx.checker.check_model(model, full_check=True)
    x = np.arange(6, dtype=np.float32).reshape(shape)
    np.testing.assert_array_equal(_run_ort(model, {"X": x})["Y"], x)


@pytest.mark.parametrize("captured", ["activation", "q", "D"])
def test_remove_qdq_keeps_values_captured_by_control_flow(captured):
    """QDQ removal leaves lexically captured tensors defined with their original types."""
    shape = [3]
    dtype = TensorProto.UINT8 if captured == "q" else TensorProto.FLOAT
    branch = helper.make_graph(
        [helper.make_node("Identity", [captured], ["branch_out"])],
        "branch",
        [],
        [_vi("branch_out", dtype, shape)],
    )
    public_output = "Y" if captured == "D" else "D"
    bridge = [helper.make_node("Identity", ["D"], ["Y"])] if captured == "D" else []
    model = helper.make_model(
        helper.make_graph(
            [
                helper.make_node("Relu", ["X"], ["activation"]),
                helper.make_node("QuantizeLinear", ["activation", "scale", "zero"], ["q"]),
                helper.make_node("DequantizeLinear", ["q", "scale", "zero"], ["D"]),
                *bridge,
                helper.make_node(
                    "If", ["condition"], ["Z"], then_branch=branch, else_branch=branch
                ),
            ],
            "captured",
            [_vi("X", TensorProto.FLOAT, shape), _vi("condition", TensorProto.BOOL, [])],
            [_vi(public_output, TensorProto.FLOAT, shape), _vi("Z", dtype, shape)],
            initializer=[
                numpy_helper.from_array(np.float32(1), "scale"),
                numpy_helper.from_array(np.uint8(0), "zero"),
            ],
            value_info=[
                _vi("activation", TensorProto.FLOAT, shape),
                _vi("q", TensorProto.UINT8, shape),
            ],
        ),
        opset_imports=[helper.make_opsetid("", 21)],
        ir_version=_MAX_IR_VERSION_FOR_ORT,
    )
    onnx.checker.check_model(model, full_check=True)
    feeds = {"X": np.array([-1, 1, 2], dtype=np.float32), "condition": np.array(True)}
    expected = _run_ort(model, feeds)
    before = model.SerializeToString()
    count = _apply(model, quant_types=["uint8"])
    onnx.checker.check_model(model, full_check=True)
    assert count == 0 and model.SerializeToString() == before
    for name, actual in _run_ort(model, feeds).items():
        np.testing.assert_array_equal(actual, expected[name])


@pytest.mark.parametrize(
    ("opset", "dtype"),
    [(10, TensorProto.FLOAT), (21, TensorProto.FLOAT16), (21, TensorProto.FLOAT)],
)
def test_input_clip_respects_opset_and_output_dtype(opset, dtype):
    """Input clipping uses a valid schema and returns values in the declared output dtype."""
    np_dtype = np.float16 if dtype == TensorProto.FLOAT16 else np.float32
    model = helper.make_model(
        helper.make_graph(
            [
                helper.make_node("QuantizeLinear", ["X", "scale", "zero"], ["q"]),
                helper.make_node("DequantizeLinear", ["q", "scale", "zero"], ["Y"]),
            ],
            "input_clip",
            [_vi("X", dtype, [3])],
            [_vi("Y", dtype, [3])],
            initializer=[
                numpy_helper.from_array(np_dtype(0.25), "scale"),
                numpy_helper.from_array(np.uint8(0), "zero"),
            ],
        ),
        opset_imports=[helper.make_opsetid("", opset)],
        ir_version=_MAX_IR_VERSION_FOR_ORT,
    )
    onnx.checker.check_model(model, full_check=True)
    assert _apply(model, quant_types=["uint8"], keep_clip_after_inputs=True) == 1
    onnx.checker.check_model(model, full_check=True)
    x = np.array([-10, 1, 100], dtype=np_dtype)
    np.testing.assert_array_equal(_run_ort(model, {"X": x})["Y"], np.clip(x, 0, 63.75))


@pytest.mark.parametrize("unsupported", ["per_axis", "dtype_change", "untyped_storage"])
def test_input_clip_refusal_preserves_all_pairs(unsupported):
    """An unsupported later pair leaves earlier valid pairs and caller state unchanged."""
    shape = [2, 3]
    output_dtype = TensorProto.FLOAT16 if unsupported == "dtype_change" else TensorProto.FLOAT
    per_axis = unsupported == "per_axis"
    missing_zero = unsupported == "untyped_storage"
    scale = np.full(3, 0.25, dtype=np.float32) if per_axis else np.float32(0.25)
    zero = np.zeros(3, dtype=np.uint8) if per_axis else np.uint8(0)
    second_inputs = ["W", "second_scale"] + ([] if missing_zero else ["second_zero"])
    options = {"axis": 1} if per_axis else {}
    q_options = {"output_dtype": TensorProto.UINT16} if missing_zero else options
    dq_scale = "half_scale" if unsupported == "dtype_change" else "second_scale"
    model = helper.make_model(
        helper.make_graph(
            [
                helper.make_node("QuantizeLinear", ["X", "scale", "zero"], ["first_q"]),
                helper.make_node("DequantizeLinear", ["first_q", "scale", "zero"], ["Y"]),
                helper.make_node("QuantizeLinear", second_inputs, ["second_q"], **q_options),
                helper.make_node(
                    "DequantizeLinear", ["second_q", dq_scale, *second_inputs[2:]], ["Z"], **options
                ),
            ],
            "clip_refusal",
            [_vi(name, TensorProto.FLOAT, shape) for name in ("X", "W")],
            [_vi("Y", TensorProto.FLOAT, shape), _vi("Z", output_dtype, shape)],
            initializer=[
                numpy_helper.from_array(np.float32(0.25), "scale"),
                numpy_helper.from_array(np.uint8(0), "zero"),
                numpy_helper.from_array(scale, "second_scale"),
                *([] if missing_zero else [numpy_helper.from_array(zero, "second_zero")]),
                *(
                    [numpy_helper.from_array(np.float16(0.25), "half_scale")]
                    if unsupported == "dtype_change"
                    else []
                ),
            ],
        ),
        opset_imports=[helper.make_opsetid("", 21)],
        ir_version=_MAX_IR_VERSION_FOR_ORT,
    )
    onnx.checker.check_model(model, full_check=True)
    before = model.SerializeToString()
    with pytest.raises((NotImplementedError, ValueError)):
        _apply(model, quant_types=["uint8", "uint16"], keep_clip_after_inputs=True)
    assert model.SerializeToString() == before
