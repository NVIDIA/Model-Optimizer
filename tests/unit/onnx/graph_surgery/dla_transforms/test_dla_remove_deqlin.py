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
# Unit tests for :mod:`modelopt.onnx.graph_surgery.dla_transforms.dla_remove_deqlin`.

"""ORT CPU parity tests for ``dla_remove_deqlin``.

Transform rule
--------------
A ``DequantizeLinear`` node whose first input (data) is a graph initializer is
replaced by a pre-computed float32 initializer:

  ``DequantizeLinear(x_init, scale, zp)``  →  ``x_init_dequantized  (float32)``

The original quantised initializer and the DQ node are removed.

Skip conditions (DQ node kept)
-------------------------------
* Conv / ConvTranspose weight input with a recognised quantised zero-point type.
* MatMul / Gemm weight that ``check_to_apply_transpose`` flags for conversion.
* DQ whose data input is NOT a static initializer (e.g. a graph input).

Coverage
--------
uint16 DQ on init → dequantized, ORT parity
int8   DQ on init feeding non-Conv → dequantized
DQ → Conv weight INT8 zp          → kept (skipped)
DQ → Conv weight UINT8 zp         → kept
DQ → ConvTranspose weight INT8 zp → kept
DQ → MatMul weight (will convert)  → kept
DQ data is a graph input           → never removed
DQ with two consumers              → dequantized, both consumers rewired
No DQ nodes                        → graph unchanged, ORT parity
"""

from __future__ import annotations

import copy

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper
from onnx.reference import ReferenceEvaluator

from modelopt.onnx.graph_surgery.dla_transforms import dla_remove_deqlin as _mod

ort = pytest.importorskip("onnxruntime", reason="onnxruntime required for ORT parity checks")

_OPSET_VERSION = 21
_MAX_IR_VERSION_FOR_ORT = 10


_apply = _mod._apply_remove_deqlin


# ── Shared utilities ──────────────────────────────────────────────────────────


def _vi(name: str, elem_type: int, shape: list) -> onnx.ValueInfoProto:
    """Create typed tensor metadata."""
    return helper.make_tensor_value_info(name, elem_type, shape)


def _clamp_ir(model: onnx.ModelProto) -> None:
    """Use an IR version supported by the test runtime."""
    model.ir_version = min(model.ir_version, _MAX_IR_VERSION_FOR_ORT)


def _prepare_model(model: onnx.ModelProto) -> None:
    """Infer intermediate metadata when the optional runtime supports the graph."""
    try:
        # ONNX Runtime is optional for these execution tests.
        from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference

        inferred = SymbolicShapeInference.infer_shapes(model)
        if inferred is not None:
            del model.graph.value_info[:]
            model.graph.value_info.extend(inferred.graph.value_info)
    except Exception:
        pass


def _run_ort(model: onnx.ModelProto, feeds: dict) -> dict:
    """Execute the graph outputs with the CPU runtime."""
    _clamp_ir(model)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"], sess_options=options
    )
    return dict(zip([output.name for output in model.graph.output], sess.run(None, feeds)))


def _has_dq(model: onnx.ModelProto) -> bool:
    """Check whether the graph retains a dequantization node."""
    return any(n.op_type == "DequantizeLinear" for n in model.graph.node)


def _init_names(model: onnx.ModelProto) -> set[str]:
    """Collect graph initializer names."""
    return {i.name for i in model.graph.initializer}


_RNG = np.random.default_rng(2024)


# ── Helpers to build DQ sub-graphs ───────────────────────────────────────────


def _make_dq_init(
    name: str,
    data: np.ndarray,
    scale: np.ndarray,
    zp: np.ndarray,
) -> tuple[onnx.TensorProto, onnx.TensorProto, onnx.TensorProto]:
    """Return (data_init, scale_init, zp_init) ONNX initializers."""
    return (
        numpy_helper.from_array(data, name=f"{name}_x"),
        numpy_helper.from_array(scale.astype(np.float32), name=f"{name}_scale"),
        numpy_helper.from_array(zp, name=f"{name}_zp"),
    )


def _make_dq_node(name: str, output_name: str) -> onnx.NodeProto:
    """Build a dequantization node using the named controls."""
    return helper.make_node(
        "DequantizeLinear",
        inputs=[f"{name}_x", f"{name}_scale", f"{name}_zp"],
        outputs=[output_name],
        name=f"dq_{name}",
    )


def test_dq_without_zero_point_dequantized():
    """DQ(data, scale) with omitted zero_point uses zero and folds into float32."""
    shape = [2, 4]
    x_data = _RNG.integers(-127, 127, shape, dtype=np.int8)
    scale_val = np.array(0.2, dtype=np.float32)

    x_init = numpy_helper.from_array(x_data, name="no_zp_x")
    scale_init = numpy_helper.from_array(scale_val, name="no_zp_scale")
    dq_node = helper.make_node(
        "DequantizeLinear",
        inputs=["no_zp_x", "no_zp_scale"],
        outputs=["no_zp_float"],
        name="dq_no_zp",
    )
    add = helper.make_node("Add", ["X", "no_zp_float"], ["Y"], name="add")

    graph = helper.make_graph(
        [dq_node, add],
        "dq_no_zero_point",
        [_vi("X", TensorProto.FLOAT, shape)],
        [_vi("Y", TensorProto.FLOAT, shape)],
        initializer=[x_init, scale_init],
        value_info=[_vi("no_zp_float", TensorProto.FLOAT, shape)],
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

    assert not _has_dq(tr)
    assert "no_zp_x_dequantized" in _init_names(tr)
    expected_init = next(i for i in tr.graph.initializer if i.name == "no_zp_x_dequantized")
    np.testing.assert_allclose(
        numpy_helper.to_array(expected_init), x_data.astype(np.float32) * scale_val
    )
    np.testing.assert_allclose(ref["Y"], got["Y"], rtol=1e-4, atol=1e-4)


# ── 3. DQ → Conv weight INT8 zp: SKIPPED ─────────────────────────────────────


def test_dq_conv_weight_int8_skipped():
    """DQ(int8 zp) feeding Conv input[1] must be kept."""
    in_c, out_c, k = 4, 8, 3
    x_data = _RNG.integers(-127, 127, [out_c, in_c, k, k], dtype=np.int8)
    scale_val = np.ones(out_c, dtype=np.float32) * 0.1
    zp_val = np.zeros(out_c, dtype=np.int8)

    x_init, scale_init, zp_init = _make_dq_init("w", x_data, scale_val, zp_val)
    dq_node = helper.make_node(
        "DequantizeLinear",
        inputs=["w_x", "w_scale", "w_zp"],
        outputs=["w_float"],
        name="dq_w",
        axis=0,
    )
    conv = helper.make_node("Conv", ["X", "w_float"], ["Y"], name="conv", pads=[1, 1, 1, 1])

    graph = helper.make_graph(
        [dq_node, conv],
        "dq_conv_skip",
        [_vi("X", TensorProto.FLOAT, [1, in_c, 8, 8])],
        [_vi("Y", TensorProto.FLOAT, [1, out_c, 8, 8])],
        initializer=[x_init, scale_init, zp_init],
        value_info=[_vi("w_float", TensorProto.FLOAT, [out_c, in_c, k, k])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", _OPSET_VERSION)])
    _clamp_ir(model)

    tr = copy.deepcopy(model)
    _prepare_model(tr)
    tr = _apply(tr)

    assert _has_dq(tr), "DQ node must be kept for Conv INT8 weight"


# ── 4. DQ → Conv weight UINT8 zp: SKIPPED ────────────────────────────────────


def test_dq_conv_weight_uint8_skipped():
    """DQ(uint8 zp) feeding Conv input[1] must be kept."""
    in_c, out_c, k = 4, 8, 3
    x_data = _RNG.integers(0, 255, [out_c, in_c, k, k], dtype=np.uint8)
    scale_val = np.ones(out_c, dtype=np.float32) * 0.1
    zp_val = np.full(out_c, 128, dtype=np.uint8)

    x_init, scale_init, zp_init = _make_dq_init("w", x_data, scale_val, zp_val)
    dq_node = helper.make_node(
        "DequantizeLinear",
        inputs=["w_x", "w_scale", "w_zp"],
        outputs=["w_float"],
        name="dq_w",
        axis=0,
    )
    conv = helper.make_node("Conv", ["X", "w_float"], ["Y"], name="conv", pads=[1, 1, 1, 1])

    graph = helper.make_graph(
        [dq_node, conv],
        "dq_conv_u8_skip",
        [_vi("X", TensorProto.FLOAT, [1, in_c, 8, 8])],
        [_vi("Y", TensorProto.FLOAT, [1, out_c, 8, 8])],
        initializer=[x_init, scale_init, zp_init],
        value_info=[_vi("w_float", TensorProto.FLOAT, [out_c, in_c, k, k])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", _OPSET_VERSION)])
    _clamp_ir(model)

    tr = copy.deepcopy(model)
    _prepare_model(tr)
    tr = _apply(tr)

    assert _has_dq(tr), "DQ node must be kept for Conv UINT8 weight"


# ── 5. DQ → ConvTranspose weight INT8 zp: SKIPPED ────────────────────────────


def test_dq_convtranspose_weight_int8_skipped():
    """DQ(int8 zp) feeding ConvTranspose input[1] must be kept."""
    in_c, out_c, k = 8, 4, 3
    x_data = _RNG.integers(-127, 127, [in_c, out_c, k, k], dtype=np.int8)
    scale_val = np.ones(out_c, dtype=np.float32) * 0.05
    zp_val = np.zeros(out_c, dtype=np.int8)

    x_init, scale_init, zp_init = _make_dq_init("w", x_data, scale_val, zp_val)
    dq_node = helper.make_node(
        "DequantizeLinear",
        inputs=["w_x", "w_scale", "w_zp"],
        outputs=["w_float"],
        name="dq_w",
        axis=1,
    )
    convt = helper.make_node(
        "ConvTranspose", ["X", "w_float"], ["Y"], name="convt", pads=[1, 1, 1, 1]
    )

    graph = helper.make_graph(
        [dq_node, convt],
        "dq_convt_skip",
        [_vi("X", TensorProto.FLOAT, [1, in_c, 6, 6])],
        [_vi("Y", TensorProto.FLOAT, [1, out_c, 6, 6])],
        initializer=[x_init, scale_init, zp_init],
        value_info=[_vi("w_float", TensorProto.FLOAT, [in_c, out_c, k, k])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", _OPSET_VERSION)])
    _clamp_ir(model)

    tr = copy.deepcopy(model)
    _prepare_model(tr)
    tr = _apply(tr)

    assert _has_dq(tr), "DQ node must be kept for ConvTranspose INT8 weight"


# ── 6. DQ → MatMul weight (will convert): SKIPPED ────────────────────────────


def test_dq_matmul_weight_will_convert_skipped():
    """DQ(int8) feeding MatMul input[1] where check_to_apply_transpose=True → DQ kept.

    Weight shape [K, N] (2D) and activation [M, K] (2D) → check_to_apply_transpose
    returns True, meaning dla_matmul_to_transpose_conv_transpose will handle this node.
    """
    m, k, n = 4, 8, 16
    w_data = _RNG.integers(-127, 127, [k, n], dtype=np.int8)
    scale_val = np.ones(n, dtype=np.float32) * 0.05
    zp_val = np.zeros(n, dtype=np.int8)

    x_init, scale_init, zp_init = _make_dq_init("w", w_data, scale_val, zp_val)
    dq_node = helper.make_node(
        "DequantizeLinear",
        inputs=["w_x", "w_scale", "w_zp"],
        outputs=["w_float"],
        name="dq_w",
        axis=0,
    )
    matmul = helper.make_node("MatMul", ["X", "w_float"], ["Y"], name="matmul")

    graph = helper.make_graph(
        [dq_node, matmul],
        "dq_matmul_skip",
        [_vi("X", TensorProto.FLOAT, [m, k])],
        [_vi("Y", TensorProto.FLOAT, [m, n])],
        initializer=[x_init, scale_init, zp_init],
        value_info=[_vi("w_float", TensorProto.FLOAT, [k, n])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", _OPSET_VERSION)])
    _clamp_ir(model)

    tr = copy.deepcopy(model)
    _prepare_model(tr)
    tr = _apply(tr)

    assert _has_dq(tr), "DQ node must be kept for MatMul that will be converted to ConvTranspose"


# ── 7. DQ data is a graph input → never removed ───────────────────────────────


def test_dq_dynamic_input_not_removed():
    """DQ whose input[0] is a graph input (not an initializer) must be left in place."""
    shape = [4, 8]
    scale_init = numpy_helper.from_array(np.array(0.1, dtype=np.float32), name="scale")
    zp_init = numpy_helper.from_array(np.array(0, dtype=np.uint16), name="zp")

    dq_node = helper.make_node(
        "DequantizeLinear",
        inputs=["X_quant", "scale", "zp"],
        outputs=["Y"],
        name="dq",
    )
    graph = helper.make_graph(
        [dq_node],
        "dq_dynamic",
        [_vi("X_quant", TensorProto.UINT16, shape)],
        [_vi("Y", TensorProto.FLOAT, shape)],
        initializer=[scale_init, zp_init],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", _OPSET_VERSION)])
    _clamp_ir(model)
    original_node_count = len(model.graph.node)

    tr = copy.deepcopy(model)
    _prepare_model(tr)
    tr = _apply(tr)

    assert len(tr.graph.node) == original_node_count, "Graph must be unchanged"
    assert _has_dq(tr), "DQ node must remain when data is a graph input"


# ── 8. DQ with two consumers → both rewired ───────────────────────────────────


@pytest.mark.parametrize(
    ("dtype", "operation", "public_output"),
    [(np.uint16, "Add", False), (np.uint16, "Add", True), (np.int8, "Mul", False)],
)
def test_dq_multiple_consumers_rewired(dtype, operation, public_output):
    """Preserve shared consumers, public output names and unrelated Constant ownership."""
    shape = [2, 4]
    bounds = np.iinfo(dtype)
    data = _RNG.integers(bounds.min, bounds.max, shape, dtype=dtype)
    initializers = _make_dq_init("c", data, np.array(0.01, np.float32), np.array(0, dtype))
    outputs = [_vi("Y", TensorProto.FLOAT, shape)]
    if public_output:
        outputs.append(_vi("c_float", TensorProto.FLOAT, shape))
    nodes = [
        helper.make_node(
            "Constant",
            [],
            ["retained"],
            name="retained",
            value=numpy_helper.from_array(np.arange(16, dtype=np.uint8)),
        ),
        _make_dq_node("c", "c_float"),
        helper.make_node(operation, ["X", "c_float"], ["first"], name="first"),
        helper.make_node(operation, ["first", "c_float"], ["result"], name="second"),
        helper.make_node(
            "Relu" if operation == "Add" else "Identity", ["result"], ["Y"], name="tail"
        ),
    ]
    model = helper.make_model(
        helper.make_graph(
            nodes,
            "dq_multi_consumer",
            [_vi("X", TensorProto.FLOAT, shape)],
            outputs,
            initializer=initializers,
        ),
        opset_imports=[helper.make_opsetid("", _OPSET_VERSION)],
        ir_version=9,
    )
    feeds = {"X": _RNG.standard_normal(shape).astype(np.float32)}
    onnx.checker.check_model(model, full_check=True)
    expected = ReferenceEvaluator(model).run(None, feeds)
    runtime_expected = _run_ort(model, feeds)
    transformed = copy.deepcopy(model)
    retained = {
        node.name: node for node in transformed.graph.node if node.op_type != "DequantizeLinear"
    }
    payload = retained["retained"].attribute[0].t
    payload_bytes = payload.SerializeToString()
    _apply(transformed)
    onnx.checker.check_model(transformed, full_check=True)
    assert not _has_dq(transformed)
    for node in transformed.graph.node:
        if node.name in retained:
            assert node is retained[node.name]
        if node.name in {"first", "second"}:
            assert "c_x_dequantized" in node.input
    assert retained["retained"].attribute[0].t is payload
    assert payload.SerializeToString() == payload_bytes
    if public_output:
        identity = next(node for node in transformed.graph.node if "c_float" in node.output)
        assert identity.op_type == "Identity"
        assert list(identity.input) == ["c_x_dequantized"]
    for actual, reference in zip(ReferenceEvaluator(transformed).run(None, feeds), expected):
        np.testing.assert_array_equal(actual, reference)
    for name, actual in _run_ort(transformed, feeds).items():
        np.testing.assert_array_equal(actual, runtime_expected[name])


# ── 9. No DQ nodes: graph unchanged ──────────────────────────────────────────


def test_no_dq_nodes_noop():
    """Graph with no DQ nodes must be left completely unchanged."""
    shape = [2, 4]
    w_init = numpy_helper.from_array(_RNG.standard_normal(shape).astype(np.float32), name="W")
    add = helper.make_node("Add", ["X", "W"], ["Y"], name="add")

    graph = helper.make_graph(
        [add],
        "no_dq",
        [_vi("X", TensorProto.FLOAT, shape)],
        [_vi("Y", TensorProto.FLOAT, shape)],
        initializer=[w_init],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", _OPSET_VERSION)])
    _clamp_ir(model)
    original_count = len(model.graph.node)

    x = _RNG.standard_normal(shape).astype(np.float32)
    _prepare_model(model)
    ref = _run_ort(model, {"X": x})

    tr = copy.deepcopy(model)
    _prepare_model(tr)
    tr = _apply(tr)
    _prepare_model(tr)
    got = _run_ort(tr, {"X": x})

    assert len(tr.graph.node) == original_count
    np.testing.assert_array_equal(ref["Y"], got["Y"])


@pytest.mark.parametrize(
    "case",
    [
        "default-axis",
        "scalar-rank-one",
        "scalar-rank-zero",
        "runtime-zero-point",
        "overridable-data",
        "overridable-scale",
        "overridable-zero-point",
        "fp16-scale",
    ],
)
def test_constant_folding_preserves_runtime_and_type_contracts(case):
    """Fold only supported fixed controls, retaining runtime overrides and output types."""
    data = np.asarray([[2, 4, 6], [8, 10, 12]], dtype=np.uint8)
    if case == "scalar-rank-zero":
        data = np.asarray(5, np.uint8)
    if case == "scalar-rank-one":
        data = data.reshape(-1)
    scale = np.asarray(
        [0.25, 0.5, 1.0] if case == "default-axis" else 0.5,
        dtype=np.float16 if case == "fp16-scale" else np.float32,
    )
    if case == "scalar-rank-zero":
        scale = np.asarray([0.5], np.float32)
    initializers = [
        numpy_helper.from_array(data, name="data"),
        numpy_helper.from_array(scale, name="scale"),
    ]
    inputs, feeds = [], {}
    node_inputs = ["data", "scale"]
    if "zero-point" in case:
        node_inputs.append("zero")
        inputs.append(_vi("zero", TensorProto.UINT8, []))
        feeds["zero"] = np.asarray(2, np.uint8)
        if case == "overridable-zero-point":
            initializers.append(numpy_helper.from_array(np.asarray(0, np.uint8), name="zero"))
    for name, value in [("data", data), ("scale", scale)]:
        if case == f"overridable-{name}":
            inputs.append(
                _vi(
                    name,
                    TensorProto.UINT8 if name == "data" else TensorProto.FLOAT,
                    list(value.shape),
                )
            )
            feeds[name] = value + np.asarray(2, dtype=value.dtype)
    out_type = TensorProto.FLOAT16 if case == "fp16-scale" else TensorProto.FLOAT
    model = helper.make_model(
        helper.make_graph(
            [helper.make_node("DequantizeLinear", node_inputs, ["Y"], name="dq")],
            case,
            inputs,
            [_vi("Y", out_type, list(data.shape))],
            initializer=initializers,
        ),
        opset_imports=[helper.make_opsetid("", 21)],
    )
    onnx.checker.check_model(model, full_check=True)
    expected = ReferenceEvaluator(model).run(None, feeds)[0]
    before = model.SerializeToString()
    _apply(model)
    onnx.checker.check_model(model, full_check=True)
    actual = ReferenceEvaluator(model).run(None, feeds)[0]
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == expected.dtype
    if case in {"default-axis", "scalar-rank-one", "scalar-rank-zero"}:
        assert not _has_dq(model)
    else:
        assert model.SerializeToString() == before


@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("omit_zero", [False, True])
def test_quantized_conv_weight_retained_for_all_consumers(shared, omit_zero):
    """Retain a required quantized weight with shared uses or an omitted zero point."""
    initializers = [
        numpy_helper.from_array(np.ones([1, 1, 1, 1], dtype=np.uint8), name="weight"),
        numpy_helper.from_array(np.asarray(0.25, np.float32), name="scale"),
    ]
    inputs = ["weight", "scale"]
    if not omit_zero:
        inputs.append("zero")
        initializers.append(numpy_helper.from_array(np.asarray(0, np.uint8), name="zero"))
    nodes = [
        helper.make_node("DequantizeLinear", inputs, ["dequantized"], name="dq"),
        helper.make_node("Conv", ["X", "dequantized"], ["Y"], name="conv"),
    ]
    outputs = [_vi("Y", TensorProto.FLOAT, [1, 1, 1, 1])]
    if shared:
        nodes.append(helper.make_node("Identity", ["dequantized"], ["shared"], name="shared"))
        outputs.append(_vi("shared", TensorProto.FLOAT, [1, 1, 1, 1]))
    model = helper.make_model(
        helper.make_graph(
            nodes,
            "quantized_weight",
            [_vi("X", TensorProto.FLOAT, [1, 1, 1, 1])],
            outputs,
            initializer=initializers,
        ),
        opset_imports=[helper.make_opsetid("", 21)],
    )
    onnx.checker.check_model(model, full_check=True)
    feeds = {"X": np.asarray([[[[2.0]]]], np.float32)}
    expected = ReferenceEvaluator(model).run(None, feeds)
    before = model.SerializeToString()
    _apply(model)
    assert model.SerializeToString() == before
    for actual, reference in zip(ReferenceEvaluator(model).run(None, feeds), expected):
        np.testing.assert_array_equal(actual, reference)


@pytest.mark.parametrize("overridable", [False, True])
def test_reused_float_initializer_preserves_override_contract(overridable):
    """Reuse equal fixed weights while avoiding initializer defaults callers may override."""
    data = np.asarray([[2, 4, 6]], np.uint8)
    expected = data.astype(np.float32) * np.float32(0.5)
    initializers = [
        numpy_helper.from_array(data, name="data"),
        numpy_helper.from_array(np.float32(0.5), name="scale"),
        numpy_helper.from_array(expected, name="data_dequantized"),
    ]
    inputs = [_vi("data_dequantized", TensorProto.FLOAT, [1, 3])] if overridable else []
    model = helper.make_model(
        helper.make_graph(
            [helper.make_node("DequantizeLinear", ["data", "scale"], ["Y"], name="dq")],
            "reused",
            inputs,
            [_vi("Y", TensorProto.FLOAT, [1, 3])],
            initializer=initializers,
        ),
        opset_imports=[helper.make_opsetid("", 21)],
    )
    onnx.checker.check_model(model, full_check=True)
    feeds = {"data_dequantized": expected + 10} if overridable else {}
    reference = ReferenceEvaluator(model).run(None, feeds)
    _apply(model)
    onnx.checker.check_model(model, full_check=True)
    assert len(model.graph.initializer) == (4 if overridable else 3)
    np.testing.assert_array_equal(ReferenceEvaluator(model).run(None, feeds)[0], reference[0])


@pytest.mark.parametrize("domain", ["custom", "com.microsoft"])
def test_custom_dequantization_function_remains_unchanged(domain):
    """Preserve custom functions that share a built-in operation name."""
    function = helper.make_function(
        domain,
        "DequantizeLinear",
        ["data", "scale"],
        ["Y"],
        [
            helper.make_node("Cast", ["data"], ["cast"], to=TensorProto.FLOAT),
            helper.make_node("Neg", ["cast"], ["Y"]),
        ],
        opset_imports=[helper.make_opsetid("", 21)],
    )
    model = helper.make_model(
        helper.make_graph(
            [
                helper.make_node(
                    "DequantizeLinear", ["data", "scale"], ["Y"], domain=domain, name="custom_dq"
                )
            ],
            "custom",
            [],
            [_vi("Y", TensorProto.FLOAT, [2])],
            initializer=[
                numpy_helper.from_array(np.asarray([2, 4], np.uint8), name="data"),
                numpy_helper.from_array(np.float32(0.5), name="scale"),
            ],
        ),
        opset_imports=[helper.make_opsetid("", 21), helper.make_opsetid(domain, 1)],
        functions=[function],
    )
    onnx.checker.check_model(model, full_check=True)
    reference = ReferenceEvaluator(model).run(None, {})
    before = model.SerializeToString()
    _apply(model)
    assert model.SerializeToString() == before
    np.testing.assert_array_equal(ReferenceEvaluator(model).run(None, {})[0], reference[0])


@pytest.mark.parametrize("dtype", [np.int8, np.uint8, np.int16, np.uint16, np.int32])
@pytest.mark.parametrize("axis", [1, -1])
def test_supported_integer_axis_folding_is_exact(dtype, axis):
    """Keep exact float32 results across supported integer widths and negative axes."""
    bounds = np.iinfo(np.int16 if dtype == np.int32 else dtype)
    data = np.asarray([[bounds.min, 0, bounds.max], [1, bounds.max, bounds.min]], dtype=dtype)
    scale = np.asarray([0.1234567, 0.25, 0.75], np.float32)
    zero = (
        np.zeros(3, dtype=dtype)
        if dtype == np.int32
        else np.asarray([bounds.max, 0, 1], dtype=dtype)
    )
    model = helper.make_model(
        helper.make_graph(
            [
                helper.make_node(
                    "DequantizeLinear", ["data", "scale", "zero"], ["Y"], axis=axis, name="dq"
                )
            ],
            "axis_dtype",
            [],
            [_vi("Y", TensorProto.FLOAT, [2, 3])],
            initializer=[
                numpy_helper.from_array(data, name="data"),
                numpy_helper.from_array(scale, name="scale"),
                numpy_helper.from_array(zero, name="zero"),
            ],
        ),
        opset_imports=[helper.make_opsetid("", 21)],
    )
    onnx.checker.check_model(model, full_check=True)
    reference = ReferenceEvaluator(model).run(None, {})
    _apply(model)
    onnx.checker.check_model(model, full_check=True)
    assert not _has_dq(model)
    np.testing.assert_array_equal(ReferenceEvaluator(model).run(None, {})[0], reference[0])


def test_supported_runtime_dequantization_schema_folds():
    """Execute the supported runtime contrib schema before and after constant folding."""
    data = np.asarray([[2, 4, 6], [8, 10, 12]], np.uint16)
    model = helper.make_model(
        helper.make_graph(
            [
                helper.make_node(
                    "DequantizeLinear",
                    ["data", "scale", "zero"],
                    ["Y"],
                    name="dq",
                    domain="com.microsoft",
                )
            ],
            "runtime_dq",
            [],
            [_vi("Y", TensorProto.FLOAT, [2, 3])],
            initializer=[
                numpy_helper.from_array(data, name="data"),
                numpy_helper.from_array(np.float32(0.5), name="scale"),
                numpy_helper.from_array(np.uint16(2), name="zero"),
            ],
        ),
        opset_imports=[helper.make_opsetid("", 21), helper.make_opsetid("com.microsoft", 1)],
    )
    _clamp_ir(model)
    onnx.checker.check_model(model, full_check=True)
    expected = _run_ort(model, {})
    _apply(model)
    onnx.checker.check_model(model, full_check=True)
    assert not _has_dq(model)
    np.testing.assert_array_equal(_run_ort(model, {})["Y"], expected["Y"])


@pytest.mark.parametrize("axis", [1, -1])
def test_large_int32_folding_preserves_established_rounding(axis):
    """Retain existing INT32 rounding while evaluating intermediate arithmetic in bounded chunks."""
    data = np.asarray(
        [[16777217, -1356484993, 1129686157], [16777219, -18865023, 17258856]], np.int32
    )
    scale = np.asarray([0.1234567, 0.25, 0.0001], np.float32)
    zero = np.zeros(3, np.int32)
    model = helper.make_model(
        helper.make_graph(
            [
                helper.make_node(
                    "DequantizeLinear", ["data", "scale", "zero"], ["Y"], name="dq", axis=axis
                )
            ],
            "large_int32",
            [],
            [_vi("Y", TensorProto.FLOAT, [2, 3])],
            initializer=[
                numpy_helper.from_array(data, name="data"),
                numpy_helper.from_array(scale, name="scale"),
                numpy_helper.from_array(zero, name="zero"),
            ],
        ),
        opset_imports=[helper.make_opsetid("", 21)],
    )
    onnx.checker.check_model(model, full_check=True)
    reference = ReferenceEvaluator(model).run(None, {})[0]
    expected = ((data - zero) * scale).astype(np.float32)
    _apply(model)
    onnx.checker.check_model(model, full_check=True)
    actual = ReferenceEvaluator(model).run(None, {})[0]
    assert not _has_dq(model)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_allclose(actual, reference, rtol=1e-6, atol=0)


def test_unknown_matmul_layout_keeps_quantized_weight():
    """Keep quantized weights until later inference can establish conversion eligibility."""
    model = helper.make_model(
        helper.make_graph(
            [
                helper.make_node(
                    "DequantizeLinear", ["weight", "scale", "zero"], ["dequantized"], name="dq"
                ),
                helper.make_node("MatMul", ["X", "dequantized"], ["Y"], name="matmul"),
            ],
            "unknown_layout",
            [_vi("X", TensorProto.FLOAT, ["M", "K"])],
            [_vi("Y", TensorProto.FLOAT, ["M", 4])],
            initializer=[
                numpy_helper.from_array(np.arange(12, dtype=np.uint8).reshape(3, 4), name="weight"),
                numpy_helper.from_array(np.float32(0.5), name="scale"),
                numpy_helper.from_array(np.uint8(0), name="zero"),
            ],
        ),
        opset_imports=[helper.make_opsetid("", 21)],
    )
    onnx.checker.check_model(model, full_check=True)
    feeds = {"X": np.ones([2, 3], np.float32)}
    expected = ReferenceEvaluator(model).run(None, feeds)
    before = model.SerializeToString()
    _apply(model)
    assert model.SerializeToString() == before
    np.testing.assert_array_equal(ReferenceEvaluator(model).run(None, feeds)[0], expected[0])


@pytest.mark.parametrize("depth", [1, 2])
@pytest.mark.parametrize("direct_consumer", [False, True])
def test_dequantization_preserves_nested_captures(depth, direct_consumer):
    """Preserve DQ outputs captured by nested control flow with or without direct uses."""
    shape = [2, 3]
    branch = helper.make_graph(
        [helper.make_node("Identity", ["dequantized"], ["branch_output"])],
        "branch",
        [],
        [_vi("branch_output", TensorProto.FLOAT, shape)],
    )
    if depth == 2:
        branch = helper.make_graph(
            [
                helper.make_node(
                    "If",
                    ["condition"],
                    ["nested"],
                    then_branch=branch,
                    else_branch=copy.deepcopy(branch),
                )
            ],
            "nested_branch",
            [],
            [_vi("nested", TensorProto.FLOAT, shape)],
        )
    nodes = [
        helper.make_node("DequantizeLinear", ["data", "scale", "zero"], ["dequantized"], name="dq")
    ]
    outputs = [_vi("Y", TensorProto.FLOAT, shape)]
    if direct_consumer:
        nodes.append(helper.make_node("Identity", ["dequantized"], ["direct"], name="direct"))
        outputs.append(_vi("direct", TensorProto.FLOAT, shape))
    nodes.append(
        helper.make_node(
            "If",
            ["condition"],
            ["Y"],
            name="capture",
            then_branch=branch,
            else_branch=copy.deepcopy(branch),
        )
    )
    model = helper.make_model(
        helper.make_graph(
            nodes,
            "captured_dq",
            [_vi("condition", TensorProto.BOOL, [])],
            outputs,
            initializer=[
                numpy_helper.from_array(np.arange(6, dtype=np.uint8).reshape(shape), name="data"),
                numpy_helper.from_array(np.asarray(0.5, np.float32), name="scale"),
                numpy_helper.from_array(np.asarray(0, np.uint8), name="zero"),
            ],
        ),
        opset_imports=[helper.make_opsetid("", 21)],
    )
    _clamp_ir(model)
    onnx.checker.check_model(model, full_check=True)
    references = {
        condition: ReferenceEvaluator(model).run(None, {"condition": np.asarray(condition)})
        for condition in [False, True]
    }
    before = model.SerializeToString()
    _apply(model)
    assert model.SerializeToString() == before
    onnx.checker.check_model(model, full_check=True)
    for condition, expected in references.items():
        for actual, reference in zip(
            ReferenceEvaluator(model).run(None, {"condition": np.asarray(condition)}), expected
        ):
            np.testing.assert_array_equal(actual, reference)
