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
_SAVED_MODELS_DIR = _REPO_ROOT / "scratch_space" / "remove_deqlin_test_models"
_OPSET_VERSION = 21
_MAX_IR_VERSION_FOR_ORT = 10


# ── Dynamic import ────────────────────────────────────────────────────────────


def _load_module():
    mod_key = "modelopt.onnx.graph_surgery.dla_transforms.dla_remove_deqlin"
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
        _REPO_ROOT / "modelopt/onnx/graph_surgery/dla_transforms/dla_remove_deqlin.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_key] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


_mod = _load_module()
_apply = _mod._apply_remove_deqlin


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


def _has_dq(model: onnx.ModelProto) -> bool:
    return any(n.op_type == "DequantizeLinear" for n in model.graph.node)


def _init_names(model: onnx.ModelProto) -> set[str]:
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
    return helper.make_node(
        "DequantizeLinear",
        inputs=[f"{name}_x", f"{name}_scale", f"{name}_zp"],
        outputs=[output_name],
        name=f"dq_{name}",
    )


# ── 1. uint16 DQ on initializer → dequantized, ORT parity ────────────────────


def test_uint16_dq_init_dequantized():
    """DQ(uint16 init) feeding Relu → folded into float32 init; ORT parity."""
    shape = [4, 8]
    scale_val = np.float32(0.01)
    zp_val = np.uint16(0)
    x_data = _RNG.integers(0, 65535, shape, dtype=np.int64).astype(np.uint16)

    x_init, scale_init, zp_init = _make_dq_init("w", x_data, np.array(scale_val), np.array(zp_val))
    dq_node = _make_dq_node("w", "w_float")
    relu = helper.make_node("Relu", ["X_add"], ["Y"], name="relu")
    add = helper.make_node("Add", ["X", "w_float"], ["X_add"], name="add")

    graph = helper.make_graph(
        [dq_node, add, relu],
        "u16_dq",
        [_vi("X", TensorProto.FLOAT, shape)],
        [_vi("Y", TensorProto.FLOAT, shape)],
        initializer=[x_init, scale_init, zp_init],
        value_info=[
            _vi("w_float", TensorProto.FLOAT, shape),
            _vi("X_add", TensorProto.FLOAT, shape),
        ],
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
    _save(model, tr, "u16_dq")

    assert not _has_dq(tr), "DQ node must be removed"
    assert "w_x_dequantized" in _init_names(tr), "float32 initializer must be added"
    # original quantised init is left for _remove_unused_initializers to clean up
    np.testing.assert_allclose(ref["Y"], got["Y"], rtol=1e-4, atol=1e-4)


# ── 2. int8 DQ feeding non-Conv op → dequantized ─────────────────────────────


def test_int8_dq_non_conv_dequantized():
    """DQ(int8) feeding a Mul (not Conv) → folded into float32."""
    shape = [2, 4]
    x_data = _RNG.integers(-127, 127, shape, dtype=np.int8)
    scale_val = np.array(0.1, dtype=np.float32)
    zp_val = np.array(0, dtype=np.int8)

    x_init, scale_init, zp_init = _make_dq_init("b", x_data, scale_val, zp_val)
    dq_node = _make_dq_node("b", "b_float")
    mul = helper.make_node("Mul", ["X", "b_float"], ["Y"], name="mul")

    graph = helper.make_graph(
        [dq_node, mul],
        "int8_dq_mul",
        [_vi("X", TensorProto.FLOAT, shape)],
        [_vi("Y", TensorProto.FLOAT, shape)],
        initializer=[x_init, scale_init, zp_init],
        value_info=[_vi("b_float", TensorProto.FLOAT, shape)],
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
    _save(model, tr, "int8_dq_mul")

    assert not _has_dq(tr)
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


def test_dq_multiple_consumers_rewired():
    """DQ folded and BOTH downstream consumers rewired to the new float init."""
    shape = [2, 4]
    x_data = _RNG.integers(0, 255, shape, dtype=np.uint16)
    scale_val = np.array(0.01, dtype=np.float32)
    zp_val = np.array(0, dtype=np.uint16)

    x_init, scale_init, zp_init = _make_dq_init("c", x_data, scale_val, zp_val)
    dq_node = _make_dq_node("c", "c_float")
    add1 = helper.make_node("Add", ["X", "c_float"], ["add1_out"], name="add1")
    add2 = helper.make_node("Add", ["add1_out", "c_float"], ["Y"], name="add2")

    graph = helper.make_graph(
        [dq_node, add1, add2],
        "dq_multi_consumer",
        [_vi("X", TensorProto.FLOAT, shape)],
        [_vi("Y", TensorProto.FLOAT, shape)],
        initializer=[x_init, scale_init, zp_init],
        value_info=[
            _vi("c_float", TensorProto.FLOAT, shape),
            _vi("add1_out", TensorProto.FLOAT, shape),
        ],
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
    _save(model, tr, "dq_multi_consumer")

    assert not _has_dq(tr), "DQ must be removed"
    # Both add1 and add2 must reference the new float init
    new_name = "c_x_dequantized"
    for node in tr.graph.node:
        if node.name in ("add1", "add2"):
            assert new_name in node.input, f"{node.name} was not rewired to {new_name!r}"
    np.testing.assert_allclose(ref["Y"], got["Y"], rtol=1e-4, atol=1e-4)


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
