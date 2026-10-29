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
# Unit tests for :mod:`modelopt.onnx.graph_surgery.dla_transforms.dla_where`.

"""ORT CPU parity tests for ``dla_where``.

Transform rule
--------------
``Where(cond, x, y)`` is replaced by a ``Mul``-based equivalent when one of
{x, y} is an all-zero static initializer:

    Where(cond, 0, y)  →  Mul(y, 1 - float(cond))
    Where(cond, x, 0)  →  Mul(x, float(cond))

Coverage
--------
BOOL init condition, true-branch zeros   → Mul(y, 1-cond_float), ORT parity
BOOL init condition, false-branch zeros  → Mul(x, cond_float),   ORT parity
Dynamic BOOL condition, true-branch zeros  → Cast+Sub+Mul,        ORT parity
Dynamic BOOL condition, false-branch zeros → Cast+Mul,            ORT parity
Greater-produced BOOL condition, false-branch zeros: Cast+Mul, ORT parity
Neither branch all-zeros                    → graph unchanged
No Where nodes                              → noop, ORT parity
"""

from __future__ import annotations

import copy

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper
from onnx.reference import ReferenceEvaluator

from modelopt.onnx.graph_surgery.dla_transforms import dla_where as _mod
from modelopt.onnx.graph_surgery.make_dla_compatible import _transform_make_dla_compatible

ort = pytest.importorskip("onnxruntime", reason="onnxruntime required for ORT parity checks")

_OPSET_VERSION = 17
_MAX_IR_VERSION_FOR_ORT = 9


_apply = _mod._apply_where


# ── Shared utilities ──────────────────────────────────────────────────────────


def _vi(name: str, elem_type: int, shape: list) -> onnx.ValueInfoProto:
    """Declare a typed tensor shape."""
    return helper.make_tensor_value_info(name, elem_type, shape)


def _clamp_ir(model: onnx.ModelProto) -> None:
    """Select an IR version supported by the CPU runtime."""
    model.ir_version = min(model.ir_version, _MAX_IR_VERSION_FOR_ORT)


def _prepare_model(model: onnx.ModelProto) -> None:
    """Populate metadata using optional runtime symbolic inference."""
    try:
        # Optional ONNX Runtime tooling is absent in some CPU installations.
        from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference
    except ImportError:
        return
    inferred = SymbolicShapeInference.infer_shapes(model, auto_merge=True)
    if inferred is not None:
        del model.graph.value_info[:]
        model.graph.value_info.extend(inferred.graph.value_info)


def _run_ort(model: onnx.ModelProto, feeds: dict) -> dict:
    """Fully check and execute all public outputs in one CPU session."""
    _clamp_ir(model)
    onnx.checker.check_model(model, full_check=True)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    sess = ort.InferenceSession(
        model.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )
    return dict(zip([o.name for o in model.graph.output], sess.run(None, feeds)))


_RNG = np.random.default_rng(2024)
_SHAPE = [2, 3]


# ── 1. Static BOOL condition, true-branch (x) is zeros ───────────────────────


def test_where_bool_init_cond_x_zeros():
    """Where(bool_init, zeros, y) → Mul(y, 1-cond_float); ORT parity.

    Where(cond, 0, y) selects y when condition is False, so the mask is (1 - cond).
    """
    cond_arr = np.array([[True, False, True], [False, True, False]])
    zeros_arr = np.zeros(_SHAPE, dtype=np.float32)
    y_arr = _RNG.standard_normal(_SHAPE).astype(np.float32)

    cond_init = numpy_helper.from_array(cond_arr, name="cond")
    x_init = numpy_helper.from_array(zeros_arr, name="x_zeros")

    where = helper.make_node("Where", ["cond", "x_zeros", "Y_in"], ["Z"], name="where")
    graph = helper.make_graph(
        [where],
        "where_bool_init_x0",
        [_vi("Y_in", TensorProto.FLOAT, _SHAPE)],
        [_vi("Z", TensorProto.FLOAT, _SHAPE)],
        initializer=[cond_init, x_init],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", _OPSET_VERSION)])
    _clamp_ir(model)

    _prepare_model(model)
    ref = _run_ort(model, {"Y_in": y_arr})

    tr = copy.deepcopy(model)
    _prepare_model(tr)
    tr = _apply(tr)
    _prepare_model(tr)
    got = _run_ort(tr, {"Y_in": y_arr})

    assert not any(n.op_type == "Where" for n in tr.graph.node), "Where must be removed"
    np.testing.assert_allclose(ref["Z"], got["Z"], rtol=1e-5, atol=1e-6)


# ── 2. Static BOOL condition, false-branch (y) is zeros ──────────────────────


def test_where_bool_init_cond_y_zeros():
    """Where(bool_init, x, zeros) → Mul(x, cond_float); ORT parity.

    Where(cond, x, 0) selects x when condition is True, so the mask is cond itself.
    """
    cond_arr = np.array([[True, False, True], [False, True, False]])
    zeros_arr = np.zeros(_SHAPE, dtype=np.float32)
    x_arr = _RNG.standard_normal(_SHAPE).astype(np.float32)

    cond_init = numpy_helper.from_array(cond_arr, name="cond2")
    y_init = numpy_helper.from_array(zeros_arr, name="y_zeros")

    where = helper.make_node("Where", ["cond2", "X_in", "y_zeros"], ["Z"], name="where")
    graph = helper.make_graph(
        [where],
        "where_bool_init_y0",
        [_vi("X_in", TensorProto.FLOAT, _SHAPE)],
        [_vi("Z", TensorProto.FLOAT, _SHAPE)],
        initializer=[cond_init, y_init],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", _OPSET_VERSION)])
    _clamp_ir(model)

    _prepare_model(model)
    ref = _run_ort(model, {"X_in": x_arr})

    tr = copy.deepcopy(model)
    _prepare_model(tr)
    tr = _apply(tr)
    _prepare_model(tr)
    got = _run_ort(tr, {"X_in": x_arr})

    assert not any(n.op_type == "Where" for n in tr.graph.node), "Where must be removed"
    np.testing.assert_allclose(ref["Z"], got["Z"], rtol=1e-5, atol=1e-6)


# ── 3. Dynamic BOOL condition, true-branch (x) is zeros ──────────────────────


def test_where_dynamic_bool_cond_x_zeros():
    """Where(bool_graph_input, zeros_init, y) → Cast+Sub+Mul; ORT parity."""
    zeros_arr = np.zeros(_SHAPE, dtype=np.float32)
    x_init = numpy_helper.from_array(zeros_arr, name="x_zeros_dyn")

    where = helper.make_node("Where", ["cond", "x_zeros_dyn", "Y_in"], ["Z"], name="where")
    graph = helper.make_graph(
        [where],
        "where_dyn_bool_x0",
        [
            _vi("cond", TensorProto.BOOL, _SHAPE),
            _vi("Y_in", TensorProto.FLOAT, _SHAPE),
        ],
        [_vi("Z", TensorProto.FLOAT, _SHAPE)],
        initializer=[x_init],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", _OPSET_VERSION)])
    _clamp_ir(model)

    cond_val = _RNG.integers(0, 2, _SHAPE).astype(bool)
    y_arr = _RNG.standard_normal(_SHAPE).astype(np.float32)
    _prepare_model(model)
    ref = _run_ort(model, {"cond": cond_val, "Y_in": y_arr})

    tr = copy.deepcopy(model)
    _prepare_model(tr)
    tr = _apply(tr)
    _prepare_model(tr)
    got = _run_ort(tr, {"cond": cond_val, "Y_in": y_arr})

    assert not any(n.op_type == "Where" for n in tr.graph.node), "Where must be removed"
    # Cast + Sub + Mul should be present
    op_types = {n.op_type for n in tr.graph.node}
    assert "Cast" in op_types, "Cast node expected for dynamic BOOL condition"
    assert "Sub" in op_types, "Sub node expected for x-is-zero inversion"
    assert "Mul" in op_types
    np.testing.assert_allclose(ref["Z"], got["Z"], rtol=1e-5, atol=1e-6)


# ── 4. Dynamic BOOL condition, false-branch (y) is zeros ─────────────────────


def test_where_dynamic_bool_cond_y_zeros():
    """Where(bool_graph_input, x, zeros_init) → Cast+Mul; ORT parity."""
    zeros_arr = np.zeros(_SHAPE, dtype=np.float32)
    y_init = numpy_helper.from_array(zeros_arr, name="y_zeros_dyn")

    where = helper.make_node("Where", ["cond", "X_in", "y_zeros_dyn"], ["Z"], name="where")
    graph = helper.make_graph(
        [where],
        "where_dyn_bool_y0",
        [
            _vi("cond", TensorProto.BOOL, _SHAPE),
            _vi("X_in", TensorProto.FLOAT, _SHAPE),
        ],
        [_vi("Z", TensorProto.FLOAT, _SHAPE)],
        initializer=[y_init],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", _OPSET_VERSION)])
    _clamp_ir(model)

    cond_val = _RNG.integers(0, 2, _SHAPE).astype(bool)
    x_arr = _RNG.standard_normal(_SHAPE).astype(np.float32)
    _prepare_model(model)
    ref = _run_ort(model, {"cond": cond_val, "X_in": x_arr})

    tr = copy.deepcopy(model)
    _prepare_model(tr)
    tr = _apply(tr)
    _prepare_model(tr)
    got = _run_ort(tr, {"cond": cond_val, "X_in": x_arr})

    assert not any(n.op_type == "Where" for n in tr.graph.node), "Where must be removed"
    op_types = {n.op_type for n in tr.graph.node}
    assert "Cast" in op_types, "Cast node expected for dynamic BOOL condition"
    assert "Sub" not in op_types, "Sub must NOT be inserted for y-is-zero case"
    assert "Mul" in op_types
    np.testing.assert_allclose(ref["Z"], got["Z"], rtol=1e-5, atol=1e-6)


# 5. Greater-produced BOOL condition, false-branch zeros


def test_where_greater_bool_cond_y_zeros():
    """Lower a Greater-produced BOOL condition with a zero false branch."""
    zeros_arr = np.zeros(_SHAPE, dtype=np.float32)
    y_init = numpy_helper.from_array(zeros_arr, name="y_zeros_f")

    greater = helper.make_node("Greater", ["X_in", "thresh"], ["cond_bool"], name="greater")
    where = helper.make_node("Where", ["cond_bool", "X_in", "y_zeros_f"], ["Z"], name="where")

    thresh_init = numpy_helper.from_array(np.array(0.0, dtype=np.float32), name="thresh")
    graph = helper.make_graph(
        [greater, where],
        "where_greater_bool_cond_y0",
        [_vi("X_in", TensorProto.FLOAT, _SHAPE)],
        [_vi("Z", TensorProto.FLOAT, _SHAPE)],
        initializer=[y_init, thresh_init],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", _OPSET_VERSION)])
    _clamp_ir(model)

    x_arr = _RNG.standard_normal(_SHAPE).astype(np.float32)
    _prepare_model(model)
    ref = _run_ort(model, {"X_in": x_arr})

    tr = copy.deepcopy(model)
    _prepare_model(tr)
    tr = _apply(tr)
    _prepare_model(tr)
    got = _run_ort(tr, {"X_in": x_arr})

    assert not any(n.op_type == "Where" for n in tr.graph.node), "Where must be removed"
    np.testing.assert_allclose(ref["Z"], got["Z"], rtol=1e-5, atol=1e-6)


# ── 6. Neither branch is all-zeros → graph unchanged ─────────────────────────


def test_where_no_zero_branch_skipped():
    """Where with no all-zero branch is left unchanged."""
    x_arr = np.ones(_SHAPE, dtype=np.float32)
    y_arr = np.full(_SHAPE, 2.0, dtype=np.float32)
    cond_arr = np.array([[True, False, True], [False, True, False]])

    cond_init = numpy_helper.from_array(cond_arr, name="cond_nz")
    x_init = numpy_helper.from_array(x_arr, name="x_nz")
    y_init = numpy_helper.from_array(y_arr, name="y_nz")

    where = helper.make_node("Where", ["cond_nz", "x_nz", "y_nz"], ["Z"], name="where")
    graph = helper.make_graph(
        [where],
        "where_no_zero",
        [],
        [_vi("Z", TensorProto.FLOAT, _SHAPE)],
        initializer=[cond_init, x_init, y_init],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", _OPSET_VERSION)])
    _clamp_ir(model)
    original_node_count = len(model.graph.node)

    _prepare_model(model)
    ref = _run_ort(model, {})

    tr = copy.deepcopy(model)
    _prepare_model(tr)
    tr = _apply(tr)
    _prepare_model(tr)
    got = _run_ort(tr, {})

    assert len(tr.graph.node) == original_node_count, "graph must be unchanged"
    np.testing.assert_array_equal(ref["Z"], got["Z"])


# ── 7. No Where nodes → noop ──────────────────────────────────────────────────


def test_no_where_nodes_noop():
    """Graph without Where nodes: transform is a noop, ORT parity preserved."""
    shape = [2, 3]
    relu_node = helper.make_node("Relu", ["X"], ["Y"], name="relu")
    graph = helper.make_graph(
        [relu_node],
        "no_where",
        [_vi("X", TensorProto.FLOAT, shape)],
        [_vi("Y", TensorProto.FLOAT, shape)],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", _OPSET_VERSION)])
    _clamp_ir(model)
    original_node_count = len(model.graph.node)

    x = _RNG.standard_normal(shape).astype(np.float32)
    _prepare_model(model)
    ref = _run_ort(model, {"X": x})

    tr = copy.deepcopy(model)
    _prepare_model(tr)
    tr = _apply(tr)
    _prepare_model(tr)
    got = _run_ort(tr, {"X": x})

    assert len(tr.graph.node) == original_node_count, "noop: node count must not change"
    np.testing.assert_array_equal(ref["Y"], got["Y"])


@pytest.mark.parametrize("overridden", ["condition", "zeros"])
@pytest.mark.parametrize("zero_index", [0, 1])
def test_where_respects_overridable_initializer_inputs(overridden, zero_index):
    """Use runtime condition defaults and decline nonconstant zero branches."""
    inputs = [_vi("data", TensorProto.FLOAT, [3]), _vi("condition", TensorProto.BOOL, [3])]
    initializers = [
        numpy_helper.from_array(np.asarray([False, False, False]), "condition"),
        numpy_helper.from_array(np.zeros(3, np.float32), "zeros"),
    ]
    if overridden == "zeros":
        inputs.append(_vi("zeros", TensorProto.FLOAT, [3]))
    branches = ["zeros", "data"] if zero_index == 0 else ["data", "zeros"]
    model = helper.make_model(
        helper.make_graph(
            [helper.make_node("Where", ["condition", *branches], ["y"])],
            "runtime_where",
            inputs,
            [_vi("y", TensorProto.FLOAT, [3])],
            initializers,
        ),
        opset_imports=[helper.make_opsetid("", 17)],
    )
    _clamp_ir(model)
    feeds = {
        "data": np.asarray([2, 3, 4], np.float32),
        "condition": np.asarray([True, False, True]),
    }
    if overridden == "zeros":
        feeds["zeros"] = np.asarray([9, 8, 7], np.float32)
    expected = _run_ort(copy.deepcopy(model), feeds)
    np.testing.assert_array_equal(_run_ort(_apply(copy.deepcopy(model)), feeds)["y"], expected["y"])


@pytest.mark.parametrize("domain", ["custom", ""])
def test_where_preserves_opaque_local_function(domain):
    """Leave a local Where function and its fixed owners intact."""
    function = helper.make_function(
        domain,
        "Where",
        ["condition", "a", "b"],
        ["c"],
        [helper.make_node("Identity", ["b"], ["c"])],
        [helper.make_opsetid("", 17)],
    )
    model = helper.make_model(
        helper.make_graph(
            [helper.make_node("Where", ["condition", "zeros", "data"], ["y"], domain=domain)],
            "opaque",
            [_vi("condition", TensorProto.BOOL, [2]), _vi("data", TensorProto.FLOAT, [2])],
            [_vi("y", TensorProto.FLOAT, [2])],
            [numpy_helper.from_array(np.zeros(2, np.float32), "zeros")],
        ),
        functions=[function],
        opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("custom", 1)],
    )
    _clamp_ir(model)
    original = model.SerializeToString()
    feeds = {"condition": np.asarray([False, True]), "data": np.asarray([2, 3], np.float32)}
    expected = ReferenceEvaluator(model).run(None, feeds)[0]
    _apply(model)
    assert model.SerializeToString() == original
    onnx.checker.check_model(model, full_check=True)
    np.testing.assert_array_equal(ReferenceEvaluator(model).run(None, feeds)[0], expected)


@pytest.mark.parametrize("legacy", [False, True])
def test_where_allocates_distinct_names_and_legacy_controls(legacy):
    """Execute repeated unnamed rewrites with occupied node names and legacy declarations."""
    inputs = [_vi("x", TensorProto.FLOAT, [2]), _vi("condition", TensorProto.BOOL, [2])]
    initializers = [numpy_helper.from_array(np.zeros(2, np.float32), "zeros")]
    nodes = [helper.make_node("Where", ["condition", "x", "zeros"], [out]) for out in ("a", "b")]
    nodes.append(helper.make_node("Identity", ["x"], ["spare"], name="_mul"))
    outputs = [_vi(out, TensorProto.FLOAT, [2]) for out in ("a", "b", "spare")]
    feeds = {"x": np.asarray([2, 3], np.float32), "condition": np.asarray([False, True])}
    if legacy:
        inputs.extend(_vi(t.name, t.data_type, list(t.dims)) for t in initializers)
    model = helper.make_model(
        helper.make_graph(nodes, "names_and_legacy", inputs, outputs, initializers),
        opset_imports=[helper.make_opsetid("", 17)],
    )
    model.ir_version = 3 if legacy else 9
    expected = _run_ort(copy.deepcopy(model), feeds)
    transformed = _apply(copy.deepcopy(model))
    names = [node.name for node in transformed.graph.node if node.name]
    assert len(names) == len(set(names))
    for name, actual in _run_ort(transformed, feeds).items():
        np.testing.assert_array_equal(actual, expected[name])


@pytest.mark.parametrize("source_kind", ["initializer", "Transpose"])
@pytest.mark.parametrize("zero_branch", [0, 1])
@pytest.mark.parametrize("unary", [False, True])
def test_registered_where_uses_materialized_promoted_shapes(source_kind, zero_branch, unary):
    """Lower selections after rank preparation even when intermediate metadata is deferred."""
    nodes, inputs, feeds = [], [], {}
    shape = [1, 3, 2] if source_kind == "initializer" and unary else [3, 2]
    values = np.arange(1, 7, dtype=np.float32).reshape(shape)
    initializers = [
        numpy_helper.from_array(np.array(True), "condition"),
        numpy_helper.from_array(np.zeros(shape, np.float32), "zero"),
    ]
    if source_kind == "initializer":
        initializers.append(numpy_helper.from_array(values, "data"))
    else:
        inputs.append(_vi("x", TensorProto.FLOAT, [2, 3]))
        feeds["x"] = values.T.copy()
        nodes.append(helper.make_node("Transpose", ["x"], ["data"], perm=[1, 0]))
    if unary:
        nodes.append(helper.make_node("Relu", ["data"], ["activated"]))
    branches = ["activated" if unary else "data"] * 2
    branches[zero_branch] = "zero"
    nodes.append(helper.make_node("Where", ["condition", *branches], ["y"]))
    model = helper.make_model(
        helper.make_graph(
            nodes,
            "promoted_where",
            inputs,
            [_vi("y", TensorProto.FLOAT, shape)],
            initializers,
            value_info=[_vi("data", TensorProto.FLOAT, shape)],
        ),
        opset_imports=[helper.make_opsetid("", 17)],
        ir_version=9,
    )
    expected = _run_ort(model, feeds)
    result = _transform_make_dla_compatible(copy.deepcopy(model), verbose=False)
    assert not any(node.op_type == "Where" for node in result.graph.node)
    onnx.checker.check_model(result, full_check=True)
    actual = _run_ort(result, feeds)
    for name in expected:
        np.testing.assert_array_equal(actual[name], expected[name])


def test_shared_fixed_bool_masks_are_reused_by_polarity():
    """Share each direct or inverted FLOAT mask without changing BOOL ownership or values."""
    shape = [1, 1, 2, 3]
    condition = np.array([True, False, True, False, True, False]).reshape(shape)
    branches = [0, 1, 0, 1]
    model = helper.make_model(
        helper.make_graph(
            [
                helper.make_node(
                    "Where",
                    ["condition", "zero" if side == 0 else "x", "x" if side == 0 else "zero"],
                    [f"y{i}"],
                )
                for i, side in enumerate(branches)
            ],
            "shared_bool_masks",
            [_vi("x", TensorProto.FLOAT, shape)],
            [_vi(f"y{i}", TensorProto.FLOAT, shape) for i in range(len(branches))],
            initializer=[
                numpy_helper.from_array(condition, "condition"),
                numpy_helper.from_array(np.float32(0), "zero"),
            ],
        ),
        opset_imports=[helper.make_opsetid("", 17)],
        ir_version=9,
    )
    feeds = {"x": np.arange(1, 7, dtype=np.float32).reshape(shape)}
    onnx.checker.check_model(model, full_check=True)
    expected = ReferenceEvaluator(model).run(None, feeds)
    runtime_expected = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    ).run(None, feeds)
    original = model.graph.initializer[0]
    payload = original.SerializeToString()
    _apply(model)
    onnx.checker.check_model(model, full_check=True)
    masks = [node.input[1] for node in model.graph.node if node.op_type == "Mul"]
    assert len(masks) == 4
    assert masks[0] == masks[2] and masks[1] == masks[3] and masks[0] != masks[1]
    assert (
        sum(t.data_type == TensorProto.FLOAT and t.name != "zero" for t in model.graph.initializer)
        == 2
    )
    assert model.graph.initializer[0] is original
    assert original.SerializeToString() == payload
    for actual, reference in zip(ReferenceEvaluator(model).run(None, feeds), expected):
        np.testing.assert_array_equal(actual, reference)
    runtime_actual = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    ).run(None, feeds)
    for actual, reference in zip(runtime_actual, runtime_expected):
        np.testing.assert_array_equal(actual, reference)
