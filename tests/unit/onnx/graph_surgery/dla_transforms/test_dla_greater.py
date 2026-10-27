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
# Unit tests for :mod:`modelopt.onnx.graph_surgery.dla_transforms.dla_greater`.

"""ORT CPU parity tests for ``dla_greater``.

Transform rule
--------------
For every ``Greater`` node, any initializer input whose dtype is INT32 or
INT64 is replaced by a new float32 initializer (name ``{original}_cast``).
Float32 / other-dtype initializers and dynamic (non-initializer) inputs are
left untouched.

Coverage
--------
INT32 initializer threshold  → cast to float32, ORT parity
INT64 initializer threshold  → cast to float32, ORT parity
float32 initializer          → unchanged (no cast)
both inputs dynamic          → graph unchanged, ORT parity
both inputs INT32 initializers → both cast to float32
"""

from __future__ import annotations

import copy

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper
from onnx.reference import ReferenceEvaluator

from modelopt.onnx.graph_surgery.dla_transforms import dla_greater as _mod

ort = pytest.importorskip("onnxruntime", reason="onnxruntime required for ORT parity checks")

_OPSET_VERSION = 17
_MAX_IR_VERSION_FOR_ORT = 9


_apply = _mod._apply_greater


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


def _init_dtype(model: onnx.ModelProto, name: str) -> int | None:
    """Read a named initializer element type."""
    for init in model.graph.initializer:
        if init.name == name:
            return init.data_type
    return None


_RNG = np.random.default_rng(2024)


# ── 1. INT32 threshold initializer → cast to float32 ─────────────────────────


def test_greater_int32_init_converted():
    """Greater(X_int32, int32_threshold) → threshold becomes float32; ORT parity.

    Both inputs are INT32 so the base model is valid for ORT.
    After the transform the threshold becomes float32; X's declared type is updated
    to float32 in the test (reflecting that upstream transforms will produce float32).
    """
    shape = [3, 4]
    threshold = np.array(5, dtype=np.int32)
    thresh_init = numpy_helper.from_array(threshold, name="thresh")

    greater = helper.make_node("Greater", ["X", "thresh"], ["Y"], name="greater")
    graph = helper.make_graph(
        [greater],
        "greater_i32",
        [_vi("X", TensorProto.INT32, shape)],
        [_vi("Y", TensorProto.BOOL, shape)],
        initializer=[thresh_init],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", _OPSET_VERSION)])
    _clamp_ir(model)

    x = (_RNG.standard_normal(shape) * 10).astype(np.int32)
    _prepare_model(model)
    ref = _run_ort(model, {"X": x})

    tr = copy.deepcopy(model)
    _prepare_model(tr)
    tr = _apply(tr)
    _prepare_model(tr)
    got = _run_ort(tr, {"X": x})

    assert _init_dtype(tr, "thresh_cast") == TensorProto.FLOAT, "threshold must become float32"
    # original INT32 init is left for _remove_unused_initializers to clean up
    np.testing.assert_array_equal(ref["Y"], got["Y"])


# ── 2. INT64 threshold initializer → cast to float32 ─────────────────────────


def test_greater_int64_init_converted():
    """Greater(X_int64, int64_threshold) → threshold becomes float32; ORT parity."""
    shape = [2, 6]
    threshold = np.array(-3, dtype=np.int64)
    thresh_init = numpy_helper.from_array(threshold, name="thresh64")

    greater = helper.make_node("Greater", ["X", "thresh64"], ["Y"], name="greater")
    graph = helper.make_graph(
        [greater],
        "greater_i64",
        [_vi("X", TensorProto.INT64, shape)],
        [_vi("Y", TensorProto.BOOL, shape)],
        initializer=[thresh_init],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", _OPSET_VERSION)])
    _clamp_ir(model)

    x = (_RNG.standard_normal(shape) * 5).astype(np.int64)
    _prepare_model(model)
    ref = _run_ort(model, {"X": x})

    tr = copy.deepcopy(model)
    _prepare_model(tr)
    tr = _apply(tr)
    _prepare_model(tr)
    got = _run_ort(tr, {"X": x})

    assert _init_dtype(tr, "thresh64_cast") == TensorProto.FLOAT
    np.testing.assert_array_equal(ref["Y"], got["Y"])


# ── 3. float32 initializer → unchanged ───────────────────────────────────────


def test_greater_float32_init_unchanged():
    """Greater(X_float32, float32_threshold) — already float32, no change at all."""
    shape = [2, 4]
    threshold = np.array(0.5, dtype=np.float32)
    thresh_init = numpy_helper.from_array(threshold, name="thresh_f32")

    greater = helper.make_node("Greater", ["X", "thresh_f32"], ["Y"], name="greater")
    graph = helper.make_graph(
        [greater],
        "greater_f32",
        [_vi("X", TensorProto.FLOAT, shape)],
        [_vi("Y", TensorProto.BOOL, shape)],
        initializer=[thresh_init],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", _OPSET_VERSION)])
    _clamp_ir(model)
    original_node_count = len(model.graph.node)
    original_init_count = len(model.graph.initializer)

    x = _RNG.standard_normal(shape).astype(np.float32)
    _prepare_model(model)
    ref = _run_ort(model, {"X": x})

    tr = copy.deepcopy(model)
    _prepare_model(tr)
    tr = _apply(tr)
    _prepare_model(tr)
    got = _run_ort(tr, {"X": x})

    # Both inputs already float32 — no Cast inserted, no new initializer, node inputs unchanged
    assert len(tr.graph.node) == original_node_count
    assert len(tr.graph.initializer) == original_init_count
    greater_node = next(n for n in tr.graph.node if n.op_type == "Greater")
    assert greater_node.input[1] == "thresh_f32"
    np.testing.assert_array_equal(ref["Y"], got["Y"])


# ── 4. Both inputs dynamic → graph unchanged ─────────────────────────────────


def test_greater_dynamic_float32_inputs_noop():
    """Greater(X_float32, Z_float32) — both already float32, no Cast inserted."""
    shape = [3, 4]
    greater = helper.make_node("Greater", ["X", "Z"], ["Y"], name="greater")
    graph = helper.make_graph(
        [greater],
        "greater_dynamic",
        [_vi("X", TensorProto.FLOAT, shape), _vi("Z", TensorProto.FLOAT, shape)],
        [_vi("Y", TensorProto.BOOL, shape)],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", _OPSET_VERSION)])
    _clamp_ir(model)
    original_node_count = len(model.graph.node)

    x = _RNG.standard_normal(shape).astype(np.float32) * 5
    z = _RNG.standard_normal(shape).astype(np.float32) * 5
    _prepare_model(model)
    ref = _run_ort(model, {"X": x, "Z": z})

    tr = copy.deepcopy(model)
    _prepare_model(tr)
    tr = _apply(tr)
    _prepare_model(tr)
    got = _run_ort(tr, {"X": x, "Z": z})

    # Both inputs are float32 — shape inference informs transform to skip them
    assert len(tr.graph.node) == original_node_count, (
        "no Cast nodes should be inserted for float32 inputs"
    )
    np.testing.assert_array_equal(ref["Y"], got["Y"])


# ── 5. Both inputs are INT32 initializers → both cast ────────────────────────


def test_greater_both_inputs_int32_converted():
    """Greater(int32_A, int32_B) — both initializers cast to float32."""
    a_init = numpy_helper.from_array(np.array([1, 2, 3, 4], dtype=np.int32), name="A")
    b_init = numpy_helper.from_array(np.array([4, 3, 2, 1], dtype=np.int32), name="B")

    greater = helper.make_node("Greater", ["A", "B"], ["Y"], name="greater")
    graph = helper.make_graph(
        [greater],
        "greater_both_int",
        [],
        [_vi("Y", TensorProto.BOOL, [4])],
        initializer=[a_init, b_init],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", _OPSET_VERSION)])
    _clamp_ir(model)

    _prepare_model(model)
    ref = _run_ort(model, {})

    tr = copy.deepcopy(model)
    _prepare_model(tr)
    tr = _apply(tr)
    _prepare_model(tr)
    got = _run_ort(tr, {})

    assert _init_dtype(tr, "A_cast") == TensorProto.FLOAT, "A must be cast to float32"
    assert _init_dtype(tr, "B_cast") == TensorProto.FLOAT, "B must be cast to float32"
    np.testing.assert_array_equal(ref["Y"], got["Y"])


@pytest.mark.parametrize("runtime_both", [False, True])
def test_greater_preserves_large_integer_ordering(runtime_both):
    """Keep adjacent large integers distinguishable instead of rounding both to FLOAT."""
    shape = [3]
    values = np.asarray([2**24, 2**24 + 1, 2**24 + 2], np.int64)
    threshold = np.full(shape, 2**24, np.int64)
    inputs = [_vi("x", TensorProto.INT64, shape)]
    initializers = []
    feeds = {"x": values}
    if runtime_both:
        inputs.append(_vi("threshold", TensorProto.INT64, shape))
        feeds["threshold"] = threshold
    else:
        initializers.append(numpy_helper.from_array(threshold, "threshold"))
    model = helper.make_model(
        helper.make_graph(
            [helper.make_node("Greater", ["x", "threshold"], ["y"])],
            "integer_ordering",
            inputs,
            [_vi("y", TensorProto.BOOL, shape)],
            initializers,
        ),
        opset_imports=[helper.make_opsetid("", 17)],
    )
    _clamp_ir(model)
    expected = _run_ort(copy.deepcopy(model), feeds)
    transformed = _apply(copy.deepcopy(model))
    np.testing.assert_array_equal(_run_ort(transformed, feeds)["y"], expected["y"])


def test_greater_preserves_runtime_initializer_threshold():
    """Do not fold an overridable comparison threshold into its default value."""
    model = helper.make_model(
        helper.make_graph(
            [helper.make_node("Greater", ["x", "threshold"], ["y"])],
            "runtime_threshold",
            [_vi("x", TensorProto.INT32, [3]), _vi("threshold", TensorProto.INT32, [])],
            [_vi("y", TensorProto.BOOL, [3])],
            [numpy_helper.from_array(np.asarray(0, np.int32), "threshold")],
        ),
        opset_imports=[helper.make_opsetid("", 17)],
    )
    _clamp_ir(model)
    feeds = {"x": np.asarray([1, 2, 3], np.int32), "threshold": np.asarray(2, np.int32)}
    expected = _run_ort(copy.deepcopy(model), feeds)
    np.testing.assert_array_equal(_run_ort(_apply(copy.deepcopy(model)), feeds)["y"], expected["y"])


@pytest.mark.parametrize("domain", ["custom", ""])
def test_greater_preserves_opaque_local_function(domain):
    """Leave model-local comparisons and their native operand types untouched."""
    function = helper.make_function(
        domain,
        "Greater",
        ["a", "b"],
        ["c"],
        [helper.make_node("Less", ["a", "b"], ["c"])],
        [helper.make_opsetid("", 17)],
    )
    model = helper.make_model(
        helper.make_graph(
            [helper.make_node("Greater", ["x", "threshold"], ["y"], domain=domain)],
            "opaque",
            [_vi("x", TensorProto.INT32, [2])],
            [_vi("y", TensorProto.BOOL, [2])],
            [numpy_helper.from_array(np.asarray(1, np.int32), "threshold")],
        ),
        functions=[function],
        opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("custom", 1)],
    )
    _clamp_ir(model)
    original = model.SerializeToString()
    feeds = {"x": np.asarray([0, 2], np.int32)}
    expected = ReferenceEvaluator(model).run(None, feeds)[0]
    _apply(model)
    assert model.SerializeToString() == original
    onnx.checker.check_model(model, full_check=True)
    np.testing.assert_array_equal(ReferenceEvaluator(model).run(None, feeds)[0], expected)


@pytest.mark.parametrize("legacy", [False, True])
def test_greater_allocates_distinct_names_and_legacy_controls(legacy):
    """Execute repeated unnamed rewrites with occupied node names and legacy declarations."""
    inputs = [_vi("x", TensorProto.INT32, [2])]
    initializers = [numpy_helper.from_array(np.asarray(1, np.int32), "threshold")]
    nodes = [helper.make_node("Greater", ["x", "threshold"], [out]) for out in ("a", "b")]
    nodes.append(helper.make_node("Identity", ["x"], ["spare"], name="x_cast_to_float32"))
    outputs = [_vi(out, TensorProto.BOOL, [2]) for out in ("a", "b")]
    outputs.append(_vi("spare", TensorProto.INT32, [2]))
    feeds = {"x": np.asarray([0, 2], np.int32)}
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


@pytest.mark.parametrize("fixed", [False, True])
def test_greater_half_precision_operands_convert_together(fixed):
    """Preserve finite and nonfinite half comparisons with matching FLOAT operand types."""
    shape = [4]
    inputs = [_vi("x", TensorProto.FLOAT16, shape)]
    initializers = []
    threshold = np.asarray([0, 0, 0, 0], np.float16)
    feeds = {"x": np.asarray([-1, 1, np.nan, np.inf], np.float16)}
    if fixed:
        initializers.append(numpy_helper.from_array(threshold, "threshold"))
    else:
        inputs.append(_vi("threshold", TensorProto.FLOAT16, shape))
        feeds["threshold"] = threshold
    model = helper.make_model(
        helper.make_graph(
            [helper.make_node("Greater", ["x", "threshold"], ["y"])],
            "half_operands",
            inputs,
            [_vi("y", TensorProto.BOOL, shape)],
            initializers,
        ),
        opset_imports=[helper.make_opsetid("", 17)],
    )
    _clamp_ir(model)
    expected = _run_ort(copy.deepcopy(model), feeds)
    np.testing.assert_array_equal(_run_ort(_apply(copy.deepcopy(model)), feeds)["y"], expected["y"])
