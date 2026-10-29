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
# Unit tests for :mod:`modelopt.onnx.graph_surgery.dla_transforms.dla_constants_to_initializers`.

"""ORT CPU parity tests for ``constants_to_initializers`` / ``_constants_to_initializers_from_model``.

Transform rule
--------------
Every ``Constant`` node is removed and replaced by a graph initializer carrying
the same tensor data.  Downstream consumers are unaffected because they
reference the tensor by name, which is preserved.

Supported Constant payload attributes
--------------------------------------
``value``          — dense TensorProto (any dtype / any shape)
``value_float``    — scalar float32
``value_floats``   — 1-D float32 list
``value_int``      — scalar int64
``value_ints``     — 1-D int64 list

Coverage
--------
value (float32 tensor)              → node removed, initializer added, ORT parity
value_float (scalar)                → node removed, initializer added, ORT parity
value_floats (1-D list)             → node removed, initializer added, ORT parity
value_int  (scalar)                 → node removed, initializer added
value_ints (1-D list)               → node removed, initializer added
no Constant nodes                   → graph unchanged, ORT output unchanged
multiple Constants                  → all converted in one pass, ORT parity
"""

from __future__ import annotations

import copy

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from modelopt.onnx.graph_surgery.dla_transforms import dla_constants_to_initializers as _mod
from modelopt.onnx.graph_surgery.make_dla_compatible import _transform_make_dla_compatible

ort = pytest.importorskip("onnxruntime", reason="onnxruntime required for ORT parity checks")


@pytest.mark.parametrize("ir_version", [3, 9])
@pytest.mark.parametrize("used", [False, True])
def test_pipeline_constant_cleanup_preserves_required_feeds(ir_version, used):
    """Pruning a folded legacy default must not turn it into a required runtime input."""
    shape = [1, 1, 2, 3]
    model = helper.make_model(
        helper.make_graph(
            [
                helper.make_node(
                    "Constant",
                    [],
                    ["weight"],
                    value=numpy_helper.from_array(np.ones(shape, np.float32)),
                ),
                helper.make_node(
                    "Add" if used else "Identity", ["x", "weight"] if used else ["x"], ["y"]
                ),
            ],
            "legacy_constant_cleanup",
            [_vi("x", TensorProto.FLOAT, shape), _vi("original_default", TensorProto.FLOAT, shape)],
            [_vi("y", TensorProto.FLOAT, shape)],
            initializer=[
                numpy_helper.from_array(np.full(shape, 2, np.float32), "original_default")
            ],
        ),
        opset_imports=[helper.make_opsetid("", 13)],
        ir_version=ir_version,
    )
    feeds = {"x": np.arange(6, dtype=np.float32).reshape(shape)}
    model.graph.input[1].doc_string = "Caller-declared optional input"
    original_input = model.graph.input[1].SerializeToString()
    original_default = model.graph.initializer[0].SerializeToString()
    onnx.checker.check_model(model, full_check=True)
    expected = _run_ort(model, feeds)
    transformed = _transform_make_dla_compatible(copy.deepcopy(model), verbose=False)
    onnx.checker.check_model(transformed, full_check=True)
    assert (
        next(
            value for value in transformed.graph.input if value.name == "original_default"
        ).SerializeToString()
        == original_input
    )
    assert (
        next(
            value for value in transformed.graph.initializer if value.name == "original_default"
        ).SerializeToString()
        == original_default
    )
    session = ort.InferenceSession(
        transformed.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    assert [value.name for value in session.get_inputs()] == ["x"]
    assert ("weight" in {value.name for value in transformed.graph.initializer}) == used
    np.testing.assert_array_equal(session.run(None, feeds)[0], expected["y"])


_OPSET_VERSION = 17
_MAX_IR_VERSION_FOR_ORT = 9


_apply = _mod._constants_to_initializers_from_model


# ── Shared utilities ──────────────────────────────────────────────────────────


def _vi(name: str, elem_type: int, shape: list) -> onnx.ValueInfoProto:
    """Build a typed tensor descriptor for a small test graph."""
    return helper.make_tensor_value_info(name, elem_type, shape)


def _clamp_ir(model: onnx.ModelProto) -> None:
    """Keep the test model compatible with the supported ORT versions."""
    model.ir_version = min(model.ir_version, _MAX_IR_VERSION_FOR_ORT)


def _prepare_model(model: onnx.ModelProto) -> None:
    """Refresh intermediate metadata and surface invalid graph or inference errors."""
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    onnx.checker.check_model(inferred, full_check=True)
    del model.graph.value_info[:]
    model.graph.value_info.extend(inferred.graph.value_info)


def _run_ort(model: onnx.ModelProto, feeds: dict) -> dict:
    """Execute the test model on the CPU and return named outputs."""
    _clamp_ir(model)
    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    return {o.name: sess.run([o.name], feeds)[0] for o in model.graph.output}


def _initializer_names(model: onnx.ModelProto) -> set[str]:
    """Collect initializer names to check that folded values remain addressable."""
    return {i.name for i in model.graph.initializer}


def _constant_node_names(model: onnx.ModelProto) -> set[str]:
    """Collect Constant output names that have not been folded."""
    return {n.output[0] for n in model.graph.node if n.op_type == "Constant"}


_RNG = np.random.default_rng(2024)


# ── 1. value attribute (dense float32 tensor) ────────────────────────────────


@pytest.mark.parametrize(("ir_version", "opset"), [(3, 7), (4, 9), (9, _OPSET_VERSION)])
def test_constant_value_tensor_converted(ir_version, opset):
    """Constant with 'value' (TensorProto) replaced by initializer; ORT output unchanged."""
    shape = [2, 4]
    const_data = _RNG.standard_normal(shape).astype(np.float32)
    const_init = numpy_helper.from_array(const_data, name="const_w")
    const_node = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["const_w"],
        name="c_w",
        value=const_init,
    )
    # Add const_w to X element-wise
    add_node = helper.make_node("Add", ["X", "const_w"], ["Y"], name="add")

    graph = helper.make_graph(
        [const_node, add_node],
        "const_value",
        [_vi("X", TensorProto.FLOAT, shape)],
        [_vi("Y", TensorProto.FLOAT, shape)],
        value_info=[_vi("const_w", TensorProto.FLOAT, shape)],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", opset)], ir_version=ir_version
    )
    _clamp_ir(model)

    x = _RNG.standard_normal(shape).astype(np.float32)
    ref = _run_ort(model, {"X": x})

    tr = copy.deepcopy(model)
    _prepare_model(tr)
    cnt = _apply(tr, verbose=False)
    onnx.checker.check_model(tr, full_check=True)
    assert [value.name for value in tr.graph.input] == (
        ["X", "const_w"] if ir_version <= 3 else ["X"]
    )
    got = _run_ort(tr, {"X": x})

    assert cnt == 1
    assert "const_w" in _initializer_names(tr)
    assert "const_w" not in _constant_node_names(tr)
    np.testing.assert_allclose(ref["Y"], got["Y"], rtol=1e-5, atol=1e-5)


# ── 2. value_float attribute (scalar float32) ────────────────────────────────


@pytest.mark.parametrize("layout", ["linear", "coordinates", "empty"])
def test_sparse_constant_is_densified_without_changing_values(layout):
    """Sparse Constant layouts become dense initializers with identical tensor values."""
    shape = [2, 3]
    values = np.array([] if layout == "empty" else [2.0, -4.0], dtype=np.float32)
    indices = np.array(
        [[0, 1], [1, 2]] if layout == "coordinates" else ([] if layout == "empty" else [1, 5]),
        dtype=np.int64,
    )
    sparse = helper.make_sparse_tensor(
        numpy_helper.from_array(values, "K"), numpy_helper.from_array(indices, "indices"), shape
    )
    model = helper.make_model(
        helper.make_graph(
            [
                helper.make_node("Constant", [], ["K"], sparse_value=sparse),
                helper.make_node("Add", ["X", "K"], ["Y"]),
            ],
            "sparse_constant",
            [_vi("X", TensorProto.FLOAT, shape)],
            [_vi("Y", TensorProto.FLOAT, shape)],
        ),
        opset_imports=[helper.make_opsetid("", _OPSET_VERSION)],
        ir_version=_MAX_IR_VERSION_FOR_ORT,
    )
    onnx.checker.check_model(model, full_check=True)
    expected = np.zeros(shape, dtype=np.float32)
    if layout != "empty":
        expected.reshape(-1)[[1, 5]] = values
    assert _apply(model, verbose=False) == 1
    assert not _constant_node_names(model)
    initializer = next(value for value in model.graph.initializer if value.name == "K")
    np.testing.assert_array_equal(numpy_helper.to_array(initializer), expected)
    onnx.checker.check_model(model, full_check=True)
    x = np.arange(6, dtype=np.float32).reshape(shape)
    np.testing.assert_array_equal(_run_ort(model, {"X": x})["Y"], x + expected)


def test_constant_value_float_converted():
    """Constant with 'value_float' attribute converted and ORT output matches."""
    shape = [3, 4]
    scalar = 2.5
    const_node = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["scalar_c"],
        name="c_f",
        value_float=scalar,
    )
    mul_node = helper.make_node("Mul", ["X", "scalar_c"], ["Y"], name="mul")

    graph = helper.make_graph(
        [const_node, mul_node],
        "const_float",
        [_vi("X", TensorProto.FLOAT, shape)],
        [_vi("Y", TensorProto.FLOAT, shape)],
        value_info=[_vi("scalar_c", TensorProto.FLOAT, [])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", _OPSET_VERSION)])
    _clamp_ir(model)

    x = _RNG.standard_normal(shape).astype(np.float32)
    ref = _run_ort(model, {"X": x})

    tr = copy.deepcopy(model)
    _prepare_model(tr)
    cnt = _apply(tr, verbose=False)
    got = _run_ort(tr, {"X": x})

    assert cnt == 1
    assert "scalar_c" in _initializer_names(tr)
    assert "scalar_c" not in _constant_node_names(tr)
    np.testing.assert_allclose(ref["Y"], got["Y"], rtol=1e-5, atol=1e-5)


# ── 3. value_floats attribute (1-D float32 list) ──────────────────────────────


def test_constant_value_floats_converted():
    """Constant with 'value_floats' (list) converted; ORT output matches."""
    floats = [1.0, 2.0, 3.0, 4.0]
    shape_x = [2, 4]
    const_node = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["bias"],
        name="c_fs",
        value_floats=floats,
    )
    add_node = helper.make_node("Add", ["X", "bias"], ["Y"], name="add")

    graph = helper.make_graph(
        [const_node, add_node],
        "const_floats",
        [_vi("X", TensorProto.FLOAT, shape_x)],
        [_vi("Y", TensorProto.FLOAT, shape_x)],
        value_info=[_vi("bias", TensorProto.FLOAT, [len(floats)])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", _OPSET_VERSION)])
    _clamp_ir(model)

    x = _RNG.standard_normal(shape_x).astype(np.float32)
    ref = _run_ort(model, {"X": x})

    tr = copy.deepcopy(model)
    _prepare_model(tr)
    cnt = _apply(tr, verbose=False)
    got = _run_ort(tr, {"X": x})

    assert cnt == 1
    assert "bias" in _initializer_names(tr)
    assert "bias" not in _constant_node_names(tr)
    np.testing.assert_allclose(ref["Y"], got["Y"], rtol=1e-5, atol=1e-5)


# ── 4. value_int attribute (scalar int64) ────────────────────────────────────


def test_constant_value_int_converted():
    """Constant with 'value_int' converted to initializer (int64 scalar).

    Uses Gather with a scalar index, which is the natural consumer of a
    scalar int64 constant (ReduceSum opset-13 requires a 1-D axes tensor).
    """
    shape = [4, 8]
    const_node = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["idx"],
        name="c_i",
        value_int=1,
    )
    # Gather selects row 1 from X → output shape [8]
    gather_node = helper.make_node("Gather", ["X", "idx"], ["Y"], name="gather", axis=0)

    graph = helper.make_graph(
        [const_node, gather_node],
        "const_int",
        [_vi("X", TensorProto.FLOAT, shape)],
        [_vi("Y", TensorProto.FLOAT, [shape[1]])],
        value_info=[_vi("idx", TensorProto.INT64, [])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", _OPSET_VERSION)])
    _clamp_ir(model)

    x = _RNG.standard_normal(shape).astype(np.float32)
    ref = _run_ort(model, {"X": x})

    tr = copy.deepcopy(model)
    _prepare_model(tr)
    cnt = _apply(tr, verbose=False)
    got = _run_ort(tr, {"X": x})

    assert cnt == 1
    assert "idx" in _initializer_names(tr)
    assert "idx" not in _constant_node_names(tr)
    np.testing.assert_allclose(ref["Y"], got["Y"], rtol=1e-5, atol=1e-5)


# ── 5. value_ints attribute (1-D int64 list) ──────────────────────────────────


def test_constant_value_ints_converted():
    """Constant with 'value_ints' (shape tensor) converted; Reshape ORT parity."""
    shape_in = [2, 4, 8]
    target_shape = [2, 32]
    const_node = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["new_shape"],
        name="c_is",
        value_ints=target_shape,
    )
    reshape_node = helper.make_node("Reshape", ["X", "new_shape"], ["Y"], name="reshape")

    graph = helper.make_graph(
        [const_node, reshape_node],
        "const_ints",
        [_vi("X", TensorProto.FLOAT, shape_in)],
        [_vi("Y", TensorProto.FLOAT, target_shape)],
        value_info=[_vi("new_shape", TensorProto.INT64, [len(target_shape)])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", _OPSET_VERSION)])
    _clamp_ir(model)

    x = _RNG.standard_normal(shape_in).astype(np.float32)
    ref = _run_ort(model, {"X": x})

    tr = copy.deepcopy(model)
    _prepare_model(tr)
    cnt = _apply(tr, verbose=False)
    got = _run_ort(tr, {"X": x})

    assert cnt == 1
    assert "new_shape" in _initializer_names(tr)
    assert "new_shape" not in _constant_node_names(tr)
    np.testing.assert_allclose(ref["Y"], got["Y"], rtol=1e-5, atol=1e-5)


# ── 6. No Constant nodes: graph unchanged ────────────────────────────────────


def test_no_constants_noop():
    """Model with no Constant nodes: graph and ORT output unchanged."""
    shape = [2, 4]
    w_init = numpy_helper.from_array(_RNG.standard_normal(shape).astype(np.float32), name="W")
    add_node = helper.make_node("Add", ["X", "W"], ["Y"], name="add")

    graph = helper.make_graph(
        [add_node],
        "no_const",
        [_vi("X", TensorProto.FLOAT, shape)],
        [_vi("Y", TensorProto.FLOAT, shape)],
        initializer=[w_init],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", _OPSET_VERSION)])
    _clamp_ir(model)
    original_node_count = len(model.graph.node)

    x = _RNG.standard_normal(shape).astype(np.float32)
    ref = _run_ort(model, {"X": x})

    tr = copy.deepcopy(model)
    _prepare_model(tr)
    cnt = _apply(tr, verbose=False)
    got = _run_ort(tr, {"X": x})

    assert cnt == 0
    assert len(tr.graph.node) == original_node_count
    np.testing.assert_array_equal(ref["Y"], got["Y"])


# ── 7. Multiple Constants in one graph ───────────────────────────────────────


def test_multiple_constants_all_converted():
    """All Constant nodes in the graph are converted in a single pass; ORT parity."""
    shape = [3, 4]
    w_data = _RNG.standard_normal(shape).astype(np.float32)
    b_data = _RNG.standard_normal([shape[1]]).astype(np.float32)

    w_const = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["W"],
        name="c_w",
        value=numpy_helper.from_array(w_data, name="W"),
    )
    b_const = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["B"],
        name="c_b",
        value=numpy_helper.from_array(b_data, name="B"),
    )
    matmul = helper.make_node("MatMul", ["X", "W"], ["mm_out"], name="mm")
    add = helper.make_node("Add", ["mm_out", "B"], ["Y"], name="add")

    graph = helper.make_graph(
        [w_const, b_const, matmul, add],
        "multi_const",
        [_vi("X", TensorProto.FLOAT, [shape[0], shape[0]])],
        [_vi("Y", TensorProto.FLOAT, [shape[0], shape[1]])],
        value_info=[
            _vi("W", TensorProto.FLOAT, shape),
            _vi("B", TensorProto.FLOAT, [shape[1]]),
            _vi("mm_out", TensorProto.FLOAT, [shape[0], shape[1]]),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", _OPSET_VERSION)])
    _clamp_ir(model)

    x = _RNG.standard_normal([shape[0], shape[0]]).astype(np.float32)
    ref = _run_ort(model, {"X": x})

    tr = copy.deepcopy(model)
    _prepare_model(tr)
    cnt = _apply(tr, verbose=False)
    got = _run_ort(tr, {"X": x})

    assert cnt == 2
    assert "W" in _initializer_names(tr)
    assert "B" in _initializer_names(tr)
    assert not any(n.op_type == "Constant" for n in tr.graph.node)
    np.testing.assert_allclose(ref["Y"], got["Y"], rtol=1e-5, atol=1e-5)
