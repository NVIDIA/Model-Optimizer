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
# Unit tests for :mod:`modelopt.onnx.graph_surgery.dla_transforms.dla_handle_qdq`.

"""ORT CPU parity tests for ``dla_handle_qdq``.

Transform rule
--------------
Every ``QuantizeLinear`` and ``DequantizeLinear`` node whose data input is
non-4D gets wrapped independently:

    data [*orig_shape]
    → Unsqueeze  → data_4d  [1,…,1, *orig_shape]
    → Q / DQ     → out_4d   [1,…,1, *orig_shape]
    → Squeeze    → out      [*orig_shape]

Scale / zero-point inputs are untouched.  The ``axis`` attribute is shifted
by ``4 - orig_rank``.  Nodes whose data input is already 4-D are skipped.

Coverage
--------
QuantizeLinear  3-D input (dynamic) → wrapped, axis adjusted, ORT parity
DequantizeLinear 3-D input (static init) → wrapped, axis adjusted, ORT parity
DequantizeLinear 2-D input (dynamic) → wrapped, ORT parity
Q already 4-D → skipped (no Unsqueeze/Squeeze inserted)
"""

from __future__ import annotations

import copy

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper
from onnx.reference import ReferenceEvaluator

from modelopt.onnx.graph_surgery.dla_transforms import dla_handle_qdq as _mod

ort = pytest.importorskip("onnxruntime", reason="onnxruntime required for ORT parity checks")

_OPSET_VERSION = 21  # INT4/UINT4 require ≥ opset 21
_MAX_IR_VERSION_FOR_ORT = 9


_apply = _mod._apply_handle_qdq


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


def _assert_wrapped(tr, op_type: str, orig_name: str) -> None:
    """Check Unsqueeze→op→Squeeze chain is present and op has the right name."""
    assert any(n.op_type == "Unsqueeze" for n in tr.graph.node), (
        f"Unsqueeze must be inserted before {op_type}"
    )
    assert any(n.op_type == "Squeeze" for n in tr.graph.node), (
        f"Squeeze must be inserted after {op_type}"
    )
    assert any(n.op_type == op_type for n in tr.graph.node), f"{op_type} node must still be present"


_RNG = np.random.default_rng(2024)


@pytest.mark.parametrize("op_type", ["QuantizeLinear", "DequantizeLinear"])
def test_runtime_per_tensor_qdq_keeps_axis_absent(op_type):
    """Runtime per-tensor QDQ retains scalar controls and its absent axis when wrapped."""
    quantize = op_type == "QuantizeLinear"
    shape = [2, 3]
    input_dtype = TensorProto.FLOAT if quantize else TensorProto.UINT8
    output_dtype = TensorProto.UINT8 if quantize else TensorProto.FLOAT
    model = helper.make_model(
        helper.make_graph(
            [helper.make_node(op_type, ["X", "scale", "zero"], ["Y"], domain="com.microsoft")],
            "runtime_per_tensor",
            [_vi("X", input_dtype, shape)],
            [_vi("Y", output_dtype, shape)],
            initializer=[
                numpy_helper.from_array(np.float32(0.25), "scale"),
                numpy_helper.from_array(np.uint8(0), "zero"),
            ],
        ),
        opset_imports=[helper.make_opsetid("", 10), helper.make_opsetid("com.microsoft", 1)],
        ir_version=_MAX_IR_VERSION_FOR_ORT,
    )
    onnx.checker.check_model(model, full_check=True)
    data = np.arange(6, dtype=np.float32 if quantize else np.uint8).reshape(shape)
    expected = _run_ort(model, {"X": data})["Y"]
    _apply(model)
    onnx.checker.check_model(model, full_check=True)
    transformed = next(node for node in model.graph.node if node.op_type == op_type)
    assert not any(attribute.name == "axis" for attribute in transformed.attribute)
    _assert_wrapped(model, op_type, transformed.name)
    np.testing.assert_array_equal(_run_ort(model, {"X": data})["Y"], expected)


# ── 1. QuantizeLinear — 3-D dynamic input, per-channel axis ──────────────────


@pytest.mark.parametrize(("opset", "domain"), [(21, ""), (10, "com.microsoft")])
@pytest.mark.parametrize("shape", [[2, 4, 8], ["batch", 4, None]])
def test_quantize_linear_3d_dynamic_wrapped(opset, domain, shape):
    """Q [2,4,8] axis=1 → Unsqueeze→Q(axis=2)→Squeeze; output shape and values match base."""
    scale = numpy_helper.from_array(np.full((4,), 0.1, dtype=np.float32), name="scale")
    zp = numpy_helper.from_array(np.zeros((4,), dtype=np.int8), name="zp")

    q = helper.make_node(
        "QuantizeLinear", ["X", "scale", "zp"], ["Y"], name="q", axis=1, domain=domain
    )
    graph = helper.make_graph(
        [q],
        "g",
        [_vi("X", TensorProto.FLOAT, shape)],
        [_vi("Y", TensorProto.INT8, shape)],
        initializer=[scale, zp],
    )
    model = helper.make_model(
        graph,
        opset_imports=[
            helper.make_opsetid("", opset),
            *([helper.make_opsetid(domain, 1)] if domain else []),
        ],
    )
    _clamp_ir(model)

    x = _RNG.standard_normal([2, 4, 8]).astype(np.float32)
    _prepare_model(model)
    ref = _run_ort(model, {"X": x})

    tr = copy.deepcopy(model)
    _prepare_model(tr)
    tr = _apply(tr)
    q_node = next(n for n in tr.graph.node if n.op_type == "QuantizeLinear")
    for name in [q_node.input[0], q_node.output[0]]:
        info = next(value for value in tr.graph.value_info if value.name == name)
        actual = [
            d.dim_value if d.HasField("dim_value") else d.dim_param or None
            for d in info.type.tensor_type.shape.dim
        ]
        assert actual == [1, *shape]
    _prepare_model(tr)
    got = _run_ort(tr, {"X": x})

    _assert_wrapped(tr, "QuantizeLinear", "q")

    # axis must have been shifted from 1 to 2 (4-3=1 delta)
    q_node = next(n for n in tr.graph.node if n.op_type == "QuantizeLinear")
    axis_attr = next((a for a in q_node.attribute if a.name == "axis"), None)
    assert axis_attr is not None and axis_attr.i == 2, (
        f"axis must be shifted to 2, got {axis_attr.i if axis_attr else 'None'}"
    )

    # scale and zp must be untouched (still 1-D)
    s_init = next(i for i in tr.graph.initializer if i.name == "scale")
    z_init = next(i for i in tr.graph.initializer if i.name == "zp")
    assert list(s_init.dims) == [4], f"scale must stay 1-D [4], got {list(s_init.dims)}"
    assert list(z_init.dims) == [4], f"zp must stay 1-D [4], got {list(z_init.dims)}"

    np.testing.assert_array_equal(ref["Y"], got["Y"])


# ── 2. DequantizeLinear — 3-D static initializer data input, per-channel ─────


@pytest.mark.parametrize(("opset", "domain"), [(21, ""), (10, "com.microsoft")])
def test_dequantize_linear_3d_static_init_wrapped(opset, domain):
    """DQ with static int8 initializer [2,4,8] axis=1 → wrapped; float output matches base."""
    shape = [2, 4, 8]
    data = numpy_helper.from_array((_RNG.integers(-5, 6, shape)).astype(np.int8), name="data")
    scale = numpy_helper.from_array(np.full((4,), 0.1, dtype=np.float32), name="dq_scale")
    zp = numpy_helper.from_array(np.zeros((4,), dtype=np.int8), name="dq_zp")

    dq = helper.make_node(
        "DequantizeLinear", ["data", "dq_scale", "dq_zp"], ["Y"], name="dq", axis=1, domain=domain
    )
    graph = helper.make_graph(
        [dq],
        "g",
        [],
        [_vi("Y", TensorProto.FLOAT, shape)],
        initializer=[data, scale, zp],
    )
    model = helper.make_model(
        graph,
        opset_imports=[
            helper.make_opsetid("", opset),
            *([helper.make_opsetid(domain, 1)] if domain else []),
        ],
    )
    _clamp_ir(model)

    _prepare_model(model)
    ref = _run_ort(model, {})

    tr = copy.deepcopy(model)
    _prepare_model(tr)
    tr = _apply(tr)
    _prepare_model(tr)
    got = _run_ort(tr, {})

    _assert_wrapped(tr, "DequantizeLinear", "dq")

    dq_node = next(n for n in tr.graph.node if n.op_type == "DequantizeLinear")
    axis_attr = next((a for a in dq_node.attribute if a.name == "axis"), None)
    assert axis_attr is not None and axis_attr.i == 2, (
        f"axis must be shifted to 2, got {axis_attr.i if axis_attr else 'None'}"
    )

    np.testing.assert_allclose(ref["Y"], got["Y"], rtol=1e-5, atol=1e-6)


# ── 3. DequantizeLinear — 2-D dynamic input ───────────────────────────────────


def test_dequantize_linear_2d_dynamic_wrapped():
    """DQ [4,8] dynamic input (per-tensor, no axis) → Unsqueeze→DQ→Squeeze; ORT parity."""
    shape = [4, 8]
    scale = numpy_helper.from_array(np.array(0.1, dtype=np.float32), name="s2")
    zp = numpy_helper.from_array(np.array(0, dtype=np.int8), name="z2")

    dq = helper.make_node("DequantizeLinear", ["X", "s2", "z2"], ["Y"], name="dq2")
    graph = helper.make_graph(
        [dq],
        "g",
        [_vi("X", TensorProto.INT8, shape)],
        [_vi("Y", TensorProto.FLOAT, shape)],
        initializer=[scale, zp],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", _OPSET_VERSION)])
    _clamp_ir(model)

    x = _RNG.integers(-5, 6, shape).astype(np.int8)
    _prepare_model(model)
    ref = _run_ort(model, {"X": x})

    tr = copy.deepcopy(model)
    _prepare_model(tr)
    tr = _apply(tr)
    _prepare_model(tr)
    got = _run_ort(tr, {"X": x})

    _assert_wrapped(tr, "DequantizeLinear", "dq2")
    np.testing.assert_allclose(ref["Y"], got["Y"], rtol=1e-5, atol=1e-6)


# ── 4. QuantizeLinear already 4-D → skipped ──────────────────────────────────


def test_quantize_linear_4d_skipped():
    """Q with 4-D input is already 4-D → no Unsqueeze/Squeeze inserted, graph unchanged."""
    shape = [1, 2, 4, 8]
    scale = numpy_helper.from_array(np.full((2,), 0.1, dtype=np.float32), name="s4d")
    zp = numpy_helper.from_array(np.zeros((2,), dtype=np.int8), name="z4d")

    q = helper.make_node("QuantizeLinear", ["X", "s4d", "z4d"], ["Y"], name="q4d", axis=1)
    graph = helper.make_graph(
        [q],
        "g",
        [_vi("X", TensorProto.FLOAT, shape)],
        [_vi("Y", TensorProto.INT8, shape)],
        initializer=[scale, zp],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", _OPSET_VERSION)])
    _clamp_ir(model)
    orig_count = len(model.graph.node)

    x = _RNG.standard_normal(shape).astype(np.float32)
    _prepare_model(model)
    ref = _run_ort(model, {"X": x})

    tr = copy.deepcopy(model)
    _prepare_model(tr)
    tr = _apply(tr)
    _prepare_model(tr)
    got = _run_ort(tr, {"X": x})

    assert len(tr.graph.node) == orig_count, "4-D input: no nodes should be added"
    assert not any(n.op_type == "Unsqueeze" for n in tr.graph.node)
    assert not any(n.op_type == "Squeeze" for n in tr.graph.node)
    np.testing.assert_array_equal(ref["Y"], got["Y"])


@pytest.mark.parametrize(
    ("opset", "collision"),
    [
        (10, None),
        (12, None),
        (21, None),
        (21, "node0_unsq_axes_0"),
        (21, "node0_in4d_0"),
        (21, "node0_out4d_0"),
        (21, "node0_unsq_0"),
    ],
)
def test_mixed_qdq_nodes_use_their_snapshot_positions(opset, collision):
    """Mixed-rank Q/DQ nodes retain values when only selected positions are wrapped."""
    specs = [
        ("QuantizeLinear", [3]),
        ("DequantizeLinear", [1, 1, 2, 2]),
        ("QuantizeLinear", [2, 2, 3]),
        ("DequantizeLinear", [2, 3]),
    ]
    nodes, inputs, outputs, feeds = [], [], [], {}
    for index, (op_type, shape) in enumerate(specs):
        input_name, output_name = f"X{index}", f"Y{index}"
        quantize = op_type == "QuantizeLinear"
        input_dtype = TensorProto.FLOAT if quantize else TensorProto.UINT8
        output_dtype = TensorProto.UINT8 if quantize else TensorProto.FLOAT
        nodes.append(
            helper.make_node(
                op_type, [input_name, "scale", "zero"], [output_name], name=f"node{index}"
            )
        )
        inputs.append(_vi(input_name, input_dtype, shape))
        outputs.append(_vi(output_name, output_dtype, shape))
        feeds[input_name] = np.arange(
            np.prod(shape), dtype=np.float32 if quantize else np.uint8
        ).reshape(shape)
    initializers = [
        numpy_helper.from_array(np.float32(0.25), "scale"),
        numpy_helper.from_array(np.uint8(0), "zero"),
    ]
    if collision == "node0_unsq_axes_0":
        initializers.append(numpy_helper.from_array(np.array([0], dtype=np.int64), collision))
    elif collision == "node0_unsq_0":
        nodes.append(helper.make_node("Identity", ["X1"], ["protected"], name=collision))
        outputs.append(_vi("protected", TensorProto.UINT8, [1, 1, 2, 2]))
    elif collision:
        inputs.append(_vi(collision, TensorProto.FLOAT, [1]))
        feeds[collision] = np.array([1], dtype=np.float32)
    model = helper.make_model(
        helper.make_graph(
            nodes,
            "mixed_qdq",
            inputs,
            outputs,
            initializer=initializers,
        ),
        opset_imports=[helper.make_opsetid("", opset)],
        ir_version=_MAX_IR_VERSION_FOR_ORT,
    )
    onnx.checker.check_model(model, full_check=True)
    expected = _run_ort(model, feeds)
    transformed = _apply(copy.deepcopy(model))
    onnx.checker.check_model(transformed, full_check=True)
    unchanged = next(node for node in transformed.graph.node if node.name == "node1")
    assert list(unchanged.input) == ["X1", "scale", "zero"]
    for name, output in _run_ort(transformed, feeds).items():
        np.testing.assert_array_equal(output, expected[name])


@pytest.mark.parametrize("op_type", ["QuantizeLinear", "DequantizeLinear"])
def test_untyped_qdq_output_keeps_its_inferred_dtype(op_type):
    """An internal output without value_info remains correctly typed after wrapping."""
    quantize = op_type == "QuantizeLinear"
    input_dtype = TensorProto.FLOAT if quantize else TensorProto.INT8
    output_dtype = TensorProto.INT8 if quantize else TensorProto.FLOAT
    shape = [2, 3]
    model = helper.make_model(
        helper.make_graph(
            [
                helper.make_node(op_type, ["X", "scale", "zero"], ["hidden"], name="qdq"),
                helper.make_node("Identity", ["hidden"], ["Y"]),
            ],
            "untyped_output",
            [_vi("X", input_dtype, shape)],
            [_vi("Y", output_dtype, shape)],
            initializer=[
                numpy_helper.from_array(np.float32(0.25), "scale"),
                numpy_helper.from_array(np.int8(0), "zero"),
            ],
        ),
        opset_imports=[helper.make_opsetid("", 21)],
        ir_version=_MAX_IR_VERSION_FOR_ORT,
    )
    onnx.checker.check_model(model, full_check=True)
    x = np.arange(6, dtype=np.float32 if quantize else np.int8).reshape(shape)
    expected = _run_ort(model, {"X": x})
    transformed = _apply(copy.deepcopy(model))
    onnx.checker.check_model(transformed, full_check=True)
    np.testing.assert_array_equal(_run_ort(transformed, {"X": x})["Y"], expected["Y"])


@pytest.mark.parametrize("op_type", ["QuantizeLinear", "DequantizeLinear"])
def test_blocked_qdq_is_preserved_until_scale_rank_is_supported(op_type):
    """Blocked quantization keeps data and scale ranks consistent by remaining unchanged."""
    quantize = op_type == "QuantizeLinear"
    input_dtype = TensorProto.FLOAT if quantize else TensorProto.INT8
    output_dtype = TensorProto.INT8 if quantize else TensorProto.FLOAT
    shape = [2, 4]
    model = helper.make_model(
        helper.make_graph(
            [helper.make_node(op_type, ["X", "scale", "zero"], ["Y"], axis=1, block_size=2)],
            "blocked",
            [_vi("X", input_dtype, shape)],
            [_vi("Y", output_dtype, shape)],
            initializer=[
                numpy_helper.from_array(np.full((2, 2), 0.25, dtype=np.float32), "scale"),
                numpy_helper.from_array(np.zeros((2, 2), dtype=np.int8), "zero"),
            ],
        ),
        opset_imports=[helper.make_opsetid("", 21)],
        ir_version=_MAX_IR_VERSION_FOR_ORT,
    )
    onnx.checker.check_model(model, full_check=True)
    x = np.arange(8, dtype=np.float32 if quantize else np.int8).reshape(shape)
    expected = ReferenceEvaluator(model).run(None, {"X": x})[0]
    before = model.SerializeToString()
    _apply(model)
    assert model.SerializeToString() == before
    onnx.checker.check_model(model, full_check=True)
    np.testing.assert_array_equal(ReferenceEvaluator(model).run(None, {"X": x})[0], expected)
