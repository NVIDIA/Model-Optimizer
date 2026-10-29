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

"""Structural and ORT parity tests for ``dla_graph_cleanup``.

The combined transform applies seven stages in order:
  1. Replace intermediary Squeeze/Unsqueeze with 4-D Reshape.
  2. Canonicalize eligible non-4D inputs.
  3. Collapse Reshape chains.
  4. Adapt graph-output Reshapes with Squeeze.
  5. Fold constants via onnx-graphsurgeon.
  6. Collapse lossless Cast chains.
  7. Remove unused initializers.

Numerical cases use ORT; float8 and custom-function cases use ONNX ReferenceEvaluator.
"""

from __future__ import annotations

import copy

import numpy as np
import onnx
import onnx_graphsurgeon as gs
import pytest
from onnx import TensorProto, helper, numpy_helper
from onnx.reference import ReferenceEvaluator

from modelopt.onnx.graph_surgery.dla_transforms import dla_graph_cleanup as _mod
from modelopt.onnx.graph_surgery.utils.graph_utils import GraphCache

pytest.importorskip("onnxruntime", reason="onnxruntime required for ORT parity checks")

import onnxruntime as ort
from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference

_OPSET_VERSION = 17
_MAX_IR_VERSION_FOR_ORT = 9


_apply = _mod._apply_graph_cleanup
_fold_consts = _mod._fold_constants
_remove_unused_inits = _mod._remove_unused_initializers


@pytest.mark.parametrize("opset", [12, 17])
@pytest.mark.parametrize("shape", [[2, 3], [2, 3, 4]])
@pytest.mark.parametrize("declared_shape", [False, True])
def test_cleanup_preserves_native_rank_with_unshaped_outputs(opset, shape, declared_shape):
    """Infer the native rank before promotion while retaining the caller's original descriptor."""
    model = helper.make_model(
        helper.make_graph(
            [helper.make_node("Relu", ["x"], ["y"])],
            "unshaped_output",
            [_vi("x", TensorProto.FLOAT, shape)],
            [_vi("y", TensorProto.FLOAT, shape if declared_shape else None)],
        ),
        opset_imports=[helper.make_opsetid("", opset)],
        ir_version=9,
    )
    onnx.checker.check_model(
        onnx.shape_inference.infer_shapes(copy.deepcopy(model)), full_check=True
    )
    feeds = {"x": np.arange(np.prod(shape), dtype=np.float32).reshape(shape) - 2}
    expected = _run_ort(model, feeds)
    signature = model.graph.output[0].SerializeToString()
    transformed = _apply(copy.deepcopy(model))
    assert transformed.graph.output[0].SerializeToString() == signature
    actual = _run_ort(transformed, feeds)
    np.testing.assert_array_equal(actual["y"], expected["y"])
    onnx.checker.check_model(
        onnx.shape_inference.infer_shapes(copy.deepcopy(transformed)), full_check=True
    )


def _canonicalize_inputs(model):
    """Canonicalize model inputs using a fresh graph cache."""
    return _mod._canonicalize_inputs(model, GraphCache(model.graph))


def _replace_sq_unsq(model):
    """Replace eligible Squeeze and Unsqueeze nodes using current shapes."""
    return _mod._replace_intermediary_squeeze_unsqueeze(model, GraphCache(model.graph))


def _remove_reshapes(model):
    """Collapse reshape chains using a fresh graph cache."""
    return _mod._remove_reshape_chains(model, GraphCache(model.graph))


def _reshape_to_squeeze(model):
    """Adapt graph-output reshapes without changing their declared outputs."""
    return _mod._replace_graph_output_reshape_with_squeeze(model, GraphCache(model.graph))


def _remove_cast_chains(model):
    """Collapse eligible cast chains using current graph metadata."""
    return _mod._remove_consecutive_casts(model, GraphCache(model.graph))


# ── Shared utilities ──────────────────────────────────────────────────────────


def _vi(name: str, elem_type: int, shape: list) -> onnx.ValueInfoProto:
    """Build a typed graph value with the declared shape."""
    return helper.make_tensor_value_info(name, elem_type, shape)


def _clamp_ir(model: onnx.ModelProto) -> None:
    """Keep fixtures within ORT's supported IR range."""
    model.ir_version = min(model.ir_version, _MAX_IR_VERSION_FOR_ORT)


def _prepare_model(model: onnx.ModelProto) -> None:
    """Populate intermediate shapes and surface unexpected inference failures."""
    inferred = SymbolicShapeInference.infer_shapes(model)
    if inferred is not None:
        del model.graph.value_info[:]
        model.graph.value_info.extend(inferred.graph.value_info)


def _run_ort(model: onnx.ModelProto, feeds: dict) -> dict:
    """Run all graph outputs with ORT's CPU execution provider."""
    _clamp_ir(model)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"], sess_options=options
    )
    return dict(zip([output.name for output in model.graph.output], sess.run(None, feeds)))


_RNG = np.random.default_rng(2024)


# ══════════════════════════════════════════════════════════════════════════════
# Step 1 — Canonicalize non-4D inputs
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("opset", [12, 17])
@pytest.mark.parametrize("input_shape", [[1], [6], [1, 6], [2, 3, 4]])
def test_canonicalize_input_promotes_safe_branch_and_preserves_outputs(opset, input_shape):
    """Keep 4D interior promotion while preserving both pointwise and Transpose outputs."""
    relu = helper.make_node("Relu", ["X"], ["Y"], name="relu")
    transpose = helper.make_node(
        "Transpose", ["X"], ["Z"], perm=list(reversed(range(len(input_shape))))
    )
    graph = helper.make_graph(
        [relu, transpose],
        "g",
        [_vi("X", TensorProto.FLOAT, input_shape)],
        [
            _vi("Y", TensorProto.FLOAT, input_shape),
            _vi("Z", TensorProto.FLOAT, list(reversed(input_shape))),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    _clamp_ir(model)
    onnx.checker.check_model(model, full_check=True)
    feeds = {"X": _RNG.standard_normal(input_shape).astype(np.float32)}
    expected = _run_ort(model, feeds)

    tr = copy.deepcopy(model)
    tr = _apply(tr)
    onnx.checker.check_model(tr, full_check=True)

    unsq_nodes = [n for n in tr.graph.node if n.op_type == "Unsqueeze"]
    assert len(unsq_nodes) == 1
    unsq = unsq_nodes[0]
    assert unsq.input[0] == "X", "Unsqueeze reads the graph input"

    # Relu must now read the Unsqueeze output, not X directly
    relu_node = next(n for n in tr.graph.node if n.op_type == "Relu")
    assert relu_node.input[0] == unsq.output[0], "Relu must be redirected to Unsqueeze output"
    assert len(GraphCache(tr.graph).get_shape(relu_node.output[0])) == 4
    transpose_node = next(n for n in tr.graph.node if n.op_type == "Transpose")
    assert transpose_node.input[0] == "X"
    actual = _run_ort(tr, feeds)
    assert list(tr.graph.output) == list(model.graph.output)
    np.testing.assert_array_equal(actual["Y"], expected["Y"])
    np.testing.assert_array_equal(actual["Z"], expected["Z"])


@pytest.mark.parametrize("opset", [12, 17])
def test_cleanup_preserves_scalar_output_after_intermediary_promotion(opset):
    """Restore a scalar result after an intermediary Squeeze is promoted to 4D."""
    squeeze = helper.make_node("Squeeze", ["X"], ["middle"], axes=[0, 1])
    initializers = []
    if opset >= 13:
        squeeze.ClearField("attribute")
        squeeze.input.append("axes")
        initializers.append(numpy_helper.from_array(np.array([0, 1], np.int64), name="axes"))
    model = helper.make_model(
        helper.make_graph(
            [squeeze, helper.make_node("Relu", ["middle"], ["Y"])],
            "scalar_boundary",
            [_vi("X", TensorProto.FLOAT, [1, 1])],
            [_vi("Y", TensorProto.FLOAT, [])],
            initializer=initializers,
        ),
        opset_imports=[helper.make_opsetid("", opset)],
    )
    _clamp_ir(model)
    onnx.checker.check_model(model, full_check=True)
    feeds = {"X": np.array([[2]], np.float32)}
    expected = _run_ort(model, feeds)["Y"]
    transformed = _apply(copy.deepcopy(model))
    onnx.checker.check_model(transformed, full_check=True)
    assert list(transformed.graph.output) == list(model.graph.output)
    np.testing.assert_array_equal(_run_ort(transformed, feeds)["Y"], expected)


@pytest.mark.parametrize("opset", [12, 13, 17])
@pytest.mark.parametrize("alias_only", [False, True])
def test_canonicalize_2d_input_unsqueeze_added(opset, alias_only):
    """Promote a 2D input when broadcasting already fixes downstream rank at four."""
    add = helper.make_node("Add", ["X", "W"], ["sum"], name="add")
    transpose = helper.make_node("Transpose", ["sum"], ["Y"], perm=[0, 1, 3, 2])
    graph = helper.make_graph(
        [add, transpose],
        "g",
        [_vi("X", TensorProto.FLOAT, [4, 5])],
        [_vi("Y", TensorProto.FLOAT, [1, 1, 5, 4])],
        [numpy_helper.from_array(np.zeros([1, 1, 4, 5], np.float32), "W")],
        value_info=[_vi("sum", TensorProto.FLOAT, [1, 1, 4, 5])],
    )
    model = helper.make_model(
        graph,
        opset_imports=[
            helper.make_opsetid("test.modelopt", 99),
            helper.make_opsetid("ai.onnx", 17),
            helper.make_opsetid("", opset),
        ],
    )
    if alias_only:
        model.ClearField("opset_import")
        model.opset_import.append(helper.make_opsetid("ai.onnx", opset))
    _clamp_ir(model)
    feeds = {"X": _RNG.standard_normal([4, 5]).astype(np.float32)}
    expected = _run_ort(model, feeds)

    tr = copy.deepcopy(model)
    _prepare_model(tr)
    tr = _canonicalize_inputs(tr)

    unsq_nodes = [n for n in tr.graph.node if n.op_type == "Unsqueeze"]
    assert len(unsq_nodes) == 1
    if opset < 13:
        axes = list(next(attr.ints for attr in unsq_nodes[0].attribute if attr.name == "axes"))
        assert len(unsq_nodes[0].input) == 1
        assert not any(i.name.endswith("_unsqueeze_axes") for i in tr.graph.initializer)
    else:
        axes_init = next(i for i in tr.graph.initializer if i.name == unsq_nodes[0].input[1])
        axes = numpy_helper.to_array(axes_init).tolist()
    assert axes == [0, 1], f"axes should be [0,1] for 2-D input, got {axes}"
    add_node = next(node for node in tr.graph.node if node.op_type == "Add")
    assert add_node.input[0] == unsq_nodes[0].output[0]
    # ONNX 1.21 requires an empty-domain import; ORT also accepts alias-only imports.
    if not alias_only:
        onnx.checker.check_model(tr, full_check=True)
    np.testing.assert_array_equal(_run_ort(tr, feeds)["Y"], expected["Y"])


def test_canonicalize_3d_input_duplicate_unsqueezes_deduped():
    """3-D input consumed by two Unsqueeze nodes → only one canonical Unsqueeze remains.

    Both existing Unsqueezes (same axes) are replaced by the single canonical one.
    """
    axes_init = numpy_helper.from_array(np.array([0], dtype=np.int64), name="ax")
    unsq_a = helper.make_node("Unsqueeze", ["X", "ax"], ["Xa"], name="unsq_a")
    unsq_b = helper.make_node("Unsqueeze", ["X", "ax"], ["Xb"], name="unsq_b")
    relu_a = helper.make_node("Relu", ["Xa"], ["Ya"], name="relu_a")
    relu_b = helper.make_node("Relu", ["Xb"], ["Yb"], name="relu_b")
    graph = helper.make_graph(
        [unsq_a, unsq_b, relu_a, relu_b],
        "g",
        [_vi("X", TensorProto.FLOAT, [2, 3, 4])],
        [_vi("Ya", TensorProto.FLOAT, [1, 2, 3, 4]), _vi("Yb", TensorProto.FLOAT, [1, 2, 3, 4])],
        initializer=[axes_init],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", _OPSET_VERSION)])
    _clamp_ir(model)

    tr = copy.deepcopy(model)
    _prepare_model(tr)
    tr = _canonicalize_inputs(tr)

    unsq_nodes = [n for n in tr.graph.node if n.op_type == "Unsqueeze"]
    # Only the single canonical Unsqueeze should remain; old ones are removed/replaced
    assert len(unsq_nodes) == 1, (
        f"exactly one Unsqueeze should remain after dedup, got {len(unsq_nodes)}"
    )


@pytest.mark.parametrize("axes_input", [False, True])
def test_canonicalize_4d_input_unchanged(axes_input):
    """Keep 4D data inputs and optional Unsqueeze control inputs unchanged."""
    inputs = [_vi("X", TensorProto.FLOAT, [1, 2, 3, 4])]
    output_shape = [1, 2, 3, 4]
    node = helper.make_node("Relu", ["X"], ["Y"], name="relu")
    feeds = {"X": np.arange(24, dtype=np.float32).reshape(1, 2, 3, 4)}
    if axes_input:
        inputs.append(_vi("axes", TensorProto.INT64, [1]))
        node = helper.make_node("Unsqueeze", ["X", "axes"], ["Y"], name="uq")
        output_shape = [1, 1, 2, 3, 4]
        feeds["axes"] = np.array([0], np.int64)
    graph = helper.make_graph(
        [node],
        "g",
        inputs,
        [_vi("Y", TensorProto.FLOAT, output_shape)],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", _OPSET_VERSION)])
    _clamp_ir(model)
    onnx.checker.check_model(model, full_check=True)
    expected = _run_ort(model, feeds)
    before = model.SerializeToString()
    tr = _canonicalize_inputs(copy.deepcopy(model))
    assert tr.SerializeToString() == before
    onnx.checker.check_model(tr, full_check=True)
    np.testing.assert_array_equal(_run_ort(tr, feeds)["Y"], expected["Y"])


# ══════════════════════════════════════════════════════════════════════════════
# Step 2 — Replace intermediary Squeeze / Unsqueeze with 4-D Reshape
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("stage", ["input", "squeeze", "unsqueeze"])
@pytest.mark.parametrize(
    "consumer", ["Transpose", "Softmax", "MatMul", "transpose_via_relu", "prelu_slope"]
)
def test_rank_promotion_preserves_rank_sensitive_consumers(stage, consumer):
    """Preserve rank-sensitive behavior, including consumers behind an elementwise node."""
    input_shape = [1, 2, 3] if stage == "squeeze" else [2, 3]
    source_shape = [1, 2, 3] if stage == "unsqueeze" else [2, 3]
    nodes, initializers, values = [], [], []
    source = "X"
    if stage != "input":
        initializers.append(numpy_helper.from_array(np.array([0], np.int64), "axes"))
        if stage == "unsqueeze":
            nodes.append(helper.make_node("Identity", ["X"], ["middle"]))
            values.append(_vi("middle", TensorProto.FLOAT, input_shape))
            source = "middle"
        nodes.append(helper.make_node(stage.title(), [source, "axes"], ["adapted"]))
        source = "adapted"
        values.append(_vi(source, TensorProto.FLOAT, source_shape))
    if consumer == "transpose_via_relu":
        nodes.append(helper.make_node("Relu", [source], ["activated"]))
        source = "activated"
        values.append(_vi(source, TensorProto.FLOAT, source_shape))
        consumer = "Transpose"
    if consumer == "Transpose":
        nodes.append(
            helper.make_node(
                consumer, [source], ["Y"], perm=list(reversed(range(len(source_shape))))
            )
        )
        output_shape = list(reversed(source_shape))
    elif consumer == "Softmax":
        nodes.append(helper.make_node(consumer, [source], ["Y"], axis=len(source_shape) - 2))
        output_shape = source_shape
    elif consumer == "prelu_slope":
        initializers.append(
            numpy_helper.from_array(
                -np.arange(1, np.prod(source_shape) + 1, dtype=np.float32).reshape(source_shape),
                "data",
            )
        )
        nodes.append(helper.make_node("PRelu", ["data", source], ["Y"]))
        output_shape = source_shape
    else:
        initializers.append(
            numpy_helper.from_array(np.arange(12, dtype=np.float32).reshape(3, 4), "W")
        )
        nodes.append(helper.make_node(consumer, [source, "W"], ["Y"]))
        output_shape = [*source_shape[:-1], 4]
    graph = helper.make_graph(
        nodes,
        "rank_sensitive",
        [_vi("X", TensorProto.FLOAT, input_shape)],
        [_vi("Y", TensorProto.FLOAT, output_shape)],
        initializers,
        value_info=values,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", _OPSET_VERSION)])
    _clamp_ir(model)
    feeds = {"X": np.arange(np.prod(input_shape), dtype=np.float32).reshape(input_shape)}
    expected = _run_ort(model, feeds)
    before = model.SerializeToString()
    transformed = copy.deepcopy(model)
    transformed = (
        _canonicalize_inputs(transformed) if stage == "input" else _replace_sq_unsq(transformed)
    )
    onnx.checker.check_model(transformed, full_check=True)
    np.testing.assert_array_equal(_run_ort(transformed, feeds)["Y"], expected["Y"])
    assert transformed.SerializeToString() == before


@pytest.mark.parametrize("stage", ["canonical", "squeeze"])
@pytest.mark.parametrize("names", ["empty", "duplicate", "existing"])
def test_generated_reshape_targets_do_not_collide(stage, names):
    """Keep distinct reshape targets for unnamed nodes and existing tensor names."""
    prefix = "uq" if stage == "canonical" else "sq"
    first_name = "" if names == "empty" else prefix
    second_name = prefix + "_other" if names == "existing" else first_name
    initializers = []
    if stage == "canonical":
        initializers.extend(
            [
                numpy_helper.from_array(np.array([2, 3], np.int64), "axes1"),
                numpy_helper.from_array(np.array([1, 3], np.int64), "axes2"),
            ]
        )
        nodes = [
            helper.make_node("Unsqueeze", ["X", "axes1"], ["Y1"], name=first_name),
            helper.make_node("Unsqueeze", ["X", "axes2"], ["Y2"], name=second_name),
        ]
        inputs = [_vi("X", TensorProto.FLOAT, [2, 3])]
        outputs = [
            _vi("Y1", TensorProto.FLOAT, [2, 3, 1, 1]),
            _vi("Y2", TensorProto.FLOAT, [2, 1, 3, 1]),
        ]
        feeds = {"X": np.arange(6, dtype=np.float32).reshape(2, 3)}
        suffix = "_reshape_shape"
    else:
        initializers.append(numpy_helper.from_array(np.array([0], np.int64), "axes"))
        nodes = [
            helper.make_node("Squeeze", ["X1", "axes"], ["Y1"], name=first_name),
            helper.make_node("Squeeze", ["X2", "axes"], ["Y2"], name=second_name),
        ]
        inputs = [_vi("X1", TensorProto.FLOAT, [1, 2, 3]), _vi("X2", TensorProto.FLOAT, [1, 3, 2])]
        outputs = [_vi("Y1", TensorProto.FLOAT, [2, 3]), _vi("Y2", TensorProto.FLOAT, [3, 2])]
        feeds = {
            "X1": np.arange(6, dtype=np.float32).reshape(1, 2, 3),
            "X2": np.arange(6, dtype=np.float32).reshape(1, 3, 2),
        }
        suffix = "_4d_shape"
    if names == "existing":
        initializers.append(numpy_helper.from_array(np.array([99], np.int64), prefix + suffix))
    model = helper.make_model(
        helper.make_graph(nodes, "target_names", inputs, outputs, initializers),
        opset_imports=[helper.make_opsetid("", _OPSET_VERSION)],
    )
    _clamp_ir(model)
    onnx.checker.check_model(model, full_check=True)
    reference = model
    if names == "duplicate":
        # ORT rejects duplicate labels; normalize only the reference fixture.
        reference = copy.deepcopy(model)
        for index, node in enumerate(reference.graph.node):
            node.name = f"reference_{index}"
    expected = _run_ort(reference, feeds)
    transformed = copy.deepcopy(model)
    transformed = (
        _canonicalize_inputs(transformed) if stage == "canonical" else _replace_sq_unsq(transformed)
    )
    onnx.checker.check_model(transformed, full_check=True)
    actual = _run_ort(transformed, feeds)
    for name, value in expected.items():
        np.testing.assert_array_equal(actual[name], value)


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_canonicalize_preserves_graph_output_shape(axis):
    """Preserve a non-4D graph-output Unsqueeze's declared shape and values."""
    target = [2, 3]
    target.insert(axis, 1)
    axes = numpy_helper.from_array(np.array([axis], np.int64), "axes")
    model = helper.make_model(
        helper.make_graph(
            [helper.make_node("Unsqueeze", ["X", "axes"], ["Y"], name="uq")],
            "output_boundary",
            [_vi("X", TensorProto.FLOAT, [2, 3])],
            [_vi("Y", TensorProto.FLOAT, target)],
            [axes],
        ),
        opset_imports=[helper.make_opsetid("", _OPSET_VERSION)],
    )
    _clamp_ir(model)
    feeds = {"X": np.arange(6, dtype=np.float32).reshape(2, 3)}
    expected = _run_ort(model, feeds)
    transformed = _canonicalize_inputs(copy.deepcopy(model))
    assert transformed.graph.output[0].type == model.graph.output[0].type
    onnx.checker.check_model(transformed, full_check=True)
    np.testing.assert_array_equal(_run_ort(transformed, feeds)["Y"], expected["Y"])


@pytest.mark.parametrize("opset", [12, 17])
@pytest.mark.parametrize("axes", [[1], [-3]])
def test_squeeze_fallback_uses_selected_axes(opset, axes):
    """Infer only selected squeeze axes when intermediate shape metadata is absent."""
    initializers = []
    if opset < 13:
        squeeze = helper.make_node("Squeeze", ["X"], ["middle"], axes=axes, name="sq")
    else:
        initializers.append(numpy_helper.from_array(np.array(axes, np.int64), "axes"))
        squeeze = helper.make_node("Squeeze", ["X", "axes"], ["middle"], name="sq")
    model = helper.make_model(
        helper.make_graph(
            [squeeze, helper.make_node("Identity", ["middle"], ["Y"])],
            "axes_fallback",
            [_vi("X", TensorProto.FLOAT, [2, 1, 1, 3])],
            [_vi("Y", TensorProto.FLOAT, [2, 1, 3])],
            initializers,
        ),
        opset_imports=[helper.make_opsetid("", opset)],
    )
    _clamp_ir(model)
    feeds = {"X": np.arange(6, dtype=np.float32).reshape(2, 1, 1, 3)}
    expected = _run_ort(model, feeds)
    transformed = _replace_sq_unsq(copy.deepcopy(model))
    reshape = next(node for node in transformed.graph.node if node.op_type == "Reshape")
    target = next(init for init in transformed.graph.initializer if init.name == reshape.input[1])
    assert numpy_helper.to_array(target).tolist() == [1, 2, 1, 3]
    transformed = _mod.infer_shapes(transformed)
    onnx.checker.check_model(transformed, full_check=True)
    np.testing.assert_array_equal(
        _run_ort(transformed, feeds)["Y"], expected["Y"].reshape(1, 2, 1, 3)
    )


@pytest.mark.parametrize("axes_mode", ["dynamic", "omitted"])
def test_squeeze_fallback_distinguishes_unknown_and_omitted_axes(axes_mode):
    """Retain unresolved axes and preserve the remove-all fallback for omitted axes."""
    inputs = [_vi("X", TensorProto.FLOAT, [1, 2, 1, 3])]
    node_inputs = ["X"]
    if axes_mode == "dynamic":
        inputs.append(_vi("axes", TensorProto.INT64, [1]))
        node_inputs.append("axes")
    squeeze = helper.make_node("Squeeze", node_inputs, ["middle"], name="sq")
    output_shape = [None, None, None] if axes_mode == "dynamic" else [2, 3]
    model = helper.make_model(
        helper.make_graph(
            [squeeze, helper.make_node("Relu", ["middle"], ["Y"])],
            "axes_resolution",
            inputs,
            [_vi("Y", TensorProto.FLOAT, output_shape)],
        ),
        opset_imports=[helper.make_opsetid("", 17)],
    )
    _clamp_ir(model)
    onnx.checker.check_model(model, full_check=True)
    transformed = _replace_sq_unsq(copy.deepcopy(model))
    if axes_mode == "dynamic":
        assert transformed.SerializeToString() == model.SerializeToString()
    else:
        transformed = _mod.infer_shapes(transformed)
    onnx.checker.check_model(transformed, full_check=True)
    for axes in ([0], [2]) if axes_mode == "dynamic" else [None]:
        feeds = {"X": np.arange(6, dtype=np.float32).reshape(1, 2, 1, 3)}
        if axes is not None:
            feeds["axes"] = np.array(axes, np.int64)
        expected = _run_ort(model, feeds)["Y"]
        if axes_mode != "dynamic":
            expected = expected.reshape(1, 1, 2, 3)
        np.testing.assert_array_equal(_run_ort(transformed, feeds)["Y"], expected)


def test_intermediary_squeeze_replaced_with_reshape():
    """Squeeze [2,1,6,8] → [2,6,8] (axis=1) is replaced by Reshape → [1,2,6,8].

    pad([2,1,6,8]) = [2,1,6,8], pad([2,6,8]) = [1,2,6,8] — shapes differ → Reshape inserted.
    The squeezed axis must be 1 (a unit dimension) for ORT symbolic shape inference to accept it.
    """
    sq_axes = numpy_helper.from_array(np.array([1], dtype=np.int64), name="sq_ax")
    relu = helper.make_node("Relu", ["X"], ["X_relu"], name="relu")
    squeeze = helper.make_node("Squeeze", ["X_relu", "sq_ax"], ["Sq_out"], name="sq")
    relu2 = helper.make_node("Relu", ["Sq_out"], ["Y"], name="relu2")
    graph = helper.make_graph(
        [relu, squeeze, relu2],
        "g",
        [_vi("X", TensorProto.FLOAT, [2, 1, 6, 8])],
        [_vi("Y", TensorProto.FLOAT, [2, 6, 8])],
        initializer=[sq_axes],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", _OPSET_VERSION)])
    _clamp_ir(model)

    tr = copy.deepcopy(model)
    _prepare_model(tr)
    tr = _replace_sq_unsq(tr)

    assert not any(n.op_type == "Squeeze" for n in tr.graph.node), "Squeeze must be removed"
    reshape_nodes = [n for n in tr.graph.node if n.op_type == "Reshape"]
    assert reshape_nodes, "a Reshape must be inserted"

    # Shape init name follows pattern {node.name}_4d_shape → "sq_4d_shape"
    shape_init = next((i for i in tr.graph.initializer if i.name == "sq_4d_shape"), None)
    assert shape_init is not None, "Reshape shape initializer must exist"
    shape = numpy_helper.to_array(shape_init).tolist()
    assert shape == [1, 2, 6, 8], f"expected [1,2,6,8], got {shape}"


@pytest.mark.parametrize(
    ("axes", "output_shape"),
    [
        ([2], [2, 4, 1, 6]),
        ([-1, -3], [2, 4, 1, 6, 1]),
        ([2, -1], [2, 4, 1, 6, 1]),
        ([-5, -3], [1, 2, 1, 4, 6]),
    ],
)
def test_intermediary_unsqueeze_replaced_with_reshape(axes, output_shape):
    """Preserve multi-axis Unsqueeze layout when output shape metadata is absent."""
    unsq_axes = numpy_helper.from_array(np.array(axes, dtype=np.int64), name="unsq_ax")
    relu = helper.make_node("Relu", ["X"], ["X_relu"], name="relu")
    unsqueeze = helper.make_node("Unsqueeze", ["X_relu", "unsq_ax"], ["Uq_out"], name="uq")
    relu2 = helper.make_node("Relu", ["Uq_out"], ["Y"], name="relu2")
    graph = helper.make_graph(
        [relu, unsqueeze, relu2],
        "g",
        [_vi("X", TensorProto.FLOAT, [2, 4, 6])],
        [_vi("Y", TensorProto.FLOAT, output_shape)],
        initializer=[unsq_axes],
        value_info=[_vi("X_relu", TensorProto.FLOAT, [2, 4, 6])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", _OPSET_VERSION)])
    _clamp_ir(model)
    feeds = {"X": _RNG.standard_normal([2, 4, 6]).astype(np.float32)}
    expected = _run_ort(model, feeds)

    tr = copy.deepcopy(model)
    tr = _replace_sq_unsq(tr)

    assert not any(n.op_type == "Unsqueeze" for n in tr.graph.node), "Unsqueeze must be removed"
    assert any(n.op_type == "Reshape" for n in tr.graph.node), "Reshape must be inserted"
    # Shape init name: {node.name}_4d_shape → "uq_4d_shape"
    shape_init = next((i for i in tr.graph.initializer if i.name == "uq_4d_shape"), None)
    assert shape_init is not None, "Reshape shape initializer must exist"
    shape = numpy_helper.to_array(shape_init).tolist()
    assert shape == output_shape
    onnx.checker.check_model(tr, full_check=True)
    np.testing.assert_array_equal(_run_ort(tr, feeds)["Y"], expected["Y"])


def test_intermediary_squeeze_at_graph_output_replaced():
    """Squeeze whose output IS a graph output is now replaced like any other intermediary Squeeze.

    Squeeze [1,2,1,5] → [1,2,5] (axis=2): pad([1,2,1,5])=[1,2,1,5] ≠ pad([1,2,5])=[1,1,2,5]
    so a Reshape is inserted and the Squeeze is removed.
    """
    sq_axes = numpy_helper.from_array(np.array([2], dtype=np.int64), name="sq_ax2")
    relu = helper.make_node("Relu", ["X"], ["X_relu"], name="relu")
    squeeze = helper.make_node("Squeeze", ["X_relu", "sq_ax2"], ["Y"], name="sq")
    graph = helper.make_graph(
        [relu, squeeze],
        "g",
        [_vi("X", TensorProto.FLOAT, [1, 2, 1, 5])],
        [_vi("Y", TensorProto.FLOAT, [1, 2, 5])],
        initializer=[sq_axes],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", _OPSET_VERSION)])
    _clamp_ir(model)

    tr = copy.deepcopy(model)
    _prepare_model(tr)
    tr = _replace_sq_unsq(tr)

    assert not any(n.op_type == "Squeeze" for n in tr.graph.node), (
        "Squeeze at graph output must now be replaced"
    )
    assert any(n.op_type == "Reshape" for n in tr.graph.node), "Reshape must be inserted"


def test_intermediary_squeeze_identity_at_graph_output_uses_reshape():
    """Squeeze identity in 4-D AND graph output → Reshape with graph output shape.

    Squeeze [1,1,5] → [1,5] (axis=0): padded shapes are identical so normally the node
    would be removed.  Because Y is a graph output, a Reshape is inserted instead and its
    target shape must equal the declared graph output shape [1,5] — not the 4D padded shape.
    """
    sq_axes = numpy_helper.from_array(np.array([0], dtype=np.int64), name="sq_ax_id")
    relu = helper.make_node("Relu", ["X"], ["X_relu"], name="relu")
    squeeze = helper.make_node("Squeeze", ["X_relu", "sq_ax_id"], ["Y"], name="sq_id")
    graph = helper.make_graph(
        [relu, squeeze],
        "g",
        [_vi("X", TensorProto.FLOAT, [1, 1, 5])],
        [_vi("Y", TensorProto.FLOAT, [1, 5])],
        initializer=[sq_axes],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", _OPSET_VERSION)])
    _clamp_ir(model)

    tr = copy.deepcopy(model)
    _prepare_model(tr)
    tr = _replace_sq_unsq(tr)

    assert not any(n.op_type == "Squeeze" for n in tr.graph.node), (
        "Squeeze must be replaced even for identity case at graph output"
    )
    assert any(n.op_type == "Reshape" for n in tr.graph.node), (
        "Reshape must be inserted to preserve the graph output name"
    )

    # Reshape target shape must equal the graph output shape [1,5], not 4D padded [1,1,1,5]
    rs_shape_init = next((i for i in tr.graph.initializer if i.name == "sq_id_4d_shape"), None)
    assert rs_shape_init is not None, "Reshape shape initializer must exist"
    assert numpy_helper.to_array(rs_shape_init).tolist() == [1, 5], (
        "Reshape target must equal graph output shape [1,5]"
    )

    assert any(o.name == "Y" for o in tr.graph.output), "Y must remain a graph output"


def test_unsqueeze_on_graph_input_skipped():
    """Unsqueeze whose data input IS a graph input must not be replaced.

    This Unsqueeze is the canonical 4D-promotion node placed by convert_ops_to_4d
    or canonicalize_inputs.  Replacing it with a Reshape would break the graph
    because the input tensor shape stays 3D but the Reshape would expect a 4D input.
    """
    unsq_axes = numpy_helper.from_array(np.array([0], dtype=np.int64), name="gi_unsq_ax")
    # Unsqueeze directly on graph input X [2,4,6] → [1,2,4,6]
    unsqueeze = helper.make_node("Unsqueeze", ["X", "gi_unsq_ax"], ["X_4d"], name="gi_unsq")
    relu = helper.make_node("Relu", ["X_4d"], ["Y"], name="relu")
    graph = helper.make_graph(
        [unsqueeze, relu],
        "g",
        [_vi("X", TensorProto.FLOAT, [2, 4, 6])],
        [_vi("Y", TensorProto.FLOAT, [1, 2, 4, 6])],
        initializer=[unsq_axes],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", _OPSET_VERSION)])
    _clamp_ir(model)
    orig_node_count = len(model.graph.node)

    tr = copy.deepcopy(model)
    _prepare_model(tr)
    tr = _replace_sq_unsq(tr)

    # Unsqueeze on graph input must be preserved exactly as-is
    assert any(n.op_type == "Unsqueeze" for n in tr.graph.node), (
        "Unsqueeze on graph input must be preserved"
    )
    assert not any(n.op_type == "Reshape" for n in tr.graph.node), (
        "No Reshape must be inserted for graph-input Unsqueeze"
    )
    assert len(tr.graph.node) == orig_node_count, "Node count must be unchanged"


# ══════════════════════════════════════════════════════════════════════════════
# Step 3 — Collapse Reshape chains
# ══════════════════════════════════════════════════════════════════════════════


def _make_reshape_graph(shapes: list[list[int]]) -> onnx.ModelProto:
    """Build a graph: X → Reshape1 → Reshape2 → … → Y, each using a fixed shape."""
    assert len(shapes) >= 2
    inits = []
    nodes = []
    prev_name = "X"
    for i, shape in enumerate(shapes[1:], 1):
        sh_init = numpy_helper.from_array(np.array(shape, dtype=np.int64), name=f"sh{i}")
        inits.append(sh_init)
        out_name = "Y" if i == len(shapes) - 1 else f"T{i}"
        nodes.append(helper.make_node("Reshape", [prev_name, f"sh{i}"], [out_name], name=f"rs{i}"))
        prev_name = out_name
    graph = helper.make_graph(
        nodes,
        "g",
        [_vi("X", TensorProto.FLOAT, shapes[0])],
        [_vi("Y", TensorProto.FLOAT, shapes[-1])],
        initializer=inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", _OPSET_VERSION)])
    _clamp_ir(model)
    return model


def test_reshape_chain_noop_removed():
    """X[1,2,3,4] → Reshape[1,2,3,4] → Reshape[1,2,3,4] → Y: both removed, ORT parity."""
    model = _make_reshape_graph([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]])
    x = _RNG.standard_normal([1, 2, 3, 4]).astype(np.float32)

    _prepare_model(model)
    ref = _run_ort(model, {"X": x})

    tr = copy.deepcopy(model)
    _prepare_model(tr)
    tr = _remove_reshapes(tr)

    reshape_nodes = [n for n in tr.graph.node if n.op_type == "Reshape"]
    assert len(reshape_nodes) == 0, "identity-reshape chain must be fully removed"

    _prepare_model(tr)
    got = _run_ort(tr, {"X": x})
    np.testing.assert_array_equal(ref["Y"], got["Y"])


def test_reshape_chain_collapsed_to_one():
    """X[6] → Reshape[2,3] → Reshape[1,1,2,3] → Y: collapsed to single Reshape, ORT parity."""
    model = _make_reshape_graph([[6], [2, 3], [1, 1, 2, 3]])
    x = _RNG.standard_normal([6]).astype(np.float32)

    _prepare_model(model)
    ref = _run_ort(model, {"X": x})

    tr = copy.deepcopy(model)
    _prepare_model(tr)
    tr = _remove_reshapes(tr)

    reshape_nodes = [n for n in tr.graph.node if n.op_type == "Reshape"]
    assert len(reshape_nodes) == 1, "two-step chain must collapse to one Reshape"

    _prepare_model(tr)
    got = _run_ort(tr, {"X": x})
    np.testing.assert_array_equal(ref["Y"], got["Y"])


def test_single_reshape_unchanged():
    """A single Reshape is not touched."""
    model = _make_reshape_graph([[6], [2, 3]])
    orig_count = len(model.graph.node)

    tr = copy.deepcopy(model)
    _prepare_model(tr)
    tr = _remove_reshapes(tr)

    assert len(tr.graph.node) == orig_count, "single Reshape must not be changed"


@pytest.mark.parametrize(
    "kind",
    ["reshape_noop", "reshape_chain", "reshape_noop_chain", "cast_chain", "unsqueeze", "control"],
)
def test_cleanup_preserves_subgraph_captures(kind):
    """Keep chain values alive when If branches capture them, including direct controls."""
    dtype = TensorProto.BOOL if kind == "control" else TensorProto.FLOAT
    input_shape = [] if kind == "control" else [6]
    target_shape = input_shape
    captured = "Y"
    captured_type = dtype
    initializers = []
    value_info = []
    if kind == "cast_chain":
        nodes = [
            helper.make_node("Cast", ["X"], ["T"], to=TensorProto.DOUBLE),
            helper.make_node("Cast", ["T"], ["Y"], to=TensorProto.FLOAT),
        ]
        captured = "T"
        captured_type = TensorProto.DOUBLE
        value_info.append(_vi("T", captured_type, input_shape))
    elif kind == "unsqueeze":
        input_shape = [2, 3]
        target_shape = [1, 1, 2, 3]
        initializers.append(numpy_helper.from_array(np.array([0, 1], np.int64), name="axes"))
        nodes = [helper.make_node("Unsqueeze", ["X", "axes"], ["Y"])]
    else:
        shapes = [[2, 3], [6]] if "chain" in kind else [input_shape]
        if kind == "reshape_chain":
            shapes[-1] = [1, 2, 3]
            captured = "T"
        nodes = []
        previous = "X"
        for index, shape in enumerate(shapes):
            name = f"shape_{index}"
            initializers.append(numpy_helper.from_array(np.array(shape, np.int64), name=name))
            output = "Y" if index == len(shapes) - 1 else "T"
            nodes.append(helper.make_node("Reshape", [previous, name], [output]))
            value_info.append(_vi(output, dtype, shape))
            previous = output
        target_shape = shapes[-1]
    captured_shape = [2, 3] if captured == "T" and kind != "cast_chain" else target_shape
    if kind in ("cast_chain", "unsqueeze"):
        value_info.append(_vi("Y", dtype, target_shape))
    inputs = [_vi("X", dtype, input_shape)]
    condition = "Y" if kind == "control" else "condition"
    if kind != "control":
        inputs.append(_vi(condition, TensorProto.BOOL, []))
    branch = helper.make_graph(
        [helper.make_node("Identity", [captured], ["branch_out"])],
        "capture",
        [],
        [_vi("branch_out", captured_type, captured_shape)],
    )
    nodes.extend(
        [
            helper.make_node("Identity", ["Y"], ["main"]),
            helper.make_node(
                "If", [condition], ["captured"], then_branch=branch, else_branch=branch
            ),
        ]
    )
    model = helper.make_model(
        helper.make_graph(
            nodes,
            "captured_chain",
            inputs,
            [_vi("main", dtype, target_shape), _vi("captured", captured_type, captured_shape)],
            initializer=initializers,
            value_info=value_info,
        ),
        opset_imports=[helper.make_opsetid("", _OPSET_VERSION)],
    )
    _clamp_ir(model)
    onnx.checker.check_model(model, full_check=True)
    feeds = {
        "X": np.array(True)
        if kind == "control"
        else np.arange(6, dtype=np.float32).reshape(input_shape)
    }
    if kind != "control":
        feeds[condition] = np.array(True)
    expected = _run_ort(model, feeds)
    transform = _remove_cast_chains if kind == "cast_chain" else _remove_reshapes
    if kind == "unsqueeze":
        transform = _canonicalize_inputs
    transformed = transform(copy.deepcopy(model))
    onnx.checker.check_model(transformed, full_check=True)
    actual = _run_ort(transformed, feeds)
    for name in expected:
        np.testing.assert_array_equal(actual[name], expected[name])


# ══════════════════════════════════════════════════════════════════════════════
# Step 4 — Fold constants
# ══════════════════════════════════════════════════════════════════════════════


def test_fold_constants_const_add_folded():
    """Constant + Add → output is a constant: the Add is folded away.

    After folding, the graph should have fewer nodes and ORT output must match.
    """
    a_init = numpy_helper.from_array(np.array([1.0, 2.0, 3.0], dtype=np.float32), name="A")
    b_init = numpy_helper.from_array(np.array([4.0, 5.0, 6.0], dtype=np.float32), name="B")
    add = helper.make_node("Add", ["A", "B"], ["Y"], name="add")
    graph = helper.make_graph(
        [add],
        "g",
        [],
        [_vi("Y", TensorProto.FLOAT, [3])],
        initializer=[a_init, b_init],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", _OPSET_VERSION)])
    _clamp_ir(model)

    _prepare_model(model)
    ref = _run_ort(model, {})

    tr = _fold_consts(copy.deepcopy(model))
    _prepare_model(tr)
    got = _run_ort(tr, {})

    # After folding the Add node should be gone (its result is a new initializer)
    assert not any(n.op_type == "Add" for n in tr.graph.node), "Add must be folded away"
    np.testing.assert_array_equal(ref["Y"], got["Y"])


def test_fold_constants_dynamic_input_not_folded():
    """Add with a dynamic input is not folded."""
    a_init = numpy_helper.from_array(np.array([1.0, 2.0, 3.0], dtype=np.float32), name="A")
    add = helper.make_node("Add", ["X", "A"], ["Y"], name="add")
    graph = helper.make_graph(
        [add],
        "g",
        [_vi("X", TensorProto.FLOAT, [3])],
        [_vi("Y", TensorProto.FLOAT, [3])],
        initializer=[a_init],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", _OPSET_VERSION)])
    _clamp_ir(model)
    orig_count = len(model.graph.node)

    tr = _fold_consts(copy.deepcopy(model))

    assert len(tr.graph.node) == orig_count, "dynamic Add must not be folded"

    x = _RNG.standard_normal([3]).astype(np.float32)
    _prepare_model(model)
    ref = _run_ort(model, {"X": x})
    _prepare_model(tr)
    got = _run_ort(tr, {"X": x})
    np.testing.assert_array_equal(ref["Y"], got["Y"])


# ══════════════════════════════════════════════════════════════════════════════
# Step 5 — Remove unused initializers
# ══════════════════════════════════════════════════════════════════════════════


def test_remove_unused_initializers():
    """Initializer not referenced by any node is dropped; ORT parity preserved."""
    used = numpy_helper.from_array(np.array([1.0, 2.0], dtype=np.float32), name="used")
    orphan = numpy_helper.from_array(np.array([99.0], dtype=np.float32), name="orphan")
    relu = helper.make_node("Relu", ["used"], ["Y"], name="relu")
    graph = helper.make_graph(
        [relu],
        "g",
        [],
        [_vi("Y", TensorProto.FLOAT, [2])],
        initializer=[used, orphan],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", _OPSET_VERSION)])
    _clamp_ir(model)

    _prepare_model(model)
    ref = _run_ort(model, {})

    tr = copy.deepcopy(model)
    tr = _remove_unused_inits(tr)

    init_names = {i.name for i in tr.graph.initializer}
    assert "used" in init_names, "'used' initializer must be kept"
    assert "orphan" not in init_names, "'orphan' initializer must be removed"

    _prepare_model(tr)
    got = _run_ort(tr, {})
    np.testing.assert_array_equal(ref["Y"], got["Y"])


def test_remove_unused_initializers_all_used_unchanged():
    """When all initializers are in use, none are removed."""
    a = numpy_helper.from_array(np.array([1.0, 2.0], dtype=np.float32), name="A")
    b = numpy_helper.from_array(np.array([3.0, 4.0], dtype=np.float32), name="B")
    add = helper.make_node("Add", ["A", "B"], ["Y"], name="add")
    graph = helper.make_graph(
        [add],
        "g",
        [],
        [_vi("Y", TensorProto.FLOAT, [2])],
        initializer=[a, b],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", _OPSET_VERSION)])
    _clamp_ir(model)
    orig_count = len(model.graph.initializer)

    tr = copy.deepcopy(model)
    tr = _remove_unused_inits(tr)

    assert len(tr.graph.initializer) == orig_count, "no initializers should be removed"


# ══════════════════════════════════════════════════════════════════════════════
# Combined: noop graph → all steps are noops, ORT parity
# ══════════════════════════════════════════════════════════════════════════════


def test_combined_noop_graph():
    """4-D input, single Relu, no Reshape chains, no constants → combined transform is a noop."""
    relu = helper.make_node("Relu", ["X"], ["Y"], name="relu")
    graph = helper.make_graph(
        [relu],
        "g",
        [_vi("X", TensorProto.FLOAT, [1, 2, 3, 4])],
        [_vi("Y", TensorProto.FLOAT, [1, 2, 3, 4])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", _OPSET_VERSION)])
    _clamp_ir(model)
    orig_count = len(model.graph.node)
    orig_inits = len(model.graph.initializer)

    x = _RNG.standard_normal([1, 2, 3, 4]).astype(np.float32)
    _prepare_model(model)
    ref = _run_ort(model, {"X": x})

    tr = copy.deepcopy(model)
    _prepare_model(tr)
    tr = _apply(tr)
    _prepare_model(tr)
    got = _run_ort(tr, {"X": x})

    assert len(tr.graph.node) == orig_count, "noop: node count unchanged"
    assert len(tr.graph.initializer) == orig_inits, "noop: initializer count unchanged"
    np.testing.assert_array_equal(ref["Y"], got["Y"])


# ══════════════════════════════════════════════════════════════════════════════
# Combined: Reshape chain + orphan init + constant fold
# ══════════════════════════════════════════════════════════════════════════════


def test_combined_reshape_and_unused_init_cleaned():
    """Combined: Reshape chain (noop) removed + orphan init removed, ORT parity."""
    x_data = _RNG.standard_normal([1, 2, 3, 4]).astype(np.float32)
    orphan = numpy_helper.from_array(np.array([0.0], dtype=np.float32), name="orphan2")
    sh1 = numpy_helper.from_array(np.array([1, 2, 3, 4], dtype=np.int64), name="sh1")
    sh2 = numpy_helper.from_array(np.array([1, 2, 3, 4], dtype=np.int64), name="sh2")

    rs1 = helper.make_node("Reshape", ["X", "sh1"], ["T1"], name="rs1")
    rs2 = helper.make_node("Reshape", ["T1", "sh2"], ["Y"], name="rs2")
    graph = helper.make_graph(
        [rs1, rs2],
        "g",
        [_vi("X", TensorProto.FLOAT, [1, 2, 3, 4])],
        [_vi("Y", TensorProto.FLOAT, [1, 2, 3, 4])],
        initializer=[orphan, sh1, sh2],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", _OPSET_VERSION)])
    _clamp_ir(model)

    _prepare_model(model)
    ref = _run_ort(model, {"X": x_data})

    tr = copy.deepcopy(model)
    _prepare_model(tr)
    tr = _apply(tr)
    _prepare_model(tr)
    got = _run_ort(tr, {"X": x_data})

    reshape_nodes = [n for n in tr.graph.node if n.op_type == "Reshape"]
    assert len(reshape_nodes) == 0, "identity-Reshape chain must be fully removed"
    assert "orphan2" not in {i.name for i in tr.graph.initializer}, "orphan init must be removed"
    np.testing.assert_array_equal(ref["Y"], got["Y"])


@pytest.mark.parametrize("names", ["named", "empty", "duplicate", "existing"])
def test_interleaved_reshape_chains_collapsed(names):
    """Two Reshape chains interleaved in node order are both collapsed correctly.

    Graph topology:
        X1 [1,2,4,4] --r1--> [1,4,2,4] --r2--> [2,4,4,1] --\\
                                                               Add --> Y [2,4,4,1]
        X2 [1,2,4,4] --r4--> [1,4,2,4] --r5--> [2,4,4,1] --/

    Node sequence in the graph: [r1, r4, r2, r5, add]
    (r4 is interleaved between r1 and r2 in the node list)

    The old sequential algorithm misses both chains:
    - Sees r1, then r4 (not consuming T1) → flushes singleton [r1], starts [r4]
    - Then r2 (not consuming T4) → flushes singleton [r4], starts [r2]
    - Then r5 (not consuming T2) → flushes singleton [r2], starts [r5]
    - End: flushes singleton [r5]
    → All 4 Reshape nodes remain unchanged.

    The new output-to-chain map algorithm finds both chains:
    - r1 → {T1: [r1]};  r4 → {T1:[r1], T4:[r4]}
    - r2 input=T1 in map → {T4:[r4], T2:[r1,r2]}
    - r5 input=T4 in map → {T2:[r1,r2], T5:[r4,r5]}
    - Flush [r1,r2]: in=[1,2,4,4] != out=[2,4,4,1] → collapse to 1 Reshape
    - Flush [r4,r5]: same → collapse to 1 Reshape
    → 2 Reshape nodes remain (one per chain), ORT parity.
    """
    # All shapes are 4D; chain input [1,2,4,4] != chain output [2,4,4,1]
    # so each 2-node chain collapses to a single Reshape (not removed as noop).
    sh_a = numpy_helper.from_array(np.array([1, 4, 2, 4], dtype=np.int64), name="sh_a")
    sh_b = numpy_helper.from_array(np.array([2, 4, 4, 1], dtype=np.int64), name="sh_b")
    sh_c = numpy_helper.from_array(np.array([1, 4, 2, 4], dtype=np.int64), name="sh_c")
    sh_d = numpy_helper.from_array(np.array([2, 4, 4, 1], dtype=np.int64), name="sh_d")

    r1 = helper.make_node("Reshape", ["X1", "sh_a"], ["T1"], name="r1")
    r2 = helper.make_node("Reshape", ["T1", "sh_b"], ["T2"], name="r2")
    r4 = helper.make_node("Reshape", ["X2", "sh_c"], ["T4"], name="r4")
    r5 = helper.make_node("Reshape", ["T4", "sh_d"], ["T5"], name="r5")
    add = helper.make_node("Add", ["T2", "T5"], ["Y"], name="add1")
    if names in ("empty", "duplicate"):
        for node in (r1, r2, r4, r5):
            node.name = "" if names == "empty" else "reshape"

    # Interleaved: r4 sits between r1 and r2 in the node list
    graph = helper.make_graph(
        [r1, r4, r2, r5, add],
        "interleaved_chains",
        [
            _vi("X1", TensorProto.FLOAT, [1, 2, 4, 4]),
            _vi("X2", TensorProto.FLOAT, [1, 2, 4, 4]),
        ],
        [_vi("Y", TensorProto.FLOAT, [2, 4, 4, 1])],
        initializer=[sh_a, sh_b, sh_c, sh_d],
        value_info=[
            _vi("T1", TensorProto.FLOAT, [1, 4, 2, 4]),
            _vi("T2", TensorProto.FLOAT, [2, 4, 4, 1]),
            _vi("T4", TensorProto.FLOAT, [1, 4, 2, 4]),
            _vi("T5", TensorProto.FLOAT, [2, 4, 4, 1]),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", _OPSET_VERSION)])
    _clamp_ir(model)

    if names == "existing":
        model.graph.initializer.append(
            numpy_helper.from_array(np.array([99], np.int64), "r2_chain_shape")
        )

    x1 = _RNG.standard_normal([1, 2, 4, 4]).astype(np.float32)
    x2 = _RNG.standard_normal([1, 2, 4, 4]).astype(np.float32)
    _prepare_model(model)
    onnx.checker.check_model(model, full_check=True)
    reference = model
    if names == "duplicate":
        reference = copy.deepcopy(model)
        for index, node in enumerate(reference.graph.node):
            node.name = f"reference_{index}"
    ref = _run_ort(reference, {"X1": x1, "X2": x2})

    tr = copy.deepcopy(model)
    _prepare_model(tr)
    tr = _remove_reshapes(tr)
    onnx.checker.check_model(tr, full_check=True)
    _prepare_model(tr)
    got = _run_ort(tr, {"X1": x1, "X2": x2})

    # Each 2-node chain collapses to 1 Reshape → 2 Reshape nodes total (down from 4)
    reshape_nodes = [n for n in tr.graph.node if n.op_type == "Reshape"]
    assert len(reshape_nodes) == 2, (
        f"Each chain collapses to 1 Reshape (2 total), got {[n.name for n in reshape_nodes]}"
    )
    # Verify ORT parity
    np.testing.assert_allclose(ref["Y"], got["Y"], rtol=1e-5, atol=1e-6)


# ══════════════════════════════════════════════════════════════════════════════
# Step 4 — Replace graph-output Reshape with Squeeze
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("opset", [12, 13, 17])
@pytest.mark.parametrize("output_shape", [[3, 4], [1, 3, 4, 1]])
def test_graph_output_reshape_replaced_with_squeeze(opset, output_shape):
    """Use opset-compatible output Squeezes while retaining identity Reshapes."""
    sh_init = numpy_helper.from_array(np.array(output_shape, dtype=np.int64), name="rs_shape")
    reshape = helper.make_node("Reshape", ["X", "rs_shape"], ["Y"], name="rs")
    graph = helper.make_graph(
        [reshape],
        "g",
        [_vi("X", TensorProto.FLOAT, [1, 3, 4, 1])],
        [_vi("Y", TensorProto.FLOAT, output_shape)],
        initializer=[sh_init],
    )
    model = helper.make_model(
        graph,
        opset_imports=[
            helper.make_opsetid("test.modelopt", 99),
            helper.make_opsetid("", opset),
        ],
    )
    _clamp_ir(model)

    x = _RNG.standard_normal([1, 3, 4, 1]).astype(np.float32)
    _prepare_model(model)
    ref = _run_ort(model, {"X": x})

    tr = copy.deepcopy(model)
    _prepare_model(tr)
    tr = _reshape_to_squeeze(tr)
    onnx.checker.check_model(tr, full_check=True)
    _prepare_model(tr)
    got = _run_ort(tr, {"X": x})

    np.testing.assert_array_equal(ref["Y"], got["Y"])
    if output_shape == [1, 3, 4, 1]:
        assert any(node.op_type == "Reshape" for node in tr.graph.node)
        assert not any(node.op_type == "Squeeze" for node in tr.graph.node)
        return
    assert not any(n.op_type == "Reshape" for n in tr.graph.node), "Reshape must be replaced"
    sq_nodes = [n for n in tr.graph.node if n.op_type == "Squeeze"]
    assert len(sq_nodes) == 1, "One Squeeze must be inserted"

    if opset < 13:
        axes = list(next(attr.ints for attr in sq_nodes[0].attribute if attr.name == "axes"))
        assert len(sq_nodes[0].input) == 1
        assert not any("squeeze_axes" in i.name for i in tr.graph.initializer)
    else:
        axes_init = next(i for i in tr.graph.initializer if i.name == sq_nodes[0].input[1])
        axes = numpy_helper.to_array(axes_init).tolist()
    assert axes == [0, 3]


def test_graph_output_reshape_squeeze_axes_left_to_right():
    """[1,1,1,512] → Reshape → [1,512]: squeeze axes must be [0,1] (leftmost 1s), not [1,2].

    Verifies that when multiple 1-dims could be squeezed, the algorithm always
    removes from the lowest (leftmost) axis indices first.
    """
    sh_init = numpy_helper.from_array(np.array([1, 512], dtype=np.int64), name="rs_shape_ltr")
    reshape = helper.make_node("Reshape", ["X", "rs_shape_ltr"], ["Y"], name="rs_ltr")
    graph = helper.make_graph(
        [reshape],
        "g",
        [_vi("X", TensorProto.FLOAT, [1, 1, 1, 512])],
        [_vi("Y", TensorProto.FLOAT, [1, 512])],
        initializer=[sh_init],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", _OPSET_VERSION)])
    _clamp_ir(model)

    x = _RNG.standard_normal([1, 1, 1, 512]).astype(np.float32)
    _prepare_model(model)
    ref = _run_ort(model, {"X": x})

    tr = copy.deepcopy(model)
    _prepare_model(tr)
    tr = _reshape_to_squeeze(tr)
    _prepare_model(tr)
    got = _run_ort(tr, {"X": x})

    assert not any(n.op_type == "Reshape" for n in tr.graph.node), "Reshape must be replaced"
    sq_nodes = [n for n in tr.graph.node if n.op_type == "Squeeze"]
    assert len(sq_nodes) == 1, "One Squeeze must be inserted"

    axes_init = next((i for i in tr.graph.initializer if "squeeze_axes" in i.name), None)
    assert axes_init is not None
    axes = numpy_helper.to_array(axes_init).tolist()
    assert axes == [0, 1], f"Expected axes [0,1] (leftmost 1s), got {axes}"

    np.testing.assert_array_equal(ref["Y"], got["Y"])


@pytest.mark.parametrize("output_shape", [[6, 4], [24]])
@pytest.mark.parametrize("opset", [12, 13, 17])
def test_graph_output_reshape_not_squeezable_padded_to_4d(output_shape, opset):
    """[2,3,4] → Reshape → [6,4] at graph output: not a pure squeeze, output is 2D (<4D).

    Pad the Reshape target to rank four and squeeze every prepended unit dimension.
    This ensures the Reshape produces a 4D tensor before being squeezed back.
    """
    sh_init = numpy_helper.from_array(np.array(output_shape, dtype=np.int64), name="rs_shape2")
    reshape = helper.make_node("Reshape", ["X", "rs_shape2"], ["Y"], name="rs2")
    graph = helper.make_graph(
        [reshape],
        "g",
        [_vi("X", TensorProto.FLOAT, [2, 3, 4])],
        [_vi("Y", TensorProto.FLOAT, output_shape)],
        initializer=[sh_init],
    )
    model = helper.make_model(
        graph,
        opset_imports=[
            helper.make_opsetid("test.modelopt", 99),
            helper.make_opsetid("", opset),
        ],
    )
    _clamp_ir(model)

    x = _RNG.standard_normal([2, 3, 4]).astype(np.float32)
    _prepare_model(model)
    ref = _run_ort(model, {"X": x})

    tr = copy.deepcopy(model)
    _prepare_model(tr)
    tr = _reshape_to_squeeze(tr)
    _prepare_model(tr)
    got = _run_ort(tr, {"X": x})

    # The boundary restores the original output while the intermediate stays 4D.
    rs_nodes = [n for n in tr.graph.node if n.op_type == "Reshape"]
    assert len(rs_nodes) == 1, "Reshape must remain (with updated shape)"
    sq_nodes = [n for n in tr.graph.node if n.op_type == "Squeeze"]
    assert len(sq_nodes) == 1
    rs_shape_init = next((i for i in tr.graph.initializer if i.name == rs_nodes[0].input[1]), None)
    assert rs_shape_init is not None
    assert (
        numpy_helper.to_array(rs_shape_init).tolist()
        == [1] * (4 - len(output_shape)) + output_shape
    )
    if opset < 13:
        axes = list(next(attr.ints for attr in sq_nodes[0].attribute if attr.name == "axes"))
        assert len(sq_nodes[0].input) == 1
        assert not any("sq0_axes" in i.name for i in tr.graph.initializer)
    else:
        axes_init = next(i for i in tr.graph.initializer if i.name == sq_nodes[0].input[1])
        axes = numpy_helper.to_array(axes_init).tolist()
    assert axes == list(range(4 - len(output_shape)))
    onnx.checker.check_model(tr, full_check=True)
    np.testing.assert_array_equal(ref["Y"], got["Y"])


def test_graph_output_reshape_4d_output_kept():
    """[5,6,7,8] → Reshape → [30,14,4] at graph output: not a pure squeeze, BUT padded to [1,30,14,4] + Squeeze([0])."""
    sh_init = numpy_helper.from_array(np.array([30, 14, 4], dtype=np.int64), name="rs_shape_3d")
    reshape = helper.make_node("Reshape", ["X", "rs_shape_3d"], ["Y"], name="rs_3d")
    graph = helper.make_graph(
        [reshape],
        "g",
        [_vi("X", TensorProto.FLOAT, [5, 6, 7, 8])],
        [_vi("Y", TensorProto.FLOAT, [30, 14, 4])],
        initializer=[sh_init],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", _OPSET_VERSION)])
    _clamp_ir(model)

    x = _RNG.standard_normal([5, 6, 7, 8]).astype(np.float32)
    _prepare_model(model)
    ref = _run_ort(model, {"X": x})

    tr = copy.deepcopy(model)
    _prepare_model(tr)
    tr = _reshape_to_squeeze(tr)
    _prepare_model(tr)
    got = _run_ort(tr, {"X": x})

    # Reshape must target [1,30,14,4]
    rs_nodes = [n for n in tr.graph.node if n.op_type == "Reshape"]
    assert len(rs_nodes) == 1
    rs_shape_init = next((i for i in tr.graph.initializer if i.name == rs_nodes[0].input[1]), None)
    assert rs_shape_init is not None
    assert numpy_helper.to_array(rs_shape_init).tolist() == [1, 30, 14, 4], (
        f"Reshape shape must be [1,30,14,4], got {numpy_helper.to_array(rs_shape_init).tolist()}"
    )
    # Squeeze([0]) strips the leading 1 back to [30,14,4]
    sq_nodes = [n for n in tr.graph.node if n.op_type == "Squeeze"]
    assert len(sq_nodes) == 1
    np.testing.assert_array_equal(ref["Y"], got["Y"])


def test_graph_output_reshape_not_at_graph_output_unchanged():
    """Reshape that is NOT a graph output must not be touched by this step."""
    sh_init = numpy_helper.from_array(np.array([3, 4], dtype=np.int64), name="rs_shape3")
    reshape = helper.make_node("Reshape", ["X", "rs_shape3"], ["T"], name="rs3")
    relu = helper.make_node("Relu", ["T"], ["Y"], name="relu")
    graph = helper.make_graph(
        [reshape, relu],
        "g",
        [_vi("X", TensorProto.FLOAT, [1, 3, 4, 1])],
        [_vi("Y", TensorProto.FLOAT, [3, 4])],
        initializer=[sh_init],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", _OPSET_VERSION)])
    _clamp_ir(model)

    tr = copy.deepcopy(model)
    _prepare_model(tr)
    tr = _reshape_to_squeeze(tr)

    # Reshape feeds Relu (not a graph output) → must stay as Reshape
    assert any(n.op_type == "Reshape" for n in tr.graph.node), (
        "Reshape not at graph output must be unchanged"
    )
    assert not any(n.op_type == "Squeeze" for n in tr.graph.node), (
        "No Squeeze must be inserted for non-output Reshape"
    )


# ══════════════════════════════════════════════════════════════════════════════
# GroupQueryAttention — Unsqueeze after GQA output must be preserved
# ══════════════════════════════════════════════════════════════════════════════


def test_unsqueeze_after_gqa_3d_to_4d_skipped():
    """Unsqueeze whose input is a GQA 3D output and whose output is 4D must not be replaced.

    Pattern:  X [b,s,h] → GQA → gqa_out [b,s,h]  (3D)
              gqa_out → Unsqueeze(axes=[0]) → gqa_4d [1,b,s,h]  (4D)  ← must be preserved
              gqa_4d → Relu → Y [1,b,s,h]

    The Unsqueeze is the canonical 3D→4D bridge after GQA and must not be replaced
    by a Reshape.
    """
    b, s, h = 2, 4, 8
    unsq_axes = numpy_helper.from_array(np.array([0], dtype=np.int64), name="gqa_unsq_ax")

    # Stub GQA node with 3D input/output (com.microsoft domain, no real attributes needed)
    gqa = helper.make_node(
        "GroupQueryAttention",
        inputs=["X", "", ""],
        outputs=["gqa_out"],
        name="gqa_node",
        domain="com.microsoft",
        num_heads=2,
        kv_num_heads=2,
    )
    unsqueeze = helper.make_node(
        "Unsqueeze", ["gqa_out", "gqa_unsq_ax"], ["gqa_4d"], name="gqa_unsq"
    )
    relu = helper.make_node("Relu", ["gqa_4d"], ["Y"], name="relu")

    graph = helper.make_graph(
        [gqa, unsqueeze, relu],
        "g",
        [_vi("X", TensorProto.FLOAT, [b, s, h])],
        [_vi("Y", TensorProto.FLOAT, [1, b, s, h])],
        value_info=[
            _vi("gqa_out", TensorProto.FLOAT, [b, s, h]),  # 3D GQA output
            _vi("gqa_4d", TensorProto.FLOAT, [1, b, s, h]),  # 4D Unsqueeze output
        ],
        initializer=[unsq_axes],
    )
    model = helper.make_model(
        graph,
        opset_imports=[
            helper.make_opsetid("", _OPSET_VERSION),
            helper.make_opsetid("com.microsoft", 1),
        ],
    )
    _clamp_ir(model)
    orig_node_count = len(model.graph.node)

    tr = copy.deepcopy(model)
    _prepare_model(tr)
    tr = _replace_sq_unsq(tr)

    # Unsqueeze after GQA (3D→4D) must be preserved unchanged
    assert any(n.op_type == "Unsqueeze" for n in tr.graph.node), (
        "Unsqueeze after GQA 3D output must be preserved"
    )
    assert not any(n.op_type == "Reshape" for n in tr.graph.node), (
        "No Reshape must replace the GQA→Unsqueeze bridge"
    )
    assert len(tr.graph.node) == orig_node_count, (
        "Node count must be unchanged — no nodes added or removed"
    )


# ── Tests: _remove_consecutive_casts ─────────────────────────────────────────


@pytest.mark.parametrize("colliding_name", [False, True])
def test_cast_chain_collapsed_to_single(colliding_name):
    """Cast(INT8→INT64) → Cast(INT64→INT32) collapsed to Cast(INT8→INT32)."""
    cast1 = helper.make_node("Cast", ["X"], ["c1"], name="cast1", to=TensorProto.INT64)
    cast2 = helper.make_node("Cast", ["c1"], ["Y"], name="cast2", to=TensorProto.INT32)
    graph = helper.make_graph(
        [cast1, cast2],
        "g",
        [_vi("X", TensorProto.INT8, [2, 4])],
        [_vi("Y", TensorProto.INT32, [2, 4])],
        value_info=[_vi("c1", TensorProto.INT64, [2, 4])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", _OPSET_VERSION)])
    _clamp_ir(model)
    if colliding_name:
        model.graph.node.append(helper.make_node("Identity", ["X"], ["Z"], name="cast2_chain"))
        model.graph.output.append(_vi("Z", TensorProto.INT8, [2, 4]))
    feeds = {"X": np.arange(8, dtype=np.int8).reshape(2, 4)}
    expected = _run_ort(model, feeds)

    tr = copy.deepcopy(model)
    _prepare_model(tr)
    tr = _remove_cast_chains(tr)

    cast_nodes = [n for n in tr.graph.node if n.op_type == "Cast"]
    assert len(cast_nodes) == 1, f"Expected 1 Cast, got {len(cast_nodes)}"
    # Final Cast should target INT32
    to_val = next(a.i for a in cast_nodes[0].attribute if a.name == "to")
    assert to_val == TensorProto.INT32
    # Input should be X, output should be Y
    assert cast_nodes[0].input[0] == "X"
    assert cast_nodes[0].output[0] == "Y"
    onnx.checker.check_model(tr, full_check=True)
    actual = _run_ort(tr, feeds)
    for name, value in expected.items():
        np.testing.assert_array_equal(actual[name], value)


def test_cast_chain_three_nodes():
    """Three consecutive Casts collapsed to one."""
    cast1 = helper.make_node("Cast", ["X"], ["c1"], name="cast1", to=TensorProto.INT64)
    cast2 = helper.make_node("Cast", ["c1"], ["c2"], name="cast2", to=TensorProto.FLOAT)
    cast3 = helper.make_node("Cast", ["c2"], ["Y"], name="cast3", to=TensorProto.INT32)
    graph = helper.make_graph(
        [cast1, cast2, cast3],
        "g",
        [_vi("X", TensorProto.INT8, [3])],
        [_vi("Y", TensorProto.INT32, [3])],
        value_info=[
            _vi("c1", TensorProto.INT64, [3]),
            _vi("c2", TensorProto.FLOAT, [3]),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", _OPSET_VERSION)])
    _clamp_ir(model)

    tr = copy.deepcopy(model)
    _prepare_model(tr)
    tr = _remove_cast_chains(tr)

    cast_nodes = [n for n in tr.graph.node if n.op_type == "Cast"]
    assert len(cast_nodes) == 1
    to_val = next(a.i for a in cast_nodes[0].attribute if a.name == "to")
    assert to_val == TensorProto.INT32


@pytest.mark.parametrize(
    ("opset", "target", "attributes", "values"),
    [
        (19, TensorProto.FLOAT8E4M3FN, {"saturate": 0}, [-700, 700, 1.25, 1.75]),
        (
            24,
            TensorProto.FLOAT8E8M0,
            {"saturate": 0, "round_mode": "down"},
            [1.25, 1.75, 3.25, 0.6],
        ),
    ],
)
def test_cast_chain_preserves_final_attributes(opset, target, attributes, values):
    """Retain float8 saturation and rounding semantics when removing a lossless cast."""
    first = helper.make_node("Cast", ["X"], ["wide"], to=TensorProto.DOUBLE)
    last = helper.make_node("Cast", ["wide"], ["Y"], to=target, **attributes)
    model = helper.make_model(
        helper.make_graph(
            [first, last],
            "cast_attributes",
            [_vi("X", TensorProto.FLOAT, [4])],
            [_vi("Y", target, [4])],
            value_info=[_vi("wide", TensorProto.DOUBLE, [4])],
        ),
        opset_imports=[helper.make_opsetid("", opset)],
    )
    onnx.checker.check_model(model, full_check=True)
    feeds = {"X": np.array(values, np.float32)}
    with np.errstate(over="ignore", invalid="ignore"):
        expected = ReferenceEvaluator(model).run(None, feeds)[0]
        transformed = _remove_cast_chains(copy.deepcopy(model))
        actual = ReferenceEvaluator(transformed).run(None, feeds)[0]
    np.testing.assert_array_equal(actual.astype(np.float32), expected.astype(np.float32))
    assert len(transformed.graph.node) == 1
    assert list(transformed.graph.node[0].attribute) == list(last.attribute)
    onnx.checker.check_model(transformed, full_check=True)


def test_custom_domain_cast_is_not_collapsed():
    """Treat a custom-domain Cast as opaque instead of assuming standard Cast semantics."""
    function = helper.make_function(
        "test.modelopt",
        "Cast",
        ["X"],
        ["Y"],
        [
            helper.make_node("Cast", ["X"], ["wide"], to=TensorProto.DOUBLE),
            helper.make_node("Neg", ["wide"], ["Y"]),
        ],
        [helper.make_opsetid("", _OPSET_VERSION)],
        attributes=["to"],
    )
    model = helper.make_model(
        helper.make_graph(
            [
                helper.make_node(
                    "Cast", ["X"], ["wide"], domain="test.modelopt", to=TensorProto.DOUBLE
                ),
                helper.make_node("Cast", ["wide"], ["Y"], to=TensorProto.FLOAT),
            ],
            "custom_cast",
            [_vi("X", TensorProto.FLOAT, [4])],
            [_vi("Y", TensorProto.FLOAT, [4])],
            value_info=[_vi("wide", TensorProto.DOUBLE, [4])],
        ),
        opset_imports=[
            helper.make_opsetid("", _OPSET_VERSION),
            helper.make_opsetid("test.modelopt", 1),
        ],
        functions=[function],
    )
    onnx.checker.check_model(model, full_check=True)
    feeds = {"X": np.arange(1, 5, dtype=np.float32)}
    expected = ReferenceEvaluator(model).run(None, feeds)[0]
    before = model.SerializeToString()
    transformed = _remove_cast_chains(copy.deepcopy(model))
    actual = ReferenceEvaluator(transformed).run(None, feeds)[0]
    np.testing.assert_array_equal(actual, expected)
    assert transformed.SerializeToString() == before
    onnx.checker.check_model(transformed, full_check=True)


@pytest.mark.parametrize(
    ("stage", "op_type", "input_shape", "output_shape", "parameters"),
    [
        ("input", "Unsqueeze", [2, 3], [1, 1, 2, 3], [0, 1]),
        ("intermediary", "Unsqueeze", [2, 3], [1, 1, 2, 3], [0, 1]),
        ("intermediary", "Squeeze", [1, 1, 2, 3], [2, 3], [0, 1]),
        ("reshape", "Reshape", [2, 3], [2, 3], [2, 3]),
        ("chain", "Reshape", [6], [2, 3], [2, 3]),
        ("output", "Reshape", [1, 2, 3, 1], [2, 3], [2, 3]),
    ],
)
def test_custom_domain_shape_operations_are_preserved(
    stage, op_type, input_shape, output_shape, parameters
):
    """Preserve local functions whose names match standard shape operations."""
    function = helper.make_function(
        "test.modelopt",
        op_type,
        ["X", "parameters"],
        ["Y"],
        [
            helper.make_node(op_type, ["X", "parameters"], ["adapted"]),
            helper.make_node("Neg", ["adapted"], ["Y"]),
        ],
        [helper.make_opsetid("", _OPSET_VERSION)],
    )
    nodes, value_info = [], []
    source = "X"
    if stage == "intermediary":
        nodes.append(helper.make_node("Identity", ["X"], ["middle"]))
        value_info.append(_vi("middle", TensorProto.FLOAT, input_shape))
        source = "middle"
    output = "T" if stage == "chain" else "Y"
    nodes.append(
        helper.make_node(op_type, [source, "parameters"], [output], domain="test.modelopt")
    )
    initializers = [numpy_helper.from_array(np.array(parameters, np.int64), name="parameters")]
    if stage == "chain":
        value_info.append(_vi("T", TensorProto.FLOAT, output_shape))
        output_shape = [1, 2, 3]
        initializers.append(numpy_helper.from_array(np.array(output_shape, np.int64), name="final"))
        nodes.append(helper.make_node("Reshape", ["T", "final"], ["Y"]))
    model = helper.make_model(
        helper.make_graph(
            nodes,
            "custom_shape_operation",
            [_vi("X", TensorProto.FLOAT, input_shape)],
            [_vi("Y", TensorProto.FLOAT, output_shape)],
            initializer=initializers,
            value_info=value_info,
        ),
        opset_imports=[
            helper.make_opsetid("", _OPSET_VERSION),
            helper.make_opsetid("test.modelopt", 1),
        ],
        functions=[function],
    )
    onnx.checker.check_model(model, full_check=True)
    feeds = {"X": np.arange(1, 7, dtype=np.float32).reshape(input_shape)}
    expected = ReferenceEvaluator(model).run(None, feeds)[0]
    transform = {
        "input": _canonicalize_inputs,
        "intermediary": _replace_sq_unsq,
        "reshape": _remove_reshapes,
        "chain": _remove_reshapes,
        "output": _reshape_to_squeeze,
    }[stage]
    transformed = transform(copy.deepcopy(model))
    onnx.checker.check_model(transformed, full_check=True)
    np.testing.assert_array_equal(ReferenceEvaluator(transformed).run(None, feeds)[0], expected)
    assert transformed.SerializeToString() == model.SerializeToString()


def test_single_cast_unchanged():
    """A lone Cast node is not modified."""
    cast1 = helper.make_node("Cast", ["X"], ["Y"], name="cast1", to=TensorProto.INT32)
    graph = helper.make_graph(
        [cast1],
        "g",
        [_vi("X", TensorProto.FLOAT, [4])],
        [_vi("Y", TensorProto.INT32, [4])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", _OPSET_VERSION)])
    _clamp_ir(model)

    tr = copy.deepcopy(model)
    _prepare_model(tr)
    tr = _remove_cast_chains(tr)

    cast_nodes = [n for n in tr.graph.node if n.op_type == "Cast"]
    assert len(cast_nodes) == 1
    assert cast_nodes[0].input[0] == "X"
    assert cast_nodes[0].output[0] == "Y"


def test_cast_chain_broken_by_multi_consumer():
    """Cast chain is broken when intermediate tensor has multiple consumers."""
    cast1 = helper.make_node("Cast", ["X"], ["c1"], name="cast1", to=TensorProto.INT64)
    cast2 = helper.make_node("Cast", ["c1"], ["Y"], name="cast2", to=TensorProto.INT32)
    relu = helper.make_node("Relu", ["c1"], ["Z"], name="relu")
    graph = helper.make_graph(
        [cast1, cast2, relu],
        "g",
        [_vi("X", TensorProto.FLOAT, [2, 4])],
        [_vi("Y", TensorProto.INT32, [2, 4]), _vi("Z", TensorProto.INT64, [2, 4])],
        value_info=[_vi("c1", TensorProto.INT64, [2, 4])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", _OPSET_VERSION)])
    _clamp_ir(model)

    tr = copy.deepcopy(model)
    _prepare_model(tr)
    tr = _remove_cast_chains(tr)

    # Both Casts should remain (chain broken by multi-consumer c1)
    cast_nodes = [n for n in tr.graph.node if n.op_type == "Cast"]
    assert len(cast_nodes) == 2


@pytest.mark.parametrize("ir_version", [8, 9, 10])
def test_constant_folding_preserves_ir_version(ir_version, monkeypatch):
    """Keep the caller's IR without changing ONNX's process-wide version constant."""
    model = helper.make_model(
        helper.make_graph(
            [helper.make_node("Add", ["a", "b"], ["y"])],
            "folded_ir",
            [],
            [_vi("y", TensorProto.FLOAT, [2])],
            [
                numpy_helper.from_array(np.array([1, 2], np.float32), "a"),
                numpy_helper.from_array(np.array([3, 4], np.float32), "b"),
            ],
        ),
        opset_imports=[helper.make_opsetid("", 17)],
        ir_version=ir_version,
    )
    global_version = onnx.IR_VERSION
    export = gs.export_onnx

    def check_global_version(*args, **kwargs):
        """Check the export's global state while delegating to graphsurgeon."""
        assert global_version == onnx.IR_VERSION
        return export(*args, **kwargs)

    monkeypatch.setattr(gs, "export_onnx", check_global_version)
    result = _fold_consts(model)
    assert result.ir_version == ir_version
    assert global_version == onnx.IR_VERSION
    onnx.checker.check_model(result, full_check=True)
    np.testing.assert_array_equal(ReferenceEvaluator(result).run(None, {})[0], [4, 6])
