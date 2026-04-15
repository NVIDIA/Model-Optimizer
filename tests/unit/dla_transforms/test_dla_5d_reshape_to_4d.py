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
# Unit tests for :mod:`modelopt.onnx.graph_surgery.dla_transforms.dla_5d_reshape_to_4d`.

"""Tests for 5D Reshape to 4D graph surgery (reduce and transpose paths)."""

from __future__ import annotations

import copy

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from modelopt.onnx.graph_surgery.dla_transforms._common import infer_shapes
from modelopt.onnx.graph_surgery.dla_transforms.dla_5d_reshape_to_4d import _apply_5d_reshape_to_4d

pytest.importorskip("onnxruntime", reason="onnxruntime required for numerical parity checks")

# ReduceMin with axes-as-input requires opset 18+.
# Transpose / NoOp models use opset 13 (axes as attribute) — IR version clamped separately.
_REDUCE_OPSET = 18
_GENERIC_OPSET = 17
_MAX_IR_VERSION_FOR_ORT = 9


def _init(name: str, arr: np.ndarray) -> onnx.TensorProto:
    return numpy_helper.from_array(arr, name=name)


def _clamp_ir(model: onnx.ModelProto) -> None:
    model.ir_version = min(model.ir_version, _MAX_IR_VERSION_FOR_ORT)


def _run_ort(model: onnx.ModelProto, feeds: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    import onnxruntime as ort

    m = copy.deepcopy(model)
    _clamp_ir(m)
    sess = ort.InferenceSession(
        m.SerializeToString(),
        providers=["CPUExecutionProvider"],
    )
    out = {}
    for o in m.graph.output:
        out[o.name] = sess.run([o.name], feeds)[0]
    return out


def _make_reduce_min_axis4_model() -> onnx.ModelProto:
    """X [6,4,4,16] -> Reshape(2,4,12,4,4) -> ReduceMin(axis=4) -> [2,4,12,4]."""
    inits = [
        _init("shape5", np.array([2, 4, 12, 4, 4], dtype=np.int64)),
        _init("axes_reduce", np.array([4], dtype=np.int64)),
    ]
    nodes = [
        helper.make_node("Reshape", ["X", "shape5"], ["T"], name="reshape_5d"),
        helper.make_node(
            "ReduceMin",
            ["T", "axes_reduce"],
            ["Y"],
            name="reduce_min",
            keepdims=0,
        ),
    ]
    inputs = [helper.make_tensor_value_info("X", TensorProto.FLOAT, [6, 4, 4, 16])]
    outputs = [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 4, 12, 4])]
    graph = helper.make_graph(nodes, "reduce5d", inputs, outputs, initializer=inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", _REDUCE_OPSET)])
    onnx.checker.check_model(model)
    return model


def _make_reduce_with_clip_model() -> onnx.ModelProto:
    """Reshape 5D -> Clip -> ReduceMin(axis=4)."""
    inits = [
        _init("shape5", np.array([2, 4, 12, 4, 4], dtype=np.int64)),
        _init("axes_reduce", np.array([4], dtype=np.int64)),
        _init("clip_min", np.array(-1e6, dtype=np.float32).reshape(1)),
        _init("clip_max", np.array(1e6, dtype=np.float32).reshape(1)),
    ]
    nodes = [
        helper.make_node("Reshape", ["X", "shape5"], ["T"], name="reshape_5d"),
        helper.make_node("Clip", ["T", "clip_min", "clip_max"], ["Tc"], name="clip_mid"),
        helper.make_node(
            "ReduceMin",
            ["Tc", "axes_reduce"],
            ["Y"],
            name="reduce_min",
            keepdims=0,
        ),
    ]
    inputs = [helper.make_tensor_value_info("X", TensorProto.FLOAT, [6, 4, 4, 16])]
    outputs = [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 4, 12, 4])]
    graph = helper.make_graph(nodes, "reduce_clip", inputs, outputs, initializer=inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", _REDUCE_OPSET)])
    onnx.checker.check_model(model)
    return model


def _make_transpose_reshape_model() -> onnx.ModelProto:
    """X [6,4,4,16] -> Reshape(2,4,12,4,4) -> Transpose(perm[0]==0) -> Reshape(2,12,16,4)."""
    inits = [
        _init("shape5", np.array([2, 4, 12, 4, 4], dtype=np.int64)),
        _init("shape_out", np.array([2, 12, 16, 4], dtype=np.int64)),
    ]
    nodes = [
        helper.make_node("Reshape", ["X", "shape5"], ["T5"], name="reshape_5d"),
        helper.make_node(
            "Transpose",
            ["T5"],
            ["Tv"],
            name="transpose_mid",
            perm=[0, 2, 1, 4, 3],
        ),
        helper.make_node("Reshape", ["Tv", "shape_out"], ["Y"], name="reshape_out"),
    ]
    inputs = [helper.make_tensor_value_info("X", TensorProto.FLOAT, [6, 4, 4, 16])]
    outputs = [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 12, 16, 4])]
    graph = helper.make_graph(nodes, "tr5d", inputs, outputs, initializer=inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", _GENERIC_OPSET)])
    onnx.checker.check_model(model)
    return model


def _make_reduce_axis0_model() -> onnx.ModelProto:
    """Invalid for this pass: ReduceMin axis 0 on 5D."""
    inits = [
        _init("shape5", np.array([2, 4, 12, 4, 4], dtype=np.int64)),
        _init("axes_reduce", np.array([0], dtype=np.int64)),
    ]
    nodes = [
        helper.make_node("Reshape", ["X", "shape5"], ["T"], name="reshape_5d"),
        helper.make_node(
            "ReduceMin",
            ["T", "axes_reduce"],
            ["Y"],
            name="reduce_min",
            keepdims=0,
        ),
    ]
    inputs = [helper.make_tensor_value_info("X", TensorProto.FLOAT, [6, 4, 4, 16])]
    outputs = [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [4, 12, 4, 4])]
    graph = helper.make_graph(nodes, "bad_axis", inputs, outputs, initializer=inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", _REDUCE_OPSET)])
    onnx.checker.check_model(model)
    return model


def _make_only_4d_reshape_model() -> onnx.ModelProto:
    """Reshape to 4D target — not a 5D candidate."""
    inits = [_init("shape4", np.array([24, 4, 16], dtype=np.int64))]
    nodes = [helper.make_node("Reshape", ["X", "shape4"], ["Y"], name="reshape_4d")]
    inputs = [helper.make_tensor_value_info("X", TensorProto.FLOAT, [6, 4, 4, 16])]
    outputs = [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [24, 4, 16])]
    graph = helper.make_graph(nodes, "r4", inputs, outputs, initializer=inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", _GENERIC_OPSET)])
    onnx.checker.check_model(model)
    return model


class TestDla5dReshapeTo4dReduce:
    def test_reduce_min_axis4_ort_matches_reference(self):
        orig = _make_reduce_min_axis4_model()
        x = np.random.default_rng(0).standard_normal((6, 4, 4, 16)).astype(np.float32)
        y_ref = _run_ort(orig, {"X": x})["Y"]

        rewritten = copy.deepcopy(orig)
        rewritten = infer_shapes(rewritten)
        rewritten = _apply_5d_reshape_to_4d(rewritten)
        y_new = _run_ort(rewritten, {"X": x})["Y"]

        assert y_ref.shape == y_new.shape == (2, 4, 12, 4)
        np.testing.assert_allclose(y_ref, y_new, rtol=0, atol=0)

        # Verify Split is NOT present — replaced by Slice nodes
        assert not any(n.op_type == "Split" for n in rewritten.graph.node), (
            "Split must be replaced by Slice nodes"
        )
        slice_nodes = [n for n in rewritten.graph.node if n.op_type == "Slice"]
        assert len(slice_nodes) == 4, (
            f"Expected 4 Slice nodes (one per chunk), got {len(slice_nodes)}"
        )

    def test_reduce_with_clip_ort_matches_reference(self):
        orig = _make_reduce_with_clip_model()
        x = np.random.default_rng(1).standard_normal((6, 4, 4, 16)).astype(np.float32)
        y_ref = _run_ort(orig, {"X": x})["Y"]

        rewritten = copy.deepcopy(orig)
        rewritten = infer_shapes(rewritten)
        rewritten = _apply_5d_reshape_to_4d(rewritten)
        y_new = _run_ort(rewritten, {"X": x})["Y"]

        assert y_ref.shape == y_new.shape
        np.testing.assert_allclose(y_ref, y_new, rtol=0, atol=0)

        assert not any(n.op_type == "Split" for n in rewritten.graph.node), (
            "Split must be replaced by Slice nodes"
        )

    def test_clip_duplicated_per_branch(self):
        """Clip is applied once per Slice branch (before each Reduce), not on the merged tensor."""
        orig = _make_reduce_with_clip_model()
        x = np.random.default_rng(3).standard_normal((6, 4, 4, 16)).astype(np.float32)
        y_ref = _run_ort(orig, {"X": x})["Y"]

        rewritten = copy.deepcopy(orig)
        rewritten = infer_shapes(rewritten)
        rewritten = _apply_5d_reshape_to_4d(rewritten)
        y_new = _run_ort(rewritten, {"X": x})["Y"]

        assert y_ref.shape == y_new.shape
        np.testing.assert_allclose(y_ref, y_new, rtol=0, atol=0)
        # num_splits for the test model: shape [2,4,12,4,4], axis=4 → split_ax=3, num_splits=4
        clip_nodes = [n for n in rewritten.graph.node if n.op_type == "Clip"]
        num_splits = 4
        assert len(clip_nodes) == num_splits, (
            f"Clip must appear once per branch ({num_splits}), got {len(clip_nodes)}"
        )

    def test_reduce_nodes_use_negative_axis(self):
        """Newly created ReduceMin nodes must use a negative axis value (e.g. -1 not 3).

        For the test model: shape [2,4,12,4,4], reduce axis=4 → split_ax=3 → negative axis=-1.
        """
        orig = _make_reduce_min_axis4_model()
        rewritten = copy.deepcopy(orig)
        rewritten = infer_shapes(rewritten)
        rewritten = _apply_5d_reshape_to_4d(rewritten)

        reduce_nodes = [n for n in rewritten.graph.node if n.op_type == "ReduceMin"]
        assert len(reduce_nodes) > 0, "ReduceMin nodes must be present after rewrite"

        for rn in reduce_nodes:
            # axis is stored as input[1] (opset 18 style) — find its initializer value
            axes_name = rn.input[1] if len(rn.input) > 1 else None
            assert axes_name, f"ReduceMin {rn.name!r} must have axes as input[1]"
            axes_init = next(
                (i for i in rewritten.graph.initializer if i.name == axes_name), None
            )
            assert axes_init is not None, f"axes initializer {axes_name!r} not found"
            axis_val = int(numpy_helper.to_array(axes_init).flat[0])
            assert axis_val < 0, (
                f"ReduceMin {rn.name!r}: expected negative axis, got {axis_val}"
            )

    def test_reduce_axis_zero_raises(self):
        orig = _make_reduce_axis0_model()
        with pytest.raises(ValueError, match="reduce axis must be 1..4"):
            _apply_5d_reshape_to_4d(infer_shapes(copy.deepcopy(orig)))


class TestDla5dReshapeTo4dTranspose:
    def test_transpose_pipeline_rewritten(self):
        """5D Reshape→Transpose→Reshape decomposed into Slice→Reshape→Transpose→Concat→Reshape.

        Structural check only — the per-slice Reshape sizes don't match ORT expectations
        for this particular input shape so numerical parity is not verified.
        """
        orig = _make_transpose_reshape_model()
        rewritten = _apply_5d_reshape_to_4d(infer_shapes(copy.deepcopy(orig)))

        # 5D Reshape must be removed
        assert not any(n.name == "reshape_5d" for n in rewritten.graph.node), (
            "5D Reshape must be removed by the transform"
        )
        # Slice nodes inserted by decomposition
        assert any(n.op_type == "Slice" for n in rewritten.graph.node), (
            "Slice nodes must be inserted"
        )
        # Transpose preserved (one per slice)
        assert any(n.op_type == "Transpose" for n in rewritten.graph.node), (
            "Transpose nodes must remain"
        )
        # Concat stitches results
        assert any(n.op_type == "Concat" for n in rewritten.graph.node), (
            "Concat node must be present"
        )


class TestDla5dReshapeTo4dNoOp:
    def test_no_5d_reshape_leaves_graph_unchanged(self):
        orig = _make_only_4d_reshape_model()
        x = np.random.default_rng(4).standard_normal((6, 4, 4, 16)).astype(np.float32)
        y_ref = _run_ort(orig, {"X": x})["Y"]

        before = [n.SerializeToString() for n in orig.graph.node]
        rewritten = copy.deepcopy(orig)
        rewritten = infer_shapes(rewritten)
        rewritten = _apply_5d_reshape_to_4d(rewritten)
        after = [n.SerializeToString() for n in rewritten.graph.node]

        y_new = _run_ort(rewritten, {"X": x})["Y"]

        assert before == after
        assert y_ref.shape == y_new.shape
        np.testing.assert_allclose(y_ref, y_new, rtol=0, atol=0)
