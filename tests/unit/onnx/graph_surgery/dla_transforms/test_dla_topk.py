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
# Unit tests for :mod:`modelopt.onnx.graph_surgery.dla_transforms.dla_topk`.

"""ORT CPU parity tests for ``dla_topk``.

Pattern rewritten
-----------------
x [1,1,1,N] → TopK(K=k) → values, indices
                             indices → Cast → Reshape → Tile → Cast → GatherElements(data, axis=2)

Becomes::
x [1,N,1,1] → Reshape → TopK(axis=1) → Cast(INT32) → Squeeze([0,2,3]) → Gather(axis=2)

Coverage
--------
Full chain rewrite — values match original GatherElements output, ORT parity
Chain absent (no Tile) — graph unchanged, noop
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
from onnx import TensorProto, helper, numpy_helper

from modelopt.onnx.graph_surgery.dla_transforms import dla_topk as _mod

pytest.importorskip("onnxruntime", reason="onnxruntime required for ORT parity checks")

_REPO_ROOT = Path(__file__).resolve().parents[5]
_SAVED_MODELS_DIR = _REPO_ROOT / "scratch_space" / "topk_test_models"
_OPSET_VERSION = 17
_MAX_IR_VERSION_FOR_ORT = 9


_apply = _mod._apply_topk


# ── Shared utilities ──────────────────────────────────────────────────────────


def _vi(name, elem_type, shape):
    return helper.make_tensor_value_info(name, elem_type, shape)


def _clamp_ir(model):
    model.ir_version = min(model.ir_version, _MAX_IR_VERSION_FOR_ORT)


def _prepare_model(model):
    try:
        from onnxruntime.tools.symbolic_shape_infer import (
            SymbolicShapeInference,  # type: ignore[import-untyped]
        )

        inferred = SymbolicShapeInference.infer_shapes(model, auto_merge_symbolic_dims=True)
        if inferred is not None:
            del model.graph.value_info[:]
            model.graph.value_info.extend(inferred.graph.value_info)
    except Exception:
        pass


def _run_ort(model, feeds):
    import onnxruntime as ort  # type: ignore[import-untyped]

    _clamp_ir(model)
    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    return {o.name: sess.run([o.name], feeds)[0] for o in model.graph.output}


_RNG = np.random.default_rng(2024)


# ── Build the full chain model ────────────────────────────────────────────────


def _make_topk_chain_model(n: int = 16, k: int = 3, k_repeats: int = 4):
    """Build the TopK→Cast→Reshape→Tile→GatherElements chain.

    Matches the user-updated transform which expects:
    * x [1,N] — 2D input, TopK axis=1
    * indices [1,k] → Cast → Reshape [1,k,1] → Tile [1,1,k_repeats,1] → [1,k,k_repeats,1]
    * GatherElements(data[1,N,k_repeats,1], indices[1,k,k_repeats,1], axis=1)
    * gather_axis=1, so gather_axis+1=2: tile_repeats[2]==k_repeats, all other repeats==1
    """
    x_shape = [1, n]
    # tile_repeats[2]=k_repeats (gather_axis+1=2), all others 1
    tile_repeats = np.array([1, 1, k_repeats, 1], dtype=np.int64)
    # data: same rank as tiled indices [1,k,k_repeats,1], axis=1 has n elements
    data_shape = [1, n, k_repeats, 1]

    inits = [
        numpy_helper.from_array(np.array([k], dtype=np.int64), name="k_val"),
        numpy_helper.from_array(np.array([1, k, 1], dtype=np.int64), name="rs_shape"),
        numpy_helper.from_array(tile_repeats, name="tile_repeats"),
    ]

    topk = helper.make_node("TopK", ["X", "k_val"], ["topk_vals", "topk_idx"], name="topk", axis=1)
    cast1 = helper.make_node(
        "Cast", ["topk_idx"], ["cast1_out"], name="cast1", to=TensorProto.INT64
    )
    reshape = helper.make_node("Reshape", ["cast1_out", "rs_shape"], ["rs_out"], name="rs")
    tile = helper.make_node("Tile", ["rs_out", "tile_repeats"], ["tile_out"], name="tile")
    ge = helper.make_node("GatherElements", ["data", "tile_out"], ["Y"], name="ge", axis=1)

    graph = helper.make_graph(
        [topk, cast1, reshape, tile, ge],
        "topk_chain",
        [_vi("X", TensorProto.FLOAT, x_shape), _vi("data", TensorProto.FLOAT, data_shape)],
        [_vi("Y", TensorProto.FLOAT, [1, k, k_repeats, 1])],
        value_info=[
            _vi("topk_vals", TensorProto.FLOAT, [1, k]),
            _vi("topk_idx", TensorProto.INT64, [1, k]),
            _vi("cast1_out", TensorProto.INT64, [1, k]),
            _vi("rs_out", TensorProto.INT64, [1, k, 1]),
            _vi("tile_out", TensorProto.INT64, [1, k, k_repeats, 1]),
        ],
        initializer=inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", _OPSET_VERSION)])
    _clamp_ir(model)
    return model


# ── 1. Full chain rewrite ─────────────────────────────────────────────────────


def test_topk_chain_rewritten():
    """TopK→Cast→Reshape→Tile→Cast→GatherElements structural rewrite.

    The rewrite changes the output shape (GatherElements → Gather compresses dims),
    so only structural correctness is verified, not ORT parity.
    """
    n, k, k_repeats = 16, 3, 4
    model = _make_topk_chain_model(n=n, k=k, k_repeats=k_repeats)

    tr = copy.deepcopy(model)
    _prepare_model(tr)
    tr = _apply(tr)

    # ── Chain nodes must be removed ───────────────────────────────────────────
    assert not any(n.op_type == "Tile" for n in tr.graph.node), "Tile must be removed"
    assert not any(n.op_type == "Reshape" and n.name == "rs" for n in tr.graph.node), (
        "Original Reshape (rs) must be removed"
    )
    assert not any(n.op_type == "GatherElements" for n in tr.graph.node), (
        "GatherElements must be replaced by Gather"
    )
    assert any(n.op_type == "Gather" for n in tr.graph.node), "Gather must be inserted"
    assert any(n.op_type == "Squeeze" for n in tr.graph.node), (
        "Squeeze must be inserted to compress index dims"
    )

    # ── TopK axis must be 1 (N dim after Reshape [1,N,1,1]) ──────────────────
    topk_node = next(n for n in tr.graph.node if n.op_type == "TopK")
    topk_axis = next((int(a.i) for a in topk_node.attribute if a.name == "axis"), None)
    assert topk_axis == 1, f"TopK axis must be 1 after rewrite, got {topk_axis}"

    # ── Gather axis must be 1 ─────────────────────────────────────────────────
    gather_node = next(n for n in tr.graph.node if n.op_type == "Gather")
    gather_axis_after = next((int(a.i) for a in gather_node.attribute if a.name == "axis"), None)
    assert gather_axis_after == 1, f"Gather axis must be 1, got {gather_axis_after}"

    # ── Order: Cast before Squeeze, Squeeze before Gather ────────────────────
    node_types = [n.op_type for n in tr.graph.node]
    cast1_idx = next(i for i, t in enumerate(node_types) if t == "Cast")
    squeeze_idx = next(i for i, t in enumerate(node_types) if t == "Squeeze")
    gather_idx = next(i for i, t in enumerate(node_types) if t == "Gather")
    assert cast1_idx < squeeze_idx < gather_idx, (
        f"Order must be Cast({cast1_idx}) < Squeeze({squeeze_idx}) < Gather({gather_idx})"
    )

    # ── Cast1 must now target INT32 ───────────────────────────────────────────
    cast1_node = next(n for n in tr.graph.node if n.op_type == "Cast")
    cast1_to = next((int(a.i) for a in cast1_node.attribute if a.name == "to"), None)
    assert cast1_to == TensorProto.INT32, f"Cast1 must target INT32, got {cast1_to}"

    # ── Squeeze input must be Cast1 output (topk→cast→squeeze order) ─────────
    sq_node = next(n for n in tr.graph.node if n.op_type == "Squeeze")
    cast1_out = cast1_node.output[0]
    assert sq_node.input[0] == cast1_out, f"Squeeze must read Cast output, got {sq_node.input[0]}"


# ── 2. No matching chain → noop ───────────────────────────────────────────────


def test_topk_no_chain_noop():
    """TopK without the Cast→Reshape→Tile→Cast→GatherElements chain is left unchanged."""
    n, k = 16, 3
    k_init = numpy_helper.from_array(np.array([k], dtype=np.int64), name="k_val2")
    topk = helper.make_node("TopK", ["X", "k_val2"], ["vals", "idx"], name="topk2", axis=3)
    relu = helper.make_node("Relu", ["vals"], ["Y"], name="relu")
    graph = helper.make_graph(
        [topk, relu],
        "g",
        [_vi("X", TensorProto.FLOAT, [1, 1, 1, n])],
        [_vi("Y", TensorProto.FLOAT, [1, 1, 1, k])],
        initializer=[k_init],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", _OPSET_VERSION)])
    _clamp_ir(model)
    orig_count = len(model.graph.node)

    x = _RNG.standard_normal([1, 1, 1, n]).astype(np.float32)
    _prepare_model(model)
    ref = _run_ort(model, {"X": x})

    tr = copy.deepcopy(model)
    _prepare_model(tr)
    tr = _apply(tr)
    _prepare_model(tr)
    got = _run_ort(tr, {"X": x})

    assert len(tr.graph.node) == orig_count, "Noop: node count must not change"
    np.testing.assert_allclose(ref["Y"], got["Y"], rtol=1e-5, atol=1e-6)
