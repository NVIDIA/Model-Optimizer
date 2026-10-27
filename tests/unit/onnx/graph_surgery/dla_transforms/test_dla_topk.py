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

"""Execute private TopK index rewrites and preserve incompatible owners and layouts."""

import copy

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper
from onnx.reference import ReferenceEvaluator

from modelopt.onnx.graph_surgery.dla_transforms.dla_topk import _apply_topk

ort = pytest.importorskip("onnxruntime")


def _vi(name, dtype, shape):
    """Declare a typed tensor shape."""
    return helper.make_tensor_value_info(name, dtype, shape)


def _make_topk_chain_model(n=16, k=3, repeats=4, *, rank_converted=False):
    """Build an executable index chain with a fixed rank-four gather layout."""
    x_shape = [1, 1, 1, n] if rank_converted else [1, n]
    rs_shape = [1, 1, k, 1] if rank_converted else [1, k, 1, 1]
    tile_repeats = [1, 1, 1, repeats] if rank_converted else [1, 1, repeats, 1]
    data_shape = [1, 1, n, repeats] if rank_converted else [1, n, repeats, 1]
    out_shape = [1, 1, k, repeats] if rank_converted else [1, k, repeats, 1]
    nodes = [
        helper.make_node(
            "TopK", ["X", "k_val"], ["values", "indices"], name="topk", axis=len(x_shape) - 1
        ),
        helper.make_node("Cast", ["indices"], ["cast1"], name="cast1", to=TensorProto.INT64),
        helper.make_node("Reshape", ["cast1", "target"], ["reshaped"], name="reshape"),
        helper.make_node("Tile", ["reshaped", "repeats"], ["tiled"], name="tile"),
    ]
    if rank_converted:
        nodes.append(
            helper.make_node("Cast", ["tiled"], ["cast2"], name="cast2", to=TensorProto.INT64)
        )
    nodes.append(
        helper.make_node(
            "GatherElements",
            ["data", "cast2" if rank_converted else "tiled"],
            ["Y"],
            name="gather",
            axis=2 if rank_converted else 1,
        )
    )
    model = helper.make_model(
        helper.make_graph(
            nodes,
            "private_indices",
            [_vi("X", TensorProto.FLOAT, x_shape), _vi("data", TensorProto.FLOAT, data_shape)],
            [_vi("Y", TensorProto.FLOAT, out_shape)],
            [
                numpy_helper.from_array(np.asarray([k], np.int64), "k_val"),
                numpy_helper.from_array(np.asarray(rs_shape, np.int64), "target"),
                numpy_helper.from_array(np.asarray(tile_repeats, np.int64), "repeats"),
            ],
        ),
        opset_imports=[helper.make_opsetid("", 17)],
    )
    model.ir_version = 9
    return onnx.shape_inference.infer_shapes(model, strict_mode=True)


def _feeds(model):
    """Use distinct scores and gather values to expose changed index selection."""
    result = {}
    for value in model.graph.input:
        shape = [dim.dim_value for dim in value.type.tensor_type.shape.dim]
        result[value.name] = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    return result


def _execute(model, feeds, *, reference=False):
    """Fully check and run the real graph on CPU or the opaque-function evaluator."""
    onnx.checker.check_model(model, full_check=True)
    if reference:
        return ReferenceEvaluator(model).run(None, feeds)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(
        model.SerializeToString(), options, providers=["CPUExecutionProvider"]
    ).run(None, feeds)


@pytest.mark.parametrize("rank_converted", [False, True])
@pytest.mark.parametrize("legacy", [False, True])
def test_topk_chain_preserves_checked_outputs(rank_converted, legacy):
    """Execute both supported layouts and declare controls for legacy model IR."""
    model = _make_topk_chain_model(rank_converted=rank_converted)
    if legacy:
        model.ir_version = 3
        model.graph.input.extend(
            _vi(t.name, t.data_type, list(t.dims)) for t in model.graph.initializer
        )
    feeds = {name: value for name, value in _feeds(model).items() if name in {"X", "data"}}
    expected = _execute(model, feeds)
    transformed = _apply_topk(copy.deepcopy(model), rank_converted=rank_converted)
    assert not any(node.op_type == "Tile" for node in transformed.graph.node)
    assert any(node.op_type == "Gather" for node in transformed.graph.node)
    for actual, original in zip(_execute(transformed, feeds), expected):
        np.testing.assert_array_equal(actual, original)


@pytest.mark.parametrize("owner", ["values", "indices", "cast1", "reshaped", "tiled", "cast2"])
def test_public_topk_owners_remain_native(owner):
    """Keep public values and middleware owners without changing their ranks or types."""
    model = _make_topk_chain_model(rank_converted=True)
    model.graph.output.append(
        copy.deepcopy(next(v for v in model.graph.value_info if v.name == owner))
    )
    before = model.SerializeToString()
    feeds = _feeds(model)
    expected = _execute(model, feeds)
    transformed = _apply_topk(copy.deepcopy(model), rank_converted=True)
    assert transformed.SerializeToString() == before
    for actual, original in zip(_execute(transformed, feeds), expected):
        np.testing.assert_array_equal(actual, original)


@pytest.mark.parametrize("owner", ["cast1", "reshaped"])
def test_captured_topk_owners_remain_native(owner):
    """Keep index producers that nested graphs capture outside the direct consumer map."""
    model = _make_topk_chain_model(rank_converted=True)
    info = copy.deepcopy(next(v for v in model.graph.value_info if v.name == owner))
    info.name = "captured"
    branch = helper.make_graph(
        [helper.make_node("Identity", [owner], ["captured"])], "capture", [], [info]
    )
    model.graph.input.append(_vi("choose", TensorProto.BOOL, []))
    model.graph.node.append(
        helper.make_node("If", ["choose"], ["capture_out"], then_branch=branch, else_branch=branch)
    )
    info = copy.deepcopy(info)
    info.name = "capture_out"
    model.graph.output.append(info)
    feeds = _feeds(model)
    feeds["choose"] = np.asarray(True)
    expected = _execute(model, feeds)
    before = model.SerializeToString()
    transformed = _apply_topk(copy.deepcopy(model), rank_converted=True)
    assert transformed.SerializeToString() == before
    for actual, original in zip(_execute(transformed, feeds), expected):
        np.testing.assert_array_equal(actual, original)


@pytest.mark.parametrize("rank_converted", [False, True])
def test_gather_elements_partial_non_axis_extent_is_not_expanded(rank_converted):
    """Decline layouts where Gather would select more non-axis rows than GatherElements."""
    model = _make_topk_chain_model(rank_converted=rank_converted)
    model.graph.input[1].type.tensor_type.shape.dim[1 if rank_converted else 0].dim_value = 2
    feeds = _feeds(model)
    expected = _execute(model, feeds)
    before = model.SerializeToString()
    transformed = _apply_topk(copy.deepcopy(model), rank_converted=rank_converted)
    assert transformed.SerializeToString() == before
    np.testing.assert_array_equal(_execute(transformed, feeds)[0], expected[0])


def test_tile_orientation_must_preserve_each_selected_index():
    """Decline same-shaped tiling that interleaves indices along the feature axis."""
    model = _make_topk_chain_model(k=2, rank_converted=True)
    for tensor in model.graph.initializer:
        if tensor.name == "target":
            tensor.CopyFrom(numpy_helper.from_array(np.asarray([1, 1, 1, 2], np.int64), "target"))
        if tensor.name == "repeats":
            tensor.CopyFrom(numpy_helper.from_array(np.asarray([1, 1, 2, 2], np.int64), "repeats"))
    next(v for v in model.graph.value_info if v.name == "reshaped").CopyFrom(
        _vi("reshaped", TensorProto.INT64, [1, 1, 1, 2])
    )
    feeds = _feeds(model)
    expected = _execute(model, feeds)
    before = model.SerializeToString()
    transformed = _apply_topk(copy.deepcopy(model), rank_converted=True)
    assert transformed.SerializeToString() == before
    np.testing.assert_array_equal(_execute(transformed, feeds)[0], expected[0])


def test_topk_float_cast_rounding_is_not_removed():
    """Preserve an intentional FLOAT16 rounding step before the final integer cast."""
    model = _make_topk_chain_model(n=4096, k=1, rank_converted=True)
    cast = next(node for node in model.graph.node if node.name == "cast1")
    next(attr for attr in cast.attribute if attr.name == "to").i = TensorProto.FLOAT16
    for value in model.graph.value_info:
        if value.name in {"cast1", "reshaped", "tiled"}:
            value.type.tensor_type.elem_type = TensorProto.FLOAT16
    feeds = _feeds(model)
    feeds["X"].fill(0)
    feeds["X"].reshape(-1)[2049] = 100
    expected = _execute(model, feeds)
    before = model.SerializeToString()
    transformed = _apply_topk(copy.deepcopy(model), rank_converted=True)
    assert transformed.SerializeToString() == before
    np.testing.assert_array_equal(_execute(transformed, feeds)[0], expected[0])


@pytest.mark.parametrize("site", ["TopK", "Cast"])
def test_opaque_index_functions_remain_native(site):
    """Leave model-local operators intact even when their names match the pattern."""
    model = _make_topk_chain_model()
    node = next(node for node in model.graph.node if node.op_type == site)
    node.domain = "custom"
    body = (
        [helper.make_node("TopK", ["a", "b"], ["v", "i"], axis=1)]
        if site == "TopK"
        else [helper.make_node("Identity", ["a"], ["v"])]
    )
    function = helper.make_function(
        "custom",
        site,
        ["a", "b"] if site == "TopK" else ["a"],
        ["v", "i"] if site == "TopK" else ["v"],
        body,
        [helper.make_opsetid("", 17)],
        attributes=["axis"] if site == "TopK" else ["to"],
    )
    model.functions.append(function)
    model.opset_import.append(helper.make_opsetid("custom", 1))
    feeds = _feeds(model)
    expected = _execute(model, feeds, reference=True)
    before = model.SerializeToString()
    transformed = _apply_topk(copy.deepcopy(model))
    assert transformed.SerializeToString() == before
    np.testing.assert_array_equal(_execute(transformed, feeds, reference=True)[0], expected[0])


@pytest.mark.parametrize("version", [11, 12])
def test_legacy_adapter_schemas_remain_native(version):
    """Avoid inserting input-based Squeeze controls into older operator sets."""
    model = _make_topk_chain_model()
    model.opset_import[0].version = version
    feeds = _feeds(model)
    expected = _execute(model, feeds)
    before = model.SerializeToString()
    transformed = _apply_topk(copy.deepcopy(model))
    assert transformed.SerializeToString() == before
    np.testing.assert_array_equal(_execute(transformed, feeds)[0], expected[0])


def test_topk_without_index_chain_is_unchanged():
    """Keep an ordinary TopK value path that has no index rewrite pattern."""
    model = helper.make_model(
        helper.make_graph(
            [
                helper.make_node("TopK", ["X", "K"], ["values", "indices"], axis=3),
                helper.make_node("Relu", ["values"], ["Y"]),
            ],
            "unmatched_topk",
            [_vi("X", TensorProto.FLOAT, [1, 1, 1, 16])],
            [_vi("Y", TensorProto.FLOAT, [1, 1, 1, 3])],
            [numpy_helper.from_array(np.asarray([3], np.int64), "K")],
        ),
        opset_imports=[helper.make_opsetid("", 17)],
    )
    model.ir_version = 9
    feeds = _feeds(model)
    expected = _execute(model, feeds)
    original = model.SerializeToString()
    _apply_topk(model)
    assert model.SerializeToString() == original
    np.testing.assert_array_equal(_execute(model, feeds)[0], expected[0])


@pytest.mark.parametrize("control", ["k_val", "target", "repeats"])
def test_topk_runtime_initializer_defaults_remain_native(control):
    """Do not freeze overridable structural controls while collapsing the index chain."""
    model = _make_topk_chain_model(rank_converted=True)
    tensor = next(item for item in model.graph.initializer if item.name == control)
    model.graph.input.append(_vi(control, tensor.data_type, list(tensor.dims)))
    feeds = {name: value for name, value in _feeds(model).items() if name in {"X", "data"}}
    expected = _execute(model, feeds)
    before = model.SerializeToString()
    changed = _apply_topk(copy.deepcopy(model), rank_converted=True)
    assert changed.SerializeToString() == before
    np.testing.assert_array_equal(_execute(changed, feeds)[0], expected[0])


def test_topk_allocates_distinct_names_and_preserves_constant_identity():
    """Reuse the shared allocator and batch updates without copying unrelated payloads."""
    model = _make_topk_chain_model(rank_converted=True)
    constant = helper.make_node(
        "Constant",
        [],
        ["topk_sq_out"],
        name="topk_reshape",
        value=numpy_helper.from_array(np.arange(16, dtype=np.float32)),
    )
    model.graph.node.insert(0, constant)
    constant = model.graph.node[0]
    original = constant.SerializeToString()
    model.graph.initializer.append(
        numpy_helper.from_array(np.asarray([42], np.int64), "topk_reshape_const")
    )
    feeds = _feeds(model)
    expected = _execute(model, feeds)
    changed = _apply_topk(model, rank_converted=True)
    assert next(node for node in changed.graph.node if node.op_type == "Constant") is constant
    assert constant.SerializeToString() == original
    definitions = [name for node in changed.graph.node for name in node.output] + [
        item.name for item in changed.graph.initializer
    ]
    assert len(definitions) == len(set(definitions))
    np.testing.assert_array_equal(_execute(changed, feeds)[0], expected[0])


def test_malformed_late_topk_refuses_before_any_mutation():
    """Validate every candidate's arity before changing an earlier valid chain."""
    model = _make_topk_chain_model(rank_converted=True)
    model.graph.node.append(helper.make_node("TopK", ["X"], ["bad_values", "bad_indices"]))
    original = model.SerializeToString()
    with pytest.raises(ValueError, match="expected"):
        _apply_topk(model, rank_converted=True)
    assert model.SerializeToString() == original
