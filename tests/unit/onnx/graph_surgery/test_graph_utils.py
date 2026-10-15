# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Shared graph queries, shape resolution, and initializer contracts."""

import copy

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper, numpy_helper

from modelopt.onnx.graph_surgery.utils.graph_utils import (
    GraphCache,
    _shape_before_boundary_promotion,
    add_initializer,
    batch_replace_nodes,
    insert_nodes_at_position,
    resolve_reshape_shape,
)


@pytest.mark.parametrize(("values", "dtype"), [([3], np.int64), ([2], np.int32)])
def test_generated_initializer_collision_leaves_graph_unchanged(values, dtype):
    graph = helper.make_graph([], "collision", [], [])
    graph.initializer.append(numpy_helper.from_array(np.array([2], dtype=np.int64), "axes"))
    before = graph.SerializeToString()
    with pytest.raises(ValueError, match="Conflicting generated initializer"):
        add_initializer(graph, "axes", np.asarray(values, dtype=dtype), with_value_info=False)
    assert graph.SerializeToString() == before


def test_original_boundary_shape_survives_promotion():
    graph = helper.make_graph(
        [helper.make_node("Unsqueeze", ["x", "axes"], ["x_unsqueeze_input"])],
        "boundary",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3, 4])],
        [helper.make_tensor_value_info("x_unsqueeze_input", TensorProto.FLOAT, [1, 2, 3, 4])],
        [numpy_helper.from_array(np.array([0], dtype=np.int64), "axes")],
    )
    cache = GraphCache(graph)
    assert _shape_before_boundary_promotion(cache, "x_unsqueeze_input") == [2, 3, 4]
    assert cache.get_shape("x_unsqueeze_input") == [1, 2, 3, 4]


@pytest.mark.parametrize("target", [[-1, -1], [4, 2]])
def test_invalid_reshape_does_not_change_requested_shape(target):
    values = np.array(target, dtype=np.int64)
    with pytest.raises(ValueError):
        resolve_reshape_shape(values, [2, 3])
    np.testing.assert_array_equal(values, target)


def test_symbolic_shape_is_not_reported_as_a_concrete_prefix():
    graph = helper.make_graph(
        [], "symbolic", [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, "N", 4])], []
    )
    assert GraphCache(graph).get_shape("x") is None


def test_cache_rebuild_refreshes_initializer_values():
    graph = helper.make_graph([], "cache", [], [], [numpy_helper.from_array(np.array([1]), "a")])
    cache = GraphCache(graph)
    np.testing.assert_array_equal(cache.get_init_array("a"), [1])
    graph.initializer[0].CopyFrom(numpy_helper.from_array(np.array([2]), "a"))
    cache.rebuild()
    np.testing.assert_array_equal(cache.get_init_array("a"), [2])


@pytest.mark.parametrize(("target", "source"), [([-1, 0], [0, 3]), ([-2, -3], [2, 3])])
def test_invalid_reshape_dimensions_fail_clearly(target, source):
    with pytest.raises(ValueError):
        resolve_reshape_shape(np.array(target), source)


def test_generated_initializer_collision_preserves_signed_zero():
    graph = helper.make_graph(
        [],
        "signed_zero",
        [],
        [],
        [numpy_helper.from_array(np.array([-0.0], dtype=np.float32), "a")],
    )
    with pytest.raises(ValueError, match="Conflicting generated initializer"):
        add_initializer(graph, "a", np.asarray([0.0], dtype=np.float32), with_value_info=False)


@pytest.mark.parametrize("replace", [False, True])
def test_node_edits_preserve_topology_and_boundary_values(replace):
    original = helper.make_model(
        helper.make_graph(
            [helper.make_node("Identity", ["x"], ["y"])],
            "topology",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 3])],
        ),
        opset_imports=[helper.make_opsetid("", 18)],
        ir_version=10,
    )
    candidate = copy.deepcopy(original)
    final_output = "y" if replace else "restored"
    nodes = [
        helper.make_node("Neg", ["x"], ["negative"]),
        helper.make_node("Neg", ["negative"], [final_output]),
    ]
    existing = candidate.graph.node[0]
    if replace:
        batch_replace_nodes(candidate.graph, {0: ([existing], nodes)})
    else:
        existing.input[0] = final_output
        insert_nodes_at_position(candidate.graph, nodes, existing)
    onnx.checker.check_model(candidate)
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    feeds = {"x": np.arange(6, dtype=np.float32).reshape(2, 3)}
    results = [
        ort.InferenceSession(
            model.SerializeToString(), options, providers=["CPUExecutionProvider"]
        ).run(None, feeds)[0]
        for model in (original, candidate)
    ]
    np.testing.assert_array_equal(*results)
