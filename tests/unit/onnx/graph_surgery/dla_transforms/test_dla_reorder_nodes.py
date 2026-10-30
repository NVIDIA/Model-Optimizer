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

"""Tests for the final DLA priority topological node reorder."""

from __future__ import annotations

import numpy as np
import pytest
from onnx import TensorProto, helper, numpy_helper

from modelopt.onnx.graph_surgery.dla_transforms.dla_reorder_nodes import (
    _apply_reorder_nodes,
    priority_topological_order,
)


def _vi(name: str, dtype: int = TensorProto.FLOAT, shape: list[int] | None = None):
    """Build a tensor descriptor for a small ordering fixture."""
    return helper.make_tensor_value_info(name, dtype, shape or [1])


def _node_names(model) -> list[str]:
    """Read node names in their current serialized order."""
    return [node.name for node in model.graph.node]


def test_priority_reorder_preserves_dependencies_and_delays_support_nodes():
    """Support nodes stay close to their consumers without breaking dependencies."""
    model = helper.make_model(
        helper.make_graph(
            [
                helper.make_node("Add", ["relu", "bias"], ["Y"], name="add"),
                helper.make_node(
                    "Constant",
                    [],
                    ["bias"],
                    name="const_bias",
                    value=numpy_helper.from_array(
                        np.array([1.0], dtype=np.float32), name="bias_value"
                    ),
                ),
                helper.make_node("QuantizeLinear", ["X", "scale", "zero"], ["xq"], name="q"),
                helper.make_node("DequantizeLinear", ["xq", "scale", "zero"], ["xdq"], name="dq"),
                helper.make_node("Relu", ["xdq"], ["relu"], name="relu"),
            ],
            "g",
            [_vi("X")],
            [_vi("Y")],
            initializer=[
                numpy_helper.from_array(np.array([0.5], dtype=np.float32), "scale"),
                numpy_helper.from_array(np.array([0], dtype=np.uint8), "zero"),
            ],
        )
    )

    _apply_reorder_nodes(model)

    assert _node_names(model) == ["q", "dq", "relu", "const_bias", "add"]


def test_duplicate_producer_consumer_edges_are_counted_once():
    """Using the same producer twice does not leave a consumer blocked."""
    model = helper.make_model(
        helper.make_graph(
            [
                helper.make_node("Add", ["hidden", "hidden"], ["Y"], name="add"),
                helper.make_node("Identity", ["X"], ["hidden"], name="identity"),
            ],
            "g",
            [_vi("X")],
            [_vi("Y")],
        )
    )

    _apply_reorder_nodes(model)

    assert _node_names(model) == ["identity", "add"]


def test_unresolved_dependency_error_uses_fallback_label_for_unnamed_nodes():
    """Cycle diagnostics identify unnamed nodes as well as named nodes."""
    graph = helper.make_graph(
        [
            helper.make_node("Identity", ["b"], ["a"]),
            helper.make_node("Identity", ["a"], ["b"], name="named_cycle"),
        ],
        "g",
        [],
        [_vi("a")],
    )

    with pytest.raises(ValueError, match=r"<unnamed Identity #0>"):
        priority_topological_order(graph)
