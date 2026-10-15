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

"""Final priority topological node reorder for DLA ONNX models."""

from __future__ import annotations

import heapq
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import onnx

from ..utils.graph_utils import _node_inputs

PRIORITIZED_OPS = {
    "Constant",
    "ConstantOfShape",
    "QuantizeLinear",
    "DequantizeLinear",
}


def _node_label(node: onnx.NodeProto, index: int) -> str:
    return node.name or f"<unnamed {node.op_type} #{index}>"


def _build_edges(nodes: list[onnx.NodeProto]) -> tuple[list[set[int]], list[list[int]]]:
    output_to_node: dict[str, int] = {}
    for index, node in enumerate(nodes):
        for output in node.output:
            if output:
                output_to_node[output] = index

    dependencies = [set() for _ in nodes]
    consumers: list[list[int]] = [[] for _ in nodes]
    seen_edges: set[tuple[int, int]] = set()
    for consumer_index, node in enumerate(nodes):
        for input_name in _node_inputs(node):
            producer_index = output_to_node.get(input_name)
            if producer_index is None:
                continue

            edge = (producer_index, consumer_index)
            if edge in seen_edges:
                continue

            seen_edges.add(edge)
            dependencies[consumer_index].add(producer_index)
            consumers[producer_index].append(consumer_index)

    return dependencies, consumers


def _unresolved_dependency_error(nodes: list[onnx.NodeProto], indegrees: list[int]) -> ValueError:
    remaining = [
        _node_label(nodes[index], index) for index, indegree in enumerate(indegrees) if indegree > 0
    ]
    return ValueError(
        "Graph contains a cycle or unresolved dependency involving: " + ", ".join(remaining[:10])
    )


def _compute_depths(
    nodes: list[onnx.NodeProto],
    dependencies: list[set[int]],
    consumers: list[list[int]],
) -> list[int]:
    depths = [0] * len(nodes)
    indegrees = [len(node_dependencies) for node_dependencies in dependencies]
    ready = [index for index, indegree in enumerate(indegrees) if indegree == 0]

    processed = 0
    read_index = 0
    while read_index < len(ready):
        index = ready[read_index]
        read_index += 1
        processed += 1

        for consumer_index in consumers[index]:
            depths[consumer_index] = max(depths[consumer_index], depths[index] + 1)
            indegrees[consumer_index] -= 1
            if indegrees[consumer_index] == 0:
                ready.append(consumer_index)

    if processed != len(nodes):
        raise _unresolved_dependency_error(nodes, indegrees)

    return depths


def priority_topological_order(
    graph: onnx.GraphProto,
) -> tuple[list[int], list[int], list[int]]:
    """Return node indices in DLA priority topological order.

    The order always respects data dependencies. Among ready nodes, generic ops
    keep the lowest priority, while Constant/QDQ-style support nodes are delayed
    toward their first consumer depth so they appear nearer to their use.
    """
    nodes = list(graph.node)
    dependencies, consumers = _build_edges(nodes)
    depths = _compute_depths(nodes, dependencies, consumers)
    sink_priority = max(depths, default=0) + 1

    def node_priority(index: int) -> int:
        if nodes[index].op_type not in PRIORITIZED_OPS:
            return 0
        if consumers[index]:
            return depths[consumers[index][0]]
        return sink_priority

    indegrees = [len(node_dependencies) for node_dependencies in dependencies]
    ready: list[tuple[int, int]] = []
    for index, indegree in enumerate(indegrees):
        if indegree == 0:
            heapq.heappush(ready, (node_priority(index), index))

    order: list[int] = []
    while ready:
        _, index = heapq.heappop(ready)
        order.append(index)

        for consumer_index in consumers[index]:
            indegrees[consumer_index] -= 1
            if indegrees[consumer_index] == 0:
                heapq.heappush(ready, (node_priority(consumer_index), consumer_index))

    if len(order) != len(nodes):
        raise _unresolved_dependency_error(nodes, indegrees)

    priorities = [node_priority(index) for index in range(len(nodes))]
    return order, depths, priorities


def _apply_reorder_nodes(model: onnx.ModelProto) -> onnx.ModelProto:
    """Reorder ``graph.node`` in-place using DLA priority topological order."""
    order, _, _ = priority_topological_order(model.graph)
    nodes = list(model.graph.node)
    sorted_nodes = [nodes[index] for index in order]
    del model.graph.node[:]
    model.graph.node.extend(sorted_nodes)
    return model
