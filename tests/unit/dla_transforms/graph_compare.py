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

"""Structural comparison of two ONNX graphs with different tensor/node names.

The comparison canonicalises both graphs via a BFS traversal from graph outputs,
producing a name-independent fingerprint for each node.  Two graphs are
structurally identical when their fingerprint sequences match.

Usage in tests::

    from graph_compare import assert_graphs_structurally_equal

    assert_graphs_structurally_equal(actual_model.graph, expected_model.graph)
"""

from __future__ import annotations

from collections import Counter, deque

import onnx
from onnx import numpy_helper

# ---------------------------------------------------------------------------
# Node fingerprint: a name-independent description of a single node
# ---------------------------------------------------------------------------


def _tensor_signature(tensor: onnx.TensorProto) -> tuple:
    """Return a hashable tensor payload signature (dtype + shape + raw bytes)."""
    arr = numpy_helper.to_array(tensor)
    return (tensor.data_type, tuple(arr.shape), arr.tobytes())


def _attr_signature(attr: onnx.AttributeProto) -> tuple:
    """Return a hashable, name-independent representation of an attribute value."""
    if attr.type == onnx.AttributeProto.INT:
        return (attr.name, "INT", attr.i)
    if attr.type == onnx.AttributeProto.INTS:
        return (attr.name, "INTS", tuple(attr.ints))
    if attr.type == onnx.AttributeProto.FLOAT:
        return (attr.name, "FLOAT", round(attr.f, 6))
    if attr.type == onnx.AttributeProto.FLOATS:
        return (attr.name, "FLOATS", tuple(round(f, 6) for f in attr.floats))
    if attr.type == onnx.AttributeProto.STRING:
        return (attr.name, "STRING", attr.s)
    if attr.type == onnx.AttributeProto.TENSOR:
        return (attr.name, "TENSOR", _tensor_signature(attr.t))
    # For other complex types, just record the type
    return (attr.name, str(attr.type))


def _node_fingerprint(node: onnx.NodeProto) -> tuple:
    """Return a hashable fingerprint for *node* that ignores tensor/node names.

    Captures: op_type, number of inputs/outputs, and sorted attribute signatures.
    """
    attrs = tuple(sorted(_attr_signature(a) for a in node.attribute))
    return (node.op_type, len(node.input), len(node.output), attrs)


# ---------------------------------------------------------------------------
# Canonical BFS traversal
# ---------------------------------------------------------------------------


def _build_producer_map(graph: onnx.GraphProto) -> dict[str, onnx.NodeProto]:
    return {out: node for node in graph.node for out in node.output if out}


def _canonical_order(graph: onnx.GraphProto) -> list[onnx.NodeProto]:
    """Return nodes in a canonical BFS order starting from graph outputs.

    The traversal walks backwards from each graph output through producers,
    then reverses the result to get a topological (inputs-first) ordering.
    Nodes unreachable from outputs are appended at the end in graph order.
    """
    producer = _build_producer_map(graph)
    visited: set[int] = set()
    order: list[onnx.NodeProto] = []
    queue: deque[str] = deque()

    # Seed with graph output tensor names
    for o in graph.output:
        queue.append(o.name)

    while queue:
        tensor = queue.popleft()
        node = producer.get(tensor)
        if node is None or id(node) in visited:
            continue
        visited.add(id(node))
        order.append(node)
        for inp in node.input:
            if inp:
                queue.append(inp)

    # Reverse: we walked output→input, but we want input→output order
    order.reverse()

    # Append any unreachable nodes (shouldn't happen in well-formed graphs)
    order.extend(node for node in graph.node if id(node) not in visited)

    return order


# ---------------------------------------------------------------------------
# Graph-level structural comparison
# ---------------------------------------------------------------------------


def graphs_structurally_equal(
    graph1: onnx.GraphProto,
    graph2: onnx.GraphProto,
) -> tuple[bool, list[str]]:
    """Compare two graphs structurally, ignoring tensor and node names.

    Returns ``(is_equal, list_of_differences)``.
    """
    diffs: list[str] = []

    # 1. Compare graph input/output counts and types
    if len(graph1.input) != len(graph2.input):
        diffs.append(f"Input count: {len(graph1.input)} vs {len(graph2.input)}")
    if len(graph1.output) != len(graph2.output):
        diffs.append(f"Output count: {len(graph1.output)} vs {len(graph2.output)}")

    # 2. Compare graph input/output shapes and dtypes (order-sensitive)
    for i, (io1, io2) in enumerate(zip(graph1.input, graph2.input)):
        tt1, tt2 = io1.type.tensor_type, io2.type.tensor_type
        if tt1.elem_type != tt2.elem_type:
            diffs.append(f"Input[{i}] dtype: {tt1.elem_type} vs {tt2.elem_type}")
        s1 = [d.dim_value for d in tt1.shape.dim] if tt1.HasField("shape") else None
        s2 = [d.dim_value for d in tt2.shape.dim] if tt2.HasField("shape") else None
        if s1 != s2:
            diffs.append(f"Input[{i}] shape: {s1} vs {s2}")

    for i, (io1, io2) in enumerate(zip(graph1.output, graph2.output)):
        tt1, tt2 = io1.type.tensor_type, io2.type.tensor_type
        if tt1.elem_type != tt2.elem_type:
            diffs.append(f"Output[{i}] dtype: {tt1.elem_type} vs {tt2.elem_type}")
        s1 = [d.dim_value for d in tt1.shape.dim] if tt1.HasField("shape") else None
        s2 = [d.dim_value for d in tt2.shape.dim] if tt2.HasField("shape") else None
        if s1 != s2:
            diffs.append(f"Output[{i}] shape: {s1} vs {s2}")

    # 3. Compare node count
    if len(graph1.node) != len(graph2.node):
        diffs.append(f"Node count: {len(graph1.node)} vs {len(graph2.node)}")

    # Compare initializer payloads name-independently. Most graph surgeries emit
    # shape/axis/repeats tensors as initializers, so their contents are part of
    # the graph semantics even when generated names differ.
    init1 = Counter(_tensor_signature(init) for init in graph1.initializer)
    init2 = Counter(_tensor_signature(init) for init in graph2.initializer)
    if init1 != init2:
        missing = list((init2 - init1).elements())[:5]
        extra = list((init1 - init2).elements())[:5]
        diffs.append(f"Initializer payloads differ; missing={missing}, extra={extra}")

    # 4. Compare op-type histogram
    ops1 = {}
    for n in graph1.node:
        ops1[n.op_type] = ops1.get(n.op_type, 0) + 1
    ops2 = {}
    for n in graph2.node:
        ops2[n.op_type] = ops2.get(n.op_type, 0) + 1
    if ops1 != ops2:
        diffs.append(f"Op histogram: {ops1} vs {ops2}")

    # 5. Compare canonical node fingerprint sequence
    order1 = _canonical_order(graph1)
    order2 = _canonical_order(graph2)

    if len(order1) != len(order2):
        diffs.append(f"Canonical order length: {len(order1)} vs {len(order2)}")
    else:
        for i, (n1, n2) in enumerate(zip(order1, order2)):
            fp1 = _node_fingerprint(n1)
            fp2 = _node_fingerprint(n2)
            if fp1 != fp2:
                diffs.append(
                    f"Node[{i}] mismatch: {fp1[0]}({fp1[1]}in,{fp1[2]}out,attrs={fp1[3]}) "
                    f"vs {fp2[0]}({fp2[1]}in,{fp2[2]}out,attrs={fp2[3]})"
                )

    return (len(diffs) == 0, diffs)


def assert_graphs_structurally_equal(
    graph1: onnx.GraphProto,
    graph2: onnx.GraphProto,
    msg: str = "",
) -> None:
    """Assert two graphs are structurally identical; raise ``AssertionError`` with diff details."""
    equal, diffs = graphs_structurally_equal(graph1, graph2)
    if not equal:
        detail = "\n  ".join(diffs[:20])  # cap at 20 diffs
        prefix = f"{msg}: " if msg else ""
        raise AssertionError(f"{prefix}Graphs differ structurally:\n  {detail}")
