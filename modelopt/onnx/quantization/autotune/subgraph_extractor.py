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

"""Extract standalone ONNX subgraphs from a full model using onnx_graphsurgeon.

Given boundary tensor names (inputs/outputs), marks them as the subgraph's
I/O, cleans up unreferenced nodes, runs shape inference, and serializes
the result to bytes for direct TensorRT consumption.
"""

import logging
from collections import deque
from typing import List, Set

import numpy as np
import onnx
import onnx_graphsurgeon as gs

logger = logging.getLogger(__name__)


def extract_subgraph(
    graph: gs.Graph,
    input_tensor_names: List[str],
    output_tensor_names: List[str],
) -> bytes:
    """Extract a standalone subgraph between specified boundary tensors.

    Args:
        graph: The full ONNX model graph (onnx_graphsurgeon).
        input_tensor_names: Tensor names to use as subgraph inputs.
        output_tensor_names: Tensor names to use as subgraph outputs.

    Returns:
        Serialized ONNX model bytes containing only the subgraph.

    Raises:
        ValueError: If boundary tensors cannot be found in the graph.
    """
    sub_graph = graph.copy()
    tensors = sub_graph.tensors()

    new_inputs = []
    for name in input_tensor_names:
        t = tensors.get(name)
        if t is None:
            raise ValueError(f"Input tensor '{name}' not found in graph")
        var = _ensure_variable(t, name)
        var.inputs.clear()
        new_inputs.append(var)

    new_outputs = []
    for name in output_tensor_names:
        t = tensors.get(name)
        if t is None:
            raise ValueError(f"Output tensor '{name}' not found in graph")
        var = _ensure_variable(t, name)
        new_outputs.append(var)

    sub_graph.inputs = new_inputs
    sub_graph.outputs = new_outputs
    sub_graph.cleanup().toposort()

    model = gs.export_onnx(sub_graph)
    try:
        model = onnx.shape_inference.infer_shapes(model)
    except Exception as e:
        logger.warning(f"Shape inference failed for subgraph, proceeding without: {e}")

    logger.debug(
        f"Extracted subgraph: {len(sub_graph.nodes)} nodes, "
        f"{len(new_inputs)} inputs, {len(new_outputs)} outputs"
    )
    return model.SerializeToString()


def extract_subgraph_by_nodes(
    graph: gs.Graph,
    node_names: List[str],
) -> bytes:
    """Extract a subgraph containing exactly the specified nodes.

    Automatically resolves boundary tensors by finding inputs produced
    outside the node set and outputs consumed outside the node set.

    Args:
        graph: The full ONNX model graph.
        node_names: List of ONNX node names to include.

    Returns:
        Serialized ONNX model bytes.
    """
    node_set = set(node_names)
    group_nodes = [n for n in graph.nodes if n.name in node_set]

    if not group_nodes:
        raise ValueError(f"None of the specified nodes found in graph")

    graph_output_names = {t.name for t in graph.outputs}
    input_tensors = []
    output_tensors = []
    seen_inputs = set()
    seen_outputs = set()

    for node in group_nodes:
        for inp in node.inputs:
            if isinstance(inp, gs.Constant):
                continue
            if not isinstance(inp, gs.Variable):
                continue
            if inp.name in seen_inputs:
                continue
            producer = inp.inputs[0] if inp.inputs else None
            if producer is None or producer.name not in node_set:
                input_tensors.append(inp.name)
                seen_inputs.add(inp.name)

        for out in node.outputs:
            if not isinstance(out, gs.Variable):
                continue
            if out.name in seen_outputs:
                continue
            if out.name in graph_output_names:
                output_tensors.append(out.name)
                seen_outputs.add(out.name)
                continue
            for consumer in out.outputs:
                if consumer.name not in node_set:
                    output_tensors.append(out.name)
                    seen_outputs.add(out.name)
                    break

    if not input_tensors:
        logger.warning(
            f"No boundary inputs found for nodes {node_names[:3]}..., "
            "using graph inputs that reach these nodes"
        )
        input_tensors = _find_reachable_graph_inputs(graph, group_nodes)

    if not output_tensors:
        for node in group_nodes:
            for out in node.outputs:
                if isinstance(out, gs.Variable) and out.name not in seen_outputs:
                    output_tensors.append(out.name)
                    seen_outputs.add(out.name)

    return extract_subgraph(graph, input_tensors, output_tensors)


def _ensure_variable(tensor, name: str) -> gs.Variable:
    """Ensure the tensor is a gs.Variable, creating one if needed."""
    if isinstance(tensor, gs.Variable):
        return tensor
    var = gs.Variable(name=name, dtype=np.float32)
    return var


def _find_reachable_graph_inputs(
    graph: gs.Graph,
    target_nodes: List[gs.Node],
) -> List[str]:
    """BFS backward from target_nodes to find graph inputs that feed into them."""
    graph_input_names = {t.name for t in graph.inputs}
    visited = set()
    queue = deque()
    result = []

    for node in target_nodes:
        for inp in node.inputs:
            if isinstance(inp, gs.Variable) and inp.name not in visited:
                visited.add(inp.name)
                queue.append(inp)

    while queue:
        tensor = queue.popleft()
        if tensor.name in graph_input_names:
            result.append(tensor.name)
            continue
        for producer in tensor.inputs:
            if not isinstance(producer, gs.Node):
                continue
            for inp in producer.inputs:
                if isinstance(inp, gs.Variable) and inp.name not in visited:
                    visited.add(inp.name)
                    queue.append(inp)

    return result
