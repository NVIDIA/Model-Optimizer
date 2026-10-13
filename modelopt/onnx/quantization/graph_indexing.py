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

"""Graph indexing and pattern matching for ONNX quantization."""

import re
from collections import defaultdict

import numpy as np
import onnx
import onnx_graphsurgeon as gs
from onnx_graphsurgeon.ir.graph import Graph
from onnx_graphsurgeon.ir.node import Node
from onnx_graphsurgeon.ir.tensor import Constant, Tensor

from modelopt.onnx.op_types import get_copy_ops
from modelopt.onnx.utils import get_child_nodes, get_parent_nodes

__all__ = [
    "expand_node_names_from_patterns",
    "find_mha_partitions",
    "get_fusible_backbone",
    "get_tensor_consumer_node_indices",
    "get_tensor_consumer_nodes",
    "get_tensor_from_name",
    "get_tensor_producer_nodes",
    "has_const_input",
    "has_path_type",
    "is_const_input",
    "match_fp8_mha_pattern",
]


def is_const_input(tensor: Tensor) -> bool:
    """Returns whether the given tensor is an initializer or produced by const-foldable nodes."""
    if isinstance(tensor, Constant):
        return True

    # Tensor is a graph input variable
    if len(tensor.inputs) == 0:
        return False

    producer_node = tensor.inputs[0]  # Generally tensors has single producer
    if producer_node.op in ["Constant", "Identity"]:
        return True

    # Second axes input to Squeeze/Unsqueeze is a constant, we need to check the first input
    if producer_node.op in ["Squeeze", "Unsqueeze"] and is_const_input(producer_node.inputs[0]):
        return True

    # Const -> Clip -> Exp -> Mul pattern matching for swin_v2
    if producer_node.op == "Exp":
        clip_node = producer_node.i()
        if clip_node.op == "Clip" and has_const_input(clip_node):
            return True

    return False


def has_const_input(node: Node) -> bool:
    """Returns whether the given node has any constant input."""
    return any(is_const_input(tensor) for tensor in node.inputs)


def has_path_type(
    node: Node,
    graph: Graph,
    path_type: list[str],
    is_forward: bool,
    wild_card_types: list[str] = [],
    path_nodes: list[Node] = [],
) -> bool:
    """Checks if the given node is start/end of a given forward/backward path type.

    Note, Path can be forward or backward wrt a node depending on the next level nodes.
    Additionally, this method can work with optional nodes and collect the traversed path.

    Args:
        node: Start node of the path.
        graph: ONNX model graph.
        path_type: Path types to match from the given node.
        is_forward: Whether to match forward or backward path.
        wild_card_types: Wild card types, these type of nodes are skipped and not matched with the path_type.
        path_nodes: Accumulated nodes in the matched path.

    Returns:
        Bool, whether the given node is start/end of the given forward/backward path type.
    """
    optional_path_types = ["BiasAdd", "ConstMul"]
    if not path_type:
        # All types matched
        return True

    # Current node type and special type conversion for optional BiasAdd and ConstMul
    # Note, matching path with Add/Mul type nodes with const input will fail
    node_type = node.op
    if node_type == "Add" and has_const_input(node):
        node_type = "BiasAdd"
    elif node_type == "Mul" and has_const_input(node):
        node_type = "ConstMul"

    # Special type conversion from NonBiasAdd to Add if all Add inputs are non-constant
    if node_type == "Add" and path_type[0] == "NonBiasAdd":
        path_type[0] = "Add"

    # Check if current non-wild node type does not match the expected path type
    # And if path type is not optional (ex. BiasAdd)
    is_match = (node_type == path_type[0]) or (node.op == path_type[0])
    is_wild_match = node_type in wild_card_types
    if not is_match and not is_wild_match and (path_type[0] not in optional_path_types):
        return False

    # Add current node name in the path
    if is_match:
        path_nodes.append(node)

    # If current node type matches the expected path type or path type is optional (ex. BiasAdd), we have a type match
    # Update the remaining path types to match
    next_path_type = path_type[:]

    # Non-repeatable optional types should be consumed
    if is_match or (path_type[0] in ["BiasAdd", "ConstMul"]):
        next_path_type = path_type[1:]

    # If current node is not wild card and didn't match, go ahead and match with the
    # remaining path types starting with the current node
    if not is_match and not is_wild_match:
        assert path_type[0] in optional_path_types
        return has_path_type(
            node,
            graph,
            next_path_type,
            is_forward,
            wild_card_types,
            path_nodes,
        )

    next_level_nodes = get_child_nodes(node) if is_forward else get_parent_nodes(node)

    # Check if any child (forward path) or parent (backward path) can match the remaining path types
    for next_node in next_level_nodes:
        sub_path = []
        if has_path_type(next_node, graph, next_path_type, is_forward, wild_card_types, sub_path):
            path_nodes.extend(sub_path)
            return True

    # Path type matches if there is no remaining types to match
    return not next_path_type


def get_fusible_backbone(node: Node, graph: Graph) -> Node | None:
    """Returns the linear backbone node for a given node if it matches the pattern.

    TensorRT fuses convolution with BN, Relu, MaxPool etc. when in some specific pattern.
    This rule tries to match some of those patterns.
    Note. BiasAdd and ConstMul are optional in path types.

    Args:
        node: Start node of the pattern.
        graph: ONNX model graph.

    Returns:
        Backbone node of the given node, None if not found.
    """

    def _get_backbone(root: Node):
        if root.op in ["Conv", "ConvTranspose"]:
            return root

        for tensor in root.inputs:
            if not isinstance(tensor, Constant) and tensor.inputs:
                parent_node = tensor.inputs[0]
                bb = _get_backbone(parent_node)
                if bb:
                    return bb

    fusible_linear_path_types = []
    for conv_type in ["Conv", "ConvTranspose"]:
        fusible_linear_path_types += [
            ["BiasAdd", "ConstMul", conv_type],
            ["Relu", "BiasAdd", "ConstMul", conv_type],
            ["BatchNormalization", "BiasAdd", conv_type],
            ["Relu", "BatchNormalization", "BiasAdd", conv_type],
            ["MaxPool", "Relu", "BatchNormalization", "BiasAdd", conv_type],
            ["Mul", "Sigmoid", "BatchNormalization", conv_type],
        ]
    for idx, path_type in enumerate(fusible_linear_path_types):
        if has_path_type(node, graph, path_type, is_forward=False, wild_card_types=get_copy_ops()):
            return _get_backbone(node)

    return None


def get_tensor_from_name(graph: onnx.GraphProto, tensor_name: str) -> onnx.ValueInfoProto | None:
    """Returns a ValueInfoProto given a tensor name.

    Args:
        graph: ONNX model graph
        tensor_name: String with tensor name.

    Returns:
        onnx.ValueInfoProto: actual graph tensor.
    """
    # Search in inputs
    vi = next((vi for vi in graph.input if vi.name == tensor_name), None)
    # If not found, search in outputs
    if vi is None:
        vi = next((vi for vi in graph.output if vi.name == tensor_name), None)
    # If not found, search in value_info (intermediate tensors)
    if vi is None:
        vi = next((vi for vi in graph.value_info if vi.name == tensor_name), None)
    return vi


def get_tensor_producer_nodes(
    graph: onnx.GraphProto,
    get_initializer_producers: bool = False,
) -> dict[str, onnx.NodeProto]:
    """Returns a dictionary of tensor name and their producer node object mapping.

    Note. we create a special Root type node as external inputs producer for ease of implementation.

    Args:
        graph: ONNX model graph.

    Returns:
        Dictionary, key is tensor name and value is their producer node object
    """
    # Create a dictionary to store tensor producer nodes
    tensor_producers = defaultdict(None)

    # Special Root type producer node
    root_node = onnx.helper.make_node(
        op_type="Root",
        inputs=[],
        outputs=[i.name for i in graph.input],
        name="root_0",
    )

    input_names = [graph_input.name for graph_input in graph.input]
    initializer_names = [initializer.name for initializer in graph.initializer]
    external_input_names = list(np.setdiff1d(input_names, initializer_names))

    # Note. We are marking external inputs as non-constant by adding a parent,
    # so that we can quantize the first node of the graph if appropriate
    for graph_input in external_input_names:
        tensor_producers[graph_input] = root_node

    # Traverse the graph to find producer nodes for each tensor
    for node in graph.node:
        for output_name in node.output:
            tensor_producers[output_name] = node

    if get_initializer_producers:
        for initializer in graph.initializer:
            tensor_producers[initializer.name] = initializer

    return tensor_producers


def get_tensor_consumer_nodes(
    graph: onnx.GraphProto,
) -> dict[str, list[onnx.NodeProto]]:
    """Returns a dictionary of tensor name and their consumer node object mapping.

    Args:
        graph: ONNX model graph.

    Returns:
        Dictionary, key is tensor name and value is their consumer node object
    """
    # Create a dictionary to store tensor consumer nodes
    tensor_consumers = defaultdict(list)

    # Traverse the graph to find consumer nodes for each tensor
    for node in graph.node:
        for input_name in node.input:
            tensor_consumers[input_name].append(node)

    return tensor_consumers


def get_tensor_consumer_node_indices(graph: onnx.GraphProto | gs.Graph) -> dict[str, list[int]]:
    """Build a mapping from tensor names to the indices of nodes that use them.

    Args:
        graph: ONNX GraphSurgeon graph to analyze
    Returns:
        Dictionary mapping tensor names to lists of node indices that consume them
    """
    tensor_consumer_map: dict[str, list[int]] = defaultdict(list)
    nodes = graph.nodes if isinstance(graph, gs.Graph) else graph.node
    for node_idx, node in enumerate(nodes):
        inputs = node.inputs if isinstance(node, gs.Node) else node.input
        for tensor in inputs:
            tensor_name = tensor
            if isinstance(tensor, str):
                tensor_name = tensor
            elif hasattr(tensor, "name") and isinstance(tensor.name, str):
                tensor_name = tensor.name
            tensor_consumer_map[tensor_name].append(node_idx)
    return tensor_consumer_map


def expand_node_names_from_patterns(
    graph: onnx.GraphProto | Graph, name_patterns: list[str] | None = None
) -> list[str]:
    """Expand the node names from the given patterns."""
    if not name_patterns:
        return []
    node_list = getattr(graph, "nodes", None) or getattr(graph, "node", None) or []

    matched_node_names = []
    for pattern in name_patterns:
        matched_node_names.extend([node.name for node in node_list if re.match(pattern, node.name)])
    return matched_node_names


def find_mha_partitions(graph):
    """Match MHA: BMM1 -> ... -> Softmax -> ... -> BMM2."""
    mha_chain_type = ["MatMul", "Softmax", "MatMul"]
    wild_card_types = [
        "Div",
        "Mul",
        "ConstMul",
        "Add",
        "BiasAdd",
        "Reshape",
        "Transpose",
        "Flatten",
        "Cast",
    ]
    mha_partitions = []
    for node in graph.nodes:
        if node.op == "MatMul":
            mha_partition = []
            if has_path_type(
                node, graph, mha_chain_type, True, wild_card_types, mha_partition
            ) and (
                len(mha_partition) == 3
                and mha_partition[0].op == "MatMul"
                and mha_partition[2].op == "MatMul"
            ):
                mha_partitions.append(mha_partition)

    return mha_partitions


def match_fp8_mha_pattern(graph: Graph, softmax_op: Node, has_fp8_qdq: bool) -> list[Node]:
    """Match FP8 fMHA v2 with the given softmax_op.

    If has_fp8_qdq == True, we match this FP8 fMHA v2 pattern:
    Q -> DQ -> BMM1 -> (Mul/Div) -> (Add) -> Softmax -> (Cast) -> Q -> DQ -> BMM2 -> Q -> DQ.
    If has_fp8_qdq == False, we match this FP8 fMHA v2 pattern:
    BMM1 -> (Mul/Div) -> (Add) -> Softmax -> (Cast) -> BMM2.

    Args:
        graph:
            The graph to match FP8 MHA pattern.
        softmax_op:
            The softmax op of FP8 MHA we want to match.
        nodes_to_exclude:
            List of Nodes to exclude from quantization.
        has_fp8_qdq:
            If True, match the FP8 MHA with Q/DQs.
            Else, match the FP8 MHA without Q/DQs.

    Returns:
        List of BMM1 node, Softmax node and BMM2 node.
    """
    if has_fp8_qdq:
        softmax_bmm1_chain_types = [
            ["Softmax", "MatMul", "DequantizeLinear", "QuantizeLinear"],
            ["Softmax", "Add", "MatMul", "DequantizeLinear", "QuantizeLinear"],
            ["Softmax", "Div", "MatMul", "DequantizeLinear", "QuantizeLinear"],
            ["Softmax", "Mul", "MatMul", "DequantizeLinear", "QuantizeLinear"],
            ["Softmax", "Add", "Div", "MatMul", "DequantizeLinear", "QuantizeLinear"],
            ["Softmax", "Add", "Mul", "MatMul", "DequantizeLinear", "QuantizeLinear"],
        ]
        softmax_bmm2_chain_type = [
            "Softmax",
            "QuantizeLinear",
            "DequantizeLinear",
            "MatMul",
            "QuantizeLinear",
            "DequantizeLinear",
        ]
    else:
        softmax_bmm1_chain_types = [
            ["Softmax", "MatMul"],
            ["Softmax", "Add", "MatMul"],
            ["Softmax", "Div", "MatMul"],
            ["Softmax", "Mul", "MatMul"],
            ["Softmax", "Add", "Div", "MatMul"],
            ["Softmax", "Add", "Mul", "MatMul"],
        ]
        softmax_bmm2_chain_type = [
            "Softmax",
            "MatMul",
        ]
    wild_card_types = [
        "Reshape",
        "Transpose",
        "Flatten",
        "Cast",
    ]

    bmm1_index = 1
    bmm2_index = 7 if has_fp8_qdq else 3
    # Maps chain_idx to bmm position offsets for indexing the fp8_mha_partition
    # chain_idx=0 -> offset=0
    # chain_idx=1,2,3 -> offset=1
    # chain_idx=4,5 -> offset=2
    bmm_offset_map = {0: 0, 1: 1, 2: 1, 3: 1, 4: 2, 5: 2}
    for chain_idx, softmax_bmm1_chain_type in enumerate(softmax_bmm1_chain_types):
        fp8_mha_partition = []
        if has_path_type(
            softmax_op, graph, softmax_bmm1_chain_type, False, wild_card_types, fp8_mha_partition
        ) and has_path_type(
            softmax_op, graph, softmax_bmm2_chain_type, True, wild_card_types, fp8_mha_partition
        ):
            offset = bmm_offset_map.get(chain_idx)
            bmm1_node = fp8_mha_partition[bmm1_index + offset]
            bmm2_node = fp8_mha_partition[bmm2_index + offset]
            assert bmm1_node.op == "MatMul" and bmm2_node.op == "MatMul"
            return [bmm1_node, softmax_op, bmm2_node]
    return []
