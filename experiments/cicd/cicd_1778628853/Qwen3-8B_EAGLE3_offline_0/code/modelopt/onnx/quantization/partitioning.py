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

"""Utilities related to partitioning the ONNX model to place QDQ nodes."""

from onnx_graphsurgeon.ir.graph import Graph
from onnx_graphsurgeon.ir.node import Node

from modelopt.onnx.logging_config import logger
from modelopt.onnx.op_types import (
    is_copy_op,
    is_linear_op,
    is_pointwise_or_elementwise_op,
    is_pooling_or_window_op,
)
from modelopt.onnx.quantization.graph_utils import (
    get_fusible_backbone,
    has_const_input,
    has_path_type,
    is_const_input,
)
from modelopt.onnx.utils import get_child_nodes, get_parent_nodes, get_variable_inputs


def _build_fusible_partition(
    cur_node: Node,
    fusible_partition: list[Node],
    partitioned_nodes: set[str],
    non_residual_inputs: dict[str, str],
    graph: Graph,
) -> None:
    """Traverses the graph starting from cur_node and updates the fusible_partition list.

    Add a nodes to the partition if any of these holds:
    1. The node is a unary or binary pointwise operation or a copy op and fusible by cask
    2. The node is BN and/or Relu and fusible with preceding Conv op (Conv-Act fusion)
    3. The node is MaxPool following a Conv-Act pattern (Conv-Act-Pool fusion)
    4. The node is a residual Add and fusible with current partition

    Args:
        cur_node: Current candidate node for the partition.
        fusible_partition: Current fusible partition.
        partitioned_nodes: Set of already partitioned nodes.
        non_residual_inputs: Non-residual input map.
        graph: ONNX model graph.

    Returns:
        Backbone node of the given pointwise op, None if not found.
    """

    def _is_on_non_residual_path(node: Node) -> bool:
        return bool(
            node.op == "Add"  # Input node should be an Add node
            # The Add node should have a non-residual input
            and non_residual_inputs[node.name]
            # Input from the current node is non-residual
            and cur_node.outputs[0].name == non_residual_inputs[node.name]
        )

    def _get_partition_node_outputs() -> list[str]:
        # Collect tensor names produced by nodes in fusible_partition
        # TODO: cache sub-partition outputs and append after them
        partition_node_outputs = []
        for partition_node in fusible_partition:
            partition_node_outputs.extend([output.name for output in partition_node.outputs])

        return partition_node_outputs

    def _is_cask_fusible(node: Node, partition_node_outputs: list[str]) -> bool:
        for tensor in node.inputs:
            if tensor.name not in partition_node_outputs and not is_const_input(tensor):
                return False
        return True

    def _is_fusible_mul(mul_node: Node) -> bool:
        # Don't consider Mul as fusible if it has the indicated ancestors.
        # Otherwise, this causes regressions in:
        #  - densenet-12 and inception-v2-9 (dangling constants): [mul_node.op, "Unsqueeze"]
        #  - faster_vit: ["Mul", "Add", "Tanh", "Mul", "Add", "Mul", "Pow"]
        #  This improves perf in various models:  mobilenet_v3, vovnet19b, coatnet_0, regnety_040.

        var_inps = get_variable_inputs(mul_node)
        if len(var_inps) <= 1:
            return True

        # Conv-Sigmoid-Mul chain is fusible
        fusible_patterns = [
            ["Mul", "Sigmoid", "Conv"],
            ["Mul", "Sigmoid", "BatchNormalization", "Conv"],
            ["Mul", "Sigmoid", "Add"],
        ]
        if any(has_path_type(mul_node, graph, p, is_forward=False) for p in fusible_patterns):
            return True

        non_fusible_patterns = [["Mul", "Sigmoid"], ["Mul", "HardSigmoid"]]
        return not any(
            has_path_type(mul_node, graph, p, is_forward=False) for p in non_fusible_patterns
        )

    # Check the Mul nodes for their fusion compatibility
    if cur_node.op == "Mul" and not _is_fusible_mul(cur_node):
        return

    # Add current node to the partition
    fusible_partition.append(cur_node)
    partitioned_nodes.add(cur_node.name)

    # If on non-residual path, return after adding the node to the partition
    # TODO: can the compiler fuse pointwise ops followed by residual Add?
    if cur_node.op == "Add" and non_residual_inputs[cur_node.name]:
        return

    consumer_nodes = get_child_nodes(cur_node)
    partition_node_outputs = _get_partition_node_outputs()

    # TODO: traverse consumer nodes in topologically sorted order
    for consumer_node in consumer_nodes:
        if consumer_node.name in partitioned_nodes:
            continue

        if (
            (
                is_copy_op(consumer_node.op)
                and _is_cask_fusible(consumer_node, partition_node_outputs)
            )
            or (
                is_pointwise_or_elementwise_op(consumer_node.op)
                and _is_cask_fusible(consumer_node, partition_node_outputs)
            )
            or (
                consumer_node.op in ["BatchNormalization", "Relu", "MaxPool"]
                and get_fusible_backbone(consumer_node, graph)
            )
            or _is_on_non_residual_path(consumer_node)
        ):
            # DFS with the consumer and find more nodes for the partition
            _build_fusible_partition(
                consumer_node,
                fusible_partition,
                partitioned_nodes,
                non_residual_inputs,
                graph,
            )


def _find_quantizable_kgen_partitions(
    graph: Graph,
    cask_fusible_partitions: list[list[Node]],
) -> list[list[Node]]:
    """Collect the partitions that can have quantized inputs, e.g. `XYZ` in `Conv-[epilogue]-XYZ`.

    Args:
        graph: Onnx model graph.
        cask_fusible_partitions: List of partitions that are fusible by CASK with Conv/MatMul backbone.

    Returns:
        List of KGEN partitions that prefer quantized inputs.
    """
    node_type_prefer_quantized_input = [
        "Resize",
        "Upsample",
    ]

    # collect the outputs from the cask fusible partitions.
    # assumption: the nodes are topologically sorted in the lists.
    cask_fusion_outputs = [partition[-1].outputs[0] for partition in cask_fusible_partitions]

    all_fusible_partitions = []
    for node in graph.nodes:
        if node.op not in node_type_prefer_quantized_input:
            continue
        if node.inputs[0] in cask_fusion_outputs:
            all_fusible_partitions.append([node])

    return all_fusible_partitions


def find_fusible_partitions(
    graph: Graph,
    partitioned_nodes: set[str],
    non_residual_inputs: dict[str, str],
) -> tuple[list[list[Node]], list[list[Node]]]:
    """Traverses the graph and collects all cask/kgen fusible partitions.

    Args:
        graph: Onnx model graph.
        partitioned_nodes: Set of already partitioned nodes.
        non_residual_inputs: Non-residual input map.

    Returns:
        List of partitions that are fusible by CASK with Conv/MatMul backbone.
        List of KGEN partitions with pointwise ops only.
    """
    logger.info("Building KGEN/CASK targeted partitions")

    def _partition_helper(fusible_root_type_checker):
        all_fusible_partitions = []  # Collects all individual partitions
        for node in graph.nodes:
            # Check if the node is already in some other partition
            if node.name in partitioned_nodes:
                continue

            # Start a partition with a linear op
            if not fusible_root_type_checker(node.op):
                continue

            # Try building a partition starting with this current linear op
            fusible_partition = []
            _build_fusible_partition(
                node,
                fusible_partition,
                partitioned_nodes,
                non_residual_inputs,
                graph,
            )

            # Gather the non-empty partitions
            if fusible_partition:
                partitioned_nodes.update([node.name for node in fusible_partition])
                all_fusible_partitions.append(fusible_partition)

        return all_fusible_partitions

    cask_fusible_partitions = _partition_helper(is_linear_op)

    kgen_partitions = _partition_helper(is_pointwise_or_elementwise_op)
    kgen_partitions += _find_quantizable_kgen_partitions(graph, cask_fusible_partitions)

    return cask_fusible_partitions, kgen_partitions


def get_skipped_output_layers(graph: Graph, partially_quantizable_nodes: list[Node]) -> list[str]:
    """Returns the name of the non-quantizable output layers."""
    # TODO: see if input producer is already quantized or not
    # TODO: filter out input layers if consumer is not quantized already
    output_layers = []
    partially_quantizable_node_names = {node.name for node in partially_quantizable_nodes}
    graph_output_names = [tensor.name for tensor in graph.outputs]

    for node in graph.nodes:
        output_layers.extend(
            [
                node.name
                for tensor in node.outputs
                if tensor.name in graph_output_names
                and (
                    node.op not in ["Conv", "Gemm", "MatMul"]
                    and node.name not in partially_quantizable_node_names
                )
            ]
        )

    return output_layers


def find_quantizable_nodes(
    graph: Graph,
    nodes_to_quantize: list[Node],
    partitioned_nodes: set[str],
    quantizable_op_types: list[str],
) -> list[Node]:
    """Return the graph ops which are quantizable but not partitioned yet."""
    logger.info(f"Finding quantizable nodes. Initial nodes to quantize: {len(nodes_to_quantize)}")

    # TODO: Check if any BatchNormalization is un-partitioned
    # Note. Maxpool quantization has +/-
    # Note. Prologue fusion is not handled, so some pointwise ops might be unnecessarily quantized
    def _has_quantizable_consumer(node: Node, quantizable_node_set: set[str]) -> bool:
        children = get_child_nodes(node)
        for child_node in children:
            if (child_node.name in quantizable_node_set) or (
                is_copy_op(child_node.op)
                and _has_quantizable_consumer(child_node, quantizable_node_set)
            ):
                return True

        return False

    def _has_quantizable_producer(node: Node, quantizable_node_set: set[str]) -> bool:
        parents = get_parent_nodes(node)
        for parent_node in parents:
            if (parent_node.name in quantizable_node_set) or (
                (is_copy_op(parent_node.op) or parent_node.op == "Cast")
                and _has_quantizable_producer(parent_node, quantizable_node_set)
            ):
                return True
        return False

    quantizable_nodes = []
    pooling_and_window_ops = []
    for node in graph.nodes:
        if node.name in partitioned_nodes or node.op not in quantizable_op_types:
            continue

        # Collect pooling and window ops for second pass
        # as they need to check their neighbor's quantization status
        if is_pooling_or_window_op(node.op):
            pooling_and_window_ops.append(node)
            logger.debug(f"Added pooling/window op to second pass: {node.name} ({node.op})")
            continue

        if is_pointwise_or_elementwise_op(node.op) and has_const_input(node):
            logger.debug(f"Skipping pointwise op with const input: {node.name}")
            continue

        quantizable_nodes.append(node)
        logger.debug(f"Added quantizable node: {node.name} ({node.op})")

    logger.info(f"Found {len(pooling_and_window_ops)} pooling/window ops")

    quantizable_node_set = set(
        [node.name for node in nodes_to_quantize] + [node.name for node in quantizable_nodes]
    )

    # TODO: Add or _has_quantizable_producer, ex. inception-v1-12.onnx
    quantizable_nodes.extend(
        node
        for node in pooling_and_window_ops
        if _has_quantizable_consumer(node, quantizable_node_set)
        or _has_quantizable_producer(node, quantizable_node_set)
    )

    return quantizable_nodes


def find_hardcoded_patterns(graph: Graph) -> list[list[Node]]:
    """Finds some non-quantizable pre-defined patterns!.

    Note. matching this tail pattern causes MTL_v1 -5.5%
    ["ReduceSum", "Add", "Div", "Mul", "ReduceSum", "Sub", "Pow", "Mul", "ReduceSum", "Sqrt"]
    """
    gelu = ["Div", "Erf", "Add", "Mul", "Mul"]

    matched_node_names = []
    for node in graph.nodes:
        for path_type in [gelu]:
            path_nodes = []
            if has_path_type(
                node,
                graph,
                path_type,
                is_forward=True,
                wild_card_types=[],
                path_nodes=path_nodes,
            ):
                matched_node_names.extend(path_nodes)

    return [matched_node_names]


def find_layer_norm_partitions(graph: Graph) -> list[list[Node]]:
    """Finds the layer norm patterns in the graph."""
    # The most common LayerNorm implementation looks like this:
    # t -> ReduceMean -> Sub -> Pow -> ReduceMean -> Add -> Sqrt -> Div -> Mul -> Add -> output
    #   \________________/  \______________________________________/
    # For simplicity, we do not match the connection between Sub and Div nodes.
    layer_norm_chain_types = [
        "ReduceMean",
        "Sub",
        "Pow",
        "ReduceMean",
        "Add",
        "Sqrt",
        "Div",
        "Mul",
        "Add",
    ]
    mean_var_norm_chain_types = layer_norm_chain_types[:-2]
    wild_card_types = ["Cast"]
    layer_norm_partitions = []

    for node in graph.nodes:
        layer_norm_partition = []
        if node.op == "LayerNormalization":
            layer_norm_partitions.append([node])
            logger.debug(f"Found direct LayerNormalization node: {node.name}")
        elif node.op == layer_norm_chain_types[0] and has_path_type(
            node, graph, layer_norm_chain_types, True, wild_card_types, layer_norm_partition
        ):
            layer_norm_partitions.append(layer_norm_partition)
            logger.debug(f"Found layer norm pattern starting at: {node.name}")
        elif node.op == mean_var_norm_chain_types[0] and has_path_type(
            node, graph, mean_var_norm_chain_types, True, wild_card_types, layer_norm_partition
        ):
            layer_norm_partitions.append(layer_norm_partition)
            logger.debug(f"Found mean-var norm pattern starting at: {node.name}")

    logger.info(f"Found {len(layer_norm_partitions)} layer norm partitions")
    return layer_norm_partitions


def find_mha_partitions(graph: Graph) -> list[list[Node]]:
    """Finds the MHA core (QK_AV) patterns in the graph that should not be quantized.

    A common MHA implementation looks like this:
    t -> MatMul -> (optional) Pointwise ops (such as Add, Mul, Sub) with constant input -> Softmax -> MatMul -> output
    Patterns that do not look like that should not be quantized (at least for now).
    """
    mha_chain_types = [
        [
            "MatMul",
            "NonBiasAdd",  # without constant inputs
            "Softmax",
            "MatMul",
        ],
    ]
    mha_partitions = []

    for node in graph.nodes:
        if node.op == "MatMul":
            for chain_type in mha_chain_types:
                mha_partition = []
                if has_path_type(node, graph, chain_type, True, [], mha_partition):
                    mha_partitions.append(mha_partition)

    logger.info(f"Found {len(mha_partitions)} MHA (QK_AV) Patterns")
    return mha_partitions


def find_non_quantizable_partitions_from_patterns(graph: Graph) -> list[list[str]]:
    """Finds fusible partition from fixed patterns.

    Certain fused kernel counterpart is often a subgraph of native ops in onnx.
    Those patterns are identified here and quantized to match compiler expectation.
    """
    hard_coded_partitions = find_hardcoded_patterns(graph)
    layer_norm_partitions = find_layer_norm_partitions(graph)
    mha_partitions = find_mha_partitions(graph)

    partitions = [
        [node.name for node in partition_nodes]
        for partition_nodes in hard_coded_partitions + layer_norm_partitions + mha_partitions
    ]

    return partitions
