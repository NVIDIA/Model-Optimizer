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

"""Q/DQ graph policy for ONNX quantization."""

import re
from typing import Any, cast

import onnx
import onnx_graphsurgeon as gs
from onnx_graphsurgeon.ir.graph import Graph
from onnx_graphsurgeon.ir.node import Node
from onnx_graphsurgeon.ir.tensor import Tensor

from modelopt.onnx.logging_config import logger
from modelopt.onnx.op_types import is_copy_op, is_linear_op
from modelopt.onnx.quantization.graph_indexing import (
    expand_node_names_from_patterns,
    get_fusible_backbone,
    get_tensor_consumer_nodes,
    has_const_input,
    has_path_type,
    is_const_input,
)
from modelopt.onnx.utils import find_lowest_common_ancestor, get_child_nodes, get_parent_nodes

DEFAULT_GATHER_BLOCK_SIZE = 32
DEFAULT_GATHER_QUANTIZE_AXIS = None

__all__ = [
    "build_non_residual_input_map",
    "classify_partially_quantized_weighted_ops",
    "classify_partition_nodes",
    "filter_quantizable_kgen_heads",
    "find_conv_to_layernorm_nodes",
    "get_concat_eliminated_tensors",
    "get_layer_info",
    "get_layer_precision_mapping",
    "get_resize_scales",
    "print_stat",
    "remove_partial_input_qdq",
    "should_quantize_to_8bit",
    "validate_8bit_layers",
]


def _is_following_cask_partition(
    node: Node, cask_partition_nodes: set[str], max_depth: int = 10
) -> bool:
    """Check if a CASK fusible partition can be reached by traversing backward through copy ops.

    Args:
        node: The node to check.
        cask_partition_nodes: Set of node names belonging to CASK partitions.
        max_depth: Maximum recursion depth to guard against pathological graphs.

    Returns:
        True if the node belongs to or follows a CASK partition through copy ops.
    """
    if node.name in cask_partition_nodes:
        return True

    if max_depth <= 0 or not is_copy_op(node.op):
        return False

    parent_nodes = get_parent_nodes(node)
    if len(parent_nodes) == 0:
        return False

    return all(
        _is_following_cask_partition(parent, cask_partition_nodes, max_depth - 1)
        for parent in parent_nodes
    )


def find_conv_to_layernorm_nodes(
    graph: Graph,
    cask_fusible_partitions: list[list[Node]],
) -> list[Node]:
    """Find LayerNormalization nodes whose input comes from a CASK (Conv) partition.

    When a Conv's output feeds into a LayerNormalization, the Conv output should be
    quantized to enable faster INT8 kernels in TRT. This function detects such patterns
    and returns the LayerNormalization nodes that should be added to the quantizable
    nodes list so that Q/DQ pairs are inserted on their input (i.e. the Conv output).

    Args:
        graph: ONNX model graph.
        cask_fusible_partitions: List of CASK fusible partitions.

    Returns:
        List of LayerNormalization nodes that consume CASK partition outputs.
    """
    cask_partition_nodes: set[str] = set()
    for partition in cask_fusible_partitions:
        cask_partition_nodes.update(node.name for node in partition)

    conv_to_ln_nodes = []
    for node in graph.nodes:
        if node.op != "LayerNormalization":
            continue

        # Check if the first input (activation) comes from a CASK partition
        # possibly through copy ops (Reshape, Transpose, etc.)
        inp_tensor = node.inputs[0]
        if inp_tensor.inputs:
            producer = inp_tensor.inputs[0]
            if _is_following_cask_partition(producer, cask_partition_nodes):
                conv_to_ln_nodes.append(node)
                logger.debug(
                    f"Found Conv->LayerNorm pattern: LayerNorm node '{node.name}' "
                    f"consumes CASK partition output"
                )

    logger.info(f"Found {len(conv_to_ln_nodes)} Conv->LayerNorm patterns to quantize")
    return conv_to_ln_nodes


def filter_quantizable_kgen_heads(
    cask_fusible_partitions: list[list[Node]],
    kgen_partitions: list[list[Node]],
    quantizable_op_types: list[str],
    graph: Graph,
) -> tuple[list[Node], list[tuple[Node, Node, str]]]:
    """Returns the list of kgen head names if it follows a CASK partition."""
    cask_partition_nodes: set[str] = set()
    for partition in cask_fusible_partitions:
        cask_partition_nodes.update(node.name for node in partition)

    cask_partition_heads = [partition[0] for partition in cask_fusible_partitions]

    def _is_mha_epilogue_pattern(node: Node, graph: Graph):
        if head_node.op != "Add":
            return False

        # Below are valid patterns:
        # (1)
        # Add -> Softmax -> MatMul
        #
        # (2)
        # Add -> Flatten -> Softmax -> Reshape -> MatMul
        #      \----------Shape-----/
        #
        mha_epilogue_path = ["Softmax", "MatMul"]
        wild_card_types = ["Flatten", "Reshape"]
        add_children = get_child_nodes(node)

        for child in add_children:
            if has_path_type(
                child,
                graph,
                mha_epilogue_path,
                is_forward=True,
                wild_card_types=wild_card_types,
            ):
                return True

        return False

    def _has_other_quantizable_consumer(
        tensor: Tensor, quantizable_kgen_heads: list[Node], head_name: str
    ):
        # Note. this is kinda approximate analysis,
        # all quantizable kgen heads may haven't got discovered yet
        quantizable_ops = [node.name for node in cask_partition_heads + quantizable_kgen_heads]

        # Look for other quantizable consumer than the current kgen head
        if head_name in quantizable_ops:
            quantizable_ops.remove(head_name)

        return any(consumer.name in quantizable_ops for consumer in tensor.outputs)

    quantizable_kgen_heads = []
    no_quantize_inputs = []  # list of tuple [(src_node_name, dst_node_name, input_name), ...]
    output_quantization_candidates = [
        "AveragePool",
        "BatchNormalization",
        "GlobalAveragePool",
        "MaxPool",
        "Mul",  # Example: VoVNet
    ]

    for partition in kgen_partitions:
        head_node = partition[0]
        # Check if partition head is of default quantizable type
        if head_node.op not in quantizable_op_types:
            continue

        # If the node has cost input, do not quantize
        if has_const_input(head_node):
            continue

        head_parents = get_parent_nodes(head_node)
        no_quantize_inputs_of_head = []
        has_quantizable_input = False

        # Check each of the parent (input producer for partition head)
        # or predecessor nodes and see if output quantization is needed for them
        # and decide which input of kgen head needs quantization
        for parent in head_parents:
            # If the head is consuming output of any quantizable op, then it is quantizable
            if (
                _is_following_cask_partition(parent, cask_partition_nodes)
                or parent.op in output_quantization_candidates
            ):
                # The mask add of MHA should not be quantized
                if _is_mha_epilogue_pattern(head_node, graph):
                    no_quantize_inputs_of_head.append(
                        (parent, partition[0], parent.outputs[0].name)
                    )
                else:
                    quantizable_kgen_heads.append(partition[0])
                    has_quantizable_input = True
            # If the input from the current parent has no other quantizable consumer, do not quantize that input
            elif not _has_other_quantizable_consumer(
                parent.outputs[0], quantizable_kgen_heads, head_node.name
            ):
                no_quantize_inputs_of_head.append((parent, partition[0], parent.outputs[0].name))

        # If at least one input of Add is quantizable, collect if there is any non-quantizable inputs
        if head_node.op == "Add" and has_quantizable_input:
            no_quantize_inputs.extend(no_quantize_inputs_of_head)

    return quantizable_kgen_heads, no_quantize_inputs


def classify_partition_nodes(
    partitions: list[list[Node]],
) -> tuple[list[Node], list[Node], list[tuple[Node, Node, str]]]:
    """We should partially quantize the partition nodes with inputs outside of the partition.

    Args:
        partitions: Partitions created by modelopt ptq algo.

    Returns:
        List of non-quantizable nodes.
        List of quantizable nodes.
        List of partially-quantizable inputs with non-quantizable input info as (src, dst, input_name)
    """
    non_quantizable_partition_nodes = []  # list of Node [node1, ...]
    quantizable_partition_nodes = []  # list of Node [node1, ...]
    no_quantize_inputs = []  # list of tuple [(src_node, dst_node, input_name), ...]

    for partition in partitions:
        partition_root_type = partition[0].op
        assert is_linear_op(partition_root_type)

        # Collect tensor names produced by partition nodes
        partition_node_outputs = []
        for node in partition:
            partition_node_outputs.extend([output.name for output in node.outputs])

        for node in partition:
            has_external_inputs = False
            internal_inputs = []  # Keeps (producer, consumer, tensor)
            for tensor in node.inputs:
                if is_const_input(tensor):
                    continue

                # If a KGEN op has external non-constant input, it is considered partially quantizable
                if tensor.name not in partition_node_outputs:
                    # partition heads will be fully quantizable and added
                    has_external_inputs = True
                else:
                    producer_node = tensor.inputs[0]
                    # format: source, target, input
                    # Note. it might happen that this node was not quantized
                    # We just ignore it from no_quantize_inputs list in post-processing
                    internal_inputs.append((producer_node, node, tensor.name))

            if not has_external_inputs:
                non_quantizable_partition_nodes.append(node)
            elif has_external_inputs and internal_inputs:
                no_quantize_inputs.extend(internal_inputs)
            else:
                # partition head is quantizable
                quantizable_partition_nodes.append(node)

    return non_quantizable_partition_nodes, quantizable_partition_nodes, no_quantize_inputs


def classify_partially_quantized_weighted_ops(
    graph: Graph, nodes_to_exclude: list[str]
) -> list[tuple[Node, Node, str]]:
    """Ensures that the input of non-quantizable weighted nodes do not get quantized."""
    no_quantize_inputs = []
    linear_nodes_to_exclude = [
        node for node in graph.nodes if node.name in nodes_to_exclude and is_linear_op(node.op)
    ]
    for node in linear_nodes_to_exclude:
        for tensor in node.inputs:
            if tensor.inputs:
                producer_node = tensor.inputs[0]
                no_quantize_inputs.append((producer_node, node, tensor.name))
    return no_quantize_inputs


def build_non_residual_input_map(
    graph: Graph,
) -> tuple[dict[str, str], list[tuple[Node, Node, str]]]:
    """Builds a map of non-residual Add input name to the Add node name from the given graph.

    This assumes that the Add layer only has 2 inputs.

    We will refer to a subgraph which has a Convolution node with a single output that is summed (element-wise)
    with another non-constant input-tensor as a "residual-add" subgraph, because it occurs in modern
    convnets that use residual connections.

    Args:
        graph: Onnx model graph.

    Returns:
        Dictionary of Add node names vs their non-residual input name.
        List of partially-quantizable inputs with non-quantizable input info as (src, dst, input_name)
    """
    non_residual_inputs = {}
    no_quantize_inputs = []
    for node in graph.nodes:
        if node.op == "Add":
            # Add nodes with constant or graph input does not have non-residual input
            # Here, A = node.inputs[0], B = node.inputs[1] and A.inputs means producer nodes of A
            # TODO: make this check a util?
            if (
                has_const_input(node)
                or len(node.inputs[0].inputs) == 0
                or len(node.inputs[1].inputs) == 0
            ):
                non_residual_inputs[node.name] = None
                continue

            input1_producer = node.i(0, 0)
            input2_producer = node.i(1, 0)

            backbone1 = get_fusible_backbone(input1_producer, graph)
            backbone2 = get_fusible_backbone(input2_producer, graph)

            # Input in the longest path to LCA is the non-residual input
            lca, d1, d2 = find_lowest_common_ancestor(input1_producer, input2_producer)

            # Generally if both the inputs have a backbone then both backbones are of the same type
            if backbone1 and backbone2:
                if backbone1 == backbone2:
                    non_residual_inputs[node.name] = None
                    continue

                if d1 > d2:
                    non_residual_inputs[node.name] = node.inputs[0].name
                    no_quantize_inputs.append((input1_producer, node, node.inputs[0].name))
                else:
                    non_residual_inputs[node.name] = node.inputs[1].name
                    no_quantize_inputs.append((input2_producer, node, node.inputs[1].name))
            elif backbone1:
                # ConvNext pattern
                # Conv ---------------------- add
                #       \---- non backbone---/
                # This case LCA being backbone itself is not residual Add case.
                if lca and lca == backbone1.name:
                    # Not a residual Add node
                    non_residual_inputs[node.name] = None
                else:
                    non_residual_inputs[node.name] = node.inputs[0].name
                    no_quantize_inputs.append((input1_producer, node, node.inputs[0].name))
            elif backbone2:
                if lca and lca == backbone2.name:
                    # Not a residual Add node
                    non_residual_inputs[node.name] = None
                else:
                    non_residual_inputs[node.name] = node.inputs[1].name
                    no_quantize_inputs.append((input2_producer, node, node.inputs[1].name))
            else:
                # Not a residual Add node
                non_residual_inputs[node.name] = None

    return non_residual_inputs, no_quantize_inputs


def remove_partial_input_qdq(
    graph: Graph,
    no_quantize_inputs: list[tuple[Node, Node, str]],
) -> None:
    """Modifies the onnx model by removing QDQ nodes from the marked inputs, ex. non-residual inputs etc.

    Args:
        graph: Onnx model graph.
        no_quantize_inputs: List non-quantizable input info as (src, dst, input_name)
    """
    logger.info("Deleting QDQ nodes from marked inputs to make certain operations fusible")
    graph_nodes = {node.name: node for node in graph.nodes}
    for source, target, non_qdq_input_name in no_quantize_inputs:
        # Note. no_quantize_inputs objects are from non-quantized input graph
        # we are deleting some QDQ from the new quantized output graph
        source_node = graph_nodes[source.name]
        try:
            dq_node = source_node.o().o()
        except Exception:
            # Reached end of the graph
            continue
        if dq_node.op == "DequantizeLinear":
            dq_output = dq_node.outputs[0]  # source_node->Q->DQ->target_node

            # Look up the specific target node in the quantized graph.
            # With DedicatedQDQPair=False, a shared Q/DQ pair may feed multiple consumers
            # (e.g. Conv activation AND Add residual). Always patch the intended target
            # rather than the first consumer of the DQ output to avoid removing Q/DQ from
            # the wrong branch.
            target_node_in_graph = graph_nodes.get(target.name)
            if target_node_in_graph is None:
                continue

            # Find the input index in the target that is connected to the DQ output
            target_input_idx_arr = [
                idx
                for idx, inp in enumerate(target_node_in_graph.inputs)
                if inp.name == dq_output.name
            ]
            # If no input index is found (dq_output is not actually connected to target node), skip rewiring to
            # prevent silent corruption of the graph.
            if not target_input_idx_arr:
                logger.warning(
                    "Expected DequantizeLinear output '%s' to be an input of node '%s', "
                    "but no matching input was found. Skipping Q/DQ bypass for this edge.",
                    dq_output.name,
                    target_node_in_graph.name,
                )
                continue
            target_input_idx = target_input_idx_arr[0]

            # Connect the target's input directly to source_node's output (bypass Q/DQ)
            target_node_in_graph.inputs[target_input_idx] = source_node.outputs[0]

    # Check for quantized residual Adds where the parallel branch is not being quantized
    for source, target, non_qdq_input_name in no_quantize_inputs:
        if target.op != "Add":
            continue

        target_node = graph_nodes[target.name]
        for inp_idx, inp in enumerate(target_node.inputs):
            if inp.inputs[0].op == "DequantizeLinear":
                try:
                    parent_node = inp.inputs[0].i().i()
                except Exception:
                    # Reached beginning of the graph
                    continue
                quant_out_count = [
                    out_idx
                    for out_idx, out in enumerate(parent_node.outputs)
                    if out.outputs[0].op == "QuantizeLinear"
                ]
                non_quant_out_count = [
                    out
                    for out in parent_node.outputs
                    for _, _, non_qdq_inp_name in no_quantize_inputs
                    if out.name == non_qdq_inp_name
                ]
                # Bypass QDQ nodes if only one branch is quantized and the parallel branch should not be quantized
                if len(quant_out_count) == 1 and non_quant_out_count:
                    target_node.inputs[inp_idx] = parent_node.outputs[quant_out_count[0]]

    graph.cleanup()
    graph.toposort()


def _find_int4_quantizable_weights(
    graph: onnx.GraphProto,
    nodes_to_exclude: list[str],
) -> list[tuple[onnx.ValueInfoProto, onnx.ValueInfoProto, bool, int, str]]:
    """Finds the int4 quantizable weights from the graph.

    Returns:
        list of tuples: (act_tensor, weight_tensor, do_transpose, gemm_io_type, node_name)
    """
    wa_pack = []
    gemm_nodes = [
        node
        for node in graph.node
        if node.op_type in ["Gemm", "MatMul"] and node.name not in nodes_to_exclude
    ]
    initializer_idxs = {initializer.name: idx for idx, initializer in enumerate(graph.initializer)}
    for gemm in gemm_nodes:
        if gemm.input[0] in initializer_idxs:
            # Ex. two const input to MatMul_115 in fastvit0.onnx
            # Note. RTN algorithm will quantize these weights though
            continue

        if gemm.input[1] not in initializer_idxs:
            continue

        weight_tensor = graph.initializer[initializer_idxs[gemm.input[1]]]
        if len(weight_tensor.dims) == 1:  # 1D blocked quantization not supported
            continue

        gemm_io_type = cast("int", weight_tensor.data_type)

        act_tensor = onnx.helper.ValueInfoProto()
        act_tensor.name = gemm.input[0]

        # TODO: support transA by transposing activation tensors in _clip_search
        do_transpose = gemm.op_type == "Gemm" and any(
            attr.name == "transB" and attr.i > 0 for attr in gemm.attribute
        )

        # Include node name for proper matching with layers_8bit_set
        wa_pack.append((act_tensor, weight_tensor, do_transpose, gemm_io_type, gemm.name))

    return wa_pack


def should_quantize_to_8bit(layer_name: str, layers_8bit: list[str]):
    """Check if layer should be quantized to 8 bits.

    The layers_8bit list contains ONNX node names like '/model/layers.13/attn/qkv_proj/MatMul'.
    The layer_name argument is an ONNX initializer name like 'model.layers.13.attn.qkv_proj.MatMul.weight'.

    To match these, we:
      - Remove the leading slash from the node name.
      - Replace all '/' with '.' to match the naming convention of the initializer.

    This allows us to correctly identify which weights should be quantized to 8 bits.
    """
    if not layers_8bit:
        return False

    # Normalize both to dot-delimited tokens and require exact token sequence match.
    def tokens(s: str) -> list[str]:
        return s.lstrip("/").replace("/", ".").split(".")

    hay = tokens(layer_name)
    for pat in layers_8bit:
        needle = tokens(pat)
        n, m = len(hay), len(needle)
        for i in range(n - m + 1):
            if hay[i : i + m] == needle:
                return True
    return False


def validate_8bit_layers(layers_str: str) -> bool:
    """Validate the format of layers_8bit string."""
    if not layers_str:
        return True
    # Allow comma-separated list of path-like tokens
    pattern = r"^\s*[/a-zA-Z0-9_.\-]+(\s*,\s*[/a-zA-Z0-9_.\-]+)*\s*$"
    return bool(re.match(pattern, layers_str))


def get_layer_precision_mapping(
    onnx_model: onnx.ModelProto,
    precision_pattern_8bit: str | None = None,
    nodes_to_exclude: list[str] | None = [r"/lm_head"],
    block_size: int = 128,
    quantize_axis: int = 0,
):
    """Generate a mapping of layer names to their quantization precision (4 bits or 8 bits) for an ONNX model.

    Args:
        onnx_model (onnx.ModelProto): The ONNX model to analyze.
        precision_pattern_8bit (str, optional): Comma-separated string of layer patterns to quantize to 8 bits.
            If None, a default set of patterns is used to select layers for 8 bits quantization.
        nodes_to_exclude (list[str], optional): List of node name patterns to exclude from quantization.
            Defaults to [r"/lm_head"].

    Returns:
        dict: A mapping from layer names to their quantization precision (e.g., {"layer_name": "8"}).
    """
    graph = onnx_model.graph

    nodes_to_exclude = expand_node_names_from_patterns(graph, nodes_to_exclude)
    # Collect quantizable weight tensors
    wa_pack = _find_int4_quantizable_weights(graph, nodes_to_exclude)

    if precision_pattern_8bit:
        if not validate_8bit_layers(precision_pattern_8bit):
            raise ValueError("Invalid format for --layers_8bit. Use comma-separated layers.")
        layers_list_8bit = [x.strip() for x in precision_pattern_8bit.split(",") if x.strip()]

    else:
        matmul_nodes = [
            node
            for node in onnx_model.graph.node
            if node.op_type in ["Gemm", "MatMul"] and "lm_head" not in node.name
        ]

        # Only include nodes matching the specified patterns for all layers present in the model
        # For example, for all i where a node exists with name:
        #   /model/layers.{i}/attn/qkv_proj/MatMul
        #   /model/layers.{i}/attn/v_proj/MatMul
        #   /model/layers.{i}/mlp/down_proj/MatMul
        pattern_regexes = [
            re.compile(r"^/model/layers\.(\d+)/attn/qkv_proj/MatMul$"),
            re.compile(r"^/model/layers\.(\d+)/attn/v_proj/MatMul$"),
            re.compile(r"^/model/layers\.(\d+)/self_attn/qkv_proj/MatMul$"),
            re.compile(r"^/model/layers\.(\d+)/self_attn/v_proj/MatMul$"),
            re.compile(r"^/model/layers\.(\d+)/mlp/down_proj/MatMul$"),
        ]

        # Filter matmul_nodes to only those matching the patterns
        filtered_matmul_nodes = []
        for node in matmul_nodes:
            for pat in pattern_regexes:
                if pat.match(node.name):
                    filtered_matmul_nodes.append(node)
                    break

        # Build a mapping from group key to list of node names (ordered by layer index if possible)
        def extract_group_key(node_name):
            # Extract the two components before 'MatMul' in the name, e.g. ...foo.bar.MatMul
            parts = node_name.split("/")
            if len(parts) >= 3:
                return ".".join(parts[-3:-1])
            return node_name

        group_to_nodes = {}
        for node in filtered_matmul_nodes:
            group_key = extract_group_key(node.name)
            group_to_nodes.setdefault(group_key, []).append(node.name)

        layers_8bit_set = set()
        for names in group_to_nodes.values():
            n = len(names)
            if n == 0:
                continue

            # Try to sort by layer index if present
            def layer_idx(name):
                m = re.search(r"layers\.(\d+)\.", name)
                return int(m.group(1)) if m else 0

            names_sorted = sorted(names, key=layer_idx)
            first_eighth = int(n // 8)
            last_eighth = int(n // 8)
            # First 1/8
            layers_8bit_set.update(names_sorted[:first_eighth])
            # Last 1/8
            if last_eighth > 0:
                layers_8bit_set.update(names_sorted[-last_eighth:])
            # Every third in the rest (excluding first and last eighth)
            rest_start = first_eighth
            rest_end = n - last_eighth
            for i in range(rest_start, rest_end):
                if (i - rest_start) % 3 == 0:
                    layers_8bit_set.add(names_sorted[i])
        layers_list_8bit = list(layers_8bit_set)
    # NEW: Create layer info mapping with precision, block_size, and axis
    layer_info = {}
    for i, (act_tensor, weight_tensor, do_transpose, gemm_io_type, node_name) in enumerate(wa_pack):
        weight_name = weight_tensor.name
        # Use node_name for matching against layers_8bit patterns
        if should_quantize_to_8bit(node_name, layers_list_8bit):
            layer_info[weight_name] = {
                "precision": 8,
                "block_size": -1,  # Per-channel for 8-bit
                "axis": 0,
            }
        else:
            layer_info[weight_name] = {
                "precision": 4,
                "block_size": block_size,  # Default block size for 4-bit
                "axis": quantize_axis,
            }

    return layer_info


def get_layer_info(
    onnx_model: onnx.ModelProto,
    nodes_to_exclude: list[str] | None = [r"/lm_head"],
    block_size: int = 128,
    quantize_axis: int = 0,
    **kwargs: Any,
):
    """Generate a mapping of weight tensor names to their quantization configuration.

    This function determines the quantization configuration (precision, block_size, axis) for each
    weight tensor in the ONNX model, based on the provided configuration. If mixed quantization
    is enabled, it uses the layer precision mapping; otherwise, it returns None.

    Args:
        onnx_model (onnx.ModelProto): The ONNX model to analyze.
        nodes_to_exclude (list[str] | None): List of node name patterns to exclude from quantization.
        **kwargs: Additional keyword arguments, such as:
            - enable_mixed_quant (bool): Whether to enable mixed quantization.
            - layers_8bit (str): Comma-separated list of layer patterns to quantize to 8 bit.
            - block_size (int): Default block size for quantization.
            - quantize_axis (int): Default quantization axis.
            - gather_block_size (int): Default block size for gather quantization.
            - gather_quantize_axis (int): Default quantization axis for gather.

    Returns:
        dict[str, dict[str, Any]] | None: A mapping from weight tensor names to their quantization
        configuration (with keys: precision, block_size, axis), or None if mixed quantization is not enabled.
    """
    layer_info = None
    enable_mixed_quant = kwargs.get("enable_mixed_quant", False)
    layers_8bit = kwargs.get("layers_8bit")
    gather_block_size = kwargs.get("gather_block_size", DEFAULT_GATHER_BLOCK_SIZE)
    gather_quantize_axis = kwargs.get("gather_quantize_axis", DEFAULT_GATHER_QUANTIZE_AXIS)
    if enable_mixed_quant or layers_8bit:
        layer_info = get_layer_precision_mapping(
            onnx_model,
            layers_8bit,
            nodes_to_exclude,
            block_size,
            quantize_axis,
        )
    else:
        layer_info = None

    if gather_quantize_axis is not None:
        if layer_info is None:
            layer_info = {}
        for node in onnx_model.graph.node:
            if node.op_type == "Gather":
                layer_info[node.input[0]] = {
                    "precision": 4,
                    "block_size": gather_block_size,
                    "axis": gather_quantize_axis,
                }
    return layer_info


def print_stat(graph: Graph) -> None:
    """Collect and print stats of the quantized model."""
    count = 0
    quantized_type_counts = {}
    quantized_nodes = []
    output_names = [output_node.name for output_node in graph.outputs]
    for node in graph.nodes:
        for tensor in node.inputs:
            if len(tensor.inputs) == 0:
                continue

            producer_node = tensor.inputs[0]
            if producer_node.op == "DequantizeLinear":
                quantized_type_counts[node.op] = quantized_type_counts.get(node.op, 0) + 1
                quantized_nodes.append(node.name)
                count += 1
                break
            else:
                # Sometimes "_DequantizeLinear_Output" is not suffix of the "DequantizeLinear" typed node,
                # if that node is also in final model output. Ex. CLIP-ViT-L-14-opset16.onnx
                assert tensor.name in output_names or producer_node.op != "DequantizeLinear"

    logger.info(f"Total number of nodes: {len(graph.nodes)}")
    logger.info(f"Total number of quantized nodes: {count}")
    logger.debug(f"Quantized type counts: {quantized_type_counts}")
    logger.debug(f"Quantized nodes: {quantized_nodes}")


def get_resize_scales(onnx_model):
    r"""Record Resize op's old scale value before converting to fp16.

    Because low precision scale will lead to wrong shape. For example, if 7 is
    resized to 6, fp32 scale should be 6/7 = 0.85714. After converting to fp16,
    it becomes 0.85693 but 7 * 0.85693 = 5.9985 < 6.
    """
    resize_scale_inits = {}
    for node in onnx_model.graph.node:
        if node.op_type == "Resize" and len(node.input) > 2 and node.input[2] is not None:
            for init in onnx_model.graph.initializer:
                if init.name == node.input[2] and init.data_type == onnx.TensorProto.FLOAT:
                    resize_scale_inits[node.name] = (init.data_type, init.raw_data)
                    break
    return resize_scale_inits


def get_concat_eliminated_tensors(
    onnx_model: onnx.ModelProto,
    nodes_to_quantize: list[str],
) -> dict[str, set[str]]:
    """Find the input tensors and output tensor of concat that will be quantized.

    We can do some perf optimization for TRT.

    For example, like the below pattern:
    (t1) q1 -> dq1 \
    (t2) q2 -> dq2 -> concat -> q4 -> dq4 (t4)
    (t3) q3 -> dq3 /

    In TRT, q4 will be propagated forward concat. It will be like:
    (t1) q1 -> dq1 -> q4 \
    (t2) q2 -> dq2 -> q4 -> concat -> dq4 (t4)
    (t3) q3 -> dq3 -> q4 /

    If the scaling factor of dq1 and q4 are different, it will cause the dq-q compute latency. If
    they are the same, then the dq-q pairs can be eliminated in TRT, and no extra dq-q compute
    latency. However, it will sacrifice the accuracy.

    Thus, this function will collect which tensors should have the same scaling factors. For the
    above example, we want the scaling factor of dq1, dq2, dq3, q4 be the same. This function will
    return like
    {
    t1: {t1,t2,t3,t4},
    t2: {t1,t2,t3,t4},
    t3: {t1,t2,t3,t4},
    t4: {t1,t2,t3,t4},
    }
    This format is convenient for calibrator to assign the same scaling factor.

    Returns:
        {current tensor name: set of tensors that should share the same scaling factor}
    """
    logger.info("Finding concat eliminated tensors")
    input_name_to_nodes = get_tensor_consumer_nodes(onnx_model.graph)
    graph = gs.import_onnx(onnx_model)

    # We'll use a Union-Find data structure to track tensor groups
    parent = {}

    def find(x):
        if x not in parent:
            parent[x] = x
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        parent[find(x)] = find(y)

    # First, identify concat ops where output is quantized
    for node in graph.nodes:
        if node.op == "Concat":
            # Check if concat output is quantized
            concat_output_name = node.outputs[0].name
            concat_consumers = input_name_to_nodes[concat_output_name]
            concat_has_qdq = any(
                consumer.name in nodes_to_quantize for consumer in concat_consumers
            )

            if concat_has_qdq:
                # Find quantized inputs to concat
                quantized_inputs = []
                for input_tensor in node.inputs:
                    input_name = input_tensor.name
                    input_consumers = input_name_to_nodes[input_name]
                    if any(consumer.name in nodes_to_quantize for consumer in input_consumers):
                        quantized_inputs.append(input_name)

                # If we have quantized inputs, merge them with the output
                if quantized_inputs:
                    # Add the concat output
                    quantized_inputs.append(concat_output_name)

                    # Use union-find to merge all related tensors
                    for i in range(1, len(quantized_inputs)):
                        union(quantized_inputs[i], quantized_inputs[0])

    # Build the final result dictionary
    result = {}
    all_tensors = set(parent.keys())

    # Group by the root parent
    groups = {}
    for tensor in all_tensors:
        root = find(tensor)
        if root not in groups:
            groups[root] = set()
        groups[root].add(tensor)

    # Build the final mapping
    for tensor in all_tensors:
        root = find(tensor)
        result[tensor] = groups[root]
    return result
