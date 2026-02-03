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

"""Provides ONNX graph related utils for QDQ placement."""

import re
from collections import defaultdict
from functools import reduce
from typing import Any, cast

import numpy as np
import onnx
import onnx_graphsurgeon as gs
from onnx_graphsurgeon.ir.graph import Graph
from onnx_graphsurgeon.ir.node import Node
from onnx_graphsurgeon.ir.tensor import Constant, Tensor, Variable
from onnxruntime.quantization.calibrate import CalibrationDataReader

from modelopt.onnx.logging_config import logger
from modelopt.onnx.op_types import get_copy_ops, is_copy_op, is_linear_op
from modelopt.onnx.quantization.ort_utils import create_inference_session
from modelopt.onnx.utils import (
    find_lowest_common_ancestor,
    get_child_nodes,
    get_parent_nodes,
    infer_shapes,
    parse_shapes_spec,
    save_onnx,
)

DEFAULT_GATHER_BLOCK_SIZE = 32
DEFAULT_GATHER_QUANTIZE_AXIS = None


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


def get_input_shapes(onnx_path: str) -> dict[str, list[int]]:
    """Returns the input shapes of the given ONNX model."""
    onnx_model = onnx.load(onnx_path)
    input_shape_dict = {}
    for input in onnx_model.graph.input:
        input_shape_dict[input.name] = [x.dim_value for x in input.type.tensor_type.shape.dim]
    return input_shape_dict


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
            tensor_name = tensor.name if isinstance(tensor, gs.Tensor) else tensor
            tensor_consumer_map[tensor_name].append(node_idx)
    return tensor_consumer_map


def filter_quantizable_kgen_heads(
    cask_fusible_partitions: list[list[Node]],
    kgen_partitions: list[list[Node]],
    quantizable_op_types: list[str],
    graph: Graph,
) -> tuple[list[Node], list[tuple[Node, Node, str]]]:
    """Returns the list of kgen head names if it follows a CASK partition."""
    cask_partition_nodes = set()
    for partition in cask_fusible_partitions:
        cask_partition_nodes.update([node.name for node in partition])

    cask_partition_heads = [partition[0] for partition in cask_fusible_partitions]

    def _is_following_cask_partition(node: Node):
        # Checking if cask fusible partition can be reached backward
        # ignoring the copy ops
        if node.name in cask_partition_nodes:
            return True

        if not is_copy_op(node.op):
            return False

        parent_nodes = get_parent_nodes(node)
        if len(parent_nodes) == 0:
            return False

        return all(_is_following_cask_partition(parent) for parent in parent_nodes)

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
            if _is_following_cask_partition(parent) or parent.op in output_quantization_candidates:
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
        if node.op in ["Add"]:
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
            dq_node = dq_node.outputs[0]  # source_node->Q->DQ->target_node0

            # Find the input index in the target connecting with source_node
            target_input_idx_arr = [
                idx for idx, inp in enumerate(dq_node.outputs[0].inputs) if inp.name == dq_node.name
            ]
            target_input_idx = target_input_idx_arr[0] if target_input_idx_arr else 0

            # Connect the output of source_node with the output of DQ
            dq_node.outputs[0].inputs[target_input_idx] = source_node.outputs[0]

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


def _find_nodes_from_op_types_to_exclude(graph: Graph, op_types_to_exclude=None) -> list[str]:
    nodes_to_exclude = []
    if op_types_to_exclude:
        nodes_to_exclude = [node.name for node in graph.nodes if node.op in op_types_to_exclude]
    return nodes_to_exclude


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


def find_nodes_to_exclude(
    graph: Graph, nodes_to_exclude: list[str], op_types_to_exclude: list[str]
):
    """Find the node names from the ONNX graph which matches user's exclusion patterns."""
    nodes_to_exclude = nodes_to_exclude or []
    nodes_to_exclude = expand_node_names_from_patterns(graph, nodes_to_exclude)
    nodes_to_exclude.extend(_find_nodes_from_op_types_to_exclude(graph, op_types_to_exclude))

    # Remove duplicates from the exclusion list
    return [*set(nodes_to_exclude)]


def get_extended_model_outputs(
    onnx_path: str,
    extended_model: onnx.ModelProto,
    use_external_data_format: bool,
    intermediate_generated_files: list[str],
    calibration_data_reader: CalibrationDataReader,
    calibration_eps: list[str],
) -> dict[str, np.ndarray]:
    """Run one inference step on an onnx model which has some intermediate tensor marked as model outputs.

    The first calibration data is used for the dummy inference. This is useful when we want to know the shape of an
    intermediate tensor given the calibration data.

    Args:
        onnx_path:
            Path to the original onnx model, used for saving the extended model nearby if it is larger than 2GB.
        extended_model:
            The onnx model with some intermediate tensors marked as model outputs.
        use_external_data_format:
            If True, external data path will be used to store the weights of the intermediate model.
        intermediate_generated_files:
            List of intermediate generated files that will be deleted after quantization.
        calibration_data_reader:
            Calibration data reader for running inference.
        calibration_eps:
            Priority order for the execution providers (EP) to calibrate the model.
            Any subset of ['cuda:x', 'cpu', 'trt'], where 'x' is the device id.

    Returns: a map with each output name pointed to the corresponding output numpy ndarray.
    """
    # Get the first calibration input data.
    inputs = calibration_data_reader.get_first()

    # Initialize ORT session.
    if use_external_data_format:
        extended_onnx_path = onnx_path.replace(".onnx", "_extended.onnx")
        save_onnx(extended_model, extended_onnx_path, save_as_external_data=True)
        intermediate_generated_files.append(extended_onnx_path)
        session = create_inference_session(extended_onnx_path, calibration_eps)
    else:
        session = create_inference_session(extended_model.SerializeToString(), calibration_eps)

    # Run extended model's inference.
    extended_model_output_names = [output.name for output in session.get_outputs()]
    outputs = session.run(extended_model_output_names, inputs)
    output_map = dict(zip(extended_model_output_names, outputs))

    return output_map


def find_nodes_from_matmul_to_exclude(
    onnx_path: str,
    use_external_data_format: bool = False,
    intermediate_generated_files: list[str] | None = None,
    calibration_data_reader: CalibrationDataReader = None,
    calibration_eps: list[str] = ["cpu", "cuda:0", "trt"],
    calibration_shapes: str | dict | None = None,
) -> list[str]:
    """Find MatMul nodes that meets gemv condition to exclude.

    Either of m or n in matmul is 1, this matmul cannot utilize
    TensorCores. The perf of adding Q/DQ layers is not good in
    TRT. Thus, in this case, do not add Q/DQ layers to this matmul.

    Args:
        onnx_path: Path to the onnx model.
        use_external_data_format: If True, external data path will be used to store the
            weights of the intermediate model.
        intermediate_generated_files: List of intermediate generated files that will be deleted after quantization.
        calibration_data_reader: Calibration data reader for running inference.
        calibration_shapes: Model input shapes for inference. If provided, symbolic shape inference will be used
            instead of calibration_data_reader.
        calibration_eps: Priority order for the execution providers (EP) to calibrate the model.
            Any subset of ['cuda:x', 'cpu', 'trt'], where 'x' is the device id.

    Returns:
        List of Nodes to exclude from quantization.
    """
    model = onnx.load(onnx_path, load_external_data=True)
    graph = gs.import_onnx(model)

    matmul_nodes = [node for node in graph.nodes if node.op in {"MatMul", "Gemm"}]
    if not matmul_nodes:
        logger.debug("No MatMul nodes found in the model")
        return []

    logger.debug(f"Found {len(matmul_nodes)} MatMul nodes to analyze")

    if calibration_shapes:
        nodes_to_exclude = _exclude_matmuls_by_shape_inference(
            model, matmul_nodes, calibration_shapes
        )
    else:
        if calibration_data_reader is None:
            raise ValueError(
                "Either calibration_shapes or calibration_data_reader must be provided"
            )
        nodes_to_exclude = _exclude_matmuls_by_inference(
            onnx_path,
            model,
            matmul_nodes,
            use_external_data_format,
            intermediate_generated_files or [],
            calibration_data_reader,
            calibration_eps,
        )

    logger.debug(f"Matmul nodes to exclude: {nodes_to_exclude}")
    return [*set(nodes_to_exclude)]


def find_nodes_from_convs_to_exclude(graph: Graph, quantize_mode: str = "int8"):
    """Find unsupported Conv nodes to exclude from quantization.

    - The input and output channels should be >= 16. The exception is for Conv layers in INT8 quantization mode,
      which supports it if the input or output channel % 8.
    - The filter size for FP8 conv kernels should be less than 32.

    Args:
        graph: Onnx model graph.
        quantize_mode: Quantize mode (int8 or fp8).

    Returns:
        List of Conv nodes.
    """
    unsupported_conv_nodes = []
    logger.info("Scanning for unsupported Conv nodes for quantization")
    for node in graph.nodes:
        if node.op == "Conv":
            weight = node.inputs[1]

            # If weight.shape is None, it means the weight is not a constant tensor.
            # Skip the convs with non-constant weights.
            if weight.shape is None:
                logger.debug(f"Skipped quantizing conv: {node.name} due to non-constant weight")
                unsupported_conv_nodes.append(node.name)
                continue

            assert 3 <= len(weight.shape) <= 5, (
                f"Invalid weight shape {weight.shape}. Only 1D, 2D, and 3D convolutions are supported"
            )
            output_channel = weight.shape[0]
            input_channel = weight.shape[1]
            group = node.attrs.get("group", 1)
            kernel_shape = node.attrs.get("kernel_shape", [1, 1])
            if output_channel < 16 or input_channel < 16:
                if quantize_mode == "int8":
                    # If standard Conv (group == 1), check for supported configs:
                    # (1) OC or IC % 8
                    # (2) Conv is the 1st layer of the graph or OC or IC >= 16
                    # (3) Kernel shape is the same in all dimensions (i.e, 3x3)
                    if (
                        group == 1
                        and (output_channel % 8 == 0 or input_channel % 8 == 0)
                        and (
                            any(inp in graph.inputs for inp in node.inputs)
                            or (output_channel >= 16 or input_channel >= 16)
                        )
                        and len(set(kernel_shape)) == 1
                    ):
                        continue

                    # If grouped Conv (group > 1), check for supported configs:
                    # (1) OC % 8
                    # (2) Kernel shape is >= 3 across all dimensions
                    if group > 1 and output_channel % 8 == 0 and all(k >= 3 for k in kernel_shape):
                        continue

                logger.debug(f"Found Conv with I/O channel size less than 16: {node.name}")
                unsupported_conv_nodes.append(node.name)
                continue

            filter_size = reduce(lambda x, y: x * y, weight.shape[2:])
            if quantize_mode == "fp8" and filter_size > 32:
                logger.debug(f"Found large filter conv for FP8: {node.name}")
                unsupported_conv_nodes.append(node.name)

    logger.info(f"Found {len(unsupported_conv_nodes)} unsupported Conv nodes for quantization")
    return unsupported_conv_nodes


def _exclude_matmuls_by_shape_inference(
    model: onnx.ModelProto, matmul_nodes: list, calibration_shapes: str | dict | None = None
) -> list[str]:
    """Use shape inference to find MatMuls with dimension 1."""
    # Prepare model for symbolic inference
    for graph_input in model.graph.input:
        for dim in graph_input.type.tensor_type.shape.dim:
            if dim.HasField("dim_param"):
                dim.Clear()
                dim.dim_value = 1

    # Apply calibration shapes if provided
    input_shapes = {}
    if calibration_shapes:
        input_shapes = (
            parse_shapes_spec(calibration_shapes)
            if isinstance(calibration_shapes, str)
            else calibration_shapes
        )
    for graph_input in model.graph.input:
        if graph_input.name in input_shapes:
            input_shape = input_shapes[graph_input.name]
            tensor_shape = graph_input.type.tensor_type.shape.dim
            if len(tensor_shape) != len(input_shape):
                raise ValueError(
                    f"{graph_input.name} expects shape of rank {len(tensor_shape)}, "
                    f"but calibration shape of rank {len(input_shape)} was passed."
                )
            for dim, new_dim_value in zip(tensor_shape, input_shape):
                dim.dim_value = new_dim_value

    model = infer_shapes(model)
    value_info_map = {vi.name: vi for vi in model.graph.value_info}
    value_info_map.update({vi.name: vi for vi in model.graph.output})

    nodes_to_exclude = []
    for matmul_node in matmul_nodes:
        output_name = matmul_node.outputs[0].name
        value_info = value_info_map.get(output_name)
        if not value_info:
            raise RuntimeError(f"Shape inference did not find shape for {output_name}.")

        dims = value_info.type.tensor_type.shape.dim
        if all(isinstance(inp, Variable) for inp in matmul_node.inputs):
            if len(dims) < 2:
                raise RuntimeError(f"Shape for {output_name} is incorrect.")

            if dims[-1].dim_value == 1 or dims[-2].dim_value == 1:
                nodes_to_exclude.append(matmul_node.name)
        elif len(dims) < 3 and any(out.dim_value == 1 for out in dims):
            nodes_to_exclude.append(matmul_node.name)

    return nodes_to_exclude


def _exclude_matmuls_by_inference(
    onnx_path: str,
    model: onnx.ModelProto,
    matmul_nodes: list,
    use_external_data_format: bool,
    intermediate_generated_files: list[str],
    calibration_data_reader: CalibrationDataReader,
    calibration_eps: list[str],
) -> list[str]:
    """Use actual inference to find MatMuls with dimension 1."""
    # Add matmul outputs to model outputs
    for matmul_node in matmul_nodes:
        model.graph.output.extend([onnx.ValueInfoProto(name=matmul_node.outputs[0].name)])

    output_map = get_extended_model_outputs(
        onnx_path,
        model,
        use_external_data_format,
        intermediate_generated_files,
        calibration_data_reader,
        calibration_eps,
    )

    nodes_to_exclude = []
    for matmul_node in matmul_nodes:
        matmul_output = output_map[matmul_node.outputs[0].name]
        if all(isinstance(inp, Variable) for inp in matmul_node.inputs):
            if (
                len(matmul_output.shape) < 2
                or matmul_output.shape[-1] == 1
                or matmul_output.shape[-2] == 1
            ):
                nodes_to_exclude.append(matmul_node.name)
        elif len(matmul_output.shape) < 3 and any(out == 1 for out in matmul_output.shape):
            nodes_to_exclude.append(matmul_node.name)

    return nodes_to_exclude


def find_nodes_from_mha_to_exclude(
    onnx_path: str,
    use_external_data_format: bool = False,
    nodes_to_exclude: list[str] | None = None,
    disable_mha_qdq: bool = False,
    quantize_mode: str = "int8",
    intermediate_generated_files: list[str] | None = None,
    calibration_data_reader: CalibrationDataReader = None,
    calibration_eps: list[str] = ["cpu", "cuda:0", "trt"],
) -> list[str]:
    """Find MatMul nodes in MHA pattern to exclude.

    If disable_mha_qdq is set, don't add Q/DQ layers to MatMuls in MHA pattern.
    else when quantize_mode == "fp8", if head_size > 256 or head_size <= 8 or
    mha doesn't meet fp8 fMHA v2 pattern, don't add Q/DQ layers to MatMuls in MHA pattern.
    else when quantize_mode == "int8", if seq_len > 512, don't add Q/DQ layers
    to MatMuls in MHA pattern.

    Args:
        onnx_path:
            Path to the onnx model.
        use_external_data_format:
            If True, external data path will be used to store the weights of the intermediate model.
        nodes_to_exclude:
            List of Nodes to exclude from quantization.
        disable_mha_qdq:
            If True, all MHA's BMM1 and BMM2 will be added to nodes_to_exclude.
            Else, each MHA will be checked whether to enable QDQ or not when is_fp8fp16 is True.
        quantize_mode:
            Quantization mode. One of 'int8' (default), 'int4' and 'fp8'.
        intermediate_generated_files:
            List of intermediate generated files that will be deleted after quantization.
        calibration_data_reader:
            Calibration data reader for running inference.
        calibration_eps:
            Priority list of execution providers (EP) for calibration.

    Returns:
        List of Nodes to exclude from quantization.
    """
    logger.info(f"Analyzing MHA nodes for {quantize_mode} quantization")
    model = onnx.load(onnx_path, load_external_data=True)
    graph = gs.import_onnx(model)

    mha_partitions = find_mha_partitions(graph)
    if len(mha_partitions) == 0:
        logger.info("No MHA partitions found in the model")
        return nodes_to_exclude  # type: ignore[return-value]

    matmul_nodes_to_exclude = []
    if disable_mha_qdq:
        logger.info("Disabling QDQ for all MHA nodes")
        for mha_partition in mha_partitions:
            matmul_nodes_to_exclude.append(mha_partition[0].name)
            matmul_nodes_to_exclude.append(mha_partition[2].name)
    elif quantize_mode in {"fp8", "int8"}:
        # Add each BMM1's second input as BS1 model's extended outputs.
        for mha_partition in mha_partitions:
            bmm1_node = mha_partition[0]
            model.graph.output.extend([onnx.ValueInfoProto(name=bmm1_node.inputs[1].name)])

        # To get head_size and seq_len of MHA of the model, we run extended model inference once to get the shape info.
        output_map = get_extended_model_outputs(
            onnx_path,
            model,
            use_external_data_format,
            intermediate_generated_files,  # type: ignore[arg-type]
            calibration_data_reader,
            calibration_eps,
        )

        # For each MHA block,
        # In quantize_mode == int8, if seq_len > 512, add bmm to nodes_to_exclude.
        # In quantize_mode == fp8, if head_size > 256 or head_size <= 8, add its bmm to nodes_to_exclude.
        for mha_partition in mha_partitions:
            bmm1_node = mha_partition[0]
            softmax_node = mha_partition[1]
            bmm1_input_name = bmm1_node.inputs[1].name
            bmm1_input = output_map[bmm1_input_name]
            seq_len = bmm1_input.shape[-1]
            head_size = bmm1_input.shape[-2]
            enable_mha_qdq = True
            if quantize_mode == "int8":
                if seq_len > 512:
                    enable_mha_qdq = False
                    logger.debug(
                        f"Disabling QDQ for MHA node {bmm1_node.name} due to seq_len {seq_len} > 512"
                    )
            elif quantize_mode == "fp8":
                if head_size > 256 or head_size <= 8:
                    enable_mha_qdq = False
                    logger.debug(
                        f"Disabling QDQ for MHA node {bmm1_node.name} due to head_size {head_size}"
                    )
                else:
                    fp8_fmha_v2_pattern = match_fp8_mha_pattern(graph, softmax_node, False)
                    if len(fp8_fmha_v2_pattern) == 0:
                        enable_mha_qdq = False
                        logger.debug(
                            f"Disabling QDQ for MHA node {bmm1_node.name} due to non-matching FP8 pattern"
                        )
            if not enable_mha_qdq:
                matmul_nodes_to_exclude.append(mha_partition[0].name)
                matmul_nodes_to_exclude.append(mha_partition[2].name)

    logger.debug(f"Matmul nodes From MHA to exclude: {matmul_nodes_to_exclude}")

    nodes_to_exclude.extend(matmul_nodes_to_exclude)  # type: ignore[union-attr]
    # Remove duplicates from the exclusion list
    return [*set(nodes_to_exclude)]  # type: ignore[arg-type]


def validate_op_types_spelling(onnx_path, op_types_to_quantize, op_types_to_exclude) -> None:
    """Validate spelling in op types."""

    def find_item_ignore_case(target, arr):
        target_lower = target.lower()
        for item in arr:
            if item.lower() == target_lower:
                return item
        return None

    model = onnx.load(onnx_path, load_external_data=True)
    op_types = {node.op_type for node in model.graph.node}
    for op_type in op_types:
        if op_types_to_quantize:
            op_to_quant = find_item_ignore_case(op_type, op_types_to_quantize)
            if op_type not in op_types_to_quantize and op_to_quant is not None:
                logger.warning(
                    f"Model contains '{op_type}' ops, but you're requesting '{op_to_quant}' "
                    f"to be quantized, which is not a match. Please ensure that the lower/uppercasing is correct."
                )
        if op_types_to_exclude:
            op_to_exclude = find_item_ignore_case(op_type, op_types_to_exclude)
            if op_type not in op_types_to_exclude and op_to_exclude is not None:
                logger.warning(
                    f"Model contains '{op_type}' ops, but you're requesting '{op_to_exclude}' "
                    f"to be excluded from quantization, which is not a match. "
                    f"Please ensure that the lower/uppercasing is correct."
                )


def cast_custom_ops(onnx_model: onnx.ModelProto, ops_to_cast: dict) -> onnx.ModelProto:
    """Adds cast_to_fp16 nodes to the inputs and cast_to_fp32 to the outputs of a layer in the requested indices."""
    logger.info("Casting custom ops in the requested inputs and outputs")
    name_dict = {}

    def _get_unique_name(old_name):
        if old_name not in name_dict:
            name_dict[old_name] = 0
            return old_name
        name_dict[old_name] = name_dict[old_name] + 1
        return old_name + "_" + str(name_dict[old_name])

    def _is_castable_tensor(tensor) -> bool:
        castable_types = ["float16", "float32", "double"]
        return tensor.dtype and tensor.dtype in castable_types

    def _add_cast_node_inp(tensor, precision="fp16", suffix=""):
        if precision == "fp16":
            onnx_precision = int(onnx.TensorProto.FLOAT16)
            np_precision = "float16"
        else:
            onnx_precision = int(onnx.TensorProto.FLOAT)
            np_precision = "float32"

        cast_out = Variable(
            name=_get_unique_name(tensor.name + f"_{precision}{suffix}"),
            dtype=np_precision,
            shape=tensor.shape,
        )
        cast_node = Node(
            op="Cast",
            name=_get_unique_name(tensor.name + f"_cast_to_{precision}{suffix}"),
            attrs={"to": onnx_precision},
            inputs=[tensor],
            outputs=[cast_out],
        )
        graph.nodes.append(cast_node)
        return cast_out

    def _add_cast_node_out(tensor, inp_precision="fp16", out_precision="fp32", suffix=""):
        cast_precision = (
            int(onnx.TensorProto.FLOAT16)
            if out_precision == "fp16"
            else int(onnx.TensorProto.FLOAT)
        )
        np_precision = "float16" if inp_precision == "fp16" else "float32"

        cast_inp = gs.Variable(
            name=_get_unique_name(tensor.name + f"_{inp_precision}{suffix}"),
            dtype=np_precision,
            shape=tensor.shape,
        )
        cast_node = gs.Node(
            op="Cast",
            name=_get_unique_name(tensor.name + f"_cast_to_{out_precision}{suffix}"),
            attrs={"to": cast_precision},
            inputs=[cast_inp],
            outputs=[tensor],
        )
        graph.nodes.append(cast_node)
        return cast_inp

    graph = gs.import_onnx(onnx_model)
    castable_nodes = [n for n in graph.nodes if n.op in ops_to_cast]

    for node in castable_nodes:
        inp_idxs = ops_to_cast[node.op]["inp"]
        out_idxs = ops_to_cast[node.op]["out"]

        # Cast relevant inputs to FP16
        for inp_idx, inp in enumerate(node.inputs):
            if inp_idx in inp_idxs and _is_castable_tensor(inp):
                cast_out = _add_cast_node_inp(inp)
                node.inputs[inp_idx] = cast_out

        # Cast relevant outputs from FP16 back to FP32
        for out_idx, out in enumerate(node.outputs):
            if out_idx in out_idxs and _is_castable_tensor(out):
                cast_inp = _add_cast_node_out(out)
                node.outputs[out_idx] = cast_inp

    graph.cleanup().toposort()

    onnx_model = gs.export_onnx(graph)
    # TODO: remove manual ir_version change once ORT supports ir_version 11
    onnx_model.ir_version = 10

    return onnx_model


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


def insert_matmul_casts(graph, matmul_node):
    """Insert three cast nodes for MatMul's two inputs and output."""
    matmul_input0 = matmul_node.inputs[0]
    matmul_input0_cast_output = gs.Variable(
        name=f"{matmul_input0.name}/Cast_output", dtype=np.float32
    )
    graph.layer(
        op="Cast",
        name=f"{matmul_input0.name}/Cast",
        inputs=[matmul_input0],
        outputs=[matmul_input0_cast_output],
        attrs={"to": np.float32},
    )
    matmul_node.inputs[0] = matmul_input0_cast_output

    matmul_input1 = matmul_node.inputs[1]
    matmul_input1_cast_output = gs.Variable(
        name=f"{matmul_input1.name}/Cast_output", dtype=np.float32
    )
    graph.layer(
        op="Cast",
        name=f"{matmul_input1.name}/Cast",
        inputs=[matmul_input1],
        outputs=[matmul_input1_cast_output],
        attrs={"to": np.float32},
    )
    matmul_node.inputs[1] = matmul_input1_cast_output

    matmul_output = matmul_node.outputs[0]
    matmul_output_cast_input = gs.Variable(
        name=f"{matmul_output.name}/Cast_output", dtype=np.float32
    )
    graph.layer(
        op="Cast",
        name=f"{matmul_output.name}/Cast",
        inputs=[matmul_output_cast_input],
        outputs=[matmul_output],
        attrs={"to": np.float16},
    )
    matmul_node.outputs[0] = matmul_output_cast_input


def insert_fp8_mha_casts(onnx_model):
    r"""Insert three cast ops.

    The first cast will be added before the input0 of MatMul to cast fp16 to fp32.
    The second cast will be added before the input1 of MatMul to cast fp16 to fp32.
    The third cast will be added after the output of MatMul to cast fp32 back to fp16.
    The insertion of Cast ops in the FP8 MHA part actually forbids the MHAs to run
    with FP16 accumulation because the compiler only has FP32 accumulation kernels for FP8 MHAs.
    """
    graph = gs.import_onnx(onnx_model)
    graph.cleanup().toposort()

    # Match FP8 MHA: Q -> DQ -> BMM1 -> (Mul/Div) -> (Add) -> Softmax -> (Cast) -> Q -> DQ -> BMM2 -> Q -> DQ
    for node in graph.nodes:
        if node.op == "Softmax":
            fp8_fmha_v2_pattern = match_fp8_mha_pattern(graph, node, True)
            # Insert cast nodes on BMM2's input and output tensors.
            if len(fp8_fmha_v2_pattern) == 3:
                insert_matmul_casts(graph, fp8_fmha_v2_pattern[0])
                insert_matmul_casts(graph, fp8_fmha_v2_pattern[2])

    graph.cleanup().toposort()

    return gs.export_onnx(graph)


def convert_fp16_io(graph):
    """Convert graph I/O to FP16."""
    convertible_dtypes = [
        onnx.TensorProto.FLOAT,
        onnx.TensorProto.DOUBLE,
        onnx.TensorProto.BFLOAT16,
    ]
    for input_tensor in graph.inputs:
        input_tensor.dtype = (
            onnx.TensorProto.FLOAT16
            if input_tensor.dtype in convertible_dtypes
            else input_tensor.dtype
        )
    for output_tensor in graph.outputs:
        output_tensor.dtype = (
            onnx.TensorProto.FLOAT16
            if output_tensor.dtype in convertible_dtypes
            else output_tensor.dtype
        )


def remove_output_initializers(graph: gs.Graph, graph_initializers: list):
    """Remove initializers that are also listed as graph outputs.

    Having initializers (constant tensors) that are also marked as outputs can lead to ONNX Runtime
    or conversion tool errors, particularly related to ambiguous 'dtype' or shape inference.
    This step ensures compatibility by detaching such initializers from the graph's outputs.
    """
    init_names = [init.name for init in graph_initializers]
    init_names_removed = []
    for output_tensor in graph.outputs:
        if output_tensor.name in init_names:
            graph.outputs.remove(output_tensor)
            init_names_removed.append(output_tensor.name)
    if init_names_removed:
        logger.info(f"Removed output initializers: {init_names_removed}")


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


def remove_redundant_cast_nodes(graph: onnx.GraphProto) -> None:
    """Remove redundant Cast nodes from the ONNX graph to optimize model performance.

    This function identifies and removes two types of redundant Cast nodes:

    1. Cast nodes where input and output types are identical
       - Before: t1 (dtype=fp16) -> cast (to=fp16) -> t2 -> Op
       - After:  t1 (dtype=fp16) -> Op

    2. Cast nodes that can be fused with initializers
       - Before: (initializer) t1 (dtype=fp32) -> cast (to=fp16) -> t2 -> Op
       - After:  (initializer) t1 (dtype=fp16) -> Op

    The function preserves Cast nodes that:
    - Have outputs that are graph outputs
    - Are necessary for type conversion
    - Have dynamic inputs (not initializers)

    Args:
        graph: ONNX graph to optimize. The graph will be modified in-place.

    Note:
        - This optimization is particularly useful for models with many Cast operations
        - The function modifies the graph in-place
        - All tensor consumers are updated to maintain graph connectivity
        - Initializer data types are converted when possible to eliminate Cast nodes
    """
    initializers = {init.name: init for init in graph.initializer}
    tensor_consumers = get_tensor_consumer_nodes(graph)
    value_info_map = {info.name: info for info in graph.value_info}
    cast_indices = []
    output_names = {output.name for output in graph.output}

    def _get_tensor_type(tensor_name: str) -> int | None:
        """Get the tensor type for a given tensor name."""
        if tensor_name in value_info_map:
            return value_info_map[tensor_name].type.tensor_type.elem_type
        if tensor_name in initializers:
            return initializers[tensor_name].data_type
        return None

    for node_idx, node in enumerate(graph.node):
        if node.op_type != "Cast":
            continue

        # Skip if output is a graph output
        if any(out_name in output_names for out_name in node.output):
            continue

        input_name = node.input[0]
        input_type = _get_tensor_type(input_name)
        if input_type is None:
            continue

        # Get target type from Cast node attributes
        attr = next((attr for attr in node.attribute if attr.name == "to"), None)
        if attr is None:
            continue

        # Pattern 1: Input and output types are the same
        if input_type == attr.i:
            cast_indices.append(node_idx)
        # Pattern 2: Convert and fuse Cast node for initializers
        elif input_name in initializers:
            cast_indices.append(node_idx)
            cast_input = onnx.numpy_helper.to_array(initializers[input_name])
            dtype = onnx.helper.tensor_dtype_to_np_dtype(attr.i)
            converted_tensor = onnx.numpy_helper.from_array(cast_input.astype(dtype), input_name)
            initializers[input_name].CopyFrom(converted_tensor)
        else:
            continue

        # Update consumer nodes
        for consumer in tensor_consumers.get(node.output[0], []):
            for i, input_tensor in enumerate(consumer.input):
                if input_tensor == node.output[0]:
                    consumer.input[i] = input_name
                    break

    # Remove Cast nodes in reverse order
    logger.info(f"Removing {len(cast_indices)} redundant Cast nodes")
    for node_idx in sorted(cast_indices, reverse=True):
        del graph.node[node_idx]
