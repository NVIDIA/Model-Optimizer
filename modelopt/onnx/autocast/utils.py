# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Utility functions for AutoCast.

This module provides common utility functions used across the AutoCast package.
It includes functions for graph traversal, tensor type inference, model validation,
and mapping setup between nodes, initializers, and value info. These utilities
support the core functionality of model precision conversion.
"""

import logging
from collections import defaultdict
from collections.abc import Callable

import onnx

import modelopt.onnx.utils as onnx_utils
from modelopt.onnx.autocast.logging_config import logger
from modelopt.onnx.utils import get_opset_version


def setup_mappings(model: onnx.ModelProto) -> tuple[dict, dict, dict]:
    """Setup and return mappings for model components.

    Args:
        model: ONNX model to create mappings for.

    Returns:
        Tuple containing:
        - value_info_map: Mapping of names to value infos.
        - initializer_map: Mapping of names to initializers.
        - node_to_init_map: Mapping of node names to their initializer inputs.
    """
    value_info_map = {}
    for container in (model.graph.value_info, model.graph.input, model.graph.output):
        value_info_map.update((vi.name, vi) for vi in container)

    initializer_map = {init.name: init for init in model.graph.initializer}

    node_to_init_map = {
        node.name: [
            initializer_map[input_name]
            for input_name in node.input
            if input_name in initializer_map
        ]
        for node in model.graph.node
    }

    return value_info_map, initializer_map, node_to_init_map


def get_unique_consumer_node(model: onnx.ModelProto, tensor_name: str) -> onnx.NodeProto:
    """Get a single consumer node and raise exception if there are multiple consumers.

    Args:
        model: The ONNX model to search.
        tensor_name: Name of the tensor to find consumer for.

    Returns:
        onnx.NodeProto: The single consumer node.

    Raises:
        Exception: If there is not exactly one consumer node.
    """
    consumers = onnx_utils.get_consumer_nodes(model, tensor_name)
    if len(consumers) != 1:
        raise Exception(f"Expected single consumer for {tensor_name}, found {len(consumers)}")
    return consumers[0]


def walk_subgraphs_recursive(
    graph: onnx.GraphProto,
    callback: Callable,
    parent_node: onnx.NodeProto = None,
    is_subgraph: bool = False,
) -> None:
    """Recursively walk through a graph and all its subgraphs, applying a callback.

    This utility function traverses an ONNX graph and all nested subgraphs by examining
    graph attributes in nodes. It works with standard control flow operators (Scan, If, Loop)
    as well as custom operators that define subgraphs using ONNX graph attributes.

    Args:
        graph: The graph to walk.
        callback: Function to call for each graph. Signature: callback(graph, parent_node, is_subgraph).
        parent_node: The parent node containing this subgraph (None for main graph).
        is_subgraph: Whether this is a subgraph (True) or the main graph (False).

    Note:
        Works with any node that has attributes of type AttributeProto.GRAPH or
        AttributeProto.GRAPHS, including custom operators.
    """
    # Apply callback to current graph
    callback(graph, parent_node, is_subgraph)

    # Recursively process subgraphs in control flow nodes
    for node in graph.node:
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.GRAPH:
                walk_subgraphs_recursive(attr.g, callback, parent_node=node, is_subgraph=True)
            elif attr.type == onnx.AttributeProto.GRAPHS:
                for subgraph in attr.graphs:
                    walk_subgraphs_recursive(subgraph, callback, parent_node=node, is_subgraph=True)


def clear_types_and_shapes_recursive(
    graph: onnx.GraphProto, clear_shapes: bool = True, is_subgraph: bool = False
) -> None:
    """Recursively clear type/shape information for a graph and all its subgraphs.

    Resets intermediate (``value_info``) and output tensor types to ``UNDEFINED`` -- and, when
    ``clear_shapes`` is True, replaces concrete dimensions with a symbolic ``"unk"`` placeholder --
    so that a subsequent :func:`modelopt.onnx.utils.infer_types` re-derives types (and shapes) from
    the operator graph instead of trusting stale annotations. For subgraphs, input types/shapes are
    cleared as well so that they are propagated from the parent graph.

    Note: this clears in place but does not change tensor *rank*. It cannot repair a stale rank (e.g.
    a leftover rank-0 scalar on a tensor that is really rank-2+); see
    :func:`modelopt.onnx.utils._reconcile_stale_output_shapes` for that.

    Args:
        graph: The ONNX graph to clear types and shapes for.
        clear_shapes: If True, also clear shape information. Pass False to keep shapes when only
            types will be re-inferred (mirrors standalone type inference).
        is_subgraph: Whether this is a subgraph (True) or the main graph (False).
    """

    def _clear_callback(g: onnx.GraphProto, parent: onnx.NodeProto, is_sub: bool) -> None:
        logger.debug(f"Clearing types/shapes in {'subgraph' if is_sub else 'main graph'}: {g.name}")

        # Clear type/shape information for inputs (only for subgraphs, not main graph inputs)
        if is_sub:
            for inp in g.input:
                if inp.type.HasField("tensor_type"):
                    inp.type.tensor_type.elem_type = onnx.TensorProto.UNDEFINED
                    if clear_shapes:
                        for idx, d in enumerate(inp.type.tensor_type.shape.dim):
                            if d.dim_value:
                                inp.type.tensor_type.shape.dim[idx].dim_param = "unk"

        if is_sub:
            # Identify which tensors are produced by nodes in this subgraph
            subgraph_outputs = set()
            for node in g.node:
                subgraph_outputs.update(node.output)

            # Clear value_info only for intermediates produced by nodes in this subgraph
            for vi in g.value_info:
                if vi.name in subgraph_outputs:
                    vi.type.tensor_type.elem_type = onnx.TensorProto.UNDEFINED
                    if clear_shapes:
                        for idx, d in enumerate(vi.type.tensor_type.shape.dim):
                            if d.dim_value:
                                vi.type.tensor_type.shape.dim[idx].dim_param = "unk"
        else:
            for vi in g.value_info:
                vi.type.tensor_type.elem_type = onnx.TensorProto.UNDEFINED
                for idx, d in enumerate(vi.type.tensor_type.shape.dim):
                    if d.dim_value:
                        vi.type.tensor_type.shape.dim[idx].dim_param = "unk"

        # Clear outputs for both main graph and subgraphs
        for out in g.output:
            out.type.tensor_type.elem_type = onnx.TensorProto.UNDEFINED
            if clear_shapes:
                for idx, d in enumerate(out.type.tensor_type.shape.dim):
                    if d.dim_value:
                        out.type.tensor_type.shape.dim[idx].dim_param = "unk"

    walk_subgraphs_recursive(graph, _clear_callback, is_subgraph=is_subgraph)


def get_op_types_not_supported_in_low_precision(
    model: onnx.ModelProto,
    min_opset: int,
    low_precision_type: str = "float16",
) -> list[str]:
    """Get a list of ops not supported in low precision for the opset_version = max(model.opset, min_opset).

    An op is considered to be supported if at least one of the inputs may be in low precision.
    Ops where only some of the inputs may be in low precision are considered supported by this function
    and may need special handling. See PrecisionConverter::_should_skip_low_precision_input_conversion.

    Args:
        model: ONNX model.
        min_opset: Minimum opset version.
        low_precision_type: Target precision to reduce to ('float16' or 'bfloat16').

    Returns:
        ops_without_support: List of ops not supported in low precision for the current opset version.
    """
    # Obtain the current model's opset version
    opset_version = max(get_opset_version(model), min_opset)

    # Get all ops precision support information
    precision = "tensor(float16)" if low_precision_type == "float16" else "tensor(bfloat16)"
    model_ops = {n.op_type for n in model.graph.node}
    schemas_dict = defaultdict(dict)
    for schema in onnx.defs.get_all_schemas_with_history():
        if schema.name not in model_ops:
            continue
        float16_supported = False
        for constr in schema.type_constraints:
            if precision in constr.allowed_type_strs:
                float16_supported = True
                break
        schemas_dict[schema.name].update({schema.since_version: float16_supported})

    # Check that all ops are supported in low precision for the current opset version.
    # Otherwise, exclude from conversion.
    ops_without_support = {}
    for op, schema in schemas_dict.items():
        supported_opsets = [k for k, v in schema.items() if v]
        if supported_opsets:
            min_supported_opset = min(supported_opsets)
            if min_supported_opset > opset_version:
                ops_without_support[op] = min_supported_opset
        else:
            ops_without_support[op] = None

    if ops_without_support:
        logging.warning(
            f"{len(ops_without_support)} ops are not supported in '{low_precision_type}' in opset {opset_version}, "
            f"skipping those from conversion. Upgrade the model's opset version as follows to run them in low "
            f" precision: {ops_without_support}."
        )

    return list(ops_without_support.keys())
