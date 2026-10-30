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

"""The three separately ordered five-dimensional reshape pattern passes."""

import numpy as np
from onnx import TensorProto, helper

from ..utils.graph_utils import GraphCache, _attribute, add_initializer, batch_replace_nodes
from .dla_5d_reshape_to_4d import (
    _build_reduce_replacement_or_raise,
    _build_transpose_replacement_or_raise,
    _duplicate_clip_qdq_chain,
    _is_5d_reshape_candidate,
    _known_non_scalar_control,
)
from .dla_graph_cleanup import _subgraph_captures
from .dla_reorder_nodes import priority_topological_order

__all__ = []


def _follower(cache, output, op_type):
    """Match an unshared standard-domain data follower without replacing public tensors."""
    consumers = cache.get_consumers(output)
    if output in cache.graph_output_names or len(consumers) != 1:
        return None
    return next(
        (
            node
            for node, port in consumers
            if port == 0
            and node.op_type == op_type
            and node.domain in {"", "ai.onnx"}
            and not node.overload
        ),
        None,
    )


def _replace(graph, replacements):
    """Apply planned replacements without copying untouched payloads and order late controls."""
    if not replacements:
        return
    removed_outputs = {
        output for node in graph.node if id(node) in replacements for output in node.output
    }
    added_outputs = {
        output for nodes in replacements.values() for node in nodes for output in node.output
    }
    removed_captures = (removed_outputs - added_outputs) & _subgraph_captures(graph)
    if removed_captures:
        raise ValueError(f"Cannot rewrite captured tensors: {sorted(removed_captures)}")
    positions = {output: index for index, node in enumerate(graph.node) for output in node.output}
    needs_reorder = any(
        positions.get(name, -1) > index and name not in removed_outputs
        for index, node in enumerate(graph.node)
        for added in replacements.get(id(node), [])
        for name in added.input
    )
    batch_replace_nodes(
        graph,
        {
            index: ([node], replacements[id(node)])
            for index, node in enumerate(graph.node)
            if id(node) in replacements
        },
    )
    if needs_reorder:
        order, _, _ = priority_topological_order(graph)
        nodes = list(graph.node)
        ranks = {id(nodes[index]): rank for rank, index in enumerate(order)}
        graph.node.sort(key=lambda node: ranks[id(node)])


def _apply_reshape_reduce_sum(model):
    """Lower the separately ordered unquantized reshape/reduction stage."""
    return _reshape_reduce_sum(model, quantized=False)


def _apply_reshape_qdq_reduce_sum(model):
    """Lower the separately ordered per-tensor QDQ reshape/reduction stage."""
    return _reshape_reduce_sum(model, quantized=True)


def _reshape_reduce_sum(model, *, quantized):
    """Plan disjoint reductions while preserving controls and restoring initializers on refusal."""
    graph = model.graph
    cache = GraphCache(graph)
    replacements = {}
    count = 0
    initializer_count = len(graph.initializer)
    try:
        for reshape in graph.node:
            if not _is_5d_reshape_candidate(cache, reshape):
                continue
            if id(reshape) in replacements:
                raise ValueError(f"{reshape.name!r}: overlapping reshape patterns")
            shape = cache.get_init_array(reshape.input[1])
            if shape is None or shape.size != 5:
                continue
            middle = []
            output = reshape.output[0]
            if quantized:
                quantize = _follower(cache, output, "QuantizeLinear")
                if quantize is None:
                    continue
                dequantize = _follower(cache, quantize.output[0], "DequantizeLinear")
                if dequantize is None:
                    continue
                middle = [quantize, dequantize]
                output = dequantize.output[0]
            reduce = _follower(cache, output, "ReduceSum")
            if reduce is None:
                continue
            if any(
                name in cache.graph_input_names for name in [reshape.input[1], *reduce.input[1:]]
            ):
                raise ValueError(
                    f"{reshape.name!r}: shape and axes must be fixed graph initializers"
                )
            if any(
                any(attr.name in {"axis", "block_size"} for attr in node.attribute)
                or any(_known_non_scalar_control(cache, name) for name in node.input[1:])
                for node in middle
            ):
                raise ValueError(f"{reshape.name!r}: only per-tensor QDQ middleware is supported")
            # This pattern splits the final width of eight into two fours.
            input_shape = cache.get_shape(reshape.input[0])
            axes_values = cache.get_init_array(reduce.input[1]) if len(reduce.input) > 1 else None
            keepdims = _attribute(reduce, "keepdims")
            fast_path = (
                input_shape is not None
                and len(input_shape) == 4
                and input_shape[-1] == 8
                and shape.tolist() == [*input_shape[:-1], 2, 4]
                and axes_values is not None
                and axes_values.reshape(-1).tolist() == [-1]
                and keepdims is not None
                and keepdims.i == 0
            )
            if not fast_path:
                removed, added = _build_reduce_replacement_or_raise(
                    model, reshape, middle, reduce, set(cache.init_names), count, cache
                )
                replacements.update({id(node): [] for node in removed})
                replacements[id(reshape)] = added
                count += 1
                continue
            prefix = f"reshape_{'qdq_' if quantized else ''}reduce_{count}"
            axes = add_initializer(
                graph,
                cache.unique_name(prefix + "_slice_axes"),
                np.asarray([3], dtype=np.int64),
                dtype=TensorProto.INT64,
                with_value_info=False,
            )
            nodes, outputs = [], []
            for branch in range(2):
                start = add_initializer(
                    graph,
                    cache.unique_name(f"{prefix}_{branch}_starts"),
                    np.asarray([branch * 4], dtype=np.int64),
                    dtype=TensorProto.INT64,
                    with_value_info=False,
                )
                end = add_initializer(
                    graph,
                    cache.unique_name(f"{prefix}_{branch}_ends"),
                    np.asarray([(branch + 1) * 4], dtype=np.int64),
                    dtype=TensorProto.INT64,
                    with_value_info=False,
                )
                current = cache.unique_name(f"{prefix}_{branch}_slice")
                nodes.append(
                    helper.make_node(
                        "Slice", [reshape.input[0], start, end, axes], [current], name=current
                    )
                )
                duplicated, current = _duplicate_clip_qdq_chain(
                    middle, current, f"{prefix}_{branch}", cache
                )
                nodes.extend(duplicated)
                output = cache.unique_name(f"{prefix}_{branch}_reduce")
                nodes.append(
                    helper.make_node(
                        "ReduceSum", [current, *reduce.input[1:]], [output], name=output
                    )
                )
                outputs.append(output)
            nodes.append(
                helper.make_node(
                    "Concat",
                    outputs,
                    list(reduce.output),
                    name=cache.unique_name(prefix + "_axis_updated"),
                    axis=3,
                )
            )
            for original in [*middle, reduce]:
                replacements[id(original)] = []
            replacements[id(reshape)] = nodes
            count += 1
        _replace(graph, replacements)
    except Exception:
        del graph.initializer[initializer_count:]
        raise
    return model


def _apply_reshape_transpose_reshape(model):
    """Reuse the validated transpose builder for the separately ordered direct pattern."""
    graph = model.graph
    cache = GraphCache(graph)
    replacements = {}
    inferred_value_info = []
    count = 0
    initializer_count = len(graph.initializer)
    try:
        for first in graph.node:
            if not _is_5d_reshape_candidate(cache, first) or "_transformed" in first.name:
                continue
            transpose = _follower(cache, first.output[0], "Transpose")
            if transpose is None:
                continue
            last = _follower(cache, transpose.output[0], "Reshape")
            if last is None:
                continue
            if id(first) in replacements:
                raise ValueError(f"{first.name!r}: overlapping reshape patterns")
            removed, added = _build_transpose_replacement_or_raise(
                model,
                first,
                [],
                transpose,
                set(cache.init_names),
                count,
                cache,
                inferred_value_info=inferred_value_info,
            )
            replacements.update({id(node): [] for node in removed})
            replacements[id(first)] = added
            count += 1
        _replace(graph, replacements)
        # Ordered reduction stages read new Concat shapes before the next global inference.
        graph.value_info.extend(inferred_value_info)
    except Exception:
        del graph.initializer[initializer_count:]
        raise
    return model
