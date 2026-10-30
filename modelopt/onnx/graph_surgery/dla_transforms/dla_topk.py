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

"""Replace proven private TopK index tiling with a vector Gather."""

from __future__ import annotations

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from ...logging_config import logger
from ..utils.graph_utils import (
    GraphCache,
    batch_replace_nodes,
    get_node_attr_i,
    resolve_reshape_shape,
    set_node_attr_i,
)
from .dla_constants_to_initializers import _add_legacy_initializer_inputs
from .dla_graph_cleanup import _subgraph_captures
from .dla_indexing import _insert_support_nodes, _operator_nodes
from .dla_rank_conversion import _runtime_input_names


def _private_consumer(cache, name, operation, standard, protected, port=0):
    """Match one standard consumer without changing a public or captured owner."""
    users = cache.get_consumers(name)
    if name in protected or len(users) != 1:
        return None
    node, index = users[0]
    return (
        node
        if index == port
        and id(node) in standard
        and node.op_type == operation
        and len(node.output) == 1
        else None
    )


def _topk_plans(model, cache, *, rank_converted=False):
    """Prove complete private index-chain eligibility without changing the graph."""
    if next((op.version for op in model.opset_import if op.domain in {"", "ai.onnx"}), 1) < 13:
        return []
    graph = model.graph
    standard_nodes = _operator_nodes(model, {"TopK", "Cast", "Reshape", "Tile", "GatherElements"})
    standard = {id(node) for node in standard_nodes}
    topks = [node for node in standard_nodes if node.op_type == "TopK"]
    for node in topks:
        if len(node.input) != 2 or len(node.output) != 2:
            raise ValueError(f"[dla_topk] TopK {node.name!r}: expected two inputs and two outputs")
    protected = cache.graph_output_names | _subgraph_captures(graph)
    runtime = _runtime_input_names(model)
    plans = []
    for node in topks:
        values, indices = node.output
        if values in protected or cache.get_consumers(values):
            continue
        cast1 = _private_consumer(cache, indices, "Cast", standard, protected)
        if cast1 is None or get_node_attr_i(cast1, "to", 0) not in {
            TensorProto.INT32,
            TensorProto.INT64,
        }:
            continue
        reshape = _private_consumer(cache, cast1.output[0], "Reshape", standard, protected)
        tile = (
            _private_consumer(cache, reshape.output[0], "Tile", standard, protected)
            if reshape
            else None
        )
        if tile is None or len(reshape.input) != 2 or len(tile.input) != 2:
            continue
        casts = []
        cast_ids = set()
        tail = tile.output[0]
        while rank_converted:
            cast = _private_consumer(cache, tail, "Cast", standard, protected)
            if (
                cast is None
                or id(cast) in cast_ids
                or get_node_attr_i(cast, "to", 0) not in {TensorProto.INT32, TensorProto.INT64}
            ):
                break
            casts.append(cast)
            cast_ids.add(id(cast))
            tail = cast.output[0]
        gather = _private_consumer(cache, tail, "GatherElements", standard, protected, 1)
        if gather is None or len(gather.input) != 2:
            continue
        axis = 2 if rank_converted else 1
        if get_node_attr_i(gather, "axis", 0) % 4 != axis:
            continue
        shape = cache.get_shape(node.input[0])
        data_shape = cache.get_shape(gather.input[0])
        if (
            shape is None
            or len(shape) != (4 if rank_converted else 2)
            or any(dim != 1 for dim in shape[:-1])
        ):
            continue
        n = shape[-1]
        if (
            not 1 < n <= np.iinfo(np.int32).max
            or get_node_attr_i(node, "axis", -1) % len(shape) != len(shape) - 1
        ):
            continue
        if data_shape is None or len(data_shape) != 4 or data_shape[axis] != n:
            continue
        controls = [node.input[1], reshape.input[1], tile.input[1]]
        if any(name in runtime or not cache.is_init(name) for name in controls):
            continue
        k_array, target, repeats = (cache.get_init_array(name) for name in controls)
        assert k_array is not None and target is not None and repeats is not None
        if k_array.shape != (1,) or k_array.dtype != np.int64 or not 0 < int(k_array[0]) <= n:
            continue
        k = int(k_array[0])
        original_indices = [*shape[:-1], k]
        reshaped = resolve_reshape_shape(target, original_indices)
        required_shape = [1, 1, 1, 1]
        required_shape[axis] = k
        required_repeats = list(data_shape)
        required_repeats[axis] = 1
        if reshaped != required_shape or repeats.tolist() != required_repeats:
            continue
        plans.append((node, cast1, reshape, tile, casts, gather, n, k))

    return plans


def _apply_topk(model: onnx.ModelProto, *, rank_converted=False) -> onnx.ModelProto:
    """Plan exact private index rewrites before changing nodes, controls or metadata."""
    if next((op.version for op in model.opset_import if op.domain in {"", "ai.onnx"}), 1) < 13:
        return model
    graph = model.graph
    cache = GraphCache(graph)
    plans = _topk_plans(model, cache, rank_converted=rank_converted)
    before = {}
    removed = set()
    for count, (node, cast1, reshape, tile, casts, gather, n, k) in enumerate(plans):
        prefix = node.name or f"topk_{count}"
        shape_name = cache.unique_name(prefix + "_reshape_const")
        reshape_out = cache.unique_name(prefix + "_reshape_out")
        axes_name = cache.unique_name(prefix + "_sq_axes")
        squeezed = cache.unique_name(prefix + "_sq_out")
        graph.initializer.extend(
            [
                numpy_helper.from_array(np.asarray([1, n, 1, 1], np.int64), shape_name),
                numpy_helper.from_array(np.asarray([0, 2, 3], np.int64), axes_name),
            ]
        )
        graph.value_info.extend(
            [
                helper.make_tensor_value_info(
                    reshape_out, cache.get_dtype(node.input[0]), [1, n, 1, 1]
                ),
                helper.make_tensor_value_info(squeezed, TensorProto.INT32, [k]),
            ]
        )
        before[id(node)] = [
            helper.make_node(
                "Reshape",
                [node.input[0], shape_name],
                [reshape_out],
                name=cache.unique_name(prefix + "_reshape"),
            )
        ]
        before[id(gather)] = [
            helper.make_node(
                "Squeeze",
                [cast1.output[0], axes_name],
                [squeezed],
                name=cache.unique_name(prefix + "_squeeze_indices"),
            )
        ]
        node.input[0] = reshape_out
        set_node_attr_i(node, "axis", 1)
        for name in [*node.output, cast1.output[0]]:
            cache.update_shape(name, [1, k, 1, 1])
        set_node_attr_i(cast1, "to", TensorProto.INT32)
        if cast1.output[0] in cache.vi_map:
            cache.vi_map[cast1.output[0]].type.tensor_type.elem_type = TensorProto.INT32
        gather.op_type = "Gather"
        gather.input[1] = squeezed
        removed.update(id(item) for item in (reshape, tile, *casts))
    nodes = []
    replacements = {}
    for index, node in enumerate(graph.node):
        if id(node) in removed:
            replacements[index] = ([node], [])
        else:
            nodes.extend(before.get(id(node), []))
            nodes.append(node)
    batch_replace_nodes(graph, replacements)
    _insert_support_nodes(graph, nodes)
    _add_legacy_initializer_inputs(model)
    logger.debug("topk: rewrote %d TopK chain(s).", len(plans))
    return model
