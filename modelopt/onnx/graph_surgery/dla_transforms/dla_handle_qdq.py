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

"""Wrap every QuantizeLinear / DequantizeLinear node with Unsqueeze → op → Squeeze.

Both Q and DQ are handled independently regardless of whether their data input
is a static initializer or a dynamic tensor.  Scale and zero-point inputs are
left untouched.  The ``axis`` attribute is shifted by ``4 - orig_rank`` to
keep per-channel quantization referencing the correct dimension.
Blocked quantization is left unchanged because its scales track the data rank.

Pattern applied to each node::

    data [*orig_shape]
    → Unsqueeze  → data_4d [1,…,1, *orig_shape]
    → Q / DQ     → out_4d  [1,…,1, *orig_shape]
    → Squeeze    → out     [*orig_shape]
"""

from __future__ import annotations

import onnx
from onnx import helper

from ...logging_config import logger
from ..utils.graph_utils import (
    GraphCache,
    add_value_info,
    axes_for_rank,
    get_node_attr_i,
    pad4d,
    set_node_attr_i,
)
from .dla_graph_cleanup import _set_axes
from .dla_remove_qdq import _qdq_output_dtype

_axes_for_rank = axes_for_rank


def _apply_handle_qdq(model: onnx.ModelProto) -> onnx.ModelProto:
    """Wrap eligible Q / DQ nodes with Unsqueeze → node → Squeeze."""
    graph = model.graph
    cache = GraphCache(graph)
    versions = {opset.domain: opset.version for opset in model.opset_import}
    opset = versions.get("", versions.get("ai.onnx", 1))

    replacements: dict[int, list[onnx.NodeProto]] = {}
    cnt = [0]

    for node_pos, node in enumerate(list(graph.node)):
        if node.domain not in {"", "ai.onnx", "com.microsoft"} or node.overload:
            continue
        if node.op_type not in ("QuantizeLinear", "DequantizeLinear"):
            continue
        if not node.input or not node.output:
            continue
        if get_node_attr_i(node, "block_size", 0) > 0:
            continue

        data_name = node.input[0]
        orig_shape = cache.get_shape(data_name)

        if orig_shape is None:
            logger.debug("%s %r: data input shape unknown; skipping.", node.op_type, node.name)
            continue
        if len(orig_shape) not in (1, 2, 3):
            continue  # already 4-D, nothing to do

        orig_rank = len(orig_shape)
        delta = 4 - orig_rank
        axes = _axes_for_rank(orig_rank)
        pfx = node.name or f"qdq_{cnt[0]}"
        c = cnt[0]
        cnt[0] += 1

        # ── Unsqueeze data input to 4D ────────────────────────────────────────
        in4d = cache.unique_name(f"{pfx}_in4d_{c}")
        unsq = helper.make_node(
            "Unsqueeze",
            inputs=[data_name],
            outputs=[in4d],
            name=cache.unique_name(f"{pfx}_unsq_{c}"),
        )
        _set_axes(model, cache, unsq, axes, f"{pfx}_unsq_axes_{c}")
        in_dtype = cache.get_dtype(data_name) or onnx.TensorProto.FLOAT
        add_value_info(graph, in4d, in_dtype, pad4d(orig_shape))
        node.input[0] = in4d

        # ── Update axis attribute (ONNX Q/DQ default: 1) ─────────────────────
        has_axis = any(a.name == "axis" for a in node.attribute)
        if (opset >= 13 or node.domain == "com.microsoft") and (has_axis or len(orig_shape) > 1):
            orig_axis = get_node_attr_i(node, "axis", 1)
            if orig_axis >= 0:
                set_node_attr_i(node, "axis", orig_axis + delta)

        # ── Squeeze output back to original rank ──────────────────────────────
        orig_out = node.output[0]
        out4d = cache.unique_name(f"{pfx}_out4d_{c}")
        out_dtype = cache.get_dtype(orig_out) or _qdq_output_dtype(node, cache)
        if out_dtype:
            add_value_info(graph, out4d, out_dtype, pad4d(orig_shape))
        node.output[0] = out4d

        c2 = cnt[0]
        cnt[0] += 1
        sq = helper.make_node(
            "Squeeze",
            inputs=[out4d],
            outputs=[orig_out],
            name=cache.unique_name(f"{pfx}_sq_{c2}"),
        )
        _set_axes(model, cache, sq, axes, f"{pfx}_sq_axes_{c2}")

        replacements[node_pos] = [unsq, node, sq]
        logger.debug(
            "%s %r: wrapped Unsqueeze→op→Squeeze (rank %d→4, axis delta=%d).",
            node.op_type,
            node.name,
            orig_rank,
            delta,
        )

    if replacements:
        nodes = [
            replacement
            for index, node in enumerate(graph.node)
            for replacement in replacements.get(index, [node])
        ]
        del graph.node[:]
        graph.node.extend(nodes)

    logger.debug("handle_qdq: wrapped %d Q/DQ node(s) with Unsqueeze/Squeeze.", len(replacements))
    return model
