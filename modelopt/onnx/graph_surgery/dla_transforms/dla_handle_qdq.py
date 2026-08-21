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

Pattern applied to each node::

    data [*orig_shape]
    → Unsqueeze  → data_4d [1,…,1, *orig_shape]
    → Q / DQ     → out_4d  [1,…,1, *orig_shape]
    → Squeeze    → out     [*orig_shape]
"""

from __future__ import annotations

import numpy as np
import onnx
from onnx import helper, numpy_helper

from ...logging_config import logger
from ._common import (
    GraphCache,
    add_unique_initializers,
    add_value_info,
    axes_for_rank,
    batch_replace_nodes,
    get_node_attr_i,
    pad4d,
    run_onnx_file_transform,
    set_node_attr_i,
)

_axes_for_rank = axes_for_rank


def _apply_handle_qdq(model: onnx.ModelProto) -> onnx.ModelProto:
    """Wrap every Q / DQ node with Unsqueeze → node → Squeeze."""
    graph = model.graph
    cache = GraphCache(graph)
    node_idx = {id(n): i for i, n in enumerate(graph.node)}
    unique_init: set[str] = set()

    replacements: dict[int, tuple[list, list]] = {}
    cnt = [0]

    for node in list(graph.node):
        if node.op_type not in ("QuantizeLinear", "DequantizeLinear"):
            continue
        if not node.input or not node.output:
            continue

        data_name = node.input[0]
        orig_shape = cache.get_shape(data_name)

        if orig_shape is None:
            logger.debug("%s %r: data input shape unknown; skipping.", node.op_type, node.name)
            continue
        if len(orig_shape) == 4:
            continue  # already 4-D, nothing to do

        orig_rank = len(orig_shape)
        delta = 4 - orig_rank
        axes = _axes_for_rank(orig_rank)
        pfx = node.name or f"qdq_{cnt[0]}"
        c = cnt[0]
        cnt[0] += 1

        # ── Unsqueeze data input to 4D ────────────────────────────────────────
        unsq_axes_init = numpy_helper.from_array(
            np.array(axes, dtype=np.int64), name=f"{pfx}_unsq_axes_{c}"
        )
        add_unique_initializers(graph, unique_init, [unsq_axes_init])
        in4d = f"{pfx}_in4d_{c}"
        unsq = helper.make_node(
            "Unsqueeze",
            inputs=[data_name, unsq_axes_init.name],
            outputs=[in4d],
            name=f"{pfx}_unsq_{c}",
        )
        in_dtype = cache.get_dtype(data_name) or onnx.TensorProto.FLOAT
        add_value_info(graph, in4d, in_dtype, pad4d(orig_shape))
        node.input[0] = in4d

        # ── Update axis attribute (ONNX Q/DQ default: 1) ─────────────────────
        has_axis = any(a.name == "axis" for a in node.attribute)
        if has_axis:
            orig_axis = get_node_attr_i(node, "axis", 1)
            if orig_axis >= 0:
                set_node_attr_i(node, "axis", orig_axis + delta)

        # ── Squeeze output back to original rank ──────────────────────────────
        orig_out = node.output[0]
        out4d = f"{pfx}_out4d_{c}"
        out_dtype = cache.get_dtype(orig_out) or in_dtype
        add_value_info(graph, out4d, out_dtype, pad4d(orig_shape))
        node.output[0] = out4d

        c2 = cnt[0]
        cnt[0] += 1
        sq_axes_init = numpy_helper.from_array(
            np.array(axes, dtype=np.int64), name=f"{pfx}_sq_axes_{c2}"
        )
        add_unique_initializers(graph, unique_init, [sq_axes_init])
        sq = helper.make_node(
            "Squeeze",
            inputs=[out4d, sq_axes_init.name],
            outputs=[orig_out],
            name=f"{pfx}_sq_{c2}",
        )

        replacements[node_idx[id(node)]] = ([node], [unsq, node, sq])
        logger.debug(
            "%s %r: wrapped Unsqueeze→op→Squeeze (rank %d→4, axis delta=%d).",
            node.op_type,
            node.name,
            orig_rank,
            delta,
        )

    batch_replace_nodes(graph, replacements)

    logger.debug("handle_qdq: wrapped %d Q/DQ node(s) with Unsqueeze/Squeeze.", len(replacements))
    return model


def dla_handle_qdq(
    model_path: str,
    output_path: str,
    *,
    use_external_data: bool = True,
    external_data_name: str | None = None,
    verbose: bool = True,
    **kwargs,
) -> onnx.ModelProto:
    """Load ONNX from ``model_path``, wrap Q/DQ nodes, save to ``output_path``."""
    return run_onnx_file_transform(
        model_path,
        output_path,
        _apply_handle_qdq,
        use_external_data=use_external_data,
        external_data_name=external_data_name,
        verbose=verbose,
        **kwargs,
    )
