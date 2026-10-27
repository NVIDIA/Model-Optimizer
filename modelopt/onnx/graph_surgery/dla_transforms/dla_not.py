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

"""Replace ``Not`` nodes (preceded by a ``Cast``) with a float32-compatible ``Clip``/``Sub`` pattern.

DLA does not support the ``Not`` op on boolean tensors.  The common pattern is:

    X  ──► Cast(to=BOOL) ──► Not ──► Y(BOOL)

This is rewritten to:

    X  ──► Cast(to=FLOAT) ──► Clip(0, 1) ──► Sub(1 - clipped) ──► Y(FLOAT)

Only unshared Cast chains with provably binary source values are transformed.
``Not`` nodes with other producers (e.g. ``Equal``) or with graph-input operands are left in place.
"""

from __future__ import annotations

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from ...logging_config import logger
from ..utils.graph_utils import (
    GraphCache,
    add_unique_initializers,
    add_value_info,
    batch_replace_nodes,
    get_node_attr_i,
)
from .dla_constants_to_initializers import _add_legacy_initializer_inputs
from .dla_graph_cleanup import _subgraph_captures
from .dla_indexing import _operator_nodes
from .dla_rank_conversion import _runtime_input_names

__all__ = []


def _binary_source(cache: GraphCache, name: str, runtime: set[str]) -> bool:
    """Prove binary values without retaining a decoded fixed tensor."""
    if cache.get_dtype(name) == TensorProto.BOOL:
        return True
    tensor = cache.get_init(name) if name not in runtime else None
    if tensor is None:
        return False
    array = numpy_helper.to_array(tensor)
    return all(
        np.all((chunk == 0) | (chunk == 1))
        for chunk in np.nditer(
            array, flags=["external_loop", "buffered", "zerosize_ok"], buffersize=65536, order="C"
        )
    )


def _apply_not(model: onnx.ModelProto) -> onnx.ModelProto:
    """Rewrite unshared casts of binary values; preserve other Not chains."""
    graph = model.graph
    cache = GraphCache(graph)
    if next((op.version for op in model.opset_import if op.domain in {"", "ai.onnx"}), 1) < 11:
        return model
    node_idx = {id(n): i for i, n in enumerate(graph.node)}
    captured = _subgraph_captures(graph)
    casts = {id(node) for node in _operator_nodes(model, {"Cast"})}
    runtime = _runtime_input_names(model)
    binary = {}
    unique_init: set[str] = set()
    replacements: dict[int, tuple[list, list]] = {}
    cnt = 0

    for node in _operator_nodes(model, {"Not"}):
        producer_node = cache.get_producer(node.input[0])
        if producer_node is None or id(producer_node) not in casts:
            logger.debug("Not %r: input not from a Cast node — skipped.", node.name)
            continue
        if (
            len(cache.get_consumers(producer_node.output[0])) != 1
            or producer_node.output[0] in cache.graph_output_names | captured
            or node.output[0] in cache.graph_output_names | captured
        ):
            continue
        source = producer_node.input[0]
        if source not in binary:
            binary[source] = _binary_source(cache, source, runtime)
        binary_source = binary[source]
        if not binary_source or any(
            user.op_type != "Cast"
            or id(user) not in casts
            or get_node_attr_i(user, "to", 0) == TensorProto.STRING
            for user, _port in cache.get_consumers(node.output[0])
        ):
            continue

        # ── Redirect Cast: to=BOOL → to=FLOAT ────────────────────────────────
        for attr in producer_node.attribute:
            if attr.name == "to":
                attr.i = TensorProto.FLOAT

        cast_out_vi = cache.vi_map.get(producer_node.output[0])
        if cast_out_vi is not None:
            cast_out_vi.type.tensor_type.elem_type = TensorProto.FLOAT

        # ── Unique initializer names scoped to this Not node ──────────────────
        min_name = cache.unique_name(f"{node.name}_clip_min")
        max_name = cache.unique_name(f"{node.name}_clip_max")
        one_name = cache.unique_name(f"{node.name}_one")

        add_unique_initializers(
            graph,
            unique_init,
            [
                numpy_helper.from_array(np.array(0.0, dtype=np.float32), name=min_name),
                numpy_helper.from_array(np.array(1.0, dtype=np.float32), name=max_name),
                numpy_helper.from_array(np.array(1.0, dtype=np.float32), name=one_name),
            ],
        )

        clip_out = cache.unique_name(f"{node.name}_clipped")
        clip_node = helper.make_node(
            "Clip",
            inputs=[producer_node.output[0], min_name, max_name],
            outputs=[clip_out],
            name=cache.unique_name(f"{node.name}_clip"),
        )
        clip_shape = cache.get_shape(producer_node.output[0])
        if clip_shape is not None:
            add_value_info(graph, clip_out, TensorProto.FLOAT, clip_shape)

        not_out_vi = cache.vi_map.get(node.output[0])
        if not_out_vi is not None:
            not_out_vi.type.tensor_type.elem_type = TensorProto.FLOAT

        sub_node = helper.make_node(
            "Sub",
            inputs=[one_name, clip_out],
            outputs=list(node.output),
            name=cache.unique_name(f"{node.name}_sub"),
        )

        replacements[node_idx[id(node)]] = ([node], [clip_node, sub_node])
        cnt += 1
        logger.debug("Not %r: replaced with Cast(FLOAT) → Clip → Sub.", node.name)

    batch_replace_nodes(graph, replacements)

    _add_legacy_initializer_inputs(model)
    logger.debug("not: transformed %d Not node(s).", cnt)
    return model
