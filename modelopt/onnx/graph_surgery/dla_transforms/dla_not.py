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

Only ``Not`` nodes whose sole input is produced by a ``Cast`` node are transformed.
``Not`` nodes with other producers (e.g. ``Equal``) or with graph-input operands are left in place.
"""

from __future__ import annotations

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from ...logging_config import logger
from ._common import (
    GraphCache,
    add_unique_initializers,
    add_value_info,
    batch_replace_nodes,
    run_onnx_file_transform,
)
from ._dla_graph_helpers import get_tensor_value_info_by_name


def _apply_not(model: onnx.ModelProto) -> onnx.ModelProto:
    """Transform ``Cast(to=BOOL) → Not`` chains to ``Cast(to=FLOAT) → Clip → Sub``.

    Shape inference is run first so that intermediate tensor value_info entries are
    available for dtype updates.
    """
    graph = model.graph
    cache = GraphCache(graph)
    node_idx = {id(n): i for i, n in enumerate(graph.node)}
    unique_init: set[str] = set()
    replacements: dict[int, tuple[list, list]] = {}
    cnt = 0

    for node in cache.nodes_by_op("Not"):
        producer_node = cache.get_producer(node.input[0])
        if producer_node is None or producer_node.op_type != "Cast":
            logger.debug("Not %r: input not from a Cast node — skipped.", node.name)
            continue

        # ── Redirect Cast: to=BOOL → to=FLOAT ────────────────────────────────
        for attr in producer_node.attribute:
            if attr.name == "to":
                attr.i = TensorProto.FLOAT

        cast_out_vi = get_tensor_value_info_by_name(model, producer_node.output[0])
        if cast_out_vi is not None:
            cast_out_vi.type.tensor_type.elem_type = TensorProto.FLOAT

        # ── Unique initializer names scoped to this Not node ──────────────────
        min_name = f"{node.name}_clip_min"
        max_name = f"{node.name}_clip_max"
        one_name = f"{node.name}_one"

        add_unique_initializers(
            graph,
            unique_init,
            [
                numpy_helper.from_array(np.array(0.0, dtype=np.float32), name=min_name),
                numpy_helper.from_array(np.array(1.0, dtype=np.float32), name=max_name),
                numpy_helper.from_array(np.array(1.0, dtype=np.float32), name=one_name),
            ],
        )

        clip_out = f"{node.name}_clipped"
        clip_node = helper.make_node(
            "Clip",
            inputs=[producer_node.output[0], min_name, max_name],
            outputs=[clip_out],
            name=f"{node.name}_clip",
        )
        clip_shape = cache.get_shape(producer_node.output[0])
        if clip_shape is not None:
            add_value_info(graph, clip_out, TensorProto.FLOAT, clip_shape)

        # Update Not output dtype to FLOAT (graph output or intermediate value_info)
        not_out_vi = get_tensor_value_info_by_name(model, node.output[0])
        if not_out_vi is not None:
            not_out_vi.type.tensor_type.elem_type = TensorProto.FLOAT

        sub_node = helper.make_node(
            "Sub",
            inputs=[one_name, clip_out],
            outputs=list(node.output),
            name=f"{node.name}_sub",
        )

        replacements[node_idx[id(node)]] = ([node], [clip_node, sub_node])
        cnt += 1
        logger.debug("Not %r: replaced with Cast(FLOAT) → Clip → Sub.", node.name)

    batch_replace_nodes(graph, replacements)

    logger.debug("not: transformed %d Not node(s).", cnt)
    return model


def dla_not(
    model_path: str,
    output_path: str,
    *,
    use_external_data: bool = True,
    external_data_name: str | None = None,
    verbose: bool = True,
    **kwargs,
) -> onnx.ModelProto:
    """Load ONNX from ``model_path``, rewrite Cast→Not chains, save to ``output_path``."""
    return run_onnx_file_transform(
        model_path,
        output_path,
        _apply_not,
        use_external_data=use_external_data,
        external_data_name=external_data_name,
        verbose=verbose,
        **kwargs,
    )
