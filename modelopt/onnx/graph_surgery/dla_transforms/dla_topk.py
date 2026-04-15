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

"""Rewrite the ``TopK → Cast → Reshape → Tile → Cast → GatherElements`` subgraph.

Pattern detected (from the image / scratch_space reference)::

    x [1,1,1,N]  ──► TopK(axis=last, K=k)
                          ├── values   [1,k]
                          └── indices  [1,k]
                               └── Cast(INT64) → [1,k]
                                    └── Reshape → [1,k,1]
                                         └── Tile(repeats=[1,k_repeats,1]) → [1,k,k_repeats]
                                              └── Cast → [1,k,k_repeats]
                                                   └── GatherElements(axis=2)
                                                            ├── data [...]
                                                            └── indices (above)

Rewritten to::

    x [1,N,1,1]  ──► Reshape → [1,N,1,1]
                      └── TopK(axis=1, K=k)
                               ├── values   [1,k,1,1]
                               └── indices  [1,k,1,1]
                                    └── Cast(INT32)
                                         └── Squeeze(axes=[0,2,3]) → [k]
                                              └── Gather(axis=2)
                                                       ├── data [...]
                                                       └── indices (above)

Conditions for the rewrite:
* ``k`` tensor is a scalar initializer.
* Indices output feeds exactly one ``Cast`` → one ``Reshape`` → one ``Tile`` → one ``Cast``
  → one ``GatherElements`` chain.
* ``GatherElements`` axis == 2 (the canonical DLA form).
* Tile repeats: all 1 except ``repeats[gather_axis + 1]`` which equals the tiled dim size.
* Input ``x`` has shape ``[1, 1, 1, N]`` (first three dims equal 1, last > 1).
"""

from __future__ import annotations

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from ...logging_config import logger
from ._common import GraphCache, insert_nodes_at_position, run_onnx_file_transform, set_node_attr_i
from ._dla_graph_helpers import update_tensor_shape_by_name


def _apply_topk(model: onnx.ModelProto) -> onnx.ModelProto:
    """Detect and rewrite the TopK→Cast→Reshape→Tile→Cast→GatherElements chain."""
    graph = model.graph
    cache = GraphCache(graph)
    nodes_to_add: list[tuple] = []  # (reference_node, [new_nodes])
    nodes_to_remove: list = []
    cnt = 0

    for node in cache.nodes_by_op("TopK"):
        if len(node.input) != 2:
            raise ValueError(
                f"[dla_topk] TopK node {node.name!r}: expected 2 inputs, got {len(node.input)}"
            )
        x_tensor, k_tensor = node.input

        if len(node.output) != 2:
            raise ValueError(
                f"[dla_topk] TopK node {node.name!r}: expected 2 outputs, got {len(node.output)}"
            )
        _, indices_tensor = node.output

        # K must be a scalar initializer
        if not cache.is_init(k_tensor):
            continue
        k_arr = cache.get_init_array(k_tensor)
        if k_arr is None or k_arr.size != 1:
            continue

        # ── Walk the chain: indices → Cast → Reshape → Tile → Cast → GatherElements ──

        def _single_consumer_of_type(tensor_name, op_type):
            """Return (consumer_node, idx) if there is exactly one consumer of op_type, else None."""
            if tensor_name is None:
                return None
            consumers = cache.get_consumers(tensor_name)
            if len(consumers) == 1:
                n, idx = consumers[0]
                if n.op_type == op_type:
                    return n
            return None

        cast1_node = _single_consumer_of_type(indices_tensor, "Cast")
        if cast1_node is None:
            continue
        cast1_output = cast1_node.output[0]

        reshape_node = _single_consumer_of_type(cast1_output, "Reshape")
        if reshape_node is None:
            continue
        reshape_output = reshape_node.output[0]

        tile_node = _single_consumer_of_type(reshape_output, "Tile")
        if tile_node is None:
            continue
        tile_output = tile_node.output[0]

        ge_node = _single_consumer_of_type(tile_output, "GatherElements")
        if ge_node is None:
            continue
        # ── Validate GatherElements conditions ───────────────────────────────
        if len(ge_node.input) != 2:
            raise ValueError(
                f"[dla_topk] GatherElements {ge_node.name!r}: expected 2 inputs, got {len(ge_node.input)}"
            )
        gather_data, gather_indices = ge_node.input
        if gather_indices != tile_output:
            continue  # indices input is not the one we walked

        gather_axis = next((int(a.i) for a in ge_node.attribute if a.name == "axis"), None)
        if gather_axis != 1:
            continue  # only axis==1 is handled

        tile_repeats_arr = cache.get_init_array(tile_node.input[1])
        if tile_repeats_arr is None:
            continue

        tile_out_shape = cache.get_shape(tile_output)
        if tile_out_shape is None:
            continue

        can_replace = True
        for dim, value in enumerate(tile_repeats_arr.tolist()):
            if dim == gather_axis + 1:
                can_replace = can_replace and (value == tile_out_shape[dim])
            else:
                can_replace = can_replace and (value == 1)
        if not can_replace:
            continue

        # ── Validate input shape ──────────────────────────────────────────────
        input_shape = cache.get_shape(x_tensor)
        if input_shape is None or len(input_shape) != 2:
            raise ValueError(f"[dla_topk] TopK {node.name!r}: expected 2D input, got {input_shape}")
        if not (int(np.prod(input_shape[:1])) == 1 and input_shape[1] > 1):
            continue  # only [1,N] layout

        k_val = int(k_arr.item())
        n_dim = input_shape[1]
        # ── Build replacement ─────────────────────────────────────────────────
        pfx = node.name or f"topk_{cnt}"
        cnt += 1
        new_shape = (1, n_dim, 1, 1)
        # Step 1 — Reshape x [1,1,1,N] → [1,N,1,1]
        rs_const_name = f"{pfx}_reshape_const"
        rs_const = numpy_helper.from_array(np.array(new_shape, dtype=np.int64), name=rs_const_name)
        graph.initializer.append(rs_const)
        rs_out_name = f"{pfx}_reshape_out"
        graph.value_info.append(
            helper.make_tensor_value_info(
                rs_out_name,
                cache.get_dtype(x_tensor),
                new_shape,
            )
        )
        rs_node = helper.make_node(
            "Reshape",
            inputs=[x_tensor, rs_const_name],
            outputs=[rs_out_name],
            name=f"{pfx}_reshape",
        )
        nodes_to_add.append((node, [rs_node]))
        # Step 2 — Redirect TopK to new input, set axis=1
        node.input[:] = [rs_out_name, k_tensor]
        _set_attr_i(node, "axis", 1)
        for outp in node.output:
            update_tensor_shape_by_name(model, outp, (1, k_val, 1, 1))
        # Step 3 — Keep Cast1 on TopK indices; change target to INT32
        _set_attr_i(cast1_node, "to", TensorProto.INT32)
        update_tensor_shape_by_name(model, cast1_output, (1, k_val, 1, 1))

        # Step 4 — Insert Squeeze(axes=[0,2,3]) after Cast1 → produces 1D indices
        sq_axes_name = f"{pfx}_sq_axes"
        graph.initializer.append(
            numpy_helper.from_array(np.array([0, 2, 3], dtype=np.int64), name=sq_axes_name)
        )
        sq_out_name = f"{pfx}_sq_out"
        graph.value_info.append(
            helper.make_tensor_value_info(
                sq_out_name,
                TensorProto.INT32,
                (k_val,),
            )
        )
        sq_node = helper.make_node(
            "Squeeze",
            inputs=[cast1_output, sq_axes_name],
            outputs=[sq_out_name],
            name=f"{pfx}_squeeze_indices",
        )
        nodes_to_add.append((ge_node, [sq_node]))

        # Step 5 — Change GatherElements → Gather, use sq_out as indices
        ge_node.op_type = "Gather"
        _set_attr_i(ge_node, "axis", 1)
        ge_node.input[:] = [gather_data, sq_out_name]
        # Step 6 — Remove Reshape, Tile
        nodes_to_remove.extend([reshape_node, tile_node])
        logger.debug(
            "TopK %r: rewrote TopK→Cast→Reshape→Tile→Cast→GatherElements chain (k=%d, N=%d).",
            node.name,
            k_val,
            n_dim,
        )

    for node in nodes_to_remove:
        if node in graph.node:
            graph.node.remove(node)
    for ref_node, new_nodes in nodes_to_add:
        insert_nodes_at_position(graph, new_nodes, ref_node)

    logger.debug("topk: rewrote %d TopK chain(s).", cnt)
    return model


_set_attr_i = set_node_attr_i


def dla_topk(
    model_path: str,
    output_path: str,
    *,
    use_external_data: bool = True,
    external_data_name: str | None = None,
    verbose: bool = True,
    **kwargs,
) -> onnx.ModelProto:
    """Load ONNX from ``model_path``, rewrite TopK chains, save to ``output_path``."""
    return run_onnx_file_transform(
        model_path,
        output_path,
        _apply_topk,
        use_external_data=use_external_data,
        external_data_name=external_data_name,
        verbose=verbose,
        **kwargs,
    )
