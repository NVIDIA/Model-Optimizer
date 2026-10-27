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

"""Convert InstanceNormalization nodes to 4D-compatible form.

Transform rule
--------------
Only 3D and 4D inputs are supported:

* **4D input** ``[N, C, H, W]`` — no transformation needed; node is left untouched.
* **3D input** ``[N, C, D]`` — the node is replaced by a four-node sequence:

  .. code-block::

      [N,C,D]  →  Reshape[N,C,D,1]  →  InstanceNorm  →  Reshape[1,N,C,D]  →  Squeeze(axis=0)  →  [N,C,D]

  The intermediate tensor after the second Reshape is ``[1,N,C,D]`` — a fully 4D tensor
  compatible with downstream DLA ops.  The final Squeeze restores the original 3D shape
  so that downstream consumers see the expected rank.

* **>4D input** — raises ``ValueError`` (unsupported).

Numerical correctness
---------------------
InstanceNorm normalises each ``(N, C)`` slice over its spatial dimensions.
For ``[N,C,D,1]`` the spatial product is ``D x 1 = D``, identical to the original
``[N,C,D]`` case, so the transform is numerically lossless.
"""

from __future__ import annotations

import numpy as np
import onnx
from onnx import helper, numpy_helper

from ...logging_config import logger
from ..utils.graph_utils import (
    GraphCache,
    add_unique_initializers,
    add_value_info,
    batch_replace_nodes,
)
from .dla_constants_to_initializers import _add_legacy_initializer_inputs
from .dla_indexing import _known_rank, _operator_nodes
from .dla_model_boundaries import _require_adapter_opset


def _apply_fix_instancenorm_channel_mismatch(model: onnx.ModelProto) -> onnx.ModelProto:
    """Replace every 3D InstanceNormalization with a 4D-compatible Reshape/InstanceNorm/Reshape/Squeeze sequence."""
    # ── Shape inference ───────────────────────────────────────────────────────
    graph = model.graph
    cache = GraphCache(graph)
    node_idx = {id(n): i for i, n in enumerate(graph.node)}
    unique_init: set = set(cache.init_names)

    replacements: dict[int, tuple[list, list]] = {}
    cnt = 0

    nodes = _operator_nodes(model, {"InstanceNormalization"})
    for node in nodes:
        rank = _known_rank(cache, node.input[0])
        if rank is not None and rank not in (3, 4):
            raise ValueError(
                f"InstanceNormalization {node.name!r}: input rank {rank} is not supported"
            )
    if any(
        (shape := cache.get_shape(node.input[0])) is not None and len(shape) == 3 and all(shape)
        for node in nodes
    ):
        _require_adapter_opset(model)

    for node in nodes:
        in_name = node.input[0]
        out_name = node.output[0]

        in_shape = cache.get_shape(in_name)
        if in_shape is None:
            logger.warning(
                "InstanceNormalization node %r: input shape unknown; skipping.", node.name
            )
            continue

        rank = len(in_shape)

        if rank == 4:
            continue  # Already 4D — nothing to do.

        if not all(in_shape):
            continue

        n, c, d = in_shape
        pfx = node.name or f"instnorm_{cnt}"

        # 1. Pre-Reshape: [N,C,D] → [N,C,D,1]  (add trailing unary spatial dim)
        pre_shape_init = numpy_helper.from_array(
            np.array([n, c, d, 1], dtype=np.int64),
            name=cache.unique_name(f"{pfx}_pre_shape_{cnt}"),
        )
        add_unique_initializers(graph, unique_init, [pre_shape_init])
        pre_out = cache.unique_name(f"{in_name}_trail4d_{cnt}")
        pre_reshape = helper.make_node(
            "Reshape",
            inputs=[in_name, pre_shape_init.name],
            outputs=[pre_out],
            name=cache.unique_name(f"{pfx}_pre_reshape_{cnt}"),
        )
        in_dtype = cache.get_dtype(in_name) or onnx.TensorProto.FLOAT
        add_value_info(graph, pre_out, in_dtype, [n, c, d, 1])

        # 2. InstanceNorm on [N,C,D,1] → [N,C,D,1]
        instrnorm_out = cache.unique_name(f"{out_name}_4d_{cnt}")
        new_instrnorm = onnx.NodeProto()
        new_instrnorm.CopyFrom(node)
        new_instrnorm.input[0] = pre_out
        new_instrnorm.output[0] = instrnorm_out
        new_instrnorm.name = cache.unique_name(f"{pfx}_4d_{cnt}")
        add_value_info(graph, instrnorm_out, in_dtype, [n, c, d, 1])

        # 3. Post-Reshape: [N,C,D,1] → [1,N,C,D]  (move spatial dim to batch position)
        post_shape_init = numpy_helper.from_array(
            np.array([1, n, c, d], dtype=np.int64),
            name=cache.unique_name(f"{pfx}_post_shape_{cnt}"),
        )
        add_unique_initializers(graph, unique_init, [post_shape_init])
        post_out = cache.unique_name(f"{out_name}_1ncd_{cnt}")
        post_reshape = helper.make_node(
            "Reshape",
            inputs=[instrnorm_out, post_shape_init.name],
            outputs=[post_out],
            name=cache.unique_name(f"{pfx}_post_reshape_{cnt}"),
        )
        add_value_info(graph, post_out, in_dtype, [1, n, c, d])

        # 4. Squeeze axis=0: [1,N,C,D] → [N,C,D]  (restore original rank)
        squeeze_axes_init = numpy_helper.from_array(
            np.array([0], dtype=np.int64),
            name=cache.unique_name(f"{pfx}_squeeze_axes_{cnt}"),
        )
        add_unique_initializers(graph, unique_init, [squeeze_axes_init])
        squeeze = helper.make_node(
            "Squeeze",
            inputs=[post_out, squeeze_axes_init.name],
            outputs=[out_name],
            name=cache.unique_name(f"{pfx}_squeeze_{cnt}"),
        )

        replacements[node_idx[id(node)]] = (
            [node],
            [pre_reshape, new_instrnorm, post_reshape, squeeze],
        )
        logger.debug(
            "InstanceNormalization %r [%s] → Reshape[%s,1]→InstanceNorm→Reshape[1,%s]→Squeeze",
            node.name,
            ",".join(str(d) for d in in_shape),
            ",".join(str(d) for d in in_shape),
            ",".join(str(d) for d in in_shape),
        )
        cnt += 1

    # ── Apply replacements ─────────────────────────────────────────────────────
    batch_replace_nodes(graph, replacements)
    _add_legacy_initializer_inputs(model)

    logger.debug(
        "fix_instancenorm_channel_mismatch: transformed %d InstanceNormalization node(s).",
        len(replacements),
    )
    return model
