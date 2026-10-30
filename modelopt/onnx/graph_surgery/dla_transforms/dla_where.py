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

"""Lower zero-branch selections to arithmetic under the existing finite-FP32 assumption.

Fixed BOOL conditions get private FLOAT masks; runtime conditions use Cast nodes.
Overridable zero branches remain native, and initializer owners are preserved.
Masked multiplication does not preserve the sign of selected zero values.
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
    pad4d,
)
from .dla_constants_to_initializers import _add_legacy_initializer_inputs
from .dla_indexing import _operator_nodes, _prepared_index_data
from .dla_rank_conversion import _runtime_input_names

__all__ = []


def _zero_initializer_index(
    cache: GraphCache, names: list[str], runtime: set[str], zeros: dict[str, bool]
) -> int | None:
    """Return the index (in ``names``) of the first all-zero initializer, or ``None``."""
    for idx, name in enumerate(names):
        if cache.is_init(name) and name not in runtime:
            if name not in zeros:
                zeros[name] = np.count_nonzero(numpy_helper.to_array(cache.get_init(name))) == 0
            if zeros[name]:
                return idx
    return None


def _apply_where(model: onnx.ModelProto) -> onnx.ModelProto:
    """Replace eligible ``Where`` nodes with ``Mul``-based equivalents.

    Only ``Where`` nodes where exactly one of the two data inputs is an all-zero
    static initializer are transformed.  All others are left untouched.
    """
    graph = model.graph
    cache = GraphCache(graph)
    node_idx = {id(n): i for i, n in enumerate(graph.node)}
    unique_init: set[str] = set()
    replacements: dict[int, tuple[list, list]] = {}
    cnt = 0
    runtime = _runtime_input_names(model)
    zeros = {}
    static_masks = {}

    selections = _operator_nodes(model, {"Where"})
    if selections:
        for tensor in graph.initializer:
            if tensor.name in cache.vi_map and tensor.name not in runtime:
                cache.vi_map[tensor.name] = helper.make_tensor_value_info(
                    tensor.name, tensor.data_type, list(tensor.dims)
                )
    prepared = (
        _prepared_index_data(
            model,
            cache,
            (),
            [name for node in selections for name in node.input],
        )
        if selections
        else {}
    )

    def current_shape(name):
        """Use proven leading layouts without changing metadata needed by other passes."""
        shape = cache.get_shape(name)
        return pad4d(shape) if shape is not None and prepared[name] else shape

    for node in selections:
        condition = node.input[0]
        x_name = node.input[1]  # true branch
        y_name = node.input[2]  # false branch

        zero_idx = _zero_initializer_index(cache, [x_name, y_name], runtime, zeros)
        if zero_idx is None:
            continue

        non_zero_name = y_name if zero_idx == 0 else x_name
        if cache.get_dtype(non_zero_name) != TensorProto.FLOAT:
            continue
        shapes = [current_shape(name) for name in [condition, x_name, y_name]]
        if any(shape is None for shape in shapes):
            continue
        if np.broadcast_shapes(*shapes) != np.broadcast_shapes(
            shapes[0], shapes[2 if zero_idx == 0 else 1]
        ):
            continue
        condition_type = cache.get_dtype(condition)
        nodes_to_add: list = []

        if cache.is_init(condition) and condition not in runtime:
            if condition_type == TensorProto.BOOL:
                mask_key = (condition, zero_idx == 0)
                float_name = static_masks.get(mask_key)
                if float_name is None:
                    cond_raw = cache.get_init_array(condition)
                    if cond_raw is None:
                        continue
                    cond_arr = cond_raw.astype(np.float32)
                    if zero_idx == 0:
                        # Where(cond, 0, y) → y * (1 - cond_float)
                        np.subtract(1.0, cond_arr, out=cond_arr)
                    float_name = cache.unique_name(f"{condition}_float32")
                    graph.initializer.add(
                        name=float_name,
                        data_type=TensorProto.FLOAT,
                        dims=cond_arr.shape,
                        raw_data=cond_arr.astype("<f4", copy=False).tobytes(),
                    )
                    static_masks[mask_key] = float_name
                mul_inputs = [non_zero_name, float_name]
            else:
                # Non-bool static condition — use directly as the float mask.
                mul_inputs = [non_zero_name, condition]

        else:
            # ── Dynamic condition tensor ──────────────────────────────────────
            if condition_type == TensorProto.BOOL:
                float_name = cache.unique_name(f"{condition}_float32")
                cast_node = helper.make_node(
                    "Cast",
                    inputs=[condition],
                    outputs=[float_name],
                    to=TensorProto.FLOAT,
                    name=cache.unique_name(f"{node.name}_cast_cond"),
                )
                nodes_to_add.append(cast_node)
                cond_shape = current_shape(condition)
                if cond_shape is not None:
                    add_value_info(graph, float_name, TensorProto.FLOAT, cond_shape)
                mask_name = float_name
            else:
                # Already a numeric type — use directly.
                mask_name = condition

            if zero_idx == 0:
                # Where(cond, 0, y) → y * (1 - mask)
                one_name = cache.unique_name(f"{node.name}_one")
                add_unique_initializers(
                    graph,
                    unique_init,
                    [numpy_helper.from_array(np.array(1.0, dtype=np.float32), name=one_name)],
                )
                inv_name = cache.unique_name(f"{node.name}_inv_cond")
                sub_node = helper.make_node(
                    "Sub",
                    inputs=[one_name, mask_name],
                    outputs=[inv_name],
                    name=cache.unique_name(f"{node.name}_inv"),
                )
                nodes_to_add.append(sub_node)
                cond_shape = current_shape(condition)
                if cond_shape is not None:
                    add_value_info(graph, inv_name, TensorProto.FLOAT, cond_shape)
                mul_inputs = [non_zero_name, inv_name]
            else:
                # Where(cond, x, 0) → x * mask
                mul_inputs = [non_zero_name, mask_name]

        mul_node = helper.make_node(
            "Mul",
            inputs=mul_inputs,
            outputs=list(node.output),
            name=cache.unique_name(f"{node.name}_mul"),
        )
        nodes_to_add.append(mul_node)
        replacements[node_idx[id(node)]] = ([node], nodes_to_add)
        cnt += 1
        logger.debug("Where %r: replaced by Mul (zero_idx=%d).", node.name, zero_idx)

    batch_replace_nodes(graph, replacements)

    _add_legacy_initializer_inputs(model)
    logger.debug("where: transformed %d Where node(s).", cnt)
    return model
