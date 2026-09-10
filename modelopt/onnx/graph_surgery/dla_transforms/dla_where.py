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

"""Replace ``Where`` nodes with a float-friendly ``Mul`` when one branch is an all-zero initializer.

DLA does not support the ``Where`` op.  When one of the two data branches is a static all-zero
tensor the operation simplifies to a scalar multiply:

    Where(cond, 0, y)  →  Mul(y, 1 - float(cond))
    Where(cond, x, 0)  →  Mul(x, float(cond))

The boolean condition is converted to float32 by:

* **Static BOOL initializer** — replaced in-place by a pre-computed float32 initializer
  (or its inverse, for the zero-true-branch case).  The original bool initializer is removed
  only if the ``Where`` node is its sole consumer.
* **Dynamic BOOL tensor** — a ``Cast(to=FLOAT)`` node is inserted before the ``Mul``.
  For the zero-true-branch case a ``Sub(1, cast_out)`` is also inserted.
* **Non-BOOL condition** — used directly as the float mask (no cast needed).

``Where`` nodes whose both data inputs are non-zero (or non-initializer) are left unchanged.
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


def _zero_initializer_index(cache: GraphCache, names: list[str]) -> int | None:
    """Return the index (in ``names``) of the first all-zero initializer, or ``None``."""
    for idx, name in enumerate(names):
        if cache.is_init(name):
            arr = cache.get_init_array(name)
            if arr is not None and np.count_nonzero(arr) == 0:
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

    for node in cache.nodes_by_op("Where"):
        condition = node.input[0]
        x_name = node.input[1]  # true branch
        y_name = node.input[2]  # false branch

        zero_idx = _zero_initializer_index(cache, [x_name, y_name])
        if zero_idx is None:
            continue

        non_zero_name = y_name if zero_idx == 0 else x_name
        condition_type = cache.get_dtype(condition)
        nodes_to_add: list = []

        if cache.is_init(condition):
            if condition_type == TensorProto.BOOL:
                cond_raw = cache.get_init_array(condition)
                if cond_raw is None:
                    continue
                cond_arr = cond_raw.astype(np.float32)
                if zero_idx == 0:
                    # Where(cond, 0, y) → y * (1 - cond_float)
                    cond_arr = 1.0 - cond_arr
                float_name = f"{condition}_float32"
                add_unique_initializers(
                    graph, unique_init, [numpy_helper.from_array(cond_arr, name=float_name)]
                )
                mul_inputs = [non_zero_name, float_name]
            else:
                # Non-bool static condition — use directly as the float mask.
                mul_inputs = [non_zero_name, condition]

        else:
            # ── Dynamic condition tensor ──────────────────────────────────────
            if condition_type == TensorProto.BOOL:
                float_name = f"{condition}_float32"
                cast_node = helper.make_node(
                    "Cast",
                    inputs=[condition],
                    outputs=[float_name],
                    to=TensorProto.FLOAT,
                    name=f"{node.name}_cast_cond",
                )
                nodes_to_add.append(cast_node)
                cond_shape = cache.get_shape(condition)
                if cond_shape is not None:
                    add_value_info(graph, float_name, TensorProto.FLOAT, cond_shape)
                mask_name = float_name
            else:
                # Already a numeric type — use directly.
                mask_name = condition

            if zero_idx == 0:
                # Where(cond, 0, y) → y * (1 - mask)
                one_name = f"{node.name}_one"
                add_unique_initializers(
                    graph,
                    unique_init,
                    [numpy_helper.from_array(np.array(1.0, dtype=np.float32), name=one_name)],
                )
                inv_name = f"{node.name}_inv_cond"
                sub_node = helper.make_node(
                    "Sub",
                    inputs=[one_name, mask_name],
                    outputs=[inv_name],
                    name=f"{node.name}_inv",
                )
                nodes_to_add.append(sub_node)
                cond_shape = cache.get_shape(condition)
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
            name=f"{node.name}_mul",
        )
        nodes_to_add.append(mul_node)
        replacements[node_idx[id(node)]] = ([node], nodes_to_add)
        cnt += 1
        logger.debug("Where %r: replaced by Mul (zero_idx=%d).", node.name, zero_idx)

    batch_replace_nodes(graph, replacements)

    logger.debug("where: transformed %d Where node(s).", cnt)
    return model


def dla_where(
    model_path: str,
    output_path: str,
    *,
    use_external_data: bool = True,
    external_data_name: str | None = None,
    verbose: bool = True,
    **kwargs,
) -> onnx.ModelProto:
    """Load ONNX from ``model_path``, rewrite eligible Where nodes, save to ``output_path``."""
    return run_onnx_file_transform(
        model_path,
        output_path,
        _apply_where,
        use_external_data=use_external_data,
        external_data_name=external_data_name,
        verbose=verbose,
        **kwargs,
    )
