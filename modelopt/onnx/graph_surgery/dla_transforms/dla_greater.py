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

"""Cast INT32 / INT64 initializer inputs of ``Greater`` nodes to float32.

``Greater`` on DLA requires float32 operands.  When a comparison threshold is
stored as an integer initializer this transform replaces it with a float32
copy so no runtime cast is needed.
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
    insert_nodes_at_position,
)

_INT_TYPES = frozenset({TensorProto.INT32, TensorProto.INT64})


def _apply_greater(model: onnx.ModelProto) -> onnx.ModelProto:
    """Ensure all inputs of every Greater node are float32.

    * Initializer inputs with INT32/INT64 dtype: replaced by a float32 initializer;
      the original integer initializer is removed.
    * Non-initializer (dynamic) inputs whose dtype is not float32: a
      ``Cast(to=FLOAT)`` node is inserted before them.  Shape inference must have
      been run so that tensor dtypes are available in ``value_info``.
    """
    graph = model.graph
    cache = GraphCache(graph)
    unique_init: set[str] = set()
    cnt = 0
    converted = {}

    for node in cache.nodes_by_op("Greater"):
        for idx, inp in enumerate(node.input):
            if not inp:
                continue
            dtype = cache.get_dtype(inp)
            if inp in converted:
                node.input[idx] = converted[inp]
                continue
            if dtype == TensorProto.FLOAT:
                continue

            if cache.is_init(inp):
                if dtype not in _INT_TYPES:
                    continue
                init_arr = cache.get_init_array(inp)
                assert init_arr is not None, f"Initializer {inp!r} not found"
                arr = init_arr.astype(np.float32)
                new_name = cache.unique_name(f"{inp}_cast")
                add_unique_initializers(
                    graph, unique_init, [numpy_helper.from_array(arr, name=new_name)]
                )

                node.input[idx] = new_name
                converted[inp] = new_name
                logger.debug(
                    "Greater %r input[%d]: initializer %r cast to float32.", node.name, idx, inp
                )
            else:
                # Dynamic tensor with non-float dtype — insert Cast before this node.
                if dtype is None:
                    continue
                cast_out = cache.unique_name(f"{inp}_to_float32")
                cast_node = helper.make_node(
                    "Cast",
                    inputs=[inp],
                    outputs=[cast_out],
                    name=f"{inp}_cast_to_float32",
                    to=TensorProto.FLOAT,
                )
                insert_nodes_at_position(graph, [cast_node], node)
                node.input[idx] = cast_out
                converted[inp] = cast_out
                inp_shape = cache.get_shape(inp)
                if inp_shape is not None:
                    add_value_info(graph, cast_out, TensorProto.FLOAT, inp_shape)
                logger.debug(
                    "Greater %r input[%d]: inserted Cast for dynamic tensor %r.",
                    node.name,
                    idx,
                    inp,
                )
            cnt += 1

    logger.debug("greater: fixed %d input(s) to float32.", cnt)
    return model
