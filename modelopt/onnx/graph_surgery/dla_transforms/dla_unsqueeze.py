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

"""Insert Cast(FLOAT) after Unsqueeze nodes that produce integer tensors."""

import onnx
from onnx import TensorProto, helper

from ._common import (
    GraphCache,
    insert_nodes_at_position,
    run_onnx_file_transform,
)


def _apply_unsqueeze(model):
    """Transform Unsqueeze node when data is of type int32 or int64."""
    cache = GraphCache(model.graph)
    nodes_to_add_at_pos = []
    for node in cache.nodes_by_op("Unsqueeze"):
        data_dtype = cache.get_dtype(node.input[0])
        if data_dtype in (TensorProto.INT32, TensorProto.INT64):
            cast_output_name = f"{node.output[0]}_float32"
            cast_node = helper.make_node(
                "Cast",
                inputs=[node.output[0]],
                outputs=[cast_output_name],
                to=TensorProto.FLOAT,
                name=f"{node.output[0]}_cast_to_float32",
            )
            node_input_pair = cache.get_consumers(node.output[0])
            for consumer, idx in node_input_pair:
                consumer.input[idx] = cast_output_name
            nodes_to_add_at_pos.append(([cast_node], node_input_pair[0][0]))
    for nodes_to_add, node_pos in nodes_to_add_at_pos:
        insert_nodes_at_position(model.graph, nodes_to_add, node_pos)
    return model


def dla_unsqueeze(
    model_path: str,
    output_path: str,
    *,
    use_external_data: bool = True,
    external_data_name: str | None = None,
    verbose: bool = True,
    **kwargs,
) -> onnx.ModelProto:
    """Load ONNX from ``model_path``, apply graph transform, save to ``output_path``."""
    return run_onnx_file_transform(
        model_path,
        output_path,
        _apply_unsqueeze,
        use_external_data=use_external_data,
        external_data_name=external_data_name,
        verbose=verbose,
        **kwargs,
    )
