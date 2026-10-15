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

from onnx import TensorProto, helper

from ..utils.graph_utils import GraphCache, add_value_info, insert_nodes_at_position


def _apply_unsqueeze(model):
    """Transform Unsqueeze node when data is of type int32 or int64."""
    cache = GraphCache(model.graph)
    nodes_to_add_at_pos = []
    for node in cache.nodes_by_op("Unsqueeze"):
        data_dtype = cache.get_dtype(node.input[0])
        if data_dtype in (TensorProto.INT32, TensorProto.INT64):
            integer_ports = {
                "Gather": {1},
                "GatherElements": {1},
                "Reshape": {1},
                "Expand": {1},
                "Tile": {1},
                "TopK": {1},
                "Slice": {1, 2, 3, 4},
                "Unsqueeze": {1},
                "Squeeze": {1},
                "ReduceSum": {1},
                "ReduceMean": {1},
                "ReduceMax": {1},
                "ReduceMin": {1},
            }
            node_input_pair = [
                (consumer, port)
                for consumer, port in cache.get_consumers(node.output[0])
                if port not in integer_ports.get(consumer.op_type, set())
            ]
            if not node_input_pair:
                continue
            cast_output_name = cache.unique_name(f"{node.output[0]}_float32")
            cast_node = helper.make_node(
                "Cast",
                inputs=[node.output[0]],
                outputs=[cast_output_name],
                to=TensorProto.FLOAT,
                name=f"{node.output[0]}_cast_to_float32",
            )
            add_value_info(
                model.graph, cast_output_name, TensorProto.FLOAT, cache.get_shape(node.output[0])
            )
            for consumer, idx in node_input_pair:
                consumer.input[idx] = cast_output_name
            if node_input_pair:
                nodes_to_add_at_pos.append(([cast_node], node_input_pair[0][0]))
    for nodes_to_add, node_pos in nodes_to_add_at_pos:
        insert_nodes_at_position(model.graph, nodes_to_add, node_pos)
    return model
