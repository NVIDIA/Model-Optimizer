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

"""Remove paired QuantizeLinear / DequantizeLinear nodes for selected quantized dtypes.

Pairs are removed when the quantizer's zero-point tensor element type matches one of the
user-configurable storage types (default: UINT16 and INT16). See :mod:`utils.graph_utils` for
name/NumPy mappings.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import onnx
from onnx import helper, numpy_helper

from ...logging_config import logger
from ..utils.graph_utils import (
    GraphCache,
    add_unique_initializers,
    calculate_clip_range,
    insert_nodes_at_position,
)


def _remove_qdq_from_model(
    model: onnx.ModelProto,
    *,
    keep_clip_after_inputs: bool,
    max_chained_qdq_pairs: int,
    quant_types_to_remove: frozenset[int],
) -> int:
    """Mutate ``model`` in place. Returns number of QDQ pairs removed."""
    q_output_to_q_node_map: dict = {}
    dq_input_to_dq_node_map: dict = defaultdict(list)
    graph = model.graph
    cache = GraphCache(graph)
    unique_initializer: set[str] = set()
    node_datatype_map: dict[str, int] = {}

    for node in graph.node:
        if node.op_type == "QuantizeLinear":
            if not node.output[0] or len(node.output) > 1:
                continue
            q_output_to_q_node_map[node.output[0]] = node
            zp_dtype = (
                cache.get_dtype(node.input[2]) if len(node.input) > 2 else onnx.TensorProto.UINT8
            )
            if zp_dtype is not None:
                node_datatype_map[node.output[0]] = zp_dtype
            in_dtype = cache.get_dtype(node.input[0])
            if in_dtype is not None:
                node_datatype_map[node.input[0]] = in_dtype
        elif node.op_type == "DequantizeLinear":
            if not node.input[0]:
                continue
            dq_input_to_dq_node_map[node.input[0]].append(node)
            out_dtype = cache.get_dtype(node.output[0])
            if out_dtype is None and len(node.input) > 1 and node.input[1]:
                # Fall back to scale dtype — DQ output has the same float type as the scale.
                out_dtype = cache.get_dtype(node.input[1])
            if out_dtype is not None:
                node_datatype_map[node.output[0]] = out_dtype

    qdq_node_pair_output_to_input_map: dict[str, str] = {}
    cnt = 0
    input_names = {inp.name for inp in model.graph.input}
    aliases = {}
    for q_output, q_node in q_output_to_q_node_map.items():
        if node_datatype_map.get(q_output) not in quant_types_to_remove:
            continue
        for dq_node in dq_input_to_dq_node_map.get(q_output, []):
            if keep_clip_after_inputs and q_node.input[0] in input_names:
                continue
            in_dtype = node_datatype_map.get(q_node.input[0])
            out_dtype = node_datatype_map.get(dq_node.output[0])
            if in_dtype is None or out_dtype is None:
                raise ValueError(
                    f"Cannot remove QDQ pair ({q_node.name!r}, {dq_node.name!r}): "
                    "dtype is unknown. Ensure shape inference has been run."
                )
            if in_dtype == out_dtype:
                aliases[dq_node.output[0]] = q_node.input[0]
    for output in aliases:
        current, visited = output, set()
        while current in aliases:
            if current in visited or len(visited) > max_chained_qdq_pairs:
                raise ValueError(
                    f"More than {max_chained_qdq_pairs} chained QDQ pairs at tensor {output!r}"
                )
            visited.add(current)
            current = aliases[current]

    for q_output, q_node in q_output_to_q_node_map.items():
        if q_output not in dq_input_to_dq_node_map:
            continue
        q_dtype = node_datatype_map.get(q_output)
        if q_dtype is None or q_dtype not in quant_types_to_remove:
            continue

        for dq_node in dq_input_to_dq_node_map[q_output]:
            if keep_clip_after_inputs and q_node.input[0] in input_names:
                clip_min, clip_max = calculate_clip_range(dq_node, model)
                clip_min_init = numpy_helper.from_array(
                    np.asarray(clip_min, dtype=np.float32), f"{dq_node.name}_clip_min"
                )
                clip_max_init = numpy_helper.from_array(
                    np.asarray(clip_max, dtype=np.float32), f"{dq_node.name}_clip_max"
                )
                add_unique_initializers(
                    model.graph, unique_initializer, [clip_min_init, clip_max_init]
                )
                clip_node = helper.make_node(
                    "Clip",
                    inputs=[
                        q_node.input[0],
                        f"{dq_node.name}_clip_min",
                        f"{dq_node.name}_clip_max",
                    ],
                    outputs=[dq_node.output[0]],
                    name=f"{dq_node.name}_clip",
                )
                insert_nodes_at_position(graph, [clip_node], q_node)
            else:
                in_dt = node_datatype_map.get(q_node.input[0])
                out_dt = node_datatype_map.get(dq_node.output[0])
                if in_dt is None:
                    raise ValueError(
                        f"Cannot remove QDQ pair ({q_node.name!r}, {dq_node.name!r}): "
                        f"dtype of input tensor {q_node.input[0]!r} is unknown. "
                        "Ensure shape inference has been run before calling this transform."
                    )
                if out_dt is None:
                    raise ValueError(
                        f"Cannot remove QDQ pair ({q_node.name!r}, {dq_node.name!r}): "
                        f"dtype of output tensor {dq_node.output[0]!r} is unknown. "
                        "Ensure shape inference has been run before calling this transform."
                    )
                if in_dt != out_dt:
                    cast_node = helper.make_node(
                        "Cast",
                        inputs=[q_node.input[0]],
                        outputs=[dq_node.output[0]],
                        name=f"{dq_node.name}_cast",
                        to=out_dt,
                    )
                    insert_nodes_at_position(graph, [cast_node], q_node)
                else:
                    qdq_node_pair_output_to_input_map[dq_node.output[0]] = q_node.input[0]

            graph.node.remove(dq_node)
            cnt += 1

        dq_ids = {id(node) for node in dq_input_to_dq_node_map[q_output]}
        if q_output not in cache.graph_output_names and all(
            id(node) in dq_ids for node, _ in cache.get_consumers(q_output)
        ):
            graph.node.remove(q_node)

    for node in graph.node:
        for i, input_name in enumerate(node.input):
            if input_name not in qdq_node_pair_output_to_input_map:
                continue
            qdq_node_pair_output = input_name
            num_connected = 0
            while True:
                qdq_node_pair_input = qdq_node_pair_output_to_input_map[qdq_node_pair_output]
                if qdq_node_pair_input not in qdq_node_pair_output_to_input_map:
                    node.input[i] = qdq_node_pair_input
                    break
                qdq_node_pair_output = qdq_node_pair_input
                num_connected += 1
                if num_connected > max_chained_qdq_pairs:
                    raise ValueError(
                        f"More than {max_chained_qdq_pairs} chained QDQ pairs at tensor "
                        f"{input_name!r}; raise max_chained_qdq_pairs if intentional."
                    )

    consumer_node_map: dict[str, list] = defaultdict(list)
    producer_node_map: dict[str, onnx.NodeProto] = {}
    for node in graph.node:
        for input_ in node.input:
            consumer_node_map[input_].append(node)
        for output in node.output:
            producer_node_map[output] = node

    renamed_outputs: dict[str, str] = {}
    graph_output_names = {output.name for output in graph.output}
    for graph_output in graph.output:
        if graph_output.name not in qdq_node_pair_output_to_input_map:
            continue
        qdq_node_pair_output = graph_output.name
        num_connected = 0
        while True:
            qdq_node_pair_input = qdq_node_pair_output_to_input_map[qdq_node_pair_output]
            if qdq_node_pair_input not in qdq_node_pair_output_to_input_map:
                node = producer_node_map.get(qdq_node_pair_input)
                if qdq_node_pair_input in graph_output_names:
                    graph.node.append(
                        helper.make_node(
                            "Identity",
                            [qdq_node_pair_input],
                            [graph_output.name],
                            name=f"identity_output_{graph_output.name}",
                        )
                    )
                    break
                if qdq_node_pair_input in renamed_outputs:
                    graph.node.append(
                        helper.make_node(
                            "Identity",
                            [renamed_outputs[qdq_node_pair_input]],
                            [graph_output.name],
                            name=f"identity_fanout_{graph_output.name}",
                        )
                    )
                    break
                if node is None:
                    # Terminal tensor is a graph input — insert Identity to bridge it.
                    identity = helper.make_node(
                        "Identity",
                        inputs=[qdq_node_pair_input],
                        outputs=[graph_output.name],
                        name=f"identity_bypass_{graph_output.name}",
                    )
                    graph.node.append(identity)
                else:
                    for i, output_name in enumerate(node.output):
                        if output_name == qdq_node_pair_input:
                            node.output[i] = graph_output.name
                            renamed_outputs[qdq_node_pair_input] = graph_output.name
                            break
                    consumers = consumer_node_map.get(qdq_node_pair_input, [])
                    for consumer in consumers:
                        for j, input_name in enumerate(consumer.input):
                            if input_name == qdq_node_pair_input:
                                consumer.input[j] = graph_output.name
                                break
                break
            qdq_node_pair_output = qdq_node_pair_input
            num_connected += 1
            if num_connected > max_chained_qdq_pairs:
                raise ValueError(
                    f"More than {max_chained_qdq_pairs} chained QDQ pairs for graph output "
                    f"{graph_output.name!r}."
                )

    logger.debug("Removed %d QuantizeLinear and DequantizeLinear pairs", cnt)
    return cnt
