# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Collapse reshape chains at the final, inferred stage of the DLA pipeline."""

import numpy as np
from onnx import TensorProto, helper

from ..utils.graph_utils import GraphCache, add_initializer, batch_replace_nodes
from .dla_graph_cleanup import _subgraph_captures
from .dla_indexing import _operator_nodes

__all__ = []


def _apply_remove_reshapes(model):
    """Collapse fixed reshape chains while retaining rank-sensitive consumers and runtime controls."""
    graph = model.graph
    cache = GraphCache(graph)
    reshapes = _operator_nodes(model, {"Reshape"})
    captured = _subgraph_captures(graph)
    chains = []
    chain = []
    for node in reshapes:
        if chain:
            consumers = cache.get_consumers(chain[-1].output[0])
            if (
                chain[-1].output[0] in cache.graph_output_names | captured
                or len(consumers) != 1
                or consumers[0][0] is not node
            ):
                chains.append(chain)
                chain = []
        chain.append(node)
    if chain:
        chains.append(chain)
    replacements = {}
    for chain in chains:
        first, last = chain[0], chain[-1]
        if any(
            node.input[1] not in cache.init_names or node.input[1] in cache.graph_input_names
            for node in chain
        ):
            continue
        input_shape = cache.get_shape(first.input[0])
        output_shape = cache.get_shape(last.output[0])
        if (
            not input_shape
            or not output_shape
            or any(d <= 0 for d in [*input_shape, *output_shape])
        ):
            continue
        removable = input_shape == output_shape
        if removable and last.output[0] not in cache.graph_output_names | captured:
            for consumer, port in cache.get_consumers(last.output[0]):
                consumer.input[port] = first.input[0]
        elif input_shape == output_shape:
            replacements[id(last)] = [
                helper.make_node(
                    "Identity",
                    [first.input[0]],
                    list(last.output),
                    name=cache.unique_name((last.name or last.output[0]) + "_identity"),
                    domain=last.domain,
                )
            ]
        elif len(chain) > 1:
            target = add_initializer(
                graph,
                cache.unique_name(last.name + "_reshape_shape"),
                np.asarray(output_shape, dtype=np.int64),
                dtype=TensorProto.INT64,
                with_value_info=False,
            )
            replacements[id(last)] = [
                helper.make_node(
                    "Reshape",
                    [first.input[0], target],
                    list(last.output),
                    name=cache.unique_name((last.name or last.output[0]) + "_reshape"),
                    domain=last.domain,
                )
            ]
        else:
            continue
        for node in chain:
            replacements.setdefault(id(node), [])
    batch_replace_nodes(
        graph,
        {
            index: ([node], replacements[id(node)])
            for index, node in enumerate(graph.node)
            if id(node) in replacements
        },
    )
    return model
