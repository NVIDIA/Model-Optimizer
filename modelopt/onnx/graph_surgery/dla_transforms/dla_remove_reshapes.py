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

from itertools import pairwise

import numpy as np
from onnx import TensorProto, helper

from ..utils.graph_utils import GraphCache, add_initializer, batch_replace_nodes
from .dla_constants_to_initializers import _add_legacy_initializer_inputs
from .dla_graph_cleanup import _subgraph_captures
from .dla_indexing import _operator_nodes
from .dla_rank_conversion import _runtime_input_names

__all__ = []


def _apply_remove_reshapes(model):
    """Collapse fixed reshape chains while retaining rank-sensitive consumers and runtime controls."""
    graph = model.graph
    cache = GraphCache(graph)
    runtime_inputs = _runtime_input_names(model)
    reshapes = _operator_nodes(model, {"Reshape"})
    captured = _subgraph_captures(graph)
    positions = {id(node): index for index, node in enumerate(reshapes)}
    chains, tails = [], {}
    for node in reshapes:
        producer = cache.get_producer(node.input[0])
        chain = tails.pop(id(producer), None)
        if chain is None:
            chain = []
            chains.append(chain)
        chain.append(node)
        consumers = cache.get_consumers(node.output[0])
        if (
            node.output[0] not in cache.graph_output_names | captured
            and len(consumers) == 1
            and consumers[0][1] == 0
        ):
            tails[id(node)] = chain
    replacements = {}
    for chain in chains:
        first, last = chain[0], chain[-1]
        if any(
            node.input[1] not in cache.init_names or node.input[1] in runtime_inputs
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
            if all(cache.get_shape(node.output[0]) == input_shape for node in chain[:-1]) and any(
                positions[id(right)] != positions[id(left)] + 1 for left, right in pairwise(chain)
            ):
                # Interleaved identity prefixes need only bypassing; retain the existing target and attributes.
                last.input[0] = first.input[0]
                for node in chain[:-1]:
                    replacements[id(node)] = []
                continue
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
    _add_legacy_initializer_inputs(model)
    return model
