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

"""Legacy transform: remove no-op Reshape nodes."""

import numpy as np
import onnx
from onnx import helper, numpy_helper

from ...logging_config import logger
from ._common import infer_shapes, run_onnx_file_transform
from ._dla_graph_helpers import (
    get_consumer_nodes_by_input,
    get_tensor_shape_by_name,
    replace_nodes_with_topological_order,
)


def _apply_remove_reshapes(model):
    """Check for chain of reshape nodes and remove or replace with one reshape node."""
    reshape_chain = []
    nodes_to_replace = []
    cnts = [0, 0, 0]
    all_reshape_nodes = [node for node in model.graph.node if node.op_type == "Reshape"]

    def replace_or_remove_reshape_chain(reshape_chain, nodes_to_replace, cnts):
        input_shape = np.array(
            get_tensor_shape_by_name(model.graph, reshape_chain[0].input[0]), dtype=np.int64
        )
        output_shape = np.array(
            get_tensor_shape_by_name(model.graph, reshape_chain[-1].output[0]), dtype=np.int64
        )
        if (input_shape == output_shape).all():
            # remove the chain from graph.
            nodes_to_replace.append((reshape_chain, []))
            consumers = get_consumer_nodes_by_input(model, reshape_chain[-1].output[0])
            for consumer in consumers:
                if consumer[0].input[consumer[1]] == reshape_chain[-1].output[0]:
                    consumer[0].input[consumer[1]] = reshape_chain[0].input[0]
            cnts[0] += len(reshape_chain)
        elif len(reshape_chain) > 1:
            # replace the chain with one reshape node.
            reshape_shape = numpy_helper.from_array(
                output_shape, reshape_chain[-1].name + "_reshape_shape"
            )
            model.graph.initializer.append(reshape_shape)
            reshape_node = helper.make_node(
                "Reshape",
                inputs=[reshape_chain[0].input[0], reshape_shape.name],
                outputs=reshape_chain[-1].output,
                name=reshape_chain[-1].name + "_reshape",
            )
            nodes_to_replace.append((reshape_chain, [reshape_node]))
            cnts[1] += 1
            cnts[2] += len(reshape_chain)

    for node in all_reshape_nodes:
        if len(reshape_chain) == 0:
            reshape_chain.append(node)
        else:
            consumers = get_consumer_nodes_by_input(model, reshape_chain[-1].output[0])
            if len(consumers) == 1 and consumers[0][0].name == node.name:
                reshape_chain.append(node)
            else:
                replace_or_remove_reshape_chain(reshape_chain, nodes_to_replace, cnts)
                reshape_chain = [node]
    if len(reshape_chain) > 0:
        replace_or_remove_reshape_chain(reshape_chain, nodes_to_replace, cnts)

    for nodes_to_remove, nodes_to_add in nodes_to_replace:
        replace_nodes_with_topological_order(model.graph, nodes_to_remove, nodes_to_add)
    logger.debug(
        "Removed %d reshape nodes and replaced %d reshape nodes with %d new reshape nodes.",
        cnts[0],
        cnts[2],
        cnts[1],
    )
    model = infer_shapes(model)
    return model


def dla_remove_reshapes(
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
        _apply_remove_reshapes,
        use_external_data=use_external_data,
        external_data_name=external_data_name,
        verbose=verbose,
        **kwargs,
    )
