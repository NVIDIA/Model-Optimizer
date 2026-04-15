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

"""Legacy transform: replace Squeeze/Unsqueeze pairs with equivalent Reshape nodes."""

import numpy as np
import onnx
from onnx import helper, numpy_helper

from ...logging_config import logger
from ._common import add_unique_initializers, infer_shapes, run_onnx_file_transform
from ._dla_graph_helpers import get_initializer_object_by_name, replace_nodes_with_topological_order


def _apply_squeeze_unsqueeze_to_reshape(model):
    """Detect and convert Squeeze-Unsqueeze pairs to a single Reshape operation.

    This optimization helps reduce unnecessary dimension operations in the graph.

    Pattern:
    Input -> Squeeze -> Unsqueeze -> Output
    becomes
    Input -> Reshape -> Output

    The Reshape operation will use a fixed shape of [1,1,-1,1]
    """
    graph = model.graph
    unique_initializer = set()
    nodes_to_remove = []
    nodes_to_add = []
    nodes_to_replace = []

    # Create a mapping of node output names to their consumers
    output_to_consumers = {}
    for node in graph.node:
        for input_name in node.input:
            if input_name not in output_to_consumers:
                output_to_consumers[input_name] = []
            output_to_consumers[input_name].append(node)

    # Find Squeeze-Unsqueeze pairs
    for node in graph.node:
        if node.op_type == "Squeeze":
            squeeze_output = node.output[0]
            consumers = output_to_consumers.get(squeeze_output, [])

            # Check if this Squeeze is followed by an Unsqueeze
            for consumer in consumers:
                if consumer.op_type == "Unsqueeze":
                    # Create a new Reshape node
                    reshape_name = f"{node.name}_to_{consumer.name}_reshape"
                    reshape_shape_name = f"{reshape_name}_shape"

                    # ensure that squeeze -> unsqueeze is similar to [1,1,-1,1]
                    squeeze_axes = get_initializer_object_by_name(model, node.input[1])
                    squeeze_axes_array = numpy_helper.to_array(squeeze_axes)
                    if squeeze_axes is None:
                        raise ValueError(
                            "[transform_squeeze_unsqueeze_to_reshape] squeeze axes is not "
                            f"initializer for node: {node.name}"
                        )
                    if squeeze_axes_array[0] not in [0, 1, 2]:
                        raise ValueError(
                            f"[transform_squeeze_unsqueeze_to_reshape] squeeze axes is not 1 for node: {node.name}"
                        )
                    unsqueeze_axes = get_initializer_object_by_name(model, consumer.input[1])
                    unsqueeze_axes_array = numpy_helper.to_array(unsqueeze_axes)
                    if unsqueeze_axes is None:
                        raise ValueError(
                            "[transform_squeeze_unsqueeze_to_reshape] unsqueeze axes is not "
                            f"initializer for node: {consumer.name}"
                        )
                    if unsqueeze_axes_array[0] != -1:
                        raise ValueError(
                            "[transform_squeeze_unsqueeze_to_reshape] unsqueeze axes is not "
                            f"-1 for node: {consumer.name}"
                        )
                    # Create shape initializer for Reshape with fixed shape [1,1,-1,1]
                    reshape_shape = numpy_helper.from_array(
                        np.array([1, 1, -1, 1], dtype=np.int64), reshape_shape_name
                    )
                    add_unique_initializers(graph, unique_initializer, [reshape_shape])
                    # Create Reshape node
                    reshape_node = helper.make_node(
                        "Reshape",
                        inputs=[node.input[0], reshape_shape_name],
                        outputs=[consumer.output[0]],
                        name=reshape_name,
                    )

                    # Add new node and mark old nodes for removal
                    nodes_to_replace.append(([node, consumer], [reshape_node]))

                    # Update consumers of the Unsqueeze output to use the Reshape output
                    for next_node in graph.node:
                        for i, input_name in enumerate(next_node.input):
                            if input_name == consumer.output[0]:
                                next_node.input[i] = reshape_node.output[0]

    # Replace nodes using topological order preservation
    for nodes_to_remove, nodes_to_add in nodes_to_replace:
        replace_nodes_with_topological_order(graph, nodes_to_remove, nodes_to_add)

    logger.debug(
        "Converted %d Squeeze-Unsqueeze pairs to Reshape operations with shape [1,1,-1,1]",
        len(nodes_to_remove) // 2,
    )
    model = infer_shapes(model)
    return model


def dla_squeeze_unsqueeze_to_reshape(
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
        _apply_squeeze_unsqueeze_to_reshape,
        use_external_data=use_external_data,
        external_data_name=external_data_name,
        verbose=verbose,
        **kwargs,
    )
