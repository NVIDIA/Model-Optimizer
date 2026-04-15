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

"""Legacy transform: remove intermediary Squeeze and Unsqueeze nodes."""

import numpy as np
import onnx
from onnx import helper, numpy_helper

from ...logging_config import logger
from ._common import (
    add_unique_initializers,
    infer_shapes,
    insert_nodes_at_position,
    pad4d,
    run_onnx_file_transform,
)
from ._dla_graph_helpers import get_tensor_shape_map


def _apply_remove_intermediary_squeeze_and_unsqueeze(model):
    """Remove intermediary Squeeze/Unsqueeze nodes and replace with Reshape operations.

    Handles special case of Squeeze with axis 2 and 5D input by converting to Reshape.
    Also replaces eligible Unsqueeze nodes with Reshape.
    """
    graph = model.graph
    unique_initializer = set()
    input_names = {input_.name for input_ in graph.input}
    output_names = {output.name for output in graph.output}
    tensor_name_dim_map = get_tensor_shape_map(graph.value_info)
    initializer_dim_map = {init.name: list(init.dims) for init in graph.initializer}
    squeeze_unsqueeze_removed_cnt = 0
    unsqueeze_replaced_cnt = 0

    def get_shape(name):
        if name in tensor_name_dim_map:
            return tensor_name_dim_map[name]
        if name in initializer_dim_map:
            return initializer_dim_map[name]
        # Check graph inputs
        for graph_input in graph.input:
            if graph_input.name == name:
                tensor_type = graph_input.type.tensor_type
                shape = []
                if tensor_type.HasField("shape"):
                    for d in tensor_type.shape.dim:
                        if d.HasField("dim_value"):
                            shape.append(d.dim_value)
                        else:
                            shape.append(0)
                return shape
        return None

    # Build output to consumers map
    output_to_consumers = {}
    for node in graph.node:
        for input_name in node.input:
            output_to_consumers.setdefault(input_name, []).append(node)

    nodes_to_remove = []
    nodes_to_add = []
    input_shape = None
    for node in graph.node:
        # --- Remove intermediary Squeeze nodes ---
        if len(node.input) > 0:
            input_shape = get_shape(
                node.input[0]
            )  #### without this if condition, it will give error for constant op

        # --- Handle special case: Squeeze with axis 2 and 5D input ---
        if (
            node.op_type == "Squeeze" and input_shape is not None and len(input_shape) == 5
        ):  # model-specific: squeeze on axis 2 of 5D input; axis and size are fixed
            # Get axes from initializer
            axes = None
            for init in graph.initializer:
                if init.name == node.input[1]:
                    axes = numpy_helper.to_array(init)
                    break
            if axes is None:
                for n in graph.node:
                    if n.op_type == "Constant" and n.output[0] == node.input[1]:
                        for attr in n.attribute:
                            if attr.name == "value":
                                axes = numpy_helper.to_array(attr.t)
                                break
                        break

            # Check if axis is 2
            if axes is not None and len(axes) == 1 and axes[0] in [2, 3]:
                # Convert squeeze to reshape with shape [1, input[1], input[3], input[4]]
                reshape_shape = [*input_shape[: axes[0]], *input_shape[axes[0] + 1 :]]

                # Create shape initializer for Reshape
                reshape_shape_name = node.name + "_reshape_shape"
                reshape_shape_init = numpy_helper.from_array(
                    np.array(reshape_shape, dtype=np.int64), reshape_shape_name
                )
                add_unique_initializers(graph, unique_initializer, [reshape_shape_init])
                # Create Reshape node
                reshape_node = helper.make_node(
                    "Reshape",
                    inputs=[node.input[0], reshape_shape_name],
                    outputs=node.output,
                    name=node.name + "_to_reshape",
                )
                nodes_to_add.append(reshape_node)
                nodes_to_remove.append(node)
                squeeze_unsqueeze_removed_cnt += 1
                # Rewire consumers
                squeeze_output = node.output[0]
                consumers = output_to_consumers.get(squeeze_output, [])
                for consumer in consumers:
                    for i, input_name in enumerate(consumer.input):
                        if input_name == squeeze_output:
                            consumer.input[i] = reshape_node.output[0]
                continue  # Don't process further
            else:
                error_msg = (
                    f"[transform_remove_intermediary_squeeze_and_unsqueeze] "
                    f"axis is not None, axis = {axes} for node: {node.name}"
                    if axes is not None
                    else f"[transform_remove_intermediary_squeeze_and_unsqueeze] axes is None for node: {node.name}"
                )
                raise ValueError(error_msg)

        #### Here we are checking if the input and output of squeeze/unsqueeze are not model inputs/outputs ####
        unsqueeze_4d = (
            node.op_type == "Unsqueeze" and input_shape is not None and len(input_shape) == 4
        )
        if (
            (node.op_type == "Squeeze" or unsqueeze_4d)
            and node.input[0] not in input_names
            and node.output[0] not in output_names
        ):
            squeeze_input = node.input[0]
            squeeze_output = node.output[0]
            squeeze_input_shape = get_shape(squeeze_input)
            squeeze_output_shape = get_shape(squeeze_output)
            # make squeeze input and output 4D by adding unary dims at front
            squeeze_input_shape = pad4d(squeeze_input_shape)
            squeeze_output_shape = pad4d(squeeze_output_shape)
            # if squeeze input and output are not 4D, replace squeeze with reshape
            if squeeze_input_shape != squeeze_output_shape:
                reshape_shape_init = numpy_helper.from_array(
                    np.array(squeeze_output_shape, dtype=np.int64), node.name + "_reshape_shape"
                )
                add_unique_initializers(graph, unique_initializer, [reshape_shape_init])
                reshape_node = helper.make_node(
                    "Reshape",
                    inputs=[squeeze_input, node.name + "_reshape_shape"],
                    outputs=[squeeze_output],
                    name=node.name + "_to_reshape",
                )
                nodes_to_add.append(reshape_node)
                nodes_to_remove.append(node)
                squeeze_unsqueeze_removed_cnt += 1
                continue  # Don't process further

            # Rewire consumers to use the input directly
            consumers = output_to_consumers.get(squeeze_output, [])
            for consumer in consumers:
                for i, input_name in enumerate(consumer.input):
                    if input_name == squeeze_output:
                        consumer.input[i] = squeeze_input
            nodes_to_remove.append(node)
            squeeze_unsqueeze_removed_cnt += 1
            continue  # Don't process further

        # --- Replace eligible Unsqueeze nodes with Reshape ---
        if (
            node.op_type == "Unsqueeze"
            and node.input[0] not in input_names
            and node.output[0] not in output_names
        ):
            input_shape = get_shape(node.input[0])
            if input_shape is None:
                continue  # Can't process if shape is unknown

            # Get axes from initializer
            axes = None
            for init in graph.initializer:
                if init.name == node.input[1]:
                    axes = numpy_helper.to_array(init)
                    break
            if axes is None:
                for n in graph.node:
                    if n.op_type == "Constant" and n.output[0] == node.input[1]:
                        for attr in n.attribute:
                            if attr.name == "value":
                                axes = numpy_helper.to_array(attr.t)
                                break
                        break
            if axes is None:
                continue  # Can't process if axes unknown
            # Compute unsqueezed shape
            out_shape = list(input_shape)
            for axis in sorted(axes):
                if axis >= 0:
                    out_shape.insert(axis, 1)
                else:
                    out_shape.insert(len(out_shape) + axis + 1, 1)
            # If input is 2D and output is 3D, prepend a 1
            if len(input_shape) == 2 and len(out_shape) == 3:
                out_shape = [1, *out_shape]

            # Only replace if:
            # - input is 2D and output is 4D
            # - input is 3D and output is 4D
            # - input is 2D and output is 3D (with extra 1 at front)
            valid_transforms = [
                (len(input_shape) == 2 and len(out_shape) == 4),
                (len(input_shape) == 3 and len(out_shape) == 4),
                (len(input_shape) == 2 and len(out_shape) == 3),
            ]
            if any(valid_transforms):
                # Create shape initializer for Reshape
                reshape_shape_name = node.name + "_reshape_shape"
                reshape_shape = numpy_helper.from_array(
                    np.array(out_shape, dtype=np.int64), reshape_shape_name
                )
                add_unique_initializers(graph, unique_initializer, [reshape_shape])
                # Create Reshape node
                reshape_node = helper.make_node(
                    "Reshape",
                    inputs=[node.input[0], reshape_shape_name],
                    outputs=node.output,
                    name=node.name + "_to_reshape",
                )
                nodes_to_add.append(reshape_node)
                nodes_to_remove.append(node)
                unsqueeze_replaced_cnt += 1
                # Rewire consumers
                unsqueeze_output = node.output[0]
                consumers = output_to_consumers.get(unsqueeze_output, [])
                for consumer in consumers:
                    for i, input_name in enumerate(consumer.input):
                        if input_name == unsqueeze_output:
                            consumer.input[i] = reshape_node.output[0]
            else:
                error_msg = (
                    f"[transform_remove_intermediary_squeeze_and_unsqueeze] valid_transforms is not True "
                    f"for node: {node.name}, input_shape: {input_shape}, out_shape: {out_shape}"
                )
                raise ValueError(error_msg)

    nodes_to_add_ind = 0
    for node in nodes_to_remove:
        if (
            nodes_to_add_ind < len(nodes_to_add)
            and node.name in nodes_to_add[nodes_to_add_ind].name
        ):
            insert_nodes_at_position(graph, [nodes_to_add[nodes_to_add_ind]], node)
            nodes_to_add_ind += 1
        graph.node.remove(node)

    logger.debug(
        "Removed %d intermediary Squeeze ops and replaced %d Unsqueeze ops with Reshape as per rules.",
        squeeze_unsqueeze_removed_cnt,
        unsqueeze_replaced_cnt,
    )
    model = infer_shapes(model)
    return model


def dla_remove_intermediary_squeeze_and_unsqueeze(
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
        _apply_remove_intermediary_squeeze_and_unsqueeze,
        use_external_data=use_external_data,
        external_data_name=external_data_name,
        verbose=verbose,
        **kwargs,
    )
