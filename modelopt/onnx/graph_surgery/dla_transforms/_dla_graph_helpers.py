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

"""Shared graph helpers for DLA transforms (split from legacy monolith)."""

from collections import defaultdict

import numpy as np
from onnx import TensorProto, helper, numpy_helper

from ...logging_config import logger


def insert_nodes_at_position(graph, nodes_to_add, reference_node):
    """Insert new nodes at the position of a reference node to maintain topological order.

    Args:
        graph: ONNX graph
        nodes_to_add: List of nodes to insert
        reference_node: Node whose position will be used for insertion

    """
    if not nodes_to_add:
        return

    # Find the position of the reference node
    node_list = list(graph.node)
    try:
        insert_position = node_list.index(reference_node)
    except ValueError:
        # Reference node not found, append at the end
        insert_position = len(node_list)

    # Insert all new nodes at the reference position
    for i, new_node in enumerate(nodes_to_add):
        graph.node.insert(insert_position + i, new_node)


def replace_nodes_with_topological_order(graph, nodes_to_remove, nodes_to_add):
    """Replace nodes while maintaining topological order.

    Args:
        graph: ONNX graph
        nodes_to_remove: List of nodes to remove
        nodes_to_add: List of nodes to add at the position of the first removed node

    """
    if not nodes_to_remove:
        if nodes_to_add:
            # Just add nodes at the end if no nodes to remove
            graph.node.extend(nodes_to_add)
        return

    # Find the position of the first node to remove
    first_removed_node = nodes_to_remove[0]
    node_list = list(graph.node)
    try:
        insert_position = node_list.index(first_removed_node)
    except ValueError:
        insert_position = len(node_list)

    # Remove all nodes to be removed
    for node in nodes_to_remove:
        if node in graph.node:
            graph.node.remove(node)

    # Insert new nodes at the position where the first removed node was
    for i, new_node in enumerate(nodes_to_add):
        graph.node.insert(insert_position + i, new_node)


def get_tensor_shape_by_name(graph, name):
    tensor_type = None
    for initializer in graph.initializer:
        if initializer.name == name:
            return numpy_helper.to_array(initializer).shape
    for value_info in graph.value_info:
        if value_info.name == name:
            tensor_type = value_info.type.tensor_type
            break
    for graph_input in graph.input:
        if graph_input.name == name:
            tensor_type = graph_input.type.tensor_type
            break
    if tensor_type is not None:
        shape = []
        if tensor_type.HasField("shape"):
            for d in tensor_type.shape.dim:
                if d.HasField("dim_value"):
                    shape.append(d.dim_value)
                else:
                    shape.append(0)
                    break
            return shape
    return None


def get_tensor_shape_map(graph_value_info):
    tensor_name_dim_map = {}
    for value_info in graph_value_info:
        tensor_type = value_info.type.tensor_type
        shape = []
        if tensor_type.HasField("shape"):
            for d in tensor_type.shape.dim:
                # the dimension may have a definite (integer) value or a symbolic identifier or neither:
                if d.HasField("dim_value"):
                    shape.append(d.dim_value)
                else:
                    shape.append(0)
                    break
        tensor_name_dim_map[value_info.name] = shape
    return tensor_name_dim_map


def get_tensor_dtype_by_name(model, name):
    for init in model.graph.initializer:
        if init.name == name:
            return init.data_type
    for vi in model.graph.value_info:
        if vi.name == name:
            return vi.type.tensor_type.elem_type
    for node in model.graph.node:
        if node.op_type == "Constant" and node.output[0] == name:
            return numpy_helper.to_array(node.attribute[0].t).dtype
    for graph_input in model.graph.input:
        if graph_input.name == name:
            return graph_input.type.tensor_type.elem_type
    for output in model.graph.output:
        if output.name == name:
            return output.type.tensor_type.elem_type
    return None


def get_initializer_by_name(model, init_name):
    for init in model.graph.initializer:
        if init.name == init_name:
            return numpy_helper.to_array(init)
    return None


def get_initializer_object_by_name(model, init_name):
    try:
        for init in model.graph.initializer:
            if init.name == init_name:
                return init
        return None
    except Exception:
        for init in model.initializer:
            if init.name == init_name:
                return init
        return None


def get_constant_by_name(model, const_name):
    for node in model.graph.node:
        if node.op_type == "Constant" and node.output[0] == const_name:
            value = None
            for attr in node.attribute:
                if attr.name == "value":
                    value = attr.t
                    break
            return numpy_helper.to_array(value)
    return None


def calculate_clip_range(node, model):
    x_scale = get_initializer_by_name(model, node.input[1])
    if x_scale is None:
        x_scale = get_constant_by_name(model, node.input[1])
    x_zero_point = get_initializer_by_name(model, node.input[2])
    if x_scale is None:
        raise ValueError(f"{node.name} should have x_scale value")
    int_max = np.int32(
        65535 if x_zero_point.dtype == np.uint16 else 255 if x_zero_point.dtype == np.uint8 else 127
    )
    int_min = np.int32(
        0 if x_zero_point.dtype == np.uint16 else 0 if x_zero_point.dtype == np.uint8 else -128
    )
    if x_zero_point is None:
        logger.info("x_zero_point is None!")
        x_zero_point = np.array(0, dtype=np.int32)
    else:
        x_zero_point = x_zero_point.astype(np.int32)
    clip_min = ((int_min - x_zero_point) * x_scale).astype(np.float32)
    clip_max = ((int_max - x_zero_point) * x_scale).astype(np.float32)
    return clip_min, clip_max


def add_unique_initializers(graph, unique_set, initializers_list):
    """Add multiple initializers to the graph only if their names are not already in the unique set.

    Args:
        graph: ONNX graph to add the initializers to
        unique_set: Set of initializer names already added (modified in-place)
        initializers_list: List of ONNX initializers to add

    Returns:
        int: Number of initializers actually added

    """
    added_count = 0
    for init in initializers_list:
        if init.name not in unique_set:
            graph.initializer.append(init)
            unique_set.add(init.name)
            added_count += 1
    return added_count


def get_producer_node_by_output(model, output):
    for node in model.graph.node:
        if output in node.output:
            return node
    return None


def get_consumer_nodes_by_input(model, input_):
    node_input_idx_pair = []
    for node in model.graph.node:
        for i, inp in enumerate(node.input):
            if inp == input_:
                node_input_idx_pair.append((node, i))
                break
    return node_input_idx_pair


def get_tensor_value_info_by_name(model, tensor_name):
    for vi in list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info):
        if vi.name == tensor_name:
            return vi
    return None


def set_value_info_shape(value_info, new_shape):
    tensor_type = value_info.type.tensor_type
    tensor_type.shape.dim.clear()
    for d in new_shape:
        dim = tensor_type.shape.dim.add()
        if d is None:
            dim.dim_param = ""
        else:
            dim.dim_value = int(d)


def update_tensor_shape_by_name(model, name, new_shape):
    for vi in list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info):
        if vi.name == name:
            set_value_info_shape(vi, new_shape)
            return True
    return False


def get_op_types_by_input(model, tensor_name):
    op_types = set()
    for node in model.graph.node:
        if tensor_name in node.input:
            op_types.add(node.op_type)
    return op_types


def is_initializer(model, name):
    return name in {init.name for init in model.graph.initializer}


def make_one_scalar_initializer(model, name):
    if not any(init.name == name for init in model.graph.initializer):
        one_initializer = helper.make_tensor(name, TensorProto.FLOAT, [], [1.0])
        model.graph.initializer.append(one_initializer)


def is_const_dq_input(init_list, input_name, graph):
    for node in graph.node:
        if (
            node.op_type == "DequantizeLinear"
            and node.output[0] == input_name
            and node.input[0] in init_list
        ):
            return node.input[0]
    return None


def is_graph_input_dq_input(input_name, graph, input_list):
    if input_name in input_list:
        return input_name
    for node in graph.node:
        if (
            node.op_type == "DequantizeLinear"
            and node.output[0] == input_name
            and node.input[0] in input_list
        ):
            return node.input[0]
    return None


def check_to_apply_transpose(conv_node, model, tensor_name_dim_map):
    initializer_names = {init.name for init in model.graph.initializer}
    # input_list = {input_.name for input_ in model.graph.input}
    bool_value = False
    if (
        conv_node.input[1] in initializer_names
        or is_const_dq_input(initializer_names, conv_node.input[1], model.graph)
    ):  #### if second input is initializer with suitable conditions, we need to trans-conv-transpose
        init_name = is_const_dq_input(initializer_names, conv_node.input[1], model.graph)
        if not init_name:
            init_name = conv_node.input[1]
        for init in model.graph.initializer:
            if init.name == init_name:
                shape = list(init.dims)
                bool_value = (
                    len(shape) == 2
                    or (len(shape) == 3 and shape[0] == 1)
                    or (len(shape) == 4 and shape[0] == 1 and shape[1] == 1)
                )
                break
    # else:
    # # if second input is graph input, no need to convert to conv.
    # input_ = conv_node.input[1]
    # if is_graph_input_dq_input(input_, model.graph, input_list) is not None:
    #     return False
    # if input_ in tensor_name_dim_map:
    #     shape = tensor_name_dim_map[input_]
    #     if len(shape) == 2:
    #         # if shape[1] != 1:  # 2D tensor added leading dims to 1x1xMxN
    #         bool_value = True
    #     if len(shape) == 3 and shape[0] == 1:  # C == 1
    #         bool_value = True
    #     if len(shape) == 4 and shape[0] == 1 and shape[1] == 1:  # N == 1, C == 1
    #         bool_value = True
    if not (bool_value):
        return bool_value
    bool_value = False
    input_ = conv_node.input[0]
    if input_ in tensor_name_dim_map:
        shape = tensor_name_dim_map[input_]
        if len(shape) in (2, 3) or (len(shape) == 4 and shape[0] == 1):
            bool_value = True
        else:
            raise ValueError(
                f"[check_to_apply_transpose] Unsupported shape {shape} for input {input_} in conv node"
            )
    else:
        raise RuntimeError(
            f"[check_to_apply_transpose] Input {input_} not found in tensor_name_dim_map for conv node"
        )
    return bool_value


def get_shape_from_graph(graph, name):
    # Check value_info
    for value_info in graph.value_info:
        if value_info.name == name:
            tensor_type = value_info.type.tensor_type
            return [d.dim_value if d.HasField("dim_value") else 0 for d in tensor_type.shape.dim]
    # Check graph inputs
    for graph_input in graph.input:
        if graph_input.name == name:
            tensor_type = graph_input.type.tensor_type
            return [d.dim_value if d.HasField("dim_value") else 0 for d in tensor_type.shape.dim]
    # Check initializers
    for init in graph.initializer:
        if init.name == name:
            return list(init.dims)

    for graph_output in graph.output:
        if graph_output.name == name:
            tensor_type = graph_output.type.tensor_type
            return [d.dim_value if d.HasField("dim_value") else 0 for d in tensor_type.shape.dim]
    return None


def count_ops(model):
    ops = defaultdict(int)
    for node in model.graph.node:
        ops[node.op_type] += 1
        if node.op_type not in ops:
            ops[node.op_type] = 0
    logger.info("=== Graph ops count ===")
    for op_name, cnt in sorted(ops.items()):
        logger.info("%d %s", cnt, op_name)


def get_input_shape_from_graph_inputs(graph, input_name):
    for graph_input in graph.input:
        if graph_input.name == input_name:
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


def add_value_info(graph, tensor_name, dtype=TensorProto.FLOAT, shape=None):
    vi = helper.make_tensor_value_info(tensor_name, dtype, shape)
    graph.value_info.append(vi)
