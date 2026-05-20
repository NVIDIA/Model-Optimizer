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

"""Common graph manipulation utilities for ONNX models."""

import os
from collections import defaultdict, deque

import numpy as np
import onnx
import torch
from onnx import TensorProto, helper, numpy_helper

from ...logging_config import logger
from .dtype_conversion import fp16_to_bf16_array


def uses_external_data(model_path: str) -> bool:
    """Check if an ONNX model uses external data files.

    Args:
        model_path: Path to the ONNX model file.

    Returns:
        True if external data files exist for the model.
    """
    model_dir = os.path.dirname(model_path) or "."
    model_name = os.path.basename(model_path)

    # Common external data naming patterns
    external_patterns = [
        model_name + "_data",
        model_name + ".data",
        model_name.replace(".onnx", ".onnx_data"),
        model_name.replace(".onnx", "_data"),
    ]

    return any(os.path.exists(os.path.join(model_dir, pattern)) for pattern in external_patterns)


def topological_sort_nodes(graph: onnx.GraphProto) -> list[onnx.NodeProto]:
    """Topologically sort nodes in the graph for clean ordering.

    Uses Kahn's algorithm to produce a valid topological ordering
    respecting data dependencies between nodes.

    Args:
        graph: ONNX graph to sort.

    Returns:
        List of nodes in topological order.
    """
    # Build dependency graph: output_name -> node that produces it
    output_to_node: dict[str, onnx.NodeProto] = {}
    for node in graph.node:
        for out in node.output:
            output_to_node[out] = node

    # Get all initializer and input names (available from the start)
    available: set[str] = set()
    for init in graph.initializer:
        available.add(init.name)
    for inp in graph.input:
        available.add(inp.name)

    # Build adjacency list and in-degree count
    node_to_idx: dict[str, int] = {node.name: i for i, node in enumerate(graph.node)}
    in_degree: dict[int, int] = dict.fromkeys(range(len(graph.node)), 0)
    adj: dict[int, list[int]] = defaultdict(list)

    for i, node in enumerate(graph.node):
        for inp in node.input:
            if inp and inp not in available:
                # This input comes from another node
                if inp in output_to_node:
                    producer = output_to_node[inp]
                    producer_idx = node_to_idx.get(producer.name)
                    if producer_idx is not None and producer_idx != i:
                        adj[producer_idx].append(i)
                        in_degree[i] += 1

    # Kahn's algorithm for topological sort
    queue: deque[int] = deque()
    for i in range(len(graph.node)):
        if in_degree[i] == 0:
            queue.append(i)

    sorted_indices: list[int] = []
    while queue:
        idx = queue.popleft()
        sorted_indices.append(idx)
        for neighbor in adj[idx]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    # If we couldn't sort all nodes, there might be a cycle or disconnected nodes
    # Just append any remaining nodes at the end
    if len(sorted_indices) < len(graph.node):
        remaining = set(range(len(graph.node))) - set(sorted_indices)
        sorted_indices.extend(remaining)

    return [graph.node[i] for i in sorted_indices]


def detect_model_dtype(model: onnx.ModelProto) -> tuple[int, np.dtype]:
    """Detect the primary floating-point dtype of the model.

    Analyzes initializers to find the most common floating-point dtype.

    Args:
        model: ONNX model to analyze.

    Returns:
        Tuple of (onnx_dtype, numpy_dtype).
    """
    dtype_counts: dict[int, int] = {}
    for init in model.graph.initializer:
        if init.data_type in [TensorProto.FLOAT, TensorProto.FLOAT16, TensorProto.BFLOAT16]:
            dtype_counts[init.data_type] = dtype_counts.get(init.data_type, 0) + 1

    if not dtype_counts:
        return TensorProto.FLOAT, np.float32

    dominant_dtype = max(dtype_counts, key=lambda k: dtype_counts[k])

    dtype_map = {
        TensorProto.FLOAT: np.float32,
        TensorProto.FLOAT16: np.float16,
        TensorProto.BFLOAT16: np.float32,  # numpy doesn't support bfloat16
    }

    return dominant_dtype, dtype_map.get(dominant_dtype, np.float32)


def get_onnx_dtype(io_dtype: str) -> int:
    """Convert string dtype to ONNX TensorProto dtype.

    Args:
        io_dtype: String representation of dtype ("float32", "float16", or "bfloat16").

    Returns:
        ONNX TensorProto data type constant.
    """
    dtype_map = {
        "float32": TensorProto.FLOAT,
        "float16": TensorProto.FLOAT16,
        "bfloat16": TensorProto.BFLOAT16,
    }
    return dtype_map.get(io_dtype, TensorProto.FLOAT16)


def find_nodes_by_pattern(graph: onnx.GraphProto, pattern: str) -> list[onnx.NodeProto]:
    """Find nodes whose name contains the pattern.

    Args:
        graph: ONNX graph to search.
        pattern: Substring pattern to match in node names.

    Returns:
        List of nodes matching the pattern.
    """
    return [n for n in graph.node if pattern in n.name]


def find_node_by_output(graph: onnx.GraphProto, output_name: str) -> onnx.NodeProto | None:
    """Find node that produces the given output.

    Args:
        graph: ONNX graph to search.
        output_name: Name of the output tensor.

    Returns:
        Node that produces the output, or None if not found.
    """
    for node in graph.node:
        if output_name in node.output:
            return node
    return None


def find_node_by_name(graph: onnx.GraphProto, name: str) -> onnx.NodeProto | None:
    """Find node by exact name.

    Args:
        graph: ONNX graph to search.
        name: Exact node name to find.

    Returns:
        Node with the given name, or None if not found.
    """
    for node in graph.node:
        if node.name == name:
            return node
    return None


def find_initializer(graph: onnx.GraphProto, name: str) -> onnx.TensorProto | None:
    """Find initializer by name.

    Args:
        graph: ONNX graph to search.
        name: Name of the initializer.

    Returns:
        Initializer tensor, or None if not found.
    """
    for init in graph.initializer:
        if init.name == name:
            return init
    return None


def get_consumers(graph: onnx.GraphProto, tensor_name: str) -> list[onnx.NodeProto]:
    """Find all nodes that consume the given tensor.

    Args:
        graph: ONNX graph to search.
        tensor_name: Name of the tensor.

    Returns:
        List of nodes that use the tensor as input.
    """
    return [node for node in graph.node if tensor_name in node.input]


def remove_node(graph: onnx.GraphProto, node: onnx.NodeProto) -> None:
    """Remove a node from the graph.

    Args:
        graph: ONNX graph to modify.
        node: Node to remove.
    """
    graph.node.remove(node)


def get_all_tensors_used(graph: onnx.GraphProto) -> set[str]:
    """Get all tensor names that are actually used in the graph.

    Args:
        graph: ONNX graph to analyze.

    Returns:
        Set of tensor names that are used in the graph.
    """
    used_tensors = set()

    # Tensors produced by nodes
    for node in graph.node:
        for inp in node.input:
            if inp:  # Skip empty strings
                used_tensors.add(inp)
        for out in node.output:
            if out:
                used_tensors.add(out)

    # Tensors that are graph outputs
    for out in graph.output:
        used_tensors.add(out.name)

    return used_tensors


def cleanup_unused_ios(graph: onnx.GraphProto) -> dict[str, int]:
    """Remove unused inputs, outputs, initializers, value_info, and orphaned nodes.

    This function iteratively removes:
    1. Orphaned nodes (nodes whose outputs are not consumed)
    2. Unused graph inputs
    3. Unused graph outputs
    4. Unused initializers
    5. Unused value_info entries

    Args:
        graph: ONNX graph to clean up.

    Returns:
        Dictionary with counts of removed items.
    """
    # First pass: Remove orphaned nodes
    total_nodes_removed = 0

    while True:
        # Get all tensor names consumed by nodes
        consumed_by_nodes = set()
        for node in graph.node:
            for inp in node.input:
                if inp:
                    consumed_by_nodes.add(inp)

        # Also include graph outputs as "consumed"
        for out in graph.output:
            consumed_by_nodes.add(out.name)

        # Find nodes whose outputs are not consumed by anyone
        nodes_to_remove = []
        for node in graph.node:
            # Skip if node has no outputs (side-effect nodes)
            if not node.output:
                continue

            # Check if ANY of the node's outputs are consumed
            any_output_consumed = False
            for out in node.output:
                if out and out in consumed_by_nodes:
                    any_output_consumed = True
                    break

            # If none of the outputs are consumed, mark for removal
            if not any_output_consumed:
                nodes_to_remove.append(node)

        if not nodes_to_remove:
            break

        for node in nodes_to_remove:
            graph.node.remove(node)

        total_nodes_removed += len(nodes_to_remove)

    # Rebuild consumed_by_nodes after node cleanup
    consumed_by_nodes = set()
    for node in graph.node:
        for inp in node.input:
            if inp:
                consumed_by_nodes.add(inp)

    for out in graph.output:
        consumed_by_nodes.add(out.name)

    # Get all tensors used
    used_tensors = get_all_tensors_used(graph)

    # Get all tensor names produced by nodes
    produced_by_nodes = set()
    for node in graph.node:
        for out in node.output:
            if out:
                produced_by_nodes.add(out)

    # Clean up inputs
    inputs_to_remove = [inp for inp in graph.input if inp.name not in consumed_by_nodes]

    for inp in inputs_to_remove:
        graph.input.remove(inp)

    # Clean up outputs
    outputs_to_remove = []
    for out in graph.output:
        input_names = {i.name for i in graph.input}
        init_names = {i.name for i in graph.initializer}
        if (
            out.name not in produced_by_nodes
            and out.name not in input_names
            and out.name not in init_names
        ):
            outputs_to_remove.append(out)

    for out in outputs_to_remove:
        graph.output.remove(out)

    # Clean up initializers
    initializers_to_remove = [
        init for init in graph.initializer if init.name not in consumed_by_nodes
    ]

    for init in initializers_to_remove:
        graph.initializer.remove(init)

    # Clean up value_info
    value_info_to_remove = [vi for vi in graph.value_info if vi.name not in used_tensors]

    for vi in value_info_to_remove:
        graph.value_info.remove(vi)

    return {
        "nodes_removed": total_nodes_removed,
        "inputs_removed": len(inputs_to_remove),
        "outputs_removed": len(outputs_to_remove),
        "initializers_removed": len(initializers_to_remove),
        "value_info_removed": len(value_info_to_remove),
    }


def initializer_to_array(init: onnx.TensorProto) -> tuple[np.ndarray, str | None]:
    """Convert ONNX initializer to numpy array, handling bfloat16.

    Args:
        init: ONNX initializer tensor.

    Returns:
        Tuple of (numpy array, dtype string if bfloat16 else None).
    """
    if init.data_type == TensorProto.BFLOAT16:
        # bfloat16 stored as raw bytes - read as int16 for manipulation
        arr = np.frombuffer(init.raw_data, dtype=np.int16).reshape(init.dims)
        return arr, "bfloat16"
    else:
        return numpy_helper.to_array(init), None


def array_to_initializer(arr: np.ndarray, name: str, is_bfloat16: bool = False) -> onnx.TensorProto:
    """Convert numpy array to ONNX initializer, handling bfloat16.

    Args:
        arr: Numpy array to convert.
        name: Name for the initializer.
        is_bfloat16: Whether to store as bfloat16.

    Returns:
        ONNX TensorProto initializer.
    """
    if is_bfloat16:
        tensor = onnx.TensorProto()
        tensor.name = name
        tensor.data_type = TensorProto.BFLOAT16
        tensor.dims.extend(arr.shape)
        tensor.raw_data = arr.tobytes()
        return tensor
    else:
        return numpy_helper.from_array(arr, name=name)


def add_initializer(
    graph: onnx.GraphProto,
    name: str,
    data: np.ndarray,
    dtype: int = TensorProto.FLOAT16,
) -> None:
    """Add an initializer (constant tensor) to the graph.

    Args:
        graph: ONNX graph to modify.
        name: Name for the initializer.
        data: Numpy array data.
        dtype: ONNX data type for the tensor.
    """
    if dtype == TensorProto.BFLOAT16:
        # For bfloat16, data comes as int16 view - create raw tensor
        tensor = onnx.TensorProto()
        tensor.name = name
        tensor.data_type = TensorProto.BFLOAT16
        tensor.dims.extend(data.shape)
        tensor.raw_data = data.tobytes()
    else:
        tensor = numpy_helper.from_array(data, name=name)
    graph.initializer.append(tensor)
    # Also add to value_info for shape inference
    value_info = helper.make_tensor_value_info(name, dtype, list(data.shape))
    graph.value_info.append(value_info)


def convert_initializers_to_dtype(graph: onnx.GraphProto, target_dtype_str: str = "float16") -> int:
    """Convert all float32 initializers to the target dtype.

    This ensures weight matrices match the model's I/O precision.

    Args:
        graph: ONNX graph to modify.
        target_dtype_str: Target dtype ("float16", "float32", or "bfloat16").

    Returns:
        Count of converted initializers.
    """
    if target_dtype_str == "float32":
        return 0

    converted_count = 0

    # Iterate over a copy since we're modifying the list
    initializers_to_convert = [
        init for init in graph.initializer if init.data_type == TensorProto.FLOAT
    ]

    for init in initializers_to_convert:
        arr = numpy_helper.to_array(init)

        if target_dtype_str == "float16":
            arr_converted = arr.astype(np.float16)
            new_init = numpy_helper.from_array(arr_converted, name=init.name)
        elif target_dtype_str == "bfloat16":
            tensor = torch.from_numpy(arr).to(torch.bfloat16)
            new_init = onnx.TensorProto()
            new_init.name = init.name
            new_init.data_type = TensorProto.BFLOAT16
            new_init.dims.extend(arr.shape)
            new_init.raw_data = tensor.view(torch.int16).numpy().tobytes()
        else:
            logger.warning(f"Unknown dtype {target_dtype_str}, skipping {init.name}")
            continue

        graph.initializer.remove(init)
        graph.initializer.append(new_init)
        converted_count += 1

    return converted_count


def convert_model_fp16_to_bf16(graph: onnx.GraphProto, verbose: bool = True) -> dict[str, int]:
    """Convert all FP16 elements in the graph to BF16.

    This converts:
    1. FP16 initializers (weights) to BF16
    2. FP16 Constant nodes to BF16
    3. Cast nodes targeting FP16 to target BF16
    4. FP16 value_info to BF16
    5. FP16 graph inputs to BF16
    6. FP16 graph outputs to BF16

    Args:
        graph: ONNX graph to modify in-place.
        verbose: Whether to print progress messages.

    Returns:
        Dictionary with conversion statistics.
    """
    stats = {
        "initializers_converted": 0,
        "constants_converted": 0,
        "casts_converted": 0,
        "value_info_converted": 0,
        "inputs_converted": 0,
        "outputs_converted": 0,
    }

    # 1. Convert FP16 initializers to BF16
    new_initializers = []
    for init in graph.initializer:
        if init.data_type == TensorProto.FLOAT16:
            fp16_array = numpy_helper.to_array(init)
            bf16_uint16 = fp16_to_bf16_array(fp16_array)

            new_init = onnx.TensorProto()
            new_init.name = init.name
            new_init.data_type = TensorProto.BFLOAT16
            new_init.dims.extend(init.dims)
            new_init.raw_data = bf16_uint16.tobytes()

            new_initializers.append(new_init)
            stats["initializers_converted"] += 1
        else:
            new_initializers.append(init)

    # Clear and replace initializers
    while len(graph.initializer) > 0:
        graph.initializer.pop()
    graph.initializer.extend(new_initializers)

    # 2. Convert Constant nodes and Cast nodes
    for node in graph.node:
        # Convert Constant nodes with FP16 values
        if node.op_type == "Constant":
            for attr in node.attribute:
                if attr.name == "value" and attr.t.data_type == TensorProto.FLOAT16:
                    fp16_array = numpy_helper.to_array(attr.t)
                    bf16_uint16 = fp16_to_bf16_array(fp16_array)

                    attr.t.data_type = TensorProto.BFLOAT16
                    attr.t.ClearField("raw_data")
                    attr.t.ClearField("float_data")
                    attr.t.ClearField("int32_data")
                    attr.t.raw_data = bf16_uint16.tobytes()

                    stats["constants_converted"] += 1

        # Convert Cast nodes targeting FP16 to target BF16
        if node.op_type == "Cast":
            for attr in node.attribute:
                if attr.name == "to" and attr.i == TensorProto.FLOAT16:
                    attr.i = TensorProto.BFLOAT16
                    stats["casts_converted"] += 1

    # 3. Convert value_info
    for vi in graph.value_info:
        if vi.type.HasField("tensor_type"):
            if vi.type.tensor_type.elem_type == TensorProto.FLOAT16:
                vi.type.tensor_type.elem_type = TensorProto.BFLOAT16
                stats["value_info_converted"] += 1

    # 4. Convert graph inputs
    for inp in graph.input:
        if inp.type.HasField("tensor_type"):
            if inp.type.tensor_type.elem_type == TensorProto.FLOAT16:
                inp.type.tensor_type.elem_type = TensorProto.BFLOAT16
                stats["inputs_converted"] += 1

    # 5. Convert graph outputs
    for out in graph.output:
        if out.type.HasField("tensor_type"):
            if out.type.tensor_type.elem_type == TensorProto.FLOAT16:
                out.type.tensor_type.elem_type = TensorProto.BFLOAT16
                stats["outputs_converted"] += 1

    if verbose:
        logger.info("FP16 to BF16 conversion statistics:")
        logger.info(f"  Initializers: {stats['initializers_converted']}")
        logger.info(f"  Constants: {stats['constants_converted']}")
        logger.info(f"  Cast nodes: {stats['casts_converted']}")
        logger.info(f"  Value_info: {stats['value_info_converted']}")
        logger.info(f"  Inputs: {stats['inputs_converted']}")
        logger.info(f"  Outputs: {stats['outputs_converted']}")

    return stats
