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

import gc
import math
import os
from collections import defaultdict, deque
from collections.abc import Iterable

import numpy as np
import onnx
import torch
from onnx import TensorProto, numpy_helper

from ...logging_config import logger
from .dtype_conversion import fp16_to_bf16_array

__all__ = [
    "DEFAULT_QDQ_REMOVE_QUANT_TYPES",
    "GraphCache",
    "add_initializer",
    "add_unique_initializers",
    "add_value_info",
    "array_to_initializer",
    "axes_for_rank",
    "batch_replace_nodes",
    "calculate_clip_range",
    "check_to_apply_transpose",
    "cleanup_unused_ios",
    "convert_initializers_to_dtype",
    "convert_model_fp16_to_bf16",
    "detect_model_dtype",
    "find_initializer",
    "find_node_by_name",
    "find_node_by_output",
    "find_nodes_by_pattern",
    "get_all_tensors_used",
    "get_constant_by_name",
    "get_consumers",
    "get_node_attr_i",
    "get_onnx_dtype",
    "infer_shapes",
    "initializer_to_array",
    "insert_nodes_at_position",
    "is_const_dq_input",
    "pad4d",
    "parse_qdq_quantized_dtype_list",
    "remove_node",
    "resolve_reshape_shape",
    "set_node_attr_i",
    "topological_sort_nodes",
    "uses_external_data",
]


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


def get_onnx_dtype(io_dtype: int | str, *, strict: bool = False) -> int:
    """Resolve ONNX codes, names, and NumPy aliases; default unknown names to FP16 unless strict."""
    if isinstance(io_dtype, int):
        return io_dtype
    key = str(io_dtype).strip()
    members = dict(TensorProto.DataType.items())
    if key.isdecimal() and int(key) in members.values():
        return int(key)
    aliases = {
        "float32": "FLOAT",
        "float64": "DOUBLE",
        "fp32": "FLOAT",
        "fp16": "FLOAT16",
        "str": "STRING",
        **{
            f"{prefix}{bits}": f"{name}{bits}"
            for prefix, name in (("u", "UINT"), ("i", "INT"))
            for bits in (8, 16, 32, 64)
        },
    }
    dtype = members.get(aliases.get(key.lower(), key.upper()))
    if dtype is not None:
        return dtype
    if strict:
        raise ValueError(
            f"Unknown dtype {io_dtype!r}. Use an ONNX type name, NumPy name, or integer code."
        )
    return TensorProto.FLOAT16


def parse_qdq_quantized_dtype_list(dtypes: Iterable[int | str]) -> frozenset[int]:
    """Normalize the storage types selected for QDQ removal."""
    return frozenset(get_onnx_dtype(dtype, strict=True) for dtype in dtypes)


DEFAULT_QDQ_REMOVE_QUANT_TYPES = frozenset({TensorProto.UINT16, TensorProto.INT16})


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
    *,
    with_value_info: bool = True,
) -> str:
    """Add or reuse an identical initializer; reject name conflicts before changing the graph."""
    is_bf16 = dtype == TensorProto.BFLOAT16
    if not is_bf16:
        data = np.asarray(data, dtype=onnx.helper.tensor_dtype_to_np_dtype(dtype))
    tensor = array_to_initializer(data, name, is_bfloat16=is_bf16)
    existing = find_initializer(graph, name)
    if existing is not None:
        if (
            existing.data_type != dtype
            or list(existing.dims) != list(tensor.dims)
            or numpy_helper.to_array(existing).tobytes() != numpy_helper.to_array(tensor).tobytes()
        ):
            raise ValueError(f"Conflicting generated initializer {name!r}")
        return name
    graph.initializer.append(tensor)
    if with_value_info:
        add_value_info(graph, name, dtype, list(data.shape))
    return name


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


def infer_shapes(model: onnx.ModelProto) -> onnx.ModelProto:
    """Run ORT symbolic shape inference and update ``value_info`` in-place."""
    try:
        # Symbolic inference is heavy; load it only when a transform requests inference.
        from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference

        inferred = SymbolicShapeInference.infer_shapes(model, auto_merge=True)
        del model.graph.value_info[:]
        model.graph.value_info.extend(inferred.graph.value_info)
        inferred_out = {o.name: o.type for o in inferred.graph.output}
        for o in model.graph.output:
            if o.name in inferred_out:
                o.type.CopyFrom(inferred_out[o.name])
        del inferred, inferred_out
        gc.collect()
        return model
    except Exception as exc:
        raise RuntimeError(f"ORT symbolic shape inference failed: {exc}") from exc


def resolve_reshape_shape(reshape_shape: np.ndarray, input_shape: list[int]) -> list[int]:
    """Resolve a Reshape target shape containing ``0`` and ``-1`` into concrete dims."""
    shape = [int(d) for d in reshape_shape.tolist()]
    if any(d <= 0 for d in input_shape) or any(d < -1 for d in shape):
        raise ValueError("Reshape resolution requires positive input dimensions and targets >= -1")
    total_elems = math.prod(input_shape)
    minus_one_indices = [i for (i, d) in enumerate(shape) if d == -1]
    if len(minus_one_indices) > 1:
        raise ValueError(f"resolve_reshape_shape: more than one -1 in shape {shape}")
    resolved = []
    for i, d in enumerate(shape):
        if d == 0:
            if i >= len(input_shape):
                raise ValueError(
                    f"resolve_reshape_shape: 0 at index {i} but input rank is {len(input_shape)}"
                )
            resolved.append(input_shape[i])
        elif d == -1:
            resolved.append(-1)
        else:
            resolved.append(d)
    if minus_one_indices:
        known_prod = 1
        for d in resolved:
            if d != -1:
                known_prod *= d
        if total_elems % known_prod != 0:
            raise ValueError(
                f"resolve_reshape_shape: cannot infer -1 dim: total={total_elems}, known_prod={known_prod}"
            )
        resolved[minus_one_indices[0]] = total_elems // known_prod
    if int(np.prod(resolved)) != total_elems:
        raise ValueError(
            f"resolve_reshape_shape: element count changes from {input_shape} to {resolved}"
        )
    return resolved


def pad4d(shape: list) -> list:
    """Pad *shape* to 4-D by prepending 1-s."""
    return [1] * (4 - len(shape)) + list(shape)


class GraphCache:
    """Pre-built index of an ONNX graph for O(1) lookups."""

    __slots__ = (
        "_arr_cache",
        "_graph",
        "_names",
        "consumer_map",
        "graph_input_names",
        "graph_output_names",
        "init_map",
        "init_names",
        "op_index",
        "producer_map",
        "vi_map",
    )

    def __init__(self, graph: onnx.GraphProto) -> None:
        """Index the current graph; rebuild after structural or initializer changes."""
        self._graph = graph
        self._arr_cache: dict[str, np.ndarray] = {}
        self.rebuild()

    def rebuild(self) -> None:
        """(Re)build every index from scratch.  Call after bulk graph mutations."""
        graph = self._graph
        self._arr_cache.clear()
        self._names = {
            v.name for v in [*graph.input, *graph.output, *graph.initializer, *graph.value_info]
        }
        for node in graph.node:
            self._names.update([node.name, *node.input, *node.output])
        self.init_map: dict[str, onnx.TensorProto] = {init.name: init for init in graph.initializer}
        self.init_names: set[str] = set(self.init_map)
        self.producer_map: dict[str, onnx.NodeProto] = {}
        for node in graph.node:
            for out in node.output:
                if out:
                    self.producer_map[out] = node
        self.consumer_map: dict[str, list[tuple]] = {}
        for node in graph.node:
            for i, inp in enumerate(node.input):
                if inp:
                    self.consumer_map.setdefault(inp, []).append((node, i))
        self.vi_map: dict[str, onnx.ValueInfoProto] = {}
        for vi in list(graph.value_info) + list(graph.input) + list(graph.output):
            self.vi_map[vi.name] = vi
        self.op_index: dict[str, list[onnx.NodeProto]] = {}
        for node in graph.node:
            self.op_index.setdefault(node.op_type, []).append(node)
        self.graph_input_names: set[str] = {inp.name for inp in graph.input}
        self.graph_output_names: set[str] = {out.name for out in graph.output}

    def get_consumers(self, tensor_name: str) -> list[tuple]:
        """Return ``[(consumer_node, input_index), ...]`` for *tensor_name*."""
        return self.consumer_map.get(tensor_name, [])

    def get_producer(self, tensor_name: str) -> onnx.NodeProto | None:
        """Return the node that produces *tensor_name*, or ``None``."""
        return self.producer_map.get(tensor_name)

    def get_shape(self, tensor_name: str) -> list[int] | None:
        """Return the concrete shape of *tensor_name*, or ``None``."""
        vi = self.vi_map.get(tensor_name)
        if vi is not None:
            tt = vi.type.tensor_type
            if tt.HasField("shape"):
                shape: list[int] = []
                for d in tt.shape.dim:
                    if d.HasField("dim_value"):
                        shape.append(d.dim_value)
                    else:
                        return None
                return shape
        init = self.init_map.get(tensor_name)
        if init is not None:
            return list(init.dims)
        return None

    def unique_name(self, preferred: str) -> str:
        """Reserve a generated name, retaining the reference name when available."""
        name = preferred
        suffix = 1
        while name in self._names:
            name = f"{preferred}_{suffix}"
            suffix += 1
        self._names.add(name)
        return name

    def get_init(self, name: str) -> onnx.TensorProto | None:
        """Return the initializer ``TensorProto`` for *name*, or ``None``."""
        return self.init_map.get(name)

    def is_init(self, name: str) -> bool:
        """Return whether *name* is an initializer."""
        return name in self.init_names

    def nodes_by_op(self, op_type: str) -> list[onnx.NodeProto]:
        """Return all nodes with the given *op_type*."""
        return self.op_index.get(op_type, [])

    def get_dtype(self, tensor_name: str) -> int | None:
        """Return the ONNX elem_type int for *tensor_name*, or ``None``."""
        vi = self.vi_map.get(tensor_name)
        if vi is not None:
            return vi.type.tensor_type.elem_type
        init = self.init_map.get(tensor_name)
        if init is not None:
            return init.data_type
        return None

    def update_shape(self, name: str, new_shape: list) -> bool:
        """Update the shape of tensor *name* in-place via ``vi_map``.  O(1) lookup."""
        vi = self.vi_map.get(name)
        if vi is None:
            return False
        tt = vi.type.tensor_type
        tt.shape.dim.clear()
        for d in new_shape:
            dim = tt.shape.dim.add()
            if d is None:
                dim.dim_param = ""
            else:
                dim.dim_value = int(d)
        return True

    def get_init_array(self, name: str) -> np.ndarray | None:
        """Return deserialized numpy array for initializer *name*, cached."""
        if name in self._arr_cache:
            return self._arr_cache[name]
        init = self.init_map.get(name)
        if init is None:
            return None
        arr = numpy_helper.to_array(init)
        self._arr_cache[name] = arr
        return arr


def insert_nodes_at_position(graph, nodes_to_add, reference_node):
    """Insert new nodes at the position of a reference node to maintain topological order."""
    if not nodes_to_add:
        return
    insert_position = next(
        (i for (i, node) in enumerate(graph.node) if node is reference_node), len(graph.node)
    )
    for i, new_node in enumerate(nodes_to_add):
        graph.node.insert(insert_position + i, new_node)


def batch_replace_nodes(graph: onnx.GraphProto, replacements: dict[int, tuple[list, list]]) -> None:
    """Batch node removal + insertion, processed in reverse index order."""
    for idx in sorted(replacements.keys(), reverse=True):
        (nodes_to_remove, nodes_to_add) = replacements[idx]
        for node in reversed(nodes_to_remove):
            position = next((i for (i, current) in enumerate(graph.node) if current is node), None)
            if position is not None:
                del graph.node[position]
        for i, new_node in enumerate(nodes_to_add):
            graph.node.insert(idx + i, new_node)


def get_constant_by_name(model: onnx.ModelProto, const_name: str) -> np.ndarray | None:
    """Read a Constant node's tensor-valued attribute by its output name."""
    for node in model.graph.node:
        if node.op_type == "Constant" and node.output[0] == const_name:
            value = None
            for attr in node.attribute:
                if attr.name == "value":
                    value = attr.t
                    break
            if value is not None:
                return numpy_helper.to_array(value)
    return None


def get_node_attr_i(node: onnx.NodeProto, name: str, default: int = 0) -> int:
    """Return integer attribute *name* from *node*, or *default* if absent."""
    attr = _attribute(node, name)
    return int(attr.i) if attr is not None else default


def set_node_attr_i(node: onnx.NodeProto, name: str, value: int) -> None:
    """Set integer attribute *name* on *node*, creating it if absent."""
    attr = _attribute(node, name)
    if attr is None:
        node.attribute.append(onnx.helper.make_attribute(name, int(value)))
    else:
        attr.i = int(value)


def axes_for_rank(rank: int) -> list[int]:
    """Leading axes to prepend when unsqueezing rank-R to 4D."""
    return {1: [0, 1, 2], 2: [0, 1], 3: [0], 4: []}.get(rank, [])


def calculate_clip_range(node, model: onnx.ModelProto) -> tuple[np.ndarray, np.ndarray]:
    """Compute floating-point clip bounds from quantization scale and zero point."""
    scale = find_initializer(model.graph, node.input[1])
    x_scale = None if scale is None else numpy_helper.to_array(scale)
    if x_scale is None:
        x_scale = get_constant_by_name(model, node.input[1])
    zero = find_initializer(model.graph, node.input[2]) if len(node.input) > 2 else None
    x_zero_point = None if zero is None else numpy_helper.to_array(zero)
    if x_scale is None:
        raise ValueError(f"{node.name} should have x_scale value")
    if x_zero_point is None:
        dtype = (
            GraphCache(model.graph).get_dtype(node.input[0])
            if node.op_type == "DequantizeLinear"
            else onnx.TensorProto.UINT8
        )
        x_zero_point = np.array(
            0, dtype=onnx.helper.tensor_dtype_to_np_dtype(dtype or onnx.TensorProto.UINT8)
        )
    bounds = np.iinfo(x_zero_point.dtype)
    (int_min, int_max) = (np.int64(bounds.min), np.int64(bounds.max))
    x_zero_point = x_zero_point.astype(np.int64)
    clip_min = ((int_min - x_zero_point) * x_scale).astype(np.float32)
    clip_max = ((int_max - x_zero_point) * x_scale).astype(np.float32)
    return (clip_min, clip_max)


def add_value_info(
    graph: onnx.GraphProto,
    name: str,
    dtype: int = onnx.TensorProto.FLOAT,
    shape: list | tuple | None = None,
) -> None:
    """Append a ``value_info`` entry for tensor *name* with *dtype* and *shape*."""
    graph.value_info.append(onnx.helper.make_tensor_value_info(name, dtype, shape))


def add_unique_initializers(graph, unique_set: set[str], initializers_list) -> int:
    """Add initializers whose names are not already in unique_set; update unique_set in place."""
    added_count = 0
    for init in initializers_list:
        if init.name not in unique_set:
            graph.initializer.append(init)
            unique_set.add(init.name)
            added_count += 1
    return added_count


def _attribute(node, name):
    """Return a node attribute by name, or None when it is absent."""
    return next((attribute for attribute in node.attribute if attribute.name == name), None)


def _shape_before_boundary_promotion(cache, name):
    """Resolve the original shape behind a generated graph-input Unsqueeze."""
    producer = cache.get_producer(name)
    if producer is not None and producer.op_type == "Unsqueeze":
        original = producer.input[0]
        if original in cache.graph_input_names and name == original + "_unsqueeze_input":
            return cache.get_shape(original)
    return cache.get_shape(name)


def is_const_dq_input(init_list, input_name, graph):
    """Find the constant weight behind a DequantizeLinear output."""
    for node in graph.node:
        if (
            node.op_type == "DequantizeLinear"
            and node.output[0] == input_name
            and (node.input[0] in init_list)
        ):
            return node.input[0]
    return None


def check_to_apply_transpose(conv_node, model, tensor_name_dim_map):
    """Check static weight and activation layouts supported by MatMul-to-Conv."""
    weights = {init.name: init for init in model.graph.initializer}
    init_name = is_const_dq_input(weights, conv_node.input[1], model.graph) or conv_node.input[1]
    weight = weights.get(init_name)
    if weight is None:
        return False
    shape = list(weight.dims)
    if not (
        len(shape) == 2
        or (len(shape) == 3 and shape[0] == 1)
        or (len(shape) == 4 and shape[:2] == [1, 1])
    ):
        return False
    input_ = conv_node.input[0]
    if input_ in tensor_name_dim_map:
        shape = tensor_name_dim_map[input_]
        if len(shape) in (2, 3) or (len(shape) == 4 and shape[0] == 1):
            return True
        else:
            raise ValueError(
                f"[check_to_apply_transpose] Unsupported shape {shape} for input {input_} in conv node"
            )
    else:
        raise RuntimeError(
            f"[check_to_apply_transpose] Input {input_} not found in tensor_name_dim_map for conv node"
        )


def _captured_names(graph):
    """Return names read by a graph but defined outside its scope."""
    bound = {v.name for v in [*graph.input, *graph.initializer]}
    bound.update(name for node in graph.node for name in node.output if name)
    used = {v.name for v in graph.output}
    for node in graph.node:
        used.update(_node_inputs(node))
    return used - bound


def _node_inputs(node):
    """Return node inputs and outer-scope names captured by its subgraphs."""
    used = {name for name in node.input if name}
    for attribute in node.attribute:
        if attribute.type == onnx.AttributeProto.GRAPH:
            used.update(_captured_names(attribute.g))
        elif attribute.type == onnx.AttributeProto.GRAPHS:
            for graph in attribute.graphs:
                used.update(_captured_names(graph))
    return used
