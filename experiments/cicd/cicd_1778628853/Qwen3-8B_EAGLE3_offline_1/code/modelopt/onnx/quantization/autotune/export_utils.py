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

"""Utilities for Q/DQ model export and insertion in ONNX autotune."""

import dataclasses

import numpy as np
import onnx
import onnx_graphsurgeon as gs

from modelopt.onnx.logging_config import logger
from modelopt.onnx.quantization.autotune.common import Config
from modelopt.onnx.quantization.autotune.insertion_points import (
    ResolvedInsertionPoint,
    merge_resolved_insertion_points,
)
from modelopt.onnx.quantization.fp8 import int8_to_fp8
from modelopt.onnx.quantization.graph_utils import get_tensor_consumer_node_indices

__all__ = [
    "build_tensor_map",
    "create_qdq_nodes",
    "export_qdq_onnx",
    "fix_zero_point_initializers",
    "get_tensor_metadata",
    "get_zero_point_for_quant_type",
    "insert_qdq_at_tensors",
    "resolve_dtype",
]

_DTYPE_MAP = {
    "int8": np.int8,
    "uint8": np.uint8,
    "float16": np.float16,
    "float32": np.float32,
}


def resolve_dtype(
    dtype_str: str, default: np.dtype = np.int8, dtype_map: dict | None = None
) -> np.dtype:
    """Resolve a dtype string (quant or DQ output) to a numpy dtype."""
    if dtype_map is None:
        dtype_map = _DTYPE_MAP
    if dtype_str == "fp8":
        try:
            return np.dtype(np.float8_e4m3fn)
        except (AttributeError, TypeError):
            logger.warning(
                "FP8 dtype not available (requires numpy >= 2.0), "
                "using uint8 as placeholder. Note: This may not produce "
                "correct results without proper FP8 support."
            )
            return np.uint8
    if hasattr(np, "bfloat16") and dtype_str == "bfloat16":
        return np.bfloat16
    if dtype_str in dtype_map:
        return dtype_map[dtype_str]
    logger.warning(f"Unknown dtype '{dtype_str}', using default {default}")
    return default


def get_zero_point_for_quant_type(quant_type: str, quant_dtype: np.dtype) -> int:
    """Return default zero point for quant type and validate it is in-range for the dtype.

    int8 uses 0 (signed); uint8 uses 128 (unsigned midpoint); fp8/other use 0.
    Raises ValueError if the default zero point is not in the valid range for quant_dtype.
    """
    default_zp = 128 if quant_type == "uint8" else 0
    if quant_dtype == np.int8:
        low, high = -128, 127
        if not (low <= default_zp <= high):
            raise ValueError(
                f"Zero point {default_zp} out of range for int8 (must be in [{low}, {high}])"
            )
    elif quant_dtype == np.uint8:
        low, high = 0, 255
        if not (low <= default_zp <= high):
            raise ValueError(
                f"Zero point {default_zp} out of range for uint8 (must be in [{low}, {high}])"
            )
    return default_zp


def build_tensor_map(graph: gs.Graph) -> dict[str, gs.Tensor]:
    """Build mapping from tensor names to tensor objects."""
    tensor_map = {t.name: t for t in graph.inputs if hasattr(t, "name") and t.name}
    for node in graph.nodes:
        for t in node.inputs:
            if hasattr(t, "name") and t.name:
                tensor_map[t.name] = t
        for t in node.outputs:
            if isinstance(t, gs.Constant) and hasattr(t, "name") and t.name:
                tensor_map[t.name] = t
    return tensor_map


def get_tensor_metadata(
    tensor: gs.Tensor, is_constant: bool, default_dtype: np.dtype
) -> tuple[tuple | None, np.dtype]:
    """Extract shape and dtype metadata from a tensor."""
    if is_constant and hasattr(tensor, "values") and tensor.values is not None:
        return tensor.values.shape, tensor.values.dtype
    if hasattr(tensor, "shape"):
        dtype = (
            tensor.dtype if hasattr(tensor, "dtype") and tensor.dtype is not None else default_dtype
        )
        return tensor.shape, dtype
    return None, default_dtype


def fix_zero_point_initializers(model: onnx.ModelProto) -> None:
    """Fix INT8 zero_point initializers to use int32_data instead of raw_data."""
    fixed_count = 0
    for initializer in model.graph.initializer:
        if (
            "_zp_" in initializer.name
            and initializer.data_type == onnx.TensorProto.INT8
            and len(initializer.raw_data) > 0
            and len(initializer.int32_data) == 0
        ):
            np_array = onnx.numpy_helper.to_array(initializer)
            int32_values = np_array.astype(np.int32).flatten().tolist()
            new_tensor = onnx.helper.make_tensor(
                initializer.name,
                onnx.TensorProto.INT8,
                list(initializer.dims),
                int32_values,
            )
            initializer.CopyFrom(new_tensor)
            fixed_count += 1
    if fixed_count > 0:
        logger.debug(f"Fixed {fixed_count} zero_point initializers (int32_data format)")


def create_qdq_nodes(
    tensor_name: str,
    qdq_input: gs.Tensor,
    output_shape: tuple | None,
    output_dtype: np.dtype,
    quant_dtype: np.dtype,
    q_scale: float,
    q_zero_point: int,
) -> tuple[gs.Node, gs.Node]:
    """Create QuantizeLinear and DequantizeLinear node pair."""
    q_name = f"QDQ_Q_{tensor_name}".replace("/", "_").replace(":", "_")
    dq_name = f"QDQ_DQ_{tensor_name}".replace("/", "_").replace(":", "_")
    dtype_map = {"float16": np.float16, "float32": np.float32}
    if hasattr(np, "bfloat16"):
        dtype_map["bfloat16"] = np.bfloat16
    scale_dtype = dtype_map.get(np.dtype(output_dtype).name, np.float32)

    logger.debug(
        f"Creating Q/DQ pair for '{tensor_name}' (scale_dtype={np.dtype(scale_dtype).name})"
    )

    q_scale_values = np.array([q_scale], dtype=scale_dtype)
    q_zp_values = np.array([q_zero_point], dtype=quant_dtype)
    q_inputs = [
        qdq_input,
        gs.Constant(f"q_scale_{tensor_name}", values=q_scale_values),
        gs.Constant(f"q_zp_{tensor_name}", values=q_zp_values),
    ]
    q_node = gs.Node(
        op="QuantizeLinear",
        name=q_name,
        inputs=q_inputs,
        outputs=[gs.Variable(f"{tensor_name}_quantized", dtype=quant_dtype, shape=output_shape)],
    )

    dq_scale_values = np.array([q_scale], dtype=scale_dtype)
    dq_zp_values = np.array([q_zero_point], dtype=quant_dtype)
    dq_inputs = [
        q_node.outputs[0],
        gs.Constant(f"dq_scale_{tensor_name}", values=dq_scale_values),
        gs.Constant(f"dq_zp_{tensor_name}", values=dq_zp_values),
    ]
    dq_node = gs.Node(
        op="DequantizeLinear",
        name=dq_name,
        inputs=dq_inputs,
        outputs=[gs.Variable(f"{tensor_name}_dequantized", dtype=output_dtype, shape=output_shape)],
    )
    return q_node, dq_node


def insert_qdq_at_tensors(
    graph: gs.Graph,
    resolved_insertion_points: set[ResolvedInsertionPoint],
    config: Config,
    *,
    tensor_users_map: dict[str, list[int]] | None = None,
) -> None:
    """Insert Q/DQ (Quantize/Dequantize) node pairs at specified locations.

    Modifies the graph in-place. Builds tensor map and tensor-to-users map,
    processes each resolved insertion point, and runs graph cleanup/toposort.

    Args:
        graph: Graph to modify in-place.
        resolved_insertion_points: Set of ResolvedInsertionPoint specifying where to insert Q/DQ.
        config: Config with default_q_scale, default_q_zero_point, default_quant_type, default_dq_dtype.
        tensor_users_map: Optional precomputed tensor name -> list of node indices. If None, computed.
    """
    q_scale = config.default_q_scale
    q_zero_point = config.default_q_zero_point
    quant_type = config.default_quant_type
    quant_dtype = resolve_dtype(quant_type, np.int8, _DTYPE_MAP)

    logger.debug(f"Q/DQ parameters: type={quant_type}, scale={q_scale}, zero_point={q_zero_point}")

    resolved_insertion_points = merge_resolved_insertion_points(graph, resolved_insertion_points)

    tensor_map = build_tensor_map(graph)
    if tensor_users_map is None:
        tensor_users_map = get_tensor_consumer_node_indices(graph)
    logger.debug(
        f"Built tensor maps: {len(tensor_map)} tensors, {len(tensor_users_map)} with users"
    )

    default_dq_dtype = resolve_dtype(config.default_dq_dtype, np.float32, _DTYPE_MAP)

    for insertion_point in resolved_insertion_points:
        tensor_name = insertion_point.tensor_name
        node_index = insertion_point.node_index
        input_index = insertion_point.input_index

        original_tensor = tensor_map[tensor_name]
        if node_index is not None:
            if node_index < 0 or node_index >= len(graph.nodes):
                raise IndexError(
                    f"Node index out of range: {node_index} (graph has {len(graph.nodes)} nodes)"
                )
            target_node = graph.nodes[node_index]
            if input_index is None:
                raise ValueError("Input index must be set when node index is set")
            if input_index < 0 or input_index >= len(target_node.inputs):
                raise IndexError(
                    f"Input index out of range for node {target_node.name}: "
                    f"{input_index} (node has {len(target_node.inputs)} inputs)"
                )
            original_tensor = target_node.inputs[input_index]
            if tensor_name != original_tensor.name:
                raise ValueError(
                    f"Tensor name mismatch for node {target_node.name} input {input_index}: "
                    f"expected {tensor_name!r}, got {original_tensor.name!r}"
                )
        else:
            if tensor_name not in tensor_map:
                raise KeyError(f"Tensor {tensor_name!r} not found in tensor map")
            if input_index is not None:
                raise ValueError("Input index must be None when node index is None")

        is_constant = isinstance(original_tensor, gs.Constant)
        output_shape, output_dtype = get_tensor_metadata(
            original_tensor, is_constant, default_dtype=default_dq_dtype
        )

        unique_suffix = "qdq"
        if node_index is not None:
            unique_suffix = f"n{node_index}_i{input_index}"
        unique_tensor_name = f"{tensor_name}_{unique_suffix}"

        q_node, dq_node = create_qdq_nodes(
            unique_tensor_name,
            original_tensor,
            output_shape,
            output_dtype,
            quant_dtype,
            q_scale,
            q_zero_point,
        )

        graph.nodes.extend([q_node, dq_node])

        if node_index is not None:
            target_node.inputs[input_index] = dq_node.outputs[0]
            logger.debug(
                f"  Q/DQ inserted: tensor '{tensor_name}' → node #{node_index} "
                f"({target_node.name}) input #{input_index}"
            )
        else:
            users = tensor_users_map[tensor_name]
            for user_index in users:
                user_node = graph.nodes[user_index]
                for i, input_tensor in enumerate(user_node.inputs):
                    if hasattr(input_tensor, "name") and input_tensor.name == tensor_name:
                        user_node.inputs[i] = dq_node.outputs[0]
                        break
            logger.debug(f"  Q/DQ inserted: tensor '{tensor_name}' → {len(users)} users")

    logger.debug("Running graph cleanup and topological sort")
    try:
        graph.cleanup().toposort()
        logger.debug("Graph cleanup completed")
    except (ValueError, RuntimeError) as exc:
        logger.error("Graph cleanup failed: %s", exc)
        raise RuntimeError(f"Graph cleanup failed after Q/DQ insertion: {exc}") from exc


def export_qdq_onnx(
    source: onnx.ModelProto | gs.Graph,
    resolved_insertion_points: set[ResolvedInsertionPoint],
    config: Config,
    *,
    insert_qdq: bool = True,
    needs_fp8_conversion: bool = False,
) -> onnx.ModelProto:
    """Export ONNX model with optional Q/DQ insertion and optional INT8→FP8 conversion.

    Does not modify the source; works on a copy of the graph.

    Args:
        source: ONNX model or GraphSurgeon graph to export from.
        resolved_insertion_points: Set of insertion points (used when insert_qdq is True).
        config: Config for Q/DQ parameters and dtypes.
        insert_qdq: If True, insert Q/DQ at resolved points before exporting.
        needs_fp8_conversion: If True, build as INT8 then convert to FP8 (e.g. when config.default_quant_type is fp8).

    Returns:
        Exported ONNX ModelProto (with Q/DQ and/or FP8 as requested).
    """
    if isinstance(source, onnx.ModelProto):
        graph_copy = gs.import_onnx(source)
    else:
        graph_copy = gs.import_onnx(gs.export_onnx(source))
    graph_copy.toposort()

    if insert_qdq and resolved_insertion_points:
        if needs_fp8_conversion:
            logger.debug("FP8 conversion: creating INT8 model first")
            config_int8 = dataclasses.replace(config, default_quant_type="int8")
            insert_qdq_at_tensors(graph_copy, resolved_insertion_points, config_int8)
        else:
            insert_qdq_at_tensors(graph_copy, resolved_insertion_points, config)

    logger.debug("Serializing to ONNX format")
    model = gs.export_onnx(graph_copy)

    if insert_qdq and resolved_insertion_points:
        fix_zero_point_initializers(model)

    if needs_fp8_conversion:
        logger.debug("Converting INT8 to FP8")
        model = int8_to_fp8(model)

    return model
