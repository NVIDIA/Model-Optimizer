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

"""Fold ``DequantizeLinear`` nodes whose data input is an initializer into a float32 initializer.

Skip conditions — DQ node is left in place when its output feeds:

* ``Conv`` weight (``input[1]``) with INT8 / UINT8 / INT4 / UINT4 zero-point.
* ``ConvTranspose`` weight with INT8 / UINT8 zero-point.
* ``MatMul`` / ``Gemm`` weight that will be converted to a ``ConvTranspose`` chain by
  ``dla_matmul_to_transpose_conv_transpose`` (detected via ``check_to_apply_transpose``).
  Pre-fusing the float32 weight would corrupt the per-channel quantization axis that the
  conversion adjusts.
"""

from __future__ import annotations

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from ...logging_config import logger
from ..utils.graph_utils import (
    GraphCache,
    add_unique_initializers,
    check_to_apply_transpose,
    get_node_attr_i,
)
from .dla_graph_cleanup import _subgraph_captures

# ── Zero-point types that trigger the skip rule ──────────────────────────────
_CONV_SKIP_ZP = frozenset(
    {TensorProto.INT8, TensorProto.UINT8, TensorProto.INT4, TensorProto.UINT4}
)
_CONVT_SKIP_ZP = frozenset({TensorProto.INT8, TensorProto.UINT8})
_MATMUL_SKIP_ZP = _CONV_SKIP_ZP


# ── Helpers ───────────────────────────────────────────────────────────────────


def _dequantize(
    x_arr: np.ndarray,
    scale_arr: np.ndarray,
    zp_arr: np.ndarray,
    axis: int,
) -> np.ndarray:
    """Compute ``(x - zero_point) * scale`` with per-channel broadcasting.

    For scalar (rank-0) ``x`` the broadcast reshape is skipped.
    """
    int32_data = x_arr.dtype == np.int32
    zp = zp_arr.astype(np.int32 if int32_data else np.float32)
    scale = scale_arr.astype(np.float32)
    if zp.size == 1:
        zp = zp.reshape(())
    if scale.size == 1:
        scale = scale.reshape(())
    if x_arr.ndim > 0 and (zp.size > 1 or scale.size > 1):
        broadcast_shape = [1] * x_arr.ndim
        broadcast_shape[axis] = -1
        zp = zp.reshape(broadcast_shape)
        scale = scale.reshape(broadcast_shape)
    if int32_data:
        # Keep established INT32 rounding without a whole-weight float64 temporary.
        result = np.empty(x_arr.shape, dtype=np.float32)
        with np.nditer(
            [x_arr, zp, scale, result],
            flags=["external_loop", "buffered", "zerosize_ok"],
            op_flags=[["readonly"], ["readonly"], ["readonly"], ["writeonly"]],
            buffersize=2**18,
        ) as chunks:
            for values, zeros, scales, output in chunks:
                output[...] = (values - zeros) * scales
        return result
    result = x_arr.astype(np.float32)
    np.subtract(result, zp, out=result)
    np.multiply(result, scale, out=result)
    return result


def _should_skip(
    dq_output: str,
    zp_dtype: int,
    consumer_map: dict,
    model: onnx.ModelProto,
    shape_map: dict,
) -> bool:
    """Retain weights required or potentially required by any downstream consumer."""
    for consumer_node, input_idx in consumer_map.get(dq_output, []):
        if input_idx != 1 or consumer_node.domain not in {"", "ai.onnx"} or consumer_node.overload:
            continue
        op = consumer_node.op_type
        if op == "Conv" and zp_dtype in _CONV_SKIP_ZP:
            return True
        if op == "ConvTranspose" and zp_dtype in _CONVT_SKIP_ZP:
            return True
        if op in ("MatMul", "Gemm") and zp_dtype in _MATMUL_SKIP_ZP:
            try:
                if check_to_apply_transpose(consumer_node, model, shape_map):
                    return True
            except (ValueError, RuntimeError):
                return True
    return False


# ── Main transform ─────────────────────────────────────────────────────────────


def _apply_remove_deqlin(model: onnx.ModelProto) -> onnx.ModelProto:
    """Replace eligible DequantizeLinear nodes with pre-computed float32 initializers."""
    graph = model.graph
    cache = GraphCache(graph)
    captured = _subgraph_captures(graph)
    local_functions = {
        (function.domain, function.name, function.overload) for function in model.functions
    }
    opsets = {entry.domain: entry.version for entry in model.opset_import}

    # Snapshot for check_to_apply_transpose (expects a plain dict)
    shape_map_snapshot: dict = {}
    for vi in list(graph.value_info) + list(graph.input) + list(graph.output):
        s = cache.get_shape(vi.name)
        if s is not None:
            shape_map_snapshot[vi.name] = s

    consumer_map = cache.consumer_map
    initializer_names = cache.init_names
    unique_init: set[str] = set(cache.init_names)
    nodes_to_remove: list = []
    cnt = 0

    for node in cache.nodes_by_op("DequantizeLinear"):
        if node.overload or (node.domain, node.op_type, node.overload) in local_functions:
            continue
        if node.domain not in {"", "ai.onnx"} and not (
            node.domain == "com.microsoft" and opsets.get(node.domain) == 1
        ):
            continue
        if any(name in cache.graph_input_names for name in node.input):
            continue
        if get_node_attr_i(node, "block_size", 0):
            continue
        if not node.input or node.input[0] not in initializer_names:
            continue  # data input is not a static initializer — skip

        dq_output = node.output[0]
        if dq_output in captured:
            continue

        # Fetch scale and zero-point initializers (inputs 1 and 2)
        scale_init = cache.get_init(node.input[1]) if len(node.input) > 1 else None
        zp_init = cache.get_init(node.input[2]) if len(node.input) > 2 else None

        if len(node.input) > 2 and node.input[2] and zp_init is None:
            continue
        x_init = cache.get_init(node.input[0])
        if x_init is None:
            continue
        if (
            x_init.data_type
            not in _CONV_SKIP_ZP | {TensorProto.INT16, TensorProto.UINT16, TensorProto.INT32}
            or (scale_init is not None and scale_init.data_type != TensorProto.FLOAT)
            or get_node_attr_i(node, "output_dtype", 0) not in {0, TensorProto.FLOAT}
        ):
            continue

        if scale_init is None:
            logger.debug(
                "DequantizeLinear %r: scale not found as initializer; skipping.",
                node.name,
            )
            continue

        if zp_init is None:
            logger.debug(
                "DequantizeLinear %r: zero_point not found as initializer; using 0 as zero_point.",
                node.name,
            )

        zp_dtype: int = zp_init.data_type if zp_init is not None else x_init.data_type

        if _should_skip(dq_output, zp_dtype, consumer_map, model, shape_map_snapshot):
            logger.debug(
                "DequantizeLinear %r: skip (downstream op keeps quantised weight).", node.name
            )
            continue

        # ── Dequantize ────────────────────────────────────────────────────────
        x_arr = numpy_helper.to_array(x_init)
        scale_arr = numpy_helper.to_array(scale_init)
        zp_arr = (
            numpy_helper.to_array(zp_init)
            if zp_init is not None
            else np.zeros(scale_arr.shape, dtype=np.int32)
        )

        axis = get_node_attr_i(node, "axis", 1)

        dequantized = _dequantize(x_arr, scale_arr, zp_arr, axis)
        del x_arr, scale_arr, zp_arr
        new_name = node.input[0] + "_dequantized"
        previous = cache.get_init(new_name)
        if previous is None:
            new_name = cache.unique_name(new_name)
        if previous is not None and (
            new_name in cache.graph_input_names
            or list(previous.dims) != list(dequantized.shape)
            or previous.data_type != TensorProto.FLOAT
            or numpy_helper.to_array(previous).tobytes() != dequantized.tobytes()
        ):
            new_name = cache.unique_name(new_name)
        new_init = (
            numpy_helper.from_array(dequantized, name=new_name)
            if new_name not in unique_init
            else None
        )
        del dequantized

        # ── Rewire all consumers ──────────────────────────────────────────────
        for consumer_node, _ in consumer_map.get(dq_output, []):
            for i, inp in enumerate(consumer_node.input):
                if inp == dq_output:
                    consumer_node.input[i] = new_name
            # no break — update every occurrence of dq_output in this node's inputs

        if new_init is not None:
            add_unique_initializers(graph, unique_init, [new_init])
            cache.init_map[new_name] = graph.initializer[-1]
        del new_init
        if dq_output in cache.graph_output_names:
            position = next(index for index, current in enumerate(graph.node) if current is node)
            graph.node.insert(
                position,
                helper.make_node(
                    "Identity",
                    [new_name],
                    [dq_output],
                    name=cache.unique_name(node.name + "_output"),
                ),
            )

        nodes_to_remove.append(node)
        cnt += 1
        logger.debug("DequantizeLinear %r → float32 initializer %r.", node.name, new_name)

    for node in nodes_to_remove:
        graph.node.remove(node)

    logger.debug(
        "remove_deqlin: folded %d DequantizeLinear node(s) into float32 initializers.", cnt
    )
    return model
