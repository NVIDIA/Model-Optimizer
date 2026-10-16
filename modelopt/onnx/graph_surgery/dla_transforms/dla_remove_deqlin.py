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
from onnx import TensorProto, numpy_helper

from ...logging_config import logger
from ._common import GraphCache, add_unique_initializers, get_node_attr_i, run_onnx_file_transform
from ._dla_graph_helpers import check_to_apply_transpose

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
    x = x_arr.astype(np.int32)
    zp = zp_arr.astype(np.int32)
    scale = scale_arr.astype(np.float32)

    if x.ndim > 0:
        # Reshape scale / zp for per-channel broadcasting along ``axis``.
        broadcast_shape = [1] * x.ndim
        broadcast_shape[axis] = -1
        zp = zp.reshape(broadcast_shape)
        scale = scale.reshape(broadcast_shape)

    return ((x - zp) * scale).astype(np.float32)


def _should_skip(
    dq_output: str,
    zp_dtype: int,
    consumer_map: dict,
    model: onnx.ModelProto,
    shape_map: dict,
) -> bool:
    """Return True if the DQ node must be kept (downstream op handles quantised weight)."""
    consumers = consumer_map.get(dq_output, [])
    if len(consumers) != 1:
        return False  # multiple consumers → always dequantize

    consumer_node, input_idx = consumers[0]
    is_weight = input_idx == 1
    op = consumer_node.op_type

    if op == "Conv" and is_weight and zp_dtype in _CONV_SKIP_ZP:
        return True
    if op == "ConvTranspose" and is_weight and zp_dtype in _CONVT_SKIP_ZP:
        return True
    if op in ("MatMul", "Gemm") and is_weight and zp_dtype in _MATMUL_SKIP_ZP:
        try:
            return check_to_apply_transpose(consumer_node, model, shape_map)
        except Exception:
            return False
    return False


# ── Main transform ─────────────────────────────────────────────────────────────


def _apply_remove_deqlin(model: onnx.ModelProto) -> onnx.ModelProto:
    """Replace eligible DequantizeLinear nodes with pre-computed float32 initializers."""
    graph = model.graph
    cache = GraphCache(graph)

    # Snapshot for check_to_apply_transpose (expects a plain dict)
    shape_map_snapshot: dict = {}
    for vi in list(graph.value_info) + list(graph.input) + list(graph.output):
        s = cache.get_shape(vi.name)
        if s is not None:
            shape_map_snapshot[vi.name] = s

    consumer_map = cache.consumer_map
    initializer_names = cache.init_names
    unique_init: set[str] = set()
    nodes_to_remove: list = []
    cnt = 0

    for node in cache.nodes_by_op("DequantizeLinear"):
        if not node.input or node.input[0] not in initializer_names:
            continue  # data input is not a static initializer — skip

        dq_output = node.output[0]

        # Fetch scale and zero-point initializers (inputs 1 and 2)
        scale_init = cache.get_init(node.input[1]) if len(node.input) > 1 else None
        zp_init = cache.get_init(node.input[2]) if len(node.input) > 2 else None

        if scale_init is None or zp_init is None:
            logger.debug(
                "DequantizeLinear %r: scale or zero_point not found as initializer; skipping.",
                node.name,
            )
            continue

        zp_dtype: int = zp_init.data_type

        if _should_skip(dq_output, zp_dtype, consumer_map, model, shape_map_snapshot):
            logger.debug(
                "DequantizeLinear %r: skip (downstream op keeps quantised weight).", node.name
            )
            continue

        # ── Dequantize ────────────────────────────────────────────────────────
        x_init = cache.get_init(node.input[0])
        x_arr = numpy_helper.to_array(x_init)
        scale_arr = numpy_helper.to_array(scale_init)
        zp_arr = numpy_helper.to_array(zp_init)

        axis = get_node_attr_i(node, "axis", 0)

        dequantized = _dequantize(x_arr, scale_arr, zp_arr, axis)
        new_name = node.input[0] + "_dequantized"
        new_init = numpy_helper.from_array(dequantized, name=new_name)

        # ── Rewire all consumers ──────────────────────────────────────────────
        for consumer_node, _ in consumer_map.get(dq_output, []):
            for i, inp in enumerate(consumer_node.input):
                if inp == dq_output:
                    consumer_node.input[i] = new_name
            # no break — update every occurrence of dq_output in this node's inputs

        add_unique_initializers(graph, unique_init, [new_init])

        nodes_to_remove.append(node)
        cnt += 1
        logger.debug("DequantizeLinear %r → float32 initializer %r.", node.name, new_name)

    for node in nodes_to_remove:
        graph.node.remove(node)

    logger.debug(
        "remove_deqlin: folded %d DequantizeLinear node(s) into float32 initializers.", cnt
    )
    return model


def dla_remove_deqlin(
    model_path: str,
    output_path: str,
    *,
    use_external_data: bool = True,
    external_data_name: str | None = None,
    verbose: bool = True,
    **kwargs,
) -> onnx.ModelProto:
    """Load ONNX from ``model_path``, fold eligible DQ nodes, save to ``output_path``."""
    return run_onnx_file_transform(
        model_path,
        output_path,
        _apply_remove_deqlin,
        use_external_data=use_external_data,
        external_data_name=external_data_name,
        verbose=verbose,
        **kwargs,
    )
