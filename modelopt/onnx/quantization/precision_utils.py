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

"""Shared runtime-precision conversion for ONNX quantization."""

import numpy as np
import onnx
import onnx_graphsurgeon as gs

from modelopt.onnx.autocast.convert import convert_to_f16
from modelopt.onnx.logging_config import logger
from modelopt.onnx.quantization.graph_utils import (
    convert_fp16_io,
    insert_fp8_mha_casts,
    remove_output_initializers,
)

__all__: list[str] = []


def _upgrade_opset_21(onnx_model: onnx.ModelProto) -> onnx.ModelProto:
    """Update an FP8 graph to the opset required for FP16 Q/DQ scales."""
    logger.info("Upgrading model to opset 21")
    graph = gs.import_onnx(onnx_model)

    for node in graph.nodes:
        if node.op in {"QuantizeLinear", "DequantizeLinear"}:
            node.domain = ""

        if node.op == "ReduceMean" and "axes" in node.attrs:
            axes = gs.Constant(
                name=node.name + "_axes", values=np.array(node.attrs["axes"], dtype=np.int64)
            )
            del node.attrs["axes"]
            node.inputs.append(axes)

    onnx_model = gs.export_onnx(graph)
    for opset_import in onnx_model.opset_import:
        if opset_import.domain == "":
            opset_import.version = 21

    return onnx_model


def _convert_to_runtime_precision(
    onnx_model: onnx.ModelProto,
    *,
    quantize_mode: str,
    high_precision_dtype: str,
    direct_io_types: bool = False,
    op_types_to_exclude_fp16: list[str] | None = None,
    custom_ops_to_cast_fp32: dict | None = None,
    trt_extra_plugin_lib_paths: list[str] | None = None,
    opset: int | None = None,
    mha_accumulation_dtype: str = "fp16",
) -> onnx.ModelProto:
    """Convert a quantized model to its requested runtime precision."""
    if high_precision_dtype not in {"fp16", "bf16"}:
        return onnx_model

    source_dtype = "float" if quantize_mode == "fp8" else "float32"
    logger.info(f"Converting {source_dtype} tensors to {high_precision_dtype}")

    if quantize_mode == "fp8":
        graph = gs.import_onnx(onnx_model)
        remove_output_initializers(graph, onnx_model.graph.initializer)
        convert_fp16_io(graph)
        onnx_model = gs.export_onnx(graph)

    onnx_model = convert_to_f16(
        onnx_model,
        keep_io_types=not direct_io_types,
        op_block_list=op_types_to_exclude_fp16 or [],
        tensor_block_dict=custom_ops_to_cast_fp32 or {},
        low_precision_type=high_precision_dtype,
        trt_plugins=trt_extra_plugin_lib_paths,
        opset=opset,
    )

    if quantize_mode != "fp8":
        return onnx_model

    current_opsets = {opset.domain: opset.version for opset in onnx_model.opset_import}
    if current_opsets.get("", 0) < 19:
        onnx_model = _upgrade_opset_21(onnx_model)

    if mha_accumulation_dtype == "fp32":
        logger.info("Inserting Cast nodes to enable FP8+FP16 MHA")
        onnx_model = insert_fp8_mha_casts(onnx_model)

    return onnx_model
