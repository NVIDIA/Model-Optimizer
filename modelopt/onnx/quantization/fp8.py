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

"""Performs FP8 GEMM only quantization of an ONNX model, and returns the ONNX ModelProto."""

import os
import tempfile
import time

import numpy as np
import onnx
import onnx_graphsurgeon as gs
from onnx import numpy_helper
from onnxruntime.quantization import CalibrationMethod
from onnxruntime.quantization.calibrate import CalibrationDataReader

import modelopt.onnx.utils as onnx_utils
from modelopt.onnx.autocast.convert import convert_to_f16
from modelopt.onnx.logging_config import configure_logging, logger
from modelopt.onnx.quantization.graph_utils import (
    convert_fp16_io,
    expand_node_names_from_patterns,
    find_nodes_from_convs_to_exclude,
    find_nodes_from_matmul_to_exclude,
    find_nodes_to_exclude,
    get_concat_eliminated_tensors,
    get_tensor_producer_nodes,
    insert_fp8_mha_casts,
    remove_output_initializers,
    remove_partial_input_qdq,
)
from modelopt.onnx.quantization.int8 import _find_nodes_to_quantize
from modelopt.onnx.quantization.ort_patching import _quantize_static as quantize_static
from modelopt.onnx.quantization.ort_utils import configure_ort
from modelopt.onnx.quantization.qdq_utils import has_qdq_nodes


def int8_to_fp8(onnx_model: onnx.ModelProto) -> onnx.ModelProto:
    """Converts the INT8 quantized model to FP8 quantized model.

    Note. This conversion works only for max calibrated INT8 models.

    Args:
        onnx_model: INT8 quantized ONNX model.

    Returns:
        FP8 quantized ONNX model.
    """
    logger.info("Starting INT8 to FP8 conversion")
    graph = onnx_model.graph
    initializers = graph.initializer
    tensor_producers = get_tensor_producer_nodes(graph)
    processed_tensor = set()
    initializer_indices = {
        initializer.name: idx for idx, initializer in enumerate(graph.initializer)
    }

    def _int8_scale_to_fp8_scale(scale: np.ndarray, scale_name: str):
        np_scale = onnx.numpy_helper.to_array(scale)
        np_fp8_scale = (np_scale * 448.0) / 127.0
        dtype = onnx.helper.tensor_dtype_to_np_dtype(scale.data_type)
        return numpy_helper.from_array(np_fp8_scale.astype(dtype), scale_name)

    def _update_tensor_type(tensor_name):
        tensor = onnx_utils.get_tensor_by_name(onnx_model, tensor_name)
        if tensor:
            tensor.type.tensor_type.elem_type = onnx.TensorProto.FLOAT8E4M3FN

    def _convert(node: onnx.NodeProto):
        scale_name = node.input[1]
        zero_point_name = node.input[2]

        if scale_name not in processed_tensor:
            scale_idx = initializer_indices.get(scale_name)
            if scale_idx is not None:
                scale = initializers[scale_idx]
                fp8_scale = _int8_scale_to_fp8_scale(scale, scale_name)
                initializers[scale_idx].CopyFrom(fp8_scale)
            else:
                producer_node = tensor_producers[scale_name]
                scale = producer_node.attribute[0].t
                fp8_scale = _int8_scale_to_fp8_scale(scale, scale_name)
                producer_node.attribute[0].t.CopyFrom(fp8_scale)
            processed_tensor.add(scale_name)

        if zero_point_name not in processed_tensor:
            zero_point_idx = initializer_indices.get(zero_point_name)
            assert zero_point_idx is not None, (
                f"Expected '{zero_point_name}' to be found in 'graph.initializer', but it was not present."
            )
            zero_point = initializers[zero_point_idx]
            dtype = onnx.helper.tensor_dtype_to_np_dtype(zero_point.data_type)
            vals = np.array(zero_point.int32_data, dtype=dtype).tobytes() or zero_point.raw_data

            np_zero_point = onnx.helper.make_tensor(
                zero_point_name, onnx.TensorProto.FLOAT8E4M3FN, zero_point.dims, vals, raw=True
            )
            initializers[zero_point_idx].CopyFrom(np_zero_point)
            processed_tensor.add(zero_point_name)

        # Update the Q input tensor type and the DQ output tensor type
        if node.op_type == "QuantizeLinear":
            for out in node.output:
                _update_tensor_type(out)
        if node.op_type == "DequantizeLinear":
            _update_tensor_type(node.input[0])

    # Iterate through the nodes and convert the scales and zero points
    # Also update the Q input tensor type and the DQ output tensor type
    for node in graph.node:
        if node.op_type in ["DequantizeLinear", "QuantizeLinear"]:
            _convert(node)

    return onnx_model


def upgrade_opset_21(onnx_model: onnx.ModelProto) -> onnx.ModelProto:
    """Modifies the ONNX graph such that it follows the opset 21 requirements.

    This is necessary for FP8+FP16 quantization since FP8 QuantizeLinear/DequantizeLinear ops do not support FP16
    scaling factors until opset 21.
    """
    logger.info("Upgrading model to opset 21")
    graph = gs.import_onnx(onnx_model)

    for node in graph.nodes:
        # QuantizeLinear/DequantizeLinear op with FP16 scales are only supported with empty domain
        # and opset_import version=21.
        if node.op in {"QuantizeLinear", "DequantizeLinear"}:
            node.domain = ""

        # ReduceMean op no longer has "axes" attribute in opset 21. Instead, it should be the second input tensor.
        if node.op == "ReduceMean" and "axes" in node.attrs:
            axes = gs.Constant(
                name=node.name + "_axes", values=np.array(node.attrs["axes"], dtype=np.int64)
            )
            del node.attrs["axes"]
            node.inputs.append(axes)

    onnx_model = gs.export_onnx(graph)

    # Set opset_import version to 21.
    for opset_import in onnx_model.opset_import:
        if opset_import.domain == "":
            opset_import.version = 21

    return onnx_model


def quantize(
    onnx_path: str,
    calibration_method: str = "entropy",
    calibration_data_reader: CalibrationDataReader = None,
    calibration_cache_path: str | None = None,
    calibration_shapes: str | dict | None = None,
    calibration_eps: list[str] = ["cpu", "cuda:0", "trt"],
    op_types_to_quantize: list[str] | None = None,
    op_types_to_exclude: list[str] | None = None,
    op_types_to_exclude_fp16: list[str] | None = None,
    custom_ops_to_cast_fp32: dict | None = None,
    nodes_to_quantize: list[str] | None = None,
    nodes_to_exclude: list[str] | None = None,
    use_external_data_format: bool = False,
    intermediate_generated_files: list[str] = [],
    trt_extra_plugin_lib_paths: list[str] | None = None,
    high_precision_dtype: str = "fp16",
    mha_accumulation_dtype: str = "fp16",
    passes: list[str] = ["concat_elimination"],
    log_level: str = "INFO",
    calibrate_per_node: bool = False,
    custom_ops_to_quantize: list[str] = [],
    direct_io_types: bool = False,
    opset: int | None = None,
    **kwargs,
) -> onnx.ModelProto:
    """Applies FP8 GEMM only quantization to an ONNX file.

    Currently, ['Conv', 'Gemm', 'MatMul', 'Residual-Add'] quantization is supported.
    """
    configure_logging(level=log_level.upper())
    logger.info("Starting FP8 quantization process")
    t_start = time.time()

    # Load the onnx graph
    logger.info(f"Loading ONNX model from {onnx_path}")
    onnx_model = onnx.load(onnx_path, load_external_data=True)
    onnx_model = onnx_utils.infer_shapes(onnx_model)
    graph = gs.import_onnx(onnx_model)
    graph.toposort()

    # If the model already has QDQ nodes, skip the quantization process
    if has_qdq_nodes(onnx_model):
        logger.info("Model already has QDQ nodes, skipping quantization")
        return onnx_model

    # The quantizable op types for FP8 are limited to Conv, Gemm, Matmul, and Residual-Add
    fp8_supported_op_types = ["Gemm", "MatMul", "Conv", "Add"]
    op_types_to_quantize = op_types_to_quantize or fp8_supported_op_types
    if not set(op_types_to_quantize) <= set(fp8_supported_op_types):
        raise RuntimeError(
            f"Unsupported op types in fp8 mode: '{set(op_types_to_quantize) - set(fp8_supported_op_types)}'"
        )
    op_types_to_quantize.extend(list(custom_ops_to_quantize))

    enable_gemv_detection_for_trt = kwargs.get("enable_gemv_detection_for_trt", True)
    if enable_gemv_detection_for_trt:
        # Either of m or n in matmul is 1, this matmul cannot utilize TensorCores.
        # The perf of adding Q/DQ layers is not good in TRT. Thus, in this case,
        # do not add Q/DQ layers to this matmul.
        logger.info("Detecting GEMV patterns for TRT optimization")
        matmul_nodes_to_exclude = find_nodes_from_matmul_to_exclude(
            onnx_path,
            use_external_data_format,
            intermediate_generated_files,
            calibration_data_reader,
            calibration_eps,
            calibration_shapes,
        )
        nodes_to_exclude.extend(matmul_nodes_to_exclude)  # type: ignore[union-attr]
        logger.debug(f"Excluding {len(matmul_nodes_to_exclude)} MatMul nodes due to GEMV pattern")

    # Collect node names to exclude from quantization
    nodes_to_exclude = find_nodes_to_exclude(graph, nodes_to_exclude, op_types_to_exclude)  # type: ignore[arg-type]
    nodes_to_exclude.extend(find_nodes_from_convs_to_exclude(graph, quantize_mode="fp8"))

    # Change the default configuration of ORT quantization
    op_types = {node.op for node in graph.nodes}
    trt_guided_options, quantizable_op_types = configure_ort(
        list(op_types),
        op_types_to_quantize,
        trt_extra_plugin_lib_paths,
        calibration_eps,
        calibrate_per_node,
        custom_ops_to_quantize,
    )
    logger.info(
        f"Quantizable op types in the model: {[t for t in op_types_to_quantize if t in op_types]}"
    )

    # Collect node names to include in quantization
    no_quantize_inputs = []
    nodes_to_quantize = expand_node_names_from_patterns(graph, nodes_to_quantize)
    if not nodes_to_quantize:
        quantizable_nodes, no_quantize_inputs = _find_nodes_to_quantize(
            graph, quantizable_op_types, nodes_to_exclude
        )
        nodes_to_quantize = [node.name for node in quantizable_nodes]

    # Update the list of nodes to quantize
    nodes_to_quantize = [
        node_name for node_name in nodes_to_quantize if node_name not in nodes_to_exclude
    ]

    if not nodes_to_quantize:
        logger.info("No node or node type is selected for quantization or model does not have them")
    else:
        logger.debug(f"Selected nodes to quantize: {nodes_to_quantize}")

        if passes and "concat_elimination" in passes:
            group_qdq_tensors = get_concat_eliminated_tensors(onnx_model, nodes_to_quantize)
            if group_qdq_tensors:
                trt_guided_options["group_qdq_tensors"] = group_qdq_tensors
                logger.debug(f"Grouping QDQ tensors for concat elimination: {group_qdq_tensors}")

        # Add disable_int32_weight_adjustment flag to extra options
        trt_guided_options["QDQDisableWeightAdjustForInt32Bias"] = True
        logger.debug("Disabled weight adjustment for INT32 bias in QDQ quantization")

        # Create a temp file for intermediate model
        tmp_onnx_file, tmp_onnx_path = tempfile.mkstemp(suffix=".onnx")
        os.close(tmp_onnx_file)
        logger.debug(f"Created temporary file for intermediate model: {tmp_onnx_path}")

        # Quantize in INT8 mode using ORT's Entropy or MinMax calibration method, with
        # ActivationSymmetric as True. For MinMax, that's equivalent to max calibration.
        logger.info(f"Starting INT8 quantization with '{calibration_method}' calibration")
        quantize_static(
            onnx_path,
            tmp_onnx_path,
            calibration_data_reader,
            op_types_to_quantize=op_types_to_quantize,
            nodes_to_quantize=nodes_to_quantize,
            per_channel=True,
            extra_options=trt_guided_options,
            use_external_data_format=use_external_data_format,
            calibrate_method=(
                CalibrationMethod.Entropy
                if calibration_method == "entropy"
                # With ActivationSymmetric as True, MinMax calibration is equivalent to max calibration
                else CalibrationMethod.MinMax
            ),
        )
        intermediate_generated_files.append(tmp_onnx_path)
        if use_external_data_format:
            intermediate_generated_files.append(tmp_onnx_path + ".data")

        # Post-processing of the onnx model after ORT quantization
        logger.info("Starting post-processing of quantized model")
        onnx_model = onnx.load(tmp_onnx_path)
        graph = gs.import_onnx(onnx_model)
        remove_partial_input_qdq(graph, no_quantize_inputs)
        onnx_model = gs.export_onnx(graph)

    if high_precision_dtype in ["fp16", "bf16"]:
        # We need to convert float to float16/bfloat16 so as to speed up layers like LayerNorm or GroupNorm.
        logger.info(f"Converting float tensors to {high_precision_dtype}")
        graph = gs.import_onnx(onnx_model)
        remove_output_initializers(graph, onnx_model.graph.initializer)
        convert_fp16_io(graph)
        onnx_model = gs.export_onnx(graph)

        # Convert to fp16/bf16 model.
        onnx_model = convert_to_f16(
            onnx_model,
            keep_io_types=not direct_io_types,
            op_block_list=op_types_to_exclude_fp16 or [],
            tensor_block_dict=custom_ops_to_cast_fp32 or {},
            low_precision_type=high_precision_dtype,
            trt_plugins=trt_extra_plugin_lib_paths,
            opset=opset,
        )

        current_opsets = {opset.domain: opset.version for opset in onnx_model.opset_import}
        opset_of_default_onnx_domain = current_opsets.get("", 0)
        if opset_of_default_onnx_domain < 19:
            # We need to convert the ONNX model to opset 19+ since FP8 QuantizeLinear/DequantizeLinear ops do not
            # support FP16 scaling factors until opset 19. So, converting here to opset-21 (19+).
            onnx_model = upgrade_opset_21(onnx_model)

        if mha_accumulation_dtype == "fp32":
            # Insert Cast nodes in MHA's BMM1 and BMM2's input and output tensors because
            # The compiler only has FP32 accumulation kernels for FP8 MHAs.
            logger.info("Inserting Cast nodes to enable FP8+FP16 MHA")
            onnx_model = insert_fp8_mha_casts(onnx_model)

    if nodes_to_quantize:
        onnx_model = int8_to_fp8(onnx_model)
        logger.info(f"FP8 quantization completed in {time.time() - t_start:.2f} seconds")

    return onnx_model
