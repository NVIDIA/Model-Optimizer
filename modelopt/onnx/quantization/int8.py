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

"""Performs INT8 quantization of an ONNX model, and returns the ONNX ModelProto."""

import os
import tempfile
import time

import onnx
import onnx_graphsurgeon as gs
from onnx_graphsurgeon.ir.graph import Graph
from onnx_graphsurgeon.ir.node import Node
from onnxruntime.quantization import CalibrationMethod
from onnxruntime.quantization.calibrate import CalibrationDataReader

from modelopt.onnx.autocast.convert import convert_to_f16
from modelopt.onnx.logging_config import configure_logging, logger
from modelopt.onnx.quantization.calib_utils import import_scales_from_calib_cache
from modelopt.onnx.quantization.graph_utils import (
    build_non_residual_input_map,
    classify_partially_quantized_weighted_ops,
    classify_partition_nodes,
    expand_node_names_from_patterns,
    filter_quantizable_kgen_heads,
    find_nodes_from_convs_to_exclude,
    find_nodes_from_matmul_to_exclude,
    find_nodes_to_exclude,
    get_concat_eliminated_tensors,
    remove_partial_input_qdq,
)
from modelopt.onnx.quantization.ort_patching import _quantize_static as quantize_static
from modelopt.onnx.quantization.ort_utils import configure_ort
from modelopt.onnx.quantization.partitioning import (
    find_fusible_partitions,
    find_non_quantizable_partitions_from_patterns,
    find_quantizable_nodes,
    get_skipped_output_layers,
)
from modelopt.onnx.quantization.qdq_utils import has_qdq_nodes, replace_scale_values


def _find_nodes_to_quantize(
    graph: Graph,
    quantizable_op_types: list[str],
    nodes_to_exclude: list[str] | None = None,
) -> tuple[list[Node], list[tuple[Node, Node, str]]]:
    logger.info("Finding nodes to quantize")
    # Build a map of add nodes to their non-residual inputs, i.e. fusible with Conv group
    logger.info("Building non-residual Add input map")
    non_residual_inputs, _ = build_non_residual_input_map(graph)

    logger.info("Searching for patterns like MHA, LayerNorm, etc")
    non_quantizable_hard_coded_partitions = find_non_quantizable_partitions_from_patterns(graph)
    logger.info(f"Found {len(non_quantizable_hard_coded_partitions)} non-quantizable partitions")

    # partitioned_nodes keeps track of nodes that are already part of some partition.
    # Certain nodes of those partitions are quantizable. For example, heads.
    partitioned_nodes = set(sum(non_quantizable_hard_coded_partitions, []) + nodes_to_exclude)  # noqa: RUF017
    cask_fusible_partitions, kgen_partitions = find_fusible_partitions(
        graph,
        partitioned_nodes,
        non_residual_inputs,
    )

    logger.info("Classifying partition nodes")
    _, quantizable_partition_nodes, no_quantize_inputs = classify_partition_nodes(
        cask_fusible_partitions,
    )

    no_quantize_inputs += classify_partially_quantized_weighted_ops(graph, nodes_to_exclude or [])

    quantizable_kgen_heads, no_quantize_kgen_inputs = filter_quantizable_kgen_heads(
        cask_fusible_partitions,
        kgen_partitions,
        quantizable_op_types,
        graph,
    )
    logger.info(
        f"Found {len(quantizable_partition_nodes)} quantizable partition "
        f"nodes and {len(quantizable_kgen_heads)} quantizable KGEN heads"
    )

    quantizable_nodes = quantizable_kgen_heads + quantizable_partition_nodes
    partially_quantizable_nodes = [dst for _, dst, _ in no_quantize_inputs]
    # Quantize all inputs of partially quantizable nodes by ORT
    # but remove QDQ from non-quantizable inputs in the post-processing step
    quantizable_nodes.extend(partially_quantizable_nodes)

    quantizable_nodes.extend(
        find_quantizable_nodes(graph, quantizable_nodes, partitioned_nodes, quantizable_op_types)
    )

    skip_list = get_skipped_output_layers(graph, partially_quantizable_nodes)
    quantizable_nodes = [node for node in quantizable_nodes if node.name not in skip_list]
    logger.info(f"Total number of quantizable nodes: {len(quantizable_nodes)}")

    return quantizable_nodes, no_quantize_inputs + no_quantize_kgen_inputs


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
    passes: list[str] = ["concat_elimination"],
    log_level: str = "INFO",
    calibrate_per_node: bool = False,
    custom_ops_to_quantize: list[str] = [],
    direct_io_types: bool = False,
    opset: int | None = None,
    **kwargs,
) -> onnx.ModelProto:
    """Applies INT8 quantization to an ONNX file using the compiler friendly heuristics.

    Quantization of ['Add', 'AveragePool', 'BatchNormalization', 'Clip', 'Conv', 'ConvTranspose',
    'Gemm', 'GlobalAveragePool', 'MatMul', 'MaxPool', 'Mul'] op types are supported.
    """
    configure_logging(level=log_level.upper())
    logger.info(f"Starting INT8 quantization with method: {calibration_method}")
    t_start = time.time()

    # Take the onnx graph
    onnx_model = onnx.load(onnx_path, load_external_data=True)

    graph = gs.import_onnx(onnx_model)
    graph.toposort()
    logger.debug(f"Loaded model with {len(graph.nodes)} nodes")

    # If the model already has QDQ nodes, skip the quantization process
    if has_qdq_nodes(onnx_model):
        logger.info("Model already has QDQ nodes, skipping quantization")
        return onnx_model

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
    nodes_to_exclude.extend(find_nodes_from_convs_to_exclude(graph, quantize_mode="int8"))

    # Change the default configuration of ORT quantization
    op_types_to_quantize = op_types_to_quantize or []
    if op_types_to_quantize:
        op_types_to_quantize.extend(custom_ops_to_quantize)
    op_types = {node.op for node in graph.nodes}
    trt_guided_options, quantizable_op_types = configure_ort(
        list(op_types),
        op_types_to_quantize,
        trt_extra_plugin_lib_paths,
        calibration_eps,
        calibrate_per_node,
        custom_ops_to_quantize,
    )
    logger.info(f"Quantizable op types: {[t for t in quantizable_op_types if t in op_types]}")

    # Collect node names to include in quantization
    no_quantize_inputs = []
    nodes_to_quantize = expand_node_names_from_patterns(graph, nodes_to_quantize)
    if not nodes_to_quantize:
        # If nodes_to_quantize is not passed, use user supplied op_types_to_quantize list
        nodes_to_quantize = [node.name for node in graph.nodes if node.op in op_types_to_quantize]

        # If op_types_to_quantize is not provided, use default QDQ placement algorithm
        if not nodes_to_quantize:
            quantizable_nodes, no_quantize_inputs = _find_nodes_to_quantize(
                graph, quantizable_op_types, nodes_to_exclude
            )
            nodes_to_quantize = [node.name for node in quantizable_nodes]

    # Read the calibration cache and quantize nodes for which activation scale values are cached
    if calibration_cache_path:
        act_scales_dict = import_scales_from_calib_cache(calibration_cache_path)
        logger.info(f"Using calibration cache from {calibration_cache_path}")
        iq_quantized_nodes = []
        quantized_tensors = [tensor_name.replace("_scale", "") for tensor_name in act_scales_dict]
        for node in graph.nodes:
            iq_quantized_nodes.extend(
                [node.name for node_input in node.inputs if node_input.name in quantized_tensors]
            )

        logger.info(
            f"Skipping quantization of nodes: {set(nodes_to_quantize) - set(iq_quantized_nodes)}"
        )
        nodes_to_quantize = list(set(nodes_to_quantize).intersection(iq_quantized_nodes))

    # Update the list of nodes to quantize
    nodes_to_quantize = [
        node_name for node_name in nodes_to_quantize if node_name not in nodes_to_exclude
    ]
    logger.info(f"Final number of nodes to quantize: {len(nodes_to_quantize)}")

    if not nodes_to_quantize:
        logger.info("No node or node type is selected for quantization or model does not have them")
    else:
        logger.debug(f"Selected {len(nodes_to_quantize)} nodes to quantize: {nodes_to_quantize}")

        if passes and "concat_elimination" in passes:
            group_qdq_tensors = get_concat_eliminated_tensors(onnx_model, nodes_to_quantize)
            if group_qdq_tensors:
                trt_guided_options["group_qdq_tensors"] = group_qdq_tensors
                logger.debug(f"Found {len(group_qdq_tensors)} tensor groups for concat elimination")

        # Add disable_int32_weight_adjustment flag to extra options
        trt_guided_options["QDQDisableWeightAdjustForInt32Bias"] = True
        logger.debug("Disabled weight adjustment for INT32 bias in QDQ quantization")

        # Create a temp file for intermediate model
        tmp_onnx_file, tmp_onnx_path = tempfile.mkstemp(suffix=".onnx")
        os.close(tmp_onnx_file)

        # Use ORT api to quantize the onnx model
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

        if calibration_cache_path:
            replace_scale_values(onnx_model.graph, act_scales_dict)

    if high_precision_dtype in ["fp16", "bf16"]:
        # We need to convert float to float16 so as to speed up layers like LayerNorm or GroupNorm.
        logger.info(f"Converting float32 tensors to {high_precision_dtype}")
        # Note: from convert_to_f16's perspective, high_precision_dtype is the precision to reduce to from FP32
        onnx_model = convert_to_f16(
            onnx_model,
            keep_io_types=not direct_io_types,
            op_block_list=op_types_to_exclude_fp16 or [],
            tensor_block_dict=custom_ops_to_cast_fp32 or {},
            low_precision_type=high_precision_dtype,
            trt_plugins=trt_extra_plugin_lib_paths,
            opset=opset,
        )

    if nodes_to_quantize:
        logger.info(f"Quantization completed successfully in {time.time() - t_start} seconds")

    return onnx_model
