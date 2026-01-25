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

"""This module is used to convert an ONNX model (w/o QDQ nodes) and calibration data into an ONNX model with QDQ nodes.

Quantization typically targets linear operations like Convolution (Conv), Matrix Multiplication (MatMul), etc., as these
transformations often yield significant performance improvements. However, many other operations can also be quantized
(i.e., they have low-precision kernels available), which can provide optimal performance with a minimal drop in
accuracy.

The default operation types that this ONNX post-training quantization (PTQ) tool quantizes in different quantization
modes are as follows:
- INT8 mode: Add, AveragePool, BatchNormalization, Clip, Conv, ConvTranspose, Gemm, GlobalAveragePool, MatMul, MaxPool,
Mul
- INT4 mode: Gemm, MatMul
- FP8 mode: Conv, Gemm, MatMul

This tool inserts Quantize-Dequantize (QDQ) nodes following compiler-friendly patterns and generates an explicit ONNX
model.
"""

import os
import platform
import shutil
import tempfile
from collections.abc import Sequence
from typing import Any

import onnx
import onnx.onnx_cpp2py_export.checker as C
import onnx_graphsurgeon as gs
import onnxslim

from modelopt.onnx.logging_config import configure_logging, logger
from modelopt.onnx.op_types import is_data_dependent_shape_op
from modelopt.onnx.quantization.calib_utils import (
    CalibrationDataProvider,
    CalibrationDataType,
    RandomDataProvider,
)
from modelopt.onnx.quantization.fp8 import quantize as quantize_fp8
from modelopt.onnx.quantization.graph_utils import (
    cast_custom_ops,
    find_nodes_from_mha_to_exclude,
    get_input_shapes,
    print_stat,
    remove_redundant_cast_nodes,
    validate_op_types_spelling,
)
from modelopt.onnx.quantization.int4 import quantize as quantize_int4
from modelopt.onnx.quantization.int8 import quantize as quantize_int8
from modelopt.onnx.quantization.ort_utils import update_trt_ep_support
from modelopt.onnx.quantization.qdq_utils import (
    qdq_to_dq,
    remove_graph_input_q,
    remove_input_dq_and_output_q,
)
from modelopt.onnx.trt_utils import interpret_trt_plugins_precision_flag, load_onnx_model
from modelopt.onnx.utils import (
    BASE_MIN_OPSET,
    QDQ_PRECISION_MIN_OPSET,
    duplicate_shared_constants,
    get_opset_version,
    name_onnx_nodes,
    save_onnx,
)

__all__ = ["quantize"]


def _normalize_quantize_mode_for_opset(quantize_mode: str) -> str:
    """Map variants like "int4_awq", "int4_rtn", "nvfp4" to their base precision types for lookup purposes."""
    mode_lower = quantize_mode.lower()
    if "int4" in mode_lower:
        return "int4"
    if "nvfp4" in mode_lower or "float4" in mode_lower:
        return "float4_e2m1fn"
    # For "int8", "fp8", etc., return as-is (fp8 falls back to BASE_MIN_OPSET which is correct)
    return quantize_mode


def _preprocess_onnx(
    onnx_path: str,
    use_external_data_format: bool,
    output_path: str,
    enable_shared_constants_duplication: bool,
    trt_plugins: list[str] | None,
    trt_plugins_precision: list[str] | None,
    override_shapes: str,
    simplify: bool = False,
    quantize_mode: str = "int8",
    opset: int | None = None,
) -> tuple[str, onnx.ModelProto, list[str], bool, bool, bool, dict, dict]:
    logger.info(f"Preprocessing the model {onnx_path}")
    intermediate_generated_files = []
    output_dir = os.path.dirname(output_path)
    model_name = os.path.splitext(os.path.basename(onnx_path))[0]

    # Load the model and weights
    onnx_model, has_custom_op, custom_ops, onnx_path, use_external_data_format = load_onnx_model(
        onnx_path,
        trt_plugins,
        override_shapes,
        use_external_data_format,
        intermediate_generated_files,
    )
    if has_custom_op:
        onnx_path = os.path.join(output_dir, f"{model_name}_ort_support.onnx")
        save_onnx(onnx_model, onnx_path, use_external_data_format)
        logger.info(
            f"Model with custom ops is saved to {onnx_path}. Model contains custom ops: {custom_ops}"
        )
        intermediate_generated_files.append(onnx_path)
    elif platform.system() != "Windows":
        logger.info(
            "No custom ops found. If that's not correct, please make sure that the 'tensorrt' python package"
            " is correctly installed and that the paths to 'libcudnn*.so' and TensorRT 'lib/' are in"
            " 'LD_LIBRARY_PATH'. If the custom op is not directly available as a plugin in TensorRT, please"
            " also make sure that the path to the compiled '.so' TensorRT plugin is also being given via the "
            " '--trt_plugins' flag (requires TRT 10+)."
        )

    # Opset 19 is the minimum required for fp16 scales in Q/DQ nodes
    # Higher opsets required for specific quantization modes (int4: 21, nvfp4: 23)
    original_opset_version = get_opset_version(onnx_model)

    # Determine minimum required opset based on quantization mode
    # Normalize quantize_mode to handle variants like "int4_awq", "nvfp4", etc.
    normalized_mode = _normalize_quantize_mode_for_opset(quantize_mode)
    mode_min_opset = QDQ_PRECISION_MIN_OPSET.get(normalized_mode, BASE_MIN_OPSET)

    # Determine target opset version
    if opset is not None:
        target_opset = opset
        # Warn if user-specified opset is below mode minimum (but still respect it)
        if opset < mode_min_opset:
            logger.warning(
                f"Opset {opset} is below the minimum opset {mode_min_opset} required for "
                f"{quantize_mode} quantization. Upgrading to opset {mode_min_opset}."
            )
            target_opset = mode_min_opset
        # Warn if user-specified opset is lower than original
        if opset < original_opset_version:
            logger.warning(
                f"Specified opset {opset} is lower than the original model's opset {original_opset_version}. "
                f"Using original model's opset {original_opset_version}."
            )
            target_opset = max(target_opset, original_opset_version)
    else:
        # Use model's opset if it's >= mode_min_opset, otherwise upgrade to mode_min_opset
        target_opset = (
            max(original_opset_version, mode_min_opset)
            if original_opset_version != 1
            else mode_min_opset
        )

    if original_opset_version < target_opset and original_opset_version != 1:
        onnx_model = onnx.version_converter.convert_version(onnx_model, target_opset)
        onnx_path = os.path.join(output_dir, f"{model_name}_opset{target_opset}.onnx")
        save_onnx(onnx_model, onnx_path, use_external_data_format)
        logger.info(f"Model is cloned to {onnx_path} with opset_version {target_opset}")
        intermediate_generated_files.append(onnx_path)

    # Simplify model if requested
    if simplify:
        logger.info("Attempting to simplify model")
        try:
            model_simp = onnxslim.slim(onnx_model, skip_fusion_patterns=["FusionGemm"])
            if model_simp:
                onnx_model = model_simp
                onnx_path = os.path.join(output_dir, f"{model_name}_simp.onnx")
                save_onnx(onnx_model, onnx_path, use_external_data_format)
                intermediate_generated_files.append(onnx_path)
                logger.info(f"Simplified model was validated and saved in {onnx_path}")
            else:
                logger.warning(
                    "Simplified ONNX model could not be validated. Will use the original model"
                )
        except Exception as e:
            logger.warning(
                f"Simplification of {onnx_path} failed with error: {e}. Will use the original model"
            )

    # Check if data-dependent shape ops are present in the model
    graph = gs.import_onnx(onnx_model)
    has_dds_op = len([node for node in graph.nodes if is_data_dependent_shape_op(node.op)]) > 0
    if has_dds_op:
        logger.debug("Found data-dependent shape operations in the model")

    # Sometimes input onnx model does not contain the node names
    # This tool depends on those names, so we assign names if needed
    graph = onnx_model.graph
    is_named = name_onnx_nodes(graph)
    is_duplicated_constant = False
    if enable_shared_constants_duplication:
        logger.info("Duplicating shared constants")
        onnx_model, is_duplicated_constant = duplicate_shared_constants(
            onnx_model
        )  # FasterViT-0, eef

    if is_named or is_duplicated_constant:
        onnx_path = os.path.join(output_dir, f"{model_name}_named.onnx")
        save_onnx(onnx_model, onnx_path, use_external_data_format)
        logger.info(f"Model is cloned to {onnx_path} after naming the nodes")
        intermediate_generated_files.append(onnx_path)

    # If custom op precisions are given, add Cast or Q/DQ where appropriate.
    custom_ops_to_cast = {}
    custom_ops_to_quantize = {}
    if trt_plugins_precision:
        custom_ops_to_cast, custom_ops_to_quantize = interpret_trt_plugins_precision_flag(
            onnx_model, trt_plugins_precision, quantize_mode
        )
        if custom_ops_to_cast.get("fp16", {}):
            onnx_model = cast_custom_ops(onnx_model, custom_ops_to_cast["fp16"])
            onnx_path = os.path.join(output_dir, f"{model_name}_castFP16.onnx")
            save_onnx(onnx_model, onnx_path, use_external_data_format)
            logger.info(f"Model is cloned to {onnx_path} after casting tensors to FP16")
            intermediate_generated_files.append(onnx_path)

    return (
        onnx_path,
        onnx_model,
        intermediate_generated_files,
        has_custom_op,
        has_dds_op,
        use_external_data_format,
        custom_ops_to_cast.get("fp32", {}),
        custom_ops_to_quantize,
    )


def quantize(
    onnx_path: str,
    quantize_mode: str = "int8",
    calibration_data: CalibrationDataType = None,
    calibration_method: str | None = None,
    calibration_cache_path: str | None = None,
    calibration_shapes: str | None = None,
    calibration_eps: list[str] = ["cpu", "cuda:0", "trt"],
    override_shapes: str | None = None,
    op_types_to_quantize: list[str] | None = None,
    op_types_to_exclude: list[str] | None = None,
    op_types_to_exclude_fp16: list[str] | None = None,
    nodes_to_quantize: list[str] | None = None,
    nodes_to_exclude: list[str] | None = None,
    use_external_data_format: bool = False,
    keep_intermediate_files: bool = False,
    output_path: str | None = None,
    log_level: str = "INFO",
    log_file: str | None = None,
    trt_plugins: list[str] | None = None,
    trt_plugins_precision: list[str] | None = None,
    high_precision_dtype: str = "fp16",
    mha_accumulation_dtype: str = "fp16",
    disable_mha_qdq: bool = False,
    dq_only: bool = False,
    block_size: int | None = None,
    use_zero_point: bool = False,
    passes: list[str] = ["concat_elimination"],
    simplify: bool = False,
    calibrate_per_node: bool = False,
    input_shapes_profile: Sequence[dict[str, str]] | None = None,
    direct_io_types: bool = False,
    opset: int | None = None,
    **kwargs: Any,
) -> None:
    """Quantizes the provided ONNX model.

    Args:
        onnx_path:
            Path to the input ONNX model.
        quantize_mode:
            Quantization mode. One of 'int8' (default), 'int4' and 'fp8'.
        calibration_data:
            Calibration data, either a numpy array or list/dict of numpy arrays.
        calibration_method:
            Calibration method choices. Options are int8/fp8: {'entropy' (default), 'max'}
            and int4: {'awq_clip' (default), 'awq_lite', 'awq_full', 'rtn_dq'}.
        calibration_cache_path:
            Path to pre-calculated activation tensor ranges, also known as calibration cache.
        calibration_shapes:
            Input shapes used for calibration process.
            It should be provided as a string representing the shape of each input tensors for one calibration step.
            Example input shapes spec: input0:1x3x256x256,input1:1x3x128x128
        calibration_eps:
            Priority order for the execution providers (EP) to calibrate the model.
            Any subset of ['NvTensorRtRtx', 'trt', 'cuda:x', 'dml:x', 'cpu'], where 'x' is the device id.

            .. note::
                If a custom op is detected in the model, 'trt' will automatically be added to the EP list.
        override_shapes:
            Override model input shapes with static shapes.
        op_types_to_quantize:
            List of op types to quantize. If None (default), all supported operators are quantized.
            This flag does not support regular expression.
        op_types_to_exclude:
            List of op types to exclude from quantization. This flag does not support regular expression.
        op_types_to_exclude_fp16:
            List of op types to exclude from FP16 conversion.
            This is only relevant if '--high_precision_dtype != fp32'.
        nodes_to_quantize:
            List of node names to quantize. If None (default), all supported nodes are quantized.
            This flag supports regular expression.
        nodes_to_exclude:
            List of node names to exclude from quantization. This flag supports regular expression.
        use_external_data_format:
            If True, separate data path will be used to store the weights of the quantized model.
        keep_intermediate_files:
            If True, keep all intermediate files generated during the ONNX model's conversion/calibration.
        output_path:
            Output filename to save the quantized ONNX model.
            If None, save in the same directory as the original ONNX model with .quant suffix.
        log_level:
            Log level. One of 'DEBUG', 'INFO', 'WARNING', 'ERROR'.
        log_file:
            Path to the log file for the quantization process.
        trt_plugins:
            A space-separated list with the custom TensorRT plugin library paths in .so format (compiled shared
            library). If this is not None or the model has custom ops, TensorrtExecutionProvider becomes the first
            choice as calibration execution provider, meaning that the TensorRT is a requirement.
        trt_plugins_precision:
            A space-separated list indicating the precision for each custom op.
            Each item should have the format <op_type>:<precision>, where precision can be fp32 (default) or fp16.
            For example: op_type_1:fp16 op_type_2:fp32.
        high_precision_dtype:
            High precision data type of the output model. If high_precision_dtype is 'fp16' or 'bf16'
            and the input model is of dtype fp32, model's weight and activation will be converted to
            'fp16' or 'bf16'.
        mha_accumulation_dtype:
            MHA accumulation dtype. One of ['fp32', 'fp16']. 'fp16' by default. If quantize_mode == 'fp8' and
            mha_accumulation_dtype == 'fp32', Cast nodes will be added to MHA's bmm1 and bmm2's input
            and output tensors.
        disable_mha_qdq:
            Don't add Q/DQ layers to MatMuls in MHA pattern.
        dq_only:
            If True, only add DQ nodes to the model. If False (default), add Q/DQ nodes to the model.
        block_size:
            Block size parameter for int4 quantization.
        use_zero_point:
            Use zero-point based quantization, if True.
        passes:
            List of optimization passes name, if set, appropriate pre/post-processing passes will be invoked.
        simplify:
            Simplify the given model before quantization.
        calibrate_per_node:
            Calibrate the model node by node instead of calibrating the entire model. This allows calibration with
            a lower system memory with the cost of longer calibration time.
        input_shapes_profile:
            This is a sequence of shapes-profile for each EP in calibration_eps. Some EPs like NvTensorRtRtx use these
            shapes profile for optimized engine generation for those input shapes. Length of this parameters should
            equal length of calibration_eps (i.e. one profile data per EP in calibration_eps, in that order).
            A shapes-profile comprises of "min", "max", and "opt" values for the shapes of model inputs
            (esp. dynamic shapes). Consider following example snippets for shape-profile data-format of some EPs.

                input_shape_profile_for_NvTensorRtrRtx_EP = {
                    "nv_profile_min_shapes":  "input1:dim1xdim2...,input2:dim1xdim2...,...",

                    "nv_profile_max_shapes":  "input1:dim1xdim2...,input2:dim1xdim2...,...",

                    "nv_profile_opt_shapes":  "input1:dim1xdim2...,input2:dim1xdim2...,...",

                }

                input_shape_profile_for_TensorRT_EP = {
                    "trt_profile_min_shapes":  "input1:dim1xdim2...,input2:dim1xdim2...,...",

                    "trt_profile_max_shapes":  "input1:dim1xdim2...,input2:dim1xdim2...,...",

                    "trt_profile_opt_shapes":  "input1:dim1xdim2...,input2:dim1xdim2...,...",

                }

            For EPs that don't require such shapes profile (e.g. CPU EP, CUDA EP, DML EP), empty profile {} can be used.
            For example, if calibration_eps are ["NvTensorRtRtx", "cpu"], then input_shapes_profile can be set to:

            - [input_shapes_profile_for_NvTensorRtRtx_EP, {}]

            If None of the calibration_eps require any such shapes profile for model inputs, then nothing needs to be
            set for this "input_shapes_profile" parameter.
            Default value is None.
        direct_io_types:
            If True, modify the I/O types in the quantized ONNX model to be lower precision whenever possible.
            If False, keep the I/O types in the quantized ONNX model the same as in the given ONNX model.
        opset:
            Target ONNX opset version for the quantized model. If None, uses required minimum opset
            (19 for int8/fp8, 21 for int4, 23 for nvfp4). If the specified opset is lower than the required minimum,
            a warning will be issued and the opset will be upgraded to the required minimum.
        kwargs:
            Additional keyword arguments for int4 quantization, including:
            - awqlite_alpha_step (float): Alpha step for lite, range [0, 1].
            - awqclip_alpha_step (float): Min alpha step for clip, range [awqclip_alpha_step, 1].
            - awqclip_alpha_min (float): Alpha step to find best alpha for clip.
            - awqclip_bsz_col (int): Batch size for processing the column dimension in clip.

    Returns:
        None, writes the quantized onnx model in the supplied output_path
        or writes to the same directory with filename like "<model_name>.quant.onnx".
    """
    configure_logging(log_level.upper(), log_file)
    logger.info(f"Starting quantization process for model: {onnx_path}")
    logger.info(f"Quantization mode: {quantize_mode}")

    if calibrate_per_node and quantize_mode not in ["int8", "fp8"]:
        raise ValueError(
            "Per node calibration is only supported for int8 and fp8 quantization modes"
        )

    # quantize_static creates a shape-inferred copy at the input model's directory
    # Needs to check if we have write permission to this directory
    assert onnx_path.endswith((".onnx", ".pb"))
    if not os.access(os.path.dirname(os.path.abspath(onnx_path)), os.W_OK):
        old_dir = os.path.dirname(os.path.abspath(onnx_path))
        tmp_dir = tempfile.mkdtemp()
        logger.info(f"Directory {old_dir} is not writable, store intermediate files in {tmp_dir}")
        onnx_path = os.path.join(tmp_dir, os.path.basename(onnx_path))

        # We assume that the model directory contains only model related weights and protobuf file
        # Anything extra in the model directory will be copied unnecessarily
        for file in os.listdir(old_dir):
            old_file_path = os.path.join(old_dir, file)
            new_file_path = os.path.join(tmp_dir, file)
            if os.path.isfile(old_file_path):
                shutil.copy(old_file_path, new_file_path)

    model_name = os.path.splitext(os.path.basename(onnx_path))[0]
    if not output_path:
        output_dir = os.path.dirname(onnx_path)
        output_path = os.path.join(output_dir, f"{model_name}.quant.onnx")
        logger.info(f"No output path specified, save quantized model to {output_path}")
    else:
        if os.path.isabs(output_path):
            output_dir = os.path.dirname(output_path)
        else:
            output_dir = os.path.dirname(os.path.join(os.getcwd(), output_path))
        assert os.path.exists(output_dir), f'output directory "{output_dir}" does not exist'

    # We need to preprocess the model with naming, weight duplication etc.
    enable_shared_constants_duplication = kwargs.get("enable_shared_constants_duplication", True)
    (
        onnx_path,
        onnx_model,
        intermediate_generated_files,
        has_custom_op,
        has_dds_op,
        use_external_data_format,
        custom_ops_to_cast_fp32,
        custom_ops_to_quantize,
    ) = _preprocess_onnx(
        onnx_path,
        use_external_data_format,
        output_path,
        enable_shared_constants_duplication,
        trt_plugins,
        trt_plugins_precision,
        override_shapes,  # type: ignore[arg-type]
        simplify,
        quantize_mode,
        opset,
    )
    trt_plugins = update_trt_ep_support(calibration_eps, has_dds_op, has_custom_op, trt_plugins)  # type: ignore[arg-type]

    # Use random scales if calibration data is not supplied
    if calibration_data is None:
        calibration_data_reader = RandomDataProvider(onnx_path, calibration_shapes)
    else:
        calibration_data_reader = CalibrationDataProvider(
            onnx_path, calibration_data, calibration_shapes
        )

    nodes_to_quantize = nodes_to_quantize or []
    nodes_to_exclude = nodes_to_exclude or []

    # Check op types spelling in 'op_types_to_exclude' and '_to_quantize'
    validate_op_types_spelling(onnx_path, op_types_to_quantize, op_types_to_exclude)

    # (1) If disable_mha_qdq is set, don't add Q/DQ layers to MatMuls in MHA pattern.
    # (2) else when quantize_mode == "int8", if seq_len > 512, don't add Q/DQ layers to
    # MatMuls in MHA pattern.
    # (3) else when quantize_mode == "fp8", if head_size > 256 or head_size <= 8
    # or mha doesn't meet fp8 fMHA v2 pattern, don't add Q/DQ layers to MatMuls in MHA pattern.
    nodes_to_exclude = find_nodes_from_mha_to_exclude(
        onnx_path,
        use_external_data_format,
        nodes_to_exclude,
        disable_mha_qdq,
        quantize_mode,
        intermediate_generated_files,
        calibration_data_reader,
        calibration_eps,
    )

    if calibrate_per_node and not calibration_shapes:
        calibration_shapes = get_input_shapes(onnx_path)

    if quantize_mode in ["fp8", "int8"]:
        quantize_func = quantize_int8 if quantize_mode == "int8" else quantize_fp8
        onnx_model = quantize_func(
            onnx_path=onnx_path,
            calibration_method=calibration_method or "entropy",
            calibration_data_reader=calibration_data_reader,
            calibration_cache_path=calibration_cache_path,
            calibration_shapes=calibration_shapes,
            calibration_eps=calibration_eps,
            op_types_to_quantize=op_types_to_quantize,
            op_types_to_exclude=op_types_to_exclude,
            op_types_to_exclude_fp16=op_types_to_exclude_fp16,
            custom_ops_to_cast_fp32=custom_ops_to_cast_fp32,
            nodes_to_quantize=nodes_to_quantize,
            nodes_to_exclude=nodes_to_exclude,
            use_external_data_format=use_external_data_format,
            intermediate_generated_files=intermediate_generated_files,
            trt_extra_plugin_lib_paths=trt_plugins,
            high_precision_dtype=high_precision_dtype,
            mha_accumulation_dtype=mha_accumulation_dtype,
            passes=passes,
            log_level=log_level,
            calibrate_per_node=calibrate_per_node,
            custom_ops_to_quantize=list(custom_ops_to_quantize.keys()),
            direct_io_types=direct_io_types,
            opset=opset,
            **kwargs,
        )
    elif "int4" in quantize_mode:
        onnx_model = quantize_int4(
            onnx_path=onnx_path,
            calibration_method=calibration_method or "awq_clip",
            calibration_data_reader=calibration_data_reader,
            calibration_eps=calibration_eps,
            use_external_data_format=use_external_data_format,
            block_size=block_size,
            nodes_to_exclude=nodes_to_exclude,
            use_zero_point=use_zero_point,
            log_level=log_level,
            input_shapes_profile=input_shapes_profile,
            **kwargs,
        )
    else:
        raise RuntimeError(f"Invalid quantization mode choice: {quantize_mode}")

    if onnx_model:
        # Fuse Q nodes for INT8/FP8 mode
        if quantize_mode in ["int8", "fp8"]:
            if dq_only:
                onnx_model = qdq_to_dq(onnx_model)
            if custom_ops_to_quantize:
                # Remove DQ nodes from the input and Q from the output of the requested custom ops
                onnx_model = remove_input_dq_and_output_q(
                    onnx_model, quantizable_custom_ops=custom_ops_to_quantize
                )
            if direct_io_types:
                onnx_model = remove_graph_input_q(onnx_model)
        else:
            # Remove redundant cast nodes in the quantized model
            # Note. This is called within the qdq_to_dq function as well
            remove_redundant_cast_nodes(onnx_model.graph)

        # Collect and print stats of the quantized model
        print_stat(gs.import_onnx(onnx_model))

        # Save the quantized model to the output path
        save_onnx(onnx_model, output_path, use_external_data_format)
        logger.info(f"Quantized onnx model is saved as {output_path}")

    # Check if intermediate files should be deleted
    if not keep_intermediate_files:
        logger.info("Cleaning up intermediate files")
        for file in intermediate_generated_files:
            if os.path.exists(file):
                os.remove(file)
            if use_external_data_format and os.path.exists(file + "_data"):
                os.remove(file + "_data")

    # Check if the quantized model is valid
    try:
        logger.info("Validating quantized model")
        onnx.checker.check_model(output_path)
    except C.ValidationError as e:
        logger.warning("ONNX model checker failed, check your deployment status")
        logger.warning(e)

    logger.info("Quantization process completed")
