# Adapted from https://github.com/microsoft/onnxruntime/blob/baeece44ba075009c6bfe95891a8c1b3d4571cb3/onnxruntime/python/tools/quantization/quant_utils.py
# and https://github.com/microsoft/onnxruntime/blob/baeece44ba075009c6bfe95891a8c1b3d4571cb3/onnxruntime/python/tools/quantization/calibrate.py
# and https://github.com/microsoft/onnxruntime/blob/2ac381c55397dffff327cc6efecf6f95a70f90a1/onnxruntime/python/tools/quantization/onnx_quantizer.py
# and https://github.com/microsoft/onnxruntime/blob/2ac381c55397dffff327cc6efecf6f95a70f90a1/onnxruntime/python/tools/quantization/quantize.py
#
# MIT License
#
# Copyright (c) Microsoft Corporation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 AND MIT
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

"""Static Q/DQ quantization built on ONNX Runtime."""

__all__ = []

import tempfile
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import onnx
from onnx import onnx_pb
from onnxruntime.quantization import calibrate
from onnxruntime.quantization.calibrate import (
    CalibrationDataReader,
    CalibrationMethod,
    DistributionCalibrater,
    EntropyCalibrater,
    HistogramCalibrater,
    MinMaxCalibrater,
    PercentileCalibrater,
    TensorData,
    TensorsData,
)
from onnxruntime.quantization.qdq_quantizer import QDQQuantizer
from onnxruntime.quantization.quant_utils import QuantFormat, QuantizationMode, QuantType
from onnxruntime.quantization.quant_utils import compute_scale_zp as _ort_compute_scale_zp
from onnxruntime.quantization.quantize import check_static_quant_arguments
from onnxruntime.quantization.registry import QDQRegistry, QLinearOpsRegistry

import modelopt.onnx.utils as onnx_utils
from modelopt.onnx.logging_config import logger
from modelopt.onnx.quantization.ort_calibration import _restore_histogram_calibration_dtypes
from modelopt.onnx.quantization.ort_session import load_model_with_shape_infer


def _compute_scale_zp(rmin, rmax, qmin, qmax, symmetric=False, min_real_range=None):
    """Retry FP16 scale calculation in FP32 when range subtraction overflows."""
    range_dtype = np.asarray(rmax).dtype
    if range_dtype != np.float16:
        return _ort_compute_scale_zp(rmin, rmax, qmin, qmax, symmetric, min_real_range)

    with np.errstate(over="ignore", invalid="ignore"):
        zero_point, scale = _ort_compute_scale_zp(rmin, rmax, qmin, qmax, symmetric, min_real_range)
        if np.all(np.isfinite(scale)):
            return zero_point, scale

        zero_point, scale = _ort_compute_scale_zp(
            np.asarray(rmin, dtype=np.float32),
            np.asarray(rmax, dtype=np.float32),
            qmin,
            qmax,
            symmetric,
            min_real_range,
        )
        return zero_point, np.asarray(scale, dtype=range_dtype)


def _check_opset_version(onnx_quantizer):
    ai_onnx_domain = [
        opset
        for opset in onnx_quantizer.model.model.opset_import
        if not opset.domain or opset.domain in ["ai.onnx", "ai.onnx.contrib"]
    ]
    opset_version = ai_onnx_domain[0].version

    if opset_version == 10:
        return 10

    if opset_version < 10:
        onnx_quantizer.model.model.opset_import.remove(ai_onnx_domain[0])
        onnx_quantizer.model.model.opset_import.extend([onnx.helper.make_opsetid("", 11)])
        opset_version = 11

    if opset_version < 19 and onnx_quantizer.weight_qType == onnx_pb.TensorProto.FLOAT8E4M3FN:
        onnx_quantizer.model.model.opset_import.remove(ai_onnx_domain[0])
        onnx_quantizer.model.model.opset_import.extend([onnx.helper.make_opsetid("", 19)])
        # Set ir_version to 10, remove it once ORT supports ir_version 11
        onnx_quantizer.model.model.ir_version = 10
        opset_version = 19

    onnx_quantizer.fuse_dynamic_quant = True
    return opset_version


def _adjust_tensor_ranges(base_quantizer):
    if base_quantizer.tensors_range is None:
        return

    for node in base_quantizer.model.nodes():
        # adjust tensor_ranges for input of Clip and Relu node
        if node.op_type in ["Clip", "Relu"]:
            if base_quantizer.is_activation_symmetric:
                continue
            if not base_quantizer.should_quantize_node(node):
                continue
            if len(base_quantizer.model.input_name_to_nodes()[node.input[0]]) != 1:
                continue
            if (
                node.input[0] not in base_quantizer.tensors_range
                or node.output[0] not in base_quantizer.tensors_range
            ):
                continue
            td = base_quantizer.tensors_range[node.output[0]]
            if not isinstance(td, TensorData):
                raise TypeError(f"Unexpected type {type(td)} for {node.output[0]!r}.")
            base_quantizer.tensors_range[node.input[0]] = td
        # Adjust Softmax to range from 0.0 to 1.0
        elif node.op_type == "Softmax":
            if node.output[0] not in base_quantizer.tensors_range:
                continue
            base_quantizer.tensors_range[node.output[0]] = TensorData(
                lowest=np.float32(0.0),
                highest=np.float32(1.0),
                avg=np.float32(0.0),
                std=np.float32(1.0),
            )

    # Patching nan values in TensorData
    # These nan values should not appear in calibration with real inputs
    for tensor_name in base_quantizer.tensors_range:
        td = base_quantizer.tensors_range[tensor_name]
        if np.isnan(td.range_value).any():
            base_quantizer.tensors_range[tensor_name] = TensorData(
                lowest=np.float32(0.0),
                highest=np.float32(448.0),
            )


def _create_calibrator_with_extra_options(
    model: str | Path,
    op_types_to_calibrate: Sequence[str] | None = None,
    augmented_model_path="augmented_model.onnx",
    calibrate_method=CalibrationMethod.MinMax,
    use_external_data_format=False,
    extra_options={},
):
    """This function overwrite is needed to pass the TRT plugin path and EP list to the inference session."""
    calibrator = None
    if calibrate_method == CalibrationMethod.MinMax:
        # default settings for min-max algorithm
        symmetric = extra_options.get("symmetric", False)
        moving_average = extra_options.get("moving_average", False)
        averaging_constant = extra_options.get("averaging_constant", 0.01)
        max_intermediate_outputs = extra_options.get("max_intermediate_outputs", None)
        calibrator = MinMaxCalibrater(
            model,
            op_types_to_calibrate,
            augmented_model_path,
            use_external_data_format=use_external_data_format,
            symmetric=symmetric,
            moving_average=moving_average,
            averaging_constant=averaging_constant,
            max_intermediate_outputs=max_intermediate_outputs,
        )
    elif calibrate_method == CalibrationMethod.Entropy:
        # default settings for entropy algorithm
        num_bins = extra_options.get("num_bins", 128)
        num_quantized_bins = extra_options.get("num_quantized_bins", 128)
        symmetric = extra_options.get("symmetric", False)
        calibrator = EntropyCalibrater(
            model,
            op_types_to_calibrate,
            augmented_model_path,
            use_external_data_format=use_external_data_format,
            symmetric=symmetric,
            num_bins=num_bins,
            num_quantized_bins=num_quantized_bins,
        )
    elif calibrate_method == CalibrationMethod.Percentile:
        # default settings for percentile algorithm
        num_bins = extra_options.get("num_bins", 2048)
        percentile = extra_options.get("percentile", 99.999)
        symmetric = extra_options.get("symmetric", True)
        calibrator = PercentileCalibrater(
            model,
            op_types_to_calibrate,
            augmented_model_path,
            use_external_data_format=use_external_data_format,
            symmetric=symmetric,
            num_bins=num_bins,
            percentile=percentile,
        )

    elif calibrate_method == CalibrationMethod.Distribution:
        # default settings for percentile algorithm
        num_bins = extra_options.get("num_bins", 2048)
        scenario = extra_options.get("scenario", "same")

        calibrator = DistributionCalibrater(
            model,
            op_types_to_calibrate,
            augmented_model_path,
            use_external_data_format=use_external_data_format,
            num_bins=num_bins,
            scenario=scenario,
        )

    if calibrator:
        calibrator.augment_graph()
        # ======== Modification: additional parameter with TRT plugin path ========
        calibrator.create_inference_session(**extra_options)
        # =========================================================================
        return calibrator

    raise ValueError(f"Unsupported calibration method {calibrate_method}")


def _quantize_static(
    model_input: str | Path | onnx.ModelProto,
    model_output: str | Path,
    calibration_data_reader: CalibrationDataReader,
    quant_format=QuantFormat.QDQ,
    op_types_to_quantize=None,
    per_channel=False,
    reduce_range=False,
    activation_type=QuantType.QInt8,
    weight_type=QuantType.QInt8,
    nodes_to_quantize=None,
    nodes_to_exclude=None,
    use_external_data_format=False,
    calibrate_method=CalibrationMethod.MinMax,
    extra_options=None,
):
    """Modification: enables TRT custom ops in the calibrator via 'TrtExtraPluginLibraryPaths' in extra_options.

    See ort.quantization.quantize.quantize_static for full function description. Additional info:

    extra_options:
        key value pair dictionary for various options in different case. Current used:
            ...
            TrtExtraPluginLibraryPaths = string :
                Default is None. Set TensorRT plugin paths if required.
            ExecutionProviders = list[string] :
                Default is [("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider",
                "TensorrtExecutionProvider"]
            TrtRtxBackend = string :
                Selects the legacy or ABI TensorRT-RTX execution provider implementation.
    """
    logger.info("Starting static quantization")
    logger.debug(f"Quantization format: {quant_format}")
    logger.debug(f"Activation type: {activation_type}")
    logger.debug(f"Weight type: {weight_type}")
    logger.debug(f"Calibration method: {calibrate_method}")
    if (
        QuantType.QFLOAT8E4M3FN in (activation_type, weight_type)
        and calibrate_method != CalibrationMethod.Distribution
    ):
        raise ValueError(
            "Only Distribution calibration method is supported for float quantization."
        )

    extra_options = extra_options or {}
    nodes_to_exclude = nodes_to_exclude or []
    nodes_to_quantize = nodes_to_quantize or []
    op_types_to_quantize = op_types_to_quantize or []
    mode = QuantizationMode.QLinearOps

    if not op_types_to_quantize or len(op_types_to_quantize) == 0:
        q_linear_ops = list(QLinearOpsRegistry.keys())
        qdq_ops = list(QDQRegistry.keys())
        op_types_to_quantize = list(set(q_linear_ops + qdq_ops))

    model = (
        onnx_utils.infer_shapes(model_input)
        if isinstance(model_input, onnx.ModelProto)
        else load_model_with_shape_infer(Path(model_input))
    )

    calib_extra_options_keys = [
        ("CalibTensorRangeSymmetric", "symmetric"),
        ("CalibMovingAverage", "moving_average"),
        ("CalibMovingAverageConstant", "averaging_constant"),
        ("CalibMaxIntermediateOutputs", "max_intermediate_outputs"),
        # ====================== Modification ======================
        ("TrtExtraPluginLibraryPaths", "trt_extra_plugin_lib_paths"),
        ("ExecutionProviders", "execution_providers"),
        ("TrtRtxBackend", "trt_rtx_backend"),
        ("group_qdq_tensors", "group_qdq_tensors"),
        ("QDQDisableWeightAdjustForInt32Bias", "disable_int32_weight_adjustment"),
        # ==========================================================
    ]
    calib_extra_options = {
        key: extra_options.get(name)
        for (name, key) in calib_extra_options_keys
        if name in extra_options
    }
    logger.debug(f"Calibration extra options: {calib_extra_options}")

    with tempfile.TemporaryDirectory(prefix="ort.quant.") as quant_tmp_dir:
        if isinstance(model_input, onnx.ModelProto):
            output_path = str(Path(quant_tmp_dir) / "model_input.onnx")
            logger.debug(f"Saving model to temporary path: {output_path}")
            onnx.save_model(
                model_input,
                output_path,
                save_as_external_data=True,
            )
            model_input = output_path

        logger.debug("Creating calibrator")
        calibrator = calibrate.create_calibrator(
            Path(model_input),
            # ======== Modification ========
            nodes_to_quantize,
            # ======== Modification ========
            augmented_model_path=Path(quant_tmp_dir).joinpath("augmented_model.onnx").as_posix(),
            calibrate_method=calibrate_method,
            use_external_data_format=use_external_data_format,
            extra_options=calib_extra_options,
        )

        logger.debug("Collecting calibration data")
        calibrator.collect_data(calibration_data_reader)
        logger.debug("Computing tensor ranges")
        tensors_range = calibrator.compute_data()
        if not isinstance(tensors_range, TensorsData):
            logger.error(f"Unexpected type {type(tensors_range)} for tensors_range")
            raise TypeError(
                f"Unexpected type {type(tensors_range)} for tensors_range and calibrator={type(calibrator)}."
            )
        if isinstance(calibrator, HistogramCalibrater):
            _restore_histogram_calibration_dtypes(calibrator.collector, tensors_range)
        del calibrator

    check_static_quant_arguments(quant_format, activation_type, weight_type)

    if quant_format is QuantFormat.QOperator:
        quantizer = QDQQuantizer(
            model,
            per_channel,
            reduce_range,
            mode,
            True,  # static
            weight_type,
            activation_type,
            tensors_range,
            nodes_to_quantize,
            nodes_to_exclude,
            op_types_to_quantize,
            extra_options,
        )
    else:
        quantizer = QDQQuantizer(
            model,
            per_channel,
            reduce_range,
            weight_type,
            activation_type,
            tensors_range,
            nodes_to_quantize,
            nodes_to_exclude,
            op_types_to_quantize,
            extra_options,
        )

    quantizer.quantize_model()
    quantizer.model.save_model_to_file(model_output, use_external_data_format)
