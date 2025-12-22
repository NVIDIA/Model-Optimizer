# Adapted from https://github.com/microsoft/onnxruntime/blob/baeece44ba075009c6bfe95891a8c1b3d4571cb3/onnxruntime/python/tools/quantization/quant_utils.py
# and https://github.com/microsoft/onnxruntime/blob/baeece44ba075009c6bfe95891a8c1b3d4571cb3/onnxruntime/python/tools/quantization/calibrate.py
# and https://github.com/microsoft/onnxruntime/blob/2ac381c55397dffff327cc6efecf6f95a70f90a1/onnxruntime/python/tools/quantization/onnx_quantizer.py
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

"""This module contains all the patched functions from ORT."""

import gc
import tempfile
import uuid
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pynvml
from onnx import onnx_pb
from onnxruntime.quantization import calibrate
from onnxruntime.quantization.base_quantizer import BaseQuantizer
from onnxruntime.quantization.calibrate import (
    CalibraterBase,
    CalibrationDataReader,
    CalibrationMethod,
    DistributionCalibrater,
    EntropyCalibrater,
    HistogramCalibrater,
    HistogramCollector,
    MinMaxCalibrater,
    PercentileCalibrater,
    TensorData,
    TensorsData,
)
from onnxruntime.quantization.qdq_quantizer import QDQQuantizer
from onnxruntime.quantization.quant_utils import (
    QuantFormat,
    QuantizationMode,
    QuantType,
    add_infer_metadata,
)
from onnxruntime.quantization.quantize import check_static_quant_arguments
from onnxruntime.quantization.registry import QDQRegistry, QLinearOpsRegistry
from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference
from tqdm import tqdm

import modelopt.onnx.utils as onnx_utils
from modelopt.onnx.logging_config import logger


def load_model_with_shape_infer(model_path: Path) -> onnx.ModelProto:
    """Load model while performing symbolic shape infer and ONNX shape inference."""
    model = onnx.load(str(model_path), load_external_data=True)
    try:
        model = onnx_utils.infer_shapes(model)
        add_infer_metadata(model)
    except Exception as e:
        logger.info(f"Failed to infer shapes for model {model_path}: {e}")
    return model


def _collect_value(histogram_collector, name_to_arr):
    """Collect histogram on real value."""
    for tensor, data_arr in tqdm(name_to_arr.items()):
        # ====================== Modification ======================
        concat_data_arr = np.asarray(data_arr[0])
        concat_data_arr = concat_data_arr.flatten()
        for i in range(1, len(data_arr)):
            curr_data_arr = np.asarray(data_arr[i])
            curr_data_arr = curr_data_arr.flatten()
            concat_data_arr = np.concatenate((concat_data_arr, curr_data_arr))

        data_arr = concat_data_arr
        # ==========================================================
        if data_arr.size > 0:
            min_value = np.min(data_arr)
            max_value = np.max(data_arr)
        else:
            min_value = np.array(0, dtype=data_arr.dtype)
            max_value = np.array(0, dtype=data_arr.dtype)

        # Change the inf and nan values to meaningful min/max
        min_value = (
            np.finfo(np.float32).tiny if np.isinf(min_value) or np.isnan(min_value) else min_value
        )
        max_value = (
            np.finfo(np.float32).max if np.isinf(max_value) or np.isnan(max_value) else max_value
        )

        threshold = max(abs(min_value), abs(max_value))

        if tensor in histogram_collector.histogram_dict:
            old_histogram = histogram_collector.histogram_dict[tensor]
            histogram_collector.histogram_dict[tensor] = histogram_collector.merge_histogram(
                old_histogram, data_arr, min_value, max_value, threshold
            )
        else:
            hist, hist_edges = np.histogram(
                data_arr, histogram_collector.num_bins, range=(-threshold, threshold)
            )
            histogram_collector.histogram_dict[tensor] = (
                hist,
                hist_edges,
                min_value,
                max_value,
                threshold,
            )


def _collect_absolute_value(histogram_collector, name_to_arr):
    """Collect histogram on absolute value."""
    for tensor, data_arr in name_to_arr.items():
        if isinstance(data_arr, list):
            for arr in data_arr:
                assert isinstance(arr, np.ndarray), (
                    f"Unexpected type {type(arr)} for tensor={tensor!r}"
                )
            dtypes = {a.dtype for a in data_arr}
            assert len(dtypes) == 1, (
                f"The calibration expects only one element type but got {dtypes} for tensor={tensor!r}"
            )
            # ====================== Modification ======================
            concat_data_arr = np.asarray(data_arr[0])
            concat_data_arr = concat_data_arr.flatten()
            for i in range(1, len(data_arr)):
                curr_data_arr = np.asarray(data_arr[i])
                curr_data_arr = curr_data_arr.flatten()
                concat_data_arr = np.concatenate((concat_data_arr, curr_data_arr))
            data_arr_np = concat_data_arr
            # ==========================================================
        elif not isinstance(data_arr, np.ndarray):
            raise ValueError(f"Unexpected type {type(data_arr)} for tensor={tensor!r}")
        else:
            data_arr_np = data_arr
        data_arr_np = data_arr_np.flatten()
        if data_arr_np.size > 0:
            min_value = np.min(data_arr_np)
            max_value = np.max(data_arr_np)
        else:
            min_value = np.array(0, dtype=data_arr_np.dtype)
            max_value = np.array(0, dtype=data_arr_np.dtype)

        data_arr_np = np.absolute(data_arr_np)  # only consider absolute value

        if tensor not in histogram_collector.histogram_dict:
            # first time it uses num_bins to compute histogram.
            hist, hist_edges = np.histogram(data_arr_np, bins=histogram_collector.num_bins)
            hist_edges = hist_edges.astype(data_arr_np.dtype)
            assert data_arr_np.dtype != np.float64, (
                "only float32 or float16 is supported, every constant must be explicitly typed"
            )
            histogram_collector.histogram_dict[tensor] = (hist, hist_edges, min_value, max_value)
        else:
            old_histogram = histogram_collector.histogram_dict[tensor]
            old_min = old_histogram[2]
            old_max = old_histogram[3]
            assert hasattr(old_min, "dtype"), (
                f"old_min should be a numpy array but is {type(old_min)}"
            )
            assert hasattr(old_max, "dtype"), (
                f"old_min should be a numpy array but is {type(old_max)}"
            )
            old_hist = old_histogram[0]
            old_hist_edges = old_histogram[1]
            temp_amax = np.max(data_arr_np)
            if temp_amax > old_hist_edges[-1]:
                # increase the number of bins
                width = old_hist_edges[1] - old_hist_edges[0]
                # NOTE: np.arange may create an extra bin after the one containing temp_amax
                new_bin_edges = np.arange(old_hist_edges[-1] + width, temp_amax + width, width)
                old_hist_edges = np.hstack((old_hist_edges, new_bin_edges))
            hist, hist_edges = np.histogram(data_arr_np, bins=old_hist_edges)
            hist_edges = hist_edges.astype(data_arr_np.dtype)
            hist[: len(old_hist)] += old_hist
            assert data_arr_np.dtype != np.float64, (
                "only float32 or float16 is supported, every constant must be explicitly typed"
            )
            histogram_collector.histogram_dict[tensor] = (
                hist,
                hist_edges,
                min(old_min, min_value),
                max(old_max, max_value),
            )


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


def _select_tensors_to_calibrate(calibrator, model: onnx.ModelProto):
    """Select input/output tensors of candidate nodes to calibrate.

    Returns:
        tensors (set): set of tensor name.
        value_infos (dict): tensor name to value info.
    """
    value_infos = {vi.name: vi for vi in model.graph.value_info}
    value_infos.update({ot.name: ot for ot in model.graph.output})
    value_infos.update({it.name: it for it in model.graph.input})
    initializer = {init.name for init in model.graph.initializer}

    tensors_to_calibrate = set()
    tensor_type_to_calibrate = {onnx_pb.TensorProto.FLOAT, onnx_pb.TensorProto.FLOAT16}

    for node in model.graph.node:
        # Hack: in calibrator.op_types_to_calibrate we pass nodes_to_quantize
        if node.name in calibrator.op_types_to_calibrate:
            for tensor_name in node.input:
                if tensor_name in value_infos:
                    vi = value_infos[tensor_name]
                    if (
                        vi.type.HasField("tensor_type")
                        and (vi.type.tensor_type.elem_type in tensor_type_to_calibrate)
                        and (tensor_name not in initializer)
                    ):
                        tensors_to_calibrate.add(tensor_name)
            for tensor_name in node.output:
                if tensor_name in value_infos:
                    vi = value_infos[tensor_name]
                    if vi.type.HasField("tensor_type") and (
                        vi.type.tensor_type.elem_type in tensor_type_to_calibrate
                    ):
                        tensors_to_calibrate.add(tensor_name)

    return tensors_to_calibrate, value_infos


def _create_inference_session_with_ep_config(calibrator, **kwargs):
    """Create an ORT InferenceSession."""
    model_path = kwargs.get("model_path")
    logger.debug("Creating inference session with Execution Provider configuration")

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess_options.add_session_config_entry("session.use_device_allocator_for_initializers", "1")
    sess_options.enable_cpu_mem_arena = False

    providers = kwargs.get("execution_providers", [])
    logger.debug(f"Execution providers: {providers}")

    # Note. This path can be an empty string, which denotes that the model has custom ops and TRT EP is needed.
    calibrator.trt_extra_plugin_lib_paths = kwargs.get("trt_extra_plugin_lib_paths")

    if calibrator.trt_extra_plugin_lib_paths is not None:
        logger.debug(f"TRT extra plugin paths: {calibrator.trt_extra_plugin_lib_paths}")
        if "TensorrtExecutionProvider" not in ort.get_available_providers():
            raise RuntimeError(
                f"Could not find `TensorrtExecutionProvider`, only {ort.get_available_providers()}"
            )
        trt_ep_options = (
            {"trt_extra_plugin_lib_paths": calibrator.trt_extra_plugin_lib_paths}
            if calibrator.trt_extra_plugin_lib_paths
            else {}
        )

        # Set GPU memory usage limit
        trt_ep_options["trt_max_workspace_size"] = 80 * (1024**3)  # 80GB
        logger.debug(f"TRT EP options: {trt_ep_options}")

        if "TensorrtExecutionProvider" in providers:
            providers.remove("TensorrtExecutionProvider")
        providers.insert(0, ("TensorrtExecutionProvider", trt_ep_options))

    def _update_provider_config(provider, config):
        if isinstance(provider, tuple) and len(provider) > 1 and isinstance(provider[1], dict):
            provider[1].update(config)
        else:
            provider = (provider, config)
        return provider

    for i in range(len(providers)):
        if any(p in providers[i] for p in ["CPUExecutionProvider", "CUDAExecutionProvider"]):
            providers[i] = _update_provider_config(
                providers[i], {"arena_extend_strategy": "kSameAsRequested"}
            )

    if model_path is None:
        # Create the inference session with EP configuration on augmented_model
        calibrator.infer_session = ort.InferenceSession(
            calibrator.augmented_model_path,
            sess_options=sess_options,
            providers=providers,
        )
    else:
        # Create the inference session with EP configuration on provided model path
        calibrator.infer_session = ort.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=providers,
        )

    # Group qdq tensors will have the same scaling factor.
    calibrator.group_qdq_tensors = kwargs.get("group_qdq_tensors")
    if calibrator.group_qdq_tensors:
        logger.debug(f"Group QDQ tensors: {calibrator.group_qdq_tensors}")


def _compute_data_minmax_calibrator(calibrator):
    """Compute the min-max range of tensor.

    :returns: dictionary mapping: {added node names: (ReduceMin, ReduceMax) pairs }
    """
    if len(calibrator.intermediate_outputs) == 0:
        return calibrator.calibrate_tensors_range

    output_names = [
        calibrator.infer_session.get_outputs()[i].name
        for i in range(len(calibrator.intermediate_outputs[0]))
    ]

    output_dicts_list = [
        dict(zip(output_names, intermediate_output), strict=True)
        for intermediate_output in calibrator.intermediate_outputs
    ]

    merged_output_dict = {}
    for d in output_dicts_list:
        for k, v in d.items():
            merged_output_dict.setdefault(k, []).append(v)

    # ====================== Modification ======================
    # Group qdq tensors should have the same scaling factor. Each tensor in group should add
    # other tensors in its merged_dict value. In this way, calibrator will generate the same
    # scaling factor.
    if calibrator.group_qdq_tensors:
        for cur, group in calibrator.group_qdq_tensors.items():
            for other in group:
                if cur == other:
                    continue
                for d in output_dicts_list:
                    for k, v in d.items():
                        cur_min = cur + "_" + "ReduceMin"
                        cur_max = cur + "_" + "ReduceMax"
                        other_min = other + "_" + "ReduceMin"
                        other_max = other + "_" + "ReduceMax"
                        if k == other_min:
                            merged_output_dict[cur_min].append(v)
                        elif k == other_max:
                            merged_output_dict[cur_max].append(v)
    # ============================================================

    added_output_names = output_names[calibrator.num_model_outputs :]
    calibrate_tensor_names = [
        added_output_names[i].rpartition("_")[0] for i in range(0, len(added_output_names), 2)
    ]  # output names

    merged_added_output_dict = {
        i: merged_output_dict[i]
        for i in merged_output_dict
        if i not in calibrator.model_original_outputs
    }

    pairs = []
    for i in range(0, len(added_output_names), 2):
        if calibrator.moving_average:
            min_value_array = np.mean(merged_added_output_dict[added_output_names[i]], axis=0)
            max_value_array = np.mean(merged_added_output_dict[added_output_names[i + 1]], axis=0)
        else:
            min_value_array = np.min(merged_added_output_dict[added_output_names[i]], axis=0)
            max_value_array = np.max(merged_added_output_dict[added_output_names[i + 1]], axis=0)

        if calibrator.symmetric:
            max_absolute_value = np.max([np.abs(min_value_array), np.abs(max_value_array)], axis=0)
            pairs.append((-max_absolute_value, max_absolute_value))
        else:
            pairs.append((min_value_array, max_value_array))

    new_calibrate_tensors_range = TensorsData(
        CalibrationMethod.MinMax, dict(zip(calibrate_tensor_names, pairs, strict=False))
    )
    if calibrator.calibrate_tensors_range:
        calibrator.calibrate_tensors_range = calibrator.merge_range(
            calibrator.calibrate_tensors_range, new_calibrate_tensors_range
        )
    else:
        calibrator.calibrate_tensors_range = new_calibrate_tensors_range

    return calibrator.calibrate_tensors_range


def _compute_data_min_max_calibrater_single_node_calibration(calibrater) -> TensorData:
    """Compute the min-max range of tensor.

    :return: dictionary mapping: {added node names: (ReduceMin, ReduceMax) pairs }

    Modification: Instead of aggregating two consecutive outputs to a MinMax pair, retrieve a MinMax pair from
        outputs of Concat.
    """
    if not calibrater.intermediate_outputs:
        return calibrater.calibrate_tensors_range

    # Get output names and merge all intermediate outputs
    output_names = [out.name for out in calibrater.infer_session.get_outputs()]

    # Merge outputs across all batches, filtering out original model outputs
    merged_outputs = {}
    for intermediate_output in calibrater.intermediate_outputs:
        for name, value in zip(output_names, intermediate_output):
            if name not in calibrater.model_original_outputs:
                merged_outputs.setdefault(name, []).append(value)

    # Compute min/max pairs for each tensor
    pairs = []
    tensor_names = []

    for output_name, values in merged_outputs.items():
        tensor_names.append(output_name.rpartition("_")[0])

        if calibrater.moving_average:
            min_val, max_val = np.mean(values, axis=0)
        else:
            stacked_values = np.stack(values, axis=0)
            min_val = np.min(stacked_values, axis=0)[0]
            max_val = np.max(stacked_values, axis=0)[1]

        if calibrater.symmetric:
            max_abs = max(np.abs(min_val), np.abs(max_val))
            pairs.append((-max_abs, max_abs))
        else:
            pairs.append((min_val, max_val))

    # Create and merge tensor range data
    new_range = TensorsData(CalibrationMethod.MinMax, dict(zip(tensor_names, pairs)))

    calibrater.calibrate_tensors_range = (
        calibrater.merge_range(calibrater.calibrate_tensors_range, new_range)
        if calibrater.calibrate_tensors_range
        else new_range
    )

    return calibrater.calibrate_tensors_range


def _collect_data_minmax_calibrator(calibrator, data_reader: CalibrationDataReader):
    """This function overwrite is needed to solve OOM issue due to the unlimited accumulation of intermediate_outputs.

    Support for: MinMax Calibrator.
    Modification: indented the last lines of code inside the while loop in order to run compute_data for each sample
        batch individually instead of the entire data at once. The assumption here is that the ONNX file has bs=N
        and the calibration data size is M (where M is a multiple of N). So the calibrator is a sequence of M/N
        samples with bs=N.
    """
    run_options = ort.RunOptions()
    try:
        pynvml.nvmlInit()
        gpu_count = pynvml.nvmlDeviceGetCount()
        pynvml.nvmlShutdown()
    except Exception as e:
        logger.error(f"Failed to get GPU count: {e}")
        gpu_count = 0
    gpu_str = ";".join([f"gpu:{i}" for i in range(gpu_count)])
    run_options.add_run_config_entry("memory.enable_memory_arena_shrinkage", f"cpu:0;{gpu_str}")
    while True:
        inputs = data_reader.get_next()
        if not inputs:
            break
        run_options = ort.RunOptions()

        calibrator.intermediate_outputs.append(
            calibrator.infer_session.run(None, inputs, run_options=run_options)
        )

        # ======== Modification: block is indentend in ========
        if len(calibrator.intermediate_outputs) == 0:
            raise ValueError("No data is collected.")

        t = calibrator.compute_data()
        if not isinstance(t, TensorsData):
            raise TypeError(f"compute_data must return a TensorsData not {type(t)}.")
        calibrator.clear_collected_data()
        # =====================================================


def _merge_range_minmax_calibrator(calibrator, old_range: TensorsData, new_range: TensorsData):
    """This function is an auxiliary function of collect_data to solve the OOM issue in the MinMax Calibrator.

    Issue fixed with this function: old_range is not a dictionary, but old_range.data is.
    TODO: create an MR in the ORT repository for this function. Alternatively, we can also file the MR fixing
            TensorData (need to at least add items() function there).
    """
    if not old_range:
        return new_range

    for key, value in old_range.data.items():
        value_tuple = value.range_value
        new_range_tuple = new_range.data[key].range_value
        if calibrator.moving_average:
            min_value = value_tuple[0] + calibrator.averaging_constant * (
                new_range_tuple[0] - value_tuple[0]
            )
            max_value = value_tuple[1] + calibrator.averaging_constant * (
                new_range_tuple[1] - value_tuple[1]
            )
        else:
            min_value = min(value_tuple[0], new_range_tuple[0])
            max_value = max(value_tuple[1], new_range_tuple[1])
        new_range.data[key] = TensorData(lowest=min_value, highest=max_value)

    return new_range


def _merge_range_min_max_calibrater_single_node_calibration(
    calibrater, old_range: TensorsData, new_range: TensorsData
):
    """This function is an auxiliary function of collect_data to solve the OOM issue in the MinMax Calibrator.

    Issue fixed with this function: old_range is not a dictionary, but old_range.data is.
    TODO: create an MR in the ORT repository for this function. Alternatively, we can also file the MR fixing
            TensorData (need to at least add items() function there).
    """
    if not old_range:
        return new_range

    def _merge_ranges(old_min, old_max, new_min, new_max):
        if calibrater.moving_average:
            alpha = calibrater.averaging_constant
            return (old_min + alpha * (new_min - old_min), old_max + alpha * (new_max - old_max))
        return min(old_min, new_min), max(old_max, new_max)

    old_data = old_range.data
    for key, new_tensor in new_range.data.items():
        if key in old_data:
            old_min, old_max = old_data[key].range_value
            new_min, new_max = new_tensor.range_value
            merged_min, merged_max = _merge_ranges(old_min, old_max, new_min, new_max)
            old_data[key] = TensorData(lowest=merged_min, highest=merged_max)
        else:
            old_data[key] = new_tensor

    return old_range


def _collect_data_histogram_calibrator(calibrator, data_reader: CalibrationDataReader):
    """This function overwrite is needed to solve OOM issue due to the unlimited accumulation of intermediate_outputs.

    Support for: Histogram Calibrator (which affects Entropy, Percentile, and DIstribution Calibrators).
    Modification: indented the last lines of code inside the while loop in order to run compute_data for each sample
        batch individually instead of the entire data at once.
    """
    while True:
        inputs = data_reader.get_next()
        if not inputs:
            break
        calibrator.intermediate_outputs.append(calibrator.infer_session.run(None, inputs))

        # ======== Modification: block is indentend in ========
        # Here, compute_date is calculated for every sample batch instead of the entire data at once.
        if len(calibrator.intermediate_outputs) == 0:
            raise ValueError("No data is collected.")

        output_names = [
            calibrator.infer_session.get_outputs()[i].name
            for i in range(len(calibrator.intermediate_outputs[0]))
        ]
        output_dicts_list = [
            dict(zip(output_names, intermediate_output))
            for intermediate_output in calibrator.intermediate_outputs
        ]

        merged_dict = {}
        for d in output_dicts_list:
            for k, v in d.items():
                merged_dict.setdefault(k, []).append(v)

        # Group qdq tensors should have the same scaling factor. Each tensor in group should add
        # other tensors in its merged_dict value. In this way, calibrator will generate the same
        # scaling factor.
        if calibrator.group_qdq_tensors:
            for cur, group in calibrator.group_qdq_tensors.items():
                for other in group:
                    if cur == other:
                        continue
                    for d in output_dicts_list:
                        for k, v in d.items():
                            if k == other:
                                merged_dict[cur].append(v)

        clean_merged_dict = {
            i: merged_dict[i] for i in merged_dict if i in calibrator.tensors_to_calibrate
        }

        if not calibrator.collector:
            calibrator.collector = HistogramCollector(
                method=calibrator.method,
                symmetric=calibrator.symmetric,
                num_bins=calibrator.num_bins,
                num_quantized_bins=calibrator.num_quantized_bins,
                percentile=calibrator.percentile,
                scenario=calibrator.scenario,
            )
        calibrator.collector.collect(clean_merged_dict)

        calibrator.clear_collected_data()
        # =====================================================


def _collect_data_min_max_calibrater_single_node_calibration(
    calibrater, data_reader: CalibrationDataReader
):
    """Collects calibration data (min/max) for a MinMax Calibrator by processing single-node models batch by batch.

    This function addresses an OOM issue by computing calibration data for each batch individually,
    rather than accumulating all intermediate outputs across the entire dataset. It assumes the ONNX model
    has a batch size of N, and the calibration data size M is a multiple of N, processing M/N batches.

    Args:
        calibrater: The calibrater object managing model inference and data collection.
        data_reader: Provides batches of input data for calibration.
    """
    input_counter = 0
    while True:
        inputs = data_reader.get_next()
        if not inputs:
            break
        logger.debug(f"Collecting tensor data and finding min & max for input #{input_counter}")

        # We are using single node model scheme. Set up model to input dependency map
        model_to_input_dep_map = {}
        for model_path, io_tensors in calibrater.single_node_model_path_map.items():
            model_to_input_dep_map[model_path] = io_tensors[0].copy()  # List of input names

        # Setup input queues
        input_queue = [model_input.name for model_input in calibrater.model.graph.input]

        pbar = tqdm(total=len(model_to_input_dep_map.keys()))
        # Resolve nodes are independent from inputs to add their outputs as inputs
        inferred_model_list = []
        for model_path, input_deps in model_to_input_dep_map.items():
            if len(input_deps) == 0:
                calibrater.create_inference_session(
                    execution_providers=calibrater.providers,
                    trt_extra_plugin_lib_paths=calibrater.trt_extra_plugin_lib_paths,
                    model_path=model_path,
                )
                outputs = calibrater.infer_session.run(None, {})

                # Add output to inputs
                need_calibration = False
                for output_idx, output in enumerate(outputs):
                    output_name = calibrater.infer_session.get_outputs()[output_idx].name
                    inputs[output_name] = output
                    input_queue.append(output_name)
                    if (
                        output_name
                        in [output_tensor.name for output_tensor in calibrater.model.graph.output]
                        and output_name not in calibrater.model_original_outputs
                    ):
                        need_calibration = True

                # Mark model path to remove it from dependency map
                inferred_model_list.append(model_path)

                # For each inference, compute data before moving to other nodes if tensor is to be calibrated
                if need_calibration:
                    calibrater.intermediate_outputs.append(outputs)
                    if len(calibrater.intermediate_outputs) == 0:
                        raise ValueError("No data is collected.")

                    t = calibrater.compute_data()
                    if not isinstance(t, TensorsData):
                        raise TypeError(f"compute_data must return a TensorsData not {type(t)}.")
                    calibrater.clear_collected_data()

                    gc.collect()

                pbar.update(1)

        # Remove inferred model from dependency map
        for model_path in inferred_model_list:
            model_to_input_dep_map.pop(model_path)

        gc.collect()

        # Process topological inference
        input_ref_count = {}
        while input_queue:
            current_input_name = input_queue.pop(0)

            # Initialize input reference count
            input_ref_count[current_input_name] = sum(
                current_input_name in input_deps for input_deps in model_to_input_dep_map.values()
            )

            # Perform inference
            inferred_model_list = []
            for model_path, input_deps in model_to_input_dep_map.items():
                if current_input_name in input_deps:
                    input_deps.remove(current_input_name)

                    # If all dependencies are met, perform inference for the node.
                    if len(input_deps) == 0:
                        # Make dictionary of only needed inputs.
                        inputs_to_feed = {}
                        for input_name_to_feed in calibrater.single_node_model_path_map[model_path][
                            0
                        ]:
                            inputs_to_feed[input_name_to_feed] = inputs[input_name_to_feed]

                        calibrater.create_inference_session(
                            execution_providers=calibrater.providers,
                            trt_extra_plugin_lib_paths=calibrater.trt_extra_plugin_lib_paths,
                            model_path=model_path,
                        )
                        outputs = calibrater.infer_session.run(None, inputs_to_feed)

                        # Mark model path to remove it from dependency map
                        inferred_model_list.append(model_path)

                        # Decrease reference count for used inputs and remove if no reference
                        for input_name in calibrater.single_node_model_path_map[model_path][0]:
                            input_ref_count[input_name] -= 1
                            if input_ref_count[input_name] == 0:
                                del inputs[input_name]
                                del input_ref_count[input_name]

                        gc.collect()

                        # Add outputs to inputs
                        need_calibration = False
                        for output_idx, output in enumerate(outputs):
                            output_name = calibrater.infer_session.get_outputs()[output_idx].name
                            inputs[output_name] = output
                            input_queue.append(output_name)
                            if (
                                output_name
                                in [
                                    output_tensor.name
                                    for output_tensor in calibrater.model.graph.output
                                ]
                                and output_name not in calibrater.model_original_outputs
                            ):
                                need_calibration = True

                        # For each inference, compute data before moving to other nodes if tensor is to be calibrated
                        if need_calibration:
                            calibrater.intermediate_outputs.append(outputs)
                            if len(calibrater.intermediate_outputs) == 0:
                                raise ValueError("No data is collected.")

                            t = calibrater.compute_data()
                            if not isinstance(t, TensorsData):
                                raise TypeError(
                                    f"compute_data must return a TensorsData not {type(t)}."
                                )
                            calibrater.clear_collected_data()

                            gc.collect()

                        pbar.update(1)

            # Remove inferred model from dependency map
            for model_path in inferred_model_list:
                model_to_input_dep_map.pop(model_path)

            gc.collect()
        pbar.close()
        input_counter += 1


def _collect_data_histogram_calibrater_single_node_calibration(calibrator, data_reader):
    """Collects histogram data for single-node calibration, processing batches to avoid OOM.

    Args:
        calibrator: Histogram calibrator instance.
        data_reader: CalibrationDataReader providing input data.
    """
    input_counter = 0
    while True:
        inputs = data_reader.get_next()
        if not inputs:
            break
        logger.debug(f"Collecting tensor data for input #{input_counter}")

        # We are using single node model scheme. Set up model to input dependency map
        model_to_input_dep_map = {}
        for model_path, io_tensors in calibrator.single_node_model_path_map.items():
            model_to_input_dep_map[model_path] = io_tensors[0].copy()  # List of input names

        # Compute data for input tensors
        input_only_model = onnx.helper.make_model(
            onnx.helper.make_graph(
                [],
                f"{calibrator.augmented_model_path[:-5]}_input_only",
                calibrator.model.graph.input,
                calibrator.model.graph.input,
            ),
            opset_imports=calibrator.model.opset_import,
            functions=calibrator.model.functions,
            ir_version=calibrator.model.ir_version,
        )
        calibrator.infer_session = ort.InferenceSession(input_only_model.SerializeToString())
        calibrator.intermediate_outputs.append(
            [
                inputs[calibrator.infer_session.get_outputs()[i].name]
                for i in range(len(calibrator.infer_session.get_outputs()))
            ]
        )
        if len(calibrator.intermediate_outputs) == 0:
            raise ValueError("No data is collected.")

        output_names = [
            calibrator.infer_session.get_outputs()[i].name
            for i in range(len(calibrator.intermediate_outputs[0]))
        ]
        output_dicts_list = [
            dict(zip(output_names, intermediate_output))
            for intermediate_output in calibrator.intermediate_outputs
        ]

        merged_dict = {}
        for d in output_dicts_list:
            for k, v in d.items():
                merged_dict.setdefault(k, []).append(v)

        clean_merged_dict = {
            i: merged_dict[i] for i in merged_dict if i in calibrator.tensors_to_calibrate
        }

        if not calibrator.collector:
            calibrator.collector = HistogramCollector(
                method=calibrator.method,
                symmetric=calibrator.symmetric,
                num_bins=calibrator.num_bins,
                num_quantized_bins=calibrator.num_quantized_bins,
                percentile=calibrator.percentile,
                scenario=calibrator.scenario,
            )
        calibrator.collector.collect(clean_merged_dict)

        calibrator.clear_collected_data()

        gc.collect()

        # Setup input queues
        input_queue = [model_input.name for model_input in calibrator.model.graph.input]

        pbar = tqdm(total=len(model_to_input_dep_map.keys()))
        # Resolve nodes are independent from inputs to add their outputs as inputs
        inferred_model_list = []
        for model_path, input_deps in model_to_input_dep_map.items():
            if len(input_deps) == 0:
                calibrator.create_inference_session(
                    execution_providers=calibrator.providers,
                    trt_extra_plugin_lib_paths=calibrator.trt_extra_plugin_lib_paths,
                    model_path=model_path,
                )
                outputs = calibrator.infer_session.run(None, {})

                # Add output to inputs
                need_calibration = False
                for output_idx, output in enumerate(outputs):
                    output_name = calibrator.infer_session.get_outputs()[output_idx].name
                    inputs[output_name] = output
                    input_queue.append(output_name)
                    if output_name in calibrator.tensors_to_calibrate:
                        need_calibration = True

                # Mark model path to remove it from dependency map
                inferred_model_list.append(model_path)

                # For each inference, compute data before moving to other nodes if tensor is to be calibrated
                if need_calibration:
                    calibrator.intermediate_outputs.append(outputs)
                    if len(calibrator.intermediate_outputs) == 0:
                        raise ValueError("No data is collected.")

                    output_names = [
                        calibrator.infer_session.get_outputs()[i].name
                        for i in range(len(calibrator.intermediate_outputs[0]))
                    ]
                    output_dicts_list = [
                        dict(zip(output_names, intermediate_output))
                        for intermediate_output in calibrator.intermediate_outputs
                    ]

                    merged_dict = {}
                    for d in output_dicts_list:
                        for k, v in d.items():
                            merged_dict.setdefault(k, []).append(v)

                    clean_merged_dict = {
                        i: merged_dict[i]
                        for i in merged_dict
                        if i in calibrator.tensors_to_calibrate
                    }

                    if not calibrator.collector:
                        calibrator.collector = HistogramCollector(
                            method=calibrator.method,
                            symmetric=calibrator.symmetric,
                            num_bins=calibrator.num_bins,
                            num_quantized_bins=calibrator.num_quantized_bins,
                            percentile=calibrator.percentile,
                            scenario=calibrator.scenario,
                        )
                    calibrator.collector.collect(clean_merged_dict)

                    calibrator.clear_collected_data()

                    gc.collect()
                pbar.update(1)

        # Remove inferred model from dependency map
        for model_path in inferred_model_list:
            model_to_input_dep_map.pop(model_path)

        gc.collect()

        # Process topological inference
        input_ref_count = {}
        while input_queue:
            current_input_name = input_queue.pop(0)

            # Initialize input reference count
            input_ref_count[current_input_name] = sum(
                current_input_name in input_deps for input_deps in model_to_input_dep_map.values()
            )

            # Perform inference
            inferred_model_list = []
            for model_path, input_deps in model_to_input_dep_map.items():
                if current_input_name in input_deps:
                    input_deps.remove(current_input_name)

                    # If all dependencies are met, perform inference for the node.
                    if len(input_deps) == 0:
                        # Make dictionary of only needed inputs.
                        inputs_to_feed = {}
                        for input_name_to_feed in calibrator.single_node_model_path_map[model_path][
                            0
                        ]:
                            inputs_to_feed[input_name_to_feed] = inputs[input_name_to_feed]

                        calibrator.create_inference_session(
                            execution_providers=calibrator.providers,
                            trt_extra_plugin_lib_paths=calibrator.trt_extra_plugin_lib_paths,
                            model_path=model_path,
                        )
                        outputs = calibrator.infer_session.run(None, inputs_to_feed)

                        # Mark model path to remove it from dependency map
                        inferred_model_list.append(model_path)

                        # Decrease reference count for used inputs and remove if no reference
                        for input_name in calibrator.single_node_model_path_map[model_path][0]:
                            input_ref_count[input_name] -= 1
                            if input_ref_count[input_name] == 0:
                                del inputs[input_name]
                                del input_ref_count[input_name]

                        gc.collect()

                        # Add outputs to inputs
                        need_calibration = False
                        for output_idx, output in enumerate(outputs):
                            output_name = calibrator.infer_session.get_outputs()[output_idx].name
                            inputs[output_name] = output
                            input_queue.append(output_name)
                            if output_name in calibrator.tensors_to_calibrate:
                                need_calibration = True

                        # For each inference, compute data before moving to other nodes if tensor is to be calibrated
                        if need_calibration:
                            calibrator.intermediate_outputs.append(outputs)
                            if len(calibrator.intermediate_outputs) == 0:
                                raise ValueError("No data is collected.")

                            output_names = [
                                calibrator.infer_session.get_outputs()[i].name
                                for i in range(len(calibrator.intermediate_outputs[0]))
                            ]
                            output_dicts_list = [
                                dict(zip(output_names, intermediate_output))
                                for intermediate_output in calibrator.intermediate_outputs
                            ]

                            merged_dict = {}
                            for d in output_dicts_list:
                                for k, v in d.items():
                                    merged_dict.setdefault(k, []).append(v)

                            clean_merged_dict = {
                                i: merged_dict[i]
                                for i in merged_dict
                                if i in calibrator.tensors_to_calibrate
                            }

                            if not calibrator.collector:
                                calibrator.collector = HistogramCollector(
                                    method=calibrator.method,
                                    symmetric=calibrator.symmetric,
                                    num_bins=calibrator.num_bins,
                                    num_quantized_bins=calibrator.num_quantized_bins,
                                    percentile=calibrator.percentile,
                                    scenario=calibrator.scenario,
                                )
                            calibrator.collector.collect(clean_merged_dict)

                            calibrator.clear_collected_data()

                            gc.collect()
                        pbar.update(1)

            # Remove inferred model from dependency map
            for model_path in inferred_model_list:
                model_to_input_dep_map.pop(model_path)

            gc.collect()
        pbar.close()
        input_counter += 1


def _collect_histogram_collector_single_node_calibration(histogram_collector, name_to_arr):
    """Collect tensor data and make histogram.

    Modification: Remove print line to make calibration per node log output cleaner.
    """
    # TODO: Currently we have different collect() for entropy and percentile method respectively.
    #       Need unified collect in the future.
    if histogram_collector.method in {"distribution", "entropy"}:
        return histogram_collector.collect_value(name_to_arr)
    elif histogram_collector.method == "percentile":
        if histogram_collector.symmetric:
            return histogram_collector.collect_absolute_value(name_to_arr)
        else:
            return histogram_collector.collect_value(name_to_arr)
    else:
        raise ValueError("Only 'entropy', 'percentile' or 'distribution' methods are supported")


def _collect_value_histogram_collector_single_node_calibration(histogram_collector, name_to_arr):
    """Collect histogram on real value."""
    for tensor, data_arr in name_to_arr.items():
        data_arr = np.asarray(data_arr).flatten()
        min_value, max_value = (np.min(data_arr), np.max(data_arr)) if data_arr.size > 0 else (0, 0)

        # Replace inf/nan with float32 min/max
        min_value = (
            np.finfo(np.float32).tiny if np.isinf(min_value) or np.isnan(min_value) else min_value
        )
        max_value = (
            np.finfo(np.float32).max if np.isinf(max_value) or np.isnan(max_value) else max_value
        )

        threshold = max(abs(min_value), abs(max_value))

        if tensor in histogram_collector.histogram_dict:
            histogram_collector.histogram_dict[tensor] = histogram_collector.merge_histogram(
                histogram_collector.histogram_dict[tensor],
                data_arr,
                min_value,
                max_value,
                threshold,
            )
        else:
            hist, hist_edges = np.histogram(
                data_arr, histogram_collector.num_bins, range=(-threshold, threshold)
            )
            histogram_collector.histogram_dict[tensor] = (
                hist,
                hist_edges,
                min_value,
                max_value,
                threshold,
            )


def _augment_graph_min_max_calibrater_single_node_calibration(calibrater):
    """Augment outputs to retrieve MinMax pair.

    Adds ReduceMin and ReduceMax nodes to all quantization_candidates op type nodes in
    model and ensures their outputs are stored as part of the graph output.

    :return: augmented ONNX model

    Modification: Add an additional Concat after Reshaped output to not rely on error-prone indexing.
        Create multiple single node ONNX models to be used to calibrate per node.
    """
    tensors, _ = calibrater.select_tensors_to_calibrate(calibrater.model)
    reshape_shape_name = str(uuid.uuid4())
    reshape_shape = onnx.numpy_helper.from_array(np.array([1], dtype=np.int64), reshape_shape_name)
    calibrater.model.graph.initializer.append(reshape_shape)

    def add_reduce_min_max(tensor_name):
        keepdims = 1
        minmax_output = tensor_name + "_MinMax"

        # Create reduce nodes
        reduce_nodes = [
            onnx.helper.make_node(
                op_name,
                [tensor_name],
                [tensor_name + "_" + op_name + "_Reshape"],
                keepdims=keepdims,
                name=tensor_name + "_" + op_name,
            )
            for op_name in ["ReduceMin", "ReduceMax"]
        ]

        # Create reshape nodes
        reshape_nodes = [
            onnx.helper.make_node(
                "Reshape",
                inputs=[node.output[0], reshape_shape_name],
                outputs=[tensor_name + "_" + op_name],
                name=node.output[0],
            )
            for node, op_name in zip(reduce_nodes, ["ReduceMin", "ReduceMax"])
        ]

        # Create concat node
        concat_node = onnx.helper.make_node(
            "Concat",
            inputs=[tensor_name + "_ReduceMin", tensor_name + "_ReduceMax"],
            outputs=[minmax_output],
            name=tensor_name + "_ReduceMin_ReduceMax_Concat",
            axis=0,
        )

        calibrater.model.graph.node.extend(reduce_nodes + reshape_nodes + [concat_node])

        # Get tensor type
        value_infos = {vi.name: vi for vi in calibrater.model.graph.value_info}
        value_infos.update({o.name: o for o in calibrater.model.graph.output})
        value_infos.update({i.name: i for i in calibrater.model.graph.input})

        if tensor_name not in value_infos:
            raise ValueError(
                f"Unable to guess tensor type for tensor {tensor_name!r}, "
                f"running shape inference before quantization may resolve this issue."
            )

        calibrater.model.graph.output.append(
            onnx.helper.make_tensor_value_info(
                minmax_output, value_infos[tensor_name].type.tensor_type.elem_type, [2]
            )
        )

    # Make sure all shapes are resolved before adding min max nodes
    calibrater.model = SymbolicShapeInference.infer_shapes(calibrater.model)

    for tensor in tensors:
        add_reduce_min_max(tensor)

    # Make sure all shapes are resolved after adding min max nodes
    calibrater.model = SymbolicShapeInference.infer_shapes(calibrater.model)

    onnx.save(
        calibrater.model,
        calibrater.augmented_model_path,
        save_as_external_data=calibrater.use_external_data_format,
    )

    # Build single node models and save them
    model_counter = 0
    initializer_name_map = {
        initializer.name: initializer for initializer in calibrater.model.graph.initializer
    }
    value_info_name_map = {
        value_info.name: value_info for value_info in calibrater.model.graph.value_info
    }
    input_name_map = {input.name: input for input in calibrater.model.graph.input}
    output_name_map = {output.name: output for output in calibrater.model.graph.output}
    for node in calibrater.model.graph.node:
        single_node_model_name = (
            f"{calibrater.augmented_model_path[:-5]}_single_node_{model_counter}"
        )
        single_node_model_node = []
        single_node_model_inputs = []
        single_node_model_outputs = []
        single_node_model_initializers = []
        single_node_model_input_names = []
        single_node_model_output_names = []

        # Add node
        single_node_model_node.append(node)

        # Process each input for node
        for input_name in node.input:
            # Skip empty tensors
            if input_name == "":
                continue

            is_input_initializer = False
            # If a node input is an initializer, add it to initializer list
            if input_name in initializer_name_map:
                single_node_model_initializers.append(initializer_name_map[input_name])
                is_input_initializer = True

            value_info_found = False
            # If a node input is not an initializer, set it as a model input
            if not is_input_initializer:
                for name_map in [value_info_name_map, input_name_map, output_name_map]:
                    if input_name in name_map:
                        single_node_model_inputs.append(name_map[input_name])
                        single_node_model_input_names.append(input_name)
                        value_info_found = True
                        break

                if not value_info_found:
                    raise ValueError(
                        f"{calibrater.augmented_model_path} is not properly shape inferenced."
                    )

        # Process each output for node
        for output_name in node.output:
            value_info_found = False
            for name_map in [value_info_name_map, output_name_map]:
                if output_name in name_map:
                    single_node_model_outputs.append(name_map[output_name])
                    single_node_model_output_names.append(output_name)
                    value_info_found = True
                    break

            if not value_info_found:
                raise ValueError(
                    f"{calibrater.augmented_model_path} is not properly shape inferenced."
                )

        # Create a new onnx model
        single_node_model = onnx.helper.make_model(
            onnx.helper.make_graph(
                single_node_model_node,
                single_node_model_name,
                single_node_model_inputs,
                single_node_model_outputs,
                single_node_model_initializers,
            ),
            opset_imports=calibrater.model.opset_import,
            functions=calibrater.model.functions,
            ir_version=calibrater.model.ir_version,
        )

        # Save it to a new onnx file
        onnx.save(single_node_model, f"{single_node_model_name}.onnx")

        # Save model info and increase model counter
        calibrater.single_node_model_path_map[f"{single_node_model_name}.onnx"] = (
            single_node_model_input_names,
            single_node_model_output_names,
        )
        model_counter += 1


def _augment_graph_histogram_calibrater_single_node_calibration(calibrater):
    """Make all quantization_candidates op type nodes as part of the graph output.

    :return: augmented ONNX model
    """
    calibrater.tensors_to_calibrate, value_infos = calibrater.select_tensors_to_calibrate(
        calibrater.model
    )
    for tensor in calibrater.tensors_to_calibrate:
        if tensor not in calibrater.model_original_outputs:
            calibrater.model.graph.output.append(value_infos[tensor])

    onnx.save(
        calibrater.model,
        calibrater.augmented_model_path,
        save_as_external_data=calibrater.use_external_data_format,
    )

    # Build single node models and save them
    initializer_name_map = {
        initializer.name: initializer for initializer in calibrater.model.graph.initializer
    }
    value_info_name_map = {
        value_info.name: value_info for value_info in calibrater.model.graph.value_info
    }
    input_name_map = {input.name: input for input in calibrater.model.graph.input}
    output_name_map = {output.name: output for output in calibrater.model.graph.output}
    model_counter = 0
    for node in calibrater.model.graph.node:
        single_node_model_name = (
            f"{calibrater.augmented_model_path[:-5]}_single_node_{model_counter}"
        )
        single_node_model_nodes = []
        single_node_model_inputs = []
        single_node_model_outputs = []
        single_node_model_initializers = []
        single_node_model_input_names = []
        single_node_model_output_names = []

        # Add node
        single_node_model_nodes.append(node)

        # Process each input for node
        for input_name in node.input:
            # Skip empty tensors
            if input_name == "":
                continue

            is_input_initializer = False
            # If a node input is an initializer, add it to initializer list
            if input_name in initializer_name_map:
                single_node_model_initializers.append(initializer_name_map[input_name])
                is_input_initializer = True

            # If a node input is not an initializer, set it as a model input
            if not is_input_initializer:
                value_info_found = False
                for name_map in [value_info_name_map, input_name_map, output_name_map]:
                    if input_name in name_map:
                        single_node_model_inputs.append(name_map[input_name])
                        single_node_model_input_names.append(input_name)
                        value_info_found = True
                        break

                if not value_info_found:
                    raise ValueError(
                        f"{calibrater.augmented_model_path} is not properly shape inferenced."
                    )

        # Process each output for node
        for output_name in node.output:
            value_info_found = False
            for name_map in [value_info_name_map, output_name_map]:
                if output_name in name_map:
                    single_node_model_outputs.append(name_map[output_name])
                    single_node_model_output_names.append(output_name)
                    value_info_found = True
                    break

            if not value_info_found:
                raise ValueError(
                    f"{calibrater.augmented_model_path} is not properly shape inferenced."
                )

        # Create a new onnx model
        single_node_model = onnx.helper.make_model(
            onnx.helper.make_graph(
                single_node_model_nodes,
                single_node_model_name,
                single_node_model_inputs,
                single_node_model_outputs,
                single_node_model_initializers,
            ),
            opset_imports=calibrater.model.opset_import,
            functions=calibrater.model.functions,
            ir_version=calibrater.model.ir_version,
        )

        # Save it to a new onnx file
        onnx.save(single_node_model, f"{single_node_model_name}.onnx")

        # Save model info and increase model counter
        calibrater.single_node_model_path_map[f"{single_node_model_name}.onnx"] = (
            single_node_model_input_names,
            single_node_model_output_names,
        )
        model_counter += 1


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


def _init_calibrater_base(
    calibrater,
    model_path: str | Path,
    op_types_to_calibrate: Sequence[str] | None = None,
    augmented_model_path="augmented_model.onnx",
    symmetric=False,
    use_external_data_format=False,
    per_channel=False,
):
    """Initialize calibrater base class.

    :param model_path: ONNX model to calibrate. It should be a model file path
    :param op_types_to_calibrate: operator types to calibrate. By default, calibrate all the float32/float16 tensors.
    :param augmented_model_path: save augmented model to this path.
    :param symmetric: make range of tensor symmetric (central point is 0).
    :param use_external_data_format: use external data format to store model which size is >= 2Gb

    Modification: Additional members including single_node_model_path_map, providers, and trt_extra_plugin_lib_paths
        were added and initialized to support calibration per node feature.
    """
    if isinstance(model_path, str):
        calibrater.model = load_model_with_shape_infer(Path(model_path))
    elif isinstance(model_path, Path):
        calibrater.model = load_model_with_shape_infer(model_path)
    else:
        raise ValueError("model_path should be model path.")

    calibrater.op_types_to_calibrate = op_types_to_calibrate
    calibrater.augmented_model_path = augmented_model_path
    calibrater.symmetric = symmetric
    calibrater.use_external_data_format = use_external_data_format
    calibrater.per_channel = per_channel
    calibrater.augment_model = None
    calibrater.infer_session = None
    calibrater.execution_providers = []

    # Add single node calibration members
    calibrater.single_node_model_path_map = {}  # {path: ([inputs], [outputs])}
    calibrater.providers = []
    calibrater.trt_extra_plugin_lib_paths = None


def patch_ort_modules(calibrate_per_node: bool = False):
    """Patches the ORT modules."""
    logger.debug("Patching ORT modules")
    if calibrate_per_node:
        MinMaxCalibrater.augment_graph = _augment_graph_min_max_calibrater_single_node_calibration
        MinMaxCalibrater.collect_data = _collect_data_min_max_calibrater_single_node_calibration
        MinMaxCalibrater.compute_data = _compute_data_min_max_calibrater_single_node_calibration
        MinMaxCalibrater.merge_range = _merge_range_min_max_calibrater_single_node_calibration
        HistogramCalibrater.augment_graph = (
            _augment_graph_histogram_calibrater_single_node_calibration
        )
        HistogramCalibrater.collect_data = (
            _collect_data_histogram_calibrater_single_node_calibration
        )
        HistogramCollector.collect = _collect_histogram_collector_single_node_calibration
        HistogramCollector.collect_value = (
            _collect_value_histogram_collector_single_node_calibration
        )
    else:
        HistogramCollector.collect_value = _collect_value
        HistogramCollector.collect_absolute_value = _collect_absolute_value
        MinMaxCalibrater.compute_data = _compute_data_minmax_calibrator
        MinMaxCalibrater.collect_data = _collect_data_minmax_calibrator
        MinMaxCalibrater.merge_range = _merge_range_minmax_calibrator
        HistogramCalibrater.collect_data = _collect_data_histogram_calibrator

    calibrate.create_calibrator = _create_calibrator_with_extra_options
    CalibraterBase.create_inference_session = _create_inference_session_with_ep_config
    CalibraterBase.select_tensors_to_calibrate = _select_tensors_to_calibrate
    QDQQuantizer.check_opset_version = _check_opset_version
    BaseQuantizer.adjust_tensor_ranges = _adjust_tensor_ranges
    CalibraterBase.__init__ = _init_calibrater_base
