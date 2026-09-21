# Adapted from https://github.com/microsoft/onnxruntime/blob/baeece44ba075009c6bfe95891a8c1b3d4571cb3/onnxruntime/python/tools/quantization/calibrate.py
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

"""Ordinary ONNX Runtime calibration patches."""

__all__ = []

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pynvml
from onnx import onnx_pb
from onnxruntime.quantization.calibrate import (
    CalibrationDataReader,
    CalibrationMethod,
    HistogramCollector,
    TensorData,
    TensorsData,
)
from tqdm import tqdm

from modelopt.onnx.logging_config import logger
from modelopt.onnx.quantization.ort_session import load_model_with_shape_infer


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


def _prepare_histogram_data(histogram_collector, tensor, data_arr):
    """Use FP32 for histogram math while remembering the source dtype."""
    if data_arr.dtype != np.float16:
        return data_arr

    original_dtypes = getattr(histogram_collector, "_modelopt_original_dtypes", {})
    original_dtypes[tensor] = data_arr.dtype
    histogram_collector._modelopt_original_dtypes = original_dtypes
    return data_arr.astype(np.float32)


def _restore_histogram_calibration_dtypes(histogram_collector, tensors_range):
    """Restore source dtypes at the calibration-to-quantization boundary."""
    original_dtypes = getattr(histogram_collector, "_modelopt_original_dtypes", {})
    for tensor, dtype in original_dtypes.items():
        if tensor not in tensors_range:
            continue
        tensor_data = tensors_range[tensor]
        dtype_limits = np.finfo(dtype)
        for attribute in ("lowest", "highest", "avg", "std"):
            if hasattr(tensor_data, attribute):
                value = np.clip(getattr(tensor_data, attribute), dtype_limits.min, dtype_limits.max)
                setattr(tensor_data, attribute, np.asarray(value, dtype=dtype))


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

        concat_data_arr = _prepare_histogram_data(histogram_collector, tensor, concat_data_arr)
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
            range_max = float(threshold)
            hist, hist_edges = np.histogram(
                data_arr, histogram_collector.num_bins, range=(-range_max, range_max)
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
