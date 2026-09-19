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

"""Per-node ONNX Runtime calibration patches."""

__all__ = []

import gc
import uuid

import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization.calibrate import (
    CalibrationDataReader,
    CalibrationMethod,
    HistogramCollector,
    TensorData,
    TensorsData,
)
from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference
from tqdm import tqdm

from modelopt.onnx.logging_config import logger
from modelopt.onnx.quantization.ort_calibration import _prepare_histogram_data


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
        data_arr = _prepare_histogram_data(histogram_collector, tensor, data_arr)
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
