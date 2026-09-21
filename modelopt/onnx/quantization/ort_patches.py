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

"""Composition of ModelOpt's ONNX Runtime monkey patches."""

__all__ = []

from onnxruntime.quantization import calibrate, qdq_quantizer
from onnxruntime.quantization.base_quantizer import BaseQuantizer
from onnxruntime.quantization.calibrate import (
    CalibraterBase,
    HistogramCalibrater,
    HistogramCollector,
    MinMaxCalibrater,
)
from onnxruntime.quantization.qdq_quantizer import QDQQuantizer

from modelopt.onnx.logging_config import logger
from modelopt.onnx.quantization.ort_calibration import (
    _collect_absolute_value,
    _collect_data_histogram_calibrator,
    _collect_data_minmax_calibrator,
    _collect_value,
    _compute_data_minmax_calibrator,
    _init_calibrater_base,
    _merge_range_minmax_calibrator,
    _select_tensors_to_calibrate,
)
from modelopt.onnx.quantization.ort_calibration_per_node import (
    _augment_graph_histogram_calibrater_single_node_calibration,
    _augment_graph_min_max_calibrater_single_node_calibration,
    _collect_data_histogram_calibrater_single_node_calibration,
    _collect_data_min_max_calibrater_single_node_calibration,
    _collect_histogram_collector_single_node_calibration,
    _collect_value_histogram_collector_single_node_calibration,
    _compute_data_min_max_calibrater_single_node_calibration,
    _merge_range_min_max_calibrater_single_node_calibration,
)
from modelopt.onnx.quantization.ort_quantization import (
    _adjust_tensor_ranges,
    _check_opset_version,
    _compute_scale_zp,
    _create_calibrator_with_extra_options,
)
from modelopt.onnx.quantization.ort_session import _create_inference_session_with_ep_config


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
    qdq_quantizer.compute_scale_zp = _compute_scale_zp
    CalibraterBase.__init__ = _init_calibrater_base
