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

"""Tests for ONNX Runtime histogram quantization patches."""

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper, numpy_helper
from onnxruntime.quantization.calibrate import (
    CalibrationDataReader,
    CalibrationMethod,
    HistogramCollector,
    TensorData,
    TensorsData,
)

from modelopt.onnx.quantization.ort_patching import (
    _collect_value,
    _collect_value_histogram_collector_single_node_calibration,
    _compute_scale_zp,
    _prepare_histogram_data,
    _quantize_static,
    _restore_histogram_calibration_dtypes,
    patch_ort_modules,
)


def test_compute_scale_zp_fp16_overflow_fallback():
    zero_point, scale = _compute_scale_zp(
        np.array(-65504, dtype=np.float16),
        np.array(65504, dtype=np.float16),
        np.array(-128, dtype=np.int8),
        np.array(127, dtype=np.int8),
        symmetric=True,
    )

    assert zero_point.dtype == np.int8
    assert zero_point == 0
    assert scale.dtype == np.float16
    assert scale == np.float16(514)


def test_quantize_static_fp16_high_range_scale(tmp_path):
    class HighRangeDataReader(CalibrationDataReader):
        def __init__(self):
            self.rewind()

        def get_next(self):
            return next(self.data, None)

        def rewind(self):
            values = np.array([[-65504, 65504, -32752, 32752]], dtype=np.float16)
            self.data = iter([{"input": values}])

    model_path = tmp_path / "model.onnx"
    output_path = tmp_path / "model.quant.onnx"
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["input", "weight"], ["output"], name="matmul")],
        "fp16_high_range",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT16, [1, 4])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT16, [1, 4])],
        [numpy_helper.from_array(np.eye(4, dtype=np.float16), name="weight")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)])
    model.ir_version = min(model.ir_version, 10)
    onnx.save(model, model_path)

    patch_ort_modules(False)
    _quantize_static(
        model_path,
        output_path,
        HighRangeDataReader(),
        nodes_to_quantize=["matmul"],
        op_types_to_quantize=["MatMul"],
        calibrate_method=CalibrationMethod.Entropy,
        extra_options={
            "ExecutionProviders": ["CPUExecutionProvider"],
            "ActivationSymmetric": True,
            "AddQDQPairToWeight": True,
        },
    )

    quantized_model = onnx.load(output_path)
    initializers = {
        initializer.name: initializer for initializer in quantized_model.graph.initializer
    }
    scale_initializers = [
        initializers[node.input[1]]
        for node in quantized_model.graph.node
        if node.op_type in {"QuantizeLinear", "DequantizeLinear"} and node.input[1] in initializers
    ]
    assert scale_initializers
    assert {initializer.data_type for initializer in scale_initializers} == {TensorProto.FLOAT16}
    assert all(
        np.isfinite(numpy_helper.to_array(initializer)).all() for initializer in scale_initializers
    )
    ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])


@pytest.mark.parametrize(
    "collect_value",
    [
        _collect_value,
        _collect_value_histogram_collector_single_node_calibration,
    ],
)
def test_collect_value_fp16_narrow_range(collect_value):
    collector = HistogramCollector(
        method="entropy",
        symmetric=False,
        num_bins=128,
        num_quantized_bins=128,
        percentile=None,
        scenario="same",
    )
    activations = np.zeros(1000, dtype=np.float16)
    for activation_max in (1e-6, 1e-6, 2e-6):
        activations[0] = np.float16(activation_max)
        collect_value(collector, {"narrow_fp16_tensor": [activations]})

    hist, edges, _, _, threshold = collector.histogram_dict["narrow_fp16_tensor"]
    assert hist.sum() == 3 * activations.size
    assert len(hist) > collector.num_bins
    assert edges.dtype == np.float32
    assert np.all(np.diff(edges) > 0), "fp16 bin edges are not strictly increasing"
    assert np.asarray(threshold).dtype.itemsize >= np.dtype(np.float32).itemsize

    tensors_range = TensorsData(CalibrationMethod.Entropy, collector.compute_collection_result())
    _restore_histogram_calibration_dtypes(collector, tensors_range)
    tensor_range = tensors_range["narrow_fp16_tensor"]
    assert tensor_range.lowest.dtype == np.float16
    assert tensor_range.highest.dtype == np.float16
    assert tensor_range.bins.dtype == np.float32


def test_restore_histogram_calibration_dtypes_clamps_fp16():
    collector = HistogramCollector(
        method="distribution",
        symmetric=False,
        num_bins=512,
        num_quantized_bins=128,
        percentile=None,
        scenario="same",
    )
    _prepare_histogram_data(collector, "tensor", np.array([], dtype=np.float16))
    _prepare_histogram_data(collector, "missing_tensor", np.array([], dtype=np.float16))

    fp32_max = np.finfo(np.float32).max
    tensor_data = TensorData(
        lowest=np.float32(-fp32_max),
        highest=np.float32(fp32_max),
        avg=np.float32(fp32_max),
        std=np.float32(fp32_max),
        hist=np.array([1]),
        hist_edges=np.array([-1, 1], dtype=np.float32),
    )
    tensors_range = TensorsData(CalibrationMethod.Distribution, {"tensor": tensor_data})

    _restore_histogram_calibration_dtypes(collector, tensors_range)

    fp16_limits = np.finfo(np.float16)
    tensor_range = tensors_range["tensor"]
    assert tensor_range.lowest == fp16_limits.min
    assert tensor_range.highest == fp16_limits.max
    assert tensor_range.avg == fp16_limits.max
    assert tensor_range.std == fp16_limits.max
    assert tensor_range.hist_edges.dtype == np.float32
    assert "missing_tensor" not in tensors_range
