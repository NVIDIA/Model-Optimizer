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

import importlib
import importlib.util
import inspect

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

import modelopt.onnx.quantization as moq

_CALIBRATION_DATA = np.array(
    [
        [-1.0, -0.5, 0.5, 1.0],
        [-1.0, -0.5, 0.5, 100.0],
    ],
    dtype=np.float32,
)


def _make_matmul_model() -> onnx.ModelProto:
    weight = np.array(
        [
            [-2.0, -1.5, -1.0],
            [-0.5, 0.0, 0.5],
            [1.0, 1.5, 2.0],
            [2.5, 3.0, 3.5],
        ],
        dtype=np.float32,
    )
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["input", "weight"], ["output"])],
        "calibrated_matmul",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3])],
        [numpy_helper.from_array(weight, "weight")],
    )
    return helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 19)],
        ir_version=10,
    )


def _producer_by_tensor(model: onnx.ModelProto) -> dict[str, onnx.NodeProto]:
    return {output: node for node in model.graph.node for output in node.output}


def _initializer_arrays(model: onnx.ModelProto) -> dict[str, np.ndarray]:
    return {
        initializer.name: numpy_helper.to_array(initializer)
        for initializer in model.graph.initializer
    }


@pytest.mark.parametrize(
    ("quantize_mode", "calibration_method", "quantized_type", "activation_scale", "weight_scales"),
    [
        pytest.param(
            "int8",
            "entropy",
            TensorProto.INT8,
            0.007997047156095505,
            [0.019685039296746254, 0.023622047156095505, 0.027559055015444756],
            id="int8-entropy",
        ),
        pytest.param(
            "int8",
            "max",
            TensorProto.INT8,
            0.787401556968689,
            [0.019685039296746254, 0.023622047156095505, 0.027559055015444756],
            id="int8-max",
        ),
        pytest.param(
            "fp8",
            "entropy",
            TensorProto.FLOAT8E4M3FN,
            0.028210056945681572,
            [0.06944013386964798, 0.08332816511392593, 0.09721619635820389],
            id="fp8-entropy",
        ),
        pytest.param(
            "fp8",
            "max",
            TensorProto.FLOAT8E4M3FN,
            2.7776055335998535,
            [0.06944013386964798, 0.08332816511392593, 0.09721619635820389],
            id="fp8-max",
        ),
    ],
)
def test_explicit_calibrated_quantization_graph_contract(
    tmp_path,
    quantize_mode,
    calibration_method,
    quantized_type,
    activation_scale,
    weight_scales,
):
    input_path = tmp_path / "matmul.onnx"
    output_path = tmp_path / "matmul.quant.onnx"
    onnx.save_model(_make_matmul_model(), input_path)

    moq.quantize(
        str(input_path),
        output_path=str(output_path),
        quantize_mode=quantize_mode,
        calibration_method=calibration_method,
        calibration_data={"input": _CALIBRATION_DATA},
        calibration_eps=["cpu"],
        high_precision_dtype="fp32",
        passes=[],
        enable_gemv_detection_for_trt=False,
    )

    model = onnx.load(output_path)
    producers = _producer_by_tensor(model)
    initializers = {initializer.name: initializer for initializer in model.graph.initializer}
    initializer_arrays = _initializer_arrays(model)
    matmul = next(node for node in model.graph.node if node.op_type == "MatMul")
    activation_dq, weight_dq = (producers[tensor] for tensor in matmul.input)
    activation_q = producers[activation_dq.input[0]]
    weight_q = producers[weight_dq.input[0]]

    assert sum(node.op_type == "QuantizeLinear" for node in model.graph.node) == 2
    assert sum(node.op_type == "DequantizeLinear" for node in model.graph.node) == 2
    assert activation_q.op_type == weight_q.op_type == "QuantizeLinear"
    assert activation_q.input[0] == model.graph.input[0].name
    assert weight_q.input[0] == "weight"
    assert activation_dq.op_type == weight_dq.op_type == "DequantizeLinear"
    assert matmul.output == [model.graph.output[0].name]
    weight_q_attributes = {
        attribute.name: helper.get_attribute_value(attribute) for attribute in weight_q.attribute
    }
    weight_dq_attributes = {
        attribute.name: helper.get_attribute_value(attribute) for attribute in weight_dq.attribute
    }
    assert weight_q_attributes == weight_dq_attributes == {"axis": 1}
    assert initializers[activation_q.input[2]].data_type == quantized_type
    assert initializers[weight_q.input[2]].data_type == quantized_type
    assert activation_q.input[1:] == activation_dq.input[1:]
    assert weight_q.input[1:] == weight_dq.input[1:]
    np.testing.assert_array_equal(
        initializer_arrays[activation_q.input[2]],
        np.zeros_like(initializer_arrays[activation_q.input[2]]),
    )
    np.testing.assert_array_equal(
        initializer_arrays[weight_q.input[2]],
        np.zeros_like(initializer_arrays[weight_q.input[2]]),
    )
    np.testing.assert_array_equal(
        initializer_arrays[activation_q.input[1]], np.float32(activation_scale)
    )
    np.testing.assert_array_equal(
        initializer_arrays[weight_q.input[1]], np.asarray(weight_scales, dtype=np.float32)
    )
    assert model.graph.input[0].type.tensor_type.elem_type == TensorProto.FLOAT
    assert model.graph.output[0].type.tensor_type.elem_type == TensorProto.FLOAT
    assert [(opset.domain, opset.version) for opset in model.opset_import] == [("", 19)]


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="Omitted FP8 calibration is expected to default to max",
)
def test_future_fp8_omitted_method_defaults_to_max(tmp_path, monkeypatch):
    quantize_module = importlib.import_module("modelopt.onnx.quantization.quantize")
    input_path = tmp_path / "matmul.onnx"
    onnx.save_model(_make_matmul_model(), input_path)
    captured = {}

    def capture_calibration_method(**kwargs):
        captured["calibration_method"] = kwargs["calibration_method"]

    monkeypatch.setattr(quantize_module, "quantize_fp8", capture_calibration_method)

    moq.quantize(
        str(input_path),
        quantize_mode="fp8",
        calibration_data={"input": _CALIBRATION_DATA},
        calibration_eps=["cpu"],
        high_precision_dtype="fp32",
        passes=[],
        enable_gemv_detection_for_trt=False,
    )

    assert captured == {"calibration_method": "max"}


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="Calibrated quantization is expected to require an explicit source",
)
def test_future_calibrated_quantization_requires_an_explicit_source(tmp_path, monkeypatch):
    quantize_module = importlib.import_module("modelopt.onnx.quantization.quantize")
    input_path = tmp_path / "matmul.onnx"
    onnx.save_model(_make_matmul_model(), input_path)

    def reject_implicit_random_data(*args, **kwargs):
        raise AssertionError("an implicit random calibration source was created")

    monkeypatch.setattr(quantize_module, "RandomDataProvider", reject_implicit_random_data)

    try:
        moq.quantize(
            str(input_path),
            quantize_mode="int8",
            calibration_eps=["cpu"],
            high_precision_dtype="fp32",
            passes=[],
            enable_gemv_detection_for_trt=False,
        )
    except ValueError:
        return

    raise AssertionError("quantization accepted a missing calibration source")


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="Calibrated quantization is expected to reject multiple sources",
)
def test_future_calibrated_quantization_rejects_two_sources(tmp_path, monkeypatch):
    quantize_module = importlib.import_module("modelopt.onnx.quantization.quantize")
    input_path = tmp_path / "matmul.onnx"
    onnx.save_model(_make_matmul_model(), input_path)
    dispatched = False

    class CalibrationReader:
        def get_next(self):
            return None

    def capture_int8_dispatch(**kwargs):
        nonlocal dispatched
        dispatched = True

    monkeypatch.setattr(quantize_module, "quantize_int8", capture_int8_dispatch)

    try:
        moq.quantize(
            str(input_path),
            quantize_mode="int8",
            calibration_data={"input": _CALIBRATION_DATA},
            calibration_data_reader=CalibrationReader(),
            calibration_eps=["cpu"],
            high_precision_dtype="fp32",
            passes=[],
            enable_gemv_detection_for_trt=False,
        )
    except ValueError:
        return

    assert not dispatched


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="Top-level quantization is expected to accept only exact mode tokens",
)
def test_future_top_level_mode_tokens_are_exact(tmp_path, monkeypatch):
    quantize_module = importlib.import_module("modelopt.onnx.quantization.quantize")
    input_path = tmp_path / "matmul.onnx"
    onnx.save_model(_make_matmul_model(), input_path)
    dispatched = False

    def capture_int4_dispatch(**kwargs):
        nonlocal dispatched
        dispatched = True

    monkeypatch.setattr(quantize_module, "quantize_int4", capture_int4_dispatch)

    try:
        moq.quantize(
            str(input_path),
            quantize_mode="int4_awq",
            calibration_data={"input": _CALIBRATION_DATA},
            calibration_eps=["cpu"],
            passes=[],
        )
    except (RuntimeError, ValueError):
        return

    assert not dispatched


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="Calibration-cache input is expected to be absent from the Python API",
)
def test_future_python_api_removes_calibration_cache_input():
    assert "calibration_cache_path" not in inspect.signature(moq.quantize).parameters


def test_legacy_graph_utils_module_is_removed():
    assert importlib.util.find_spec("modelopt.onnx.quantization.graph_utils") is None


@pytest.mark.parametrize(
    ("module_name", "removed_symbol"),
    [
        pytest.param("modelopt.onnx.quantization.int8", "quantize", id="int8-mode-function"),
        pytest.param("modelopt.onnx.quantization.fp8", "quantize", id="fp8-mode-function"),
        pytest.param("modelopt.onnx.quantization.ort_patching", None, id="ort-patching-module"),
        pytest.param("modelopt.onnx.quantization.qdq_utils", None, id="qdq-utils-module"),
        pytest.param(
            "modelopt.onnx.quantization.qdq_utils",
            "quantize_weights_to_int4",
            id="int4-exporter-helper",
        ),
        pytest.param(
            "modelopt.onnx.quantization.qdq_utils",
            "quantize_weights_to_mxfp8",
            id="mxfp8-exporter-helper",
        ),
        pytest.param(
            "modelopt.onnx.quantization.qdq_utils",
            "fp4qdq_to_2dq",
            id="fp4-to-2dq-exporter-helper",
        ),
    ],
)
@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="Legacy calibrated implementation imports are expected to be unavailable",
)
def test_future_legacy_calibrated_imports_are_removed(module_name, removed_symbol):
    module_spec = importlib.util.find_spec(module_name)
    if removed_symbol is None:
        assert module_spec is None
    elif module_spec is not None:
        module = importlib.import_module(module_name)
        assert not hasattr(module, removed_symbol)


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="AutoTune is expected to export get_quantized_tensors",
)
def test_future_autotune_exports_get_quantized_tensors():
    autotune = importlib.import_module("modelopt.onnx.quantization.autotune")
    assert hasattr(autotune, "get_quantized_tensors")
