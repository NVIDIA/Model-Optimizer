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

from unittest.mock import Mock

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper, numpy_helper

from modelopt.onnx import trt_utils
from modelopt.onnx.quantization import quantize


@pytest.fixture
def unavailable_tensorrt(monkeypatch):
    # Exercise the real loader and inspector without requiring CUDA on the CPU CI runner.
    trt = Mock()
    trt.Builder.side_effect = TypeError("pybind11::init(): factory function returned nullptr")
    trt.get_plugin_registry.return_value.all_creators = []
    monkeypatch.setattr(trt_utils, "TRT_PYTHON_AVAILABLE", True)
    monkeypatch.setattr(trt_utils, "trt", trt, raising=False)
    monkeypatch.setattr(trt_utils.platform, "system", lambda: "Linux")
    return trt


def _save_model(tmp_path, op_type="MatMul", domain="", nested=False, external=False):
    node = helper.make_node(op_type, ["x", "weight"], ["y"], domain=domain, name="operation")
    shape = ["batch", 4]
    output = helper.make_tensor_value_info("y", TensorProto.FLOAT, shape)
    graph = helper.make_graph([node], "body", [], [output])
    if nested:
        node = helper.make_node("If", ["condition"], ["y"], then_branch=graph, else_branch=graph)
    inputs = [helper.make_tensor_value_info("x", TensorProto.FLOAT, shape)]
    weight = numpy_helper.from_array(np.eye(4, dtype=np.float32), "weight")
    initializers = [weight]
    if nested:
        initializers.append(numpy_helper.from_array(np.array(True), "condition"))
    graph = helper.make_graph([node], "cpu_only", inputs, [output], initializer=initializers)
    imports = [helper.make_opsetid("", 13)]
    if domain:
        imports.append(helper.make_opsetid(domain, 1))
    model = helper.make_model(graph, opset_imports=imports, ir_version=10)
    path = tmp_path / "model.onnx"
    onnx.save_model(
        model,
        path,
        save_as_external_data=external,
        all_tensors_to_one_file=True,
        location="weights.data",
        size_threshold=0,
    )
    return path


@pytest.mark.parametrize("external", [False, True])
@pytest.mark.parametrize("override_shapes", [None, "x:2x4"])
@pytest.mark.parametrize("builder_error", [TypeError, RuntimeError])
def test_load_onnx_model_without_cuda(
    tmp_path, unavailable_tensorrt, external, override_shapes, builder_error
):
    unavailable_tensorrt.Builder.side_effect = builder_error("CUDA initialization failed")
    path = _save_model(tmp_path, external=external)
    loaded, has_custom, custom_ops, _, uses_external = trt_utils.load_onnx_model(
        str(path), override_shapes=override_shapes, use_external_data_format=external
    )

    assert not has_custom and not custom_ops
    assert uses_external == external
    session = ort.InferenceSession(loaded.SerializeToString(), providers=["CPUExecutionProvider"])
    values = np.arange(8, dtype=np.float32).reshape(2, 4)
    np.testing.assert_array_equal(session.run(None, {"x": values})[0], values)
    if override_shapes:
        assert loaded.graph.input[0].type.tensor_type.shape.dim[0].dim_value == 2


@pytest.mark.parametrize("domain", ["", "test.plugins"])
@pytest.mark.parametrize("nested", [False, True])
def test_load_onnx_model_rejects_registered_plugin(tmp_path, unavailable_tensorrt, domain, nested):
    path = _save_model(tmp_path, op_type="IdentityConv", domain=domain, nested=nested)
    parameter = onnx.defs.OpSchema.FormalParameter
    schema = onnx.defs.OpSchema(
        "IdentityConv",
        domain,
        1,
        inputs=[parameter("X", "T"), parameter("W", "T")],
        outputs=[parameter("Y", "T")],
        type_constraints=[("T", ["tensor(float)"], "")],
    )
    onnx.defs.register_schema(schema)
    try:
        onnx.checker.check_model(str(path), check_custom_domain=True)
        with pytest.raises(RuntimeError):
            trt_utils.load_onnx_model(str(path))
    finally:
        onnx.defs.deregister_schema(schema.name, schema.since_version, schema.domain)


@pytest.mark.parametrize("explicit_plugin", [False, True])
def test_load_onnx_model_requires_tensorrt_for_plugins(
    tmp_path, unavailable_tensorrt, explicit_plugin
):
    path = _save_model(tmp_path, op_type="IdentityConv" if not explicit_plugin else "MatMul")
    with pytest.raises(RuntimeError):
        trt_utils.load_onnx_model(str(path), trt_plugins=["plugin.so"] if explicit_plugin else None)


def test_load_onnx_model_preserves_parser_failure(tmp_path, unavailable_tensorrt):
    path = _save_model(tmp_path)
    unavailable_tensorrt.Builder.side_effect = None
    unavailable_tensorrt.OnnxParser.return_value.parse_from_file.return_value = False
    unavailable_tensorrt.OnnxParser.return_value.num_errors = 0
    with pytest.raises(Exception, match="Failed to parse ONNX file"):
        trt_utils.load_onnx_model(str(path))


@pytest.mark.parametrize("passes", [[], ["concat_elimination"]])
def test_cpu_quantization_without_cuda(tmp_path, unavailable_tensorrt, passes):
    rng = np.random.default_rng(6975097)
    weights = rng.normal(size=(16, 16)).astype(np.float32)
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["x", "weight"], ["y"], name="operation")],
        "cpu_quantization",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 16])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 16])],
        initializer=[numpy_helper.from_array(weights, "weight")],
    )
    path = tmp_path / "model.onnx"
    onnx.save(
        helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=10), path
    )
    output = tmp_path / "quantized.onnx"
    quantize(
        str(path),
        calibration_data=rng.normal(size=(128, 16)).astype(np.float32),
        calibration_method="max",
        calibration_eps=["cpu"],
        nodes_to_quantize=["operation"],
        high_precision_dtype="fp32",
        passes=passes,
        output_path=str(output),
    )
    model = onnx.load(output)
    onnx.checker.check_model(model)
    assert {"QuantizeLinear", "DequantizeLinear"} <= {node.op_type for node in model.graph.node}
    session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    values = rng.normal(size=(2, 16)).astype(np.float32)
    actual = session.run(None, {"x": values})[0]
    expected = values @ weights
    assert session.get_providers() == ["CPUExecutionProvider"]
    assert np.isfinite(actual).all()
    assert np.linalg.norm(actual - expected) / np.linalg.norm(expected) < 0.1
