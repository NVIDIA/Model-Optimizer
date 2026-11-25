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

import numpy as np
import onnx
import pytest
import torch
from _test_utils.import_helper import skip_if_no_tensorrt, skip_if_no_trtexec
from _test_utils.torch.deploy.lib_test_models import (
    LeNet5,
    LeNet5Ooo,
    LeNet5TwoInputs,
    LeNet5TwoOutputs,
)
from _test_utils.torch.misc import compare_outputs
from _test_utils.torch.vision_models import get_vision_models, process_model_and_inputs

trt = pytest.importorskip("tensorrt")


from modelopt.torch._deploy._runtime.tensorrt import engine_builder
from modelopt.torch._deploy._runtime.tensorrt.constants import SHA_256_HASH_LENGTH
from modelopt.torch._deploy._runtime.trt_client import TRTLocalClient
from modelopt.torch._deploy.compilation import compile
from modelopt.torch._deploy.utils import OnnxBytes

deployment = {
    "runtime": "TRT",
    "accelerator": "GPU",
    "precision": "fp16",
    "onnx_opset": "20",
}

vision_models = get_vision_models()


@pytest.mark.parametrize(
    ("model", "args", "kwargs"),
    [
        (LeNet5(), (LeNet5.gen_input(),), {}),
        (
            LeNet5TwoInputs(),
            (LeNet5TwoInputs.gen_input(0), LeNet5TwoInputs.gen_input(1)),
            {},
        ),
        (LeNet5TwoOutputs(), (LeNet5TwoOutputs.gen_input(),), {}),
        (LeNet5Ooo(), (LeNet5Ooo.gen_input(0), LeNet5Ooo.gen_input(1)), {}),
    ],
)
def test_compile_and_profile_lenet5(model, args, kwargs):
    model, args, kwargs = process_model_and_inputs(model, args, kwargs, on_gpu=True)
    _compile_and_profile(model, args, kwargs)


@pytest.mark.manual(reason="slow test, run with --run-manual")
@pytest.mark.parametrize("get_model_and_input", vision_models.values(), ids=vision_models.keys())
def test_compile_and_profile_benchmark_models(get_model_and_input):
    model, args, kwargs = get_model_and_input(on_gpu=True)
    _compile_and_profile(model, args, kwargs)


def _compile_and_profile(model, args, kwargs):
    skip_if_no_tensorrt()
    skip_if_no_trtexec()
    device_model = compile(model, (*args, kwargs), deployment)
    assert isinstance(device_model.client, TRTLocalClient)
    latency, detailed_results = device_model.profile()
    device_model_outputs = device_model(*args, **kwargs)
    torch_model_outputs = model(*args, **kwargs)
    assert latency > 0.0
    assert detailed_results is not None
    assert device_model is not None
    compare_outputs(device_model_outputs, torch_model_outputs, rtol=1e-2, atol=1e-1)


@pytest.mark.parametrize("precision", ["fp16", "bf16"])
def test_trt11_precision_graph_and_inference(tmp_path, monkeypatch, precision):
    if int(trt.__version__.split(".", 1)[0]) < 11:
        pytest.skip("TensorRT 11 precision migration")
    if not torch.cuda.is_available():
        pytest.skip("A CUDA device is required")
    if precision == "bf16" and not torch.cuda.is_bf16_supported():
        pytest.skip("BF16 requires a supported GPU")
    skip_if_no_trtexec()
    model, args, kwargs = process_model_and_inputs(
        LeNet5(), (LeNet5.gen_input(batch_size=2),), {}, on_gpu=True
    )
    run_trtexec = engine_builder._run_trtexec_with_logging
    observed = []

    def inspect_and_compile(arguments, cwd=None):
        path = next(arg.split("=", 1)[1] for arg in arguments if arg.startswith("--onnx="))
        graph = onnx.load(path)
        expected_type = {"fp16": onnx.TensorProto.FLOAT16, "bf16": onnx.TensorProto.BFLOAT16}[
            precision
        ]
        floating_weights = [
            t
            for t in graph.graph.initializer
            if t.data_type
            in (onnx.TensorProto.FLOAT, onnx.TensorProto.FLOAT16, onnx.TensorProto.BFLOAT16)
        ]
        assert floating_weights and all(t.data_type == expected_type for t in floating_weights)
        assert all(
            value.type.tensor_type.elem_type == onnx.TensorProto.FLOAT
            for value in list(graph.graph.input) + list(graph.graph.output)
        )
        observed.append(expected_type)
        return run_trtexec(arguments, cwd=cwd)

    # Observe the temporary graph, then delegate to the real native compiler.
    monkeypatch.setattr(engine_builder, "_run_trtexec_with_logging", inspect_and_compile)
    device = compile(
        model,
        (*args, kwargs),
        {**deployment, "precision": precision},
        compilation_args={"engine_path": tmp_path / "model.engine"},
    )
    actual = device(*args, **kwargs)
    with torch.no_grad():
        expected = model(*args, **kwargs)
    assert observed
    compare_outputs(actual, expected, rtol=1e-2, atol=1e-1)


@pytest.mark.parametrize("precision", ["fp16", "bf16"])
@pytest.mark.parametrize("layout", ["unnamed", "input_output", "initializer_output"])
def test_trt11_public_outputs(tmp_path, monkeypatch, precision, layout):
    if int(trt.__version__.split(".", 1)[0]) < 11:
        pytest.skip("TensorRT 11 precision migration")
    skip_if_no_trtexec()
    helper, tensor = onnx.helper, onnx.TensorProto
    weight = np.eye(4, dtype=np.float32)
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["x", "weight"], ["y"])],
        "public_outputs",
        [helper.make_tensor_value_info("x", tensor.FLOAT, [1, 4])],
        [helper.make_tensor_value_info("y", tensor.FLOAT, [1, 4])],
        initializer=[onnx.numpy_helper.from_array(weight, "weight")],
    )
    if layout == "input_output":
        graph.output.append(graph.input[0])
    elif layout == "initializer_output":
        graph.initializer.append(onnx.numpy_helper.from_array(weight, "constant"))
        graph.output.append(helper.make_tensor_value_info("constant", tensor.FLOAT, [4, 4]))
    path = tmp_path / "model.onnx"
    onnx.save_model(
        helper.make_model(graph, opset_imports=[helper.make_opsetid("", 22)], ir_version=10), path
    )
    run_trtexec = engine_builder._run_trtexec_with_logging

    def inspect_and_compile(arguments, cwd=None):
        converted_path = next(
            arg.split("=", 1)[1] for arg in arguments if arg.startswith("--onnx=")
        )
        converted = onnx.load(converted_path)
        assert (
            next(t for t in converted.graph.initializer if t.name == "weight").data_type
            == {
                "fp16": tensor.FLOAT16,
                "bf16": tensor.BFLOAT16,
            }[precision]
        )
        return run_trtexec(arguments, cwd=cwd)

    monkeypatch.setattr(engine_builder, "_run_trtexec_with_logging", inspect_and_compile)
    engine, log = engine_builder.build_engine(
        OnnxBytes(str(path)), trt_mode=precision, output_dir=tmp_path / "artifacts"
    )
    assert engine is not None, log
    runtime = trt.Runtime(trt.Logger(trt.Logger.ERROR))
    shapes = {
        value.name: [dim.dim_value for dim in value.type.tensor_type.shape.dim]
        for value in (*graph.input, *graph.output)
    }
    session = TRTLocalClient.TRTSession(
        engine[SHA_256_HASH_LENGTH:], runtime, torch.cuda.Stream(), "cuda", shapes
    )
    x = torch.arange(4, device="cuda", dtype=torch.float32).reshape(1, 4)
    outputs = session.run([x])
    expected = [x]
    if layout == "input_output":
        expected.append(x)
    elif layout == "initializer_output":
        expected.append(torch.tensor(weight, device="cuda"))
    assert len(outputs) == len(expected)
    for actual, reference in zip(outputs, expected):
        torch.testing.assert_close(actual, reference)


@pytest.mark.parametrize("input_kind", ["static", "static_inferred", "dynamic", "shape_tensor"])
def test_trt_session_inference(input_kind):
    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    config = builder.create_builder_config()
    dynamic = input_kind == "dynamic"
    x = network.add_input("input", trt.float32, (-1, 4) if dynamic else (1, 4))
    batch = 2 if dynamic else 1
    inputs = [torch.arange(batch * 4, dtype=torch.float32, device="cuda").reshape(batch, 4)]
    io_shapes = {"input": [batch, 4], "output": [batch, 4]}
    if dynamic:
        profile = builder.create_optimization_profile()
        profile.set_shape("input", (1, 4), (2, 4), (4, 4))
        config.add_optimization_profile(profile)
    if input_kind == "shape_tensor":
        shape = network.add_input("shape", trt.int32, (2,))
        layer = network.add_shuffle(x)
        layer.set_input(1, shape)
        profile = builder.create_optimization_profile()
        profile.set_shape_input("shape", (1, 1), (2, 2), (4, 4))
        config.add_optimization_profile(profile)
        inputs.append(torch.tensor([2, 2], dtype=torch.int32))
        io_shapes.update({"shape": [2], "output": [2, 2]})
    else:
        layer = network.add_identity(x)
    output = layer.get_output(0)
    output.name = "output"
    network.mark_output(output)
    serialized = builder.build_serialized_network(network, config)
    assert serialized is not None
    runtime = trt.Runtime(logger)
    session = TRTLocalClient.TRTSession(
        bytes(serialized),
        runtime,
        torch.cuda.Stream(),
        "cuda",
        {} if input_kind == "static_inferred" else io_shapes,
    )
    result = session.run(inputs)[0]
    torch.testing.assert_close(result, inputs[0].reshape(io_shapes["output"]))
    with pytest.raises(AssertionError, match="input tensors"):
        session.run(inputs[:-1])
    if input_kind == "shape_tensor":
        inputs[1] = torch.tensor([3, 3], dtype=torch.int32)
        with pytest.raises(RuntimeError, match="infer_shapes"):
            session.run(inputs)
    if dynamic:
        with pytest.raises(AssertionError, match="Invalid shape for input tensor input"):
            TRTLocalClient.TRTSession(
                bytes(serialized),
                runtime,
                torch.cuda.Stream(),
                "cuda",
                {"input": [5, 4], "output": [5, 4]},
            )
