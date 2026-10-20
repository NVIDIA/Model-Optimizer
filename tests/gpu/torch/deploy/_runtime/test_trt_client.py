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

skip_if_no_tensorrt()
skip_if_no_trtexec()


import tensorrt as trt

from modelopt.torch._deploy._runtime.tensorrt.tensorrt_utils import prepend_hash_to_bytes
from modelopt.torch._deploy._runtime.trt_client import TRTLocalClient
from modelopt.torch._deploy.compilation import compile

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


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("provide_shapes", [False, True])
def test_trt_session_shape_inference(dynamic, provide_shapes):
    """Infer outputs with name-based APIs, retaining fixed-buffer batch padding."""
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    input_tensor = network.add_input("input", trt.float32, (-1, 2) if dynamic else (2, 2))
    output = network.add_identity(input_tensor).get_output(0)
    output.name = "output"
    network.mark_output(output)
    config = builder.create_builder_config()
    if dynamic:
        profile = builder.create_optimization_profile()
        assert profile.set_shape("input", (2, 2), (3, 2), (4, 2))
        config.add_optimization_profile(profile)
    engine = builder.build_serialized_network(network, config)
    assert engine is not None
    compiled_model = prepend_hash_to_bytes(bytes(engine))
    client = TRTLocalClient(deployment)
    io_shapes = {"input": [2, 2], "output": [2, 2]} if provide_shapes else {}
    for batch in (2, 1, 2):
        inputs = torch.arange(batch * 2, dtype=torch.float32, device="cuda").reshape(batch, 2)
        expected = torch.zeros((2, 2), device="cuda")
        expected[:batch] = inputs
        (actual,) = client.inference(compiled_model, [inputs], io_shapes)
        torch.testing.assert_close(actual, expected)
    with pytest.raises(ValueError):
        client.inference(compiled_model, [], io_shapes)
    if dynamic:
        with pytest.raises(ValueError):
            TRTLocalClient.TRTSession(
                engine, client.trt_runtime, client.stream, "cuda", {"input": [5, 2]}
            )


def test_trt_session_shape_tensor_values():
    """Cached sessions must infer output shapes from each call's shape-tensor values."""
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    data = network.add_input("data", trt.float32, (6,))
    shape = network.add_input("shape", trt.int32, (2,))
    shape.location = trt.TensorLocation.HOST
    shuffle = network.add_shuffle(data)
    shuffle.set_input(1, shape)
    output = shuffle.get_output(0)
    output.name = "output"
    network.mark_output(output)
    config = builder.create_builder_config()
    profile = builder.create_optimization_profile()
    assert profile.set_shape_input("shape", (1, 2), (2, 3), (3, 6))
    config.add_optimization_profile(profile)
    engine = builder.build_serialized_network(network, config)
    assert engine is not None
    compiled_model = prepend_hash_to_bytes(bytes(engine))
    client = TRTLocalClient(deployment)
    for dims in ((2, 3), (3, 2), (1, 6)):
        inputs = torch.arange(6, dtype=torch.float32, device="cuda")
        (actual,) = client.inference(
            compiled_model, [inputs, torch.tensor(dims, dtype=torch.int32)]
        )
        torch.testing.assert_close(actual, inputs.reshape(dims))
    assert len(client.inference_sessions) == 1


@pytest.mark.manual(reason="slow test, run with --run-manual")
@pytest.mark.parametrize("get_model_and_input", vision_models.values(), ids=vision_models.keys())
def test_compile_and_profile_benchmark_models(get_model_and_input):
    model, args, kwargs = get_model_and_input(on_gpu=True)
    _compile_and_profile(model, args, kwargs)


def _compile_and_profile(model, args, kwargs):
    device_model = compile(model, (*args, kwargs), deployment)
    assert isinstance(device_model.client, TRTLocalClient)
    latency, detailed_results = device_model.profile()
    device_model_outputs = device_model(*args, **kwargs)
    torch_model_outputs = model(*args, **kwargs)
    assert latency > 0.0
    assert detailed_results is not None
    assert device_model is not None
    compare_outputs(device_model_outputs, torch_model_outputs, rtol=1e-2, atol=1e-1)
