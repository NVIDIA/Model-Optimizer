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

import hashlib

import pytest
import torch
from _test_utils.import_helper import skip_if_no_trtexec
from _test_utils.torch.deploy.lib_test_models import (
    LeNet5,
    LeNet5Ooo,
    LeNet5TwoInputs,
    LeNet5TwoOutputs,
)
from _test_utils.torch.misc import compare_outputs
from _test_utils.torch.vision_models import get_vision_models, process_model_and_inputs

trt = pytest.importorskip("tensorrt")


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


@pytest.mark.manual(reason="slow test, run with --run-manual")
@pytest.mark.parametrize("get_model_and_input", vision_models.values(), ids=vision_models.keys())
def test_compile_and_profile_benchmark_models(get_model_and_input):
    model, args, kwargs = get_model_and_input(on_gpu=True)
    _compile_and_profile(model, args, kwargs)


def _compile_and_profile(model, args, kwargs):
    skip_if_no_trtexec()
    test_deployment = deployment
    if not hasattr(trt.BuilderFlag, "FP16"):
        test_deployment = {**deployment, "precision": "stronglyTyped"}
    device_model = compile(model, (*args, kwargs), test_deployment)
    assert isinstance(device_model.client, TRTLocalClient)
    latency, detailed_results = device_model.profile()
    device_model_outputs = device_model(*args, **kwargs)
    torch_model_outputs = model(*args, **kwargs)
    assert latency > 0.0
    assert detailed_results is not None
    assert device_model is not None
    compare_outputs(device_model_outputs, torch_model_outputs, rtol=1e-2, atol=1e-1)


@pytest.fixture(scope="module")
def trt_logger():
    return trt.Logger(trt.Logger.ERROR)


def _runtime_client(logger, *, dynamic=False, shape_input=False):
    builder = trt.Builder(logger)
    strongly_typed = getattr(trt.NetworkDefinitionCreationFlag, "STRONGLY_TYPED", None)
    network = builder.create_network(0 if strongly_typed is None else 1 << int(strongly_typed))
    config = builder.create_builder_config()
    config.builder_optimization_level = 0
    shape = (-1,) if shape_input else (-1, 3) if dynamic else (2, 3)
    value = network.add_input("x", trt.float32, shape)
    if shape_input:
        target = network.add_input("target", trt.int32, (2,))
        layer = network.add_shuffle(value)
        layer.set_input(1, target)
    else:
        layer = network.add_identity(value)
    layer.get_output(0).name = "y"
    network.mark_output(layer.get_output(0))
    if dynamic or shape_input:
        profile = builder.create_optimization_profile()
        if shape_input:
            profile.set_shape("x", (1,), (6,), (36,))
            profile.set_shape_input("target", (1, 1), (2, 3), (6, 6))
        else:
            profile.set_shape("x", (2, 3), (3, 3), (4, 3))
        config.add_optimization_profile(profile)
    plan = builder.build_serialized_network(network, config)
    assert plan is not None
    plan = bytes(plan)
    client = TRTLocalClient.__new__(TRTLocalClient)
    client.inference_sessions = {}
    client.trt_runtime = trt.Runtime(logger)
    client.stream = torch.cuda.Stream()
    return client, hashlib.sha256(plan).digest() + plan


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("explicit_shapes", [False, True])
def test_trt_session_shapes_and_padding(trt_logger, dynamic, explicit_shapes):
    client, plan = _runtime_client(trt_logger, dynamic=dynamic)
    for batch in [1, 2, 4, 1] if dynamic else [2, 1, 2]:
        value = torch.arange(batch * 3, dtype=torch.float32, device="cuda").reshape(batch, 3)
        buffer_batch = max(2, batch)
        io = {"x": [buffer_batch, 3], "y": [buffer_batch, 3]} if explicit_shapes else {}
        expected = torch.zeros((buffer_batch, 3), device="cuda")
        expected[:batch] = value
        (output,) = client.inference(plan, [value], io)
        torch.testing.assert_close(output, expected)
    assert len(client.inference_sessions) == 1


@pytest.mark.parametrize("shape_device", ["cpu", "cuda"])
@pytest.mark.parametrize("explicit_shapes", [False, True])
def test_trt_session_shape_values(trt_logger, shape_device, explicit_shapes):
    client, plan = _runtime_client(trt_logger, shape_input=True)
    value = torch.arange(6, dtype=torch.float32, device="cuda")
    for shape in [(2, 3), (3, 2), (1, 6)]:
        target = torch.tensor(shape, dtype=torch.int32, device=shape_device)
        io = {"x": [6], "target": [2], "y": list(shape)} if explicit_shapes else {}
        (output,) = client.inference(plan, [value, target], io)
        torch.testing.assert_close(output, value.reshape(shape))


@pytest.mark.parametrize("io_shapes", [{"x": [5, 3]}, {"y": [8, 3]}])
def test_trt_session_invalid_shapes(trt_logger, io_shapes):
    client, plan = _runtime_client(trt_logger, dynamic=True)
    value = torch.ones((2, 3), device="cuda")
    with pytest.raises(ValueError):
        client.inference(plan, [value], io_shapes)
    torch.testing.assert_close(client.inference(plan, [value])[0], value)


@pytest.mark.parametrize("missing_all", [False, True])
def test_trt_session_missing_input(trt_logger, missing_all):
    client, plan = _runtime_client(trt_logger, shape_input=True)
    with pytest.raises(ValueError, match="Expected 2 inputs"):
        client.inference(plan, [] if missing_all else [torch.ones(6, device="cuda")])
