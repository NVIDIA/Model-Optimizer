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
    device_model = compile(model, (*args, kwargs), deployment)
    assert isinstance(device_model.client, TRTLocalClient)
    latency, detailed_results = device_model.profile()
    device_model_outputs = device_model(*args, **kwargs)
    torch_model_outputs = model(*args, **kwargs)
    assert latency > 0.0
    assert detailed_results is not None
    assert device_model is not None
    compare_outputs(device_model_outputs, torch_model_outputs, rtol=1e-2, atol=1e-1)
