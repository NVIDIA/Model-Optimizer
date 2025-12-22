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

import inspect
import io

import onnxruntime
import torch

import modelopt.torch.quantization as mtq

from .models import SimpleConv, SimpleConvLinear, SimpleLinear

TEST_MODELS = {SimpleLinear, SimpleConv, SimpleConvLinear}


def onnx_export_tester(model, device, num_bits, per_channel_quantization, constant_folding, dtype):
    axis = 0 if per_channel_quantization else None
    config = {
        "quant_cfg": {
            "*weight_quantizer": {"num_bits": num_bits, "axis": axis},
            "*input_quantizer": {"num_bits": num_bits},
            "default": {"enable": False},
        },
        "algorithm": "max",
    }

    model.eval()
    model = model.to(device).to(dtype)
    dummy_input = model.get_input().to(device).to(dtype)

    OPSET = 17  # noqa: N806

    def forward_loop(model):
        model(dummy_input)

    model = mtq.quantize(model, config, forward_loop=forward_loop)

    input_names = ["input"]
    output_names = ["output"]

    buffer = io.BytesIO()

    if "enable_onnx_checker" in inspect.signature(torch.onnx.export).parameters:
        kwargs = {"enable_onnx_checker": False}
    else:
        kwargs = {}
    torch.onnx.export(
        model,
        dummy_input,
        f=buffer,
        opset_version=OPSET,
        input_names=input_names,
        output_names=output_names,
        do_constant_folding=constant_folding,
        dynamo=False,
        **kwargs,
    )

    # TODO: ort output correctness check for fp8
    # ONNXRuntime does not seem to be supporting bf16 gemms for cpu
    # TODO: Setup correctness check on GPU. ORT on GPU requires removing onnxruntime and installing onnxruntime_gpu
    if num_bits == 8 and dtype != torch.bfloat16 and device == "cpu":
        buffer.seek(0)
        providers = ["CUDAExecutionProvider"] if device != "cpu" else ["CPUExecutionProvider"]
        ort_session = onnxruntime.InferenceSession(buffer.read(), providers=providers)
        ort_result = ort_session.run([], {"input": dummy_input.cpu().numpy()})
        ort_result = torch.tensor(ort_result[0]).to(device)
        torch_result = model(dummy_input)
        print(ort_result, torch_result)
        assert torch.allclose(ort_result, torch_result, atol=1e-4, rtol=1e-4)
