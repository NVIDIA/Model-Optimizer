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

import onnx
import pytest
import torch
from _test_utils.torch.quantization.models import SimpleLinear

import modelopt.torch.quantization as mtq
from modelopt.torch._deploy.utils import OnnxBytes, get_onnx_bytes_and_metadata
from modelopt.torch.quantization.utils import is_quantized_linear

dtype_to_onnx_type_name = {
    "Float": "FLOAT",
    "BFloat16": "BFLOAT16",
    "Half": "FLOAT16",
}


def _get_initializer_type_by_name(model, initializer_name):
    for initializer in model.graph.initializer:
        if initializer.name == initializer_name:
            return onnx.TensorProto.DataType.Name(initializer.data_type)
    return None


def _check_gemm_quantized(model, dtype):
    nodes = model.graph.node
    for node in nodes:
        if node.op_type == "Gemm":
            for input_name in node.input:
                # Both input nodes are of type DequantizeLinear, optionally passed through Cast
                assert (
                    "DequantizeLinear" in input_name
                    or "TRT_FP4QDQ" in input_name
                    or "bias" in input_name
                )
        if node.op_type in ["DequantizeLinear", "TRT_FP4DynamicQuantize"]:
            scale_type = _get_initializer_type_by_name(model, node.input[1])
            if "fp8_scale" in node.input[1]:
                assert scale_type == dtype_to_onnx_type_name["Float"]
            elif "fp16_scale" in node.input[1]:
                assert scale_type == dtype_to_onnx_type_name[dtype]

            output_type = _get_initializer_type_by_name(model, node.output[0])
            assert not output_type or output_type == dtype_to_onnx_type_name[dtype]


@pytest.mark.parametrize(
    "dtype",
    ["Float", "Half", "BFloat16"],
)
@pytest.mark.parametrize("weights_dtype", ["fp32", "fp16"])
def test_simple_linear(dtype: str, weights_dtype: str):
    def forward_loop(model, run_backward=False):
        for batch in calib_data:
            output = model(batch)
            if run_backward:
                output.sum().backward()

    config = mtq.NVFP4_DEFAULT_CFG
    torch_dtype = {"Float": torch.float32, "Half": torch.float16, "BFloat16": torch.bfloat16}[dtype]
    model = SimpleLinear().cuda().to(torch_dtype)
    calib_data = [model.get_input().cuda().to(torch_dtype) for _ in range(8)]
    sample_input = model.get_input().cuda().to(torch_dtype)

    model = mtq.quantize(model, config, forward_loop=forward_loop)
    for module in model.modules():
        assert not isinstance(module, torch.nn.Linear) or is_quantized_linear(module)
        if isinstance(module, torch.nn.Linear):
            module.input_quantizer._trt_high_precision_dtype = dtype
            module.input_quantizer._onnx_quantizer_type = "dynamic"
            module.weight_quantizer._onnx_quantizer_type = "static"

    onnx_bytes, _ = get_onnx_bytes_and_metadata(
        model,
        (sample_input,),
        weights_dtype=weights_dtype,
    )
    onnx_model = onnx.load_model_from_string(
        OnnxBytes.from_bytes(onnx_bytes).get_onnx_model_file_bytes()
    )
    onnx.checker.check_model(onnx_model, full_check=True)
    _check_gemm_quantized(onnx_model, "Half" if weights_dtype == "fp16" else dtype)
