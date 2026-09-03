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

"""GPU integration tests for Dynamo ONNX export of quantized models."""

import copy

import onnx
import pytest
import torch
from _test_utils.torch.misc import minimum_sm

import modelopt.torch.quantization as mtq
from modelopt.torch._deploy.utils import OnnxBytes, get_onnx_bytes_and_metadata


class _AlignedLinear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(128, 128, bias=False)

    def forward(self, inputs):
        return self.linear(inputs)


_CASES = [
    pytest.param(
        "fp8",
        mtq.FP8_DEFAULT_CFG,
        {"QuantizeLinear", "DequantizeLinear"},
        id="fp8",
    ),
    pytest.param(
        "int8",
        mtq.INT8_DEFAULT_CFG,
        {"QuantizeLinear", "DequantizeLinear"},
        id="int8",
    ),
    pytest.param(
        "mxfp8",
        mtq.MXFP8_DEFAULT_CFG,
        {"TRT_MXFP8DynamicQuantize", "TRT_MXFP8DequantizeLinear"},
        id="mxfp8",
    ),
    pytest.param(
        "nvfp4",
        mtq.NVFP4_DEFAULT_CFG,
        {"TRT_FP4DynamicQuantize", "DequantizeLinear"},
        marks=minimum_sm(100),
        id="nvfp4",
    ),
]


@pytest.mark.timeout(600)
@pytest.mark.parametrize(("quant_format", "config", "expected_ops"), _CASES)
def test_quantized_model_dynamo_export(tmp_path, quant_format, config, expected_ops):
    model = _AlignedLinear().eval().cuda()
    sample_input = torch.randn(2, 128, device="cuda")
    model = mtq.quantize(
        model,
        copy.deepcopy(config),
        forward_loop=lambda candidate: candidate(sample_input),
    )

    onnx_bytes, _ = get_onnx_bytes_and_metadata(
        model,
        (sample_input,),
        model_name=f"aligned_linear_{quant_format}_dynamo",
        dynamo_export=True,
        onnx_opset=24,
        weights_dtype="fp16",
    )
    onnx_package = OnnxBytes.from_bytes(onnx_bytes)
    export_dir = tmp_path / quant_format
    onnx_package.write_to_disk(str(export_dir))
    exported = onnx.load(
        export_dir / f"{onnx_package.model_name}.onnx",
        load_external_data=True,
    )

    onnx.checker.check_model(exported)
    ops = {node.op_type for node in exported.graph.node}
    assert expected_ops <= ops
    assert "quantize_op" not in ops
    assert "dynamic_block_quantize_op" not in ops
    initializer_dtypes = {initializer.data_type for initializer in exported.graph.initializer}
    if quant_format == "mxfp8":
        assert onnx.TensorProto.FLOAT8E4M3FN in initializer_dtypes
        assert onnx.TensorProto.UINT8 in initializer_dtypes
    elif quant_format == "nvfp4":
        assert "TRT_FP4QDQ" not in ops
        assert onnx.TensorProto.FLOAT4E2M1 in initializer_dtypes
        assert onnx.TensorProto.FLOAT8E4M3FN in initializer_dtypes
