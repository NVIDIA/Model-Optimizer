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

"""Unit tests for ONNX export for CPU quantization."""

import inspect
import io

import onnx
import pytest
import torch

pytest.importorskip("onnxruntime")

from _test_utils.torch.misc import set_seed
from _test_utils.torch.quantization.models import SimpleLinear
from _test_utils.torch.quantization.onnx_export import TEST_MODELS, onnx_export_tester

import modelopt.torch.quantization as mtq
import modelopt.torch.quantization.tensor_quant as tensor_quant
from modelopt.onnx.export import NVFP4QuantExporter
from modelopt.torch.quantization.utils import is_quantized_linear


@pytest.mark.parametrize("model_cls", TEST_MODELS)
@pytest.mark.parametrize(
    ("num_bits", "per_channel_quantization", "constant_folding"),
    [
        (8, True, True),
        (8, False, True),
        (8, True, False),
        (8, False, False),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_onnx_export_cpu(model_cls, num_bits, per_channel_quantization, constant_folding, dtype):
    # TODO: ORT output correctness tests sometimes fails due to random seed.
    # It needs to be investigated closer (lower priority). Lets set a seed for now.
    set_seed(90)
    onnx_export_tester(
        model_cls(), "cpu", num_bits, per_channel_quantization, constant_folding, dtype
    )


def test_nvfp4_exported_onnx_is_topologically_sorted(monkeypatch):
    def forward_loop(model):
        model(sample_input)

    def cpu_dynamic_block_quantize(inputs, *args):
        return inputs

    monkeypatch.setattr(tensor_quant, "dynamic_block_quantize_op", cpu_dynamic_block_quantize)

    model = SimpleLinear().eval()
    sample_input = model.get_input()
    model = mtq.quantize(model, mtq.NVFP4_DEFAULT_CFG, forward_loop=forward_loop)

    for module in model.modules():
        assert not isinstance(module, torch.nn.Linear) or is_quantized_linear(module)
        if isinstance(module, torch.nn.Linear):
            module.input_quantizer.disable()
            module.weight_quantizer._onnx_quantizer_type = "static"

    buffer = io.BytesIO()
    if "enable_onnx_checker" in inspect.signature(torch.onnx.export).parameters:
        kwargs = {"enable_onnx_checker": False}
    else:
        kwargs = {}

    torch.onnx.export(
        model,
        sample_input,
        buffer,
        input_names=["input"],
        output_names=["output"],
        export_params=True,
        opset_version=21,
        dynamo=False,
        **kwargs,
    )

    buffer.seek(0)
    exported_model = onnx.load_model_from_string(buffer.read())
    assert any(node.op_type == "TRT_FP4QDQ" for node in exported_model.graph.node)

    converted_model = NVFP4QuantExporter.process_model(exported_model)
    assert not any(node.op_type == "TRT_FP4QDQ" for node in converted_model.graph.node)
    onnx.checker.check_model(converted_model)
