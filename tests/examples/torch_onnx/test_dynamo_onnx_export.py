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

"""TensorRT execution tests for helper-owned quantized Dynamo ONNX export."""

import copy

import onnx
import pytest
import timm
import torch
from _test_utils.import_helper import skip_if_no_trtexec
from _test_utils.torch.misc import minimum_sm, set_seed
from packaging.version import Version
from torch import nn

from examples.torch_onnx.torch_quant_to_onnx import _prepare_auto_quantize_format, get_quant_config

skip_if_no_trtexec()

trt = pytest.importorskip("tensorrt")

import modelopt.torch.quantization as mtq
from modelopt.torch._deploy._runtime.trt_client import TRTLocalClient
from modelopt.torch._deploy.device_model import DeviceModel
from modelopt.torch._deploy.utils import OnnxBytes, get_onnx_bytes_and_metadata

_DEPLOYMENT = {
    "runtime": "TRT",
    "accelerator": "GPU",
    "precision": "stronglyTyped",
    "onnx_opset": "23",
}


def _model():
    return (
        timm.create_model(
            "vit_tiny_patch16_224",
            pretrained=False,
            img_size=32,
            depth=1,
            embed_dim=128,
            num_heads=4,
            mlp_ratio=1,
            num_classes=16,
        )
        .cuda()
        .eval()
    )


def _quantize(model, sample_input, config):
    return mtq.quantize(
        model,
        copy.deepcopy(config),
        forward_loop=lambda candidate: candidate(sample_input),
    )


def _compile_and_compare(model, inputs, *, rtol, atol):
    expected = model(*inputs)
    payload, metadata = get_onnx_bytes_and_metadata(
        model,
        inputs,
        dynamo_export=True,
    )
    client = TRTLocalClient(_DEPLOYMENT)
    compiled = client.ir_to_compiled(payload, {"decomposable_attentions": True})
    actual = DeviceModel(client, compiled, metadata)(*inputs)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    client._teardown_all_sessions()


@pytest.mark.timeout(600)
@pytest.mark.parametrize(
    ("qformat", "rtol", "atol", "minimum_compute_capability"),
    [
        ("fp8", 0.1, 0.05, (8, 9)),
        ("int8", 0.05, 0.1, (0, 0)),
        ("mxfp8", 0.1, 0.1, (8, 9)),
        ("nvfp4", 0.1, 0.5, (10, 0)),
    ],
    ids=["fp8", "int8", "mxfp8", "nvfp4"],
)
def test_quantized_dynamo_onnx_executes_with_tensorrt(
    qformat, rtol, atol, minimum_compute_capability
):
    if torch.cuda.get_device_capability() < minimum_compute_capability:
        pytest.skip(f"Requires compute capability {minimum_compute_capability} or newer")
    if (qformat in {"mxfp8", "nvfp4"}) and Version(trt.__version__) < Version("10.11"):
        pytest.skip("Block quantization requires TensorRT 10.11 or newer")

    set_seed()
    calibration_input = torch.randn(2, 3, 32, 32, device="cuda")
    model = _quantize(_model(), calibration_input, get_quant_config(qformat))
    _compile_and_compare(
        model,
        (torch.randn_like(calibration_input),),
        rtol=rtol,
        atol=atol,
    )


@pytest.mark.timeout(600)
def test_int4_awq_dynamo_onnx_exports_and_checks(tmp_path):
    set_seed()
    calibration_input = torch.randn(2, 3, 32, 32, device="cuda")
    model = _quantize(_model(), calibration_input, get_quant_config("int4_awq"))

    payload, _ = get_onnx_bytes_and_metadata(
        model,
        (torch.randn_like(calibration_input),),
        dynamo_export=True,
        weights_dtype="fp16",
    )
    package = OnnxBytes.from_bytes(payload)
    package.write_to_disk(str(tmp_path))
    exported = onnx.load(tmp_path / f"{package.model_name}.onnx", load_external_data=True)

    onnx.checker.check_model(exported, full_check=True)
    assert any(
        initializer.data_type == onnx.TensorProto.INT4 for initializer in exported.graph.initializer
    )


@minimum_sm(100)
@pytest.mark.timeout(600)
def test_fp8_nvfp4_autoquant_executes_with_tensorrt():
    if Version(trt.__version__) < Version("10.11"):
        pytest.skip("NVFP4 requires TensorRT 10.11 or newer")

    set_seed()
    calibration_input = torch.randn(2, 3, 32, 32, device="cuda")
    model, _ = mtq.auto_quantize(
        _model(),
        constraints={"effective_bits": 6.25},
        quantization_formats=[
            _prepare_auto_quantize_format("nvfp4"),
            _prepare_auto_quantize_format("fp8"),
        ],
        data_loader=[calibration_input],
        forward_step=lambda candidate, batch: candidate(batch),
        loss_func=lambda output, _batch: output.float().square().mean(),
        num_calib_steps=1,
        num_score_steps=1,
    )
    selected_formats = {
        tuple(module.weight_quantizer.num_bits)
        for module in model.modules()
        if isinstance(module, nn.Linear) and module.weight_quantizer.is_enabled
    }
    assert {(2, 1), (4, 3)} <= selected_formats
    _compile_and_compare(
        model,
        (torch.randn_like(calibration_input),),
        rtol=0.1,
        atol=0.5,
    )
