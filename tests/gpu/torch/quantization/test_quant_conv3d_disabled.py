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

"""Kernel dispatch of NVFP4 Conv3d: implicit GEMM only for quantized activations."""

import copy
import warnings

import pytest
import torch
import torch.nn.functional as F

import modelopt.torch.quantization as mtq
import modelopt.torch.quantization.nn.modules.quant_conv as quant_conv


class _VisionLanguageModel(torch.nn.Module):
    """Qwen-VL-style Conv3d vision patch embed, often kept in high precision, and a quantized LM."""

    def __init__(self):
        super().__init__()
        self.visual = torch.nn.Conv3d(3, 64, kernel_size=(2, 14, 14), stride=(2, 14, 14))
        self.language_model = torch.nn.Linear(64, 64)

    def forward(self, x):
        return self.language_model(self.visual(x).flatten(2).transpose(1, 2))


def _quantized_model(disabled_quantizers=()):
    torch.manual_seed(0)
    model = _VisionLanguageModel().cuda().eval()
    x = torch.randn(1, 3, 2, 56, 56, device="cuda")
    reference = model.visual(x)
    cfg = copy.deepcopy(mtq.NVFP4_DEFAULT_CFG)
    # NVFP4_DEFAULT_CFG keeps "*visual*" in high precision; re-enable it to choose per case.
    cfg["quant_cfg"].append({"quantizer_name": "*visual.*_quantizer", "enable": True})
    cfg["quant_cfg"].append({"quantizer_name": "*visual.output_quantizer", "enable": False})
    cfg["quant_cfg"] += [{"quantizer_name": name, "enable": False} for name in disabled_quantizers]
    mtq.quantize(model, cfg, lambda m: m(x))
    return model, x, reference


@pytest.fixture
def kernel_calls(monkeypatch):
    calls = []
    kernel = quant_conv.conv3d_implicit_gemm_cuda

    def counting_kernel(*args, **kwargs):
        calls.append(kwargs)
        return kernel(*args, **kwargs)

    monkeypatch.setattr(quant_conv, "conv3d_implicit_gemm_cuda", counting_kernel)
    return calls


@pytest.fixture(autouse=True)
def _nvfp4_gpu_fp32(cuda_capability, monkeypatch):
    if cuda_capability < (8, 9):
        pytest.skip("NVFP4 quantization requires compute capability >= 8.9")
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", False)


@pytest.mark.parametrize("case", ["config_disabled", "disable_quant"])
@torch.no_grad()
def test_conv3d_without_quantization_runs_cudnn(kernel_calls, case):
    if case == "config_disabled":
        model, x, reference = _quantized_model(["*visual*"])
    else:
        model, x, reference = _quantized_model()
        model.visual.input_quantizer.disable_quant()
        model.visual.weight_quantizer.disable_quant()

    output = model.visual(x)

    assert not kernel_calls
    torch.testing.assert_close(output, reference, rtol=0, atol=1e-5)


@torch.no_grad()
def test_conv3d_weight_only_runs_cudnn_with_weight_blocked_along_k(kernel_calls):
    model, x, _ = _quantized_model(["*visual.input_quantizer"])
    conv = model.visual
    weight = quant_conv._nvfp4_quantize_weight_along_k(conv.weight, conv.weight_quantizer)

    output = conv(x)

    assert not kernel_calls
    torch.testing.assert_close(
        output, F.conv3d(x, weight, conv.bias, conv.stride), rtol=0, atol=1e-5
    )


@torch.no_grad()
def test_conv3d_quantized_activations_run_implicit_gemm(kernel_calls):
    model, x, _ = _quantized_model()

    model.visual(x)

    assert len(kernel_calls) == 1
    assert kernel_calls[0]["quant_act"]


@torch.no_grad()
def test_conv3d_training_falls_back_to_cudnn(kernel_calls):
    model, x, _ = _quantized_model()
    model.visual.train()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.visual(x)

    assert not kernel_calls
    assert any("inference-only" in str(w.message) for w in caught)
