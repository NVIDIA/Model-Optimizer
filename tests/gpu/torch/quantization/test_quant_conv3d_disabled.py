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

"""Kernel dispatch of NVFP4 Conv3d when its quantizers are disabled."""

import copy

import pytest
import torch

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


@pytest.mark.parametrize("disabled_quantizers", [["*visual*"], ["*visual.input_quantizer"]])
@torch.no_grad()
def test_disabled_nvfp4_conv3d_dispatch(cuda_capability, monkeypatch, disabled_quantizers):
    if cuda_capability < (8, 9):
        pytest.skip("NVFP4 quantization requires compute capability >= 8.9")
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", False)

    torch.manual_seed(0)
    model = _VisionLanguageModel().cuda().eval()
    x = torch.randn(1, 3, 2, 56, 56, device="cuda")
    reference = model.visual(x)

    cfg = copy.deepcopy(mtq.NVFP4_DEFAULT_CFG)
    cfg["quant_cfg"] += [{"quantizer_name": name, "enable": False} for name in disabled_quantizers]
    mtq.quantize(model, cfg, lambda m: m(x))

    kernel_calls = []
    kernel = quant_conv.conv3d_implicit_gemm_cuda

    def counting_kernel(*args, **kwargs):
        kernel_calls.append(kwargs)
        return kernel(*args, **kwargs)

    monkeypatch.setattr(quant_conv, "conv3d_implicit_gemm_cuda", counting_kernel)
    output = model.visual(x)

    if model.visual.weight_quantizer.is_enabled:
        assert len(kernel_calls) == 1
        assert not kernel_calls[0]["quant_act"]
    else:
        assert not kernel_calls
        torch.testing.assert_close(output, reference, rtol=0, atol=1e-5)
