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

"""GPU regression tests for quantized convolutions."""

import pytest
import torch

import modelopt.torch.quantization as mtq
import modelopt.torch.quantization.nn.modules.quant_conv as quant_conv


@pytest.mark.parametrize("conv_kind", ["conv3d", "wan", "wan_cached"])
@pytest.mark.parametrize("quantize_weight", [False, True])
@torch.no_grad()
def test_nvfp4_conv3d_disable_quant(cuda_capability, conv_kind, quantize_weight):
    """Disabling activation QDQ must match bypassing its quantizer, even in the fused kernel."""
    if cuda_capability < (8, 9):
        pytest.skip("NVFP4 quantization requires compute capability >= 8.9")

    torch.manual_seed(1234)
    conv_cls = torch.nn.Conv3d
    if conv_kind != "conv3d":
        # Diffusers and its quantization plugin are optional for the regular Conv3d case.
        pytest.importorskip("onnx")
        pytest.importorskip("diffusers")
        from diffusers.models.autoencoders.autoencoder_kl_wan import WanCausalConv3d

        import modelopt.torch.quantization.plugins.diffusion.diffusers  # noqa: F401

        conv_cls = WanCausalConv3d

    model = conv_cls(16, 16, kernel_size=3, padding=1).to("cuda", torch.bfloat16).eval()
    x = torch.randn(1, 16, 3, 6, 6, device="cuda", dtype=torch.bfloat16)
    kwargs = {}
    if conv_kind == "wan_cached":
        kwargs["cache_x"] = torch.randn(1, 16, 1, 6, 6, device="cuda", dtype=torch.bfloat16)
    model = mtq.quantize(model, mtq.NVFP4_DEFAULT_CFG, lambda m: m(x, **kwargs))
    if not quantize_weight:
        model.weight_quantizer.disable_quant()

    enabled = model(x, **kwargs)
    model.input_quantizer.disable_quant()
    disabled = model(x, **kwargs)

    # Bypass the activation quantizer entirely as a control, keeping the weight state fixed.
    model.input_quantizer.disable()
    bypassed = model(x, **kwargs)
    torch.testing.assert_close(disabled, bypassed, rtol=0, atol=0)
    assert not torch.equal(enabled, disabled)

    model.input_quantizer.enable()
    model.input_quantizer.enable_quant()
    torch.testing.assert_close(model(x, **kwargs), enabled, rtol=0, atol=0)


@pytest.fixture
def nvfp4_conv3d(cuda_capability, monkeypatch):
    """A Conv3d quantized with NVFP4, its input, its unquantized output, and the kernel calls."""
    if cuda_capability < (8, 9):
        pytest.skip("NVFP4 quantization requires compute capability >= 8.9")
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", False)

    kernel_calls = []
    kernel = quant_conv.conv3d_implicit_gemm_cuda

    def counting_kernel(*args, **kwargs):
        kernel_calls.append(kwargs)
        return kernel(*args, **kwargs)

    monkeypatch.setattr(quant_conv, "conv3d_implicit_gemm_cuda", counting_kernel)

    torch.manual_seed(0)
    # Qwen-VL-style vision patch embed.
    model = torch.nn.Conv3d(3, 64, kernel_size=(2, 14, 14), stride=(2, 14, 14)).cuda().eval()
    x = torch.randn(1, 3, 2, 56, 56, device="cuda")
    with torch.no_grad():
        reference = model(x)
    mtq.quantize(model, mtq.NVFP4_DEFAULT_CFG, lambda m: m(x))
    kernel_calls.clear()  # count only the forward under test, not calibration
    return model, x, reference, kernel_calls


@pytest.mark.parametrize("switch_off", ["disable", "disable_quant"])
@torch.no_grad()
def test_nvfp4_conv3d_without_quantization_runs_cudnn(nvfp4_conv3d, switch_off):
    """A Conv3d that quantizes nothing, like a vision tower kept in high precision, is nn.Conv3d."""
    model, x, reference, kernel_calls = nvfp4_conv3d
    for quantizer in (model.input_quantizer, model.weight_quantizer):
        getattr(quantizer, switch_off)()

    output = model(x)

    assert not kernel_calls
    torch.testing.assert_close(output, reference, rtol=0, atol=1e-5)


@torch.no_grad()
def test_nvfp4_conv3d_training_falls_back_to_cudnn(nvfp4_conv3d):
    model, x, _, kernel_calls = nvfp4_conv3d
    model.train()

    with pytest.warns(UserWarning, match="inference-only"):
        model(x)

    assert not kernel_calls
