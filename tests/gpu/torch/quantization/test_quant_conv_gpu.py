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
