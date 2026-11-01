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

import torch

import modelopt.torch.quantization.ggml.iq1_s as iq1_s_module
from modelopt.torch.quantization.extensions import get_cuda_ext_ggml
from modelopt.torch.quantization.ggml.iq1_s import dequantize_iq1_s, iq1_s_grid, quantize_iq1_s


def _extension():
    extension = get_cuda_ext_ggml(raise_if_failed=True)
    assert extension is not None
    return extension


def test_iq1_s_cuda_pack_matches_pytorch_encoder_and_is_decodable(monkeypatch):
    generator = torch.Generator(device="cuda").manual_seed(1234)
    weight = torch.randn((8, 256), generator=generator, device="cuda", dtype=torch.bfloat16)

    packed = _extension().iq1_s_pack(weight, iq1_s_grid("cuda")).reshape(8, 1, 50)
    packed_again = _extension().iq1_s_pack(weight, iq1_s_grid("cuda")).reshape(8, 1, 50)
    monkeypatch.setattr(iq1_s_module, "get_cuda_ext_ggml", lambda: None)
    reference, shape = quantize_iq1_s(weight)
    reconstructed = dequantize_iq1_s(packed, shape)

    assert packed.shape == (8, 1, 50)
    assert torch.equal(packed, packed_again)
    assert torch.equal(packed, reference)
    assert shape.device.type == "cpu"
    normalized_mse = (
        reconstructed.float() - weight.float()
    ).square().mean() / weight.float().square().mean()
    assert normalized_mse < 0.25


def test_iq1_s_cuda_zero_encoding_matches_ggml_block_layout():
    weight = torch.zeros((1, 256), device="cuda", dtype=torch.bfloat16)
    packed = _extension().iq1_s_pack(weight, iq1_s_grid("cuda")).reshape(1, 1, 50)
    shape = torch.tensor(weight.shape, device="cuda")

    assert not packed.any()
    assert torch.equal(dequantize_iq1_s(packed, shape), weight)


def test_iq1_s_cuda_nonfinite_policy_matches_pytorch_encoder(monkeypatch):
    weight = torch.randn((1, 256), device="cuda", dtype=torch.bfloat16)
    weight[0, :3] = torch.tensor([torch.nan, torch.inf, -torch.inf], device="cuda")

    packed = _extension().iq1_s_pack(weight, iq1_s_grid("cuda")).reshape(1, 1, 50)
    monkeypatch.setattr(iq1_s_module, "get_cuda_ext_ggml", lambda: None)
    reference, _ = quantize_iq1_s(weight)

    assert torch.equal(packed, reference)


def test_iq1_s_cuda_falls_back_to_pytorch_encoder(monkeypatch):
    monkeypatch.setattr(iq1_s_module, "get_cuda_ext_ggml", lambda: None)
    generator = torch.Generator(device="cuda").manual_seed(1234)
    weight = torch.randn((2, 256), generator=generator, device="cuda", dtype=torch.bfloat16)

    packed, shape = quantize_iq1_s(weight)
    reconstructed = dequantize_iq1_s(packed, shape)
    normalized_mse = (
        reconstructed.float() - weight.float()
    ).square().mean() / weight.float().square().mean()

    assert packed.shape == (2, 1, 50)
    assert normalized_mse < 0.25


def test_iq1_s_cuda_float64_matches_pytorch_encoder():
    """float64 weights inside the float32 range must pack identically on both paths."""
    weight = torch.randn(4, 256, dtype=torch.float64, generator=torch.Generator().manual_seed(7))

    reference, _ = quantize_iq1_s(weight)
    packed, _ = quantize_iq1_s(weight.cuda())

    assert torch.equal(reference, packed.cpu())


def test_iq1_s_cuda_saturates_finite_values_above_the_float32_range():
    """The extension saturates such values rather than dropping them to zero.

    Byte parity with the reference encoder is not asserted here: at these magnitudes the
    squared-error objective overflows to infinity in float32, so every codebook candidate ties
    and the two search implementations break that tie differently. The saturation policy is
    what both paths must agree on.
    """
    weight = torch.randn(1, 256, dtype=torch.float64, device="cuda")
    weight[0, 7] = 1e100
    saturated = weight.clone()
    saturated[0, 7] = torch.finfo(torch.float32).max
    zeroed = weight.clone()
    zeroed[0, 7] = 0.0

    packed, _ = quantize_iq1_s(weight)

    assert torch.equal(packed, quantize_iq1_s(saturated)[0])
    assert not torch.equal(packed, quantize_iq1_s(zeroed)[0])
