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

import modelopt.torch.quantization.ggml.iq2_xs as iq2_xs_module
from modelopt.torch.quantization.extensions import get_cuda_ext_ggml
from modelopt.torch.quantization.ggml.iq2_xs import dequantize_iq2_xs, iq2_xs_grid, quantize_iq2_xs


def _extension():
    extension = get_cuda_ext_ggml(raise_if_failed=True)
    assert extension is not None
    return extension


def _pack(weight):
    blocks = weight.contiguous().reshape(-1, 256)
    scales = iq2_xs_module._predict_iq2_xs_scales(blocks)
    return _extension().iq2_xs_pack(weight, iq2_xs_grid("cuda"), scales)


def test_iq2_xs_cuda_pack_matches_pytorch_encoder_and_is_decodable(monkeypatch):
    generator = torch.Generator(device="cuda").manual_seed(1234)
    weight = torch.randn((8, 512), generator=generator, device="cuda", dtype=torch.bfloat16)

    packed = _pack(weight).reshape(8, 2, 74)
    packed_again = _pack(weight).reshape(8, 2, 74)
    monkeypatch.setattr(iq2_xs_module, "get_cuda_ext_ggml", lambda: None)
    reference, shape = quantize_iq2_xs(weight)
    reconstructed = dequantize_iq2_xs(packed, shape)

    assert packed.shape == (8, 2, 74)
    assert torch.equal(packed, packed_again)
    assert torch.equal(packed, reference)
    assert shape.device.type == "cpu"
    normalized_mse = (
        reconstructed.float() - weight.float()
    ).square().mean() / weight.float().square().mean()
    assert normalized_mse < 0.1


def test_iq2_xs_cuda_zero_encoding_matches_ggml_block_layout():
    weight = torch.zeros((1, 256), device="cuda", dtype=torch.bfloat16)
    packed = _pack(weight).reshape(1, 1, 74)
    shape = torch.tensor(weight.shape, device="cuda")

    assert not packed.any()
    assert torch.equal(dequantize_iq2_xs(packed, shape), weight)


def test_iq2_xs_cuda_underflowed_scale_has_canonical_zero_encoding():
    weight = torch.full((1, 256), -1e-6, device="cuda", dtype=torch.bfloat16)
    packed = _pack(weight).reshape(1, 1, 74)

    assert not packed.any()


def test_iq2_xs_cuda_nonfinite_policy_matches_pytorch_encoder(monkeypatch):
    weight = torch.randn((1, 256), device="cuda", dtype=torch.bfloat16)
    weight[0, :3] = torch.tensor([torch.nan, torch.inf, -torch.inf], device="cuda")

    packed = _pack(weight).reshape(1, 1, 74)
    monkeypatch.setattr(iq2_xs_module, "get_cuda_ext_ggml", lambda: None)
    reference, _ = quantize_iq2_xs(weight)

    assert torch.equal(packed, reference)


def test_iq2_xs_cuda_falls_back_to_pytorch_encoder(monkeypatch):
    monkeypatch.setattr(iq2_xs_module, "get_cuda_ext_ggml", lambda: None)
    generator = torch.Generator(device="cuda").manual_seed(1234)
    weight = torch.randn((2, 256), generator=generator, device="cuda", dtype=torch.bfloat16)

    packed, shape = quantize_iq2_xs(weight)
    reconstructed = dequantize_iq2_xs(packed, shape)
    normalized_mse = (
        reconstructed.float() - weight.float()
    ).square().mean() / weight.float().square().mean()

    assert packed.shape == (2, 1, 74)
    assert normalized_mse < 0.1


def test_iq2_xs_cuda_float64_matches_pytorch_encoder():
    """float64 weights inside the float32 range must pack identically on both paths."""
    weight = torch.randn(4, 256, dtype=torch.float64, generator=torch.Generator().manual_seed(7))

    reference, _ = quantize_iq2_xs(weight)
    packed, _ = quantize_iq2_xs(weight.cuda())

    assert torch.equal(reference, packed.cpu())


def test_iq2_xs_cuda_saturates_finite_values_above_the_float32_range():
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

    packed, _ = quantize_iq2_xs(weight)

    assert torch.equal(packed, quantize_iq2_xs(saturated)[0])
    assert not torch.equal(packed, quantize_iq2_xs(zeroed)[0])
