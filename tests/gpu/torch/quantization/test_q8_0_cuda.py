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

import modelopt.torch.quantization.ggml.q8_0 as q8_0_module
from modelopt.torch.quantization.extensions import get_cuda_ext_ggml
from modelopt.torch.quantization.ggml.q8_0 import dequantize_q8_0, quantize_q8_0


def _extension():
    extension = get_cuda_ext_ggml(raise_if_failed=True)
    assert extension is not None
    return extension


def test_q8_0_cuda_pack_matches_pytorch_encoder_and_is_decodable(monkeypatch):
    generator = torch.Generator(device="cuda").manual_seed(1234)
    weight = torch.randn((8, 64), generator=generator, device="cuda", dtype=torch.bfloat16)

    packed = _extension().q8_0_pack(weight).reshape(8, 2, 34)
    packed_again = _extension().q8_0_pack(weight).reshape(8, 2, 34)
    monkeypatch.setattr(q8_0_module, "get_cuda_ext_ggml", lambda: None)
    reference, shape = quantize_q8_0(weight)
    reconstructed = dequantize_q8_0(packed, shape)

    assert packed.shape == (8, 2, 34)
    assert torch.equal(packed, packed_again)
    assert torch.equal(packed, reference)
    assert shape.device.type == "cpu"
    normalized_mse = (
        reconstructed.float() - weight.float()
    ).square().mean() / weight.float().square().mean()
    assert normalized_mse < 1e-4


def test_q8_0_cuda_zero_and_nonfinite_encoding_match_reference(monkeypatch):
    weight = torch.zeros((2, 32), device="cuda", dtype=torch.bfloat16)
    weight[1, :3] = torch.tensor([torch.nan, torch.inf, -torch.inf], device="cuda")

    packed = _extension().q8_0_pack(weight).reshape(2, 1, 34)
    monkeypatch.setattr(q8_0_module, "get_cuda_ext_ggml", lambda: None)
    reference, shape = quantize_q8_0(weight)

    assert not packed.any()
    assert torch.equal(packed, reference)
    assert torch.equal(dequantize_q8_0(packed, shape), torch.zeros_like(weight))


def test_q8_0_cuda_falls_back_to_pytorch_encoder(monkeypatch):
    monkeypatch.setattr(q8_0_module, "get_cuda_ext_ggml", lambda: None)
    generator = torch.Generator(device="cuda").manual_seed(1234)
    weight = torch.randn((2, 32), generator=generator, device="cuda", dtype=torch.bfloat16)

    packed, shape = quantize_q8_0(weight)
    reconstructed = dequantize_q8_0(packed, shape)

    assert packed.shape == (2, 1, 34)
    assert torch.isfinite(reconstructed).all()


def test_q8_0_cuda_float64_matches_pytorch_encoder():
    weight = torch.randn(4, 32, dtype=torch.float64, generator=torch.Generator().manual_seed(7))

    reference, _ = quantize_q8_0(weight)
    packed, _ = quantize_q8_0(weight.cuda())

    assert torch.equal(reference, packed.cpu())
