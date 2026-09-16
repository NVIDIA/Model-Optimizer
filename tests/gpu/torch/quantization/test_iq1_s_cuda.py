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

import modelopt.torch.quantization.extensions as extensions
from modelopt.torch.quantization.extensions import get_cuda_ext_iq1_s
from modelopt.torch.quantization.ggml.iq1_s import dequantize_iq1_s, iq1_s_grid, quantize_iq1_s


def _extension():
    extension = get_cuda_ext_iq1_s(raise_if_failed=True)
    assert extension is not None
    return extension


def test_iq1_s_cuda_pack_is_deterministic_and_decodable():
    generator = torch.Generator(device="cuda").manual_seed(1234)
    weight = torch.randn((8, 256), generator=generator, device="cuda", dtype=torch.bfloat16)

    packed = _extension().pack(weight, iq1_s_grid("cuda")).reshape(8, 1, 50)
    packed_again = _extension().pack(weight, iq1_s_grid("cuda")).reshape(8, 1, 50)
    dispatched, shape = quantize_iq1_s(weight)
    reconstructed = dequantize_iq1_s(packed, shape)

    assert packed.shape == (8, 1, 50)
    assert torch.equal(packed, packed_again)
    assert torch.equal(packed, dispatched)
    normalized_mse = (
        reconstructed.float() - weight.float()
    ).square().mean() / weight.float().square().mean()
    assert normalized_mse < 0.25


def test_iq1_s_cuda_zero_encoding_matches_ggml_block_layout():
    weight = torch.zeros((1, 256), device="cuda", dtype=torch.bfloat16)
    packed = _extension().pack(weight, iq1_s_grid("cuda")).reshape(1, 1, 50)
    shape = torch.tensor(weight.shape, device="cuda")

    assert not packed.any()
    assert torch.equal(dequantize_iq1_s(packed, shape), weight)


def test_iq1_s_cuda_falls_back_to_pytorch_encoder(monkeypatch):
    monkeypatch.setattr(extensions, "get_cuda_ext_iq1_s", lambda: None)
    generator = torch.Generator(device="cuda").manual_seed(1234)
    weight = torch.randn((2, 256), generator=generator, device="cuda", dtype=torch.bfloat16)

    packed, shape = quantize_iq1_s(weight)
    reconstructed = dequantize_iq1_s(packed, shape)
    normalized_mse = (
        reconstructed.float() - weight.float()
    ).square().mean() / weight.float().square().mean()

    assert packed.shape == (2, 1, 50)
    assert normalized_mse < 0.25
