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

import pytest
import torch

from modelopt.torch.quantization.ggml.iq1_s import (
    IQ1_S_BLOCK_BYTES,
    dequantize_iq1_s,
    iq1_s_fake_quant,
    iq1_s_grid,
    quantize_iq1_s,
)


def test_iq1_s_canonical_grid():
    grid = iq1_s_grid()

    assert grid.shape == (2048, 8)
    assert grid.dtype == torch.float32
    assert set(grid.unique().tolist()) == {-1.0, 0.0, 1.0}
    assert grid[0].tolist() == [-1.0] * 8


def test_iq1_s_zero_block_has_canonical_zero_encoding():
    weight = torch.zeros((2, 256), dtype=torch.bfloat16)

    packed, shape = quantize_iq1_s(weight)

    assert packed.shape == (2, 1, IQ1_S_BLOCK_BYTES)
    assert packed.dtype == torch.uint8
    assert not packed.any()
    assert shape.tolist() == [2, 256]
    assert torch.equal(dequantize_iq1_s(packed, shape), weight)


def test_iq1_s_dequantizes_ggml_metadata_bit_fields():
    packed = torch.zeros((1, 1, 50), dtype=torch.uint8)
    d = torch.tensor([2.0], dtype=torch.float16).view(torch.uint8)
    packed[0, 0, :2] = d
    entries = torch.tensor([0, 256, 511, 2047], dtype=torch.int64)
    packed[0, 0, 2:6] = (entries & 0xFF).to(torch.uint8)
    qh = (
        ((entries[0] >> 8) & 7)
        | (((entries[1] >> 8) & 7) << 3)
        | (((entries[2] >> 8) & 7) << 6)
        | (((entries[3] >> 8) & 7) << 9)
        | (3 << 12)
        | (1 << 15)
    )
    packed[0, 0, 34] = (qh & 0xFF).to(torch.uint8)
    packed[0, 0, 35] = (qh >> 8).to(torch.uint8)

    decoded = dequantize_iq1_s(packed, torch.tensor([1, 256]), dtype=torch.float32)
    expected = (iq1_s_grid()[entries] - 0.125) * 14.0

    assert torch.equal(decoded[0, :32].reshape(4, 8), expected)


def test_iq1_s_round_trip_and_payload_fields():
    generator = torch.Generator().manual_seed(1234)
    weight = torch.randn((2, 256), generator=generator, dtype=torch.bfloat16)

    packed, shape = quantize_iq1_s(weight, block_chunk_size=1)
    reconstructed = dequantize_iq1_s(packed, shape)

    assert packed.shape == (2, 1, 50)
    assert reconstructed.shape == weight.shape
    assert reconstructed.dtype == torch.bfloat16
    normalized_mse = (
        reconstructed.float() - weight.float()
    ).square().mean() / weight.float().square().mean()
    assert normalized_mse < 0.25

    blocks = packed.reshape(-1, 50)
    qh = blocks[:, 34:50:2].to(torch.int64) | (blocks[:, 35:50:2].to(torch.int64) << 8)
    assert torch.all(((qh >> 12) & 0x7) < 8)
    assert torch.all((qh & 0xFFF) < 0x1000)


def test_iq1_s_requires_complete_last_dimension_blocks():
    with pytest.raises(ValueError, match="last weight dimension"):
        quantize_iq1_s(torch.ones(2, 257))


def test_iq1_s_fake_quant_has_pass_through_gradient():
    class Quantizer:
        num_bits = "iq1_s"
        backend_extra_args = {"search_impl": "auto"}

    weight = torch.randn(1, 256, requires_grad=True)
    output = iq1_s_fake_quant(weight, Quantizer())
    output.sum().backward()

    assert torch.equal(weight.grad, torch.ones_like(weight))
