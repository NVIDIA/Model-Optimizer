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

from modelopt.torch.quantization.ggml.q8_0 import (
    Q8_0_BLOCK_BYTES,
    dequantize_q8_0,
    q8_0_fake_quant,
    quantize_q8_0,
)


def test_q8_0_zero_block_has_canonical_zero_encoding():
    weight = torch.zeros((2, 32), dtype=torch.bfloat16)

    packed, shape = quantize_q8_0(weight)

    assert packed.shape == (2, 1, Q8_0_BLOCK_BYTES)
    assert packed.dtype == torch.uint8
    assert not packed.any()
    assert torch.equal(dequantize_q8_0(packed, shape), weight)


def test_q8_0_uses_roundf_ties_away_from_zero():
    weight = torch.zeros((1, 32), dtype=torch.float32)
    weight[0, :5] = torch.tensor([127.0, 0.5, -0.5, 1.5, -1.5])

    packed, shape = quantize_q8_0(weight)

    assert packed[0, 0, :2].contiguous().view(torch.float16).item() == 1.0
    assert packed[0, 0, 2:7].contiguous().view(torch.int8).tolist() == [127, 1, -1, 2, -2]
    expected = weight.clone()
    expected[0, 1:5] = torch.tensor([1.0, -1.0, 2.0, -2.0])
    assert torch.equal(dequantize_q8_0(packed, shape, dtype=torch.float32), expected)


def test_q8_0_round_trip_and_payload_fields():
    generator = torch.Generator().manual_seed(1234)
    weight = torch.randn((2, 64), generator=generator, dtype=torch.bfloat16)

    packed, shape = quantize_q8_0(weight, block_chunk_size=1)
    reconstructed = dequantize_q8_0(packed, shape, dtype=torch.float32, block_chunk_size=1)

    assert packed.shape == (2, 2, 34)
    normalized_mse = (
        reconstructed - weight.float()
    ).square().mean() / weight.float().square().mean()
    assert normalized_mse < 1e-4
    assert torch.all(packed.reshape(-1, 34)[:, :2].contiguous().view(torch.float16) > 0)


def test_q8_0_requires_complete_last_dimension_blocks():
    with pytest.raises(ValueError, match="divisible by 32"):
        quantize_q8_0(torch.ones(2, 33))


def test_q8_0_treats_nonfinite_values_as_zero():
    weight = torch.randn(1, 32)
    weight[0, :3] = torch.tensor([torch.nan, torch.inf, -torch.inf])

    packed, _ = quantize_q8_0(weight)
    expected, _ = quantize_q8_0(torch.nan_to_num(weight, nan=0.0, posinf=0.0, neginf=0.0))

    assert torch.equal(packed, expected)


@pytest.mark.parametrize("shape", [[1, 31], [0, 32]])
def test_q8_0_rejects_invalid_shape_metadata(shape):
    packed = torch.zeros((1, 1, 34), dtype=torch.uint8)

    with pytest.raises(ValueError, match="logical weight shape"):
        dequantize_q8_0(packed, torch.tensor(shape))


def test_q8_0_fake_quant_has_pass_through_gradient():
    class Quantizer:
        num_bits = "q8_0"

    weight = torch.randn(1, 32, requires_grad=True)
    q8_0_fake_quant(weight, Quantizer()).sum().backward()

    assert torch.equal(weight.grad, torch.ones_like(weight))
