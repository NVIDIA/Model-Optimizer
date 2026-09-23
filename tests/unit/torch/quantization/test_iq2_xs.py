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

import modelopt.torch.quantization.ggml.iq2_xs as iq2_xs_module
from modelopt.torch.quantization.ggml.iq2_xs import (
    IQ2_XS_BLOCK_BYTES,
    dequantize_iq2_xs,
    iq2_xs_fake_quant,
    iq2_xs_grid,
    quantize_iq2_xs,
)


def test_iq2_xs_canonical_grid():
    grid = iq2_xs_grid()

    assert grid.shape == (512, 8)
    assert grid.dtype == torch.float32
    assert set(grid.unique().tolist()) == {8.0, 25.0, 43.0}
    assert grid[0].tolist() == [8.0] * 8
    assert grid[-1].tolist() == [43.0] * 8


def test_iq2_xs_grid_normalizes_unindexed_cuda_device(monkeypatch):
    cached = torch.empty(0)
    indexed_device = torch.device("cuda", 7)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 7)
    monkeypatch.setitem(iq2_xs_module._GRID_CACHE, indexed_device, cached)

    assert iq2_xs_grid("cuda") is cached


def test_iq2_xs_zero_block_has_canonical_zero_encoding():
    weight = torch.zeros((2, 256), dtype=torch.bfloat16)

    packed, shape = quantize_iq2_xs(weight)

    assert packed.shape == (2, 1, IQ2_XS_BLOCK_BYTES)
    assert packed.dtype == torch.uint8
    assert not packed.any()
    assert shape.tolist() == [2, 256]
    assert torch.equal(dequantize_iq2_xs(packed, shape), weight)


def test_iq2_xs_underflowed_scale_has_canonical_zero_encoding():
    weight = torch.full((1, 256), -1e-6, dtype=torch.bfloat16)

    packed, shape = quantize_iq2_xs(weight)

    assert not packed.any()
    assert torch.equal(dequantize_iq2_xs(packed, shape), torch.zeros_like(weight))


def test_iq2_xs_round_trip_and_payload_fields():
    generator = torch.Generator().manual_seed(1234)
    weight = torch.randn((2, 512), generator=generator, dtype=torch.bfloat16)

    packed, shape = quantize_iq2_xs(weight, block_chunk_size=2)
    reconstructed = dequantize_iq2_xs(packed, shape, block_chunk_size=1)
    default_reconstructed = dequantize_iq2_xs(packed, shape)

    assert packed.shape == (2, 2, 74)
    assert reconstructed.shape == weight.shape
    assert reconstructed.dtype == torch.bfloat16
    assert torch.equal(reconstructed, default_reconstructed)
    normalized_mse = (
        reconstructed.float() - weight.float()
    ).square().mean() / weight.float().square().mean()
    assert normalized_mse < 0.1

    blocks = packed.reshape(-1, 74)
    codes = blocks[:, 2:66:2].to(torch.int64) | (blocks[:, 3:66:2].to(torch.int64) << 8)
    assert torch.all((codes & 0x1FF) < 512)
    assert torch.all((codes >> 9) < 128)


def test_iq2_xs_dequantizes_pinned_scale_factor():
    packed = torch.zeros((1, 1, 74), dtype=torch.uint8)
    packed[0, 0, :2] = torch.tensor([1.0], dtype=torch.float16).view(torch.uint8)
    packed[0, 0, 2:66:2] = 0xFF
    packed[0, 0, 3:66:2] = 0x01
    packed[0, 0, 66:] = 0xFF

    decoded = dequantize_iq2_xs(packed, torch.tensor([1, 256]), dtype=torch.float32)

    # Entry 511 contains eight 43s and local code 15 gives (2 * 15 + 1) / 8.
    assert torch.equal(decoded, torch.full((1, 256), 43 * 31 / 8, dtype=torch.float32))


@pytest.mark.parametrize("sign_index", [0b0000001, 0b0000011, 0b0000111, 0b1010101, 0b1111111])
def test_iq2_xs_dequantizes_the_implied_eighth_sign_bit(sign_index):
    """The eighth sign is not stored; it is the parity of the seven that are.

    The pinned-scale test above only covers sign_index 0, where every value is positive
    whether or not the implied bit is derived correctly. These indices vary which payload
    bits are set so the parity actually has to be computed.
    """
    codes = 511 | (sign_index << 9)  # entry 511 holds eight 43s
    packed = torch.zeros((1, 1, 74), dtype=torch.uint8)
    packed[0, 0, :2] = torch.tensor([1.0], dtype=torch.float16).view(torch.uint8)
    packed[0, 0, 2:66:2] = codes & 0xFF
    packed[0, 0, 3:66:2] = codes >> 8
    packed[0, 0, 66:] = 0xFF  # local code 15 in both nibbles

    decoded = dequantize_iq2_xs(packed, torch.tensor([1, 256]), dtype=torch.float32)

    payload_bits = [(sign_index >> bit) & 1 for bit in range(7)]
    magnitude = 43 * 31 / 8  # entry value 43, local code 15 -> (2 * 15 + 1) / 8
    expected_group = torch.tensor(
        [-magnitude if bit else magnitude for bit in (*payload_bits, sum(payload_bits) % 2)],
        dtype=torch.float32,
    )
    assert torch.equal(decoded.reshape(32, 8), expected_group.expand(32, 8))


def test_iq2_xs_requires_complete_last_dimension_blocks():
    with pytest.raises(ValueError, match="last weight dimension"):
        quantize_iq2_xs(torch.ones(2, 257))


def test_iq2_xs_treats_nonfinite_values_as_zero():
    weight = torch.randn(1, 256)
    weight[0, :3] = torch.tensor([torch.nan, torch.inf, -torch.inf])

    packed, _ = quantize_iq2_xs(weight)
    expected, _ = quantize_iq2_xs(torch.nan_to_num(weight, nan=0.0, posinf=0.0, neginf=0.0))

    assert torch.equal(packed, expected)


@pytest.mark.parametrize(
    "weight_shape",
    [
        torch.tensor(256),
        torch.tensor([[1, 256]]),
        torch.tensor([1.0, 256.0]),
        torch.tensor([0, 256]),
    ],
)
def test_iq2_xs_rejects_invalid_shape_metadata(weight_shape):
    packed = torch.zeros((1, 1, 74), dtype=torch.uint8)

    with pytest.raises(ValueError, match=r"weight_shape|logical weight shape"):
        dequantize_iq2_xs(packed, weight_shape)


def test_iq2_xs_rejects_scalar_packed_payload():
    with pytest.raises(ValueError, match="packed_weights"):
        dequantize_iq2_xs(torch.tensor(0, dtype=torch.uint8), torch.tensor([1, 256]))


def test_iq2_xs_fake_quant_has_pass_through_gradient():
    class Quantizer:
        num_bits = "iq2_xs"

    weight = torch.randn(1, 256, requires_grad=True)
    output = iq2_xs_fake_quant(weight, Quantizer())
    output.sum().backward()

    assert torch.equal(weight.grad, torch.ones_like(weight))


def test_iq2_xs_saturates_finite_values_above_the_float32_range():
    """float64 weights are accepted, so a finite value too large for float32 must saturate.

    Converting before sanitizing would turn it into infinity and then zero, which silently
    encodes a large weight as nothing and diverges from the CUDA ``load_float`` policy.
    """
    weight = torch.randn(1, 256, dtype=torch.float64)
    weight[0, 7] = 1e100
    saturated = weight.clone()
    saturated[0, 7] = torch.finfo(torch.float32).max
    zeroed = weight.clone()
    zeroed[0, 7] = 0.0

    packed, _ = quantize_iq2_xs(weight)

    assert torch.equal(packed, quantize_iq2_xs(saturated)[0])
    assert not torch.equal(packed, quantize_iq2_xs(zeroed)[0])
