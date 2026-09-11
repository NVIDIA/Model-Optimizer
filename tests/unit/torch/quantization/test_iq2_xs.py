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


def test_iq2_xs_zero_block_has_canonical_zero_encoding():
    weight = torch.zeros((2, 256), dtype=torch.bfloat16)

    packed, shape = quantize_iq2_xs(weight)

    assert packed.shape == (2, 1, IQ2_XS_BLOCK_BYTES)
    assert packed.dtype == torch.uint8
    assert not packed.any()
    assert shape.tolist() == [2, 256]
    assert torch.equal(dequantize_iq2_xs(packed, shape), weight)


def test_iq2_xs_round_trip_and_payload_fields():
    generator = torch.Generator().manual_seed(1234)
    weight = torch.randn((2, 512), generator=generator, dtype=torch.bfloat16)

    packed, shape = quantize_iq2_xs(weight, block_chunk_size=2)
    reconstructed = dequantize_iq2_xs(packed, shape)

    assert packed.shape == (2, 2, 74)
    assert reconstructed.shape == weight.shape
    assert reconstructed.dtype == torch.bfloat16
    normalized_mse = (
        reconstructed.float() - weight.float()
    ).square().mean() / weight.float().square().mean()
    assert normalized_mse < 0.1

    blocks = packed.reshape(-1, 74)
    codes = blocks[:, 2:66:2].to(torch.int64) | (blocks[:, 3:66:2].to(torch.int64) << 8)
    assert torch.all((codes & 0x1FF) < 512)
    assert torch.all((codes >> 9) < 128)


def test_iq2_xs_requires_complete_last_dimension_blocks():
    with pytest.raises(ValueError, match="last weight dimension"):
        quantize_iq2_xs(torch.ones(2, 257))


def test_iq2_xs_fake_quant_has_pass_through_gradient():
    class Quantizer:
        num_bits = "iq2_xs"
        backend_extra_args = {"search_impl": "auto"}

    weight = torch.randn(1, 256, requires_grad=True)
    output = iq2_xs_fake_quant(weight, Quantizer())
    output.sum().backward()

    assert torch.equal(weight.grad, torch.ones_like(weight))
