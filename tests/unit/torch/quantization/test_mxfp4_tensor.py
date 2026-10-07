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

"""Tests for MXFP4QTensor quantization on CPU."""

import torch

from modelopt.torch.quantization.qtensor import MXFP4QTensor


def test_mxfp4_quantize_rounds_half_to_even():
    """Values halfway between two E2M1 values round to the even code, like the CUDA kernel."""
    bounds = torch.tensor([0, 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5])
    expected = torch.tensor([0, 0, 1, 1, 2, 2, 4, 4], dtype=torch.float32)
    sign = torch.tensor([1, -1, 1, -1, -1, 1, -1, 1])
    weight = (bounds * sign).repeat(4).reshape(1, 32)

    qtensor, scale = MXFP4QTensor.quantize(weight, block_size=32)
    dequantized = qtensor.dequantize(dtype=torch.float32, scale=scale, block_sizes={-1: 32})

    assert torch.equal(scale, torch.tensor([[127]], dtype=torch.uint8))
    assert torch.equal(dequantized, (expected * sign).repeat(4).reshape(1, 32))


def test_mxfp4_quantize_zero_has_no_sign_bit():
    weight = torch.zeros(1, 32)
    weight[0, 0] = 6.0

    qtensor, _ = MXFP4QTensor.quantize(weight, block_size=32)

    codes = torch.stack([qtensor._quantized_data & 0xF, qtensor._quantized_data >> 4], dim=-1)
    assert torch.equal(codes.flatten()[1:], torch.zeros(31, dtype=torch.uint8))
