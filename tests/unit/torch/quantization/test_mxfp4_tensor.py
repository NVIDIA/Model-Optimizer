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

"""CPU tests for MXFP4 E8M0 scale handling and packed export."""

import pytest
import torch

from modelopt.torch.quantization.qtensor import MXFP4QTensor


def test_e8m0_encode_decode_clamps_and_rounds_up():
    scale = torch.tensor([0.0, 2.0**-128, 0.75, 1.0, 1.01, 2.0**128])

    encoded = MXFP4QTensor.encode_e8m0_scale(scale)
    decoded = MXFP4QTensor.decode_e8m0_scale(encoded)

    assert encoded.tolist() == [0, 0, 127, 127, 128, 254]
    assert torch.equal(
        decoded,
        torch.tensor([2.0**-127, 2.0**-127, 1.0, 1.0, 2.0, 2.0**127]),
    )


def test_e8m0_projection_has_identity_gradient():
    scale = torch.tensor([0.75, 1.01], requires_grad=True)

    projected = MXFP4QTensor.project_e8m0_scale_ste(scale)
    projected.sum().backward()

    assert torch.equal(projected, torch.tensor([1.0, 2.0]))
    assert torch.equal(scale.grad, torch.ones_like(scale))


def test_quantize_with_scale_controls_packed_values():
    weight = torch.cat(
        [
            torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]),
            torch.zeros(24),
        ]
    ).reshape(1, 32)
    scale = MXFP4QTensor.encode_e8m0_scale(torch.tensor([[1.0]]))

    quantized = MXFP4QTensor.quantize_with_scale(weight, scale, block_size=32)
    restored = quantized.dequantize(scale=scale, block_sizes={-1: 32})

    # Existing MXFP4 encoding preserves zero as the negative-zero nibble (0x8).
    assert quantized._quantized_data[0, :4].tolist() == [24, 50, 84, 118]
    assert torch.equal(restored, weight)


def test_quantize_with_scale_rejects_wrong_scale_count():
    with pytest.raises(ValueError, match="Expected 2 E8M0 scales"):
        MXFP4QTensor.quantize_with_scale(
            torch.zeros(2, 32), torch.ones(1, dtype=torch.uint8), block_size=32
        )


@pytest.mark.parametrize("block_size", [0, -1, 2.0])
def test_quantize_with_scale_rejects_invalid_block_size(block_size):
    with pytest.raises(ValueError, match="positive integer"):
        MXFP4QTensor.quantize_with_scale(
            torch.zeros(1, 32), torch.ones(1, dtype=torch.uint8), block_size=block_size
        )


def test_quantize_with_scale_rejects_partial_last_dimension():
    with pytest.raises(ValueError, match="last dimension 16"):
        MXFP4QTensor.quantize_with_scale(
            torch.zeros(2, 16), torch.ones(1, dtype=torch.uint8), block_size=32
        )


def test_quantize_with_scale_rejects_non_e8m0_scale():
    with pytest.raises(ValueError, match=r"torch\.uint8"):
        MXFP4QTensor.quantize_with_scale(torch.zeros(1, 32), torch.ones(1), block_size=32)
