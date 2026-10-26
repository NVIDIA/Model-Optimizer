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

"""NF4 double quantization tests that span more than one int8 scale group.

With ``scale_bits=8`` the per-block scales are stored as int8, grouped by
``scale_block_sizes``. Each group carries its own ``double_scale``, so the scales have to be
dequantized group-wise before the zero padding at the tail is trimmed. When the input has
enough blocks to span several groups the dequantize step used to broadcast the flattened
scale vector against the group axis and blow up.
"""

from __future__ import annotations

import pytest
import torch

from modelopt.torch.quantization.config import QuantizerAttributeConfig
from modelopt.torch.quantization.nn import TensorQuantizer

# 4 scales per int8 group, so 8 blocks of 2 elements give 2 groups.
_BLOCK_SIZES = {-1: 2, "scale_bits": 8, "scale_block_sizes": {-1: 4}}


def _dequantize(x: torch.Tensor, block_sizes: dict) -> torch.Tensor:
    quantizer = TensorQuantizer(
        QuantizerAttributeConfig(num_bits=4, block_sizes=block_sizes, fake_quant=False)
    )
    # The first call does the real quantize, the second one dequantizes back.
    return quantizer(quantizer(x))


@pytest.mark.parametrize(
    ("test_input", "test_output"),
    [
        # 16 elements -> 8 blocks -> 2 scale groups.
        (
            torch.arange(16, dtype=torch.bfloat16).view(1, 16),
            torch.tensor(
                [
                    [
                        0.0,
                        1.0,
                        2.1875,
                        3.0312,
                        3.6094,
                        5.0,
                        5.0625,
                        7.0,
                        9.0,
                        9.0,
                        11.0,
                        11.0,
                        13.0,
                        13.0,
                        15.0,
                        15.0,
                    ]
                ],
                dtype=torch.bfloat16,
            ),
        ),
        # Same but the tail has to be padded: 15 elements -> 8 blocks -> 2 scale groups.
        (
            torch.arange(15, dtype=torch.bfloat16).view(1, 15),
            torch.tensor(
                [
                    [
                        0.0,
                        1.0,
                        2.1719,
                        3.0,
                        3.6094,
                        5.0,
                        5.0625,
                        7.0,
                        9.0,
                        9.0,
                        11.0,
                        11.0,
                        13.0,
                        13.0,
                        14.0,
                    ]
                ],
                dtype=torch.bfloat16,
            ),
        ),
    ],
)
def test_nf4_double_quant_multiple_scale_groups(test_input, test_output):
    """Dequantizing a tensor with several int8 scale groups matches the expected values."""
    quantizer = TensorQuantizer(
        QuantizerAttributeConfig(num_bits=4, block_sizes=_BLOCK_SIZES, fake_quant=False)
    )
    deq_x = quantizer(quantizer(test_input))

    # Guard the premise of this test: a single group would not exercise the group axis.
    assert quantizer._scale.shape[0] > 1

    assert deq_x.shape == test_input.shape
    assert torch.equal(deq_x, test_output)


def test_nf4_double_quant_roundtrip_wide_input():
    """A wide input spanning many scale groups round-trips without blowing up."""
    torch.manual_seed(0)
    block_sizes = {-1: 16, "scale_bits": 8, "scale_block_sizes": {-1: 4}}
    x = torch.rand(256, 32, dtype=torch.bfloat16)

    quantizer = TensorQuantizer(
        QuantizerAttributeConfig(num_bits=4, block_sizes=block_sizes, fake_quant=False)
    )
    deq_x = quantizer(quantizer(x))

    # 512 blocks / 4 scales per group -> 128 groups.
    assert quantizer._scale.shape[0] == 128
    assert deq_x.shape == x.shape
    assert torch.allclose(deq_x, x, rtol=1e-1, atol=1e-1)
