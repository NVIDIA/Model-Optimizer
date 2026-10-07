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

"""Tests for FP8QTensor quantization on CPU."""

import pytest
import torch

from modelopt.torch.quantization.qtensor import FP8QTensor


@pytest.mark.parametrize(
    ("quant_kwargs", "dequant_kwargs"),
    [
        ({}, {}),
        ({"axis": 0}, {}),
        ({"block_sizes": {-1: 8}}, {"block_sizes": {-1: 8}}),
    ],
)
def test_fp8_quantize_all_zero_block(quant_kwargs, dequant_kwargs):
    """A block, channel or tensor that is all zeros must dequantize to zeros, not NaN."""
    torch.manual_seed(0)
    weight = torch.randn(4, 16)
    weight[0] = 0
    if not quant_kwargs:
        weight = torch.zeros(4, 16)

    qtensor, scale = FP8QTensor.quantize(weight, **quant_kwargs)
    dequantized = qtensor.dequantize(dtype=torch.float32, scale=scale, **dequant_kwargs)

    assert not torch.isnan(dequantized).any()
    assert torch.equal(dequantized[0], torch.zeros(16))
    # Small values can underflow to FP8 subnormals or zero, so allow a small absolute error.
    torch.testing.assert_close(dequantized, weight, rtol=0.07, atol=1e-3)
