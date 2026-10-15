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

"""Tests for INT4 quantization of small FP16 blocks."""

import pytest
import torch

from modelopt.torch.quantization.qtensor import INT4QTensor


@pytest.mark.parametrize("block_size", [64, 128])
@pytest.mark.parametrize("amax", [2**-14, 2**-20, 2**-24])
def test_int4_quantize_keeps_small_fp16_blocks(block_size, amax):
    x = torch.tensor([[amax, -amax] * (block_size // 2)], dtype=torch.float16)

    qtensor, scale = INT4QTensor.quantize(x, block_size)
    dequantized = qtensor.dequantize(scale=scale, block_sizes={-1: block_size})

    assert torch.isfinite(scale).all()
    torch.testing.assert_close(scale, torch.full((1, 1), 7 / amax), rtol=0, atol=0)
    torch.testing.assert_close(dequantized, x, rtol=0, atol=0)
