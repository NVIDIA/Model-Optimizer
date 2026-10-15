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

"""Tests for INT8QTensor real quantization of small FP16/BF16 channels on CPU."""

import copy

import pytest
import torch

import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.qtensor import INT8QTensor

# Rows: amax 2^-20 (its FP16 scale amax / 127 rounds to zero), amax 1e-5 (FP16 subnormal
# scale, rounded down by ~25%), all zeros, and a normal channel.
WEIGHT = [
    [2**-20, -(2**-20), 0.0, 2**-21],
    [1e-5, -5e-6, 2e-6, 0.0],
    [0.0, 0.0, 0.0, 0.0],
    [1.0, -1.0, 0.0, 0.5],
]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    ("layout", "weight"),
    [
        ({"axis": 0}, WEIGHT),
        ({"block_sizes": {-1: 2}}, WEIGHT),
        # A per-tensor scale is 0-dim and does not promote `weight / scale` out of the dtype.
        ({}, [list(range(8))]),
    ],
)
def test_int8_quantize_matches_fp32(dtype, layout, weight):
    """Real-quantizing low-precision weights must give the FP32 result, cast to that dtype."""
    weight = torch.tensor(weight, dtype=dtype)
    block_sizes = layout.get("block_sizes")

    def round_trip(w):
        qtensor, scale = INT8QTensor.quantize(w, **layout)
        return qtensor.dequantize(dtype=dtype, scale=scale, block_sizes=block_sizes)

    dequantized = round_trip(weight)

    assert torch.equal(dequantized, round_trip(weight.float()))
    # No nonzero value here is small enough to round to zero against its own amax.
    assert torch.equal(dequantized == 0, weight == 0)


def test_int8_compress_keeps_small_fp16_channel():
    """mtq.compress must not zero a small FP16 channel that fake quantization keeps."""
    model = torch.nn.Sequential(torch.nn.Linear(4, 2, bias=False)).half()
    with torch.no_grad():
        model[0].weight.copy_(torch.tensor([WEIGHT[0], WEIGHT[3]]))
    x = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float16)

    mtq.quantize(model, copy.deepcopy(mtq.INT8_DEFAULT_CFG), lambda m: m(x))
    with torch.no_grad():
        fake_quant_output = model(x)
        mtq.compress(model)
        real_quant_output = model(x)

    assert fake_quant_output[0, 0] == 2**-20
    assert torch.equal(real_quant_output, fake_quant_output)
