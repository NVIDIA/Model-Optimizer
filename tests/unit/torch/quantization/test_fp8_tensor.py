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

"""FP8 checkpoint conversion with native UE8M0 block scales."""

import pytest
import torch

from modelopt.torch.quantization.qtensor import FP8QTensor


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_ue8m0_scale_boundary_neighbors(dtype):
    """The first value above a boundary requires the next scale, even near underflow."""
    exponents = torch.arange(-127, 128 if dtype == torch.float64 else 120)
    boundary = torch.ldexp(torch.full(exponents.shape, 448.0, dtype=dtype), exponents)
    weight = torch.stack(
        (
            torch.nextafter(boundary, torch.zeros_like(boundary)),
            boundary,
            torch.nextafter(boundary, torch.full_like(boundary, float("inf"))),
        ),
        dim=-1,
    )
    _, scales = FP8QTensor.quantize(weight, block_sizes={-1: 1}, scale_dtype=torch.float8_e8m0fnu)
    expected = torch.stack((exponents + 127, exponents + 127, exponents + 128), dim=-1)
    torch.testing.assert_close(scales.view(torch.uint8), expected.clamp(max=254).to(torch.uint8))


@pytest.mark.parametrize(("block", "row_blocks"), [(32, 2), (128, 34)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float64])
def test_ue8m0_roundtrip_partial_and_zero_blocks(block, row_blocks, dtype):
    """Known representable values retain their codes and scales, including zero blocks."""
    codes = torch.tensor([448.0, -224.0, 1.125, -0.5, 0.0], dtype=dtype).repeat(
        row_blocks * block, block
    )
    codes = codes[:, : 2 * block - 3]
    exponents = torch.tensor(
        [[122, -127], [127, 0]] if dtype == torch.float64 else [[-3, -127], [2, 0]]
    ).repeat(row_blocks // 2, 1)
    rows = torch.arange(codes.shape[0]) // block
    cols = torch.arange(codes.shape[1]) // block
    codes[rows % 2 == 0, block:] = 0
    weight = (codes * 2.0 ** exponents[rows[:, None], cols]).to(dtype)
    blocks = {-2: block, -1: block}

    quantized, scales = FP8QTensor.quantize(
        weight, block_sizes=blocks, scale_dtype=torch.float8_e8m0fnu
    )
    assert scales.dtype == torch.float8_e8m0fnu
    assert torch.equal(scales.view(torch.uint8), (exponents + 127).to(torch.uint8))
    assert torch.equal(quantized._quantized_data.float(), codes)
    restored = quantized.dequantize(scale=scales, block_sizes=blocks)
    assert restored.dtype == dtype
    assert torch.equal(restored, weight)

    requantized, new_scales = FP8QTensor.quantize(restored, scales=scales, block_sizes=blocks)
    assert torch.equal(
        requantized._quantized_data.view(torch.uint8), quantized._quantized_data.view(torch.uint8)
    )
    assert torch.equal(new_scales.view(torch.uint8), scales.view(torch.uint8))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_ue8m0_dequantizes_before_casting(dtype):
    """Extreme scales must not overflow/underflow in the output dtype before multiplication."""
    weight = torch.tensor([[0.5, 0.0, 0.125, 2.0, 2.0]], dtype=torch.float8_e4m3fn)
    scales = torch.tensor([[254, 254, 0, 254, 255]], dtype=torch.uint8).view(torch.float8_e8m0fnu)
    quantized = FP8QTensor(weight.shape, dtype, weight)
    result = quantized.dequantize(scale=scales, block_sizes={-1: 1})
    expected = torch.tensor(
        [[2.0**126, 0.0, 2.0**-130, 2.0**128, float("nan")]], dtype=torch.float64
    ).to(dtype)
    torch.testing.assert_close(result, expected, rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize("shape", [(4, 0), (0, 4), (0,)])
def test_ue8m0_empty_roundtrip(shape):
    weight = torch.empty(shape, dtype=torch.bfloat16)
    quantized, scales = FP8QTensor.quantize(
        weight, block_sizes={-1: 4}, scale_dtype=torch.float8_e8m0fnu
    )
    restored = quantized.dequantize(scale=scales, block_sizes={-1: 4})
    torch.testing.assert_close(restored, weight)


def test_ue8m0_nan_block():
    weight = torch.tensor([[float("nan"), 1.0], [448.0, 0.0]])
    quantized, scales = FP8QTensor.quantize(
        weight, block_sizes={-1: 2}, scale_dtype=torch.float8_e8m0fnu
    )
    assert scales.view(torch.uint8).tolist() == [[255], [127]]
    expected = torch.tensor([[float("nan"), float("nan")], [448.0, 0.0]])
    torch.testing.assert_close(
        quantized.dequantize(scale=scales, block_sizes={-1: 2}), expected, equal_nan=True
    )


@pytest.mark.parametrize("native_scales", [False, True])
def test_block_scales_for_batched_weights(native_scales):
    """The optional format preserves ordinary FP8 and also handles batched expert weights."""
    weight = torch.arange(1, 1 + 3 * 4 * 8, dtype=torch.float32).reshape(3, 4, 8)
    blocks = {-1: 4}
    options = {"scale_dtype": torch.float8_e8m0fnu} if native_scales else {}
    quantized, scales = FP8QTensor.quantize(weight, block_sizes=blocks, **options)
    actual_scales = scales.float()
    expanded = actual_scales.repeat_interleave(4, -1)
    expected_codes = (weight / expanded).to(torch.float8_e4m3fn)
    assert torch.equal(quantized._quantized_data.float(), expected_codes.float())
    torch.testing.assert_close(
        quantized.dequantize(scale=scales, block_sizes=blocks),
        expected_codes.float() * expanded,
        rtol=0,
        atol=0,
    )
    if not native_scales:
        torch.testing.assert_close(scales, weight.reshape(3, 4, 2, 4).amax(-1) / 448.0)


def test_ue8m0_rejects_inconsistent_scale_configuration():
    weight = torch.ones(4, 8)
    with pytest.raises(ValueError):
        FP8QTensor.quantize(weight, scale_dtype=torch.float8_e8m0fnu)
    with pytest.raises(ValueError):
        FP8QTensor.quantize(weight, block_sizes={-1: 4}, scale_dtype=torch.float16)
    with pytest.raises(ValueError):
        FP8QTensor.quantize(
            weight,
            scales=torch.ones(4, 2),
            block_sizes={-1: 4},
            scale_dtype=torch.float8_e8m0fnu,
        )
    quantized = FP8QTensor(weight.shape, weight.dtype, weight.to(torch.float8_e4m3fn))
    with pytest.raises(AssertionError):
        quantized.dequantize(scale=torch.ones(4, 3).to(torch.float8_e8m0fnu), block_sizes={-1: 4})
