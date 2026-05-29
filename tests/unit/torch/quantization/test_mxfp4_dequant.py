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

"""Tests for the standalone MXFP4-packed -> BF16 dequantization entry point.

These cover:

* A. mathematical correctness — LUT, nibble order, per-group scale,
  UE8M0 exponent math.
* B. layout correctness — DS 2D shape and rank-agnostic prefix.
* C. randomized cross-validation against an explicit byte-by-byte decoder
  local to this test.
"""

import pytest
import torch

from modelopt.torch.quantization.qtensor.mxfp4_tensor import MXFP4QTensor

# Signed E2M1 value table indexed by 4-bit pattern (bit 3 = sign, bits 2..0 = magnitude).
E2M1_SIGNED = [
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
]


def _pack_fp4(indices: torch.Tensor) -> torch.Tensor:
    """Pack ``(..., K)`` tensor of 4-bit nibble indices into ``(..., K//2)`` uint8.

    Convention matches the checkpoint layout: low nibble = even element,
    high nibble = odd element (i.e. ``value[..., 2k]`` lives in the low
    4 bits of ``byte[..., k]``).
    """
    assert indices.shape[-1] % 2 == 0, "Last dim must be even"
    low = indices[..., 0::2].to(torch.uint8) & 0x0F
    high = indices[..., 1::2].to(torch.uint8) & 0x0F
    return low | (high << 4)


def _reference_dequantize_packed(
    blocks: torch.Tensor,
    scales: torch.Tensor,
    *,
    block_size: int = 32,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Straight-line reference decoder independent of MXFP4QTensor internals.

    It intentionally uses Python byte iteration instead of the vectorized implementation
    under test: unpack low/high nibbles, look up signed E2M1 values, then multiply each
    group by its UE8M0 power-of-two scale.
    """
    assert block_size % 2 == 0
    b = blocks.contiguous().view(torch.uint8).cpu()
    s = scales.contiguous().view(torch.uint8).cpu()
    assert b.shape[:-1] == s.shape[:-1]
    assert 2 * b.shape[-1] == s.shape[-1] * block_size

    bytes_per_group = block_size // 2
    flat_b = b.reshape(-1, b.shape[-1])
    flat_s = s.reshape(-1, s.shape[-1])
    rows: list[list[float]] = []
    for row_b, row_s in zip(flat_b, flat_s):
        row: list[float] = []
        for group_idx, scale_byte in enumerate(row_s.tolist()):
            scale = 2.0 ** (scale_byte - 127)
            start = group_idx * bytes_per_group
            end = start + bytes_per_group
            for byte in row_b[start:end].tolist():
                row.append(E2M1_SIGNED[byte & 0x0F] * scale)
                row.append(E2M1_SIGNED[(byte >> 4) & 0x0F] * scale)
        rows.append(row)

    out = torch.tensor(rows, dtype=torch.float32).reshape(*b.shape[:-1], b.shape[-1] * 2)
    return out.to(dtype)


class TestMathematicalCorrectness:
    def test_lut_exhaustive_all_16_values(self):
        """Every 4-bit pattern decodes to its E2M1 signed value at scale=2^0."""
        # 32 nibbles: pattern 0..15 twice so all 16 values appear in low and high nibble.
        indices = torch.arange(32, dtype=torch.uint8) % 16
        blocks = _pack_fp4(indices)  # shape (16,)
        scales = torch.tensor([127], dtype=torch.uint8)  # UE8M0 byte 127 -> 2^0
        out = MXFP4QTensor.dequantize_packed(blocks, scales, block_size=32, dtype=torch.float32)
        expected = torch.tensor([E2M1_SIGNED[i] for i in indices.tolist()], dtype=torch.float32)
        assert torch.equal(out, expected)

    def test_nibble_ordering_low_is_even(self):
        """Low nibble of each byte must correspond to the even element index."""
        # Byte[0] = (high=0xA, low=0x3) -> even=1.5, odd=-1.0
        # Byte[1] = (high=0xF, low=0x9) -> even=-0.5, odd=-6.0
        indices = torch.tensor([0x3, 0xA, 0x9, 0xF] + [0] * 28, dtype=torch.uint8)
        blocks = _pack_fp4(indices)
        scales = torch.tensor([127], dtype=torch.uint8)
        out = MXFP4QTensor.dequantize_packed(blocks, scales, block_size=32, dtype=torch.float32)
        assert out[0].item() == 1.5
        assert out[1].item() == -1.0
        assert out[2].item() == -0.5
        assert out[3].item() == -6.0

    def test_per_group_scale_independence(self):
        """Two adjacent groups of 32 use their own scales; values across the group
        boundary are scaled independently."""
        # Every fp4 index is 1 (value 0.5). 64 values -> 2 groups.
        indices = torch.full((64,), 1, dtype=torch.uint8)
        blocks = _pack_fp4(indices)
        # Group 0: 2^0; Group 1: 2^5 = 32
        scales = torch.tensor([127, 127 + 5], dtype=torch.uint8)
        out = MXFP4QTensor.dequantize_packed(blocks, scales, block_size=32, dtype=torch.float32)
        assert out[31].item() == 0.5
        assert out[32].item() == 0.5 * 32

    def test_ue8m0_scale_math(self):
        """Scale bytes decode as 2^(byte - 127). Verify at a few exponents."""
        # 32 nibbles: only first is index 1 (value 0.5), rest are 0.
        indices = torch.tensor([1] + [0] * 31, dtype=torch.uint8)
        blocks = _pack_fp4(indices)
        for byte, expected_multiplier in [(127, 1.0), (117, 2**-10), (137, 2**10), (97, 2**-30)]:
            scales = torch.tensor([byte], dtype=torch.uint8)
            out = MXFP4QTensor.dequantize_packed(blocks, scales, block_size=32, dtype=torch.float32)
            assert out[0].item() == pytest.approx(0.5 * expected_multiplier), f"byte={byte}"


class TestLayoutCorrectness:
    def test_ds_2d_expert_layout(self):
        """Canonical DS V4 expert-weight input: (M, K//2) bytes + (M, K//32) scales."""
        m, k, block_size = 4, 128, 32
        torch.manual_seed(0)
        indices = torch.randint(0, 16, (m, k), dtype=torch.uint8)
        blocks = _pack_fp4(indices)
        scales = torch.randint(100, 151, (m, k // block_size), dtype=torch.uint8)
        out = MXFP4QTensor.dequantize_packed(blocks, scales, block_size=block_size)
        assert out.shape == (m, k)
        assert out.dtype == torch.bfloat16

    def test_rank_agnostic_prefix(self):
        """3D prefix gives same result as 2D-flattened call on reshaped inputs."""
        e, m, k, block_size = 3, 4, 64, 32
        torch.manual_seed(1)
        indices = torch.randint(0, 16, (e, m, k), dtype=torch.uint8)
        blocks = _pack_fp4(indices)
        scales = torch.randint(100, 151, (e, m, k // block_size), dtype=torch.uint8)

        out_3d = MXFP4QTensor.dequantize_packed(blocks, scales, block_size=block_size)
        out_flat = MXFP4QTensor.dequantize_packed(
            blocks.reshape(e * m, k // 2),
            scales.reshape(e * m, k // block_size),
            block_size=block_size,
        )
        assert out_3d.shape == (e, m, k)
        assert torch.equal(out_3d.reshape(e * m, k), out_flat)


class TestCrossValidationWithIndependentReference:
    """Randomized checks against a byte-by-byte MXFP4 decoder local to this test."""

    def test_matches_reference_on_2d_ds_shape(self):
        """DS V4 experts arrive as 2D (M, K//2). Make sure the 2D path agrees too."""
        m, k, block_size = 16, 128, 32
        num_groups = k // block_size
        torch.manual_seed(7)
        blocks_2d = torch.randint(0, 256, (m, num_groups * 16), dtype=torch.uint8)
        scales_2d = torch.randint(100, 151, (m, num_groups), dtype=torch.uint8)

        ref = _reference_dequantize_packed(blocks_2d, scales_2d, block_size=block_size)
        out = MXFP4QTensor.dequantize_packed(blocks_2d, scales_2d, block_size=block_size)
        assert torch.equal(out, ref)
