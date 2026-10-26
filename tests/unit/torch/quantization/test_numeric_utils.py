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
"""Unit tests for ``modelopt.torch.quantization.utils.numeric_utils`` — the
closed-form MXFP4 -> NVFP4 cast numerics."""

from types import SimpleNamespace

import pytest
import torch

from modelopt.torch.quantization.utils import numeric_utils as nu

# ---------- mxfp4_to_nvfp4_global_amax --------------------------------------


def test_global_amax_basic_in_range():
    """Mixed in-range scales: m = k_max - 8, global_amax = 6*448*2^m, lossless = 100%."""
    # k values in [-3, 3] (spread = 6), all blocks lossless.
    k = torch.tensor([0, -3, 3, 1, -1, 2], dtype=torch.int32)
    e8m0 = (k + nu.E8M0_BIAS).to(torch.uint8)

    global_amax, info = nu.mxfp4_to_nvfp4_global_amax(e8m0)
    assert info["k_min"] == -3
    assert info["k_max"] == 3
    assert info["m"] == 3 - 8  # k_max - 8 = -5
    expected = 6.0 * 448.0 * 2.0 ** info["m"]
    assert global_amax == pytest.approx(expected)
    assert info["n_total_blocks"] == 6
    assert info["n_lossless_blocks"] == 6
    assert info["pct_lossless"] == pytest.approx(100.0)
    assert info["n_zero_blocks"] == 0


def test_global_amax_with_zero_blocks():
    """Zero (e8m0=0, k=-127) blocks should be ignored when computing k_max."""
    e8m0 = torch.tensor([0, 0, 130, 125], dtype=torch.uint8)  # ks: -127, -127, 3, -2
    global_amax, info = nu.mxfp4_to_nvfp4_global_amax(e8m0)
    assert info["k_max"] == 3  # ignores zero blocks
    assert info["n_zero_blocks"] == 2
    # Both nonzero blocks satisfy k_max - k_j <= 17, plus zero blocks count as
    # lossless because their reconstruction is 0 regardless of scale.
    assert info["n_lossless_blocks"] == 4


def test_global_amax_with_oor_blocks():
    """A block 18 powers below k_max is OOR (k_max - k = 18 > 17)."""
    # k values: 5, 5, -13 → spread = 18, last block is OOR.
    k = torch.tensor([5, 5, -13], dtype=torch.int32)
    e8m0 = (k + nu.E8M0_BIAS).to(torch.uint8)
    _, info = nu.mxfp4_to_nvfp4_global_amax(e8m0)
    assert info["k_max"] == 5
    assert info["n_total_blocks"] == 3
    assert info["n_lossless_blocks"] == 2  # the k=-13 block is OOR


def test_global_amax_all_zero():
    """All-zero scales should not crash; k_max defaults to 0."""
    e8m0 = torch.zeros(4, dtype=torch.uint8)
    global_amax, info = nu.mxfp4_to_nvfp4_global_amax(e8m0)
    assert info["k_min"] == 0 and info["k_max"] == 0
    assert info["n_zero_blocks"] == 4
    # All blocks count as "lossless" (their dequant is 0 regardless of scale).
    assert info["n_lossless_blocks"] == 4


# ---------- mxfp4_to_nvfp4_per_block_amax -----------------------------------


def _make_blocks_with_max_nibble(num_blocks: int, max_idx_per_block: list[int]) -> torch.Tensor:
    """Build a (num_blocks, 16) byte tensor where block i has E2M1 magnitude
    index ``max_idx_per_block[i]`` as its largest nibble; other nibbles are 0.

    Magnitude index goes in the low 3 bits of one nibble; we place it in the
    high nibble of byte 0 (so the first byte = (max_idx << 4)). Every other
    nibble is 0, so the block-wise max is exactly ``max_idx_per_block[i]``.
    """
    assert len(max_idx_per_block) == num_blocks
    blocks = torch.zeros((num_blocks, 16), dtype=torch.uint8)
    for i, idx in enumerate(max_idx_per_block):
        assert 0 <= idx < 8
        blocks[i, 0] = (idx & 0x07) << 4
    return blocks


def test_per_block_amax_in_range_returns_closed_form():
    """Every block in-range -> 6 * 2^k_j, regardless of actual nibble content."""
    # k = [0, -2, 4]; k_max = 4, k_min = -2, spread 6 (in-range).
    k = torch.tensor([0, -2, 4], dtype=torch.int32)
    e8m0 = (k + nu.E8M0_BIAS).to(torch.uint8)
    # Blocks have varying max_nibbles, but in-range path ignores them.
    blocks = _make_blocks_with_max_nibble(3, [3, 7, 1])  # max nibbles: 1.5, 6, 0.5

    out = nu.mxfp4_to_nvfp4_per_block_amax(blocks, e8m0)
    expected_mxfp4 = 6.0 * torch.exp2(k.float())  # ignores max_nibble
    expected_nvfp4 = expected_mxfp4.repeat_interleave(2, dim=-1)
    assert torch.allclose(out, expected_nvfp4)


def test_per_block_amax_oor_uses_data_derived():
    """OOR blocks should use ``max_nibble * 2^k_j`` (data-derived)."""
    # k_max=10 → m=2. OOR-low blocks have k_j - m < -9, i.e. k_j < -7.
    k = torch.tensor([10, -10], dtype=torch.int32)  # second is OOR-low
    e8m0 = (k + nu.E8M0_BIAS).to(torch.uint8)
    # Block 0 max nibble idx 7 (value 6); block 1 max nibble idx 4 (value 2).
    blocks = _make_blocks_with_max_nibble(2, [7, 4])

    out = nu.mxfp4_to_nvfp4_per_block_amax(blocks, e8m0)

    # Block 0 (in-range): 6 * 2^10 = 6144.
    # Block 1 (OOR):       2 * 2^-10 (max_nibble=2 since idx=4 -> 2.0).
    expected_mxfp4 = torch.tensor([6.0 * 2**10, 2.0 * 2**-10], dtype=torch.float32)
    expected_nvfp4 = expected_mxfp4.repeat_interleave(2, dim=-1)
    assert torch.allclose(out, expected_nvfp4)


def test_per_block_amax_doubles_last_dim():
    """Two NVFP4 blocks per MXFP4 block share the same per-block amax."""
    e8m0 = torch.tensor([130, 124], dtype=torch.uint8)  # ks: 3, -3
    blocks = _make_blocks_with_max_nibble(2, [7, 7])  # in-range
    out = nu.mxfp4_to_nvfp4_per_block_amax(blocks, e8m0)
    assert out.shape == (4,)
    # Each pair of consecutive entries should be equal.
    assert out[0] == out[1]
    assert out[2] == out[3]


def test_per_block_amax_preserves_leading_dims():
    """Leading dims (E, F, ...) flow through unchanged; only last dim doubles."""
    # shape (E=2, F=3, num_mxfp4_blocks=4)
    e8m0 = torch.full((2, 3, 4), 128, dtype=torch.uint8)  # all k=1, in-range
    blocks = torch.zeros((2, 3, 4, 16), dtype=torch.uint8)
    out = nu.mxfp4_to_nvfp4_per_block_amax(blocks, e8m0)
    assert out.shape == (2, 3, 8)


def test_per_block_amax_shape_mismatch_raises():
    """Mismatched leading dims should raise ``ValueError``."""
    blocks = torch.zeros((4, 16), dtype=torch.uint8)
    e8m0 = torch.zeros(3, dtype=torch.uint8)  # different num_blocks
    with pytest.raises(ValueError, match="shape mismatch"):
        nu.mxfp4_to_nvfp4_per_block_amax(blocks, e8m0)


# ---------- magnitude table cache ------------------------------------------


def test_e2m1_magnitude_table_cached_per_device():
    t1 = nu._e2m1_magnitude_table(torch.device("cpu"))
    t2 = nu._e2m1_magnitude_table(torch.device("cpu"))
    assert t1 is t2  # cached: same object
    assert t1.tolist() == nu._E2M1_MAGNITUDE


# ----------- fp8_max_for_normalization -------------------------------------
def test_fp8_max_for_normalization_default():
    """Without four_over_six, normalization max is the full E4M3 range (448)."""
    q = SimpleNamespace(block_sizes={-1: 16, "type": "static", "scale_bits": (4, 3)})
    assert nu.fp8_max_for_normalization(q) == nu.E4M3_MAX


def test_fp8_max_for_normalization_four_over_six():
    """With four_over_six enabled, normalization max is 256 (4/6 mode)."""
    q = SimpleNamespace(
        block_sizes={-1: 16, "type": "static", "scale_bits": (4, 3), "four_over_six": True}
    )
    assert nu.fp8_max_for_normalization(q) == nu.E4M3_MAX_46


def test_fp8_max_for_normalization_missing_block_sizes():
    """Missing or None block_sizes should fall back to the default E4M3 max."""
    assert nu.fp8_max_for_normalization(SimpleNamespace(block_sizes=None)) == nu.E4M3_MAX
    assert nu.fp8_max_for_normalization(SimpleNamespace()) == nu.E4M3_MAX


@pytest.mark.parametrize(
    ("four_over_six", "expected"),
    [
        (False, nu.E4M3_MAX),
        (True, nu.E4M3_MAX_46),
        (0, nu.E4M3_MAX),
        (1, nu.E4M3_MAX_46),
    ],
)
def test_fp8_max_for_normalization_truthy_flag(four_over_six, expected):
    """four_over_six is coerced with bool(); only truthy values select 256."""
    q = SimpleNamespace(block_sizes={"four_over_six": four_over_six})
    assert nu.fp8_max_for_normalization(q) == expected
