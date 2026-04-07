# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Tests for TriAttention RoPE utilities."""

import pytest
import torch

from modelopt.torch.sparsity.kv_cache.triattention.rope_utils import (
    build_geometric_offsets,
    invert_rope,
    rotate_half,
    to_complex_pairs,
)


def test_rotate_half_roundtrip():
    """rotate_half applied twice returns the negated original."""
    x = torch.randn(2, 4, 8)
    result = rotate_half(rotate_half(x))
    torch.testing.assert_close(result, -x)


def test_invert_rope_recovers_original():
    """Inverting RoPE-rotated tensor recovers the pre-RoPE original."""
    head_dim = 16
    seq_len = 8
    x = torch.randn(1, seq_len, head_dim)

    freqs = torch.arange(head_dim // 2, dtype=torch.float32) / head_dim
    positions = torch.arange(seq_len, dtype=torch.float32).unsqueeze(-1)
    angles = positions * freqs
    cos = torch.cos(angles).repeat(1, 2).unsqueeze(0)
    sin = torch.sin(angles).repeat(1, 2).unsqueeze(0)
    scale = 1.0

    rotated = x * cos + rotate_half(x) * sin
    recovered = invert_rope(rotated * scale, cos, sin, scale)
    torch.testing.assert_close(recovered, x, atol=1e-5, rtol=1e-5)


def test_invert_rope_zero_scale_raises():
    """Zero scale raises ValueError."""
    with pytest.raises(ValueError, match="non-zero"):
        invert_rope(torch.randn(1, 4, 8), torch.ones(1, 4, 8), torch.ones(1, 4, 8), 0.0)


def test_to_complex_pairs_shape():
    """Complex pairs halves the last dimension."""
    c = to_complex_pairs(torch.randn(4, 16))
    assert c.shape == (4, 8)
    assert c.dtype == torch.complex64


def test_to_complex_pairs_values():
    """Half style: real = first half, imag = second half."""
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]])
    c = to_complex_pairs(x)
    assert c[0, 0].real == 1.0
    assert c[0, 0].imag == 5.0
    assert c[0, 3].real == 4.0
    assert c[0, 3].imag == 8.0


def test_to_complex_pairs_odd_dim_raises():
    """Odd head dimension raises ValueError."""
    with pytest.raises(ValueError, match="even"):
        to_complex_pairs(torch.randn(4, 7))


def test_build_geometric_offsets():
    """Geometric offsets are powers of 2 up to max_length."""
    offsets = build_geometric_offsets(16, torch.device("cpu"))
    expected = torch.tensor([1.0, 2.0, 4.0, 8.0, 16.0])
    torch.testing.assert_close(offsets, expected)


def test_build_geometric_offsets_single():
    """Single offset when max_length is 1."""
    offsets = build_geometric_offsets(1, torch.device("cpu"))
    torch.testing.assert_close(offsets, torch.tensor([1.0]))


def test_build_geometric_offsets_zero_raises():
    """max_length < 1 raises ValueError."""
    with pytest.raises(ValueError, match="must be >= 1"):
        build_geometric_offsets(0, torch.device("cpu"))
