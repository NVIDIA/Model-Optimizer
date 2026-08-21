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

# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""RoPE inversion and frequency-domain utilities for TriAttention.

These functions support the TriAttention scoring algorithm by:
- Inverting RoPE rotations to recover pre-RoPE Q/K representations
- Converting real-valued tensors to complex frequency-domain representations
- Building geometric offset sequences for multi-distance scoring
"""

from __future__ import annotations

import torch

__all__ = [
    "build_geometric_offsets",
    "invert_rope",
    "rotate_half",
    "to_complex_pairs",
]


def rotate_half(x: torch.Tensor, *, style: str = "half") -> torch.Tensor:
    """Rotate tensor for RoPE. Supports 'half' (front/back) and 'interleaved' (even/odd)."""
    if style == "interleaved":
        x_even = x[..., ::2]
        x_odd = x[..., 1::2]
        return torch.stack((-x_odd, x_even), dim=-1).flatten(-2)
    d = x.shape[-1] // 2
    x1 = x[..., :d]
    x2 = x[..., d:]
    return torch.cat((-x2, x1), dim=-1)


def invert_rope(
    rotated: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    scale: float,
    *,
    style: str = "half",
) -> torch.Tensor:
    """Invert RoPE rotation to recover pre-RoPE representation.

    Args:
        rotated: RoPE-rotated tensor.
        cos: Cosine table from rotary embedding.
        sin: Sine table from rotary embedding.
        scale: Attention scaling factor applied during RoPE.
        style: RoPE pairing style ('half' or 'interleaved').

    Returns:
        Pre-RoPE tensor with RoPE rotation undone.
    """
    if scale == 0:
        raise ValueError("attention scaling factor must be non-zero")
    scale_t = torch.tensor(scale, device=rotated.device, dtype=rotated.dtype)
    base = rotated / scale_t
    cos_unit = cos / scale_t
    sin_unit = sin / scale_t
    if style == "interleaved":
        even = base[..., ::2]
        odd = base[..., 1::2]
        cos_even = cos_unit[..., ::2]
        cos_odd = cos_unit[..., 1::2]
        sin_even = sin_unit[..., ::2]
        sin_odd = sin_unit[..., 1::2]
        det = cos_even * cos_odd + sin_even * sin_odd
        det = det.clamp_min(1e-12)
        orig_even = (even * cos_odd + odd * sin_even) / det
        orig_odd = (odd * cos_even - even * sin_odd) / det
        restored = torch.empty_like(base)
        restored[..., ::2] = orig_even
        restored[..., 1::2] = orig_odd
        return restored
    return base * cos_unit - rotate_half(base, style=style) * sin_unit


def to_complex_pairs(tensor: torch.Tensor, *, style: str = "half") -> torch.Tensor:
    """Convert real tensor to complex representation for frequency analysis.

    Maps head_dim real values to head_dim/2 complex values. For 'half' style:
    real part = first half of dimensions, imag part = second half.

    Args:
        tensor: Real-valued tensor with even last dimension.
        style: RoPE pairing style ('half' or 'interleaved').

    Returns:
        Complex tensor with last dimension halved.
    """
    if tensor.size(-1) % 2 != 0:
        raise ValueError("Head dimension must be even to form complex pairs")
    real_dtype = torch.float32 if tensor.dtype in (torch.bfloat16, torch.float16) else tensor.dtype
    tensor_real = tensor.to(dtype=real_dtype)
    if style == "interleaved":
        real = tensor_real[..., ::2].contiguous()
        imag = tensor_real[..., 1::2].contiguous()
        return torch.complex(real, imag)
    freq_count = tensor.shape[-1] // 2
    real = tensor_real[..., :freq_count].contiguous()
    imag = tensor_real[..., freq_count:].contiguous()
    return torch.complex(real, imag)


def build_geometric_offsets(max_length: int, device: torch.device) -> torch.Tensor:
    """Build geometric offset sequence [1, 2, 4, 8, ..., max_length].

    Used for multi-distance scoring in TriAttention — each offset represents a
    future distance at which the key's importance is evaluated.

    Args:
        max_length: Maximum offset value (must be >= 1).
        device: Device for the output tensor.

    Returns:
        1D float tensor of powers of 2 up to max_length.
    """
    if max_length < 1:
        raise ValueError("max_length must be >= 1")
    offsets: list[float] = []
    value = 1
    while value <= max_length:
        offsets.append(float(value))
        value *= 2
    return torch.tensor(offsets, device=device, dtype=torch.float32)
