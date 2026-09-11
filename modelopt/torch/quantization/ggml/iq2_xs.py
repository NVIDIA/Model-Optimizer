# This file includes the IQ2_XS codebook adapted from:
# https://github.com/ggml-org/llama.cpp/blob/9b05354ec6fb58b4e665e9a39ebc40285c015638/ggml/src/ggml-common.h
#
# MIT License
#
# Copyright (c) 2023-2026 The ggml authors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 AND MIT
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

"""IQ2_XS fake quantization and GGML-compatible block packing.

The encoder follows the PSX-LUTS search_impl="auto" search. Every 256
logical values become one 74-byte block_iq2_xs payload:

* bytes 0..1: little-endian FP16 super-block scale d
* bytes 2..65: 32 little-endian uint16 codes (9-bit grid + 7-bit sign)
* bytes 66..73: 16 four-bit local scales, two per byte

The canonical 512 x 8 magnitude grid below comes from llama.cpp
ggml-common.h revision 9b05354ec6fb58b4e665e9a39ebc40285c015638.
"""

import base64
from functools import cache

import torch

from .common import GGML_BLOCK_SIZE, validate_packed_weights, validate_weight

__all__ = [
    "IQ2_XS_BLOCK_BYTES",
    "IQ2_XS_BLOCK_SIZE",
    "IQ2_XS_EFFECTIVE_BITS",
    "dequantize_iq2_xs",
    "iq2_xs_fake_quant",
    "iq2_xs_grid",
    "quantize_iq2_xs",
]

IQ2_XS_BLOCK_SIZE = GGML_BLOCK_SIZE
IQ2_XS_BLOCK_BYTES = 74
IQ2_XS_EFFECTIVE_BITS = IQ2_XS_BLOCK_BYTES * 8 / IQ2_XS_BLOCK_SIZE

# Compact byte representation of the canonical [512, 8] grid. Values are only
# 8, 25, and 43. Keeping this as checkpoint-independent package data avoids
# adding a pickle-backed torch.save artifact to the wheel.
_IQ2_XS_GRID_B64 = (
    "CAgICAgICAgrCAgICAgICBkZCAgICAgICCsICAgICAgrKwgICAgICBkIGQgICAgICBkZCAgICAgrGRkICAgICBkrGQgICAgICAgr"
    "CAgICAgrCCsICAgICBkZKwgICAgICCsrCAgICAgZCAgZCAgICAgZCBkICAgIKxkIGQgICAgZKwgZCAgICAgIGRkICAgIKwgZGQgI"
    "CAgZGRkZCAgICAgrGRkICAgIGQgrGQgICAgIGSsZCAgICAgICCsICAgIKwgIKwgICAgZGQgrCAgICAgrCCsICAgIGQgZKwgICAgI"
    "GRkrCAgICBkrGSsICAgICAgrKwgICAgZCAgIGQgICAgZCAgZCAgIKxkICBkICAgZKwgIGQgICAgIGQgZCAgIKwgZCBkICAgZGRkI"
    "GQgICAgrGQgZCAgIKysZCBkICAgZCCsIGQgICAgZKwgZCAgICAgIGRkICAgrCAgZGQgICBkZCBkZCAgICCsIGRkICAgZCBkZGQgI"
    "CAgZGRkZCAgICAgrGRkICAgIKysZGQgICBkICCsZCAgICBkIKxkICAgICBkrGQgICAgICAgrCAgIKwgICCsICAgZGQgIKwgICAgr"
    "CAgrCAgIGQgZCCsICAgIGRkIKwgICAgIKwgrCAgIGQgIGSsICAgIGQgZKwgICAgIGRkrCAgIGRkZGSsICAgICAgrKwgICCsrCCsr"
    "CAgIGQgICAgZCAgIGQgICBkICCsZCAgIGQgIGSsICAgZCAgICBkICBkICCsIGQgIGQgIGRkZCAgZCAgIKxkICBkICBkIKwgIGQgI"
    "CBkrCAgZCAgICAgZCBkICCsICBkIGQgIGRkIGQgZCAgIKwgZCBkICBkIGRkIGQgICBkZGQgZCAgrGRkZCBkICAgIKxkIGQgIGQgI"
    "KwgZCAgIGQgrCBkICAgIGSsIGQgICAgICBkZCAgrCAgIGRkICBkZCAgZGQgICCsICBkZCAgZCBkIGRkICAgZGQgZGQgICAgrCBkZ"
    "CAgZCAgZGRkICAgZCBkZGQgICAgZGRkZCAgZCCsZGRkICAgICCsZGQgIGQgICCsZCAgIGQgIKxkICAgIGQgrGQgIKxkrCCsZCAgI"
    "CAgZKxkICCsICBkrGQgICBkIKysZCAgICAgICCsICCsICAgIKwgIGRkICAgrCAgIKwgICCsICCsrCAgIKwgIGQgZCAgrCAgIGRkI"
    "CCsICAgIKwgIKwgIGRkrCAgrCAgZCAgZCCsICAgZCBkIKwgICAgZGQgrCAgIKxkZCCsICAgICCsIKwgICAgrKwgrCAgrKysrCCsI"
    "CBkICAgZKwgICBkICBkrCAgICBkIGSsICAgICBkZKwgIGQgIKxkrCAgZKwgrGSsICAgICAgrKwgICAgrCCsrCAgIKysIKysICCsZ"
    "GSsrKwgICAgrKysrCAgZCAgICAgZCAgZCAgICBkIKxkICAgIGQgZKwgICAgZCAgIGQgICBkIKwgZCAgIGQgZGRkICAgZCAgrGQgI"
    "CBkIGQgrCAgIGQgIGSsICAgZCAgICBkICBkIKwgIGQgIGQgZGQgZCAgZCAgrCBkICBkIGQgZGQgIGQgIGRkZCAgZCAgIKxkICBkI"
    "KysrGQgIGQgZCAgrCAgZCAgZCCsICBkICAgZKwgIGQgICAgIGQgZCCsICAgZCBkIGRkICBkIGQgIKwgIGQgZCBkIGQgZCBkICBkZ"
    "CBkIGQgICCsIGQgZCBkICBkZCBkICBkIGRkIGQgICBkZGQgZCAgICCsZCBkICBkZKxkIGQgrGRkrGQgZCBkICAgrCBkICBkICCsI"
    "GQgrGQgIKwgZCAgIGQgrCBkICAgIGSsIGQgICCsZKwgZCAgICAgIGRkIKwgICAgZGQgZGQgICBkZCAgrCAgIGRkIGQgZCAgZGQgI"
    "GRkICBkZCAgIKwgIGRkIGQgIGQgZGQgIGQgZCBkZCBkrCBkIGRkICAgZGQgZGQgIGSsZCBkZCAgICCsIGRkIGQgICBkZGQgIGQgI"
    "GRkZCAgIGQgZGRkICAgIGRkZGQgICAgIKxkZCAgZGQgrGRkIGSsIGSsZGQgZCAgICCsZCAgZCAgIKxkICAgZCAgrGQgrCBkICCsZ"
    "CAgICBkIKxkICBkZGQgrGQgrGQgrCCsZCAgICAgZKxkIGRkICBkrGQgrGSsZGSsZCBkIGRkrKxkIGSsrKysrGQgICAgICAgrCCsI"
    "CAgICCsIGRkICAgIKwgIKwgICAgrCCsrCAgICCsIGQgZCAgIKwgIGRkICAgrCAgIKwgICCsIGQgIGQgIKwgIGQgZCAgrCAgIGRkI"
    "CCsICAgIKwgIKwgICCsrCAgrCBkICAgZCCsICBkICBkIKwgICBkIGQgrCAgICBkZCCsICCsIGRkIKwgZGSsZGQgrCAgICAgrCCsI"
    "KwgrCCsIKwgICAgrKwgrCAgrKysrCCsIGQgICAgZKwgIGQgICBkrCAgIGQgIGSsIGSsrCAgZKwgICAgZCBkrCAgICAgZGSsIGQgI"
    "GRkZKwgrCBkZGRkrCBkrGSsZGSsIGQgICCsZKwgrKxkIKxkrCCsZKysrGSsICAgICAgrKwgIKwgICCsrCCsrCAgIKysICAgrCAgr"
    "KwgZGRkZCCsrCAgrCCsIKysIKwgrKwgrKwgIKysZGSsrCAgIGSsZKysICCsICCsrKwgICCsIKysrCCsICCsrKysICCsIKysrKwgr"
    "KwgrKysrCBkICAgICAgZCBkICAgICBkrGQgICAgIGRkrCAgICAgZCAgZCAgICBkrCBkICAgIGRkZGQgICAgZCCsZCAgICBkZCCsI"
    "CAgIGQgZKwgICAgZCAgIGQgICBkrCAgZCAgIGRkZCBkICAgZCCsIGQgICBkrKwgZCAgIGRkIGRkICAgZCBkZGQgICBkICCsZCAgI"
    "GRkZKxkICAgZGQgIKwgICBkIGQgrCAgIGQgIGSsICAgZCAgICBkICBkrCAgIGQgIGRkZCAgZCAgZCCsICBkICBkZCBkIGQgIGQgZ"
    "GQgZCAgZCAgrCBkICBkZCAgZGQgIGQgZCBkZCAgZCAgZGRkICBkICAgrGQgIGRkZCCsZCAgZKwgrKxkICBkZCAgIKwgIGQgZCAgr"
    "CAgZCAgZCCsICBkrCBkIKwgIGRkrKwgrCAgZCAgIGSsICBkICAgICBkIGSsICAgIGQgZGRkICAgZCBkIKwgICBkIGRkIGQgIGQgZ"
    "CBkZCAgZCBkZKxkICBkIGQgIKwgIGQgZGQgIGQgZCBkIGQgZCBkIGQgIGRkIGQgZCAgIKwgZCBkIGRkrCBkIGRkICAgZGQgZCBkI"
    "CBkZCBkICBkIGRkIGQgZKwgZGQgZCAgIGRkZCBkrKxkrGRkIGQgICAgrGQgZKysICCsZCBkIGQgZKxkIGQgIGRkrGQgZGQgICAgr"
    "CBkIGQgICCsIGQgIGQgIKwgZCAgIGQgrCBkZGQgZCCsIGQgZGRkIKwgZKwgrGQgrCBkICAgIGSsIGRkIGQgZKwgZCBkIGRkrCBkI"
    "CBkZGSsIGRkrKxkZKwgZCBkICCsrCBkICAgICAgZGSsICAgICBkZGRkICAgIGRkIKwgICAgZGRkIGQgICBkZCBkZCAgIGRkICCsI"
    "CAgZGQgrKwgICBkZGQgIGQgIGRkIGQgZCAgZGQgIGRkICBkZCAgIKwgIGRkZCAgIGQgZGQgZCAgZCBkZCAgZCBkIGRkZGRkIGQgZ"
    "GQgICBkZCBkZKwgIGRkIGRkICAgIKwgZGQgZCBkrCBkZKysrKysIGRkZCAgICBkZGQgZCAgIGRkZCAgZCAgZGRkZCCsICBkZGQgI"
    "CBkIGRkZCAgrGQgZGRkZCAgrCBkZGRkIKysIGRkZCAgICBkZGRkIKwgIGRkZGQgICCsZGRkZCCsIKxkZGRkZCCsIKxkZGQgrKxkr"
    "GRkZGQgrKysZGRkICAgICCsZGQgZGQgIKxkZGQgIGQgrGRkICBkZCCsZGRkrGSsIKxkZKysZCBkrGRkICAgZGSsZGSsICBkZKxkZ"
    "GRkIKysrGRkZCAgICAgrGQgZCAgICCsZCAgZCAgIKxkICAgZCAgrGQgZGRkICCsZKwgrGQgIKxkrGQgrCAgrGRkrKysICCsZCAgI"
    "CBkIKxkIGSsIKwgrGSsrCBkrCCsZKwgZKysIKxkICAgICBkrGSsZGQgIGSsZCAgZCBkZKxkICAgZGRkrGRkZCBkZGSsZCBkrKxkZ"
    "KxkZCAgICCsrGSsrKxkIKysZGRkrCBkrKxkrGQgIKysrGQgZGRkrKysZKwgrGSsrKxkICAgICAgIKysICAgICAgrGRkICAgICCsI"
    "KwgICAgIKxkIGQgICAgrCBkZCAgICCsICCsICAgIKysrKwgICAgrGQgIGQgICCsIGQgZCAgIKwgIGRkICAgrCAgIKwgICCsrCAgr"
    "CAgIKwgrKysICAgrKysrKwgICCsZCAgIGQgIKwgZCAgZCAgrKxkICBkICCsICBkIGQgIKwgICBkZCAgrGQgZGRkICCsZKxkZGQgI"
    "KwgICAgrCAgrCAgrCCsICCsICAgrKwgIKysICCsrCAgrCAgrKysICCsIKysrKwgIKxkICAgIGQgrCBkICAgZCCsICBkICBkIKysI"
    "GQgIGQgrGRkZCAgZCCsICAgZCBkIKwgIKxkIGQgrGSsIKwgZCCsICAgIGRkIKwgZCBkZGQgrGRkrKxkZCCsIKxkIKxkIKysrKxkr"
    "GQgrCAgICAgrCCsIKwgICCsIKxkZKwgIKwgrKysZGQgrCCsICAgrCCsIKysICCsIKwgrCCsrKwgrCCsrGQgIGSsIKysIKwgrKwgr"
    "CAgIKysrCCsIKwgrKysIKysZGSsrKwgrCCsrKysrCCsZCAgICAgZKwgZCAgICBkrCAgZCAgIGSsICAgZCAgZKysZGRkICBkrCBkI"
    "KwgIGSsICAgIGQgZKysIKwgZCBkrCBkrGRkIGSsrGRkZKwgZKxkrCCsrCBkrCAgICAgZGSsZGQgICBkZKwgZCBkIGRkrCAgZGQgZ"
    "GSsIKxkZCBkZKxkrKwgZGRkrCAgZKxkZGSsrCBkrGRkZKxkICBkrGRkrGQgZGQgrGSsrGSsrCCsZKxkrCBkZKxkrGRkZCCsrGSsI"
    "CCsZKysZKwgICAgICCsrKwgICAgIKysIKwgICAgrKysrCAgICCsrCAgrCAgIKysrKysICAgrKwgIKysICCsrGQgZGRkIKysZKxkZ"
    "GQgrKysZKysZCCsrCAgICCsIKysrCAgIKwgrKwgrCAgrCCsrKysrCCsIKysICAgrKwgrKwgIKysrCCsrCAgIGQgZKysZGRkrCBkr"
    "KxkZKxkrGSsrCCsZKysZKysrKwgICCsrKwgIKwgIKysrKwgrCAgrKysIKysICCsrKwgIKysIKysrCCsrKwgrKysIGQgIGSsrKwgZ"
    "CCsZKysrKxkIKxkrKysIKysIKysrKysrKwgrKysrGQgZKysrKysrKysrKysrKw=="
)

_GRID_CACHE: dict[torch.device, torch.Tensor] = {}


@cache
def _grid_bytes() -> bytes:
    return base64.b64decode(_IQ2_XS_GRID_B64)


def iq2_xs_grid(device: torch.device | str | None = None) -> torch.Tensor:
    """Return the canonical IQ2_XS magnitude grid as float32."""
    resolved_device = torch.device(device or "cpu")
    if resolved_device not in _GRID_CACHE:
        values = torch.tensor(list(_grid_bytes()), dtype=torch.float32)
        _GRID_CACHE[resolved_device] = values.reshape(512, 8).to(device=resolved_device)
    return _GRID_CACHE[resolved_device]


def _encode_blocks(blocks: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    """Encode a moderate-size batch of flattened 256-value blocks."""
    x = blocks.float()
    block_count = x.shape[0]
    vectors = x.reshape(block_count, 32, 8)
    magnitudes = vectors.abs()
    negative = vectors < 0
    odd_parity = negative.sum(dim=-1).remainder(2).bool()

    amax = x.abs().amax(dim=1)
    rms = x.square().mean(dim=1).sqrt()
    peak_to_rms = torch.where(rms > 0, amax / rms, torch.zeros_like(rms))
    anchor_ratio = (1.0 - 0.035 * peak_to_rms).clamp(0.65, 0.92)
    d = ((amax / 166.625) * anchor_ratio).clamp(max=65504.0).to(torch.float16)
    d_float = d.float()

    xnorm = vectors.square().sum(dim=-1)
    qnorm = grid.square().sum(dim=-1)
    best_error = torch.full((block_count, 32, 16), torch.inf, dtype=torch.float32, device=x.device)
    best_entry = torch.zeros((block_count, 32, 16), dtype=torch.int64, device=x.device)
    # Search the codebook in tiles to cap temporary memory. Strict comparison
    # preserves the lowest grid index on equal error, matching the CUDA key.
    for entry_start in range(0, 512, 64):
        grid_tile = grid[entry_start : entry_start + 64]
        products = magnitudes.unsqueeze(2) * grid_tile.reshape(1, 1, -1, 8)
        dot = products.sum(dim=-1)
        dot = torch.where(odd_parity.unsqueeze(-1), dot - 2.0 * products.amin(dim=-1), dot)
        tile_qnorm = qnorm[entry_start : entry_start + 64].reshape(1, 1, -1)

        for local in range(16):
            scale = d_float.reshape(-1, 1, 1) * ((2 * local + 1) / 8.0)
            error = (
                xnorm.unsqueeze(-1) - 2.0 * scale * dot + scale.square() * tile_qnorm
            ).clamp_min_(0)
            tile_error, tile_index = error.min(dim=-1)
            replace = tile_error < best_error[:, :, local]
            best_error[:, :, local] = torch.where(replace, tile_error, best_error[:, :, local])
            best_entry[:, :, local] = torch.where(
                replace, tile_index + entry_start, best_entry[:, :, local]
            )

    group_error = best_error.reshape(block_count, 16, 2, 16).sum(dim=2)
    selected_local = group_error.argmin(dim=-1)
    vector_local = selected_local.repeat_interleave(2, dim=1)
    selected_entry = best_entry.gather(2, vector_local.unsqueeze(-1)).squeeze(-1)

    selected_grid = grid[selected_entry]
    weakest_index = (magnitudes * selected_grid).argmin(dim=-1)
    flip = torch.nn.functional.one_hot(weakest_index, num_classes=8).bool()
    encoded_negative = negative ^ (flip & odd_parity.unsqueeze(-1))
    sign_bits = torch.arange(8, dtype=torch.int64, device=x.device)
    sign_mask = (encoded_negative.to(torch.int64) << sign_bits).sum(dim=-1)

    codes = selected_entry | ((sign_mask & 0x7F) << 9)
    packed = torch.empty((block_count, IQ2_XS_BLOCK_BYTES), dtype=torch.uint8, device=x.device)
    packed[:, :2] = d.contiguous().view(torch.uint8).reshape(block_count, 2)
    packed[:, 2:66:2] = (codes & 0xFF).to(torch.uint8)
    packed[:, 3:66:2] = (codes >> 8).to(torch.uint8)
    packed[:, 66:] = (selected_local[:, 0::2] | (selected_local[:, 1::2] << 4)).to(torch.uint8)
    return packed


@torch.no_grad()
def quantize_iq2_xs(
    weight: torch.Tensor, *, block_chunk_size: int = 64
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack a floating-point weight into GGML-compatible IQ2_XS blocks.

    Returned shapes are [*weight.shape[:-1], weight.shape[-1] // 256, 74]
    and [weight.ndim]. Both tensors remain on the weight's device.
    """
    validate_weight(weight, "IQ2_XS")
    if block_chunk_size <= 0:
        raise ValueError(f"block_chunk_size must be positive, got {block_chunk_size}")

    logical_shape = torch.tensor(weight.shape, dtype=torch.int64, device=weight.device)
    blocks = weight.contiguous().reshape(-1, IQ2_XS_BLOCK_SIZE)
    grid = iq2_xs_grid(weight.device)
    if weight.is_cuda:
        from ..extensions import get_cuda_ext_iq2_xs

        extension = get_cuda_ext_iq2_xs()
        if extension is not None:
            packed = extension.pack(blocks, grid)
            packed_shape = (
                *weight.shape[:-1],
                weight.shape[-1] // IQ2_XS_BLOCK_SIZE,
                IQ2_XS_BLOCK_BYTES,
            )
            return packed.reshape(packed_shape), logical_shape

    chunks = [
        _encode_blocks(blocks[start : start + block_chunk_size], grid)
        for start in range(0, blocks.shape[0], block_chunk_size)
    ]
    packed_shape = (
        *weight.shape[:-1],
        weight.shape[-1] // IQ2_XS_BLOCK_SIZE,
        IQ2_XS_BLOCK_BYTES,
    )
    return torch.cat(chunks).reshape(packed_shape), logical_shape


@torch.no_grad()
def dequantize_iq2_xs(
    packed_weights: torch.Tensor,
    weight_shape: torch.Tensor,
    *,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Decode GGML-compatible IQ2_XS payload bytes."""
    shape = validate_packed_weights(
        packed_weights, weight_shape, block_bytes=IQ2_XS_BLOCK_BYTES, format_name="IQ2_XS"
    )

    blocks = packed_weights.contiguous().reshape(-1, IQ2_XS_BLOCK_BYTES)
    d = blocks[:, :2].contiguous().view(torch.float16).reshape(-1).float()
    codes = blocks[:, 2:66:2].to(torch.int64) | (blocks[:, 3:66:2].to(torch.int64) << 8)
    entries = codes & 0x1FF
    sign_index = codes >> 9

    parity = torch.zeros_like(sign_index)
    for bit in range(7):
        parity ^= (sign_index >> bit) & 1
    sign_mask = sign_index | (parity << 7)
    bit_positions = torch.arange(8, dtype=torch.int64, device=blocks.device)
    signs = 1.0 - 2.0 * ((sign_mask.unsqueeze(-1) >> bit_positions) & 1).float()

    scale_bytes = blocks[:, 66:].to(torch.int64)
    local = torch.empty((blocks.shape[0], 16), dtype=torch.int64, device=blocks.device)
    local[:, 0::2] = scale_bytes & 0x0F
    local[:, 1::2] = scale_bytes >> 4
    scales = d.unsqueeze(-1) * (2 * local + 1).float() / 8.0
    values = iq2_xs_grid(blocks.device)[entries] * signs
    decoded = values * scales.repeat_interleave(2, dim=1).unsqueeze(-1)
    return decoded.reshape(shape).to(dtype)


def iq2_xs_fake_quant(inputs: torch.Tensor, quantizer) -> torch.Tensor:
    """IQ2_XS backend for TensorQuantizer, with pass-through backward."""
    if getattr(quantizer, "num_bits", None) != "iq2_xs":
        raise ValueError("The psx_luts IQ2_XS backend requires num_bits='iq2_xs'")
    extra_args = getattr(quantizer, "backend_extra_args", None) or {}
    search_impl = extra_args.get("search_impl", extra_args.get("iq_search_impl", "auto"))
    if search_impl != "auto":
        raise NotImplementedError("Only IQ2_XS search_impl='auto' is currently supported")
    packed, shape = quantize_iq2_xs(inputs)
    reconstructed = dequantize_iq2_xs(packed, shape, dtype=inputs.dtype)
    return inputs + (reconstructed - inputs).detach()
