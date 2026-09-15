# This file includes the IQ1_S codebook adapted from:
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

"""IQ1_S fake quantization and GGML-compatible block packing.

The encoder follows the PSX-LUTS ``search_impl="auto"`` search. Every 256
logical values become one 50-byte ``block_iq1_s`` payload:

* bytes 0..1: little-endian FP16 super-block scale ``d``
* bytes 2..33: low eight bits of 32 codebook indices
* bytes 34..49: eight little-endian uint16 metadata words

Each metadata word describes four consecutive eight-value vectors. Bits 0..11
hold the three high index bits, bits 12..14 select one of eight local scales,
and bit 15 selects the shared -0.125 rather than +0.125 delta. The canonical
2048 x 8 ternary grid below comes from llama.cpp ``ggml-common.h`` revision
9b05354ec6fb58b4e665e9a39ebc40285c015638.
"""

import base64
import zlib
from functools import cache

import torch

from .common import GGML_BLOCK_SIZE, validate_packed_weights, validate_weight

__all__ = [
    "IQ1_S_BLOCK_BYTES",
    "IQ1_S_BLOCK_SIZE",
    "IQ1_S_EFFECTIVE_BITS",
    "dequantize_iq1_s",
    "iq1_s_fake_quant",
    "iq1_s_grid",
    "quantize_iq1_s",
]

IQ1_S_BLOCK_SIZE = GGML_BLOCK_SIZE
IQ1_S_BLOCK_BYTES = 50
IQ1_S_EFFECTIVE_BITS = IQ1_S_BLOCK_BYTES * 8 / IQ1_S_BLOCK_SIZE
_IQ1_S_DELTA = 0.125
_IQ1_S_NATIVE_MAX = 16.875

# zlib-compressed little-endian bytes of the canonical uint64_t table. The
# decoded int8 values are -1, 0, and 1.
_IQ1_S_GRID_ZLIB_B64 = (
    "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"
    "jDMYrMeWQNZ5e68uzuCwPq7TO3/ss/TvHzM5DA8="
)

_GRID_CACHE: dict[torch.device, torch.Tensor] = {}


@cache
def _grid_bytes() -> bytes:
    return zlib.decompress(base64.b64decode(_IQ1_S_GRID_ZLIB_B64))


def iq1_s_grid(device: torch.device | str | None = None) -> torch.Tensor:
    """Return the canonical IQ1_S ternary grid as float32."""
    resolved_device = torch.device(device or "cpu")
    if resolved_device not in _GRID_CACHE:
        raw = torch.tensor(list(_grid_bytes()), dtype=torch.uint8).view(torch.int8)
        _GRID_CACHE[resolved_device] = raw.reshape(2048, 8).to(
            device=resolved_device, dtype=torch.float32
        )
    return _GRID_CACHE[resolved_device]


def _encode_blocks(blocks: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    """Encode a moderate-size batch of flattened 256-value blocks."""
    x = blocks.float()
    block_count = x.shape[0]
    vectors = x.reshape(block_count, 32, 8)
    xnorm = vectors.square().sum(dim=-1)
    xsum = vectors.sum(dim=-1)

    amax = x.abs().amax(dim=1)
    d = ((amax / _IQ1_S_NATIVE_MAX) * 0.61).clamp(max=65504.0).to(torch.float16)
    d_float = d.float()

    best_error = torch.full((block_count, 32, 16), torch.inf, device=x.device)
    best_entry = torch.zeros((block_count, 32, 16), dtype=torch.int64, device=x.device)
    grid_norm = grid.square().sum(dim=-1)
    grid_sum = grid.sum(dim=-1)

    # Tile the 2048-entry codebook to bound temporary memory. A strict update
    # retains the lowest codebook index when two candidates have equal error.
    for entry_start in range(0, 2048, 128):
        grid_tile = grid[entry_start : entry_start + 128]
        dot = torch.matmul(vectors, grid_tile.T)
        tile_norm = grid_norm[entry_start : entry_start + 128].reshape(1, 1, -1)
        tile_sum = grid_sum[entry_start : entry_start + 128].reshape(1, 1, -1)

        for shift in range(2):
            delta = -_IQ1_S_DELTA if shift else _IQ1_S_DELTA
            shifted_dot = dot + delta * xsum.unsqueeze(-1)
            shifted_norm = tile_norm + 2 * delta * tile_sum + 8 * delta * delta
            for local in range(8):
                choice = shift * 8 + local
                scale = d_float.reshape(-1, 1, 1) * (2 * local + 1)
                error = (
                    xnorm.unsqueeze(-1) - 2 * scale * shifted_dot + scale.square() * shifted_norm
                )
                tile_error, tile_index = error.min(dim=-1)
                replace = tile_error < best_error[:, :, choice]
                best_error[:, :, choice] = torch.where(
                    replace, tile_error, best_error[:, :, choice]
                )
                best_entry[:, :, choice] = torch.where(
                    replace, tile_index + entry_start, best_entry[:, :, choice]
                )

    group_error = best_error.reshape(block_count, 8, 4, 16).sum(dim=2)
    selected_choice = group_error.argmin(dim=-1)
    vector_choice = selected_choice.repeat_interleave(4, dim=1)
    selected_entry = best_entry.gather(2, vector_choice.unsqueeze(-1)).squeeze(-1)
    selected_local = selected_choice & 0x7
    selected_shift = selected_choice >> 3

    high = (selected_entry >> 8).reshape(block_count, 8, 4)
    qh = (
        high[:, :, 0]
        | (high[:, :, 1] << 3)
        | (high[:, :, 2] << 6)
        | (high[:, :, 3] << 9)
        | (selected_local << 12)
        | (selected_shift << 15)
    )

    packed = torch.empty((block_count, IQ1_S_BLOCK_BYTES), dtype=torch.uint8, device=x.device)
    packed[:, :2] = d.contiguous().view(torch.uint8).reshape(block_count, 2)
    packed[:, 2:34] = (selected_entry & 0xFF).to(torch.uint8)
    packed[:, 34:50:2] = (qh & 0xFF).to(torch.uint8)
    packed[:, 35:50:2] = (qh >> 8).to(torch.uint8)
    packed[d_float == 0] = 0
    return packed


@torch.no_grad()
def quantize_iq1_s(
    weight: torch.Tensor, *, block_chunk_size: int = 4
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack a floating-point weight into GGML-compatible IQ1_S blocks.

    Returned shapes are ``[*weight.shape[:-1], weight.shape[-1] // 256, 50]``
    and ``[weight.ndim]``. Both tensors remain on the weight's device.
    """
    validate_weight(weight, "IQ1_S")
    if block_chunk_size <= 0:
        raise ValueError(f"block_chunk_size must be positive, got {block_chunk_size}")

    logical_shape = torch.tensor(weight.shape, dtype=torch.int64, device=weight.device)
    blocks = weight.contiguous().reshape(-1, IQ1_S_BLOCK_SIZE)
    grid = iq1_s_grid(weight.device)
    if weight.is_cuda:
        from ..extensions import get_cuda_ext_iq1_s

        extension = get_cuda_ext_iq1_s()
        if extension is not None:
            packed = extension.pack(blocks, grid)
            packed_shape = (
                *weight.shape[:-1],
                weight.shape[-1] // IQ1_S_BLOCK_SIZE,
                IQ1_S_BLOCK_BYTES,
            )
            return packed.reshape(packed_shape), logical_shape

    chunks = [
        _encode_blocks(blocks[start : start + block_chunk_size], grid)
        for start in range(0, blocks.shape[0], block_chunk_size)
    ]
    packed_shape = (
        *weight.shape[:-1],
        weight.shape[-1] // IQ1_S_BLOCK_SIZE,
        IQ1_S_BLOCK_BYTES,
    )
    return torch.cat(chunks).reshape(packed_shape), logical_shape


@torch.no_grad()
def dequantize_iq1_s(
    packed_weights: torch.Tensor,
    weight_shape: torch.Tensor,
    *,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Decode GGML-compatible IQ1_S payload bytes."""
    shape = validate_packed_weights(
        packed_weights, weight_shape, block_bytes=IQ1_S_BLOCK_BYTES, format_name="IQ1_S"
    )

    blocks = packed_weights.contiguous().reshape(-1, IQ1_S_BLOCK_BYTES)
    d = blocks[:, :2].contiguous().view(torch.float16).reshape(-1).float()
    low = blocks[:, 2:34].to(torch.int64).reshape(-1, 8, 4)
    qh = blocks[:, 34:50:2].to(torch.int64) | (blocks[:, 35:50:2].to(torch.int64) << 8)
    shifts = torch.tensor([0, 3, 6, 9], dtype=torch.int64, device=blocks.device)
    high = (qh.unsqueeze(-1) >> shifts) & 0x7
    entries = low | (high << 8)

    local = (qh >> 12) & 0x7
    delta = torch.where((qh & 0x8000).bool(), -_IQ1_S_DELTA, _IQ1_S_DELTA)
    values = iq1_s_grid(blocks.device)[entries] + delta.unsqueeze(-1).unsqueeze(-1)
    scales = d.unsqueeze(-1) * (2 * local + 1).float()
    decoded = values * scales.unsqueeze(-1).unsqueeze(-1)
    return decoded.reshape(shape).to(dtype)


def iq1_s_fake_quant(inputs: torch.Tensor, quantizer) -> torch.Tensor:
    """IQ1_S backend for TensorQuantizer, with pass-through backward."""
    if getattr(quantizer, "num_bits", None) != "iq1_s":
        raise ValueError("The psx_luts IQ1_S backend requires num_bits='iq1_s'")
    extra_args = getattr(quantizer, "backend_extra_args", None) or {}
    search_impl = extra_args.get("search_impl", extra_args.get("iq_search_impl", "auto"))
    if search_impl != "auto":
        raise NotImplementedError("Only IQ1_S search_impl='auto' is currently supported")
    packed, shape = quantize_iq1_s(inputs)
    reconstructed = dequantize_iq1_s(packed, shape, dtype=inputs.dtype)
    return inputs + (reconstructed - inputs).detach()
