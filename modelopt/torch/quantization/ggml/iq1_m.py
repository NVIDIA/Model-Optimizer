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

"""IQ1_M fake quantization and GGML-compatible block packing.

The encoder performs a single-pass squared-error grid search at a fixed,
empirically anchored super-block scale, mirroring :mod:`.iq1_s`. Every 256
logical values become one 56-byte block_iq1_m payload:

* bytes 0..31: 32 low bytes of the grid index, four per sub-block
* bytes 32..47: 16 bytes holding, per nibble, three grid-index high bits and
  one delta-shift bit -- two groups per byte
* bytes 48..55: four little-endian uint16 holding four 3-bit local scales each
  in bits 0..11, and one nibble of the FP16 super-block scale in bits 12..15

IQ1_M has no dedicated ``d`` field: the FP16 super-block scale is reassembled
from the top nibble of each of the four scale words. It is also finer-grained
than IQ1_S -- a local scale covers two groups rather than four, and the delta
shift is chosen per group rather than per sub-block -- which is where its extra
0.1875 bits per weight go.

The grid is the same canonical 2048 x 8 ternary table IQ1_S uses, carried from
llama.cpp ggml-common.h revision 9b05354ec6fb58b4e665e9a39ebc40285c015638.
The matching dequantization formula is in ggml-quants.c at the same revision:
https://github.com/ggml-org/llama.cpp/blob/9b05354ec6fb58b4e665e9a39ebc40285c015638/ggml/src/ggml-quants.c#L2573-L2620
"""

import torch

from ..extensions import get_cuda_ext_ggml
from .common import (
    GGML_BLOCK_SIZE,
    fake_quantize_with_cache,
    narrow_to_float32,
    validate_block_chunk_size,
    validate_packed_weights,
    validate_weight,
)
from .iq1_s import iq1_s_grid

__all__ = [
    "IQ1_M_BLOCK_BYTES",
    "IQ1_M_BLOCK_SIZE",
    "IQ1_M_EFFECTIVE_BITS",
    "dequantize_iq1_m",
    "iq1_m_fake_quant",
    "iq1_m_grid",
    "quantize_iq1_m",
]

IQ1_M_BLOCK_SIZE = GGML_BLOCK_SIZE
IQ1_M_BLOCK_BYTES = 56
IQ1_M_EFFECTIVE_BITS = IQ1_M_BLOCK_BYTES * 8 / IQ1_M_BLOCK_SIZE
_IQ1_M_DELTA = 0.125
# Largest representable magnitude: (1 + delta) at local scale 7 -> 15 * 1.125.
_IQ1_M_NATIVE_MAX = 16.875
# IQ1_M anchors differently from IQ1_S: the ratio rises with a block's peak-to-RMS
# instead of being flat, and it is allowed closer to full range. Values follow the
# reference predictor these encoders are derived from.
_IQ1_M_SCALE_ANCHOR_BASE = 0.58
_IQ1_M_SCALE_ANCHOR_MIN = 0.65
_IQ1_M_SCALE_ANCHOR_MAX = 0.95
_IQ1_M_PEAK_TO_RMS_TAPER = 0.035
_IQ1_M_GROUPS = 32
_IQ1_M_SUBBLOCKS = 8
_DEFAULT_BLOCK_CHUNK_SIZE = 1024
_DEFAULT_DECODE_CHUNK_SIZE = 4096
_SCALE_BLOCK_CHUNK_SIZE = 4096


def iq1_m_grid(device: torch.device | str | None = None) -> torch.Tensor:
    """Return the canonical IQ1_M ternary grid as float32.

    IQ1_M indexes the same 2048-entry table as IQ1_S; this alias exists so every format
    exposes a grid accessor under its own name.
    """
    return iq1_s_grid(device)


def _predict_iq1_m_scales(blocks: torch.Tensor) -> torch.Tensor:
    """Predict one FP16 super-block scale for each flattened block."""
    x = narrow_to_float32(blocks)
    amax = x.abs().amax(dim=1)
    rms = x.square().mean(dim=1).sqrt()
    peak_to_rms = torch.where(rms > 0, amax / rms, torch.zeros_like(rms))
    anchor_ratio = (_IQ1_M_SCALE_ANCHOR_BASE + _IQ1_M_PEAK_TO_RMS_TAPER * peak_to_rms).clamp(
        _IQ1_M_SCALE_ANCHOR_MIN, _IQ1_M_SCALE_ANCHOR_MAX
    )
    return ((amax / _IQ1_M_NATIVE_MAX) * anchor_ratio).clamp(max=65504.0).to(torch.float16)


def _encode_blocks(blocks: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    """Encode a moderate-size batch of flattened 256-value blocks."""
    x = narrow_to_float32(blocks)
    block_count = x.shape[0]
    vectors = x.reshape(block_count, _IQ1_M_GROUPS, 8)
    xnorm = vectors.square().sum(dim=-1)
    xsum = vectors.sum(dim=-1)

    d = _predict_iq1_m_scales(x)
    d_float = d.float()

    # Choice index is shift * 8 + local, as in IQ1_S, but here the shift is free per group.
    shape = (block_count, _IQ1_M_GROUPS, 16)
    best_error = torch.full(shape, torch.inf, dtype=torch.float32, device=x.device)
    best_entry = torch.zeros(shape, dtype=torch.int64, device=x.device)
    grid_norm = grid.square().sum(dim=-1)
    grid_sum = grid.sum(dim=-1)

    for entry_start in range(0, 2048, 128):
        grid_tile = grid[entry_start : entry_start + 128]
        dot = torch.matmul(vectors, grid_tile.T)
        tile_norm = grid_norm[entry_start : entry_start + 128].reshape(1, 1, -1)
        tile_sum = grid_sum[entry_start : entry_start + 128].reshape(1, 1, -1)

        for shift in range(2):
            delta = -_IQ1_M_DELTA if shift else _IQ1_M_DELTA
            shifted_dot = dot + delta * xsum.unsqueeze(-1)
            shifted_norm = tile_norm + 2 * delta * tile_sum + 8 * delta * delta
            for local in range(8):
                choice = shift * 8 + local
                scale = d_float.reshape(-1, 1, 1) * (2 * local + 1)
                error = (
                    xnorm.unsqueeze(-1) - 2 * scale * shifted_dot + scale.square() * shifted_norm
                ).clamp_min_(0)
                tile_error, tile_index = error.min(dim=-1)
                replace = tile_error < best_error[:, :, choice]
                best_error[:, :, choice] = torch.where(
                    replace, tile_error, best_error[:, :, choice]
                )
                best_entry[:, :, choice] = torch.where(
                    replace, tile_index + entry_start, best_entry[:, :, choice]
                )

    # Each group picks its own shift; only the 3-bit local scale is shared, over two groups.
    per_shift_error = best_error.reshape(block_count, _IQ1_M_GROUPS, 2, 8)
    per_shift_entry = best_entry.reshape(block_count, _IQ1_M_GROUPS, 2, 8)
    local_error, shift_index = per_shift_error.min(dim=2)
    local_entry = per_shift_entry.gather(2, shift_index.unsqueeze(2)).squeeze(2)

    pair_error = local_error.reshape(block_count, 16, 2, 8).sum(dim=2)
    selected_local = pair_error.argmin(dim=-1)
    group_local = selected_local.repeat_interleave(2, dim=1)
    selected_entry = local_entry.gather(2, group_local.unsqueeze(-1)).squeeze(-1)
    selected_shift = shift_index.gather(2, group_local.unsqueeze(-1)).squeeze(-1)

    packed = torch.empty((block_count, IQ1_M_BLOCK_BYTES), dtype=torch.uint8, device=x.device)
    packed[:, :32] = (selected_entry & 0xFF).to(torch.uint8)

    high = (selected_entry >> 8).reshape(block_count, _IQ1_M_SUBBLOCKS, 4)
    shift_bits = selected_shift.reshape(block_count, _IQ1_M_SUBBLOCKS, 4)
    qh = torch.empty((block_count, _IQ1_M_SUBBLOCKS, 2), dtype=torch.int64, device=x.device)
    for half in range(2):
        low_group, high_group = 2 * half, 2 * half + 1
        qh[:, :, half] = (
            (high[:, :, low_group] & 0x7)
            | (shift_bits[:, :, low_group] << 3)
            | ((high[:, :, high_group] & 0x7) << 4)
            | (shift_bits[:, :, high_group] << 7)
        )
    packed[:, 32:48] = qh.reshape(block_count, 16).to(torch.uint8)

    # Four scale words, each carrying two sub-blocks' local scales plus one nibble of d.
    d_bits = d.contiguous().view(torch.int16).to(torch.int64) & 0xFFFF
    locals_by_sub = selected_local.reshape(block_count, _IQ1_M_SUBBLOCKS, 2)
    words = torch.zeros((block_count, 4), dtype=torch.int64, device=x.device)
    for word in range(4):
        for parity in range(2):
            sub = 2 * word + parity
            base = 6 * parity
            words[:, word] |= locals_by_sub[:, sub, 0] << base
            words[:, word] |= locals_by_sub[:, sub, 1] << (base + 3)
        words[:, word] |= ((d_bits >> (4 * word)) & 0xF) << 12
    packed[:, 48:56:2] = (words & 0xFF).to(torch.uint8)
    packed[:, 49:56:2] = ((words >> 8) & 0xFF).to(torch.uint8)
    return torch.where((d_float == 0).unsqueeze(1), 0, packed)


@torch.no_grad()
def quantize_iq1_m(
    weight: torch.Tensor, *, block_chunk_size: int = _DEFAULT_BLOCK_CHUNK_SIZE
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack a floating-point weight into GGML-compatible IQ1_M blocks.

    Returned shapes are ``[*weight.shape[:-1], weight.shape[-1] // 256, 56]``
    and ``[weight.ndim]``.
    """
    validate_weight(weight, "IQ1_M")
    validate_block_chunk_size(block_chunk_size)

    logical_shape = torch.tensor(weight.shape, dtype=torch.int64)
    blocks = weight.contiguous().reshape(-1, IQ1_M_BLOCK_SIZE)
    grid = iq1_s_grid(weight.device)
    packed_shape = (*weight.shape[:-1], weight.shape[-1] // IQ1_M_BLOCK_SIZE, IQ1_M_BLOCK_BYTES)
    if weight.is_cuda:
        extension = get_cuda_ext_ggml()
        if extension is not None:
            scale_chunks = [
                _predict_iq1_m_scales(blocks[start : start + _SCALE_BLOCK_CHUNK_SIZE])
                for start in range(0, blocks.shape[0], _SCALE_BLOCK_CHUNK_SIZE)
            ]
            packed = extension.iq1_m_pack(blocks, grid, torch.cat(scale_chunks))
            return packed.reshape(packed_shape), logical_shape

    chunks = [
        _encode_blocks(blocks[start : start + block_chunk_size], grid)
        for start in range(0, blocks.shape[0], block_chunk_size)
    ]
    return torch.cat(chunks).reshape(packed_shape), logical_shape


@torch.no_grad()
def dequantize_iq1_m(
    packed_weights: torch.Tensor,
    weight_shape: torch.Tensor,
    *,
    dtype: torch.dtype = torch.bfloat16,
    block_chunk_size: int = _DEFAULT_DECODE_CHUNK_SIZE,
) -> torch.Tensor:
    """Decode GGML-compatible IQ1_M payload bytes."""
    shape = validate_packed_weights(
        packed_weights, weight_shape, block_bytes=IQ1_M_BLOCK_BYTES, format_name="IQ1_M"
    )
    validate_block_chunk_size(block_chunk_size)

    blocks = packed_weights.contiguous().reshape(-1, IQ1_M_BLOCK_BYTES)
    grid = iq1_s_grid(blocks.device)
    decoded = torch.empty((blocks.shape[0], IQ1_M_BLOCK_SIZE), dtype=dtype, device=blocks.device)
    for start in range(0, blocks.shape[0], block_chunk_size):
        stop = min(start + block_chunk_size, blocks.shape[0])
        block_chunk = blocks[start:stop]
        count = block_chunk.shape[0]
        low = block_chunk[:, :32].to(torch.int64).reshape(count, _IQ1_M_SUBBLOCKS, 4)
        qh = block_chunk[:, 32:48].to(torch.int64).reshape(count, _IQ1_M_SUBBLOCKS, 2)
        words = block_chunk[:, 48:56:2].to(torch.int64) | (
            block_chunk[:, 49:56:2].to(torch.int64) << 8
        )

        # The FP16 super-block scale is the four words' top nibbles, low word first.
        d_bits = (
            (words[:, 0] >> 12)
            | ((words[:, 1] >> 8) & 0x00F0)
            | ((words[:, 2] >> 4) & 0x0F00)
            | (words[:, 3] & 0xF000)
        )
        d = d_bits.to(torch.int16).view(torch.float16).float()

        entries = torch.empty((count, _IQ1_M_SUBBLOCKS, 4), dtype=torch.int64, device=blocks.device)
        deltas = torch.empty(
            (count, _IQ1_M_SUBBLOCKS, 4), dtype=torch.float32, device=blocks.device
        )
        for group in range(4):
            byte = qh[:, :, group // 2]
            nibble_shift = 4 * (group % 2)
            entries[:, :, group] = low[:, :, group] | (((byte >> nibble_shift) & 0x7) << 8)
            negative = (byte >> (nibble_shift + 3)) & 1
            deltas[:, :, group] = torch.where(
                negative.bool(), -_IQ1_M_DELTA, torch.tensor(_IQ1_M_DELTA, device=blocks.device)
            )

        scales = torch.empty(
            (count, _IQ1_M_SUBBLOCKS, 2), dtype=torch.float32, device=blocks.device
        )
        for sub in range(_IQ1_M_SUBBLOCKS):
            word = words[:, sub // 2]
            base = 6 * (sub % 2)
            scales[:, sub, 0] = d * (2 * ((word >> base) & 0x7) + 1).float()
            scales[:, sub, 1] = d * (2 * ((word >> (base + 3)) & 0x7) + 1).float()

        values = grid[entries] + deltas.unsqueeze(-1)
        # Groups 0,1 take the first local scale and groups 2,3 the second, so the repeat
        # runs along the half axis: [h0, h0, h1, h1], not [h0, h1, h0, h1].
        group_scale = scales.repeat_interleave(2, dim=2)
        chunk_decoded = values * group_scale.unsqueeze(-1)
        decoded[start:stop] = chunk_decoded.reshape(-1, IQ1_M_BLOCK_SIZE)
    return decoded.reshape(shape)


def iq1_m_fake_quant(
    inputs: torch.Tensor,
    quantizer,
    *,
    block_chunk_size: int = _DEFAULT_BLOCK_CHUNK_SIZE,
    decode_chunk_size: int = _DEFAULT_DECODE_CHUNK_SIZE,
) -> torch.Tensor:
    """IQ1_M weight backend for TensorQuantizer, with pass-through backward."""
    if getattr(quantizer, "num_bits", None) != "iq1_m":
        raise ValueError("The ggml IQ1_M backend requires num_bits='iq1_m'")
    return fake_quantize_with_cache(
        inputs,
        quantizer,
        format_name="iq1_m",
        block_chunk_size=block_chunk_size,
        decode_chunk_size=decode_chunk_size,
        quantize=quantize_iq1_m,
        dequantize=dequantize_iq1_m,
    )
