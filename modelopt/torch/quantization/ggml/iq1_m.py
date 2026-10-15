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
empirically anchored super-block scale, mirroring :mod:`.iq1_s`, including its
optional per-input-column importance. Every 256 logical values become one
56-byte block_iq1_m payload:

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
    GGMLFormat,
    chunk_importance,
    importance_blocks,
    narrow_to_float32,
    validate_block_chunk_size,
    validate_packed_weights,
    validate_weight,
)
from .iq1_s import _search_shifted_grid, iq1_s_grid

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


def _encode_blocks(
    blocks: torch.Tensor, grid: torch.Tensor, weights: torch.Tensor | None = None
) -> torch.Tensor:
    """Encode a moderate-size batch of flattened 256-value blocks."""
    x = narrow_to_float32(blocks)
    block_count = x.shape[0]
    d = _predict_iq1_m_scales(x)
    d_float = d.float()
    best_error, best_entry = _search_shifted_grid(
        x.reshape(block_count, _IQ1_M_GROUPS, 8), d_float, grid, weights
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

    # Each group's nibble is three index-high bits and its shift bit; two groups share a byte,
    # low nibble first.
    nibbles = ((selected_entry >> 8) & 0x7) | (selected_shift << 3)
    packed[:, 32:48] = (nibbles[:, 0::2] | (nibbles[:, 1::2] << 4)).to(torch.uint8)

    # Four scale words, each carrying four 3-bit local scales (two sub-blocks, two halves each)
    # in bits 0..11 and one nibble of d in bits 12..15, low nibble in the first word. The fields
    # are disjoint, so summing them is the same as OR-ing them.
    d_bits = d.contiguous().view(torch.int16).to(torch.int64) & 0xFFFF
    local_shifts = torch.tensor([0, 3, 6, 9], dtype=torch.int64, device=x.device)
    nibble_shifts = torch.tensor([0, 4, 8, 12], dtype=torch.int64, device=x.device)
    words = (selected_local.reshape(block_count, 4, 4) << local_shifts).sum(dim=-1)
    words |= ((d_bits.unsqueeze(-1) >> nibble_shifts) & 0xF) << 12
    packed[:, 48:56:2] = (words & 0xFF).to(torch.uint8)
    packed[:, 49:56:2] = ((words >> 8) & 0xFF).to(torch.uint8)
    return torch.where((d_float == 0).unsqueeze(1), 0, packed)


@torch.no_grad()
def quantize_iq1_m(
    weight: torch.Tensor,
    *,
    block_chunk_size: int = _DEFAULT_BLOCK_CHUNK_SIZE,
    importance: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack a floating-point weight into GGML-compatible IQ1_M blocks.

    Returned shapes are ``[*weight.shape[:-1], weight.shape[-1] // 256, 56]``
    and ``[weight.ndim]``. ``importance``, a non-negative per-input-column vector, weights each
    value's squared error in the search, as llama.cpp's imatrix does.
    """
    validate_weight(weight, "IQ1_M")
    validate_block_chunk_size(block_chunk_size)
    block_importance = importance_blocks(importance, weight, "IQ1_M")

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
            packed = extension.iq1_m_pack(blocks, grid, torch.cat(scale_chunks), block_importance)
            return packed.reshape(packed_shape), logical_shape

    chunks = [
        _encode_blocks(
            blocks[start : start + block_chunk_size],
            grid,
            chunk_importance(
                block_importance, start, min(start + block_chunk_size, blocks.shape[0])
            ),
        )
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
    if blocks.is_cuda:
        extension = get_cuda_ext_ggml()
        if extension is not None:
            grid = iq1_s_grid(blocks.device)
            return extension.iq1_m_unpack(blocks, grid, dtype).reshape(shape)
    grid = iq1_s_grid(blocks.device)
    scale_shifts = torch.tensor([0, 3, 6, 9], dtype=torch.int64, device=blocks.device)
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

        # Each qh byte carries two groups, low nibble first: three high index bits and a delta sign.
        nibbles = torch.stack((qh & 0xF, qh >> 4), dim=-1).reshape(count, _IQ1_M_SUBBLOCKS, 4)
        entries = low | ((nibbles & 0x7) << 8)
        deltas = torch.where((nibbles & 0x8).bool(), -_IQ1_M_DELTA, _IQ1_M_DELTA)

        # Each word packs four 3-bit local scales below its nibble of d: two sub-blocks, two
        # halves each, so word w, slot k is sub-block 2 * w + k // 2, half k % 2.
        local = ((words.unsqueeze(-1) >> scale_shifts) & 0x7).reshape(count, _IQ1_M_SUBBLOCKS, 2)
        scales = d.unsqueeze(-1).unsqueeze(-1) * (2 * local + 1).float()

        values = grid[entries] + deltas.unsqueeze(-1)
        # Groups 0,1 take the first local scale and groups 2,3 the second, so the repeat
        # runs along the half axis: [h0, h0, h1, h1], not [h0, h1, h0, h1].
        group_scale = scales.repeat_interleave(2, dim=2)
        chunk_decoded = values * group_scale.unsqueeze(-1)
        decoded[start:stop] = chunk_decoded.reshape(-1, IQ1_M_BLOCK_SIZE)
    return decoded.reshape(shape)


IQ1_M_FORMAT = GGMLFormat(
    name="iq1_m",
    block_size=IQ1_M_BLOCK_SIZE,
    block_bytes=IQ1_M_BLOCK_BYTES,
    quantize=quantize_iq1_m,
    dequantize=dequantize_iq1_m,
    block_chunk_size=_DEFAULT_BLOCK_CHUNK_SIZE,
    decode_chunk_size=_DEFAULT_DECODE_CHUNK_SIZE,
    weighted=True,
)

# Kept for callers of the per-format entry point. The record captured quantize_iq1_m and
# dequantize_iq1_m when it was built, so patching those module functions changes neither backend
# dispatch nor this alias; substitute a format's encoder or decoder in IQ_FORMAT_REGISTRY.
iq1_m_fake_quant = IQ1_M_FORMAT.fake_quant
