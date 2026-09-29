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

"""IQ2_XXS fake quantization and GGML-compatible block packing.

The encoder performs a single-pass squared-error grid search at a fixed,
empirically anchored super-block scale, mirroring :mod:`.iq2_xs`. Every 256
logical values become one 66-byte block_iq2_xxs payload:

* bytes 0..1: little-endian FP16 super-block scale d
* bytes 2..65: eight 8-byte sub-block records, each holding four 8-bit grid
  indices followed by a little-endian uint32 of four 7-bit sign indices
  (bits 0..27) and one 4-bit local scale (bits 28..31)

IQ2_XXS differs from IQ2_XS in three ways: the grid is 256 entries rather than
512 so an index needs no high bits, one local scale covers a whole 32-value
sub-block rather than 16 values, and the scale shares a word with the signs
instead of living in a trailing array.

The canonical 256 x 8 magnitude grid lives in :mod:`.codebooks`, carried from
llama.cpp ggml-common.h revision 9b05354ec6fb58b4e665e9a39ebc40285c015638.
The matching dequantization formula is in ggml-quants.c at the same revision:
https://github.com/ggml-org/llama.cpp/blob/9b05354ec6fb58b4e665e9a39ebc40285c015638/ggml/src/ggml-quants.c#L2489-L2514
"""

import torch

from ..extensions import get_cuda_ext_ggml
from .codebooks import iq2_xxs_grid_bytes
from .common import (
    GGML_BLOCK_SIZE,
    IQFormat,
    narrow_to_float32,
    validate_block_chunk_size,
    validate_packed_weights,
    validate_weight,
)

__all__ = [
    "IQ2_XXS_BLOCK_BYTES",
    "IQ2_XXS_BLOCK_SIZE",
    "IQ2_XXS_EFFECTIVE_BITS",
    "dequantize_iq2_xxs",
    "iq2_xxs_fake_quant",
    "iq2_xxs_grid",
    "quantize_iq2_xxs",
]

IQ2_XXS_BLOCK_SIZE = GGML_BLOCK_SIZE
IQ2_XXS_BLOCK_BYTES = 66
IQ2_XXS_EFFECTIVE_BITS = IQ2_XXS_BLOCK_BYTES * 8 / IQ2_XXS_BLOCK_SIZE
_IQ2_XXS_GRID_ENTRIES = 256
_IQ2_XXS_LOCAL_SCALES = 16
_IQ2_XXS_GROUPS = 32
_IQ2_XXS_GROUPS_PER_SUBBLOCK = 4
_IQ2_XXS_SUBBLOCKS = _IQ2_XXS_GROUPS // _IQ2_XXS_GROUPS_PER_SUBBLOCK
# Largest representable magnitude: grid entry 43 at local scale 15 -> (0.5 + 15) * 0.25.
_IQ2_XXS_NATIVE_MAX = 43 * 31 / 8
_IQ2_XXS_SCALE_ANCHOR_MIN = 0.65
_IQ2_XXS_SCALE_ANCHOR_MAX = 0.92
_IQ2_XXS_PEAK_TO_RMS_TAPER = 0.035
# Bounds the encode search temporaries; see the note in .iq2_xs. The grid is half the size
# of IQ2_XS's, so the same chunk holds half the search tile.
_DEFAULT_BLOCK_CHUNK_SIZE = 512
# The decode runs on every forward and is launch-bound, so it takes a much larger chunk.
_DEFAULT_DECODE_CHUNK_SIZE = 4096
_SCALE_BLOCK_CHUNK_SIZE = 4096

_GRID_CACHE: dict[torch.device, torch.Tensor] = {}


def iq2_xxs_grid(device: torch.device | str | None = None) -> torch.Tensor:
    """Return the canonical IQ2_XXS magnitude grid as float32."""
    resolved_device = torch.device(device or "cpu")
    if resolved_device.type == "cuda" and resolved_device.index is None:
        resolved_device = torch.device("cuda", torch.cuda.current_device())
    if resolved_device not in _GRID_CACHE:
        values = torch.tensor(list(iq2_xxs_grid_bytes()), dtype=torch.float32)
        _GRID_CACHE[resolved_device] = values.reshape(_IQ2_XXS_GRID_ENTRIES, 8).to(
            device=resolved_device
        )
    return _GRID_CACHE[resolved_device]


def _predict_iq2_xxs_scales(blocks: torch.Tensor) -> torch.Tensor:
    """Predict one FP16 super-block scale for each flattened block."""
    x = narrow_to_float32(blocks)
    amax = x.abs().amax(dim=1)
    rms = x.square().mean(dim=1).sqrt()
    peak_to_rms = torch.where(rms > 0, amax / rms, torch.zeros_like(rms))
    anchor_ratio = (1.0 - _IQ2_XXS_PEAK_TO_RMS_TAPER * peak_to_rms).clamp(
        _IQ2_XXS_SCALE_ANCHOR_MIN, _IQ2_XXS_SCALE_ANCHOR_MAX
    )
    return ((amax / _IQ2_XXS_NATIVE_MAX) * anchor_ratio).clamp(max=65504.0).to(torch.float16)


def _encode_blocks(blocks: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    """Encode a moderate-size batch of flattened 256-value blocks."""
    x = narrow_to_float32(blocks)
    block_count = x.shape[0]
    vectors = x.reshape(block_count, _IQ2_XXS_GROUPS, 8)
    magnitudes = vectors.abs()
    negative = vectors < 0
    # Only seven sign bits are stored; the eighth is their parity, so an odd sign pattern
    # has to flip one element. Account for that cost while searching, not after.
    odd_parity = negative.sum(dim=-1).remainder(2).bool()

    d = _predict_iq2_xxs_scales(x)
    d_float = d.float()

    xnorm = vectors.square().sum(dim=-1)
    qnorm = grid.square().sum(dim=-1)
    shape = (block_count, _IQ2_XXS_GROUPS, _IQ2_XXS_LOCAL_SCALES)
    best_error = torch.full(shape, torch.inf, dtype=torch.float32, device=x.device)
    best_entry = torch.zeros(shape, dtype=torch.int64, device=x.device)
    # Search the codebook in tiles to cap temporary memory. Strict comparison
    # preserves the lowest grid index on equal error.
    for entry_start in range(0, _IQ2_XXS_GRID_ENTRIES, 64):
        grid_tile = grid[entry_start : entry_start + 64]
        products = magnitudes.unsqueeze(2) * grid_tile.reshape(1, 1, -1, 8)
        dot = products.sum(dim=-1)
        dot = torch.where(odd_parity.unsqueeze(-1), dot - 2.0 * products.amin(dim=-1), dot)
        tile_qnorm = qnorm[entry_start : entry_start + 64].reshape(1, 1, -1)

        for local in range(_IQ2_XXS_LOCAL_SCALES):
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

    # One local scale covers four groups here, against two for IQ2_XS.
    subblock_error = best_error.reshape(
        block_count, _IQ2_XXS_SUBBLOCKS, _IQ2_XXS_GROUPS_PER_SUBBLOCK, _IQ2_XXS_LOCAL_SCALES
    ).sum(dim=2)
    selected_local = subblock_error.argmin(dim=-1)
    group_local = selected_local.repeat_interleave(_IQ2_XXS_GROUPS_PER_SUBBLOCK, dim=1)
    selected_entry = best_entry.gather(2, group_local.unsqueeze(-1)).squeeze(-1)

    selected_grid = grid[selected_entry]
    weakest_index = (magnitudes * selected_grid).argmin(dim=-1)
    flip = torch.nn.functional.one_hot(weakest_index, num_classes=8).bool()
    encoded_negative = negative ^ (flip & odd_parity.unsqueeze(-1))
    sign_bits = torch.arange(8, dtype=torch.int64, device=x.device)
    sign_mask = (encoded_negative.to(torch.int64) << sign_bits).sum(dim=-1) & 0x7F

    signs = sign_mask.reshape(block_count, _IQ2_XXS_SUBBLOCKS, _IQ2_XXS_GROUPS_PER_SUBBLOCK)
    aux = (
        signs[:, :, 0]
        | (signs[:, :, 1] << 7)
        | (signs[:, :, 2] << 14)
        | (signs[:, :, 3] << 21)
        | (selected_local << 28)
    )

    body = torch.empty((block_count, _IQ2_XXS_SUBBLOCKS, 8), dtype=torch.uint8, device=x.device)
    body[:, :, 0:4] = (
        selected_entry.reshape(block_count, _IQ2_XXS_SUBBLOCKS, _IQ2_XXS_GROUPS_PER_SUBBLOCK)
    ).to(torch.uint8)
    for byte in range(4):
        body[:, :, 4 + byte] = ((aux >> (8 * byte)) & 0xFF).to(torch.uint8)

    packed = torch.empty((block_count, IQ2_XXS_BLOCK_BYTES), dtype=torch.uint8, device=x.device)
    packed[:, :2] = d.contiguous().view(torch.uint8).reshape(block_count, 2)
    packed[:, 2:] = body.reshape(block_count, -1)
    return torch.where((d_float == 0).unsqueeze(1), 0, packed)


@torch.no_grad()
def quantize_iq2_xxs(
    weight: torch.Tensor, *, block_chunk_size: int = _DEFAULT_BLOCK_CHUNK_SIZE
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack a floating-point weight into GGML-compatible IQ2_XXS blocks.

    Returned shapes are ``[*weight.shape[:-1], weight.shape[-1] // 256, 66]``
    and ``[weight.ndim]``. The packed payload remains on the weight's device;
    the logical-shape metadata is kept on CPU. Non-finite input elements are
    treated as zero during packing.
    """
    validate_weight(weight, "IQ2_XXS")
    validate_block_chunk_size(block_chunk_size)

    logical_shape = torch.tensor(weight.shape, dtype=torch.int64)
    blocks = weight.contiguous().reshape(-1, IQ2_XXS_BLOCK_SIZE)
    grid = iq2_xxs_grid(weight.device)
    packed_shape = (
        *weight.shape[:-1],
        weight.shape[-1] // IQ2_XXS_BLOCK_SIZE,
        IQ2_XXS_BLOCK_BYTES,
    )
    if weight.is_cuda:
        extension = get_cuda_ext_ggml()
        if extension is not None:
            scale_chunks = [
                _predict_iq2_xxs_scales(blocks[start : start + _SCALE_BLOCK_CHUNK_SIZE])
                for start in range(0, blocks.shape[0], _SCALE_BLOCK_CHUNK_SIZE)
            ]
            packed = extension.iq2_xxs_pack(blocks, grid, torch.cat(scale_chunks))
            return packed.reshape(packed_shape), logical_shape

    chunks = [
        _encode_blocks(blocks[start : start + block_chunk_size], grid)
        for start in range(0, blocks.shape[0], block_chunk_size)
    ]
    return torch.cat(chunks).reshape(packed_shape), logical_shape


@torch.no_grad()
def dequantize_iq2_xxs(
    packed_weights: torch.Tensor,
    weight_shape: torch.Tensor,
    *,
    dtype: torch.dtype = torch.bfloat16,
    block_chunk_size: int = _DEFAULT_DECODE_CHUNK_SIZE,
) -> torch.Tensor:
    """Decode GGML-compatible IQ2_XXS payload bytes."""
    shape = validate_packed_weights(
        packed_weights, weight_shape, block_bytes=IQ2_XXS_BLOCK_BYTES, format_name="IQ2_XXS"
    )
    validate_block_chunk_size(block_chunk_size)

    blocks = packed_weights.contiguous().reshape(-1, IQ2_XXS_BLOCK_BYTES)
    bit_positions = torch.arange(8, dtype=torch.int64, device=blocks.device)
    sign_shifts = 7 * torch.arange(
        _IQ2_XXS_GROUPS_PER_SUBBLOCK, dtype=torch.int64, device=blocks.device
    )
    grid = iq2_xxs_grid(blocks.device)
    decoded = torch.empty((blocks.shape[0], IQ2_XXS_BLOCK_SIZE), dtype=dtype, device=blocks.device)
    for start in range(0, blocks.shape[0], block_chunk_size):
        stop = min(start + block_chunk_size, blocks.shape[0])
        block_chunk = blocks[start:stop]
        count = block_chunk.shape[0]
        d = block_chunk[:, :2].contiguous().view(torch.float16).reshape(-1).float()
        body = block_chunk[:, 2:].reshape(count, _IQ2_XXS_SUBBLOCKS, 8).to(torch.int64)
        entries = body[:, :, 0:4]
        aux = body[:, :, 4] | (body[:, :, 5] << 8) | (body[:, :, 6] << 16) | (body[:, :, 7] << 24)
        # Top nibble is the sub-block scale; the low 28 bits are four 7-bit sign indices.
        scales = d.unsqueeze(-1) * (0.5 + ((aux >> 28) & 0xF).float()) * 0.25
        sign_index = (aux.unsqueeze(-1) >> sign_shifts) & 0x7F
        folded = sign_index ^ (sign_index >> 4)
        folded ^= folded >> 2
        folded ^= folded >> 1
        sign_mask = sign_index | ((folded & 1) << 7)
        signs = 1.0 - 2.0 * ((sign_mask.unsqueeze(-1) >> bit_positions) & 1).float()
        values = grid[entries] * signs
        chunk_decoded = values * scales.unsqueeze(-1).unsqueeze(-1)
        decoded[start:stop] = chunk_decoded.reshape(-1, IQ2_XXS_BLOCK_SIZE)
    return decoded.reshape(shape)


IQ2_XXS_FORMAT = IQFormat(
    name="iq2_xxs",
    block_size=IQ2_XXS_BLOCK_SIZE,
    block_bytes=IQ2_XXS_BLOCK_BYTES,
    quantize=quantize_iq2_xxs,
    dequantize=dequantize_iq2_xxs,
    block_chunk_size=_DEFAULT_BLOCK_CHUNK_SIZE,
    decode_chunk_size=_DEFAULT_DECODE_CHUNK_SIZE,
)

# Kept for callers of the per-format entry point. The record captured quantize_iq2_xxs and
# dequantize_iq2_xxs when it was built, so patching those module functions changes neither backend
# dispatch nor this alias; substitute a format's encoder or decoder in IQ_FORMAT_REGISTRY.
iq2_xxs_fake_quant = IQ2_XXS_FORMAT.fake_quant
