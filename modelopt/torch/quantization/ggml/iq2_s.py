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

"""IQ2_S fake quantization and GGML-compatible block packing.

The encoder performs a single-pass squared-error grid search at a fixed,
empirically anchored super-block scale, mirroring :mod:`.iq2_xs`. Every 256
logical values become one 82-byte block_iq2_s payload:

* bytes 0..1: little-endian FP16 super-block scale d
* bytes 2..33: 32 low bytes of the grid index, four per sub-block
* bytes 34..65: 32 sign masks, four per sub-block
* bytes 66..73: eight bytes holding the grid index high 2 bits, four per byte
* bytes 74..81: 16 four-bit local scales, two per byte

IQ2_S stores a full eight-bit sign mask per group rather than the seven-bit
parity-coded index used by IQ2_XS and IQ2_XXS, so the encoder can take the
input signs directly instead of flipping the weakest element to fix parity.

The canonical 1024 x 8 magnitude grid lives in :mod:`.codebooks`, carried from
llama.cpp ggml-common.h revision 9b05354ec6fb58b4e665e9a39ebc40285c015638.
The matching dequantization formula is in ggml-quants.c at the same revision:
https://github.com/ggml-org/llama.cpp/blob/9b05354ec6fb58b4e665e9a39ebc40285c015638/ggml/src/ggml-quants.c#L2540-L2571
"""

import torch

from ..extensions import get_cuda_ext_ggml
from .codebooks import iq2_s_grid_bytes
from .common import (
    GGML_BLOCK_SIZE,
    fake_quantize_with_cache,
    narrow_to_float32,
    validate_block_chunk_size,
    validate_packed_weights,
    validate_weight,
)

__all__ = [
    "IQ2_S_BLOCK_BYTES",
    "IQ2_S_BLOCK_SIZE",
    "IQ2_S_EFFECTIVE_BITS",
    "dequantize_iq2_s",
    "iq2_s_fake_quant",
    "iq2_s_grid",
    "quantize_iq2_s",
]

IQ2_S_BLOCK_SIZE = GGML_BLOCK_SIZE
IQ2_S_BLOCK_BYTES = 82
IQ2_S_EFFECTIVE_BITS = IQ2_S_BLOCK_BYTES * 8 / IQ2_S_BLOCK_SIZE
_IQ2_S_GRID_ENTRIES = 1024
_IQ2_S_LOCAL_SCALES = 16
_IQ2_S_GROUPS = 32
_IQ2_S_SUBBLOCKS = 8
_IQ2_S_NATIVE_MAX = 43 * 31 / 8
_IQ2_S_SCALE_ANCHOR_MIN = 0.65
_IQ2_S_SCALE_ANCHOR_MAX = 0.92
_IQ2_S_PEAK_TO_RMS_TAPER = 0.035
# The grid is twice IQ2_XS's, so the same search tile costs twice the memory.
_DEFAULT_BLOCK_CHUNK_SIZE = 128
_DEFAULT_DECODE_CHUNK_SIZE = 4096
_SCALE_BLOCK_CHUNK_SIZE = 4096

_GRID_CACHE: dict[torch.device, torch.Tensor] = {}


def iq2_s_grid(device: torch.device | str | None = None) -> torch.Tensor:
    """Return the canonical IQ2_S magnitude grid as float32."""
    resolved_device = torch.device(device or "cpu")
    if resolved_device.type == "cuda" and resolved_device.index is None:
        resolved_device = torch.device("cuda", torch.cuda.current_device())
    if resolved_device not in _GRID_CACHE:
        values = torch.tensor(list(iq2_s_grid_bytes()), dtype=torch.float32)
        _GRID_CACHE[resolved_device] = values.reshape(_IQ2_S_GRID_ENTRIES, 8).to(
            device=resolved_device
        )
    return _GRID_CACHE[resolved_device]


def _predict_iq2_s_scales(blocks: torch.Tensor) -> torch.Tensor:
    """Predict one FP16 super-block scale for each flattened block."""
    x = narrow_to_float32(blocks)
    amax = x.abs().amax(dim=1)
    rms = x.square().mean(dim=1).sqrt()
    peak_to_rms = torch.where(rms > 0, amax / rms, torch.zeros_like(rms))
    anchor_ratio = (1.0 - _IQ2_S_PEAK_TO_RMS_TAPER * peak_to_rms).clamp(
        _IQ2_S_SCALE_ANCHOR_MIN, _IQ2_S_SCALE_ANCHOR_MAX
    )
    return ((amax / _IQ2_S_NATIVE_MAX) * anchor_ratio).clamp(max=65504.0).to(torch.float16)


def _encode_blocks(blocks: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    """Encode a moderate-size batch of flattened 256-value blocks."""
    x = narrow_to_float32(blocks)
    block_count = x.shape[0]
    vectors = x.reshape(block_count, _IQ2_S_GROUPS, 8)
    magnitudes = vectors.abs()
    negative = vectors < 0

    d = _predict_iq2_s_scales(x)
    d_float = d.float()

    xnorm = vectors.square().sum(dim=-1)
    qnorm = grid.square().sum(dim=-1)
    shape = (block_count, _IQ2_S_GROUPS, _IQ2_S_LOCAL_SCALES)
    best_error = torch.full(shape, torch.inf, dtype=torch.float32, device=x.device)
    best_entry = torch.zeros(shape, dtype=torch.int64, device=x.device)
    # All eight signs are storable, so the search compares magnitudes directly.
    for entry_start in range(0, _IQ2_S_GRID_ENTRIES, 64):
        grid_tile = grid[entry_start : entry_start + 64]
        dot = (magnitudes.unsqueeze(2) * grid_tile.reshape(1, 1, -1, 8)).sum(dim=-1)
        tile_qnorm = qnorm[entry_start : entry_start + 64].reshape(1, 1, -1)

        for local in range(_IQ2_S_LOCAL_SCALES):
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

    # One local scale covers two groups (16 values), as in IQ2_XS.
    pair_error = best_error.reshape(block_count, 16, 2, _IQ2_S_LOCAL_SCALES).sum(dim=2)
    selected_local = pair_error.argmin(dim=-1)
    group_local = selected_local.repeat_interleave(2, dim=1)
    selected_entry = best_entry.gather(2, group_local.unsqueeze(-1)).squeeze(-1)

    sign_bits = torch.arange(8, dtype=torch.int64, device=x.device)
    sign_mask = (negative.to(torch.int64) << sign_bits).sum(dim=-1)

    packed = torch.empty((block_count, IQ2_S_BLOCK_BYTES), dtype=torch.uint8, device=x.device)
    packed[:, :2] = d.contiguous().view(torch.uint8).reshape(block_count, 2)
    packed[:, 2:34] = (selected_entry & 0xFF).to(torch.uint8)
    packed[:, 34:66] = sign_mask.to(torch.uint8)
    high = (selected_entry >> 8).reshape(block_count, _IQ2_S_SUBBLOCKS, 4)
    packed[:, 66:74] = (
        high[:, :, 0] | (high[:, :, 1] << 2) | (high[:, :, 2] << 4) | (high[:, :, 3] << 6)
    ).to(torch.uint8)
    packed[:, 74:] = (selected_local[:, 0::2] | (selected_local[:, 1::2] << 4)).to(torch.uint8)
    return torch.where((d_float == 0).unsqueeze(1), 0, packed)


@torch.no_grad()
def quantize_iq2_s(
    weight: torch.Tensor, *, block_chunk_size: int = _DEFAULT_BLOCK_CHUNK_SIZE
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack a floating-point weight into GGML-compatible IQ2_S blocks.

    Returned shapes are ``[*weight.shape[:-1], weight.shape[-1] // 256, 82]``
    and ``[weight.ndim]``.
    """
    validate_weight(weight, "IQ2_S")
    validate_block_chunk_size(block_chunk_size)

    logical_shape = torch.tensor(weight.shape, dtype=torch.int64)
    blocks = weight.contiguous().reshape(-1, IQ2_S_BLOCK_SIZE)
    grid = iq2_s_grid(weight.device)
    packed_shape = (*weight.shape[:-1], weight.shape[-1] // IQ2_S_BLOCK_SIZE, IQ2_S_BLOCK_BYTES)
    if weight.is_cuda:
        extension = get_cuda_ext_ggml()
        if extension is not None:
            scale_chunks = [
                _predict_iq2_s_scales(blocks[start : start + _SCALE_BLOCK_CHUNK_SIZE])
                for start in range(0, blocks.shape[0], _SCALE_BLOCK_CHUNK_SIZE)
            ]
            packed = extension.iq2_s_pack(blocks, grid, torch.cat(scale_chunks))
            return packed.reshape(packed_shape), logical_shape

    chunks = [
        _encode_blocks(blocks[start : start + block_chunk_size], grid)
        for start in range(0, blocks.shape[0], block_chunk_size)
    ]
    return torch.cat(chunks).reshape(packed_shape), logical_shape


@torch.no_grad()
def dequantize_iq2_s(
    packed_weights: torch.Tensor,
    weight_shape: torch.Tensor,
    *,
    dtype: torch.dtype = torch.bfloat16,
    block_chunk_size: int = _DEFAULT_DECODE_CHUNK_SIZE,
) -> torch.Tensor:
    """Decode GGML-compatible IQ2_S payload bytes."""
    shape = validate_packed_weights(
        packed_weights, weight_shape, block_bytes=IQ2_S_BLOCK_BYTES, format_name="IQ2_S"
    )
    validate_block_chunk_size(block_chunk_size)

    blocks = packed_weights.contiguous().reshape(-1, IQ2_S_BLOCK_BYTES)
    bit_positions = torch.arange(8, dtype=torch.int64, device=blocks.device)
    high_shifts = torch.tensor([0, 2, 4, 6], dtype=torch.int64, device=blocks.device)
    grid = iq2_s_grid(blocks.device)
    decoded = torch.empty((blocks.shape[0], IQ2_S_BLOCK_SIZE), dtype=dtype, device=blocks.device)
    for start in range(0, blocks.shape[0], block_chunk_size):
        stop = min(start + block_chunk_size, blocks.shape[0])
        block_chunk = blocks[start:stop]
        count = block_chunk.shape[0]
        d = block_chunk[:, :2].contiguous().view(torch.float16).reshape(-1).float()
        low = block_chunk[:, 2:34].to(torch.int64).reshape(count, _IQ2_S_SUBBLOCKS, 4)
        sign_mask = block_chunk[:, 34:66].to(torch.int64).reshape(count, _IQ2_S_SUBBLOCKS, 4)
        qh = block_chunk[:, 66:74].to(torch.int64).reshape(count, _IQ2_S_SUBBLOCKS)
        scale_bytes = block_chunk[:, 74:].to(torch.int64)

        entries = low | (((qh.unsqueeze(-1) >> high_shifts) & 0x3) << 8)
        local = torch.empty((count, 16), dtype=torch.int64, device=blocks.device)
        local[:, 0::2] = scale_bytes & 0x0F
        local[:, 1::2] = scale_bytes >> 4
        scales = d.unsqueeze(-1) * (0.5 + local.float()) * 0.25
        signs = 1.0 - 2.0 * ((sign_mask.unsqueeze(-1) >> bit_positions) & 1).float()
        values = grid[entries] * signs
        chunk_decoded = values.reshape(count, 16, 2, 8) * scales.unsqueeze(-1).unsqueeze(-1)
        decoded[start:stop] = chunk_decoded.reshape(-1, IQ2_S_BLOCK_SIZE)
    return decoded.reshape(shape)


def iq2_s_fake_quant(
    inputs: torch.Tensor,
    quantizer,
    *,
    block_chunk_size: int = _DEFAULT_BLOCK_CHUNK_SIZE,
    decode_chunk_size: int = _DEFAULT_DECODE_CHUNK_SIZE,
) -> torch.Tensor:
    """IQ2_S weight backend for TensorQuantizer, with pass-through backward."""
    if getattr(quantizer, "num_bits", None) != "iq2_s":
        raise ValueError("The ggml IQ2_S backend requires num_bits='iq2_s'")
    return fake_quantize_with_cache(
        inputs,
        quantizer,
        format_name="iq2_s",
        block_chunk_size=block_chunk_size,
        decode_chunk_size=decode_chunk_size,
        quantize=quantize_iq2_s,
        dequantize=dequantize_iq2_s,
    )
