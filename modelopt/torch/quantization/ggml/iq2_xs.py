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

"""IQ2_XS fake quantization and GGML-compatible block packing.

The encoder performs a single-pass squared-error grid search at a fixed,
empirically anchored super-block scale. It does not iteratively refine the
scale or apply importance weights. Every 256 logical values become one 74-byte
block_iq2_xs payload:

* bytes 0..1: little-endian FP16 super-block scale d
* bytes 2..65: 32 little-endian uint16 codes (9-bit grid + 7-bit sign)
* bytes 66..73: 16 four-bit local scales, two per byte

The canonical 512 x 8 magnitude grid lives in :mod:`.codebooks`, carried from
llama.cpp ggml-common.h revision 9b05354ec6fb58b4e665e9a39ebc40285c015638.
The matching dequantization formula is in ggml-quants.c at the same revision:
https://github.com/ggml-org/llama.cpp/blob/9b05354ec6fb58b4e665e9a39ebc40285c015638/ggml/src/ggml-quants.c#L2516-L2538
"""

import torch

from ..extensions import get_cuda_ext_ggml
from .codebooks import iq2_xs_grid_bytes
from .common import (
    GGML_BLOCK_SIZE,
    IQFormat,
    narrow_to_float32,
    validate_block_chunk_size,
    validate_packed_weights,
    validate_weight,
)

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
_IQ2_XS_NATIVE_MAX = 43 * 31 / 8
_IQ2_XS_SCALE_ANCHOR_MIN = 0.65
_IQ2_XS_SCALE_ANCHOR_MAX = 0.92
_IQ2_XS_PEAK_TO_RMS_TAPER = 0.035
# Bounds the torch encode fallback, whose codebook search holds the large temporaries: at
# 256 blocks the largest is about 16 MiB in FP32. The CUDA encoder ignores this entirely.
# This is four times smaller than the IQ1_S bound because the IQ2_XS search sweeps sixteen
# local scales per grid tile.
_DEFAULT_BLOCK_CHUNK_SIZE = 256
# The decode's temporaries are far smaller, so it is launch-bound rather than memory-bound
# and wants a bigger chunk -- and unlike packing it is not cached, so it runs on every
# forward. Sharing the encode bound above is what made IQ2_XS four times slower end to end
# than IQ1_S. Measured decoding a 2048x5632 weight: 91.4 ms at 256 blocks, 22.9 ms at 1024,
# 5.8 ms at 4096, where the transient peak is +56 MiB.
_DEFAULT_DECODE_CHUNK_SIZE = 4096
_SCALE_BLOCK_CHUNK_SIZE = 4096


_GRID_CACHE: dict[torch.device, torch.Tensor] = {}


def iq2_xs_grid(device: torch.device | str | None = None) -> torch.Tensor:
    """Return the canonical IQ2_XS magnitude grid as float32."""
    resolved_device = torch.device(device or "cpu")
    if resolved_device.type == "cuda" and resolved_device.index is None:
        resolved_device = torch.device("cuda", torch.cuda.current_device())
    if resolved_device not in _GRID_CACHE:
        values = torch.tensor(list(iq2_xs_grid_bytes()), dtype=torch.float32)
        _GRID_CACHE[resolved_device] = values.reshape(512, 8).to(device=resolved_device)
    return _GRID_CACHE[resolved_device]


def _predict_iq2_xs_scales(blocks: torch.Tensor) -> torch.Tensor:
    """Predict one FP16 super-block scale for each flattened block."""
    x = narrow_to_float32(blocks)
    amax = x.abs().amax(dim=1)
    rms = x.square().mean(dim=1).sqrt()
    peak_to_rms = torch.where(rms > 0, amax / rms, torch.zeros_like(rms))
    # The fixed-scale search favors a compressed super-block scale. This
    # empirical predictor tapers the anchor for outlier-heavy blocks.
    anchor_ratio = (1.0 - _IQ2_XS_PEAK_TO_RMS_TAPER * peak_to_rms).clamp(
        _IQ2_XS_SCALE_ANCHOR_MIN, _IQ2_XS_SCALE_ANCHOR_MAX
    )
    return ((amax / _IQ2_XS_NATIVE_MAX) * anchor_ratio).clamp(max=65504.0).to(torch.float16)


def _encode_blocks(
    blocks: torch.Tensor, grid: torch.Tensor, scales: torch.Tensor | None = None
) -> torch.Tensor:
    """Encode a moderate-size batch of flattened 256-value blocks."""
    x = narrow_to_float32(blocks)
    block_count = x.shape[0]
    vectors = x.reshape(block_count, 32, 8)
    magnitudes = vectors.abs()
    negative = vectors < 0
    odd_parity = negative.sum(dim=-1).remainder(2).bool()

    d = _predict_iq2_xs_scales(x) if scales is None else scales
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
    return torch.where((d_float == 0).unsqueeze(1), 0, packed)


@torch.no_grad()
def quantize_iq2_xs(
    weight: torch.Tensor, *, block_chunk_size: int = _DEFAULT_BLOCK_CHUNK_SIZE
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack a floating-point weight into GGML-compatible IQ2_XS blocks.

    Returned shapes are ``[*weight.shape[:-1], weight.shape[-1] // 256, 74]``
    and ``[weight.ndim]``. The packed payload remains on the weight's device;
    the logical-shape metadata is kept on CPU. Non-finite input elements are
    treated as zero during packing.
    """
    validate_weight(weight, "IQ2_XS")
    validate_block_chunk_size(block_chunk_size)

    logical_shape = torch.tensor(weight.shape, dtype=torch.int64)
    blocks = weight.contiguous().reshape(-1, IQ2_XS_BLOCK_SIZE)
    grid = iq2_xs_grid(weight.device)
    if weight.is_cuda:
        extension = get_cuda_ext_ggml()
        if extension is not None:
            scale_chunks = [
                _predict_iq2_xs_scales(blocks[start : start + _SCALE_BLOCK_CHUNK_SIZE])
                for start in range(0, blocks.shape[0], _SCALE_BLOCK_CHUNK_SIZE)
            ]
            packed = extension.iq2_xs_pack(blocks, grid, torch.cat(scale_chunks))
            packed_shape = (
                *weight.shape[:-1],
                weight.shape[-1] // IQ2_XS_BLOCK_SIZE,
                IQ2_XS_BLOCK_BYTES,
            )
            return packed.reshape(packed_shape), logical_shape

    chunks = []
    for start in range(0, blocks.shape[0], block_chunk_size):
        block_chunk = blocks[start : start + block_chunk_size]
        scales = _predict_iq2_xs_scales(block_chunk)
        chunks.append(_encode_blocks(block_chunk, grid, scales))
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
    block_chunk_size: int = _DEFAULT_DECODE_CHUNK_SIZE,
) -> torch.Tensor:
    """Decode GGML-compatible IQ2_XS payload bytes."""
    shape = validate_packed_weights(
        packed_weights, weight_shape, block_bytes=IQ2_XS_BLOCK_BYTES, format_name="IQ2_XS"
    )
    validate_block_chunk_size(block_chunk_size)

    blocks = packed_weights.contiguous().reshape(-1, IQ2_XS_BLOCK_BYTES)
    bit_positions = torch.arange(8, dtype=torch.int64, device=blocks.device)
    grid = iq2_xs_grid(blocks.device)
    decoded = torch.empty((blocks.shape[0], IQ2_XS_BLOCK_SIZE), dtype=dtype, device=blocks.device)
    for start in range(0, blocks.shape[0], block_chunk_size):
        stop = min(start + block_chunk_size, blocks.shape[0])
        block_chunk = blocks[start:stop]
        d = block_chunk[:, :2].contiguous().view(torch.float16).reshape(-1).float()
        codes = block_chunk[:, 2:66:2].to(torch.int64) | (
            block_chunk[:, 3:66:2].to(torch.int64) << 8
        )
        entries = codes & 0x1FF
        sign_index = codes >> 9
        # XOR-fold the seven payload bits down to bit 0 to recover the eighth sign bit.
        # The loop this replaces cost seven elementwise passes per chunk, and unlike packing
        # the decode is not cached -- it runs again on every forward.
        folded = sign_index ^ (sign_index >> 4)
        folded ^= folded >> 2
        folded ^= folded >> 1
        sign_mask = sign_index | ((folded & 1) << 7)
        signs = 1.0 - 2.0 * ((sign_mask.unsqueeze(-1) >> bit_positions) & 1).float()

        scale_bytes = block_chunk[:, 66:].to(torch.int64)
        local = torch.empty((block_chunk.shape[0], 16), dtype=torch.int64, device=blocks.device)
        local[:, 0::2] = scale_bytes & 0x0F
        local[:, 1::2] = scale_bytes >> 4
        # Pinned format rule: d * (0.5 + local) * 0.25 == d * (2 * local + 1) / 8.
        scales = d.unsqueeze(-1) * (2 * local + 1).float() / 8.0
        values = grid[entries] * signs
        chunk_decoded = values * scales.repeat_interleave(2, dim=1).unsqueeze(-1)
        decoded[start:stop] = chunk_decoded.reshape(-1, IQ2_XS_BLOCK_SIZE)
    return decoded.reshape(shape)


IQ2_XS_FORMAT = IQFormat(
    name="iq2_xs",
    block_size=IQ2_XS_BLOCK_SIZE,
    block_bytes=IQ2_XS_BLOCK_BYTES,
    quantize=quantize_iq2_xs,
    dequantize=dequantize_iq2_xs,
    block_chunk_size=_DEFAULT_BLOCK_CHUNK_SIZE,
    decode_chunk_size=_DEFAULT_DECODE_CHUNK_SIZE,
)

# Kept for callers of the per-format entry point. The record captured quantize_iq2_xs and
# dequantize_iq2_xs when it was built, so patching those module functions changes neither backend
# dispatch nor this alias; substitute a format's encoder or decoder in IQ_FORMAT_REGISTRY.
iq2_xs_fake_quant = IQ2_XS_FORMAT.fake_quant
