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

"""IQ1_S fake quantization and GGML-compatible block packing.

The encoder performs a single-pass squared-error grid search at a fixed,
empirically anchored super-block scale. It does not iteratively refine the
scale or apply importance weights. Every 256 logical values become one 50-byte
``block_iq1_s`` payload:

* bytes 0..1: little-endian FP16 super-block scale ``d``
* bytes 2..33: low eight bits of 32 codebook indices
* bytes 34..49: eight little-endian uint16 metadata words

Each metadata word describes four consecutive eight-value vectors. Bits 0..11
hold the three high index bits, bits 12..14 select one of eight local scales,
and bit 15 selects the shared -0.125 rather than +0.125 delta. The canonical
2048 x 8 ternary grid lives in :mod:`.codebooks`, carried from llama.cpp
``ggml-common.h`` revision
9b05354ec6fb58b4e665e9a39ebc40285c015638.
"""

import torch

from ..extensions import get_cuda_ext_ggml
from .codebooks import iq1_s_grid_bytes
from .common import (
    GGML_BLOCK_SIZE,
    fake_quantize_with_cache,
    narrow_to_float32,
    validate_block_chunk_size,
    validate_packed_weights,
    validate_weight,
)

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
_IQ1_S_SCALE_ANCHOR = 0.61
# Bounds the torch encode fallback, whose codebook search holds the large temporaries: at
# 1024 blocks each is about 16 MiB in FP32. The CUDA encoder ignores this entirely.
_DEFAULT_BLOCK_CHUNK_SIZE = 1024
# The decode's temporaries are far smaller, so it is launch-bound rather than memory-bound
# and wants a bigger chunk -- and unlike packing it is not cached, so it runs on every
# forward. Measured decoding a 2048x5632 weight: 66.5 ms at 256 blocks, 4.2 ms at 4096,
# where the transient peak is +42 MiB.
_DEFAULT_DECODE_CHUNK_SIZE = 4096


_GRID_CACHE: dict[torch.device, torch.Tensor] = {}


def iq1_s_grid(device: torch.device | str | None = None) -> torch.Tensor:
    """Return the canonical IQ1_S ternary grid as float32."""
    resolved_device = torch.device(device or "cpu")
    if resolved_device.type == "cuda" and resolved_device.index is None:
        resolved_device = torch.device("cuda", torch.cuda.current_device())
    if resolved_device not in _GRID_CACHE:
        raw = torch.tensor(list(iq1_s_grid_bytes()), dtype=torch.uint8).view(torch.int8)
        _GRID_CACHE[resolved_device] = raw.reshape(2048, 8).to(
            device=resolved_device, dtype=torch.float32
        )
    return _GRID_CACHE[resolved_device]


def _encode_blocks(blocks: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    """Encode a moderate-size batch of flattened 256-value blocks."""
    x = narrow_to_float32(blocks)
    block_count = x.shape[0]
    vectors = x.reshape(block_count, 32, 8)
    xnorm = vectors.square().sum(dim=-1)
    xsum = vectors.sum(dim=-1)

    amax = x.abs().amax(dim=1)
    # The fixed-scale search favors a compressed super-block scale. This
    # empirical anchor initializes d below the full-range value.
    d = ((amax / _IQ1_S_NATIVE_MAX) * _IQ1_S_SCALE_ANCHOR).clamp(max=65504.0).to(torch.float16)
    d_float = d.float()

    best_error = torch.full((block_count, 32, 16), torch.inf, dtype=torch.float32, device=x.device)
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
                ).clamp_min_(0)
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
    return torch.where((d_float == 0).unsqueeze(1), 0, packed)


@torch.no_grad()
def quantize_iq1_s(
    weight: torch.Tensor, *, block_chunk_size: int = _DEFAULT_BLOCK_CHUNK_SIZE
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack a floating-point weight into GGML-compatible IQ1_S blocks.

    Returned shapes are ``[*weight.shape[:-1], weight.shape[-1] // 256, 50]``
    and ``[weight.ndim]``. The packed payload remains on the weight's device;
    the logical-shape metadata is kept on CPU. Non-finite input elements are
    treated as zero during packing.
    """
    validate_weight(weight, "IQ1_S")
    validate_block_chunk_size(block_chunk_size)

    logical_shape = torch.tensor(weight.shape, dtype=torch.int64)
    blocks = weight.contiguous().reshape(-1, IQ1_S_BLOCK_SIZE)
    grid = iq1_s_grid(weight.device)
    if weight.is_cuda:
        extension = get_cuda_ext_ggml()
        if extension is not None:
            packed = extension.iq1_s_pack(blocks, grid)
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
    block_chunk_size: int = _DEFAULT_DECODE_CHUNK_SIZE,
) -> torch.Tensor:
    """Decode GGML-compatible IQ1_S payload bytes."""
    shape = validate_packed_weights(
        packed_weights, weight_shape, block_bytes=IQ1_S_BLOCK_BYTES, format_name="IQ1_S"
    )
    validate_block_chunk_size(block_chunk_size)

    blocks = packed_weights.contiguous().reshape(-1, IQ1_S_BLOCK_BYTES)
    shifts = torch.tensor([0, 3, 6, 9], dtype=torch.int64, device=blocks.device)
    grid = iq1_s_grid(blocks.device)
    decoded = torch.empty((blocks.shape[0], IQ1_S_BLOCK_SIZE), dtype=dtype, device=blocks.device)
    for start in range(0, blocks.shape[0], block_chunk_size):
        stop = min(start + block_chunk_size, blocks.shape[0])
        block_chunk = blocks[start:stop]
        d = block_chunk[:, :2].contiguous().view(torch.float16).reshape(-1).float()
        low = block_chunk[:, 2:34].to(torch.int64).reshape(-1, 8, 4)
        qh = block_chunk[:, 34:50:2].to(torch.int64) | (
            block_chunk[:, 35:50:2].to(torch.int64) << 8
        )
        high = (qh.unsqueeze(-1) >> shifts) & 0x7
        entries = low | (high << 8)
        local = (qh >> 12) & 0x7
        delta = torch.where((qh & 0x8000).bool(), -_IQ1_S_DELTA, _IQ1_S_DELTA)
        values = grid[entries] + delta.unsqueeze(-1).unsqueeze(-1)
        scales = d.unsqueeze(-1) * (2 * local + 1).float()
        chunk_decoded = values * scales.unsqueeze(-1).unsqueeze(-1)
        decoded[start:stop] = chunk_decoded.reshape(-1, IQ1_S_BLOCK_SIZE)
    return decoded.reshape(shape)


def iq1_s_fake_quant(
    inputs: torch.Tensor,
    quantizer,
    *,
    block_chunk_size: int = _DEFAULT_BLOCK_CHUNK_SIZE,
    decode_chunk_size: int = _DEFAULT_DECODE_CHUNK_SIZE,
) -> torch.Tensor:
    """IQ1_S weight backend for TensorQuantizer, with pass-through backward."""
    if getattr(quantizer, "num_bits", None) != "iq1_s":
        raise ValueError("The ggml IQ1_S backend requires num_bits='iq1_s'")
    return fake_quantize_with_cache(
        inputs,
        quantizer,
        format_name="iq1_s",
        block_chunk_size=block_chunk_size,
        decode_chunk_size=decode_chunk_size,
        quantize=quantize_iq1_s,
        dequantize=dequantize_iq1_s,
    )
