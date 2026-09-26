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

"""Q8_0 fake quantization and GGML-compatible block packing.

Every 32 logical values become one 34-byte ``block_q8_0`` payload: a
little-endian FP16 block scale followed by 32 signed int8 quants.
"""

import torch

from ..extensions import get_cuda_ext_ggml
from .common import (
    GGMLFormat,
    narrow_to_float32,
    validate_block_chunk_size,
    validate_packed_weights,
    validate_weight,
)

__all__ = [
    "Q8_0_BLOCK_BYTES",
    "Q8_0_BLOCK_SIZE",
    "Q8_0_EFFECTIVE_BITS",
    "Q8_0_FORMAT",
    "dequantize_q8_0",
    "q8_0_fake_quant",
    "quantize_q8_0",
]

Q8_0_BLOCK_SIZE = 32
Q8_0_BLOCK_BYTES = 34
Q8_0_EFFECTIVE_BITS = Q8_0_BLOCK_BYTES * 8 / Q8_0_BLOCK_SIZE
_Q8_0_MAX_QUANT = 127
_DEFAULT_BLOCK_CHUNK_SIZE = 4096
_DEFAULT_DECODE_CHUNK_SIZE = 16384


def _encode_blocks(blocks: torch.Tensor) -> torch.Tensor:
    """Encode a moderate-size batch of flattened 32-value blocks."""
    x = narrow_to_float32(blocks)
    block_count = x.shape[0]
    amax = x.abs().amax(dim=1)
    # Keep the serialized fp16 scale finite. This extends the existing GGML backend policy for
    # finite float64 values outside the float32/fp16 scale range.
    d_float = (amax / _Q8_0_MAX_QUANT).clamp(max=65504.0)
    d = d_float.to(torch.float16)
    inverse = torch.where(d_float > 0, d_float.reciprocal(), torch.zeros_like(d_float))
    normalized = x * inverse.unsqueeze(1)
    # C roundf, used by the canonical Q8_0 encoder, rounds half-way cases away from zero.
    rounded = normalized.sign() * (normalized.abs() + 0.5).floor()
    quants = rounded.clamp(-_Q8_0_MAX_QUANT, _Q8_0_MAX_QUANT).to(torch.int8)

    packed = torch.empty((block_count, Q8_0_BLOCK_BYTES), dtype=torch.uint8, device=x.device)
    packed[:, :2] = d.contiguous().view(torch.uint8).reshape(block_count, 2)
    packed[:, 2:] = quants.view(torch.uint8)
    return torch.where((d_float == 0).unsqueeze(1), 0, packed)


@torch.no_grad()
def quantize_q8_0(
    weight: torch.Tensor, *, block_chunk_size: int = _DEFAULT_BLOCK_CHUNK_SIZE
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack a floating-point weight into GGML-compatible Q8_0 blocks."""
    validate_weight(weight, "Q8_0", block_size=Q8_0_BLOCK_SIZE)
    validate_block_chunk_size(block_chunk_size)

    logical_shape = torch.tensor(weight.shape, dtype=torch.int64)
    blocks = weight.contiguous().reshape(-1, Q8_0_BLOCK_SIZE)
    if weight.is_cuda:
        extension = get_cuda_ext_ggml()
        if extension is not None:
            packed = extension.q8_0_pack(blocks)
            packed_shape = (
                *weight.shape[:-1],
                weight.shape[-1] // Q8_0_BLOCK_SIZE,
                Q8_0_BLOCK_BYTES,
            )
            return packed.reshape(packed_shape), logical_shape

    chunks = [
        _encode_blocks(blocks[start : start + block_chunk_size])
        for start in range(0, blocks.shape[0], block_chunk_size)
    ]
    packed_shape = (
        *weight.shape[:-1],
        weight.shape[-1] // Q8_0_BLOCK_SIZE,
        Q8_0_BLOCK_BYTES,
    )
    return torch.cat(chunks).reshape(packed_shape), logical_shape


@torch.no_grad()
def dequantize_q8_0(
    packed_weights: torch.Tensor,
    weight_shape: torch.Tensor,
    *,
    dtype: torch.dtype = torch.bfloat16,
    block_chunk_size: int = _DEFAULT_DECODE_CHUNK_SIZE,
) -> torch.Tensor:
    """Decode GGML-compatible Q8_0 payload bytes."""
    shape = validate_packed_weights(
        packed_weights,
        weight_shape,
        block_bytes=Q8_0_BLOCK_BYTES,
        block_size=Q8_0_BLOCK_SIZE,
        format_name="Q8_0",
    )
    validate_block_chunk_size(block_chunk_size)

    blocks = packed_weights.contiguous().reshape(-1, Q8_0_BLOCK_BYTES)
    decoded = torch.empty((blocks.shape[0], Q8_0_BLOCK_SIZE), dtype=dtype, device=blocks.device)
    for start in range(0, blocks.shape[0], block_chunk_size):
        stop = min(start + block_chunk_size, blocks.shape[0])
        block_chunk = blocks[start:stop]
        d = block_chunk[:, :2].contiguous().view(torch.float16).reshape(-1).float()
        quants = block_chunk[:, 2:].contiguous().view(torch.int8).float()
        decoded[start:stop] = (d.unsqueeze(1) * quants).to(dtype)
    return decoded.reshape(shape)


Q8_0_FORMAT = GGMLFormat(
    name="q8_0",
    block_size=Q8_0_BLOCK_SIZE,
    block_bytes=Q8_0_BLOCK_BYTES,
    quantize=quantize_q8_0,
    dequantize=dequantize_q8_0,
    block_chunk_size=_DEFAULT_BLOCK_CHUNK_SIZE,
    decode_chunk_size=_DEFAULT_DECODE_CHUNK_SIZE,
)

# Keep the public per-format entry point on the same record used by backend dispatch.
q8_0_fake_quant = Q8_0_FORMAT.fake_quant
