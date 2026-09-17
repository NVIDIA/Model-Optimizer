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

"""Shared validation for GGML-compatible block quantizers."""

import math
import weakref
from collections.abc import Callable
from dataclasses import dataclass

import torch

GGML_BLOCK_SIZE = 256


@dataclass
class _PackedWeightCache:
    input_ref: weakref.ReferenceType
    input_key: tuple[object, ...]
    format_name: str
    block_chunk_size: int
    packed_weights: torch.Tensor
    weight_shape: torch.Tensor


def _input_cache_key(inputs: torch.Tensor) -> tuple[object, ...] | None:
    try:
        version = inputs._version
    except RuntimeError:
        # Inference tensors can omit version counters, so changes cannot be detected safely.
        return None
    return (
        inputs.data_ptr(),
        tuple(inputs.shape),
        tuple(inputs.stride()),
        inputs.dtype,
        inputs.device,
        version,
    )


def fake_quantize_with_cache(
    inputs: torch.Tensor,
    quantizer,
    *,
    format_name: str,
    block_chunk_size: int,
    quantize: Callable[..., tuple[torch.Tensor, torch.Tensor]],
    dequantize: Callable[..., torch.Tensor],
) -> torch.Tensor:
    """Fake-quantize a weight while caching its compact packed representation."""
    input_key = _input_cache_key(inputs)
    cache = getattr(quantizer, "_quantizer_cache", None)
    if (
        isinstance(cache, _PackedWeightCache)
        and input_key is not None
        and cache.input_ref() is inputs
        and cache.input_key == input_key
        and cache.format_name == format_name
        and cache.block_chunk_size == block_chunk_size
    ):
        packed_weights, weight_shape = cache.packed_weights, cache.weight_shape
    else:
        packed_weights, weight_shape = quantize(inputs, block_chunk_size=block_chunk_size)
        if input_key is not None:
            quantizer._quantizer_cache = _PackedWeightCache(
                input_ref=weakref.ref(inputs),
                input_key=input_key,
                format_name=format_name,
                block_chunk_size=block_chunk_size,
                packed_weights=packed_weights,
                weight_shape=weight_shape,
            )
        else:
            quantizer._quantizer_cache = None

    reconstructed = dequantize(
        packed_weights,
        weight_shape,
        dtype=inputs.dtype,
        block_chunk_size=block_chunk_size,
    )
    return inputs + (reconstructed - inputs).detach()


def validate_weight(weight: torch.Tensor, format_name: str) -> None:
    """Validate weight metadata accepted by the current GGML block encoders."""
    if weight.numel() == 0:
        raise ValueError(f"{format_name} requires a non-empty weight")
    if weight.dim() == 0 or weight.shape[-1] % GGML_BLOCK_SIZE:
        raise ValueError(
            f"{format_name} requires the last weight dimension to be divisible by "
            f"{GGML_BLOCK_SIZE}, got shape {tuple(weight.shape)}"
        )
    if not weight.is_floating_point():
        raise TypeError(f"{format_name} requires a floating-point weight, got {weight.dtype}")


def validate_block_chunk_size(block_chunk_size: int) -> None:
    """Validate the common encoder and decoder block-chunk limit."""
    if isinstance(block_chunk_size, bool) or not isinstance(block_chunk_size, int):
        raise TypeError("block_chunk_size must be an integer")
    if block_chunk_size <= 0:
        raise ValueError(f"block_chunk_size must be positive, got {block_chunk_size}")


def validate_packed_weights(
    packed_weights: torch.Tensor,
    weight_shape: torch.Tensor,
    *,
    block_bytes: int,
    format_name: str,
) -> tuple[int, ...]:
    """Validate a packed payload and return its logical shape."""
    if (
        packed_weights.dim() == 0
        or packed_weights.dtype != torch.uint8
        or packed_weights.shape[-1] != block_bytes
    ):
        raise ValueError(
            f"packed_weights must be uint8 with last dimension {block_bytes}, "
            f"got {packed_weights.dtype} {tuple(packed_weights.shape)}"
        )
    integral_dtypes = {torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64}
    if weight_shape.dim() != 1 or weight_shape.dtype not in integral_dtypes:
        raise ValueError("weight_shape must be a one-dimensional integral tensor")
    shape = tuple(int(v) for v in weight_shape.detach().cpu().tolist())
    if not shape or any(dimension <= 0 for dimension in shape) or shape[-1] % GGML_BLOCK_SIZE:
        raise ValueError(f"invalid {format_name} logical weight shape: {shape}")
    expected_payload_values = math.prod(shape) // GGML_BLOCK_SIZE * block_bytes
    if packed_weights.numel() != expected_payload_values:
        raise ValueError("packed_weights size does not match weight_shape")
    return shape
