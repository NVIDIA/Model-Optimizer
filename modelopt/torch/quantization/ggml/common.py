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
    """One weight's packed payload, reused across forwards.

    ``base_ref`` points at the parameter, not at the tensor the backend was handed.
    TensorQuantizer passes a fresh view of the weight on every forward, so a weakref to that
    view dies as soon as the forward returns and an identity check against it never matches
    again -- which is what kept this cache from ever hitting.

    Keeping it a weakref matters: a strong reference would pin full-precision storage alive and
    defeat offloaded or meta-device flows. Tying the entry to the parameter's lifetime means the
    payload stops being reused exactly when the weight it came from is released.
    """

    base_ref: weakref.ReferenceType
    input_key: tuple[object, ...]
    format_name: str
    block_chunk_size: int
    packed_weights: torch.Tensor
    weight_shape: torch.Tensor


def _cache_base(inputs: torch.Tensor) -> torch.Tensor:
    """The tensor whose lifetime the cached payload should follow.

    ``inputs`` is a per-forward view; ``inputs._base`` is the parameter behind it, which lives
    as long as the module does.
    """
    base = inputs._base
    return inputs if base is None else base


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
    decode_chunk_size: int,
    quantize: Callable[..., tuple[torch.Tensor, torch.Tensor]],
    dequantize: Callable[..., torch.Tensor],
) -> torch.Tensor:
    """Fake-quantize a weight while caching its compact packed representation."""
    input_key = _input_cache_key(inputs)
    cache_base = _cache_base(inputs)
    cache = getattr(quantizer, "_quantizer_cache", None)
    if (
        isinstance(cache, _PackedWeightCache)
        and input_key is not None
        and cache.base_ref() is cache_base
        and cache.input_key == input_key
        and cache.format_name == format_name
        and cache.block_chunk_size == block_chunk_size
    ):
        packed_weights, weight_shape = cache.packed_weights, cache.weight_shape
    else:
        packed_weights, weight_shape = quantize(inputs, block_chunk_size=block_chunk_size)
        if input_key is not None:
            quantizer._quantizer_cache = _PackedWeightCache(
                base_ref=weakref.ref(cache_base),
                input_key=input_key,
                format_name=format_name,
                block_chunk_size=block_chunk_size,
                packed_weights=packed_weights,
                weight_shape=weight_shape,
            )
        else:
            quantizer._quantizer_cache = None

    # Sized separately from the encode chunk: packing happens once per weight and is bounded
    # by its search temporaries, while this runs on every forward and is bounded by launches.
    reconstructed = dequantize(
        packed_weights,
        weight_shape,
        dtype=inputs.dtype,
        block_chunk_size=decode_chunk_size,
    )
    return inputs + (reconstructed - inputs).detach()


@dataclass(frozen=True)
class IQFormat:
    """Everything backend dispatch and export need to know about one IQ format.

    Each format module declares one of these beside its encoder and decoder, and
    :data:`~modelopt.torch.quantization.ggml.registry.IQ_FORMAT_REGISTRY` lists them. The
    per-format pieces -- codebook, search, payload layout -- stay in the format's module; what
    lives here is the part every format does the same way.
    """

    name: str
    block_size: int
    block_bytes: int
    quantize: Callable[..., tuple[torch.Tensor, torch.Tensor]]
    dequantize: Callable[..., torch.Tensor]
    # Encode and decode are chunked separately: packing runs once per weight and is bounded by
    # its search temporaries, decoding runs every forward and is bounded by kernel launches.
    block_chunk_size: int
    decode_chunk_size: int

    @property
    def effective_bits(self) -> float:
        """Packed storage cost per weight."""
        return self.block_bytes * 8 / self.block_size

    def fake_quant(
        self,
        inputs: torch.Tensor,
        quantizer,
        *,
        block_chunk_size: int | None = None,
        decode_chunk_size: int | None = None,
    ) -> torch.Tensor:
        """TensorQuantizer backend for this format, with pass-through backward."""
        if getattr(quantizer, "num_bits", None) != self.name:
            raise ValueError(
                f"The ggml {self.name.upper()} backend requires num_bits={self.name!r}"
            )
        return fake_quantize_with_cache(
            inputs,
            quantizer,
            format_name=self.name,
            block_chunk_size=(
                self.block_chunk_size if block_chunk_size is None else block_chunk_size
            ),
            decode_chunk_size=(
                self.decode_chunk_size if decode_chunk_size is None else decode_chunk_size
            ),
            quantize=self.quantize,
            dequantize=self.dequantize,
        )


def narrow_to_float32(blocks: torch.Tensor) -> torch.Tensor:
    """Narrow ``blocks`` to float32 the way the CUDA ``load_float`` helper does.

    Non-finite elements become zero, and finite elements outside the float32 range saturate
    instead of overflowing to infinity and then being zeroed. Sanitizing at the source precision
    is what keeps the reference encoders byte-identical to the extension for float64 weights;
    converting first would turn a finite 1e100 into zero on this path and into the float32
    maximum on the CUDA one.
    """
    finite = torch.nan_to_num(blocks, nan=0.0, posinf=0.0, neginf=0.0)
    if finite.dtype == torch.float64:
        # Only float64 can hold a finite value the narrowing would overflow. The float32 bounds
        # do not fit in the narrower dtypes, so clamping them would raise rather than no-op.
        info = torch.finfo(torch.float32)
        finite = finite.clamp(info.min, info.max)
    return finite.float()


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
