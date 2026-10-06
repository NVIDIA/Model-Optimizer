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
from functools import partial
from typing import NamedTuple

import torch

GGML_BLOCK_SIZE = 256
_SIGNED_INT_OF_SIZE = {2: torch.int16, 4: torch.int32, 8: torch.int64}


class _CacheKey(NamedTuple):
    """What a cached payload was packed from; any change means it must be packed again."""

    data_ptr: int
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: torch.dtype
    device: torch.device
    version: int


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
    input_key: _CacheKey
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


def _input_cache_key(inputs: torch.Tensor) -> _CacheKey | None:
    try:
        version = inputs._version
    except RuntimeError:
        # Inference tensors can omit version counters, so changes cannot be detected safely.
        return None
    return _CacheKey(
        inputs.data_ptr(),
        tuple(inputs.shape),
        tuple(inputs.stride()),
        inputs.dtype,
        inputs.device,
        version,
    )


def _is_contiguous(shape: tuple[int, ...], stride: tuple[int, ...]) -> bool:
    expected = 1
    for size, step in zip(reversed(shape), reversed(stride)):
        if size != 1 and step != expected:
            return False
        expected *= size
    return True


def _same_elements(key: _CacheKey, other: _CacheKey) -> bool:
    """Whether two cache keys are contiguous views of the same values in the same order."""
    return (
        (key.data_ptr, key.dtype, key.device, key.version)
        == (other.data_ptr, other.dtype, other.device, other.version)
        and math.prod(key.shape) == math.prod(other.shape)
        and _is_contiguous(key.shape, key.stride)
        and _is_contiguous(other.shape, other.stride)
    )


def _matching_cache(
    quantizer, inputs: torch.Tensor, format_name: str, *, any_contiguous_view: bool = False
) -> _PackedWeightCache | None:
    """The quantizer's cached payload if it was packed from ``inputs``, else None.

    It matches only while the weight is unchanged since it was packed: the key carries the
    tensor's version counter, so an in-place update misses. With ``any_contiguous_view`` it also
    matches another contiguous view of the same values: TensorQuantizer hands fake quant the
    weight reshaped into 256-value blocks, so export, which holds the weight itself, sees a
    different shape over the same elements in the same order, and so the same GGML blocks.
    """
    cache = getattr(quantizer, "_quantizer_cache", None)
    input_key = _input_cache_key(inputs)
    if not (
        isinstance(cache, _PackedWeightCache)
        and input_key is not None
        and cache.base_ref() is _cache_base(inputs)
        and cache.format_name == format_name
    ):
        return None
    if cache.input_key == input_key or (
        any_contiguous_view and _same_elements(cache.input_key, input_key)
    ):
        return cache
    return None


def pin_packed_weight(quantizer, format_name: str, packed_weights: torch.Tensor) -> None:
    """Make ``packed_weights`` the ``format_name`` payload of ``quantizer``'s weight while it fits.

    For payloads an algorithm chose itself, such as GPTQ, which leaves the weight equal to their
    decoding: encoding that decoding again does not return the same codes. The pin is a buffer, so
    it moves with offloading and is kept by ``mto.save``/``restore`` and layerwise checkpoints. Fake
    quant and export use it while the weight, or the rows an export takes from it, is exactly its
    decoding, and encode afresh otherwise -- including once the quantizer is given another format.
    """
    quantizer._set_buffer(_pin_name(format_name), packed_weights)
    quantizer._quantizer_cache = None


def pinned_packed_weight(quantizer, format_name: str) -> torch.Tensor | None:
    """The ``format_name`` payload pinned to ``quantizer``, if any."""
    return getattr(quantizer, _pin_name(format_name), None)


def _pin_name(format_name: str) -> str:
    return f"_ggml_pinned_{format_name}"


def _matching_rows(table: torch.Tensor, rows: torch.Tensor) -> torch.Tensor | None:
    """For each row of ``rows``, the index of an equal row of ``table``; None if one has none."""
    table_keys, row_keys = _row_keys(table), _row_keys(rows)
    order = table_keys.argsort()
    found = order[torch.searchsorted(table_keys[order], row_keys).clamp_max(len(order) - 1)]
    return found if torch.equal(table[found], rows) else None


def _row_keys(x: torch.Tensor) -> torch.Tensor:
    """A 64-bit hash of each row's bits; equal rows always get equal keys.

    Integer sums wrap and are associative, so unlike a floating-point projection the key does not
    depend on how the reduction is ordered, which differs between differently shaped products.
    """
    multipliers = torch.randint(
        -(2**62), 2**62, (x.shape[-1],), generator=torch.Generator().manual_seed(0)
    )
    bits = x.contiguous().view(_SIGNED_INT_OF_SIZE[x.element_size()]).to(torch.int64)
    return (bits * multipliers.to(x.device)).sum(-1)


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
    cache = _matching_cache(quantizer, inputs, format_name)
    if cache is not None and cache.block_chunk_size == block_chunk_size:
        packed_weights, weight_shape = cache.packed_weights, cache.weight_shape
    else:
        packed_weights, weight_shape = quantize(inputs, block_chunk_size=block_chunk_size)
        input_key = _input_cache_key(inputs)
        if input_key is not None:
            quantizer._quantizer_cache = _PackedWeightCache(
                base_ref=weakref.ref(_cache_base(inputs)),
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
class GGMLFormat:
    """Everything backend dispatch and export need to know about one GGML format.

    Each format module declares one of these beside its encoder and decoder, and
    :data:`~modelopt.torch.quantization.ggml.registry.GGML_FORMAT_REGISTRY` lists them. The
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
            quantize=partial(self._pinned_or_quantize, quantizer),
            dequantize=self.dequantize,
        )

    def pack(self, weight: torch.Tensor, quantizer=None) -> torch.Tensor:
        """The packed payload of ``weight``, reusing the one fake quant cached if it matches.

        Export calls this. A model that ran a forward since quantization already holds each
        weight's payload, and reusing it both skips a second search and exports exactly the
        bytes the evaluated model decoded. Any other tensor is packed afresh, or takes the payload
        pinned to ``quantizer`` if it is that payload's decoding.
        """
        cache = (
            _matching_cache(quantizer, weight, self.name, any_contiguous_view=True)
            if quantizer is not None
            else None
        )
        if cache is None:
            return self._pinned_or_quantize(quantizer, weight)[0]
        return cache.packed_weights.reshape(
            *weight.shape[:-1], weight.shape[-1] // self.block_size, self.block_bytes
        )

    def _pinned_or_quantize(
        self, quantizer, weight: torch.Tensor, *, block_chunk_size: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``quantize(weight)``, or the pinned payload's rows if ``weight`` decodes them."""
        packed = self._pinned_rows(quantizer, weight)
        if packed is not None:
            return packed, torch.tensor(weight.shape, dtype=torch.int64)
        if block_chunk_size is None:
            return self.quantize(weight)
        return self.quantize(weight, block_chunk_size=block_chunk_size)

    def _pinned_rows(self, quantizer, weight: torch.Tensor) -> torch.Tensor | None:
        """The pinned payload of ``weight``, a view of the pinned weight or rows taken from it.

        Fake quant hands over the pinned weight itself in 256-value rows; export may hand over row
        slices of it, such as the gate and up halves of a fused projection. The comparison syncs,
        but it only runs for weights the identity cache has not seen.
        """
        pinned = pinned_packed_weight(quantizer, self.name)
        width = weight.shape[-1] // self.block_size
        if (
            pinned is None
            or weight.shape[-1] % self.block_size
            or pinned.numel() % (width * self.block_bytes)
        ):
            return None
        pinned = pinned.to(weight.device).reshape(-1, width, self.block_bytes)
        table = self.dequantize(
            pinned, torch.tensor((pinned.shape[0], weight.shape[-1])), dtype=torch.float32
        ).to(weight.dtype)
        rows = weight.reshape(-1, weight.shape[-1])
        if rows.shape == table.shape and torch.equal(rows, table):
            index = slice(None)
        else:
            index = _matching_rows(table, rows)
            if index is None:
                return None
        return pinned[index].reshape(*weight.shape[:-1], width, self.block_bytes)


# Compatibility alias for callers that imported the record type before the registry was
# generalized from the IQ family to every supported GGML block format.
IQFormat = GGMLFormat


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


def validate_weight(
    weight: torch.Tensor, format_name: str, *, block_size: int = GGML_BLOCK_SIZE
) -> None:
    """Validate weight metadata accepted by the current GGML block encoders."""
    if weight.numel() == 0:
        raise ValueError(f"{format_name} requires a non-empty weight")
    if weight.dim() == 0 or weight.shape[-1] % block_size:
        raise ValueError(
            f"{format_name} requires the last weight dimension to be divisible by "
            f"{block_size}, got shape {tuple(weight.shape)}"
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
    block_size: int = GGML_BLOCK_SIZE,
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
    if not shape or any(dimension <= 0 for dimension in shape) or shape[-1] % block_size:
        raise ValueError(f"invalid {format_name} logical weight shape: {shape}")
    expected_payload_values = math.prod(shape) // block_size * block_bytes
    if packed_weights.numel() != expected_payload_values:
        raise ValueError("packed_weights size does not match weight_shape")
    return shape
