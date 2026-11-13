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

import itertools
import math
import weakref
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import NamedTuple

import torch
from torch.optim.optimizer import register_optimizer_step_post_hook

GGML_BLOCK_SIZE = 256
_SIGNED_INT_OF_SIZE = {2: torch.int16, 4: torch.int32, 8: torch.int64}
# Columns hashed at a time, bounding _row_keys' int64 temporaries to this slice of the weight.
_ROW_KEY_CHUNK = 256
# Blocks hashed at once by _block_keys, which bounds its float32 and int64 temporaries.
_KEY_BLOCKS_PER_CHUNK = 4096
# Dtypes a model may be cast to after calibration (``model.bfloat16()``, Megatron's
# ``Float16Module``); a pin's source fingerprint also holds the source rounded to each.
_CAST_DTYPES = (torch.bfloat16, torch.float16)


class _CacheKey(NamedTuple):
    """What a cached payload was packed from; any change means it must be packed again."""

    data_ptr: int
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: torch.dtype
    device: torch.device
    version: int


@dataclass(eq=False)
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
    # Set by the next optimizer step when the weight is trainable; see _track_trainable.
    stale: bool = False


# Payloads packed from a trainable weight, marked stale by the next optimizer step. Their key
# carries the weight's version counter, which an optimizer writing the weight through a shared
# buffer or ``.data`` leaves as it was: Megatron's distributed optimizer copies updated weights into
# the DDP parameter buffer and all-gathers it, so without this a QAD job keeps the codes of its
# first forward.
_TRAINABLE_CACHES: weakref.WeakSet = weakref.WeakSet()
_STEP_HOOK: list = []


def _mark_trainable_caches_stale(*_) -> None:
    for cache in list(_TRAINABLE_CACHES):
        cache.stale = True
    _TRAINABLE_CACHES.clear()


def _track_trainable(cache: _PackedWeightCache, base: torch.Tensor) -> None:
    """Have the next optimizer step mark ``cache`` stale if ``base`` is a trainable weight."""
    if not base.requires_grad:
        return
    if not _STEP_HOOK:
        _STEP_HOOK.append(register_optimizer_step_post_hook(_mark_trainable_caches_stale))
    _TRAINABLE_CACHES.add(cache)


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
    tensor's version counter, so an in-place update misses, and an optimizer step retires a
    trainable weight's payload however the step wrote the weight. With ``any_contiguous_view`` it
    also matches another contiguous view of the same values: TensorQuantizer hands fake quant the
    weight reshaped into 256-value blocks, so export, which holds the weight itself, sees a
    different shape over the same elements in the same order, and so the same GGML blocks.
    """
    cache = getattr(quantizer, "_quantizer_cache", None)
    input_key = _input_cache_key(inputs)
    if not (
        isinstance(cache, _PackedWeightCache)
        and input_key is not None
        and not cache.stale
        and cache.base_ref() is _cache_base(inputs)
        and cache.format_name == format_name
    ):
        return None
    if cache.input_key == input_key or (
        any_contiguous_view and _same_elements(cache.input_key, input_key)
    ):
        return cache
    return None


def pin_packed_weight(
    quantizer,
    format_name: str,
    packed_weights: torch.Tensor,
    importance: torch.Tensor | None = None,
    source: torch.Tensor | None = None,
    search_scales: torch.Tensor | None = None,
) -> None:
    """Make ``packed_weights`` the ``format_name`` payload of ``quantizer``'s weight.

    For payloads an algorithm chose itself, such as GPTQ. The pin is a buffer, so it moves with
    offloading and is kept by ``mto.save``/``restore`` and layerwise checkpoints. A format with
    :attr:`GGMLFormat.payload_scales` then encodes every weight under the pinned scales and the
    ``importance`` its search was weighted with: the weight the payload was encoded from and its
    decoding both encode to the payload, and a weight QAD updated to the codes nearest it under
    those scales. Other formats use the pin while the weight, or the rows an export takes from it,
    is exactly its decoding, and encode afresh otherwise.

    ``source`` is the weight the payload's per-group choices were picked for, when a search other
    than the encoder's own picked them, such as a local-Hessian one. Encoding again would re-pick
    them, so each block still equal to that block of ``source``, or to its rounding by a cast to
    bfloat16 or float16, keeps its pinned codes; a block QAD changed is encoded as above, its local
    scales searched again under the pinned scale.

    ``search_scales`` are the FP16 super-block scales the codes were searched at, when the pinned
    ones were raised after the search, as :func:`quantize_energy_matched` raises them. Encoding again
    then searches at these and keeps the pinned ones, so a block QAD changed is not searched at the
    raised scale, which would shrink it. They are kept as their bits, which casting the model leaves.
    """
    quantizer._set_buffer(_pin_name(format_name), packed_weights)
    importance_name = PINNED_IMPORTANCE_PREFIX + format_name
    if importance is not None:
        quantizer._set_buffer(importance_name, importance.float())
    elif hasattr(quantizer, importance_name):
        delattr(quantizer, importance_name)
    source_name = _source_name(format_name)
    if source is not None:
        blocks_shape = packed_weights.shape[:-1]
        keys = [_block_keys(source, blocks_shape, dtype) for dtype in (None, *_CAST_DTYPES)]
        quantizer._set_buffer(source_name, torch.stack(keys, dim=-1))
    elif hasattr(quantizer, source_name):
        delattr(quantizer, source_name)
    search_name = _search_scales_name(format_name)
    if search_scales is not None:
        bits = search_scales.to(torch.float16).contiguous().view(torch.int16)
        quantizer._set_buffer(search_name, bits.reshape(packed_weights.shape[:-1]))
    elif hasattr(quantizer, search_name):
        delattr(quantizer, search_name)
    quantizer._quantizer_cache = None


def pinned_packed_weight(quantizer, format_name: str) -> torch.Tensor | None:
    """The ``format_name`` payload pinned to ``quantizer``, if any."""
    return getattr(quantizer, _pin_name(format_name), None)


def pinned_importance(quantizer, format_name: str) -> torch.Tensor | None:
    """The per-input-column importance the ``format_name`` payload pinned to ``quantizer`` used."""
    return getattr(quantizer, PINNED_IMPORTANCE_PREFIX + format_name, None)


# Quantizer attribute that makes its pinned payload stand in for whatever weight it is handed.
# AutoQuantize sets it while it keeps the source weight for other candidates, so a candidate whose
# calibration rewrote the weight (GPTQ) still replays its own codes.
PIN_OVERRIDES_WEIGHT = "_ggml_pin_overrides_weight"


# Prefix of the buffer holding a pinned payload; checkpoint code keeps buffers with it.
PINNED_PAYLOAD_PREFIX = "_ggml_pinned_"
# Prefix of the buffer holding the importance a pinned payload was searched with, a vector over
# the weight's input columns.
PINNED_IMPORTANCE_PREFIX = "_ggml_importance_"


def _pin_name(format_name: str) -> str:
    return f"{PINNED_PAYLOAD_PREFIX}{format_name}"


def _search_scales_name(format_name: str) -> str:
    """The buffer holding the FP16 bits of the scales a pin's codes were searched at, ``[*rows, blocks]``."""
    return f"{_pin_name(format_name)}_search_scales"


def _pinned_search_scales(quantizer, format_name: str) -> torch.Tensor | None:
    """The FP16 scales the ``format_name`` payload pinned to ``quantizer`` was searched at, if kept."""
    bits = getattr(quantizer, _search_scales_name(format_name), None)
    return None if bits is None else bits.reshape(-1).view(torch.float16)


def drop_search_scales(quantizer) -> None:
    """Forget the scales ``quantizer``'s pin was searched at, once its weight became its decoding.

    The search scales are for the weight the codes were searched from; its decoding encodes to them
    at the pinned scales instead, as :func:`~modelopt.torch.quantization.initialize_qdq_weights`
    leaves it. A quantizer with no such pin is left alone.
    """
    num_bits = getattr(quantizer, "num_bits", None)
    if isinstance(num_bits, str) and hasattr(quantizer, _search_scales_name(num_bits)):
        delattr(quantizer, _search_scales_name(num_bits))
        quantizer._quantizer_cache = None


def _source_name(format_name: str) -> str:
    """The buffer holding a pin's source fingerprint, ``[*rows, blocks, keys]``.

    It shares the payload's prefix, so checkpointing and sharding treat it as part of the pin, and
    shards like the payload's leading dimensions.
    """
    return f"{_pin_name(format_name)}_source"


def _block_keys(
    weight: torch.Tensor, blocks_shape: tuple[int, ...], dtype: torch.dtype | None = None
) -> torch.Tensor:
    """A :func:`_row_keys` hash of each 256-value block of ``weight`` as float32, ``blocks_shape``.

    Hashing float32 values lets a lossless upcast of the weight, as folding hands over, match.
    With ``dtype``, each block is first rounded to it, as a cast of the weight would.
    """
    blocks = weight.reshape(-1, GGML_BLOCK_SIZE)
    keys = torch.empty(blocks.shape[0], dtype=torch.int64, device=weight.device)
    for start in range(0, blocks.shape[0], _KEY_BLOCKS_PER_CHUNK):
        chunk = blocks[start : start + _KEY_BLOCKS_PER_CHUNK]
        chunk = chunk if dtype is None else chunk.to(dtype)
        keys[start : start + _KEY_BLOCKS_PER_CHUNK] = _row_keys(chunk.float())
    return keys.reshape(blocks_shape)


def _matching_rows(table: torch.Tensor, rows: torch.Tensor) -> torch.Tensor | None:
    """For each row of ``rows``, the index of an equal row of ``table``; None if one has none."""
    table_keys, row_keys = _row_keys(table), _row_keys(rows)
    order = table_keys.argsort()
    found = order[torch.searchsorted(table_keys[order], row_keys).clamp_max(len(order) - 1)]
    return found if torch.equal(table[found], rows) else None


def _row_keys(x: torch.Tensor) -> torch.Tensor:
    """A 64-bit hash of each row's bits; equal rows always get equal keys.

    Integer sums wrap and are associative, so unlike a floating-point projection the key does not
    depend on how the reduction is ordered, which differs between differently shaped products. That
    is also what lets the columns be summed a chunk at a time.
    """
    multipliers = torch.randint(
        -(2**62), 2**62, (x.shape[-1],), generator=torch.Generator().manual_seed(0)
    ).to(x.device)
    bits = x.contiguous().view(_SIGNED_INT_OF_SIZE[x.element_size()])
    keys = torch.zeros(x.shape[:-1], dtype=torch.int64, device=x.device)
    for start in range(0, x.shape[-1], _ROW_KEY_CHUNK):
        columns = slice(start, start + _ROW_KEY_CHUNK)
        keys += (bits[..., columns].to(torch.int64) * multipliers[columns]).sum(-1)
    return keys


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
        # A weight frozen when it was packed may since have been made trainable in place.
        _track_trainable(cache, _cache_base(inputs))
    else:
        packed_weights, weight_shape = quantize(inputs, block_chunk_size=block_chunk_size)
        input_key = _input_cache_key(inputs)
        if input_key is not None:
            base = _cache_base(inputs)
            quantizer._quantizer_cache = _PackedWeightCache(
                base_ref=weakref.ref(base),
                input_key=input_key,
                format_name=format_name,
                block_chunk_size=block_chunk_size,
                packed_weights=packed_weights,
                weight_shape=weight_shape,
            )
            _track_trainable(quantizer._quantizer_cache, base)
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
    # Exactly the reconstruction export decodes, with a straight-through gradient. The usual
    # ``inputs + (reconstructed - inputs).detach()`` rounds twice in a narrow dtype, so it can
    # differ wherever the two are far apart, as when a pinned payload replays over a source weight.
    return reconstructed + (inputs - inputs.detach())


@dataclass(frozen=True)
class ScaleFit:
    """How a format's super-block scale follows from the values it encodes.

    Each ``group``-value sub-block is fit in weighted least squares by one scale times three
    ``levels`` -- magnitudes for IQ2, signed values once per delta sign (one level triple each) for
    IQ1 -- and the super-block scale is the largest fitted scale over ``top``, the largest local
    multiplier, so the local scales span every sub-block without an empirical anchor.
    """

    group: int
    top: int
    levels: tuple[tuple[float, float, float], ...]

    @property
    def signed(self) -> bool:
        """Whether the levels fit signed values rather than magnitudes."""
        return min(min(levels) for levels in self.levels) < 0


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
    # Every decoded weight is an int8 level times a float32 step shared by ``int8_group_size``
    # weights. ``unpack_int8`` returns both for a chunk of blocks, and the torch decoder
    # multiplies them out, so the integer view and the decoding cannot disagree.
    int8_group_size: int
    unpack_int8: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]
    # Whether ``quantize`` takes a per-input-column ``importance`` that weights its search.
    weighted: bool = False
    # Each block picks one of ``num_choices`` options -- a local scale, and for IQ1_S its delta --
    # for each of its ``choice_groups`` groups; ``quantize`` takes them as ``choices``. Zero for
    # formats without one.
    choice_groups: int = 0
    num_choices: int = 0
    # Reads the FP16 super-block scale of each ``[blocks, block_bytes]`` payload row, for formats
    # whose ``quantize`` takes them as ``scales``; see :func:`pin_packed_weight`.
    payload_scales: Callable[[torch.Tensor], torch.Tensor] | None = None
    # How :func:`quantize_energy_matched` derives each block's super-block scale, and writes one
    # back into ``[blocks, block_bytes]`` payload rows.
    scale_fit: ScaleFit | None = None
    replace_payload_scales: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None

    def __post_init__(self):
        if (self.scale_fit is None) != (self.replace_payload_scales is None):
            raise ValueError(f"{self.name}: scale_fit needs replace_payload_scales, and vice versa")

    @property
    def effective_bits(self) -> float:
        """Packed storage cost per weight."""
        return self.block_bytes * 8 / self.block_size

    def decode_int8(
        self, packed_weights: torch.Tensor, weight_shape: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode a payload into int8 levels and the float32 step of each ``int8_group_size``.

        ``levels * steps`` (each step repeated over its group) equals ``dequantize`` in float32
        exactly, so a weight-only integer kernel with these group scales reproduces the format.

        Returns:
            ``levels`` shaped like the weight and ``steps`` shaped
            ``[*weight_shape[:-1], weight_shape[-1] // int8_group_size]``.
        """
        shape = validate_packed_weights(
            packed_weights,
            weight_shape,
            block_bytes=self.block_bytes,
            block_size=self.block_size,
            format_name=self.name.upper(),
        )
        blocks = packed_weights.contiguous().reshape(-1, self.block_bytes)
        groups = self.block_size // self.int8_group_size
        levels = torch.empty((len(blocks), self.block_size), dtype=torch.int8, device=blocks.device)
        steps = torch.empty((len(blocks), groups), dtype=torch.float32, device=blocks.device)
        for start in range(0, len(blocks), self.decode_chunk_size):
            stop = start + self.decode_chunk_size
            chunk_levels, chunk_steps = self.unpack_int8(blocks[start:stop])
            levels[start:stop] = chunk_levels
            steps[start:stop] = chunk_steps
        return levels.reshape(shape), steps.reshape(*shape[:-1], -1)

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
        """``quantize(weight)``, under the pinned scales or as the pinned rows ``weight`` decodes."""
        shape = torch.tensor(weight.shape, dtype=torch.int64)
        kwargs = {} if block_chunk_size is None else {"block_chunk_size": block_chunk_size}
        pinned = pinned_packed_weight(quantizer, self.name)
        keeps_scales = pinned is not None and self.payload_scales is not None
        # Unless the pin stands in for the weight (PIN_OVERRIDES_WEIGHT): AutoQuantize then keeps
        # the source weight, which these scales would encode to other codes than the pinned ones.
        if (
            pinned is not None
            and self.payload_scales is not None
            and not getattr(quantizer, PIN_OVERRIDES_WEIGHT, False)
        ):
            # Encode in the pinned weight's shape, whose last dimension the importance spans; fake
            # quant hands the weight over in 256-value rows.
            pinned_shape = (*pinned.shape[:-2], pinned.shape[-2] * self.block_size)
            if math.prod(pinned_shape) == weight.numel():
                source = getattr(quantizer, _source_name(self.name), None)
                unchanged = None
                if source is not None:
                    # A fingerprint pinned before it held cast keys is [*rows, blocks].
                    source = source.to(weight.device).reshape(*pinned.shape[:-1], -1)
                    keys = _block_keys(weight, pinned.shape[:-1])
                    unchanged = (keys[..., None] == source).any(-1)
                    if bool(unchanged.all()):
                        packed = pinned.to(weight.device)
                        return packed.reshape(*weight.shape[:-1], -1, self.block_bytes), shape
                importance = pinned_importance(quantizer, self.name)
                scales = self.payload_scales(pinned.reshape(-1, self.block_bytes)).to(weight.device)
                searched = _pinned_search_scales(quantizer, self.name)
                packed, _ = self.quantize(
                    weight.reshape(pinned_shape),
                    scales=scales if searched is None else searched.to(weight.device),
                    importance=None if importance is None else importance.to(weight.device),
                    **kwargs,
                )
                if searched is not None and self.replace_payload_scales is not None:
                    rows = self.replace_payload_scales(packed.reshape(-1, self.block_bytes), scales)
                    packed = rows.reshape(packed.shape)
                if unchanged is not None:
                    packed = torch.where(unchanged[..., None], pinned.to(packed.device), packed)
                return packed.reshape(*weight.shape[:-1], -1, self.block_bytes), shape
        # Rows of a decoding, as an export splitting a fused weight hands over.
        packed = self._pinned_rows(quantizer, weight)
        if packed is not None:
            return packed, shape
        if keeps_scales:
            raise ValueError(
                f"{self.name.upper()} keeps the scales pinned to its quantizer, but was handed "
                f"{tuple(weight.shape)} rows that are neither the pinned weight nor rows of its "
                "decoding; see pin_current_codes."
            )
        return self.plain_quantize(weight, **kwargs)

    def plain_quantize(
        self, weight: torch.Tensor, *, block_chunk_size: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode a weight no calibration chose codes for, as its forwards and export do.

        Formats with a :attr:`scale_fit` take :func:`quantize_energy_matched`, unweighted.
        """
        kwargs = {} if block_chunk_size is None else {"block_chunk_size": block_chunk_size}
        if self.scale_fit is None:
            return self.quantize(weight, **kwargs)
        packed, shape, _ = quantize_energy_matched(self, weight, **kwargs)
        return packed, shape

    def pin_current_codes(
        self, weight: torch.Tensor, quantizer, dtype: torch.dtype | None = None
    ) -> torch.Tensor:
        """Pin the codes ``weight`` encodes to under its pinned scales; return their decoding.

        An export that splits a fused weight into row slices hands :meth:`pack` slices of this
        decoding in its ``dtype``, which it finds among the pinned rows. Unless ``quantizer``
        keeps pinned scales, ``weight`` is returned in ``dtype`` as it is.
        """
        dtype = weight.dtype if dtype is None else dtype
        pinned = None if quantizer is None else pinned_packed_weight(quantizer, self.name)
        if pinned is None or self.payload_scales is None:
            return weight.to(dtype)
        packed, shape = self._pinned_or_quantize(quantizer, weight)
        # The source's keys stay valid: its unchanged blocks are the ones the new pin kept.
        source = getattr(quantizer, _source_name(self.name), None)
        pin_packed_weight(
            quantizer,
            self.name,
            packed.reshape(pinned.shape).to(pinned.device),
            pinned_importance(quantizer, self.name),
            search_scales=_pinned_search_scales(quantizer, self.name),
        )
        if source is not None:
            quantizer._set_buffer(_source_name(self.name), source)
        return self.dequantize(packed, shape, dtype=dtype)

    def _pinned_rows(self, quantizer, weight: torch.Tensor) -> torch.Tensor | None:
        """The pinned payload of ``weight``, a view of the pinned weight or rows taken from it.

        Fake quant hands over the pinned weight itself in 256-value rows; export may hand over row
        slices of it, such as the gate and up halves of a fused projection. Folding hands over a
        float32 copy of a narrower weight, which holds the decoding rounded to that narrower dtype,
        so a float32 weight also matches the decoding rounded through BF16 or FP16. The comparison
        syncs, but it only runs for weights the identity cache has not seen.
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
        if (
            getattr(quantizer, PIN_OVERRIDES_WEIGHT, False)
            and self.full_pin(quantizer, weight) is not None
        ):
            return pinned.reshape(*weight.shape[:-1], width, self.block_bytes)
        shape = torch.tensor((pinned.shape[0], weight.shape[-1]))
        rows = weight.reshape(-1, weight.shape[-1])
        storage_dtypes = [weight.dtype]
        if weight.dtype == torch.float32:
            storage_dtypes += [torch.bfloat16, torch.float16]

        def decoding(storage_dtype):
            # Decoding straight into the storage dtype rounds once, as GPTQ did writing the weight,
            # and holds one table at a time rather than a float32 copy beside each rounding.
            return self.dequantize(pinned, shape, dtype=storage_dtype).to(weight.dtype)

        # Every whole-weight comparison first: matching rows builds int64 keys several times the
        # weight's size, and only row slices need it.
        if rows.shape[0] == pinned.shape[0]:
            for storage_dtype in storage_dtypes:
                if torch.equal(rows, decoding(storage_dtype)):
                    return pinned.reshape(*weight.shape[:-1], width, self.block_bytes)
        for storage_dtype in storage_dtypes:
            index = _matching_rows(decoding(storage_dtype), rows)
            if index is not None:
                return pinned[index].reshape(*weight.shape[:-1], width, self.block_bytes)
        return None

    def full_pin(self, quantizer, weight: torch.Tensor) -> torch.Tensor | None:
        """The payload ``quantizer`` pins for all of ``weight``; None if it pins none or some rows."""
        pinned = pinned_packed_weight(quantizer, self.name)
        if pinned is None or (
            pinned.numel() != weight.numel() // self.block_size * self.block_bytes
        ):
            return None
        return pinned

    def decode_pinned(self, quantizer, weight: torch.Tensor) -> torch.Tensor:
        """``weight`` as its pinned payload decodes, rounded once to its dtype as GPTQ wrote it."""
        pinned = self.full_pin(quantizer, weight)
        if pinned is None:
            raise ValueError(
                f"No {self.name.upper()} payload pinned for a {tuple(weight.shape)} weight"
            )
        return self.dequantize(
            pinned.to(weight.device).reshape(-1, self.block_bytes),
            torch.tensor(weight.shape),
            dtype=weight.dtype,
        )


def quantize_energy_matched(
    ggml_format: GGMLFormat,
    weight: torch.Tensor,
    second_moments: torch.Tensor | None = None,
    *,
    block_chunk_size: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Encode ``weight`` at derived super-block scales, then match each block's output energy.

    ``second_moments`` is the floored per-input-column ``E[x_j^2]``, uniform when ``None``. Each
    block's scale comes from its sub-blocks' least-squares fits (:class:`ScaleFit`) and the format's
    search picks the codes and local scales at it, both weighted by the root of ``second_moments``
    as GPTQ's search is: the raw statistic is heavy-tailed. The stored scale is then rescaled so the
    block decodes with the original's output energy, ``sum_j v_j q_j^2 = sum_j v_j w_j^2`` under
    ``v = second_moments``. A least-squares search shrinks each block's output by its error; a gain
    of one instead overshoots, adding the error's energy on top. Matching the energy lands between
    the two and needs no per-format constant.

    Returns:
        The payload, the logical shape, and the FP16 scale each block's codes were searched at.
    """
    fit, replace = ggml_format.scale_fit, ggml_format.replace_payload_scales
    if fit is None or replace is None:
        raise ValueError(f"{ggml_format.name.upper()} has no super-block scale to derive")
    validate_weight(weight, ggml_format.name.upper(), block_size=ggml_format.block_size)
    blocks = weight.detach().reshape(-1, GGML_BLOCK_SIZE)
    columns = importance_blocks(second_moments, weight, ggml_format.name.upper())
    importance = None if second_moments is None else second_moments.sqrt()
    scales = torch.cat(
        [
            _derived_scales(blocks[start : start + _FIT_BLOCKS_PER_CHUNK], columns, start, fit)
            for start in range(0, len(blocks), _FIT_BLOCKS_PER_CHUNK)
        ]
    )
    kwargs = {} if block_chunk_size is None else {"block_chunk_size": block_chunk_size}
    packed, shape = ggml_format.quantize(weight, scales=scales, importance=importance, **kwargs)
    rows = packed.reshape(-1, ggml_format.block_bytes)
    chunk = ggml_format.decode_chunk_size
    factors = torch.cat(
        [
            _energy_factors(ggml_format, blocks, rows, columns, start, start + chunk)
            for start in range(0, len(blocks), chunk)
        ]
    )
    stored = (scales.float() * factors).clamp(max=65504.0).to(torch.float16)
    return replace(rows, stored).reshape(packed.shape), shape, scales


def pin_energy_matched(
    ggml_format: GGMLFormat, quantizer, weight: torch.Tensor, second_moments: torch.Tensor | None
) -> None:
    """Pin the :func:`quantize_energy_matched` codes of ``weight`` with the scales they were searched at.

    ``weight`` itself is left as it is. Encoding it again, as QAD does once it changes a block,
    searches at those scales and keeps the raised ones, so the codes stay where the calibration put
    them (see :func:`pin_packed_weight`).
    """
    if ggml_format.payload_scales is None:
        raise ValueError(f"{ggml_format.name.upper()} does not keep its pinned scales")
    extra_args = quantizer.backend_extra_args or {}
    block_chunk_size = extra_args.get("block_chunk_size", ggml_format.block_chunk_size)
    packed, _, searched = quantize_energy_matched(
        ggml_format, weight, second_moments, block_chunk_size=block_chunk_size
    )
    importance = None if second_moments is None else second_moments.sqrt()
    pin_packed_weight(quantizer, ggml_format.name, packed, importance, search_scales=searched)


# Blocks fitted at a time: a 32-value sub-block scores 561 splits, a 16-value one 153.
_FIT_BLOCKS_PER_CHUNK = 1024


def _derived_scales(
    blocks: torch.Tensor, columns: torch.Tensor | None, start: int, fit: ScaleFit
) -> torch.Tensor:
    """The FP16 super-block scale of each block: its largest sub-block fit over ``fit.top``."""
    x = narrow_to_float32(blocks)
    to_unit, from_unit = _unit_powers_of_two(x)
    x = x * to_unit
    # llama.cpp's element weights, which favour a block's large values over its small ones.
    weights = (x.square().mean(dim=-1, keepdim=True) + x.square()).sqrt()
    if columns is not None:
        weights = weights * _block_rows(columns, start, start + len(x)).sqrt()
    values = (x if fit.signed else x.abs()).reshape(len(x), -1, fit.group)
    scale = _fit_sub_blocks(values, weights.reshape(values.shape), fit.levels)
    scale = scale.amax(dim=-1) / fit.top * from_unit.squeeze(-1)
    return scale.clamp(max=65504.0).to(torch.float16)


def _fit_sub_blocks(
    values: torch.Tensor, weights: torch.Tensor, levels: tuple[tuple[float, float, float], ...]
) -> torch.Tensor:
    """Each sub-block's least-squares scale under its best split over its best level triple.

    With a positive scale each value's level rises with it, so the best fit splits the sorted
    values into three runs; every split is scored at once from prefix sums.
    """
    group = values.shape[-1]
    sorted_values, order = values.sort(dim=-1)
    sorted_weights = weights.gather(-1, order)
    zero = sorted_values.new_zeros((*sorted_values.shape[:-1], 1))
    cum_w = torch.cat([zero, sorted_weights.cumsum(dim=-1)], dim=-1)
    cum_wx = torch.cat([zero, (sorted_weights * sorted_values).cumsum(dim=-1)], dim=-1)
    first, second = torch.triu_indices(group + 1, group + 1, device=values.device)
    bounds = (torch.zeros_like(first), first, second, torch.full_like(first, group))
    runs = list(itertools.pairwise(bounds))
    w_runs = [cum_w[..., high] - cum_w[..., low] for low, high in runs]
    wx_runs = [cum_wx[..., high] - cum_wx[..., low] for low, high in runs]
    best_score = best_scale = None
    for triple in levels:
        sum_qx = sum(level * wx for level, wx in zip(triple, wx_runs))
        sum_qq = sum(level * level * w for level, w in zip(triple, w_runs))
        score = torch.where(
            (sum_qq > 0) & (sum_qx > 0), sum_qx.square() / sum_qq.clamp_min(1e-30), 0.0
        )
        score, best = score.max(dim=-1)
        scale = sum_qx.gather(-1, best[..., None]) / sum_qq.gather(-1, best[..., None])
        scale = torch.where(score > 0, scale.squeeze(-1), 0.0)
        if best_score is None:
            best_score, best_scale = score, scale
        else:
            better = score > best_score
            best_score = torch.where(better, score, best_score)
            best_scale = torch.where(better, scale, best_scale)
    return best_scale


def _unit_powers_of_two(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per row of ``x``, ``2**k`` bringing its largest magnitude into [0.5, 1), and ``2**-k``.

    Scaling by them keeps the squares and sums of a block near the float32 maximum finite, and
    changes no result that did not overflow: they are built from their bits, so the scaling is
    exact, and ``k`` is clamped so both stay normal.
    """
    k = (-torch.frexp(x.abs().amax(dim=-1, keepdim=True)).exponent).clamp(-126, 126)
    return ((k + 127) << 23).view(torch.float32), ((127 - k) << 23).view(torch.float32)


def _energy_factors(
    ggml_format: GGMLFormat,
    blocks: torch.Tensor,
    payload: torch.Tensor,
    columns: torch.Tensor | None,
    start: int,
    stop: int,
) -> torch.Tensor:
    """For blocks ``start:stop``, the factor on the scale that gives each its original energy."""
    x = narrow_to_float32(blocks[start:stop])
    decoded = ggml_format.dequantize(
        payload[start:stop],
        torch.tensor(x.shape),
        dtype=torch.float32,
        block_chunk_size=ggml_format.decode_chunk_size,
    ).reshape(x.shape)
    to_unit, _ = _unit_powers_of_two(x)
    x, decoded = x * to_unit, decoded * to_unit
    weights = 1.0 if columns is None else _block_rows(columns, start, start + len(x))
    target = (weights * x.square()).sum(dim=-1)
    energy = (weights * decoded.square()).sum(dim=-1)
    return torch.where(energy > 0, (target / energy.clamp_min(1e-30)).sqrt(), 1.0)


def leading_fp16_scales(blocks: torch.Tensor) -> torch.Tensor:
    """The FP16 super-block scale in the first two bytes of each ``[blocks, block_bytes]`` row."""
    return blocks[:, :2].contiguous().view(torch.float16).reshape(-1)


def with_leading_fp16_scales(blocks: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """``[blocks, block_bytes]`` rows with their leading FP16 super-block scales replaced."""
    replaced = blocks.clone()
    replaced[:, :2] = scales.to(torch.float16).contiguous().view(torch.uint8).reshape(-1, 2)
    return replaced


def super_block_scales(
    blocks: torch.Tensor,
    predict: Callable[[torch.Tensor], torch.Tensor],
    scales: torch.Tensor | None,
    chunk_size: int,
) -> torch.Tensor:
    """Each block's FP16 super-block scale: ``scales`` if given, else predicted a chunk at a time."""
    if scales is None:
        return torch.cat(
            [
                predict(blocks[start : start + chunk_size])
                for start in range(0, len(blocks), chunk_size)
            ]
        )
    if scales.dtype != torch.float16 or scales.numel() != len(blocks):
        raise ValueError(
            f"scales must hold one float16 value per block ({len(blocks)}), got "
            f"{scales.dtype} {tuple(scales.shape)}"
        )
    return scales.reshape(-1).to(blocks.device)


def dequantize_blocks(
    blocks: torch.Tensor,
    unpack_int8: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
    *,
    block_size: int,
    group_size: int,
    dtype: torch.dtype,
    block_chunk_size: int,
) -> torch.Tensor:
    """Decode ``[blocks, block_bytes]`` payload rows a chunk at a time as ``levels * steps``."""
    decoded = torch.empty((len(blocks), block_size), dtype=dtype, device=blocks.device)
    for start in range(0, len(blocks), block_chunk_size):
        stop = start + block_chunk_size
        levels, steps = unpack_int8(blocks[start:stop])
        decoded[start:stop] = (
            levels.reshape(len(levels), -1, group_size) * steps.unsqueeze(-1)
        ).reshape(-1, block_size)
    return decoded


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


# Floor on the per-column second moments, relative to their mean, so a column calibration never
# activated still has some weight in the search.
_IMPORTANCE_FLOOR = 1e-4


def floored_second_moments(second_moments: torch.Tensor | None) -> torch.Tensor | None:
    """The per-input-column ``E[x_j^2]``, floored at 1e-4 of its mean.

    ``None`` when there is no statistic or it has no positive mean: a module no calibration token
    reached (an idle MoE expert) would make every code score zero error, so it keeps the plain
    search.
    """
    if second_moments is None:
        return None
    second_moments = second_moments.float()
    mean = second_moments.mean()
    if not mean > 0:
        return None
    return second_moments.clamp_min(_IMPORTANCE_FLOOR * mean)


def importance_blocks(
    importance: torch.Tensor | None, weight: torch.Tensor, format_name: str
) -> torch.Tensor | None:
    """A per-input-column ``importance`` as ``[weight.shape[-1] // 256, 256]`` float32 block rows.

    Flattened block ``i`` of ``weight`` is weighted by row ``i % (weight.shape[-1] // 256)``.
    """
    if importance is None:
        return None
    if tuple(importance.shape) != (weight.shape[-1],) or not importance.is_floating_point():
        raise ValueError(
            f"{format_name} importance must be a floating-point [{weight.shape[-1]}] vector, got "
            f"{importance.dtype} {tuple(importance.shape)}"
        )
    importance = importance.to(device=weight.device, dtype=torch.float32)
    if not (torch.isfinite(importance) & (importance >= 0)).all():
        raise ValueError(f"{format_name} importance must be finite and non-negative")
    return importance.reshape(-1, GGML_BLOCK_SIZE)


def choice_blocks(
    choices: torch.Tensor | None,
    weight: torch.Tensor,
    groups: int,
    num_choices: int,
    format_name: str,
    every_choice: bool = False,
) -> torch.Tensor | None:
    """Per-group ``choices`` as validated ``[weight.numel() // 256, groups]`` int64 rows.

    ``choices`` is shaped ``[*weight.shape[:-1], weight.shape[-1] // 256, groups]``. A given choice
    replaces the encoder's own pick for that group; the codebook entries and signs are still
    searched under it. ``every_choice`` requests every choice's payload instead, so excludes it.
    """
    if every_choice and choices is not None:
        raise ValueError(f"{format_name} takes choices or every_choice, not both")
    if choices is None:
        return None
    expected = (*weight.shape[:-1], weight.shape[-1] // GGML_BLOCK_SIZE, groups)
    if tuple(choices.shape) != expected or choices.is_floating_point() or choices.is_complex():
        raise ValueError(
            f"{format_name} choices must be an integer {list(expected)} tensor, got "
            f"{choices.dtype} {tuple(choices.shape)}"
        )
    choices = choices.to(device=weight.device, dtype=torch.int64).reshape(-1, groups)
    if not ((choices >= 0) & (choices < num_choices)).all():
        raise ValueError(f"{format_name} choices must lie in [0, {num_choices})")
    return choices


def encode_every_choice(
    quantize: Callable[..., tuple[torch.Tensor, torch.Tensor]],
    weight: torch.Tensor,
    groups: int,
    num_choices: int,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``quantize``'s payload with each choice taken by every group, ``[num_choices, *packed]``.

    The torch encoders' path to ``every_choice``, one encode per choice; the CUDA encoders produce
    the same payloads in one pass over the codebook.
    """
    fixed = torch.empty(
        (*weight.shape[:-1], weight.shape[-1] // GGML_BLOCK_SIZE, groups),
        dtype=torch.int64,
        device=weight.device,
    )
    payloads = [quantize(weight, choices=fixed.fill_(c), **kwargs)[0] for c in range(num_choices)]
    return torch.stack(payloads), torch.tensor(weight.shape, dtype=torch.int64)


def chunk_importance(
    blocks_importance: torch.Tensor | None, start: int, stop: int
) -> torch.Tensor | None:
    """The importance of flattened blocks ``start:stop``, as ``[stop - start, 32, 8]``."""
    if blocks_importance is None:
        return None
    return _block_rows(blocks_importance, start, stop).reshape(-1, GGML_BLOCK_SIZE // 8, 8)


def _block_rows(blocks_importance: torch.Tensor, start: int, stop: int) -> torch.Tensor:
    """The ``[stop - start, 256]`` per-column rows of flattened blocks ``start:stop``."""
    index = torch.arange(start, stop, device=blocks_importance.device) % len(blocks_importance)
    return blocks_importance[index]


def iq2_tile_terms(
    magnitudes: torch.Tensor,
    grid_tile: torch.Tensor,
    tile_qnorm: torch.Tensor,
    weights: torch.Tensor | None = None,
    odd_parity: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The (weighted) ``|x| . q`` and ``|q|^2`` of every 8-value vector against a grid tile.

    ``magnitudes`` and ``weights`` are ``[blocks, vectors, 8]``, ``grid_tile`` is ``[entries, 8]``
    and ``tile_qnorm`` its unweighted norms; both results broadcast to ``[blocks, vectors,
    entries]``. With ``odd_parity`` the dot takes the even-parity sign rule: the coordinate
    contributing least is flipped.
    """
    products = magnitudes.unsqueeze(2) * grid_tile.reshape(1, 1, -1, 8)
    if weights is None:
        qnorm = tile_qnorm.reshape(1, 1, -1)
    else:
        products = products * weights.unsqueeze(2)
        qnorm = (weights.unsqueeze(2) * grid_tile.square().reshape(1, 1, -1, 8)).sum(dim=-1)
    dot = products.sum(dim=-1)
    if odd_parity is not None:
        dot = torch.where(odd_parity.unsqueeze(-1), dot - 2.0 * products.amin(dim=-1), dot)
    return dot, qnorm


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
