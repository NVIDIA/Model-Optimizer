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

"""Stable source identities and bounded fingerprints for quantized weight caches."""

import math
from typing import NamedTuple

import torch

__all__ = []

_SIGNED_INT_OF_SIZE = {2: torch.int16, 4: torch.int32, 8: torch.int64}
_ROW_KEY_CHUNK = 256


class _CacheKey(NamedTuple):
    """What a cached payload was packed from; any change means it must be packed again."""

    data_ptr: int
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: torch.dtype
    device: torch.device
    version: int


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
