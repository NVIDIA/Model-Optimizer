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

"""Decoded GGML weights transcoded to NVFP4, as a W1A4 or W2A4 GEMM consumes them.

The block-scaled FP4 MMA takes E2M1 values with E4M3 scales on both operands, so with NVFP4
activations the IQ weights are NVFP4 inside it as well. Each 16-weight block takes the E4M3 scale
minimising its squared error under nearest-E2M1 rounding, searched over the codes -4..+10 around
``amax / 6``; a search over every code picked the same scale on each block tested. IQ1's levels
(+-1, 7, 9 eighths of a step) and IQ2's (8 : 25 : 43) are not E2M1 ratios, so ``amax / 6`` alone
adds about 10% error to IQ1, where the search adds 4%.

The per-tensor scale is a power of two. E4M3 values repeat in every octave, so short of E4M3's
range ends a block's best scale is the same whichever tensor set that power of two: GPTQ can
transcode one column group at a time, and a converter can share one per-tensor scale across fused
projections, and both still reproduce the transcode of each whole weight.
"""

import torch

from ..qtensor.nvfp4_tensor import e2m1_bounds, e2m1_values

__all__ = ["nvfp4_global_scale", "nvfp4_transcode", "nvfp4_transcode_scales"]

NVFP4_BLOCK_SIZE = 16
# Scale codes searched around the one mapping a block's amax to 6.
_SEARCH_OFFSETS = range(-4, 11)
# Blocks searched at a time: each takes len(_SEARCH_OFFSETS) * 16 float32 temporaries.
_SEARCH_CHUNK = 1 << 16
# Blocks fake-quantized at a time, bounding that pass's temporaries to 64 MiB each.
_TRANSCODE_CHUNK = 1 << 20
_TABLES: dict[torch.device, tuple[torch.Tensor, ...]] = {}


def _tables(device: torch.device) -> tuple[torch.Tensor, ...]:
    if device not in _TABLES:
        # Every positive finite E4M3 value, 2^-9 .. 448, ascending.
        e4m3 = torch.arange(1, 127, dtype=torch.uint8).view(torch.float8_e4m3fn).float()
        _TABLES[device] = (
            e4m3.to(device),
            e2m1_bounds.to(device),
            e2m1_bounds[[1, 3, 5]].to(device),
            e2m1_values[:8].to(device),
        )
    return _TABLES[device]


def _round_e2m1(x: torch.Tensor) -> torch.Tensor:
    """``x`` rounded to E2M1, ties to even and saturating at 6, as ``NVFP4QTensor`` packs it."""
    _, bounds, odd_bounds, values = _tables(x.device)
    magnitude = x.abs()
    index = torch.searchsorted(bounds, magnitude, out_int32=True) + torch.isin(
        magnitude, odd_bounds
    )
    return values[index].copysign(x)


def nvfp4_global_scale(weight: torch.Tensor) -> torch.Tensor:
    """The power of two at least ``amax / (3 * 448)``, as a float32 scalar.

    It puts the largest block's ``amax / 6`` in (112, 224], an octave below E4M3's 448, so the
    search can still raise that block's scale.
    """
    scale = torch.exp2(torch.ceil(torch.log2(weight.abs().amax().float() / (3 * 448))))
    return torch.where(scale > 0, scale, torch.ones_like(scale))


@torch.no_grad()
def nvfp4_transcode_scales(
    weight: torch.Tensor, global_scale: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """The E4M3 block scales and float32 per-tensor scale of ``weight``'s NVFP4 transcode.

    ``weight`` is a decoded GGML weight whose last dimension is a multiple of 16. The block scales
    are shaped ``[*weight.shape[:-1], weight.shape[-1] // 16]``. ``global_scale`` defaults to
    :func:`nvfp4_global_scale`; pass a shared one for projections a deployment fuses.
    """
    if weight.shape[-1] % NVFP4_BLOCK_SIZE:
        raise ValueError(f"NVFP4 needs a last dimension divisible by 16, got {tuple(weight.shape)}")
    if global_scale is None:
        global_scale = nvfp4_global_scale(weight)
    e4m3, *_ = _tables(weight.device)
    offsets = torch.tensor(_SEARCH_OFFSETS, device=weight.device)
    blocks = weight.reshape(-1, NVFP4_BLOCK_SIZE)
    codes = torch.empty(len(blocks), dtype=torch.int64, device=weight.device)
    for start in range(0, len(blocks), _SEARCH_CHUNK):
        x = blocks[start : start + _SEARCH_CHUNK].float()
        first = torch.searchsorted(e4m3, x.abs().amax(-1) / (6 * global_scale))
        candidates = (first.unsqueeze(-1) + offsets).clamp_(0, len(e4m3) - 1)
        scales = (e4m3[candidates] * global_scale).unsqueeze(-1)
        error = (x.unsqueeze(1) - _round_e2m1(x.unsqueeze(1) / scales) * scales).square().sum(-1)
        # argmin keeps the first minimum, the smallest scale among equally good ones.
        codes[start : start + len(x)] = candidates.gather(-1, error.argmin(-1, keepdim=True))[:, 0]
    scales = e4m3[codes].to(torch.float8_e4m3fn)
    return scales.reshape(*weight.shape[:-1], -1), global_scale


@torch.no_grad()
def nvfp4_transcode(
    weight: torch.Tensor,
    block_scales: torch.Tensor | None = None,
    global_scale: torch.Tensor | None = None,
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """``weight`` fake-quantized to NVFP4 with its transcode scales, as ``dtype``.

    ``block_scales`` and ``global_scale`` are what :func:`nvfp4_transcode_scales` returns for
    ``weight``, searched when not given. Every result is an E2M1 value times an E4M3 value times a
    power of two, so a BF16 or FP16 ``dtype`` holds it exactly.
    """
    if block_scales is None:
        block_scales, global_scale = nvfp4_transcode_scales(weight, global_scale)
    blocks = weight.reshape(-1, NVFP4_BLOCK_SIZE)
    block_scales = block_scales.reshape(-1, 1)
    transcoded = torch.empty(blocks.shape, dtype=dtype, device=weight.device)
    for start in range(0, len(blocks), _TRANSCODE_CHUNK):
        chunk = slice(start, start + _TRANSCODE_CHUNK)
        scales = block_scales[chunk].float() * global_scale
        transcoded[chunk] = _round_e2m1(blocks[chunk].float() / scales) * scales
    return transcoded.reshape(weight.shape)
