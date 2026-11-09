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

"""Fixed candidates of complete native IQ sub-blocks for GSQ refinement."""

from dataclasses import dataclass

import torch

from .common import chunk_importance, importance_blocks, iq2_tile_terms, narrow_to_float32
from .iq1_s import _search_shifted_grid, iq1_s_grid
from .iq2_s import iq2_s_grid
from .iq2_xs import iq2_xs_grid
from .iq2_xxs import iq2_xxs_grid
from .registry import IQ_FORMAT_REGISTRY

__all__ = ["build_candidates"]


@dataclass
class IQCandidates:
    """Legal sub-block alternatives with a shared native scale per 256 weights."""

    coefficients: torch.Tensor
    base_scale: torch.Tensor
    valid_mask: torch.Tensor
    _format_name: str
    _entries: torch.Tensor
    _auxiliary: torch.Tensor
    _local: torch.Tensor

    def pack(self, choice: torch.Tensor, native_d: torch.Tensor) -> torch.Tensor:
        """Pack selected complete bundles and round their common scale to FP16."""
        if choice.shape != self.coefficients.shape[:3] or native_d.shape != self.base_scale.shape:
            raise ValueError("IQ candidate choices or shared scales have an invalid shape")
        index = choice.long().unsqueeze(-1).unsqueeze(-1)
        entries = self._entries.gather(-2, index.expand(*index.shape[:-1], self._entries.shape[-1]))
        auxiliary = self._auxiliary.gather(-2, index.expand_as(entries))
        local = self._local.gather(-1, choice.long().unsqueeze(-1)).squeeze(-1)
        rows, blocks = self.base_scale.shape
        payload = _pack_fields(
            self._format_name,
            entries.reshape(-1, 32),
            auxiliary.reshape(-1, 32),
            local.reshape(rows * blocks, -1),
            native_d.reshape(-1),
        )
        return payload.reshape(rows, blocks, -1)


@torch.no_grad()
def build_candidates(
    format_name: str,
    weight: torch.Tensor,
    packed: torch.Tensor,
    importance: torch.Tensor | None = None,
    max_candidates: int = 5,
) -> IQCandidates:
    """Build at most five complete native bundles at the incumbent scale.

    ``coefficients`` has shape ``[rows, blocks, bundles, candidates, bundle_width]``
    and assumes ``d=1``. Candidate zero retains the supplied codes; ``valid_mask`` marks
    unique alternatives. ``pack(choice, d)`` rounds the common scale and preserves those codes.
    """
    if format_name not in IQ_FORMAT_REGISTRY:
        raise ValueError(f"Unsupported IQ format: {format_name}")
    if isinstance(max_candidates, bool) or not isinstance(max_candidates, int):
        raise TypeError("max_candidates must be an integer")
    if not 1 <= max_candidates <= 5:
        raise ValueError("max_candidates must be between one and five")
    fmt = IQ_FORMAT_REGISTRY[format_name]
    if weight.ndim != 2 or not weight.numel() or weight.shape[-1] % 256:
        raise ValueError("IQ candidates require a matrix with complete 256-weight blocks")
    if not weight.is_floating_point():
        raise TypeError("IQ candidate weights must be floating point")
    rows, columns = weight.shape
    if packed.dtype != torch.uint8 or packed.shape != (rows, columns // 256, fmt.block_bytes):
        raise ValueError("IQ candidate payload does not match the weight")
    if packed.device != weight.device:
        raise ValueError("IQ candidate payload and weight must be on the same device")
    block_importance = importance_blocks(importance, weight, format_name)
    entries, auxiliary, local, d = _unpack_fields(format_name, packed.reshape(-1, fmt.block_bytes))
    groups = local.shape[-1]
    vectors = 32 // groups
    entries = entries.reshape(-1, groups, 1, vectors).to(torch.int16)
    auxiliary = auxiliary.reshape_as(entries).to(torch.int16)
    local = local.reshape(-1, groups, 1).to(torch.int16)
    valid = torch.ones_like(local, dtype=torch.bool)
    # Larger CUDA batches amortize the per-grid-tile kernel launches.
    block_chunk_size = 2048 if weight.is_cuda else 64
    if max_candidates > 1:
        alternatives = []
        flat_weight = weight.reshape(-1, 256)
        for start in range(0, len(d), block_chunk_size):
            stop = min(start + block_chunk_size, len(d))
            alt_entries, alt_auxiliary, alt_local, error = _alternatives(
                format_name,
                narrow_to_float32(flat_weight[start:stop]),
                d[start:stop],
                chunk_importance(block_importance, start, stop),
            )
            incumbent = slice(start, stop)
            duplicate = (
                (alt_entries == entries[incumbent]).all(-1)
                & (alt_auxiliary == auxiliary[incumbent]).all(-1)
                & (alt_local == local[incumbent])
            )
            error = error.masked_fill(duplicate | (d[incumbent] == 0)[:, None, None], torch.inf)
            order = error.argsort(dim=-1, stable=True)[..., : max_candidates - 1]
            alt_valid = torch.isfinite(error.gather(-1, order))
            index = order.unsqueeze(-1).expand(*order.shape, vectors)
            alternatives.append(
                (
                    alt_entries.gather(-2, index).to(torch.int16),
                    alt_auxiliary.gather(-2, index).to(torch.int16),
                    alt_local.gather(-1, order).to(torch.int16),
                    alt_valid,
                )
            )
        alt_entries, alt_auxiliary, alt_local, alt_valid = [
            torch.cat(parts) for parts in zip(*alternatives)
        ]
        entries = torch.cat((entries, torch.where(alt_valid[..., None], alt_entries, entries)), -2)
        auxiliary = torch.cat(
            (auxiliary, torch.where(alt_valid[..., None], alt_auxiliary, auxiliary)), -2
        )
        local = torch.cat((local, torch.where(alt_valid, alt_local, local)), -1)
        valid = torch.cat((valid, alt_valid), -1)
    coefficients = torch.empty(
        (len(d), groups, max_candidates, 256 // groups), dtype=torch.float32, device=weight.device
    )
    for start in range(0, len(d), block_chunk_size):
        stop = min(start + block_chunk_size, len(d))
        unit = _pack_fields(
            format_name,
            entries[start:stop].transpose(1, 2).reshape(-1, 32),
            auxiliary[start:stop].transpose(1, 2).reshape(-1, 32),
            local[start:stop].transpose(1, 2).reshape(-1, groups),
            torch.ones((stop - start) * max_candidates, dtype=torch.float32, device=weight.device),
        )
        levels, steps = fmt.unpack_int8(unit)
        coefficients[start:stop] = (
            (levels.reshape(-1, groups, 256 // groups) * steps.unsqueeze(-1))
            .reshape(stop - start, max_candidates, groups, 256 // groups)
            .transpose(1, 2)
        )
    coefficients = coefficients.reshape(rows, columns // 256, groups, max_candidates, 256 // groups)
    entries = entries.reshape(rows, columns // 256, groups, max_candidates, vectors)
    auxiliary = auxiliary.reshape_as(entries)
    local = local.reshape(rows, columns // 256, groups, max_candidates)
    return IQCandidates(
        coefficients,
        d.reshape(rows, -1),
        valid.reshape_as(local),
        format_name,
        entries,
        auxiliary,
        local,
    )


def _alternatives(name, weight, d, weights):
    count = weight.shape[0]
    vectors = weight.reshape(count, 32, 8)
    if name.startswith("iq1"):
        error, entries = _search_shifted_grid(vectors, d, iq1_s_grid(weight.device), weights)
        if name == "iq1_s":
            error = error.reshape(count, 8, 4, 16).sum(2)
            entries = entries.reshape(count, 8, 4, 16).transpose(-1, -2)
            choice = torch.arange(16, device=weight.device).reshape(1, 1, 16)
            auxiliary = (choice >> 3).unsqueeze(-1).expand_as(entries)
            local = (choice & 7).expand_as(error)
        else:
            error, shifts = error.reshape(count, 32, 2, 8).min(2)
            entries = entries.reshape(count, 32, 2, 8).gather(2, shifts.unsqueeze(2)).squeeze(2)
            error = error.reshape(count, 16, 2, 8).sum(2)
            entries = entries.reshape(count, 16, 2, 8).transpose(-1, -2)
            auxiliary = shifts.reshape(count, 16, 2, 8).transpose(-1, -2)
            local = torch.arange(8, device=weight.device).reshape(1, 1, 8).expand_as(error)
    else:
        grid = {"iq2_s": iq2_s_grid, "iq2_xs": iq2_xs_grid, "iq2_xxs": iq2_xxs_grid}[name](
            weight.device
        )
        negative = vectors < 0
        odd = negative.sum(-1).remainder(2).bool() if name != "iq2_s" else None
        xnorm = (vectors.square() if weights is None else weights * vectors.square()).sum(-1)
        error = torch.full((count, 32, 16), torch.inf, dtype=torch.float32, device=weight.device)
        entries = torch.zeros((count, 32, 16), dtype=torch.int64, device=weight.device)
        qnorm = grid.square().sum(-1)
        for start in range(0, len(grid), 64):
            tile = grid[start : start + 64]
            dot, norm = iq2_tile_terms(vectors.abs(), tile, qnorm[start : start + 64], weights, odd)
            for code in range(16):
                scale = d[:, None, None] * ((2 * code + 1) / 8)
                values = (xnorm[..., None] - 2 * scale * dot + scale.square() * norm).clamp_min_(0)
                best, index = values.min(-1)
                replace = best < error[..., code]
                error[..., code] = torch.where(replace, best, error[..., code])
                entries[..., code] = torch.where(replace, index + start, entries[..., code])
        signs = negative.unsqueeze(-2).expand(count, 32, 16, 8)
        if odd is not None:
            products = vectors.abs().unsqueeze(-2) * grid[entries]
            if weights is not None:
                products *= weights.unsqueeze(-2)
            flip = torch.nn.functional.one_hot(products.argmin(-1), 8).bool()
            signs = signs ^ (flip & odd[..., None, None])
        auxiliary = (signs.long() << torch.arange(8, device=weight.device)).sum(-1)
        if odd is not None:
            auxiliary &= 127
        groups = 8 if name == "iq2_xxs" else 16
        error = error.reshape(count, groups, 32 // groups, 16).sum(2)
        entries = entries.reshape(count, groups, 32 // groups, 16).transpose(-1, -2)
        auxiliary = auxiliary.reshape(count, groups, 32 // groups, 16).transpose(-1, -2)
        local = torch.arange(16, device=weight.device).reshape(1, 1, 16).expand_as(error)
    return entries, auxiliary, local, error


def _unpack_fields(name, packed):
    count = packed.shape[0]
    d = packed[:, :2].contiguous().view(torch.float16).reshape(-1).float()
    if name == "iq1_s":
        high = packed[:, 34:50:2].long() | (packed[:, 35:50:2].long() << 8)
        shifts = torch.arange(4, device=packed.device) * 3
        entries = packed[:, 2:34].long().reshape(count, 8, 4) | (
            ((high.unsqueeze(-1) >> shifts) & 7) << 8
        )
        auxiliary = (high >> 15).repeat_interleave(4, -1)
        local = (high >> 12) & 7
    elif name == "iq1_m":
        high = packed[:, 32:48].long()
        nibbles = torch.stack((high & 15, high >> 4), -1).reshape(count, 32)
        entries = packed[:, :32].long() | ((nibbles & 7) << 8)
        auxiliary = nibbles >> 3
        words = packed[:, 48:56:2].long() | (packed[:, 49:56:2].long() << 8)
        bits = ((words >> 12) << (torch.arange(4, device=packed.device) * 4)).sum(-1)
        d = bits.to(torch.int16).view(torch.float16).float()
        local = ((words.unsqueeze(-1) >> (torch.arange(4, device=packed.device) * 3)) & 7).reshape(
            count, 16
        )
    elif name == "iq2_xxs":
        body = packed[:, 2:].long().reshape(count, 8, 8)
        entries = body[..., :4].reshape(count, 32)
        bits = (body[..., 4:] << (torch.arange(4, device=packed.device) * 8)).sum(-1)
        auxiliary = (
            (bits.unsqueeze(-1) >> (torch.arange(4, device=packed.device) * 7)) & 127
        ).reshape(count, 32)
        local = bits >> 28
    elif name == "iq2_xs":
        codes = packed[:, 2:66:2].long() | (packed[:, 3:66:2].long() << 8)
        entries, auxiliary = codes & 511, codes >> 9
        local = torch.stack((packed[:, 66:].long() & 15, packed[:, 66:].long() >> 4), -1).reshape(
            count, 16
        )
    else:
        high = (
            (packed[:, 66:74].long().unsqueeze(-1) >> (torch.arange(4, device=packed.device) * 2))
            & 3
        ).reshape(count, 32)
        entries = packed[:, 2:34].long() | (high << 8)
        auxiliary = packed[:, 34:66].long()
        local = torch.stack((packed[:, 74:].long() & 15, packed[:, 74:].long() >> 4), -1).reshape(
            count, 16
        )
    return entries.reshape(count, 32), auxiliary, local, d


def _pack_fields(name, entries, auxiliary, local, d):
    fmt = IQ_FORMAT_REGISTRY[name]
    entries, auxiliary, local = entries.long(), auxiliary.long(), local.long()
    count = entries.shape[0]
    packed = torch.zeros((count, fmt.block_bytes), dtype=torch.uint8, device=entries.device)
    native_d = d.clamp(-65504, 65504).to(torch.float16)
    packed[:, :2] = native_d.contiguous().view(torch.uint8).reshape(count, 2)
    if name == "iq1_s":
        packed[:, 2:34] = (entries & 255).to(torch.uint8)
        shifts = torch.arange(4, device=entries.device) * 3
        high = (((entries.reshape(count, 8, 4) >> 8) & 7) << shifts).sum(-1)
        high |= (local << 12) | (auxiliary[:, ::4] << 15)
        packed[:, 34:50:2] = (high & 255).to(torch.uint8)
        packed[:, 35:50:2] = (high >> 8).to(torch.uint8)
    elif name == "iq1_m":
        packed[:, :32] = (entries & 255).to(torch.uint8)
        nibbles = (entries >> 8) | (auxiliary << 3)
        packed[:, 32:48] = (nibbles[:, ::2] | (nibbles[:, 1::2] << 4)).to(torch.uint8)
        words = (local.reshape(count, 4, 4) << (torch.arange(4, device=entries.device) * 3)).sum(-1)
        d_bits = native_d.contiguous().view(torch.int16).long() & 65535
        words |= ((d_bits.unsqueeze(-1) >> (torch.arange(4, device=entries.device) * 4)) & 15) << 12
        packed[:, 48:56:2] = (words & 255).to(torch.uint8)
        packed[:, 49:56:2] = (words >> 8).to(torch.uint8)
    elif name == "iq2_xxs":
        body = packed[:, 2:].reshape(count, 8, 8)
        body[..., :4] = entries.reshape(count, 8, 4).to(torch.uint8)
        bits = (auxiliary.reshape(count, 8, 4) << (torch.arange(4, device=entries.device) * 7)).sum(
            -1
        )
        bits |= local << 28
        body[..., 4:] = (
            (bits.unsqueeze(-1) >> (torch.arange(4, device=entries.device) * 8)) & 255
        ).to(torch.uint8)
    elif name == "iq2_xs":
        codes = entries | (auxiliary << 9)
        packed[:, 2:66:2] = (codes & 255).to(torch.uint8)
        packed[:, 3:66:2] = (codes >> 8).to(torch.uint8)
        packed[:, 66:] = (local[:, ::2] | (local[:, 1::2] << 4)).to(torch.uint8)
    else:
        packed[:, 2:34] = (entries & 255).to(torch.uint8)
        packed[:, 34:66] = auxiliary.to(torch.uint8)
        high = (
            (entries.reshape(count, 8, 4) >> 8) << (torch.arange(4, device=entries.device) * 2)
        ).sum(-1)
        packed[:, 66:74] = high.to(torch.uint8)
        packed[:, 74:] = (local[:, ::2] | (local[:, 1::2] << 4)).to(torch.uint8)
    return torch.where((native_d == 0).unsqueeze(-1), 0, packed)
