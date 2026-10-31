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

"""local_hessian for GGML formats: each group's local scale picked by its inputs' local Hessian."""

import torch

from ..utils.calib_utils import LocalHessianEncoder, register_local_hessian_encoder
from .common import (
    GGML_BLOCK_SIZE,
    GGMLFormat,
    floored_second_moments,
    pin_packed_weight,
    pinned_packed_weight,
    validate_weight,
)
from .registry import GGML_FORMAT_REGISTRY

__all__ = ["GGMLLocalHessianEncoder", "search_choices"]

# Bounds the candidates held at once, num_choices x rows x in_features float32 values (1 GiB).
_CANDIDATE_ELEMENTS = 1 << 28
# Passes of the group-by-group refinement; a third rarely changes a choice.
_REFINE_SWEEPS = 2


def search_choices(
    ggml_format: GGMLFormat,
    weight: torch.Tensor,
    hessian: torch.Tensor,
    importance: torch.Tensor | None = None,
    block_chunk_size: int | None = None,
    decode_chunk_size: int | None = None,
    perc_damp: float = 0.0,
) -> torch.Tensor:
    """Each group's choice minimizing the layer-output error ``dw H dw^T``, ``[*rows, blocks, groups]``.

    Every choice is encoded for every group, its codebook entries searched as the encoder does
    (weighted by ``importance``). Each group first takes the choice with the least error under its
    own diagonal block of ``hessian``, ``[in_features // 256, 256, 256]``; groups are then revisited
    in turn, each taking the choice that most lowers its block's error with the others held.
    ``perc_damp`` adds that fraction of the mean diagonal to ``H``'s diagonal first, as GPTQ damps.
    """
    rows = weight.reshape(-1, weight.shape[-1])
    blocks, groups = weight.shape[-1] // GGML_BLOCK_SIZE, ggml_format.choice_groups
    group_size = GGML_BLOCK_SIZE // groups
    hessian = hessian.to(device=weight.device, dtype=torch.float32)
    if perc_damp:
        ridge = perc_damp * hessian.diagonal(dim1=1, dim2=2).mean()
        hessian = hessian + ridge * torch.eye(GGML_BLOCK_SIZE, device=weight.device)
    group_rows = hessian.reshape(blocks, groups, group_size, GGML_BLOCK_SIZE)
    group_hessian = (
        group_rows.reshape(blocks, groups, group_size, groups, group_size)
        .diagonal(dim1=1, dim2=3)
        .permute(0, 3, 1, 2)
    )  # [blocks, groups, group_size, group_size]
    chunk_args = {} if block_chunk_size is None else {"block_chunk_size": block_chunk_size}
    decode_args = {} if decode_chunk_size is None else {"block_chunk_size": decode_chunk_size}

    picks = []
    step = max(1, _CANDIDATE_ELEMENTS // (ggml_format.num_choices * weight.shape[-1]))
    for start in range(0, rows.shape[0], step):
        chunk = rows[start : start + step]
        count = chunk.shape[0]
        reference = chunk.float().reshape(count, blocks, groups, group_size)
        errors = torch.empty(
            (ggml_format.num_choices, *reference.shape), dtype=torch.float32, device=weight.device
        )
        for choice in range(ggml_format.num_choices):
            fixed = torch.full((count, blocks, groups), choice, device=weight.device)
            packed, shape = ggml_format.quantize(
                chunk, importance=importance, choices=fixed, **chunk_args
            )
            decoded = ggml_format.dequantize(packed, shape, dtype=torch.float32, **decode_args)
            torch.sub(decoded.reshape(reference.shape), reference, out=errors[choice])

        pick = torch.einsum("crngi,ngij,crngj->crng", errors, group_hessian, errors).argmin(0)
        current = errors.gather(0, pick[None, ..., None].expand(1, *pick.shape, group_size))[0]
        # gradient = current @ H per block; switching group g by delta changes the block's error
        # by 2 delta . gradient_g + delta H_gg delta.
        gradient = torch.einsum("rnk,nkl->rnl", current.reshape(count, blocks, -1), hessian)
        for _ in range(_REFINE_SWEEPS):
            for group in range(groups):
                columns = slice(group * group_size, (group + 1) * group_size)
                delta = errors[:, :, :, group] - current[None, :, :, group]
                change = 2 * (delta * gradient[None, :, :, columns]).sum(-1) + torch.einsum(
                    "crni,nij,crnj->crn", delta, group_hessian[:, group], delta
                )
                pick[:, :, group] = change.argmin(0)
                chosen = errors[:, :, :, group].gather(
                    0, pick[None, :, :, group, None].expand(1, count, blocks, group_size)
                )[0]
                gradient += torch.einsum(
                    "rni,nik->rnk", chosen - current[:, :, group], group_rows[:, group]
                )
                current[:, :, group] = chosen
        picks.append(pick)
    return torch.cat(picks).reshape(*weight.shape[:-1], blocks, groups)


class GGMLLocalHessianEncoder(LocalHessianEncoder):
    """Encodes a ``ggml``-backend weight with Hessian-picked local scales and pins its codes.

    The codebook search keeps imatrix's weighting, the square root of the Hessian diagonal; the
    per-group choice that search would take by that diagonal is instead picked by the full
    256-column block of the Hessian (:func:`search_choices`). A weight no calibration input
    reached is encoded plainly. Like imatrix, the pinned payload marks the weight as final, so a
    quantizer that already holds one, such as GPTQ's, is left alone.
    """

    block_size = GGML_BLOCK_SIZE

    def accepts(self, quantizer) -> bool:
        """Enabled formats with per-group choices that do not hold final codes yet."""
        num_bits = getattr(quantizer, "num_bits", None)
        if not isinstance(num_bits, str) or not quantizer.is_enabled:
            return False
        ggml_format = GGML_FORMAT_REGISTRY.get(num_bits)
        return (
            ggml_format is not None
            and ggml_format.num_choices > 0
            and pinned_packed_weight(quantizer, num_bits) is None
        )

    def validate(self, weight, quantizer):
        """Refuse a weight the format cannot encode, such as one whose width is not a block multiple."""
        ggml_format = GGML_FORMAT_REGISTRY[quantizer.num_bits]
        validate_weight(weight, ggml_format.name.upper(), block_size=ggml_format.block_size)

    def encode(self, weight, quantizer, hessian, perc_damp=0.0):
        """Encode ``weight`` with the choices :func:`search_choices` picks, and pin the payload."""
        ggml_format = GGML_FORMAT_REGISTRY[quantizer.num_bits]
        extra_args = quantizer.backend_extra_args or {}
        block_chunk_size = extra_args.get("block_chunk_size", ggml_format.block_chunk_size)
        decode_chunk_size = extra_args.get("decode_chunk_size", ggml_format.decode_chunk_size)
        kwargs = {}
        if hessian is not None:
            moments = floored_second_moments(hessian.diagonal(dim1=1, dim2=2).reshape(-1))
            if moments is not None:
                if ggml_format.weighted:
                    kwargs["importance"] = moments.sqrt().to(weight.device)
                kwargs["choices"] = search_choices(
                    ggml_format,
                    weight,
                    hessian,
                    kwargs.get("importance"),
                    block_chunk_size,
                    decode_chunk_size,
                    perc_damp,
                )
        packed, shape = ggml_format.quantize(
            weight.contiguous(), block_chunk_size=block_chunk_size, **kwargs
        )
        pin_packed_weight(quantizer, quantizer.num_bits, packed)
        decoded = ggml_format.dequantize(
            packed, shape, dtype=torch.float32, block_chunk_size=decode_chunk_size
        )
        return decoded.to(weight.dtype)


register_local_hessian_encoder("ggml", GGMLLocalHessianEncoder())
