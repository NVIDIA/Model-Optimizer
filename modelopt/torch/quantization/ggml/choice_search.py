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

"""The local-Hessian choice search: each group's choice picked by the layer-output error."""

import torch

from .common import GGML_BLOCK_SIZE, GGMLFormat

__all__ = ["search_choices"]

# Bounds the candidate errors held at once, num_choices x rows x 256 float32 values (1 GiB).
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
) -> torch.Tensor:
    """Each group's choice minimizing the layer-output error ``dw H dw^T``, ``[*rows, blocks, groups]``.

    ``hessian`` holds one ``[256, 256]`` Hessian per block of columns, ``[in_features // 256, 256,
    256]``; the result is the ``choices`` the format's ``quantize`` takes. Every choice is encoded
    for every group, its codebook entries searched as the encoder does (weighted by ``importance``).
    In each block, each group first takes the choice with the least error under its own diagonal
    block of the Hessian; groups are then revisited in turn, each taking the choice that most lowers
    the block's error with the others held.
    """
    rows = weight.reshape(-1, weight.shape[-1])
    blocks, groups = weight.shape[-1] // GGML_BLOCK_SIZE, ggml_format.choice_groups
    hessian = hessian.to(device=weight.device, dtype=torch.float32)
    chunk_args = {} if block_chunk_size is None else {"block_chunk_size": block_chunk_size}
    decode_args = {} if decode_chunk_size is None else {"block_chunk_size": decode_chunk_size}
    step = max(1, _CANDIDATE_ELEMENTS // (ggml_format.num_choices * GGML_BLOCK_SIZE))
    picks = torch.empty((rows.shape[0], blocks, groups), dtype=torch.int64, device=weight.device)
    for block in range(blocks):
        columns = slice(block * GGML_BLOCK_SIZE, (block + 1) * GGML_BLOCK_SIZE)
        block_importance = None if importance is None else importance[columns]
        for start in range(0, rows.shape[0], step):
            picks[start : start + step, block] = _search_block(
                ggml_format,
                rows[start : start + step, columns],
                hessian[block],
                block_importance,
                chunk_args,
                decode_args,
            )
    return picks.reshape(*weight.shape[:-1], blocks, groups)


def _search_block(ggml_format, block, hessian, importance, chunk_args, decode_args):
    """:func:`search_choices` on one block of columns, ``[rows, 256]``: picks ``[rows, groups]``."""
    rows, groups = block.shape[0], ggml_format.choice_groups
    group_size = GGML_BLOCK_SIZE // groups
    reference = block.float()
    errors = torch.empty(
        (ggml_format.num_choices, rows, GGML_BLOCK_SIZE), dtype=torch.float32, device=block.device
    )
    for choice in range(ggml_format.num_choices):
        fixed = torch.full((rows, 1, groups), choice, device=block.device)
        packed, shape = ggml_format.quantize(
            block.contiguous(), importance=importance, choices=fixed, **chunk_args
        )
        decoded = ggml_format.dequantize(packed, shape, dtype=torch.float32, **decode_args)
        torch.sub(decoded, reference, out=errors[choice])
    errors = errors.reshape(ggml_format.num_choices, rows, groups, group_size)

    group_rows = hessian.reshape(groups, group_size, GGML_BLOCK_SIZE)
    group_hessian = (
        hessian.reshape(groups, group_size, groups, group_size)
        .diagonal(dim1=0, dim2=2)
        .permute(2, 0, 1)
    )  # [groups, group_size, group_size]
    pick = torch.einsum("crgi,gij,crgj->crg", errors, group_hessian, errors).argmin(0)
    current = errors.gather(0, pick[None, ..., None].expand(1, *pick.shape, group_size))[0]
    # gradient = current @ H; switching group g by delta changes the error by
    # 2 delta . gradient_g + delta H_gg delta.
    gradient = current.reshape(rows, GGML_BLOCK_SIZE) @ hessian
    for _ in range(_REFINE_SWEEPS):
        for group in range(groups):
            columns = slice(group * group_size, (group + 1) * group_size)
            delta = errors[:, :, group] - current[None, :, group]
            change = 2 * (delta * gradient[None, :, columns]).sum(-1) + torch.einsum(
                "cri,ij,crj->cr", delta, group_hessian[group], delta
            )
            pick[:, group] = change.argmin(0)
            chosen = errors[:, :, group].gather(
                0, pick[None, :, group, None].expand(1, rows, group_size)
            )[0]
            gradient += (chosen - current[:, group]) @ group_rows[group]
            current[:, group] = chosen
    return pick
