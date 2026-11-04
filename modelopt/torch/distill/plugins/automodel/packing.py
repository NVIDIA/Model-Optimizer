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

"""One native packing/CP contract for repair and held-out diagnostics."""

from contextlib import nullcontext
from functools import partial

import torch

from ...local_units import _valid_token_mask
from .families import native_family

__all__ = []


def prepare_packed_batch(model, batch):
    """Resolve compact document IDs through AutoModel's native packing contract."""
    family = native_family(model)
    if family is not None:
        batch = family.packed_batch(batch)
    documents = batch.get("_packed_seq_ids")
    mask = batch.get("attention_mask")
    if (
        not isinstance(documents, torch.Tensor)
        or documents.ndim != 2
        or (isinstance(mask, torch.Tensor) and mask.ndim != 2)
        or batch.get("qkv_format") == "thd"
    ):
        return batch
    # Legacy native owners (including GLM Flash) consume compact IDs directly.
    if getattr(model, "_owns_packed_attention", False) and not getattr(
        model, "requires_packed_sequence_metadata", False
    ):
        return batch
    from nemo_automodel.components.datasets.utils import neat_packed_collater
    from nemo_automodel.components.models.common.packing import (
        get_model_attn_implementation,
        get_packing_capabilities,
    )

    capabilities = get_packing_capabilities(get_model_attn_implementation(model), model=model)
    if (
        capabilities.packed_mask_type == "block_causal"
        and not bool((documents > 1).any())
        and not capabilities.requires_packed_sequence_metadata
    ):
        return {**batch, "attention_mask": documents.ne(0)}
    positions = batch.get("position_ids")
    if not isinstance(positions, torch.Tensor) or positions.shape != documents.shape:
        raise ValueError(
            "Compact packed input requires document-reset position_ids or native collated inputs."
        )
    tokens = batch["input_ids"]
    labels = batch.get("labels", torch.full_like(tokens, -100))
    rows = [
        {
            "input_ids": tokens[index],
            "labels": labels[index],
            "position_ids": positions[index],
            "attention_mask": documents[index],
        }
        for index in range(tokens.shape[0])
    ]
    packed = neat_packed_collater(rows, packing=capabilities)
    result = {**batch, **packed}
    if capabilities.packed_mask_type == "flash_varlen":
        result.pop("attention_mask", None)
    return family.packed_batch(result) if family is not None else result


def validate_packed_cp(sharder, batch, valid):
    """Reject a native CP strategy that would discard packed document boundaries."""
    cp_mesh = sharder._cp_mesh
    if cp_mesh is None or cp_mesh.size() <= 1:
        return
    from nemo_automodel.components.distributed.context_parallel.sharder import (
        shard_batch_load_balanced,
    )

    strategy = sharder.shard_batch
    while isinstance(strategy, partial):
        strategy = strategy.func
    if strategy is shard_batch_load_balanced:
        documents = batch.get("_packed_seq_ids", batch.get("attention_mask"))
        positions = batch.get("position_ids")
        adjacent = valid[:, 1:] & valid[:, :-1]
        packed = (
            isinstance(documents, torch.Tensor)
            and documents.shape == valid.shape
            and bool(((documents[:, 1:] != documents[:, :-1]) & adjacent).any())
        )
        reset = (
            isinstance(positions, torch.Tensor)
            and positions.shape == valid.shape
            and bool(((positions[:, 1:] <= positions[:, :-1]) & adjacent).any())
        )
        if packed or reset:
            raise ValueError(
                "Generic SDPA CP drops document masks; use a model-owned packed CP "
                "sharder or a supported packed attention backend."
            )


def prepare_local_batch(model, batch, mesh_context=None, *, padding_token_id=0):
    """Pack and shard tokens, labels and validity through the same native layout."""
    batch = dict(prepare_packed_batch(model, batch))
    valid = _valid_token_mask(batch)
    if valid is None:
        valid = torch.ones_like(batch["input_ids"], dtype=torch.bool)
    if mesh_context is None:
        return nullcontext, {**batch, "valid_mask": valid}
    from nemo_automodel.components.distributed.context_parallel import ContextParallelSharder

    family = native_family(model)
    batch["valid_mask"] = valid
    if family is not None and mesh_context.cp_size > 1:
        batch = family.packed_cp_batch(batch, cp_size=mesh_context.cp_size)
    valid = batch.pop("valid_mask", valid)
    sharder = ContextParallelSharder(
        model,
        mesh_context.device_mesh,
        batch,
        padding_token_id=padding_token_id,
        loss_mask=family.cp_loss_mask(valid) if family is not None else valid,
    )
    validate_packed_cp(sharder, batch, valid)
    context, local = sharder.shard(batch)
    local["valid_mask"] = sharder.shard_token_tensor(valid, seq_dim=1, fill=False)
    return context, local
