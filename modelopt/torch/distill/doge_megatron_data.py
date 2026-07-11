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

"""Megatron-Bridge data helpers for DoGE distillation."""

from collections.abc import Iterator
from copy import copy
from dataclasses import dataclass

import torch
from megatron.bridge.data.loaders import cyclic_iter
from megatron.bridge.data.samplers import build_pretraining_data_loader
from megatron.bridge.data.utils import get_dataset_provider
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.gpt_step import get_batch
from megatron.bridge.training.state import GlobalState
from megatron.bridge.training.utils.pg_utils import get_pg_collection
from megatron.core.datasets.utils import get_blend_from_list
from megatron.core.models.gpt import GPTModel
from megatron.core.rerun_state_machine import RerunDataIterator
from megatron.core.utils import get_model_config

__all__ = [
    "DoGEDataIterators",
    "_GPTBatch",
    "_build_doge_data_iterators",
    "_next_doge_batches",
]

_GPTBatch = tuple[
    torch.Tensor,  # tokens
    torch.Tensor,  # labels
    torch.Tensor,  # loss_mask
    torch.Tensor | None,  # attention_mask
    torch.Tensor,  # position_ids
    dict[str, object] | None,  # packed_sequence_metadata
]


@dataclass
class DoGEDataIterators:
    """Data iterators required by one DoGE step.

    Attributes:
        source_iterators: One iterator per tunable training dataset path from ``--data_paths``.
        target_iterator: Iterator over the fixed target objective.
    """

    source_iterators: dict[str, Iterator]
    target_iterator: Iterator


def _build_blend_iterator(
    config: ConfigContainer, model: GPTModel, data_paths: tuple[str, ...]
) -> RerunDataIterator:
    """Build one cyclic iterator from Megatron WEIGHT PATH pairs."""
    if config.train.global_batch_size != config.train.micro_batch_size:
        raise NotImplementedError("DoGE data iterators currently require --gbs == --mbs")

    # Derive from the Bridge-initialized dataset config because
    # megatron.bridge.training.setup.setup() populates runtime fields such as the tokenizer.
    # DoGE only changes the blend and split.
    dataset_config = copy(config.dataset)
    if getattr(dataset_config, "mock", False):
        # MockGPTDatasetConfig intentionally ignores source paths. Each DoGE mock iterator is an
        # independent mock stream with the same settings; this validates iterator plumbing, not
        # source-specific data selection.
        pass
    else:
        dataset_config.blend = get_blend_from_list(list(data_paths))
        dataset_config.blend_per_split = None
        dataset_config.data_path = None
    dataset_config.split = "100,0,0"
    dataset_config.finalize()

    num_samples = config.train.train_iters * config.train.global_batch_size
    # Mirrors the data path used by megatron.bridge.training.pretrain.pretrain(): resolve the
    # dataset provider with get_dataset_provider(), then build the dataloader after setup() has
    # initialized distributed state, tokenizer, model, and optimizer.
    # TODO: Validate parity with megatron.bridge.data.loaders.setup_data_iterators().
    dataset_provider = get_dataset_provider(dataset_config)
    train_dataset, _, _ = dataset_provider(
        [num_samples, 0, 0],
        dataset_config,
    )
    pg_collection = get_pg_collection(model)
    data_loader = build_pretraining_data_loader(
        train_dataset,
        consumed_samples=0,
        dataloader_type="cyclic",
        micro_batch_size=config.train.micro_batch_size,
        num_workers=dataset_config.num_workers,
        data_sharding=dataset_config.data_sharding,
        collate_fn=train_dataset.collate_fn if hasattr(train_dataset, "collate_fn") else None,
        pin_memory=dataset_config.pin_memory,
        persistent_workers=dataset_config.persistent_workers,
        data_parallel_rank=torch.distributed.get_rank(group=pg_collection.dp),
        data_parallel_size=torch.distributed.get_world_size(group=pg_collection.dp),
        global_batch_size=config.train.global_batch_size,
    )
    return RerunDataIterator(iter(cyclic_iter(data_loader)))


def _build_doge_data_iterators(
    config: ConfigContainer,
    model: GPTModel,
    blend_weights: dict[str, float],
    target_data_paths: tuple[str, ...],
) -> DoGEDataIterators:
    """Build per-source and target iterators after Megatron-Bridge setup.

    The implementation reuses Megatron-Bridge/MCore dataset construction with the initialized
    distributed state, creating one iterator for each training dataset path and one iterator for
    the target objective.
    """
    if config.model.pipeline_model_parallel_size != 1:
        raise NotImplementedError("DoGE distillation PoC currently supports only --pp_size 1.")

    source_iterators = {
        path: _build_blend_iterator(config, model, ("1.0", path)) for path in blend_weights
    }
    target_iterator = _build_blend_iterator(config, model, target_data_paths)
    return DoGEDataIterators(source_iterators=source_iterators, target_iterator=target_iterator)


def _get_doge_batch(state: GlobalState, model: GPTModel, data_iterator: Iterator) -> _GPTBatch:
    """Return the next Megatron-Bridge GPT batch from a DoGE iterator."""
    # Mirrors megatron.bridge.training.gpt_step._forward_step_common(), which calls get_batch()
    # before the model forward pass.
    model_config = get_model_config(model)
    use_mtp = (getattr(model_config, "mtp_num_layers", None) or 0) > 0
    batch = get_batch(
        data_iterator,
        state.cfg,
        use_mtp,
        pg_collection=get_pg_collection(model),
    )
    # TODO: Drop this compatibility shim once the minimum Megatron-Bridge version is fixed.
    # NeMo 26.02/26.06 containers return packed-sequence fields separately, while newer Bridge main
    # returns a single packed_sequence_metadata object.
    if len(batch) == 6:
        return batch

    (
        tokens,
        labels,
        loss_mask,
        attention_mask,
        position_ids,
        cu_seqlens,
        cu_seqlens_argmin,
        max_seqlen,
        cu_seqlens_unpadded,
        cu_seqlens_unpadded_argmin,
    ) = batch
    packed_sequence_metadata = None
    if cu_seqlens is not None:
        packed_sequence_metadata = {
            "cu_seqlens": cu_seqlens,
            "cu_seqlens_argmin": cu_seqlens_argmin,
            "max_seqlen": max_seqlen,
            "cu_seqlens_unpadded": cu_seqlens_unpadded,
            "cu_seqlens_unpadded_argmin": cu_seqlens_unpadded_argmin,
        }
    return tokens, labels, loss_mask, attention_mask, position_ids, packed_sequence_metadata


def _next_doge_batches(
    state: GlobalState, model: GPTModel, data_iterators: DoGEDataIterators
) -> tuple[dict[str, _GPTBatch], _GPTBatch]:
    """Return Megatron GPT batches for DoGE source and target losses.

    Source batches are keyed by the same dataset paths as ``data_iterators.source_iterators``.
    Batch construction uses the same ``get_batch`` path as Megatron-Bridge GPT training, including
    CUDA transfer, pipeline-stage filtering, and context-parallel partitioning.
    """
    source_batches = {
        path: _get_doge_batch(state, model, iterator)
        for path, iterator in data_iterators.source_iterators.items()
    }
    target_batch = _get_doge_batch(state, model, data_iterators.target_iterator)
    return source_batches, target_batch
