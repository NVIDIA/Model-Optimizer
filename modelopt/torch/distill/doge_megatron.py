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

"""Megatron-Bridge DoGE distillation helpers."""

from collections.abc import Iterable, Iterator
from copy import copy
from dataclasses import dataclass
from functools import partial

import torch
from megatron.bridge.data.loaders import cyclic_iter
from megatron.bridge.data.samplers import build_pretraining_data_loader
from megatron.bridge.data.utils import get_dataset_provider
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.state import GlobalState
from megatron.bridge.training.utils.pg_utils import get_pg_collection
from megatron.core.datasets.utils import get_blend_from_list
from megatron.core.models.gpt import GPTModel
from megatron.core.rerun_state_machine import RerunDataIterator

from modelopt.torch.distill.doge import DoGEWeightUpdater, normalize_data_path_weights

__all__ = ["DoGEDataIterators", "DoGEForwardStep"]


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


@dataclass
class DoGEDataIterators:
    """Data iterators required by one DoGE step.

    Attributes:
        source_iterators: One iterator per tunable training dataset path from ``--data_paths``.
        target_iterator: Iterator over the fixed target objective.
    """

    source_iterators: dict[str, Iterator]
    target_iterator: Iterator


class DoGEForwardStep:
    """Callable forward-step placeholder to pass into Megatron-Bridge ``pretrain``."""

    def __init__(
        self,
        data_paths: list[str],
        target_data_paths: list[str],
        meta_lr: float,
    ) -> None:
        """Initialize the callable state used by Megatron-Bridge ``pretrain``.

        Args:
            data_paths: Initial training-data blend in Megatron WEIGHT PATH format. The weights
                are normalized into ``self.blend_weights`` and updated during DoGE.
            target_data_paths: Fixed target-objective blend in Megatron WEIGHT PATH format. The
                weights are normalized into ``self.target_blend_weights`` and are not updated.
            meta_lr: Learning rate for exponentiated blend-weight updates.
        """
        self.data_paths = tuple(data_paths)
        self.target_data_paths = tuple(target_data_paths)
        self.updater = DoGEWeightUpdater(meta_lr=meta_lr)
        self.blend_weights: dict[str, float] = normalize_data_path_weights(data_paths)
        self.target_blend_weights: dict[str, float] = normalize_data_path_weights(target_data_paths)
        self.doge_data_iterators: DoGEDataIterators | None = None

    def _build_doge_data_iterators(self, state: GlobalState, model: GPTModel) -> DoGEDataIterators:
        """Build per-source and target iterators after Megatron-Bridge setup.

        The implementation should reuse Megatron-Bridge/MCore dataset construction with the
        initialized distributed state, creating one iterator for each training dataset path and one
        iterator for the target objective.
        """
        source_iterators = {
            path: _build_blend_iterator(state.cfg, model, ("1.0", path))
            for path in self.blend_weights
        }
        target_iterator = _build_blend_iterator(state.cfg, model, self.target_data_paths)
        return DoGEDataIterators(source_iterators=source_iterators, target_iterator=target_iterator)

    def _next_doge_batches(
        self,
    ) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, torch.Tensor]]:
        """Return raw Megatron GPT batches for DoGE source and target losses.

        Source batches are keyed by the same dataset paths as ``self.blend_weights``. Each batch is
        the raw dictionary returned by the Megatron data iterator, typically containing tensors such
        as ``tokens``, ``labels``, ``loss_mask``, ``attention_mask``, and ``position_ids``.
        """
        if self.doge_data_iterators is None:
            raise RuntimeError("DoGE data iterators must be built before sampling batches.")

        # Mirrors the raw ``next(data_iterator)`` call in
        # megatron.bridge.training.gpt_step.get_batch_from_iterator(). TODO: when implementing
        # DoGE losses, reuse Bridge's batch processing for CUDA transfer, pipeline-stage filtering,
        # and context-parallel partitioning instead of manually interpreting these raw batches.
        source_batches = {
            path: next(iterator)
            for path, iterator in self.doge_data_iterators.source_iterators.items()
        }
        target_batch = next(self.doge_data_iterators.target_iterator)
        return source_batches, target_batch

    def _compute_alignment_scores(
        self,
        source_batches: dict[str, dict[str, torch.Tensor]],
        target_batch: dict[str, torch.Tensor],
        model: GPTModel,
    ) -> dict[str, float]:
        """Compute source-to-target gradient-alignment scores for each training source.

        The returned scores are keyed by the same dataset paths as ``source_batches`` and
        ``self.blend_weights``. Higher scores should increase a source's DoGE blend weight.
        """
        raise NotImplementedError("DoGE gradient-alignment scoring is not implemented yet.")

    def _weighted_source_forward_step(
        self,
        state: GlobalState,
        source_batches: dict[str, dict[str, torch.Tensor]],
        model: GPTModel,
        return_schedule_plan: bool,
    ) -> tuple[torch.Tensor, partial]:
        """Return Megatron's inner-loop weighted source loss.

        This method should compute one source KD loss per batch in ``source_batches`` and combine
        them using the current ``self.blend_weights``. Megatron-Bridge then backpropagates the
        returned loss and updates the student with its normal optimizer step.
        """
        raise NotImplementedError("DoGE weighted source forward step is not implemented yet.")

    def __call__(
        self,
        state: GlobalState,
        data_iterator: Iterable,
        model: GPTModel,
        return_schedule_plan: bool = False,
    ) -> tuple[torch.Tensor, partial]:
        """Run as Megatron-Bridge ``pretrain`` forward step for one DoGE iteration.

        The DoGE outer loop updates the training blend weights from source-to-target gradient
        alignment scores. The inner loop returns the weighted source loss that Megatron-Bridge
        backpropagates with its normal optimizer step.
        """
        if self.doge_data_iterators is None:
            self.doge_data_iterators = self._build_doge_data_iterators(state, model)

        source_batches, target_batch = self._next_doge_batches()

        # Outer loop: use the target batch to score each source batch and update the data-blend
        # weights. This changes only ``self.blend_weights``, not the student model.
        scores = self._compute_alignment_scores(source_batches, target_batch, model)
        self.blend_weights = dict(self.updater.update(self.blend_weights, scores))

        # Inner loop: train the student on source batches mixed with the updated DoGE weights.
        # Megatron-Bridge backpropagates the returned loss and performs the optimizer step.
        # TODO: Reuse source gradients from the outer-loop scoring pass to avoid recomputing source
        # forward/backward work once the PoC no longer relies on Megatron-Bridge's normal loss path.
        return self._weighted_source_forward_step(
            state,
            source_batches,
            model,
            return_schedule_plan,
        )
