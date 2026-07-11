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

from collections.abc import Iterable
from functools import partial

import torch
from megatron.bridge.training.state import GlobalState
from megatron.core.models.gpt import GPTModel

from modelopt.torch.distill.doge import DoGEWeightUpdater, normalize_data_path_weights
from modelopt.torch.distill.doge_megatron_data import (
    DoGEDataIterators,
    _build_doge_data_iterators,
    _GPTBatch,
    _next_doge_batches,
)
from modelopt.torch.distill.doge_megatron_loss import _weighted_source_forward_step

__all__ = ["DoGEForwardStep"]


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

    def _compute_alignment_scores(
        self,
        source_batches: dict[str, _GPTBatch],
        target_batch: _GPTBatch,
        model: GPTModel,
    ) -> dict[str, float]:
        """Compute source-to-target gradient-alignment scores for each training source.

        The returned scores are keyed by the same dataset paths as ``source_batches`` and
        ``self.blend_weights``. Higher scores should increase a source's DoGE blend weight.
        """
        # PoC bootstrap: keep blend weights fixed while validating DoGE data iteration and weighted
        # source-loss plumbing. Real source-target gradient alignment will replace this.
        return dict.fromkeys(source_batches, 0.0)

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
            self.doge_data_iterators = _build_doge_data_iterators(
                state.cfg,
                model,
                self.blend_weights,
                self.target_data_paths,
            )

        source_batches, target_batch = _next_doge_batches(state, model, self.doge_data_iterators)

        # Outer loop: use the target batch to score each source batch and update the data-blend
        # weights. This changes only ``self.blend_weights``, not the student model.
        scores = self._compute_alignment_scores(source_batches, target_batch, model)
        self.blend_weights = dict(self.updater.update(self.blend_weights, scores))

        # Inner loop: train the student on source batches mixed with the updated DoGE weights.
        # Megatron-Bridge backpropagates the returned loss and performs the optimizer step.
        # TODO: Reuse source gradients from the outer-loop scoring pass to avoid recomputing source
        # forward/backward work once the PoC no longer relies on Megatron-Bridge's normal loss path.
        return _weighted_source_forward_step(
            state,
            source_batches,
            model,
            self.blend_weights,
            return_schedule_plan,
        )
