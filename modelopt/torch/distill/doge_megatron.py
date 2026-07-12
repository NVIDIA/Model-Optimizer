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
import torch.distributed as dist
from megatron.bridge.training.state import GlobalState
from megatron.core.models.gpt import GPTModel

from modelopt.torch.distill.doge import DoGEWeightUpdater, normalize_data_path_weights
from modelopt.torch.distill.doge_megatron_data import (
    DoGEDataIterators,
    _build_doge_data_iterators,
    _GPTBatch,
    _next_doge_batches,
)
from modelopt.torch.distill.doge_megatron_loss import (
    calc_alignment_gradient_vector,
    weighted_source_forward_step,
)

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
        state: GlobalState,
        source_batches: dict[str, _GPTBatch],
        target_batch: _GPTBatch,
        model: GPTModel,
    ) -> dict[str, float]:
        """Compute source-to-target gradient-alignment scores for each training source.

        The returned scores are keyed by the same dataset paths as ``source_batches`` and
        ``self.blend_weights``. Higher scores should increase a source's DoGE blend weight.
        """
        target_gradient = calc_alignment_gradient_vector(state, target_batch, model)
        scores = {}
        for path, batch in source_batches.items():
            source_gradient = calc_alignment_gradient_vector(state, batch, model)
            # Compute cosine similarity: dot(source, target) / (norm(source) * norm(target)).
            # This measures gradient direction instead of raw gradient scale and is in [-1, 1].
            denominator = source_gradient.norm() * target_gradient.norm()
            score = (
                torch.zeros((), dtype=source_gradient.dtype, device=source_gradient.device)
                if denominator.item() == 0
                else torch.dot(source_gradient, target_gradient) / denominator
            )
            if dist.is_available() and dist.is_initialized():
                # PoC synchronization: average the scalar score across the default process group so
                # every rank applies the same DoGE weight update. This is acceptable for the current
                # Qwen3-8B setup with PP=DP=CP=1 and TP-only sharding. TODO: replace with exact
                # sharded cosine reduction by summing dot/norm components over the relevant model
                # parallel group before forming the cosine score.
                dist.all_reduce(score, op=dist.ReduceOp.SUM)
                score /= dist.get_world_size()
            scores[path] = score.item()
        return scores

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

        if not model.training:
            # Megatron-Bridge reuses the forward step for validation under no-grad/eval mode.
            # DoGE scoring requires gradients, so validation reports the current weighted blend
            # without updating DoGE weights.
            return weighted_source_forward_step(
                state,
                source_batches,
                model,
                self.blend_weights,
                return_schedule_plan,
            )

        # Outer loop: use the target batch to score each source batch and update the data-blend
        # weights. This changes only ``self.blend_weights``, not the student model.
        scores = self._compute_alignment_scores(state, source_batches, target_batch, model)
        self.blend_weights = dict(self.updater.update(self.blend_weights, scores))

        # Inner loop: train the student on source batches mixed with the updated DoGE weights.
        # Megatron-Bridge backpropagates the returned loss and performs the optimizer step.
        # TODO: Reuse source gradients from the outer-loop scoring pass to avoid recomputing source
        # forward/backward work once the PoC no longer relies on Megatron-Bridge's normal loss path.
        return weighted_source_forward_step(
            state,
            source_batches,
            model,
            self.blend_weights,
            return_schedule_plan,
        )
