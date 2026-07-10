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
from dataclasses import dataclass
from functools import partial

import torch
from megatron.bridge.training.gpt_step import forward_step_modelopt
from megatron.bridge.training.state import GlobalState
from megatron.core.models.gpt import GPTModel

from modelopt.torch.distill.doge import DoGEWeightUpdater, normalize_data_path_weights

__all__ = ["DoGEDataIterators", "DoGEForwardStep"]


@dataclass
class DoGEDataIterators:
    """Data iterators required by one DoGE step.

    Attributes:
        source_iterators: One iterator per tunable training dataset path from ``--data_paths``.
        target_iterator: Iterator over the fixed target objective.
    """

    source_iterators: dict[str, Iterable]
    target_iterator: Iterable


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
        raise NotImplementedError("DoGE data iterator construction is not implemented yet.")

    def _next_doge_batches(
        self,
    ) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, torch.Tensor]]:
        """Return raw Megatron GPT batches for DoGE source and target losses.

        Source batches are keyed by the same dataset paths as ``self.blend_weights``. Each batch is
        the raw dictionary returned by the Megatron data iterator, typically containing tensors such
        as ``tokens``, ``labels``, ``loss_mask``, ``attention_mask``, and ``position_ids``.
        """
        raise NotImplementedError("DoGE batch sampling is not implemented yet.")

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

    def __call__(
        self,
        state: GlobalState,
        data_iterator: Iterable,
        model: GPTModel,
        return_schedule_plan: bool = False,
    ) -> tuple[torch.Tensor, partial]:
        """Run as Megatron-Bridge ``pretrain`` forward step for one DoGE iteration.

        Returns the ``(output_tensor, loss_function)`` pair expected by Megatron-Bridge
        after the DoGE implementation computes updated blend weights for the training datasets.
        """
        return forward_step_modelopt(state, data_iterator, model, return_schedule_plan)
