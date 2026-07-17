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

import json
from collections.abc import Iterable
from functools import partial
from pathlib import Path

import torch
from megatron.bridge.training.gpt_step import forward_step_modelopt
from megatron.bridge.training.state import GlobalState
from megatron.core.models.gpt import GPTModel

import modelopt.torch.utils.distributed as dist
from modelopt.torch.distill.doge import DoGEWeightUpdater, normalize_data_path_weights
from modelopt.torch.distill.doge_megatron_data import (
    DoGEDataIterators,
    _build_doge_data_iterators,
    _next_doge_batches,
)
from modelopt.torch.distill.doge_megatron_loss import (
    compute_alignment_scores,
    weighted_source_forward_step,
    zero_weighted_source_forward_step,
)
from modelopt.torch.utils import print_rank_0

__all__ = ["DoGEForwardStep"]


class DoGEForwardStep:
    """Callable forward-step placeholder to pass into Megatron-Bridge ``pretrain``."""

    def __init__(
        self,
        data_paths: list[str],
        target_data_paths: list[str],
        meta_lr: float,
        output_dir: str | Path,
        freeze_student: bool = False,
        freeze_blend: bool = False,
    ) -> None:
        """Initialize the callable state used by Megatron-Bridge ``pretrain``.

        Args:
            data_paths: Initial training-data blend in Megatron WEIGHT PATH format. The weights
                are normalized into ``self.blend_weights`` and updated during DoGE.
            target_data_paths: Fixed target-objective blend in Megatron WEIGHT PATH format. The
                weights are normalized into ``self.target_blend_weights`` and are not updated.
            meta_lr: Learning rate for exponentiated blend-weight updates.
            output_dir: Directory where DoGE writes the weight trajectory.
            freeze_student: Log DoGE scores without updating student weights.
            freeze_blend: Log candidate blend-weight updates without applying them.
        """
        self.data_paths = tuple(data_paths)
        self.target_data_paths = tuple(target_data_paths)
        self.updater = DoGEWeightUpdater(meta_lr=meta_lr)
        self.blend_weights: dict[str, float] = normalize_data_path_weights(data_paths)
        self.target_blend_weights: dict[str, float] = normalize_data_path_weights(target_data_paths)
        self.doge_data_iterators: DoGEDataIterators | None = None
        self.trajectory_path = Path(output_dir) / "doge_weights.jsonl"
        self.freeze_student = freeze_student
        self.freeze_blend = freeze_blend

    def write_trajectory_record(
        self,
        iteration: int,
        alignment_scores: dict[str, float] | None = None,
        alignment_debug: dict[str, dict[str, float | int]] | None = None,
        source_probe_kd_loss: dict[str, float] | None = None,
        target_probe_kd_loss: float | None = None,
        candidate_blend_weights: dict[str, float] | None = None,
    ) -> None:
        """Write one DoGE weight-trajectory record on rank 0."""
        if not dist.is_master():
            return

        record = {
            "iteration": iteration,
            "alignment_scores": alignment_scores or {},
            "blend_weights": self.blend_weights,
        }
        if candidate_blend_weights is not None:
            # Candidate weights are the next blend DoGE would apply from the current scores. They
            # can differ from ``blend_weights`` when ``--doge_freeze_blend`` keeps training fixed.
            record["candidate_blend_weights"] = candidate_blend_weights
        if alignment_debug is not None:
            record["alignment_debug"] = alignment_debug
        if source_probe_kd_loss is not None:
            record["source_probe_kd_loss"] = source_probe_kd_loss
        if target_probe_kd_loss is not None:
            record["target_probe_kd_loss"] = target_probe_kd_loss

        self.trajectory_path.parent.mkdir(parents=True, exist_ok=True)
        mode = "w" if iteration == 0 else "a"
        with self.trajectory_path.open(mode, encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")

        summary_parts = []
        for path, weight in self.blend_weights.items():
            part = f"{Path(path).name} weight={weight:.4f}"
            if alignment_scores is not None:
                part += f" alignment={alignment_scores[path]:.4f}"
            if source_probe_kd_loss is not None:
                part += f" probe_kd={source_probe_kd_loss[path]:.4f}"
            summary_parts.append(part)
        summary = " | ".join(summary_parts)
        if target_probe_kd_loss is not None:
            summary += f" | target_probe_kd={target_probe_kd_loss:.4f}"
        print_rank_0(f"DoGE iteration {iteration} | {summary}")

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
        if not model.training:
            # Megatron-Bridge reuses the forward step for validation under no-grad/eval mode.
            # DoGE scoring requires gradients, so validation should use Bridge's validation
            # iterator instead of DoGE's per-source training iterators. This reports the fixed
            # validation split derived from the initial --data_paths, not the DoGE target objective.
            # TODO: Add a DoGE-specific validation schedule where target validation is the primary
            # metric and this initial-blend validation is either renamed or skipped.
            return forward_step_modelopt(state, data_iterator, model, return_schedule_plan)

        if self.doge_data_iterators is None:
            self.doge_data_iterators = _build_doge_data_iterators(
                state.cfg,
                model,
                self.blend_weights,
                self.target_data_paths,
            )

        source_batches, target_batch = _next_doge_batches(state, model, self.doge_data_iterators)

        # Outer loop: use the target batch to score each source batch and propose a data-blend
        # update. This changes only ``self.blend_weights`` unless blend updates are frozen.
        alignment_result = compute_alignment_scores(
            state,
            source_batches,
            target_batch,
            model,
            self.blend_weights,
        )

        candidate_blend_weights = dict(
            self.updater.update(self.blend_weights, alignment_result.scores)
        )
        if not self.freeze_blend:
            self.blend_weights = candidate_blend_weights
        self.write_trajectory_record(
            state.train_state.step + 1,
            alignment_result.scores,
            alignment_result.alignment_debug,
            alignment_result.source_probe_kd_loss,
            alignment_result.target_probe_kd_loss,
            candidate_blend_weights,
        )
        if self.freeze_student:
            return zero_weighted_source_forward_step(
                state,
                source_batches,
                model,
                self.blend_weights,
                return_schedule_plan,
            )

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
