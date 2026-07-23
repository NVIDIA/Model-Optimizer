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
from typing import Literal

import torch
from megatron.bridge.training.gpt_step import forward_step_modelopt
from megatron.bridge.training.state import GlobalState
from megatron.core.models.gpt import GPTModel

import modelopt.torch.utils.distributed as dist
from modelopt.torch.distill.doge import (
    DoGEWeightUpdater,
    normalize_data_path_weights,
    sample_data_path_by_weight,
)
from modelopt.torch.distill.doge_megatron_data import (
    DoGEDataIterators,
    _build_doge_data_iterators,
    _next_doge_batches,
)
from modelopt.torch.distill.doge_megatron_loss import (
    DoGEAlignmentDiagnostics,
    DoGEAlignmentParamScope,
    DoGEVirtualStepDiagnostic,
    compute_alignment_scores,
    compute_virtual_step_diagnostics,
    sampled_source_forward_step,
    weighted_source_forward_step,
    zero_sampled_source_forward_step,
    zero_weighted_source_forward_step,
)
from modelopt.torch.utils import print_rank_0

DoGETrainLossMode = Literal["weighted", "sampled"]
DoGEWeightUpdateStrategy = Literal["alignment", "kd_gap", "target_kd_gap"]

__all__ = ["DoGEForwardStep", "DoGETrainLossMode", "DoGEWeightUpdateStrategy"]


class DoGEForwardStep:
    """Callable forward-step placeholder to pass into Megatron-Bridge ``pretrain``."""

    def __init__(
        self,
        data_paths: list[str],
        target_data_paths: list[str],
        meta_lr: float,
        output_dir: str | Path,
        min_blend_weight: float = 0.0,
        freeze_student: bool = False,
        freeze_blend: bool = False,
        schedule_end_data_paths: list[str] | None = None,
        virtual_step_candidate_weights: list[list[float]] | None = None,
        virtual_step_lr: float | None = None,
        virtual_step_num_steps: int = 1,
        alignment_param_scope: DoGEAlignmentParamScope = "final_mlp",
        train_loss_mode: DoGETrainLossMode = "weighted",
        weight_update_strategy: DoGEWeightUpdateStrategy = "alignment",
        sampling_seed: int = 1234,
    ) -> None:
        """Initialize the callable state used by Megatron-Bridge ``pretrain``.

        Args:
            data_paths: Initial training-data blend in Megatron WEIGHT PATH format. The weights
                are normalized into ``self.blend_weights`` and updated during DoGE.
            target_data_paths: Fixed target-objective blend in Megatron WEIGHT PATH format. The
                weights are normalized into ``self.target_blend_weights`` and are not updated.
            meta_lr: Learning rate for exponentiated blend-weight updates.
            output_dir: Directory where DoGE writes the weight trajectory.
            min_blend_weight: Optional minimum normalized weight for each source after every DoGE
                update.
            freeze_student: Log DoGE scores without updating student weights.
            freeze_blend: Log candidate blend-weight updates without applying them.
            schedule_end_data_paths: Optional final training-data blend in Megatron WEIGHT PATH
                format. When provided, DoGE linearly interpolates from ``data_paths`` to this
                blend over the training run and skips adaptive blend updates.
            virtual_step_candidate_weights: Optional source-order candidate blend weights used
                for frozen-model/frozen-blend virtual-step diagnostics.
            virtual_step_lr: Learning rate for virtual selected-parameter diagnostic steps.
            virtual_step_num_steps: Number of repeated virtual updates per candidate diagnostic.
            alignment_param_scope: Parameter scope used for DoGE gradient scoring and virtual-step
                diagnostics. ``all_trainable`` is intended for expensive diagnostic runs.
            train_loss_mode: How to construct the real student-update loss. ``weighted`` computes
                every source loss each step and combines them by weight. ``sampled`` samples one
                source by current weights and returns that unweighted loss.
            weight_update_strategy: How to turn per-source diagnostics into the next blend.
                ``alignment`` uses the DoGE gradient-alignment update. ``kd_gap`` sets weights
                proportional to per-source KD loss as a naive PASER-style baseline.
                ``target_kd_gap`` applies the KD-gap update only to sources that are also in the
                target blend and sets non-target source weights to zero.
            sampling_seed: Seed used for deterministic source sampling in ``sampled`` mode.
        """
        if train_loss_mode not in ("weighted", "sampled"):
            raise ValueError(f"Unsupported DoGE train loss mode: {train_loss_mode!r}")
        if weight_update_strategy not in ("alignment", "kd_gap", "target_kd_gap"):
            raise ValueError(f"Unsupported DoGE weight update strategy: {weight_update_strategy!r}")
        self.data_paths = tuple(data_paths)
        self.target_data_paths = tuple(target_data_paths)
        self.updater = DoGEWeightUpdater(meta_lr=meta_lr, min_weight=min_blend_weight)
        self.min_blend_weight = min_blend_weight
        self.blend_weights: dict[str, float] = normalize_data_path_weights(data_paths)
        self.target_blend_weights: dict[str, float] = normalize_data_path_weights(target_data_paths)
        self.doge_data_iterators: DoGEDataIterators | None = None
        self.trajectory_path = Path(output_dir) / "doge_weights.jsonl"
        self.freeze_student = freeze_student
        self.freeze_blend = freeze_blend
        self.schedule_start_iteration: int | None = None
        self.schedule_start_blend_weights = dict(self.blend_weights)
        self.schedule_end_blend_weights = _normalize_schedule_end_weights(
            schedule_end_data_paths, tuple(self.blend_weights)
        )
        self.virtual_step_candidate_blend_weights = _normalize_virtual_step_candidate_weights(
            virtual_step_candidate_weights, tuple(self.blend_weights)
        )
        self.virtual_step_lr = virtual_step_lr
        self.virtual_step_num_steps = virtual_step_num_steps
        self.alignment_param_scope = alignment_param_scope
        self.train_loss_mode = train_loss_mode
        self.weight_update_strategy = weight_update_strategy
        self.sampling_seed = sampling_seed

    def write_trajectory_record(
        self,
        iteration: int,
        alignment_scores: dict[str, float] | None = None,
        alignment_debug: dict[str, dict[str, float | int]] | None = None,
        source_probe_kd_loss: dict[str, float] | None = None,
        target_probe_kd_loss: float | None = None,
        candidate_blend_weights: dict[str, float] | None = None,
        virtual_step_diagnostics: dict[str, DoGEVirtualStepDiagnostic] | None = None,
        sampled_source_path: str | None = None,
        weight_update_scores: dict[str, float] | None = None,
    ) -> None:
        """Write one DoGE weight-trajectory record on rank 0."""
        if not dist.is_master():
            return

        record = {
            "iteration": iteration,
            "alignment_scores": alignment_scores or {},
            "blend_weights": self.blend_weights,
            "train_loss_mode": self.train_loss_mode,
            "weight_update_strategy": self.weight_update_strategy,
        }
        if sampled_source_path is not None:
            record["sampled_source_path"] = sampled_source_path
        if candidate_blend_weights is not None:
            # Candidate weights are the next blend DoGE would apply from the current scores. They
            # can differ from ``blend_weights`` when ``--doge_freeze_blend`` keeps training fixed.
            record["candidate_blend_weights"] = candidate_blend_weights
        if alignment_debug is not None:
            record["alignment_debug"] = alignment_debug
        if weight_update_scores is not None:
            record["weight_update_scores"] = weight_update_scores
        if source_probe_kd_loss is not None:
            record["source_probe_kd_loss"] = source_probe_kd_loss
        if target_probe_kd_loss is not None:
            record["target_probe_kd_loss"] = target_probe_kd_loss
        if virtual_step_diagnostics is not None:
            record["virtual_step_diagnostics"] = virtual_step_diagnostics

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
        if sampled_source_path is not None:
            summary += f" | train_source={Path(sampled_source_path).name}"
        if virtual_step_diagnostics:
            best_label, best_diagnostics = min(
                virtual_step_diagnostics.items(),
                key=lambda item: item[1]["delta_target_probe_kd"],
            )
            summary += (
                f" | virtual_best={best_label}"
                f" delta_target_probe_kd={best_diagnostics['delta_target_probe_kd']:.4g}"
            )
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
        alignment scores. The inner loop returns either the weighted multi-source loss or one
        sampled source loss for Megatron-Bridge to backpropagate with its normal optimizer step.
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

        self._apply_scheduled_blend_weights(state)
        source_batches, target_batch = _next_doge_batches(state, model, self.doge_data_iterators)

        # Outer loop: use the target batch to score each source batch and propose a data-blend
        # update. This changes only ``self.blend_weights`` unless blend updates are frozen.
        alignment_result = compute_alignment_scores(
            state,
            source_batches,
            target_batch,
            model,
            self.blend_weights,
            self.alignment_param_scope,
        )
        virtual_step_diagnostics = None
        if self.virtual_step_candidate_blend_weights:
            if self.virtual_step_lr is None:
                raise RuntimeError("DoGE virtual-step diagnostics require virtual_step_lr.")
            virtual_step_diagnostics = compute_virtual_step_diagnostics(
                state,
                source_batches,
                alignment_result.source_gradients,
                target_batch,
                model,
                self.virtual_step_candidate_blend_weights,
                self.virtual_step_lr,
                alignment_result.target_probe_kd_loss,
                self.alignment_param_scope,
                self.virtual_step_num_steps,
            )

        weight_update_scores = self._get_weight_update_scores(alignment_result)
        candidate_blend_weights = self._get_candidate_blend_weights(weight_update_scores)
        if self.schedule_end_blend_weights is None and not self.freeze_blend:
            self.blend_weights = candidate_blend_weights

        sampled_source_path = None
        if self.train_loss_mode == "sampled":
            sampled_source_path = sample_data_path_by_weight(
                self.blend_weights,
                iteration=state.train_state.step + 1,
                seed=self.sampling_seed,
            )
        self.write_trajectory_record(
            state.train_state.step + 1,
            alignment_result.scores,
            alignment_result.alignment_debug,
            alignment_result.source_probe_kd_loss,
            alignment_result.target_probe_kd_loss,
            candidate_blend_weights,
            virtual_step_diagnostics,
            sampled_source_path,
            weight_update_scores,
        )
        if self.train_loss_mode == "sampled":
            if sampled_source_path is None:
                raise RuntimeError("DoGE sampled train loss did not select a source.")
            if self.freeze_student:
                return zero_sampled_source_forward_step(
                    state,
                    source_batches,
                    sampled_source_path,
                    model,
                    return_schedule_plan,
                )
            return sampled_source_forward_step(
                state,
                source_batches,
                sampled_source_path,
                model,
                return_schedule_plan,
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
        # In weighted mode this means one weighted loss from all sources; in sampled mode this
        # branch is skipped above and only one source contributes to the optimizer step.
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

    def _get_weight_update_scores(
        self, alignment_result: DoGEAlignmentDiagnostics
    ) -> dict[str, float]:
        """Return the per-source scores used to propose the next blend."""
        if self.weight_update_strategy == "alignment":
            return alignment_result.scores
        if self.weight_update_strategy in ("kd_gap", "target_kd_gap"):
            return alignment_result.source_probe_kd_loss
        raise RuntimeError(
            f"Unsupported DoGE weight update strategy: {self.weight_update_strategy!r}"
        )

    def _get_candidate_blend_weights(
        self, weight_update_scores: dict[str, float]
    ) -> dict[str, float]:
        """Return candidate weights from the configured update strategy."""
        if self.weight_update_strategy == "alignment":
            return dict(self.updater.update(self.blend_weights, weight_update_scores))
        if self.weight_update_strategy == "kd_gap":
            return _normalize_scores_as_weights(weight_update_scores, self.min_blend_weight)
        if self.weight_update_strategy == "target_kd_gap":
            return _normalize_target_scores_as_weights(
                weight_update_scores,
                self.blend_weights,
                self.target_blend_weights,
                self.min_blend_weight,
            )
        raise RuntimeError(
            f"Unsupported DoGE weight update strategy: {self.weight_update_strategy!r}"
        )

    def _apply_scheduled_blend_weights(self, state: GlobalState) -> None:
        """Update ``blend_weights`` from the configured linear schedule, if any."""
        if self.schedule_end_blend_weights is None:
            return

        current_iteration = state.train_state.step + 1
        if self.schedule_start_iteration is None:
            self.schedule_start_iteration = current_iteration

        end_iteration = state.cfg.train.train_iters
        if current_iteration >= end_iteration or end_iteration <= self.schedule_start_iteration:
            progress = 1.0
        else:
            progress = (current_iteration - self.schedule_start_iteration) / (
                end_iteration - self.schedule_start_iteration
            )

        scheduled_weights = {
            path: (1.0 - progress) * self.schedule_start_blend_weights[path]
            + progress * self.schedule_end_blend_weights[path]
            for path in self.blend_weights
        }
        total_weight = sum(scheduled_weights.values())
        self.blend_weights = {
            path: weight / total_weight for path, weight in scheduled_weights.items()
        }


def _normalize_virtual_step_candidate_weights(
    candidate_weights: list[list[float]] | None, source_paths: tuple[str, ...]
) -> dict[str, dict[str, float]]:
    """Return normalized candidate blend weights keyed by readable candidate labels."""
    if candidate_weights is None:
        return {}

    candidates = {}
    for weights in candidate_weights:
        if len(weights) != len(source_paths):
            raise ValueError(
                "Each --doge_virtual_step_candidate_weights entry must provide one weight per "
                f"training source: expected {len(source_paths)}, got {len(weights)}."
            )
        if any(weight < 0 for weight in weights):
            raise ValueError("--doge_virtual_step_candidate_weights must be non-negative.")
        total_weight = sum(weights)
        if total_weight <= 0:
            raise ValueError("--doge_virtual_step_candidate_weights must sum to a positive value.")

        label = "/".join(f"{weight:g}" for weight in weights)
        candidates[label] = {
            path: weight / total_weight for path, weight in zip(source_paths, weights)
        }
    return candidates


def _normalize_scores_as_weights(
    scores: dict[str, float], min_blend_weight: float = 0.0
) -> dict[str, float]:
    """Convert non-negative per-source scores into normalized blend weights."""
    if not scores:
        raise ValueError("Cannot normalize empty DoGE weight-update scores.")
    if min_blend_weight * len(scores) >= 1:
        raise ValueError(
            "min_blend_weight is too large for the number of sources: "
            f"{min_blend_weight} * {len(scores)} must be less than 1."
        )

    clipped_scores = {path: max(score, 0.0) for path, score in scores.items()}
    total_score = sum(clipped_scores.values())
    if total_score == 0:
        normalized_scores = {path: 1.0 / len(clipped_scores) for path in clipped_scores}
    else:
        normalized_scores = {path: score / total_score for path, score in clipped_scores.items()}
    if min_blend_weight == 0:
        return normalized_scores

    remaining_weight = 1.0 - min_blend_weight * len(normalized_scores)
    return {
        path: min_blend_weight + remaining_weight * weight
        for path, weight in normalized_scores.items()
    }


def _normalize_target_scores_as_weights(
    scores: dict[str, float],
    blend_weights: dict[str, float],
    target_blend_weights: dict[str, float],
    min_blend_weight: float = 0.0,
) -> dict[str, float]:
    """Normalize KD-gap scores only over sources that are also target sources."""
    target_scores = {
        path: scores[path]
        for path in blend_weights
        if path in target_blend_weights and path in scores
    }
    if not target_scores:
        raise ValueError(
            "target_kd_gap requires at least one identical path in --data_paths and "
            "--target_data_paths."
        )

    target_weights = _normalize_scores_as_weights(target_scores, min_blend_weight)
    return {path: target_weights.get(path, 0.0) for path in blend_weights}


def _normalize_schedule_end_weights(
    schedule_end_data_paths: list[str] | None, source_paths: tuple[str, ...]
) -> dict[str, float] | None:
    """Return normalized schedule-end weights keyed by the initial source paths."""
    if schedule_end_data_paths is None:
        return None

    weights = normalize_data_path_weights(schedule_end_data_paths)
    if set(weights) != set(source_paths):
        raise ValueError(
            "--doge_schedule_end_data_paths must contain exactly the same source paths as "
            "--data_paths."
        )
    return {path: weights[path] for path in source_paths}
