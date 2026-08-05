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

"""Model and ModelOpt-distillation helpers for QAD."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
from torch import nn

import modelopt.torch.distill as mtd

if TYPE_CHECKING:
    from collections.abc import Sequence

try:
    from nemo_automodel.components.distributed.parallelizer import (
        PARALLELIZATION_STRATEGIES,
        DefaultParallelizationStrategy,
        register_parallel_strategy,
    )
except ImportError as exc:
    raise ImportError(
        "The FastGen QAD example requires nemo_automodel. Install "
        "examples/diffusers/fastgen/requirements.txt."
    ) from exc


class _QwenImageParallelizationStrategy(DefaultParallelizationStrategy):
    """Checkpoint complete Qwen transformer blocks before AutoModel applies FSDP."""

    def parallelize(
        self,
        model,
        device_mesh,
        activation_checkpointing: bool = False,
        **kwargs,
    ):
        if activation_checkpointing:
            from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
                CheckpointImpl,
                checkpoint_wrapper,
            )

            blocks = getattr(model, "transformer_blocks", None)
            if blocks is None:
                raise AttributeError(
                    "QwenImageTransformer2DModel does not expose transformer_blocks."
                )
            for index, block in enumerate(blocks):
                blocks[index] = checkpoint_wrapper(
                    block,
                    checkpoint_impl=CheckpointImpl.NO_REENTRANT,
                )
            logging.info(
                "[QAD] Qwen-Image activation checkpointing enabled for %d full blocks",
                len(blocks),
            )

        return super().parallelize(
            model,
            device_mesh,
            activation_checkpointing=False,
            **kwargs,
        )


def register_qwen_image_parallelization_strategy() -> None:
    """Register the Qwen strategy unless AutoModel already ships a native strategy."""
    model_class_name = "QwenImageTransformer2DModel"
    if model_class_name not in PARALLELIZATION_STRATEGIES:
        register_parallel_strategy(name=model_class_name)(_QwenImageParallelizationStrategy)


register_qwen_image_parallelization_strategy()


def _extract_tensor(output: Any, selector: str) -> torch.Tensor:
    """Select a tensor from a Diffusers root or Qwen dual-stream block output."""
    normalized = selector.lower()
    if torch.is_tensor(output):
        return output
    if hasattr(output, "sample") and normalized in {"sample", "output", "tensor"}:
        return output.sample
    if isinstance(output, dict):
        if selector not in output:
            raise KeyError(f"Selector {selector!r} is not present in layer output keys.")
        selected = output[selector]
        if not torch.is_tensor(selected):
            raise TypeError(f"Layer output {selector!r} is not a tensor.")
        return selected
    if isinstance(output, tuple | list):
        index_by_name = {
            "sample": 0,
            "output": 0,
            "tensor": 0,
            "first": 0,
            "encoder_hidden_states": 0,
            "text": 0,
            "hidden_states": 1,
            "image": 1,
            "last": -1,
        }
        if normalized not in index_by_name:
            raise ValueError(
                f"Unsupported tuple selector {selector!r}; use hidden_states/image, "
                "encoder_hidden_states/text, first, last, or sample."
            )
        selected = output[index_by_name[normalized]]
        if not torch.is_tensor(selected):
            raise TypeError(f"Selected {selector!r} output is not a tensor.")
        return selected
    raise TypeError(f"Cannot select {selector!r} from output type {type(output).__name__}.")


class TensorOutputDelegate(nn.Module):
    """Forward to a model while exposing only its final tensor to ModelOpt KD.

    The wrapped model is deliberately stored outside ``nn.Module._modules``. This
    keeps the controller parameter-free and prevents its state_dict from duplicating
    either the FSDP student or teacher. ``get_submodule`` still routes layerwise
    criterion paths to the live wrapped transformer.
    """

    def __init__(self, target: nn.Module):
        super().__init__()
        self.__dict__["_qad_target"] = target

    @property
    def target(self) -> nn.Module:
        return self.__dict__["_qad_target"]

    def forward(self, *args, **kwargs) -> torch.Tensor:
        return _extract_tensor(self.target(*args, **kwargs), "sample")

    def get_submodule(self, target: str) -> nn.Module:
        if target == "":
            return self
        return self.target.get_submodule(target)


class SelectedMSELoss(nn.modules.loss._Loss):
    """FP32 MSE after selecting a stream from a captured layer output."""

    def __init__(self, selector: str = "sample"):
        super().__init__(reduction="mean")
        self.selector = selector

    def forward(self, student_output: Any, teacher_output: Any) -> torch.Tensor:
        student = _extract_tensor(student_output, self.selector)
        teacher = _extract_tensor(teacher_output, self.selector)
        return F.mse_loss(student.float(), teacher.float(), reduction="mean")


class AdditiveLossBalancer(mtd.DistillationLossBalancer):
    """Apply independent, additive weights to task and KD loss terms."""

    def __init__(self, *, task_weight: float, kd_weights: Sequence[float]):
        super().__init__()
        self.task_weight = float(task_weight)
        self.kd_weights = tuple(float(weight) for weight in kd_weights)

    def forward(self, losses: dict[str, torch.Tensor]) -> torch.Tensor:
        losses = dict(losses)
        student_loss = losses.pop("student_loss", None)
        if not losses:
            raise RuntimeError("QAD received no KD loss terms.")
        total = None
        if self.task_weight != 0.0:
            if student_loss is None:
                raise RuntimeError("A nonzero QAD task weight requires student_loss.")
            total = student_loss * self.task_weight

        if len(losses) != len(self.kd_weights):
            raise RuntimeError(
                "ModelOpt returned an unexpected number of KD losses: "
                f"expected {len(self.kd_weights)}, got {len(losses)}."
            )
        for loss, weight in zip(losses.values(), self.kd_weights):
            # Multiplying a disabled NaN/Inf term by zero would still poison the
            # objective. Skip disabled terms completely while retaining their
            # detached diagnostics in the pipeline.
            if weight == 0.0:
                continue
            weighted_loss = loss * weight
            total = weighted_loss if total is None else total + weighted_loss
        if total is None:
            raise RuntimeError("QAD has no nonzero loss coefficient.")
        return total


def build_distillation_controller(
    *,
    student: nn.Module,
    teacher: nn.Module,
    output_weight: float,
    task_weight: float,
    layer_pairs: Sequence[dict[str, Any]],
) -> tuple[nn.Module, tuple[str, ...]]:
    """Create a parameter-free ModelOpt KD controller around live FSDP models."""
    criterion: dict[tuple[str, str], nn.modules.loss._Loss] = {("", ""): SelectedMSELoss("sample")}
    names = ["output_mse"]
    weights = [float(output_weight)]
    seen_pairs = {("", "")}

    for index, pair in enumerate(layer_pairs):
        student_layer = str(pair["student_layer"])
        teacher_layer = str(pair.get("teacher_layer", student_layer))
        selector = str(pair.get("selector", "hidden_states"))
        weight = float(pair.get("weight", 1.0))
        key = (student_layer, teacher_layer)
        if key in seen_pairs:
            raise ValueError(f"Duplicate QAD layer pair: {key!r}")
        seen_pairs.add(key)
        criterion[key] = SelectedMSELoss(selector)
        names.append(f"layer_{index}_{student_layer}_{selector}_mse")
        weights.append(weight)

    controller = mtd.convert(
        TensorOutputDelegate(student),
        mode=[
            (
                "kd_loss",
                {
                    "teacher_model": TensorOutputDelegate(teacher),
                    "criterion": criterion,
                    "loss_balancer": AdditiveLossBalancer(
                        task_weight=task_weight,
                        kd_weights=weights,
                    ),
                    "expose_minimal_state_dict": True,
                },
            )
        ],
    )
    return controller, tuple(names)


def clear_captured_outputs(controller: nn.Module) -> None:
    """Release activation references before forwards and after checkpoint recompute."""
    for student_layer, teacher_layer in controller._layers_to_loss:
        student_layer._intermediate_output = None
        teacher_layer._intermediate_output = None
