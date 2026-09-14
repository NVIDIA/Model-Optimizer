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

"""PDD objective adapter for AutoModel's diffusion training loop."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

if TYPE_CHECKING:
    from collections.abc import Callable

    from modelopt.torch.fastgen import PDDPipeline


@dataclass
class _TrajectorySlot:
    """Transient Algorithm 3 state for one gradient-accumulation lane."""

    state: torch.Tensor
    condition: tuple[torch.Tensor, torch.Tensor]
    n: torch.Tensor


class PDDFlowMatchingStepAdapter:
    """Expose the PDD objective through AutoModel's flow-matching ``step`` API."""

    def __init__(
        self,
        pipeline: PDDPipeline,
        *,
        grad_acc_steps: int = 1,
        latent_shape: tuple[int, ...] | None = None,
        optimizer_step_getter: Callable[[], int] | None = None,
    ) -> None:
        self.pipeline = pipeline
        if type(grad_acc_steps) is not int or grad_acc_steps <= 0:
            raise ValueError("grad_acc_steps must be a positive integer.")
        if pipeline.config.data_free:
            if latent_shape is None or not latent_shape:
                raise ValueError("latent_shape is required for data-free PDD.")
            if any(type(dimension) is not int or dimension <= 0 for dimension in latent_shape):
                raise ValueError("latent_shape must contain positive integers.")
        self._latent_shape = latent_shape
        self._optimizer_step_getter = optimizer_step_getter
        self._slots: list[_TrajectorySlot | None] = [None] * grad_acc_steps
        self._active_global_step: int | None = None
        self._slot_cursor = 0

    @staticmethod
    def _condition(
        batch: dict[str, Any],
        *,
        device: torch.device,
        dtype: torch.dtype,
        prefix: str = "",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            batch[f"{prefix}text_embeddings"].to(
                device=device,
                dtype=dtype,
                non_blocking=True,
            ),
            batch[f"{prefix}text_embeddings_mask"].to(device=device, non_blocking=True),
        )

    def _slot_index(self, global_step: int) -> int:
        optimizer_step = (
            global_step
            if self._optimizer_step_getter is None
            else int(self._optimizer_step_getter())
        )
        if self._active_global_step != optimizer_step:
            self._active_global_step = optimizer_step
            self._slot_cursor = 0
        if self._slot_cursor >= len(self._slots):
            raise RuntimeError(
                "AutoModel supplied more microbatches than the configured gradient "
                "accumulation steps."
            )
        index = self._slot_cursor
        self._slot_cursor += 1
        return index

    def _fresh_state(
        self,
        batch_size: int,
        *,
        device: torch.device,
    ) -> torch.Tensor:
        if self._latent_shape is None:  # guarded by __init__
            raise RuntimeError("data-free PDD latent shape was not configured.")
        noise = torch.randn(
            (batch_size, *self._latent_shape),
            device=device,
            dtype=torch.float32,
        )
        return (noise.to(torch.float64) * self.pipeline.config.grid_max_t).to(torch.float32)

    def step(
        self,
        model: nn.Module,
        batch: dict[str, Any],
        device: torch.device = torch.device("cuda"),
        dtype: torch.dtype = torch.bfloat16,
        global_step: int = 0,
        collect_metrics: bool = True,
        check_loss: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, None, dict[str, Any]]:
        """Prepare one AutoModel batch and return its PDD loss in the stock tuple shape."""
        del model

        incoming_condition = self._condition(batch, device=device, dtype=dtype)
        negative_condition = None
        if self.pipeline.config.guidance_scale is not None:
            negative_condition = self._condition(
                batch,
                device=device,
                dtype=dtype,
                prefix="negative_",
            )

        if self.pipeline.config.data_free:
            slot_index = self._slot_index(global_step)
            slot = self._slots[slot_index]
            if slot is None:
                batch_size = incoming_condition[0].shape[0]
                state = self._fresh_state(batch_size, device=device)
                condition = incoming_condition
                n = torch.zeros(batch_size, device=device, dtype=torch.long)
            else:
                state = slot.state
                condition = slot.condition
                n = slot.n

            loss, metrics, next_state, next_n = self.pipeline.compute_data_free_loss(
                state,
                n=n,
                condition=condition,
                negative_condition=negative_condition,
                collect_metrics=collect_metrics,
            )
            if bool(torch.all(next_n == self.pipeline.config.grid_size)):
                self._slots[slot_index] = None
            else:
                self._slots[slot_index] = _TrajectorySlot(
                    state=next_state,
                    condition=condition,
                    n=next_n,
                )
        else:
            data = batch["image_latents"].to(device=device, dtype=dtype, non_blocking=True)
            loss, metrics = self.pipeline.compute_loss(
                data,
                condition=incoming_condition,
                negative_condition=negative_condition,
                collect_metrics=collect_metrics,
            )
        if check_loss and not bool(torch.isfinite(loss)):
            raise FloatingPointError("PDD loss is non-finite.")

        per_sample_loss = metrics["student_target_mse"]
        return per_sample_loss, loss, None, metrics if collect_metrics else {}
