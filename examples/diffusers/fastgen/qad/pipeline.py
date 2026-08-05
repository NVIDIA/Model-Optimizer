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

"""QAD loss pipeline layered on AutoModel's flow-matching input preparation."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .modeling import clear_captured_outputs


class QADPipeline:
    """Run teacher/student on identical inputs and aggregate ModelOpt KD losses."""

    def __init__(self, flow_matching_pipeline, controller: nn.Module, loss_names: tuple[str, ...]):
        self.flow_matching_pipeline = flow_matching_pipeline
        self.controller = controller
        self.loss_names = loss_names

    def step(
        self,
        *,
        batch: dict[str, Any],
        device: torch.device,
        dtype: torch.dtype,
        global_step: int,
        check_loss: bool,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        clear_captured_outputs(self.controller)
        _, task_loss, _, _ = self.flow_matching_pipeline.step(
            model=self.controller,
            batch=batch,
            device=device,
            dtype=dtype,
            global_step=global_step,
            collect_metrics=False,
            # The flow target is optional in QAD. Validate the actual combined loss below.
            check_loss=False,
        )
        losses = self.controller.compute_kd_loss(
            student_loss=task_loss,
            skip_balancer=True,
        )
        total = self.controller.loss_balancer(losses)
        if check_loss and not bool(torch.isfinite(total.detach()).all()):
            raise FloatingPointError(f"Non-finite QAD loss at step {global_step}.")

        kd_values = [value for key, value in losses.items() if key != "student_loss"]
        if len(kd_values) != len(self.loss_names):
            raise RuntimeError(
                "QAD loss-name mapping is out of sync with ModelOpt's returned losses."
            )
        metrics = {"task_loss": task_loss.detach(), "total_loss": total.detach()}
        metrics.update({name: value.detach() for name, value in zip(self.loss_names, kd_values)})
        return total, metrics

    def clear(self) -> None:
        clear_captured_outputs(self.controller)
