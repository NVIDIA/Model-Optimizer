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

from typing import TYPE_CHECKING, Any

import torch
from torch import nn

if TYPE_CHECKING:
    from modelopt.torch.fastgen import PDDPipeline


class PDDFlowMatchingStepAdapter:
    """Expose the PDD objective through AutoModel's flow-matching ``step`` API."""

    def __init__(self, pipeline: PDDPipeline) -> None:
        self.pipeline = pipeline

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
        del model, global_step

        data = batch["image_latents"].to(device=device, dtype=dtype, non_blocking=True)
        condition = (
            batch["text_embeddings"].to(device=device, dtype=dtype, non_blocking=True),
            batch["text_embeddings_mask"].to(device=device, non_blocking=True),
        )
        negative_condition = None
        if self.pipeline.config.guidance_scale is not None:
            negative_condition = (
                batch["negative_text_embeddings"].to(
                    device=device,
                    dtype=dtype,
                    non_blocking=True,
                ),
                batch["negative_text_embeddings_mask"].to(device=device, non_blocking=True),
            )

        loss, metrics = self.pipeline.compute_loss(
            data,
            condition=condition,
            negative_condition=negative_condition,
            collect_metrics=collect_metrics,
        )
        if check_loss and not bool(torch.isfinite(loss)):
            raise FloatingPointError("PDD loss is non-finite.")

        per_sample_loss = metrics["student_target_mse"]
        return per_sample_loss, loss, None, metrics if collect_metrics else {}
