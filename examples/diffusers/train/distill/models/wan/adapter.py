# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Wan2.2 training forward adapter.

Handles the conversion between the unified batch format [B, C, F, H, W]
and the native WanModel's forward interface (list-of-tensors: x, t, context, seq_len),
plus noise application and loss.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch import Tensor

from ...interfaces import BackboneInputs

if TYPE_CHECKING:
    from collections.abc import Callable


class WanTrainingForwardAdapter:
    """Flow matching training forward adapter for Wan2.2."""

    MOCK_TEXT_EMBED_DIM: int = 4096

    def __init__(self, variant: str | None = None) -> None:
        from .loader import get_variant_config

        var = get_variant_config(variant)
        cfg = var["config"]()
        self.patch_size: tuple[int, int, int] = cfg.patch_size
        self.MOCK_LATENT_SHAPE: tuple[int, ...] = (var["z_dim"], 4, 32, 32)

    def prepare_inputs(
        self,
        batch: dict[str, Tensor],
        noise: Tensor,
        timesteps: Tensor,
        pipeline=None,
    ) -> BackboneInputs:
        latents = batch["latents"]  # [B, C, F, H, W]
        text_embeds = batch["text_embeds"]  # [B, L, D]

        b, _c, n_f, n_h, n_w = latents.shape

        sigmas = timesteps.view(-1, 1, 1, 1, 1)
        noisy_latents = (1.0 - sigmas) * latents + sigmas * noise
        targets = noise - latents

        loss_mask = torch.ones(b, dtype=torch.float32, device=latents.device)

        pt, ph, pw = self.patch_size
        seq_len = (n_f // pt) * (n_h // ph) * (n_w // pw)

        return BackboneInputs(
            noisy_input=noisy_latents,
            targets=targets,
            loss_mask=loss_mask,
            forward_kwargs={
                "x": list(noisy_latents.unbind(0)),
                "t": (timesteps * 1000.0).unsqueeze(1).expand(-1, seq_len),
                "context": list(text_embeds.unbind(0)),
                "seq_len": seq_len,
            },
        )

    def forward_model(self, model: nn.Module, inputs: BackboneInputs) -> Tensor:
        # WanModel norms promote to float32 internally; autocast keeps
        # matmuls in bf16 to match the original Wan inference code.
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            output_list = model(**inputs.forward_kwargs)
        return torch.stack(output_list)

    def compute_task_loss(self, model_output: Tensor, inputs: BackboneInputs) -> Tensor:
        return (model_output - inputs.targets).pow(2).mean()

    def compute_distillation_loss(
        self, student_output: Tensor, teacher_output: Tensor, inputs: BackboneInputs
    ) -> Tensor:
        return (student_output - teacher_output).pow(2).mean()

    def get_output_transforms(self, model: nn.Module) -> dict[str, Callable]:
        # WanAttentionBlock.forward returns x: Tensor directly -- no transform needed.
        return {}


def create_wan_adapter(variant: str | None = None) -> WanTrainingForwardAdapter:
    return WanTrainingForwardAdapter(variant)
