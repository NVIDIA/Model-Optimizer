# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared fixtures for ``tests/unit/torch/fastgen/``.

Keeps the duplicated ``_ToyTransformer`` / ``_ToyDiscriminator`` / pipeline-builder
helpers in one place so individual test files can focus on assertions rather than
wiring.
"""

from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from modelopt.torch.fastgen import DMDConfig, DMDPipeline


class ToyTransformer(nn.Module):
    """Minimal diffusers-shaped transformer: output = Linear(hidden_states).

    Accepts ``hidden_states`` / ``timestep`` / ``encoder_hidden_states`` / **kwargs
    but ignores all of them except the first.
    """

    def __init__(self, d: int) -> None:
        super().__init__()
        self.linear = nn.Linear(d, d, bias=False)

    def forward(self, hidden_states, timestep=None, encoder_hidden_states=None, **kwargs):
        return self.linear(hidden_states)


class ToyDiscriminator(nn.Module):
    """Consumes ``list[Tensor]`` and returns 2D logits ``(B, 1)`` by averaging features."""

    def forward(self, features):
        x = features[0]
        return x.flatten(start_dim=1).mean(dim=-1, keepdim=True)


@pytest.fixture
def toy_transformer_factory():
    """Return a callable ``d -> ToyTransformer(d)`` factory."""
    return ToyTransformer


@pytest.fixture
def toy_discriminator_factory():
    """Return a callable ``() -> ToyDiscriminator()`` factory."""
    return ToyDiscriminator


@pytest.fixture
def build_pipeline():
    """Factory that constructs a :class:`DMDPipeline` with toy student/teacher/fake_score.

    Usage::

        pipeline = build_pipeline(
            d=4, pred_type="flow", gan_loss_weight_gen=0.03, discriminator=ToyDiscriminator()
        )
    """

    def _build(
        d: int,
        *,
        pred_type: str = "flow",
        fake_score_pred_type: str | None = None,
        num_train_timesteps: int | None = None,
        gan_loss_weight_gen: float = 0.0,
        gan_use_same_t_noise: bool = False,
        gan_r1_reg_weight: float = 0.0,
        ema=None,
        discriminator: nn.Module | None = None,
        seed: int = 0,
    ) -> DMDPipeline:
        torch.manual_seed(seed)
        student = ToyTransformer(d)
        teacher = ToyTransformer(d)
        fake_score = copy.deepcopy(teacher)
        cfg = DMDConfig(
            pred_type=pred_type,
            fake_score_pred_type=fake_score_pred_type,
            num_train_timesteps=num_train_timesteps,
            gan_loss_weight_gen=gan_loss_weight_gen,
            gan_use_same_t_noise=gan_use_same_t_noise,
            gan_r1_reg_weight=gan_r1_reg_weight,
            ema=ema,
        )
        return DMDPipeline(
            student,
            teacher,
            fake_score,
            cfg,
            discriminator=discriminator,
        )

    return _build
