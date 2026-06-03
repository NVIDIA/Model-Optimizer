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

"""Unit tests for the cosine-with-warmup LR scheduler used by bypass distillation.

``_get_lr`` is the scheduler invoked every step inside ``train``. An off-by-one
in the cosine ramp would silently degrade convergence — bypass jobs run for
hours and produce subtly worse student weights. The degenerate-budget guard
matters for tests and short sweeps where ``training_tokens`` is small.

Schedule shape (warmup_steps=W, lr_decay_steps=D):

    step ∈ [0, W]:        linear ramp 0 → base_lr (warmup branch)
    step ∈ (W, D]:        cosine decay base_lr → min_lr (cosine branch)
    step > D:             clamped to min_lr (post-decay branch)

The cosine uses ``decay_ratio = (step - W) / (D - W)`` so the boundary cases
align: at step=W+1 the cosine has just started (decay_ratio = 1/(D-W)) and at
step=D it reaches min_lr exactly (decay_ratio=1, coeff=0).
"""

import math

import pytest
from omegaconf import OmegaConf

from modelopt.torch.puzzletron.bypass_distillation.training_loop import _get_lr


def _make_cfg(
    *,
    warmup_steps: int,
    lr_decay_steps: int,
    learning_rate: float = 1.0,
    min_lr: float = 0.1,
):
    return OmegaConf.create(
        {
            "bypass": {
                "training": {
                    "warmup_steps": warmup_steps,
                    "lr_decay_steps": lr_decay_steps,
                    "learning_rate": learning_rate,
                    "min_lr": min_lr,
                }
            }
        }
    )


def test_degenerate_budget_returns_base_lr():
    """When ``lr_decay_steps <= warmup_steps`` (tiny test budgets), the scheduler
    must short-circuit to ``learning_rate`` rather than divide by zero."""
    for warmup_steps, lr_decay_steps, learning_rate in [(10, 10, 0.5), (20, 10, 0.7)]:
        cfg = _make_cfg(
            warmup_steps=warmup_steps,
            lr_decay_steps=lr_decay_steps,
            learning_rate=learning_rate,
        )
        assert _get_lr(cfg, step=0) == learning_rate
        assert _get_lr(cfg, step=99) == learning_rate


def test_lr_schedule_matches_key_points():
    cfg = _make_cfg(warmup_steps=10, lr_decay_steps=100, learning_rate=1.0)
    for step, expected, name in [
        (0, 0.0, "warmup start"),
        (5, 0.5, "warmup midpoint"),
        (10, 1.0, "warmup end"),
    ]:
        assert _get_lr(cfg, step=step) == pytest.approx(expected), name

    cfg = _make_cfg(warmup_steps=10, lr_decay_steps=20, learning_rate=1.0, min_lr=0.0)
    cosine_start = 0.5 * (1.0 + math.cos(math.pi * 0.1))
    cosine_midpoint = 0.5 * (1.0 + math.cos(math.pi * 0.5))
    for step, expected, name in [
        (11, cosine_start, "cosine starts immediately after warmup"),
        (15, cosine_midpoint, "cosine midpoint"),
        (20, 0.0, "cosine endpoint"),
        (21, 0.0, "post-decay clamp"),
        (1000, 0.0, "long post-decay clamp"),
    ]:
        assert _get_lr(cfg, step=step) == pytest.approx(expected), name
    assert _get_lr(cfg, step=11) < 1.0
