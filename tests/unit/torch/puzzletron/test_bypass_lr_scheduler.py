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


@pytest.mark.parametrize(
    ("warmup_steps", "lr_decay_steps", "learning_rate"),
    [
        (10, 10, 0.5),
        (20, 10, 0.7),
    ],
)
def test_degenerate_budget_returns_base_lr(warmup_steps, lr_decay_steps, learning_rate):
    """When ``lr_decay_steps <= warmup_steps`` (tiny test budgets), the scheduler
    must short-circuit to ``learning_rate`` rather than divide by zero."""
    cfg = _make_cfg(
        warmup_steps=warmup_steps,
        lr_decay_steps=lr_decay_steps,
        learning_rate=learning_rate,
    )
    assert _get_lr(cfg, step=0) == learning_rate
    assert _get_lr(cfg, step=99) == learning_rate


def test_warmup_linear_ramp():
    cfg = _make_cfg(warmup_steps=10, lr_decay_steps=100, learning_rate=1.0)
    assert _get_lr(cfg, step=0) == pytest.approx(0.0)
    assert _get_lr(cfg, step=5) == pytest.approx(0.5)
    assert _get_lr(cfg, step=10) == pytest.approx(1.0)


def test_cosine_starts_decaying_immediately_after_warmup():
    """At ``step == warmup_steps + 1`` the cosine branch is entered with
    ``decay_ratio = 1/(D-W)`` — already a small step below base LR, not a
    duplicate plateau at base LR. This is the boundary the previous formula
    got wrong (it used ``step - W - 1`` and gave ``decay_ratio == 0`` here)."""
    cfg = _make_cfg(warmup_steps=10, lr_decay_steps=20, learning_rate=1.0, min_lr=0.0)
    # decay_ratio = (11 - 10) / 10 = 0.1
    expected = 0.5 * (1.0 + math.cos(math.pi * 0.1))
    assert _get_lr(cfg, step=11) == pytest.approx(expected)
    # Strictly below base LR — the cosine has begun.
    assert _get_lr(cfg, step=11) < 1.0


def test_cosine_endpoint_returns_min_lr():
    """At ``step == lr_decay_steps`` the cosine branch reaches its endpoint:
    ``decay_ratio == 1`` → ``coeff == 0`` → returns ``min_lr`` exactly. The
    post-decay clamp at ``step == lr_decay_steps + 1`` is then a no-op
    continuation, not a correction for an off-by-one."""
    cfg = _make_cfg(warmup_steps=10, lr_decay_steps=20, learning_rate=1.0, min_lr=0.1)
    assert _get_lr(cfg, step=20) == pytest.approx(0.1)


def test_cosine_midpoint_is_halfway():
    """At the cosine midpoint, ``coeff == 0.5`` → returns ``(lr + min_lr) / 2``."""
    cfg = _make_cfg(warmup_steps=10, lr_decay_steps=20, learning_rate=1.0, min_lr=0.0)
    # Midpoint of the post-warmup window: step such that decay_ratio == 0.5.
    # decay_ratio = (step - 10) / (20 - 10) → step = 15 gives ratio 0.5.
    expected_coeff = 0.5 * (1.0 + math.cos(math.pi * 0.5))
    assert _get_lr(cfg, step=15) == pytest.approx(expected_coeff)


def test_post_decay_clamps_to_min_lr():
    """``step > lr_decay_steps`` always returns ``min_lr`` exactly."""
    cfg = _make_cfg(warmup_steps=10, lr_decay_steps=20, learning_rate=1.0, min_lr=0.1)
    assert _get_lr(cfg, step=21) == 0.1
    assert _get_lr(cfg, step=1000) == 0.1
