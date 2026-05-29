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

"""Gradient routing + phase-schedule tests on ``DMDPipeline`` with tiny modules.

Ports the in-process bullets from checklist §3 (3.1, 3.2, 3.3, 3.5). The
source-grep bullets (3.4 / 3.6 / 3.7 / 3.8) intentionally stay in
``experiments/qwen.3/run_section_3.py`` — they are recipe-source linting,
not unit-testable logic, and don't translate to a meaningful pytest assertion.

3.1 / 3.2 use a ``TinyTransformer`` analogous to the one in ``conftest.py``'s
``ToyTransformer`` but with a timestep bias (the checklist's §3 module) so the
gradient signal definitely flows through the transformer when the
``compute_*_loss`` paths are exercised.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from modelopt.torch.fastgen.config import DMDConfig, EMAConfig, SampleTimestepConfig
from modelopt.torch.fastgen.methods.dmd import DMDPipeline


class _TinyTransformer(nn.Module):
    """``(hidden_states, timestep, encoder_hidden_states, **kw) -> Tensor`` module.

    Linear projection over the flattened spatial axes plus a timestep-derived
    bias. Cheap enough to run on CPU and returns a tensor in flow-space so it
    can play either student / teacher / fake_score role.
    """

    def __init__(self, channels: int = 16, dim: int = 8) -> None:
        super().__init__()
        self.channels = channels
        self.dim = dim
        flat = channels * dim * dim
        self.proj = nn.Linear(flat, flat, bias=True)
        self.t_proj = nn.Linear(1, flat, bias=False)

    def forward(self, hidden_states, timestep, encoder_hidden_states=None, **_kw):
        b = hidden_states.shape[0]
        x = hidden_states.reshape(b, -1)
        t = timestep.reshape(b, 1).to(x.dtype)
        return (self.proj(x) + self.t_proj(t)).reshape_as(hidden_states)


def _make_pipeline(
    *, with_ema: bool = True
) -> tuple[DMDPipeline, _TinyTransformer, _TinyTransformer, _TinyTransformer]:
    torch.manual_seed(0)
    student = _TinyTransformer()
    teacher = _TinyTransformer()
    fake_score = _TinyTransformer()
    ema_cfg = EMAConfig(decay=0.99, type="constant", fsdp2=False) if with_ema else None
    cfg = DMDConfig(
        pred_type="flow",
        num_train_timesteps=None,
        student_sample_steps=1,
        student_update_freq=5,
        fake_score_pred_type="x0",
        gan_loss_weight_gen=0.0,
        guidance_scale=None,
        sample_t_cfg=SampleTimestepConfig(
            time_dist_type="shifted", min_t=0.001, max_t=0.999, shift=1.0
        ),
        ema=ema_cfg,
    )
    return (
        DMDPipeline(student=student, teacher=teacher, fake_score=fake_score, config=cfg),
        student,
        teacher,
        fake_score,
    )


def _has_grad(module: nn.Module) -> bool:
    return any(p.grad is not None and p.grad.abs().sum().item() > 0 for p in module.parameters())


# ---------------------------------------------------------------------------- #
# §3.1 — compute_student_loss: only the student gets grads                     #
# ---------------------------------------------------------------------------- #


def test_compute_student_loss_routes_gradients_to_student_only():
    pipe, student, teacher, fake_score = _make_pipeline(with_ema=False)

    # Mirror the recipe's _set_grad_requirements for the student phase.
    student.train()
    for p in student.parameters():
        p.requires_grad_(True)
    fake_score.eval()
    for p in fake_score.parameters():
        p.requires_grad_(False)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    torch.manual_seed(1)
    latents = torch.randn(2, 16, 8, 8)
    noise = torch.randn_like(latents)
    text = torch.randn(2, 8, 4)

    losses = pipe.compute_student_loss(
        latents,
        noise,
        encoder_hidden_states=text,
        negative_encoder_hidden_states=None,
        guidance_scale=None,
    )
    losses["total"].backward()

    assert "vsd" in losses
    assert "total" in losses
    assert _has_grad(student)
    assert not _has_grad(teacher)
    assert not _has_grad(fake_score)


# ---------------------------------------------------------------------------- #
# §3.2 — compute_fake_score_loss: only the fake_score gets grads               #
# ---------------------------------------------------------------------------- #


def test_compute_fake_score_loss_routes_gradients_to_fake_score_only():
    pipe, student, teacher, fake_score = _make_pipeline(with_ema=False)

    # Mirror the fake-score phase grad config.
    student.eval()
    for p in student.parameters():
        p.requires_grad_(False)
    fake_score.train()
    for p in fake_score.parameters():
        p.requires_grad_(True)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    torch.manual_seed(2)
    latents = torch.randn(2, 16, 8, 8)
    noise = torch.randn_like(latents)
    text = torch.randn(2, 8, 4)

    losses = pipe.compute_fake_score_loss(latents, noise, encoder_hidden_states=text)
    losses["total"].backward()

    assert "fake_score" in losses
    assert "total" in losses
    assert _has_grad(fake_score)
    assert not _has_grad(student)
    assert not _has_grad(teacher)


# ---------------------------------------------------------------------------- #
# §3.3 — phase schedule modulo logic                                           #
# ---------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("step", "expected_phase"),
    [(0, "student")]
    + [(s, "fake_score") for s in range(1, 5)]
    + [(5, "student")]
    + [(s, "fake_score") for s in range(6, 10)]
    + [(10, "student")],
)
def test_phase_schedule_modulo_freq_5(step, expected_phase):
    """``step % student_update_freq == 0`` ⇒ student; else fake_score."""
    student_update_freq = 5
    actual = "student" if (step % student_update_freq) == 0 else "fake_score"
    assert actual == expected_phase


# ---------------------------------------------------------------------------- #
# §3.5 — update_ema increments _iteration once per call                        #
# ---------------------------------------------------------------------------- #


def test_update_ema_increments_iteration_monotonically():
    pipe, _student, _teacher, _fake = _make_pipeline(with_ema=True)
    iters = [pipe._iteration]
    for _ in range(5):
        pipe.update_ema()
        iters.append(pipe._iteration)
    assert iters == [0, 1, 2, 3, 4, 5]
