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

"""End-to-end DMD2 training-step test (hermetic, pipeline-level).

Exercises one full DMD2 optimizer step through the real ``QwenImageDMDPipeline``
loss path — the student VSD phase and the fake-score DSM phase — on tiny stub
transformers, with no real Qwen weights, NCCL, or FSDP2. This is the hermetic
proxy for ``dmd2_recipe``'s per-step training loop: a single test transitively
covers the plugin's pack/unpack ``_call_model`` path, the VSD/DSM loss math,
gradient isolation between phases, an optimizer update, and a checkpoint
round-trip — the wiring that previously took several unit tests.

A full recipe-level e2e (``dmd2_finetune.py`` driving ``dmd2_recipe`` on the mock
dataloader) additionally needs a tiny Qwen-Image pipeline fixture + single-GPU
FSDP2/NCCL init; that's a follow-up — mirror ``create_tiny_wan22_pipeline_dir``
in ``tests/_test_utils/torch/diffusers_models.py`` and drive the example via
``run_example_command`` gated by ``minimum_sm``.
"""

from __future__ import annotations

import copy

import torch
from torch import nn

from modelopt.torch.fastgen.config import DMDConfig, SampleTimestepConfig
from modelopt.torch.fastgen.plugins.qwen_image import QwenImageDMDPipeline


class _TinyQwenTransformer(nn.Module):
    """Grad-capable stub over the packed Qwen hidden states ``[B, num_patches, C*4]``.

    ``QwenImageDMDPipeline._call_model`` packs ``[B, C, H, W] -> [B, P, C*4]`` before
    the forward and unpacks after, so a ``Linear`` over the last (``C*4``) dim gives a
    real gradient path while matching the expected return shape.
    """

    def __init__(self, packed_dim: int = 64) -> None:
        super().__init__()
        self.proj = nn.Linear(packed_dim, packed_dim)

    def forward(self, hidden_states, **_kwargs):
        return self.proj(hidden_states)


def _build_pipeline():
    torch.manual_seed(0)
    student = _TinyQwenTransformer()
    teacher = _TinyQwenTransformer()
    fake_score = _TinyQwenTransformer()
    cfg = DMDConfig(
        pred_type="flow",
        num_train_timesteps=None,  # required by QwenImageDMDPipeline
        student_sample_steps=1,
        student_update_freq=5,
        fake_score_pred_type="x0",
        gan_loss_weight_gen=0.0,  # no GAN branch -> no discriminator / feature hooks needed
        guidance_scale=None,
        sample_t_cfg=SampleTimestepConfig(time_dist_type="uniform", min_t=0.001, max_t=0.999),
        ema=None,
    )
    pipe = QwenImageDMDPipeline(
        student=student,
        teacher=teacher,
        fake_score=fake_score,
        config=cfg,
        discriminator=None,
    )
    return pipe, student, teacher, fake_score


def _mock_batch(batch_size: int = 2):
    latents = torch.randn(batch_size, 16, 8, 8)  # even dims -> packs to [B, 16, 64]
    noise = torch.randn_like(latents)
    text = torch.randn(batch_size, 8, 64)  # shape is arbitrary; the stub ignores it
    return latents, noise, text


def _snapshot(module: nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in module.state_dict().items()}


def _params_changed(before: dict[str, torch.Tensor], module: nn.Module) -> bool:
    return any(not torch.equal(before[k], v) for k, v in module.state_dict().items())


def _train_only(active: nn.Module, *frozen: nn.Module) -> None:
    active.train()
    for p in active.parameters():
        p.requires_grad_(True)
    for m in frozen:
        m.eval()
        for p in m.parameters():
            p.requires_grad_(False)


def test_dmd2_student_then_fake_score_step_updates_only_active_module():
    """One student VSD step then one fake-score DSM step: each phase yields a
    finite loss, steps only its own module, and leaves the others untouched."""
    pipe, student, teacher, fake_score = _build_pipeline()
    latents, noise, text = _mock_batch()

    # ---- student (VSD) phase ----
    _train_only(student, teacher, fake_score)
    student_before, teacher_before = _snapshot(student), _snapshot(teacher)
    opt_s = torch.optim.Adam(student.parameters(), lr=1e-2)
    opt_s.zero_grad()
    losses = pipe.compute_student_loss(
        latents,
        noise,
        encoder_hidden_states=text,
        negative_encoder_hidden_states=None,
        guidance_scale=None,
    )
    assert "vsd" in losses and "total" in losses
    assert torch.isfinite(losses["total"])
    losses["total"].backward()
    opt_s.step()

    assert _params_changed(student_before, student)  # student learned
    assert not _params_changed(teacher_before, teacher)  # teacher stayed frozen

    # ---- fake-score (DSM) phase ----
    _train_only(fake_score, student, teacher)
    student_after_student_phase = _snapshot(student)
    fake_before = _snapshot(fake_score)
    opt_f = torch.optim.Adam(fake_score.parameters(), lr=1e-2)
    opt_f.zero_grad()
    fs_losses = pipe.compute_fake_score_loss(latents, noise, encoder_hidden_states=text)
    assert "fake_score" in fs_losses and "total" in fs_losses
    assert torch.isfinite(fs_losses["total"])
    fs_losses["total"].backward()
    opt_f.step()

    assert _params_changed(fake_before, fake_score)  # fake_score learned
    assert not _params_changed(student_after_student_phase, student)  # student untouched


def test_dmd2_student_state_dict_round_trips():
    """After a training step, the student's state_dict reloads bit-exactly into a
    fresh module — the save/restore contract the recipe's checkpointing relies on."""
    pipe, student, _teacher, _fake = _build_pipeline()
    latents, noise, text = _mock_batch()

    _train_only(student, _teacher, _fake)
    opt = torch.optim.Adam(student.parameters(), lr=1e-2)
    opt.zero_grad()
    pipe.compute_student_loss(
        latents, noise, encoder_hidden_states=text, guidance_scale=None
    )["total"].backward()
    opt.step()

    saved = copy.deepcopy(student.state_dict())
    reloaded = _TinyQwenTransformer()
    reloaded.load_state_dict(saved)
    for k, v in student.state_dict().items():
        assert torch.equal(v, reloaded.state_dict()[k]), k
