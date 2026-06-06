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

"""DMD math parity tests against the FastGen reference implementation.

Ports checklist §2 bullets — flow-matching identities, dsm/vsd/gan losses, CFG
formula, and the fake-score flow→x0→DSM conversion chain. The FastGen reference
math is inlined verbatim from
``source/FastGen/fastgen/methods/common_loss.py`` and
``source/FastGen/fastgen/methods/distribution_matching/dmd2.py`` so the test is
hermetic — no FastGen import required.

Numerical tolerance is ``1e-6`` for floating-point losses; pack/permute paths
(``add_noise``, ``pred_x0_from_flow``, ``x0_to_flow``, CFG) use ``torch.equal``
because both implementations route through fp64 intermediates with the same
operation order.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from modelopt.torch.fastgen.config import DMDConfig, SampleTimestepConfig
from modelopt.torch.fastgen.flow_matching import (
    add_noise,
    pred_x0_from_flow,
    rf_alpha,
    rf_sigma,
    x0_to_flow,
)
from modelopt.torch.fastgen.losses import dsm_loss, gan_disc_loss, gan_gen_loss, vsd_loss
from modelopt.torch.fastgen.methods import dmd as dmd_module
from modelopt.torch.fastgen.methods.dmd import DMDPipeline
from modelopt.torch.fastgen.utils import classifier_free_guidance

# ---------------------------------------------------------------------------- #
# FastGen reference impls (math only) — inlined verbatim                        #
# ---------------------------------------------------------------------------- #


def _expand_like(t: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    while t.ndim < target.ndim:
        t = t.unsqueeze(-1)
    return t


def _fastgen_forward_process(x: torch.Tensor, eps: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """``BaseNoiseSchedule.forward_process`` with RF alpha/sigma inlined."""
    original_dtype = x.dtype
    t64 = t.to(torch.float64)
    x64 = x.to(torch.float64)
    eps64 = eps.to(torch.float64)
    alpha_t = _expand_like(1.0 - t64, x64)
    sigma_t = _expand_like(t64, eps64)
    return (x64 * alpha_t + eps64 * sigma_t).to(original_dtype)


def _fastgen_dsm(pred_type, net_pred, *, x0=None, eps=None, t=None):
    """common_loss.py:12-60. RF scheduler so alpha=1-t, sigma=t for 'v'."""
    if pred_type == "x0":
        return F.mse_loss(x0, net_pred, reduction="mean")
    if pred_type == "eps":
        return F.mse_loss(eps, net_pred, reduction="mean")
    if pred_type == "v":
        alpha_t = _expand_like((1.0 - t).to(dtype=x0.dtype), x0).to(device=x0.device)
        sigma_t = _expand_like(t.to(dtype=x0.dtype), x0).to(device=x0.device)
        v = alpha_t * eps - sigma_t * x0
        return F.mse_loss(v, net_pred, reduction="mean")
    if pred_type == "flow":
        return F.mse_loss(eps - x0, net_pred, reduction="mean")
    raise NotImplementedError(pred_type)


def _fastgen_vsd(gen_data, teacher_x0, fake_score_x0):
    """common_loss.py:63-103."""
    dims = tuple(range(1, teacher_x0.ndim))
    with torch.no_grad():
        original_dtype = gen_data.dtype
        gen_fp32 = gen_data.float()
        teacher_fp32 = teacher_x0.float()
        diff_abs_mean = (gen_fp32 - teacher_fp32).abs().mean(dim=dims, keepdim=True)
        w = (1 / (diff_abs_mean + 1e-6)).to(dtype=original_dtype)
        pseudo_target = gen_data - (fake_score_x0 - teacher_x0) * w
    return 0.5 * F.mse_loss(gen_data, pseudo_target, reduction="mean")


def _fastgen_cfg(cond, uncond, scale):
    """dmd2.py:184 — ``teacher_x0 + (scale - 1) * (teacher_x0 - teacher_x0_neg)``."""
    return cond + (scale - 1) * (cond - uncond)


def _fastgen_gan_gen(fake_logits):
    return F.softplus(-fake_logits).mean()


def _fastgen_gan_disc(real_logits, fake_logits):
    return F.softplus(fake_logits).mean() + F.softplus(-real_logits).mean()


def _fastgen_r1(real_logits, perturbed_real_logits):
    """Approximate R1: MSE between logits on clean vs perturbed real data."""
    return F.mse_loss(real_logits, perturbed_real_logits, reduction="mean")


# ---------------------------------------------------------------------------- #
# §2.1 — RF forward process                                                    #
# ---------------------------------------------------------------------------- #


def test_rf_forward_matches_fastgen():
    torch.manual_seed(0)
    x0 = torch.randn(2, 16, 8, 8, dtype=torch.float32)
    eps = torch.randn_like(x0)
    t = torch.tensor([0.1, 0.7], dtype=torch.float64)
    assert torch.equal(add_noise(x0, eps, t), _fastgen_forward_process(x0, eps, t))


# ---------------------------------------------------------------------------- #
# §2.2 / §2.3 — student input for single-step and multi-step                   #
# ---------------------------------------------------------------------------- #


def _student_input_pipeline(*, sample_steps: int, t_list=None) -> DMDPipeline:
    cfg = DMDConfig(
        pred_type="flow",
        num_train_timesteps=None,
        student_sample_steps=sample_steps,
        student_update_freq=5,
        sample_t_cfg=SampleTimestepConfig(
            time_dist_type="shifted",
            min_t=0.001,
            max_t=0.999,
            shift=5.0,
            t_list=t_list,
        ),
    )
    return DMDPipeline(
        student=nn.Identity(), teacher=nn.Identity(), fake_score=nn.Identity(), config=cfg
    )


def test_build_student_input_single_step_matches_max_t_noise():
    pipe = _student_input_pipeline(sample_steps=1)
    latents = torch.randn(2, 16, 8, 8, dtype=torch.float32)
    noise = torch.randn_like(latents)
    input_student, t_student = pipe._build_student_input(latents, noise)
    max_t = float(pipe.config.sample_t_cfg.max_t)
    expected_input = (noise.to(torch.float64) * max_t).to(noise.dtype)
    expected_t = torch.full((2,), max_t, dtype=torch.float32)
    assert torch.equal(input_student, expected_input)
    assert torch.equal(t_student, expected_t)


def test_build_student_input_multi_step_uses_t_list_prefix_and_add_noise():
    pipe = _student_input_pipeline(sample_steps=2, t_list=[0.999, 0.5, 0.0])
    torch.manual_seed(0)
    latents = torch.randn(8, 16, 4, 4, dtype=torch.float32)
    noise = torch.randn_like(latents)
    input_student, t_student = pipe._build_student_input(latents, noise)
    allowed = list(pipe.config.sample_t_cfg.t_list[:-1])
    actual = t_student.detach().cpu().tolist()
    # fp32 round-trip — compare with tolerance, not set membership.
    assert all(any(abs(v - a) < 1e-5 for a in allowed) for v in actual)
    assert torch.equal(input_student, add_noise(latents, noise, t_student))


class _ZeroFlow(nn.Module):
    def forward(self, hidden_states, timestep, encoder_hidden_states=None, **_kwargs):
        return torch.zeros_like(hidden_states)


def _backward_simulation_pipeline(*, sample_type: str = "ode") -> DMDPipeline:
    cfg = DMDConfig(
        pred_type="flow",
        num_train_timesteps=None,
        student_sample_steps=2,
        student_sample_type=sample_type,
        backward_simulation=True,
        sample_t_cfg=SampleTimestepConfig(
            time_dist_type="uniform",
            min_t=0.001,
            max_t=0.9,
            t_list=[0.9, 0.5, 0.0],
        ),
    )
    model = _ZeroFlow()
    return DMDPipeline(student=model, teacher=model, fake_score=model, config=cfg)


def test_build_student_input_backward_simulation_uses_generated_distribution(monkeypatch):
    def _fixed_randint(low, high, size, *, device=None, dtype=None, **_kwargs):
        assert low == 0
        assert high == 2
        return torch.ones(size, device=device, dtype=dtype or torch.long)

    def _fixed_randn_like(x, *args, **kwargs):
        return torch.full_like(x, 2.0)

    monkeypatch.setattr(torch, "randint", _fixed_randint)
    monkeypatch.setattr(torch, "randn_like", _fixed_randn_like)

    pipe = _backward_simulation_pipeline()
    latents = torch.zeros(2, 16, 4, 4, dtype=torch.float32)
    final_noise = torch.full_like(latents, 3.0)
    input_student, t_student = pipe._build_student_input(latents, final_noise)

    expected_t = torch.full((2,), 0.5, dtype=torch.float32)
    generated_x0 = torch.full_like(latents, 2.0 * 0.9)
    expected_input = add_noise(generated_x0, final_noise, expected_t)
    assert torch.equal(t_student, expected_t)
    assert torch.equal(input_student, expected_input)
    assert not torch.equal(input_student, add_noise(latents, final_noise, expected_t))


def test_backward_simulation_selected_rung_is_broadcast(monkeypatch):
    calls = []

    def _fixed_randint(low, high, size, *, device=None, dtype=None, **_kwargs):
        assert low == 0
        assert high == 2
        return torch.ones(size, device=device, dtype=dtype or torch.long)

    def _broadcast_to_first_rung(tensor, src):
        assert src == 0
        calls.append(tensor.clone())
        tensor.zero_()

    monkeypatch.setattr(torch, "randint", _fixed_randint)
    monkeypatch.setattr(dmd_module.dist, "is_available", lambda: True)
    monkeypatch.setattr(dmd_module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dmd_module.dist, "broadcast", _broadcast_to_first_rung)

    pipe = _backward_simulation_pipeline()
    latents = torch.zeros(2, 16, 4, 4, dtype=torch.float32)
    final_noise = torch.full_like(latents, 3.0)
    input_student, t_student = pipe._build_student_input(latents, final_noise)

    expected_t = torch.full((2,), 0.9, dtype=torch.float32)
    expected_input = (final_noise.to(torch.float64) * 0.9).to(final_noise.dtype)
    assert len(calls) == 1
    assert torch.equal(calls[0], torch.ones(1, dtype=torch.long))
    assert torch.equal(t_student, expected_t)
    assert torch.equal(input_student, expected_input)


# ---------------------------------------------------------------------------- #
# §2.4 / §2.5 — flow ↔ x0 identities                                            #
# ---------------------------------------------------------------------------- #


def test_pred_x0_from_flow_matches_identity():
    torch.manual_seed(1)
    x_t = torch.randn(2, 16, 8, 8, dtype=torch.float32)
    flow = torch.randn_like(x_t)
    t = torch.tensor([0.3, 0.7], dtype=torch.float32)
    mo = pred_x0_from_flow(flow, x_t, t)
    t64 = _expand_like(t.to(torch.float64), x_t.to(torch.float64))
    ref = (x_t.to(torch.float64) - t64 * flow.to(torch.float64)).to(x_t.dtype)
    assert torch.equal(mo, ref)


def test_x0_to_flow_matches_identity():
    torch.manual_seed(2)
    x0 = torch.randn(2, 16, 8, 8, dtype=torch.float32)
    x_t = torch.randn_like(x0)
    t = torch.tensor([0.3, 0.7], dtype=torch.float32)
    mo = x0_to_flow(x0, x_t, t)
    t64 = _expand_like(t.to(torch.float64), x0.to(torch.float64))
    ref = ((x_t.to(torch.float64) - x0.to(torch.float64)) / t64.clamp_min(1e-6)).to(x0.dtype)
    assert torch.equal(mo, ref)


# ---------------------------------------------------------------------------- #
# §2.6 — dsm_loss for x0 / eps / flow / v                                       #
# ---------------------------------------------------------------------------- #


@pytest.mark.parametrize("pred_type", ["x0", "eps", "flow", "v"])
def test_dsm_loss_matches_fastgen(pred_type):
    torch.manual_seed(3)
    x0 = torch.randn(2, 16, 8, 8, dtype=torch.float32)
    eps = torch.randn_like(x0)
    t = torch.tensor([0.4, 0.6], dtype=torch.float32)
    net_pred = torch.randn_like(x0)
    kwargs = {"x0": x0, "eps": eps, "t": t}
    if pred_type == "v":
        kwargs["alpha_fn"] = rf_alpha
        kwargs["sigma_fn"] = rf_sigma
    mo = dsm_loss(pred_type, net_pred, **kwargs).item()
    fg = _fastgen_dsm(pred_type, net_pred, x0=x0, eps=eps, t=t).item()
    assert abs(mo - fg) < 1e-6


# ---------------------------------------------------------------------------- #
# §2.7 — vsd_loss                                                              #
# ---------------------------------------------------------------------------- #


def test_vsd_loss_matches_fastgen():
    torch.manual_seed(4)
    gen_data = torch.randn(2, 16, 8, 8, dtype=torch.float32, requires_grad=True)
    teacher_x0 = torch.randn_like(gen_data).detach()
    fake_score_x0 = torch.randn_like(gen_data).detach()
    mo = vsd_loss(gen_data, teacher_x0, fake_score_x0).item()
    fg = _fastgen_vsd(gen_data.detach(), teacher_x0, fake_score_x0).item()
    assert abs(mo - fg) < 1e-6


# ---------------------------------------------------------------------------- #
# §2.8 — fake-score DSM target: ModelOpt flow→x0→DSM matches FastGen direct DSM('x0') #
# ---------------------------------------------------------------------------- #


def test_fake_score_flow_to_x0_dsm_matches_fastgen():
    torch.manual_seed(5)
    x0_real = torch.randn(2, 16, 8, 8, dtype=torch.float32)
    eps = torch.randn_like(x0_real)
    t = torch.tensor([0.3, 0.7], dtype=torch.float32)
    x_t = add_noise(x0_real, eps, t)
    raw_flow = torch.randn_like(x0_real)
    x0_pred_modelopt = DMDPipeline._raw_to_x0(raw_flow, x_t, t, native_pred_type="flow")
    loss_modelopt = dsm_loss("x0", x0_pred_modelopt, x0=x0_real).item()

    # FastGen "direct" reference: x0 = x_t - t * flow.
    t64 = _expand_like(t.to(torch.float64), x_t.to(torch.float64))
    x0_pred_ref = (x_t.to(torch.float64) - t64 * raw_flow.to(torch.float64)).to(x0_real.dtype)
    loss_fastgen = _fastgen_dsm("x0", x0_pred_ref, x0=x0_real).item()
    assert abs(loss_modelopt - loss_fastgen) < 1e-6


# ---------------------------------------------------------------------------- #
# §2.9 — classifier-free guidance                                              #
# ---------------------------------------------------------------------------- #


def test_classifier_free_guidance_matches_fastgen():
    torch.manual_seed(6)
    cond = torch.randn(2, 16, 8, 8, dtype=torch.float32)
    uncond = torch.randn_like(cond)
    assert torch.equal(classifier_free_guidance(cond, uncond, 4.0), _fastgen_cfg(cond, uncond, 4.0))


class _RecordingFlow(nn.Module):
    def __init__(self):
        super().__init__()
        self.masks: list[torch.Tensor | None] = []

    def forward(self, hidden_states, timestep, encoder_hidden_states=None, **kwargs):
        mask = kwargs.get("encoder_hidden_states_mask")
        self.masks.append(mask.detach().clone() if torch.is_tensor(mask) else None)
        return torch.zeros_like(hidden_states)


def test_compute_student_loss_uses_separate_negative_cfg_mask():
    cfg = DMDConfig(
        pred_type="flow",
        num_train_timesteps=None,
        student_sample_steps=1,
        guidance_scale=4.0,
        sample_t_cfg=SampleTimestepConfig(time_dist_type="uniform", min_t=0.001, max_t=0.999),
    )
    student = _RecordingFlow()
    teacher = _RecordingFlow()
    fake_score = _RecordingFlow()
    pipe = DMDPipeline(student=student, teacher=teacher, fake_score=fake_score, config=cfg)

    torch.manual_seed(7)
    latents = torch.randn(2, 16, 4, 4)
    noise = torch.randn_like(latents)
    text = torch.randn(2, 8, 4)
    neg_text = torch.randn(2, 3, 4)
    text_mask = torch.ones(2, 8, dtype=torch.long)
    neg_mask = torch.ones(2, 3, dtype=torch.long)

    pipe.compute_student_loss(
        latents,
        noise,
        encoder_hidden_states=text,
        encoder_hidden_states_mask=text_mask,
        negative_encoder_hidden_states=neg_text,
        negative_encoder_hidden_states_mask=neg_mask,
    )

    assert torch.equal(teacher.masks[0], text_mask)
    assert torch.equal(teacher.masks[1], neg_mask)


# ---------------------------------------------------------------------------- #
# §2.10 — GAN gen/disc/R1 losses                                               #
# ---------------------------------------------------------------------------- #


def test_gan_gen_loss_matches_fastgen():
    torch.manual_seed(7)
    fake_logits = torch.randn(8, 1)
    assert abs(gan_gen_loss(fake_logits).item() - _fastgen_gan_gen(fake_logits).item()) < 1e-6


def test_gan_disc_loss_matches_fastgen():
    torch.manual_seed(7)
    fake_logits = torch.randn(8, 1)
    real_logits = torch.randn(8, 1)
    assert (
        abs(
            gan_disc_loss(real_logits, fake_logits).item()
            - _fastgen_gan_disc(real_logits, fake_logits).item()
        )
        < 1e-6
    )


def test_r1_loss_matches_fastgen():
    try:
        from modelopt.torch.fastgen.losses import r1_loss
    except ImportError:
        pytest.skip("modelopt.torch.fastgen.losses.r1_loss not present")
    torch.manual_seed(7)
    real_logits = torch.randn(8, 1)
    perturbed = real_logits + 0.01 * torch.randn_like(real_logits)
    assert (
        abs(r1_loss(real_logits, perturbed).item() - _fastgen_r1(real_logits, perturbed).item())
        < 1e-6
    )
