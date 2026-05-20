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

"""Regression tests for fastgen pred_type conversion, timestep rescaling, and EMA dtype promotion.

Covers three round-1 fix guards:

1. ``fake_score_pred_type`` regression — under ``pred_type='flow'`` with
   ``fake_score_pred_type='x0'``, the fake-score DSM loss must operate on the
   ``x_0`` projection of the raw flow output, not on the raw flow tensor.
2. ``num_train_timesteps`` rescale — the pipeline must scale the RF
   ``t ∈ [0, 1]`` to ``num_train_timesteps * t`` before passing it to the
   transformer, when the knob is set.
3. EMA shadow dtype promotion — by default the shadow lives in ``float32``
   even when the live model is ``bfloat16``.
"""

from __future__ import annotations

import copy
from unittest import mock

import torch
import torch.nn.functional as F
from torch import nn

from modelopt.torch.fastgen import DMDConfig, DMDPipeline, EMAConfig, ExponentialMovingAverage
from modelopt.torch.fastgen.flow_matching import add_noise, pred_x0_from_flow
from modelopt.torch.fastgen.methods import dmd as dmd_module


class _ToyTransformer(nn.Module):
    """Minimal diffusers-shaped transformer. Output is linear in ``hidden_states`` and
    ignores ``timestep`` / ``encoder_hidden_states`` — keeps the expected-value
    reconstruction analytic."""

    def __init__(self, d: int) -> None:
        super().__init__()
        self.linear = nn.Linear(d, d, bias=False)

    def forward(self, hidden_states, timestep=None, encoder_hidden_states=None, **kwargs):
        return self.linear(hidden_states)


class _TimestepEchoModel(nn.Module):
    """Returns ``hidden_states + timestep`` broadcast — used to observe the rescale knob."""

    def forward(self, hidden_states, timestep, encoder_hidden_states=None, **kwargs):
        return hidden_states + timestep.view(-1, 1).to(hidden_states.dtype)


def _build_pipeline(
    d: int,
    *,
    pred_type: str = "flow",
    fake_score_pred_type: str | None = "x0",
    num_train_timesteps: int | None = None,
):
    torch.manual_seed(0)
    student = _ToyTransformer(d)
    teacher = _ToyTransformer(d)
    fake_score = copy.deepcopy(teacher)
    cfg = DMDConfig(
        pred_type=pred_type,
        fake_score_pred_type=fake_score_pred_type,
        num_train_timesteps=num_train_timesteps,
    )
    pipeline = DMDPipeline(student, teacher, fake_score, cfg)
    return pipeline, student, teacher, fake_score, cfg


def test_fake_score_dsm_matches_manual_flow_to_x0():
    """compute_fake_score_loss under ``(flow, x0)`` must equal the manual
    ``F.mse_loss(gen_data, pred_x0_from_flow(raw, x_t, t))`` reconstruction."""
    d, b = 4, 2
    pipeline, student, _teacher, fake_score, cfg = _build_pipeline(
        d, pred_type="flow", fake_score_pred_type="x0"
    )
    latents = torch.randn(b, d)
    noise = torch.randn(b, d)

    fixed_t = torch.full((b,), 0.5)
    fixed_eps = torch.randn(b, d)

    with (
        mock.patch.object(pipeline, "sample_timesteps", return_value=fixed_t),
        mock.patch.object(dmd_module.torch, "randn_like", return_value=fixed_eps),
    ):
        actual = pipeline.compute_fake_score_loss(latents, noise)["fake_score"]

    with torch.no_grad():
        max_t = cfg.sample_t_cfg.max_t
        t_student = torch.full((b,), max_t, dtype=torch.float32)
        input_student = noise * max_t
        gen_data = pred_x0_from_flow(
            student(hidden_states=input_student, timestep=t_student),
            input_student,
            t_student,
        )
        perturbed = add_noise(gen_data, fixed_eps, fixed_t)
        fake_raw = fake_score(hidden_states=perturbed, timestep=fixed_t)
        pred_x0 = pred_x0_from_flow(fake_raw, perturbed, fixed_t)
        expected = F.mse_loss(gen_data, pred_x0)

    assert torch.allclose(actual, expected, atol=1e-6), (
        f"compute_fake_score_loss={actual.item():.3e}, manual={expected.item():.3e}"
    )


def test_student_vsd_sees_x0_not_raw_flow():
    """compute_student_loss must feed ``vsd_loss`` the x0-converted flow output of the
    fake score (the prior bug was to forward the raw flow tensor instead)."""
    d, b = 4, 2
    pipeline, student, _teacher, fake_score, cfg = _build_pipeline(
        d, pred_type="flow", fake_score_pred_type="x0"
    )
    latents = torch.randn(b, d)
    noise = torch.randn(b, d)
    fixed_t = torch.full((b,), 0.5)
    fixed_eps = torch.randn(b, d)

    captured: dict[str, torch.Tensor] = {}
    orig_vsd_loss = dmd_module.vsd_loss

    def spy(gen_data, teacher_x0, fake_score_x0, additional_scale=None):
        captured["fake_score_x0"] = fake_score_x0.detach().clone()
        return orig_vsd_loss(gen_data, teacher_x0, fake_score_x0, additional_scale)

    with (
        mock.patch.object(pipeline, "sample_timesteps", return_value=fixed_t),
        mock.patch.object(dmd_module.torch, "randn_like", return_value=fixed_eps),
        mock.patch.object(dmd_module, "vsd_loss", side_effect=spy),
    ):
        pipeline.compute_student_loss(latents, noise)

    with torch.no_grad():
        max_t = cfg.sample_t_cfg.max_t
        t_student = torch.full((b,), max_t, dtype=torch.float32)
        input_student = noise * max_t
        gen_data_expected = pred_x0_from_flow(
            student(hidden_states=input_student, timestep=t_student),
            input_student,
            t_student,
        )
        perturbed = add_noise(gen_data_expected, fixed_eps, fixed_t)
        fake_raw = fake_score(hidden_states=perturbed, timestep=fixed_t)
        fake_x0_expected = pred_x0_from_flow(fake_raw, perturbed, fixed_t)

    assert torch.allclose(captured["fake_score_x0"], fake_x0_expected, atol=1e-6)


def test_call_model_rescales_timestep_when_num_train_timesteps_set():
    """``num_train_timesteps=1000`` rescales ``t`` by 1000 inside ``_call_model``
    and casts it to ``hidden_states.dtype`` (matching FastGen's VaceWan wrapper);
    ``None`` leaves ``t`` untouched."""
    d, b = 3, 2
    model = _TimestepEchoModel()
    x = torch.zeros(b, d)
    t = torch.tensor([0.1, 0.7])

    pipe_scaled = DMDPipeline(
        _ToyTransformer(d),
        _ToyTransformer(d),
        _ToyTransformer(d),
        DMDConfig(pred_type="x0", num_train_timesteps=1000),
    )
    out_scaled = pipe_scaled._call_model(model, x, t)
    assert torch.allclose(out_scaled, x + (t * 1000.0).view(-1, 1))

    pipe_none = DMDPipeline(
        _ToyTransformer(d),
        _ToyTransformer(d),
        _ToyTransformer(d),
        DMDConfig(pred_type="x0", num_train_timesteps=None),
    )
    out_none = pipe_none._call_model(model, x, t)
    assert torch.allclose(out_none, x + t.view(-1, 1))

    # bf16 hidden_states: rescaled timestep must be cast to bf16 so the
    # addition inside the model returns a bf16 tensor (parity with FastGen's
    # ``.to(dtype=x_t.dtype)``). fp32 timestep + bf16 hidden_states without the
    # cast would either upcast the result or push dtype juggling into the model.
    x_bf16 = torch.zeros(b, d, dtype=torch.bfloat16)
    out_bf16 = pipe_scaled._call_model(model, x_bf16, t)
    assert out_bf16.dtype == torch.bfloat16
    assert torch.allclose(
        out_bf16,
        x_bf16 + (t * 1000.0).to(torch.bfloat16).view(-1, 1),
    )


def test_ema_shadow_dtype_promotion():
    """``EMAConfig.dtype='float32'`` on a bf16 student gives an fp32 shadow;
    ``dtype=None`` falls back to the live parameter's dtype."""
    torch.manual_seed(0)
    student_bf16 = nn.Linear(4, 4).to(torch.bfloat16)

    cfg_fp32 = EMAConfig(fsdp2=False, dtype="float32")
    ema_fp32 = ExponentialMovingAverage(student_bf16, cfg_fp32)
    for name, shadow in ema_fp32.state_dict().items():
        assert shadow.dtype == torch.float32, f"{name}: expected float32, got {shadow.dtype}"

    cfg_none = EMAConfig(fsdp2=False, dtype=None)
    ema_none = ExponentialMovingAverage(student_bf16, cfg_none)
    for name, shadow in ema_none.state_dict().items():
        assert shadow.dtype == torch.bfloat16, f"{name}: expected bfloat16, got {shadow.dtype}"
