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

"""Distribution Matching Distillation (DMD2) pipeline.

:class:`DMDPipeline` holds references to the student / teacher / fake-score / (optional)
discriminator and exposes the three loss-computation entry points that a training loop
calls from each update step:

- :meth:`DMDPipeline.compute_student_loss` — variational score-distillation loss plus an optional
  GAN generator term.
- :meth:`DMDPipeline.compute_fake_score_loss` — denoising score matching against the student's
  generated samples.
- :meth:`DMDPipeline.compute_discriminator_loss` — GAN discriminator loss plus an optional R1
  regularizer.

The pipeline does **not** own optimizers, schedulers, gradient toggles, or device placement.
Callers drive the alternation between student / fake-score / discriminator updates, toggle
``requires_grad``, and call the appropriate ``compute_*_loss`` each step.

Math is a close port of ``FastGen/fastgen/methods/distribution_matching/dmd2.py`` (lines
45-455). See the docstrings on the individual methods for line-level references.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from torch import nn

from ..ema import ExponentialMovingAverage
from ..flow_matching import (
    add_noise,
    pred_noise_to_pred_x0,
    pred_x0_from_flow,
    rf_alpha,
    rf_sigma,
    sample_from_t_list,
    x0_to_eps,
    x0_to_flow,
)
from ..losses import dsm_loss, gan_disc_loss, gan_gen_loss, r1_loss, vsd_loss
from ..pipeline import DistillationPipeline
from ..utils import classifier_free_guidance

if TYPE_CHECKING:
    from ..config import DMDConfig

__all__ = ["DMDPipeline"]


# ---------------------------------------------------------------------------- #
#  Feature capture helper (duck-typed so tests can bypass the wan22 plugin)    #
# ---------------------------------------------------------------------------- #


def _drain_if_hooked(module: nn.Module) -> list[torch.Tensor] | None:
    """Drain the feature-capture buffer on ``module`` if hooks are attached.

    Returns the captured tensors in insertion order (clearing the buffer in-place), or
    ``None`` when no hooks are installed. Non-raising by design so pipeline-internal
    call sites can drain unconditionally after every teacher forward — this prevents
    the buffer from growing across steps when hooks are attached but the GAN branch is
    disabled (e.g. an ablation). Callers that need the strict "did you forget to attach
    hooks?" failure mode should call :func:`_require_hooked` on the result, or use
    :func:`modelopt.torch.fastgen.plugins.wan22.pop_captured_features` directly.
    """
    captured = getattr(module, "_fastgen_captured", None)
    if captured is None:
        return None
    out = list(captured)
    captured.clear()
    return out


def _require_hooked(
    features: list[torch.Tensor] | None,
    *,
    which: str,
) -> list[torch.Tensor]:
    """Adapter that turns a ``None`` drain result into a clear ``RuntimeError``.

    Use at pipeline sites that *must* consume captured features (i.e. the GAN-enabled
    paths in ``compute_student_loss`` / ``compute_discriminator_loss``). Keeps the
    non-raising ``_drain_if_hooked`` primitive for the "drain-and-discard" sites.

    The message names the attribute the pipeline looks for
    (``teacher._fastgen_captured``) so a debugger can grep straight to the hook
    installation site.
    """
    if features is None:
        raise RuntimeError(
            f"Feature-capture hooks are required on the teacher ({which} branch): "
            "teacher._fastgen_captured is missing. Call "
            "modelopt.torch.fastgen.plugins.wan22.attach_feature_capture(teacher, ...) "
            "before running this loss."
        )
    return features


# ---------------------------------------------------------------------------- #
#  DMDPipeline                                                                 #
# ---------------------------------------------------------------------------- #


class DMDPipeline(DistillationPipeline):
    """DMD2 loss pipeline.

    Args:
        student: Trainable student module. Must be callable with ``(hidden_states, timestep,
            encoder_hidden_states=..., **kwargs)`` and return either a ``Tensor``, a
            ``(Tensor, ...)`` tuple (as diffusers returns with ``return_dict=False``), or an
            object with a ``.sample`` attribute.
        teacher: Frozen reference module with the same call signature. If ``discriminator``
            is provided, feature-capture hooks must be attached to ``teacher`` before
            calling ``compute_*_loss`` — see :func:`modelopt.torch.fastgen.plugins.wan22.attach_feature_capture`.
        fake_score: Trainable auxiliary module (same signature as teacher/student). Used to
            approximate the student's generated distribution for the VSD gradient.
        config: :class:`~modelopt.torch.fastgen.config.DMDConfig` with the hyperparameters.
        discriminator: Optional discriminator. Required when ``config.gan_loss_weight_gen > 0``.
            Must accept ``list[Tensor]`` (the captured teacher features) and return a 2D logit tensor.
    """

    def __init__(
        self,
        student: nn.Module,
        teacher: nn.Module,
        fake_score: nn.Module,
        config: DMDConfig,
        *,
        discriminator: nn.Module | None = None,
    ) -> None:
        """Wire up student / teacher / fake-score / discriminator and create the EMA tracker."""
        super().__init__(student, teacher, config)
        self.fake_score = fake_score
        self.discriminator = discriminator
        self._ema: ExponentialMovingAverage | None = (
            ExponentialMovingAverage(student, config.ema) if config.ema is not None else None
        )
        self._iteration = 0

        if config.gan_loss_weight_gen > 0 and discriminator is None:
            raise ValueError(
                "gan_loss_weight_gen > 0 requires a discriminator to be provided to DMDPipeline."
            )

    # Re-declare config at the class level so type checkers see ``DMDConfig`` here
    # even though the base class stores it as ``DistillationConfig``. At runtime the
    # attribute is set by :meth:`DistillationPipeline.__init__`.
    config: DMDConfig  # type: ignore[assignment]

    @property
    def ema(self) -> ExponentialMovingAverage | None:
        """Reference to the student EMA tracker, if configured."""
        return self._ema

    # ================================================================== #
    #  Model-call helpers                                                #
    # ================================================================== #

    def _call_model(
        self,
        model: nn.Module,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        **model_kwargs: Any,
    ) -> torch.Tensor:
        """Forward a diffusers-style transformer and return the raw prediction tensor.

        Assumes the target module accepts ``hidden_states`` / ``timestep`` /
        ``encoder_hidden_states`` as kwargs and returns one of:

        * a ``torch.Tensor`` (custom modules),
        * a ``tuple`` whose first element is the prediction (diffusers ``return_dict=False``),
        * an object with a ``.sample`` attribute (diffusers ``return_dict=True``).

        **Timestep convention.** ``timestep`` is passed verbatim to the model by default.
        If :attr:`DistillationConfig.num_train_timesteps` is set, the continuous RF time
        ``t ∈ [0, 1]`` is rescaled to ``num_train_timesteps * t`` before the call — which
        matches the diffusers training convention for Wan 2.2, SD3, Flux. Leave
        ``num_train_timesteps=None`` when the upstream model wrapper (e.g. a VaceWan-style
        module) already scales the timestep internally.

        Subclass and override this method for modules with non-diffusers signatures
        (positional-only args, alternate kwarg names) or bespoke timestep transforms.
        """
        call_kwargs: dict[str, Any] = dict(model_kwargs)
        call_kwargs["hidden_states"] = hidden_states
        if self.config.num_train_timesteps is not None:
            # Cast to match the hidden-state dtype, mirroring FastGen's VaceWan
            # wrapper (``noise_scheduler.rescale_t(t).to(dtype=x_t.dtype)``).
            timestep = (timestep * float(self.config.num_train_timesteps)).to(
                dtype=hidden_states.dtype
            )
        call_kwargs["timestep"] = timestep
        if encoder_hidden_states is not None:
            call_kwargs["encoder_hidden_states"] = encoder_hidden_states

        out = model(**call_kwargs)
        if isinstance(out, torch.Tensor):
            return out
        if isinstance(out, tuple):
            return out[0]
        if hasattr(out, "sample"):
            return out.sample
        raise TypeError(
            f"DMDPipeline._call_model could not extract a tensor from output of type "
            f"{type(out).__name__!r}. Override ``_call_model`` in a subclass to handle "
            f"custom module signatures."
        )

    @staticmethod
    def _raw_to_x0(
        raw: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
        *,
        native_pred_type: str,
    ) -> torch.Tensor:
        """Convert a raw model output in ``native_pred_type`` space to an ``x_0`` estimate.

        ``native_pred_type`` is the parameterization the module *actually* predicts
        (i.e. its architecture's native output), not the space a downstream loss
        wants to operate in. Under RF, ``flow`` and ``v`` are equivalent (both are
        ``eps - x_0``).
        """
        if native_pred_type == "x0":
            return raw
        if native_pred_type == "eps":
            return pred_noise_to_pred_x0(raw, x_t, t)
        if native_pred_type in ("flow", "v"):
            return pred_x0_from_flow(raw, x_t, t)
        raise ValueError(f"Unsupported native_pred_type={native_pred_type!r}")

    @staticmethod
    def _x0_to_raw(
        x0: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
        *,
        target_pred_type: str,
    ) -> torch.Tensor:
        """Inverse of :meth:`_raw_to_x0` — project an ``x_0`` estimate into ``target_pred_type`` space."""
        if target_pred_type == "x0":
            return x0
        if target_pred_type == "eps":
            return x0_to_eps(x0, x_t, t)
        if target_pred_type in ("flow", "v"):
            return x0_to_flow(x0, x_t, t)
        raise ValueError(f"Unsupported target_pred_type={target_pred_type!r}")

    def _convert_pred(
        self,
        raw: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
        *,
        from_pred_type: str,
        to_pred_type: str,
    ) -> torch.Tensor:
        """Project a prediction between parameterizations via the ``x_0`` hub.

        Used by :meth:`compute_fake_score_loss` to land the fake-score's raw
        output in the DSM loss's target space. Short-circuits to identity when
        both spaces agree.
        """
        if from_pred_type == to_pred_type:
            return raw
        x0 = self._raw_to_x0(raw, x_t, t, native_pred_type=from_pred_type)
        if to_pred_type == "x0":
            return x0
        return self._x0_to_raw(x0, x_t, t, target_pred_type=to_pred_type)

    def _predict_x0(
        self,
        model: nn.Module,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        *,
        native_pred_type: str | None = None,
        **model_kwargs: Any,
    ) -> torch.Tensor:
        """Forward ``model`` and return its ``x_0`` estimate.

        ``native_pred_type`` declares the module's **architectural** output
        parameterization — NOT any downstream loss's target space. In the DMD2
        setup the student / teacher / fake_score are arch-twins, so this defaults
        to :attr:`DistillationConfig.pred_type`; callers should only override it
        when wiring in a model whose architecture genuinely differs.
        """
        raw = self._call_model(
            model, hidden_states, timestep, encoder_hidden_states, **model_kwargs
        )
        native_pred_type = native_pred_type or self.config.pred_type
        return self._raw_to_x0(raw, hidden_states, timestep, native_pred_type=native_pred_type)

    # ================================================================== #
    #  Noise / timestep sampling                                         #
    # ================================================================== #

    def _build_student_input(
        self,
        latents: torch.Tensor,
        noise: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Construct ``(input_student, t_student)`` for the student forward pass.

        - ``student_sample_steps == 1``: Use the maximum training timestep and set the
          student's input to ``sigma(max_t) * noise = max_t * noise`` (RF).
        - ``student_sample_steps > 1``: Sample a random intermediate timestep from
          ``config.sample_t_cfg.t_list`` and noise the real latents up to that timestep.
        """
        cfg = self.config
        batch_size = latents.shape[0]
        device = latents.device

        if cfg.student_sample_steps == 1:
            max_t = cfg.sample_t_cfg.max_t
            t_student = torch.full((batch_size,), max_t, device=device, dtype=torch.float32)
            # Under RF, ``sigma(max_t) = max_t``; scalar-multiply is fine (no per-sample shape needed).
            input_student = noise * max_t
        else:
            if cfg.sample_t_cfg.t_list is None:
                raise ValueError(
                    "student_sample_steps > 1 requires DMDConfig.sample_t_cfg.t_list to be set."
                )
            t_student = sample_from_t_list(
                batch_size,
                cfg.sample_t_cfg.t_list,
                device=device,
                dtype=torch.float32,
            )
            input_student = add_noise(latents, noise, t_student)
        return input_student, t_student

    # ================================================================== #
    #  Public API                                                        #
    # ================================================================== #

    def compute_student_loss(
        self,
        latents: torch.Tensor,
        noise: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        *,
        negative_encoder_hidden_states: torch.Tensor | None = None,
        guidance_scale: float | None = None,
        **model_kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """Compute the student update losses.

        The returned dict always contains ``"vsd"`` and ``"total"``. When the GAN branch
        is enabled (``discriminator is not None`` and ``config.gan_loss_weight_gen > 0``),
        ``"gan_gen"`` is also present.

        Gradient flow summary:

        - VSD gradient: flows through ``student`` only (``teacher_x0`` is detached,
          ``fake_score_x0`` is computed under ``torch.no_grad()``).
        - GAN generator gradient: flows through ``student`` via the feature-capture
          hooks on the teacher. The teacher forward is therefore **not** wrapped in
          ``torch.no_grad()`` when the GAN branch is active.

        Args:
            latents: Real clean-data latents ``x_0``. Used only when
                ``student_sample_steps > 1`` to construct ``input_student``.
            noise: Pure Gaussian noise tensor matching ``latents`` in shape/dtype.
            encoder_hidden_states: Positive conditioning passed unchanged to all three
                models.
            negative_encoder_hidden_states: Negative conditioning used by classifier-free
                guidance. Required when ``guidance_scale`` (or :attr:`DMDConfig.guidance_scale`)
                is not ``None``.
            guidance_scale: Overrides :attr:`DMDConfig.guidance_scale` for this call.
                ``None`` keeps the config-level value.
            **model_kwargs: Forwarded verbatim to ``student``, ``teacher``, and ``fake_score``.

        Returns:
            Dictionary with keys ``"vsd"``, ``"total"``, and optionally ``"gan_gen"``.
        """
        cfg = self.config
        batch_size = latents.shape[0]
        device = latents.device
        gan_enabled = self.discriminator is not None and cfg.gan_loss_weight_gen > 0

        # 1. Student input.
        input_student, t_student = self._build_student_input(latents, noise)

        # 2. Student forward -> x0.
        gen_data = self._predict_x0(
            self.student,
            input_student,
            t_student,
            encoder_hidden_states=encoder_hidden_states,
            **model_kwargs,
        )

        # 3. Sample perturbation timesteps and noise, perturb gen_data.
        t = self.sample_timesteps(batch_size, device=device, dtype=torch.float32)
        eps = torch.randn_like(latents)
        perturbed = add_noise(gen_data, eps, t)

        # 4. Fake score prediction (no grad).
        #
        # VSD always operates in x_0 space, regardless of ``fake_score_pred_type``
        # (which controls the DSM loss space on the fake-score side — see
        # :meth:`compute_fake_score_loss`). The fake_score's architecture matches the
        # student's in the DMD2 setup, so its native output parameterization is
        # ``cfg.pred_type``; ``_predict_x0`` converts from that to x_0 automatically.
        with torch.no_grad():
            fake_score_x0 = self._predict_x0(
                self.fake_score,
                perturbed,
                t,
                encoder_hidden_states=encoder_hidden_states,
                **model_kwargs,
            )

        # 5. Teacher forward.
        fake_feat: list[torch.Tensor] | None = None
        if gan_enabled:
            # Grad must flow through the teacher for the GAN generator term, since the
            # captured features depend on perturbed -> gen_data -> student weights.
            teacher_x0 = self._predict_x0(
                self.teacher,
                perturbed,
                t,
                encoder_hidden_states=encoder_hidden_states,
                **model_kwargs,
            )
            fake_feat = _require_hooked(_drain_if_hooked(self.teacher), which="student-fake")
        else:
            with torch.no_grad():
                teacher_x0 = self._predict_x0(
                    self.teacher,
                    perturbed,
                    t,
                    encoder_hidden_states=encoder_hidden_states,
                    **model_kwargs,
                )
            # Drain any hooks attached but not consumed (e.g. hooks left over from a
            # previous GAN-enabled ablation). No-op when hooks aren't installed.
            _ = _drain_if_hooked(self.teacher)

        # 6. Classifier-free guidance (applied to teacher_x0 prior to detach).
        effective_scale = guidance_scale if guidance_scale is not None else cfg.guidance_scale
        if effective_scale is not None:
            if negative_encoder_hidden_states is None:
                raise ValueError(
                    "guidance_scale is set but negative_encoder_hidden_states was not provided."
                )
            with torch.no_grad():
                teacher_x0_neg = self._predict_x0(
                    self.teacher,
                    perturbed,
                    t,
                    encoder_hidden_states=negative_encoder_hidden_states,
                    **model_kwargs,
                )
            # Negative-branch features are never used for GAN — drain unconditionally so
            # the buffer stays clean for subsequent calls.
            _ = _drain_if_hooked(self.teacher)
            teacher_x0 = classifier_free_guidance(teacher_x0, teacher_x0_neg, effective_scale)

        teacher_x0 = teacher_x0.detach()

        # 7. Losses.
        vsd = vsd_loss(gen_data, teacher_x0, fake_score_x0)

        if gan_enabled:
            # ``fake_feat`` is guaranteed non-None by ``_require_hooked`` above.
            gan_gen = gan_gen_loss(self.discriminator(fake_feat))
            total = vsd + cfg.gan_loss_weight_gen * gan_gen
            return {"vsd": vsd, "gan_gen": gan_gen, "total": total}

        return {"vsd": vsd, "total": vsd}

    def compute_fake_score_loss(
        self,
        latents: torch.Tensor,
        noise: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        **model_kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """Compute the fake-score (auxiliary) update loss.

        The fake score is trained with denoising score matching against the student's
        generated samples. The student forward is wrapped in ``torch.no_grad()`` — the
        gradient here is w.r.t. ``fake_score`` only.

        Returns a dict with ``"fake_score"`` and ``"total"`` (both equal).
        """
        cfg = self.config
        batch_size = latents.shape[0]
        device = latents.device

        # 1. Build student input.
        input_student, t_student = self._build_student_input(latents, noise)

        # 2. Generate data from student (no grad).
        with torch.no_grad():
            gen_data = self._predict_x0(
                self.student,
                input_student,
                t_student,
                encoder_hidden_states=encoder_hidden_states,
                **model_kwargs,
            )

            # 3. Perturb gen_data.
            t = self.sample_timesteps(batch_size, device=device, dtype=torch.float32)
            eps = torch.randn_like(latents)
            perturbed = add_noise(gen_data, eps, t)

        # 4. Fake-score forward (grad flows here).
        #
        # The fake_score's architectural output parameterization is ``cfg.pred_type``
        # (same arch as teacher/student in DMD2). ``fake_score_pred_type`` controls
        # which space the DSM loss is computed in — it is a loss-side knob, not a
        # model-side one. When the two differ (e.g. the Wan 2.2 recipe with
        # flow-native models and ``fake_score_pred_type='x0'``), we project the raw
        # output through the ``x_0`` hub into the loss space before calling
        # ``dsm_loss``. When they agree, ``_convert_pred`` short-circuits to identity.
        fake_pred_type = cfg.fake_score_pred_type or cfg.pred_type
        raw = self._call_model(
            self.fake_score,
            perturbed,
            t,
            encoder_hidden_states=encoder_hidden_states,
            **model_kwargs,
        )
        pred_in_loss_space = self._convert_pred(
            raw,
            perturbed,
            t,
            from_pred_type=cfg.pred_type,
            to_pred_type=fake_pred_type,
        )

        # 5. DSM loss in the chosen parameterization.
        loss = dsm_loss(
            fake_pred_type,
            pred_in_loss_space,
            x0=gen_data,
            eps=eps,
            t=t,
            alpha_fn=rf_alpha,
            sigma_fn=rf_sigma,
        )
        return {"fake_score": loss, "total": loss}

    def compute_discriminator_loss(
        self,
        latents: torch.Tensor,
        noise: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        **model_kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """Compute the discriminator update loss (GAN + optional R1).

        Teacher and student forwards are wrapped in ``torch.no_grad()``; gradient flows
        only through the discriminator.

        Returns a dict with ``"gan_disc"`` and ``"total"``. When
        ``config.gan_r1_reg_weight > 0`` the dict also contains ``"r1"``.
        """
        cfg = self.config
        if self.discriminator is None:
            raise RuntimeError(
                "compute_discriminator_loss requires a discriminator to be set on DMDPipeline."
            )
        batch_size = latents.shape[0]
        device = latents.device

        # 1. Build student input and generate gen_data (no grad).
        input_student, t_student = self._build_student_input(latents, noise)
        with torch.no_grad():
            gen_data = self._predict_x0(
                self.student,
                input_student,
                t_student,
                encoder_hidden_states=encoder_hidden_states,
                **model_kwargs,
            )

            # 2. Sample fake-branch timesteps and noise.
            t = self.sample_timesteps(batch_size, device=device, dtype=torch.float32)
            eps = torch.randn_like(latents)
            perturbed_fake = add_noise(gen_data, eps, t)

            # 3. Teacher forward on fake data to capture features.
            _ = self._predict_x0(
                self.teacher,
                perturbed_fake,
                t,
                encoder_hidden_states=encoder_hidden_states,
                **model_kwargs,
            )
            fake_feat = _require_hooked(_drain_if_hooked(self.teacher), which="disc-fake")

            # 4. Real branch: same t/eps or re-sampled.
            if cfg.gan_use_same_t_noise:
                t_real = t
                eps_real = eps
            else:
                t_real = self.sample_timesteps(batch_size, device=device, dtype=torch.float32)
                eps_real = torch.randn_like(latents)
            perturbed_real = add_noise(latents, eps_real, t_real)

            _ = self._predict_x0(
                self.teacher,
                perturbed_real,
                t_real,
                encoder_hidden_states=encoder_hidden_states,
                **model_kwargs,
            )
            real_feat = _require_hooked(_drain_if_hooked(self.teacher), which="disc-real")

        # 5. Discriminator on real / fake (grad required).
        real_logits = self.discriminator(real_feat)
        fake_logits = self.discriminator(fake_feat)
        disc = gan_disc_loss(real_logits, fake_logits)

        result: dict[str, torch.Tensor] = {"gan_disc": disc}

        # 6. Optional R1 regularization.
        if cfg.gan_r1_reg_weight > 0:
            with torch.no_grad():
                perturbed_real_alpha = latents + cfg.gan_r1_reg_alpha * torch.randn_like(latents)
                _ = self._predict_x0(
                    self.teacher,
                    perturbed_real_alpha,
                    t_real,
                    encoder_hidden_states=encoder_hidden_states,
                    **model_kwargs,
                )
                real_feat_alpha = _require_hooked(_drain_if_hooked(self.teacher), which="disc-r1")
            real_logits_alpha = self.discriminator(real_feat_alpha)
            r1 = r1_loss(real_logits, real_logits_alpha)
            total = disc + cfg.gan_r1_reg_weight * r1
            result["r1"] = r1
            result["total"] = total
        else:
            result["total"] = disc
        return result

    # ================================================================== #
    #  EMA                                                               #
    # ================================================================== #

    def update_ema(self, *, iteration: int | None = None) -> None:
        """Update the student EMA tracker (no-op if ``config.ema`` is ``None``).

        Typically called after the student optimizer step. If ``iteration`` is not
        provided, an internal counter is auto-incremented.
        """
        if self._ema is None:
            return
        if iteration is not None:
            self._iteration = iteration
        else:
            self._iteration += 1
        self._ema.update(self.student, iteration=self._iteration)
