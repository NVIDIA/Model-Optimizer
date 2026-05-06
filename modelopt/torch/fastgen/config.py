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

"""Pydantic configuration classes for the fastgen distillation pipelines.

Configurations are layered so a method-specific config (e.g. :class:`DMDConfig`) inherits
shared diffusion-distillation hyperparameters from :class:`DistillationConfig`. All classes
inherit :class:`modelopt.torch.opt.config.ModeloptBaseConfig`, which provides torch-safe
serialization and dict-like iteration.

The default values in :class:`DMDConfig` mirror the FastGen Wan 2.2 5B experiment at
``FastGen/fastgen/configs/experiments/WanT2V/config_dmd2_wan22_5b.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import Field, model_validator

from modelopt.torch.opt.config import ModeloptBaseConfig, ModeloptField

if TYPE_CHECKING:
    from pathlib import Path

__all__ = [
    "DMDConfig",
    "DistillationConfig",
    "EMAConfig",
    "SampleTimestepConfig",
]

PredType = Literal["x0", "eps", "v", "flow"]
TimeDistType = Literal["uniform", "logitnormal", "lognormal", "shifted", "polynomial"]


class SampleTimestepConfig(ModeloptBaseConfig):
    """Timestep sampling distribution for diffusion training."""

    time_dist_type: TimeDistType = ModeloptField(
        default="shifted",
        title="Timestep distribution",
        description=(
            "Distribution used to sample the training timestep ``t``. Rectified-flow models"
            " typically use ``shifted`` (Wan 2.2) or ``logitnormal`` (SD3, Flux)."
        ),
    )
    min_t: float = ModeloptField(
        default=0.001,
        title="Minimum t",
        description="Lower bound of the sampling range (clamped before use).",
    )
    max_t: float = ModeloptField(
        default=0.999,
        title="Maximum t",
        description="Upper bound of the sampling range (clamped before use).",
    )
    shift: float = ModeloptField(
        default=5.0,
        title="Shift factor",
        description="Shift factor for ``time_dist_type='shifted'``; must be >= 1.",
    )
    p_mean: float = ModeloptField(
        default=0.0,
        title="Distribution mean (log-space)",
        description="Mean of the underlying normal for ``logitnormal`` / ``lognormal``.",
    )
    p_std: float = ModeloptField(
        default=1.0,
        title="Distribution std (log-space)",
        description="Standard deviation of the underlying normal for ``logitnormal`` / ``lognormal``.",
    )
    t_list: list[float] | None = ModeloptField(
        default=None,
        title="Multi-step student timesteps",
        description=(
            "Explicit timestep schedule used when ``DMDConfig.student_sample_steps > 1``."
            " The final element must be ``0.0``."
        ),
    )

    @model_validator(mode="after")
    def _check_bounds(self) -> SampleTimestepConfig:
        assert 0.0 <= self.min_t < self.max_t, (
            f"require 0 <= min_t < max_t, got min_t={self.min_t}, max_t={self.max_t}"
        )
        assert self.shift >= 1.0, f"shift must be >= 1, got {self.shift}"
        if self.t_list is not None:
            assert len(self.t_list) >= 2, "t_list must contain at least 2 entries (including t=0)"
            assert self.t_list[-1] == 0.0, f"t_list[-1] must be 0.0, got {self.t_list[-1]}"
        return self


class EMAConfig(ModeloptBaseConfig):
    """Exponential moving average (EMA) hyperparameters for the student network."""

    decay: float = ModeloptField(
        default=0.9999,
        title="EMA decay",
        description="Decay coefficient for ``type='constant'``. Ignored for ``halflife``/``power``.",
    )
    type: Literal["constant", "halflife", "power"] = ModeloptField(
        default="constant",
        title="EMA decay schedule",
        description="Schedule used to compute the per-step decay coefficient.",
    )
    start_iter: int = ModeloptField(
        default=0,
        title="EMA start iteration",
        description="Iteration at which EMA tracking begins (EMA is initialized from the live weights at this step).",
    )
    gamma: float = ModeloptField(
        default=16.97,
        title="Power schedule gamma",
        description="Exponent for ``type='power'`` (``beta = (1 - 1/iter)**(gamma + 1)``).",
    )
    halflife_kimg: float = ModeloptField(
        default=500.0,
        title="Halflife (kimg)",
        description="Halflife in thousands of images for ``type='halflife'``.",
    )
    rampup_ratio: float | None = ModeloptField(
        default=0.05,
        title="Halflife rampup ratio",
        description="Rampup fraction for ``type='halflife'``; pass ``None`` to disable rampup.",
    )
    batch_size: int = ModeloptField(
        default=1,
        title="Effective batch size",
        description="Per-step global batch size used to convert iterations to nimg for the halflife schedule.",
    )
    fsdp2: bool = ModeloptField(
        default=True,
        title="FSDP2 enabled",
        description="If True, the EMA uses ``DTensor.full_tensor()`` to gather sharded parameters before updating.",
    )
    mode: Literal["full_tensor", "local_shard"] = ModeloptField(
        default="full_tensor",
        title="FSDP2 gather mode",
        description=(
            "``full_tensor`` performs an all_gather per parameter (higher memory, exact global EMA)."
            " ``local_shard`` updates each rank's local DTensor shard in place (low memory fallback)."
        ),
    )
    dtype: Literal["float32", "bfloat16", "float16"] | None = ModeloptField(
        default="float32",
        title="EMA shadow dtype",
        description=(
            "Precision of the EMA parameter shadows. Defaults to ``float32`` so EMA updates"
            " remain numerically meaningful even when the live model is bf16/fp16 (cf. FastGen,"
            " which instantiates its EMA module in the net's construction dtype — typically"
            " fp32). Pass ``None`` to keep param shadows in the live parameter's dtype."
            " Buffer shadows always track the live dtype regardless of this setting."
        ),
    )


class DistillationConfig(ModeloptBaseConfig):
    """Shared hyperparameters for diffusion step-distillation methods.

    Concrete methods subclass this config to add method-specific fields
    (see :class:`DMDConfig`).
    """

    pred_type: PredType = ModeloptField(
        default="flow",
        title="Network prediction parameterization",
        description="Quantity predicted by the teacher / student network.",
    )
    guidance_scale: float | None = ModeloptField(
        default=None,
        title="CFG scale",
        description="Classifier-free guidance scale. If ``None`` CFG is disabled.",
    )
    # ``ModeloptField`` hard-asserts on ``default_factory``; use Pydantic's ``Field``
    # directly for this mutable sub-config so each DMDConfig instance gets its own
    # SampleTimestepConfig instead of sharing a single mutable default.
    sample_t_cfg: SampleTimestepConfig = Field(
        default_factory=SampleTimestepConfig,
        title="Timestep sampling",
        description="Timestep distribution used for both the teacher forward and the VSD / DSM losses.",
    )
    student_sample_steps: int = ModeloptField(
        default=1,
        title="Student inference steps",
        description="Number of denoising steps the distilled student performs at inference.",
    )
    student_sample_type: Literal["sde", "ode"] = ModeloptField(
        default="ode",
        title="Student sampling mode",
        description=(
            "Integrator used when unrolling the student over ``student_sample_steps > 1`` steps."
            " Not read by DMDPipeline at training time — consumed by inference samplers that"
            " unroll the student over ``student_sample_steps > 1`` steps."
        ),
    )
    num_train_timesteps: int | None = ModeloptField(
        default=None,
        title="Training-time discrete timestep count",
        description=(
            "If set, the pipeline rescales the continuous RF timestep ``t ∈ [0, 1]`` to"
            " ``num_train_timesteps * t`` before passing it to the model. Matches the"
            " diffusers convention used by Wan 2.2 / SD3 / Flux (``num_train_timesteps = 1000``)."
            " Leave ``None`` when the model wrapper already handles the rescaling internally."
        ),
    )


class DMDConfig(DistillationConfig):
    """Hyperparameters for DMD / DMD2 distribution-matching distillation.

    Default values are tuned for Wan 2.2 5B; callers fine-tune them per model.
    See ``FastGen/fastgen/configs/experiments/WanT2V/config_dmd2_wan22_5b.py``.
    """

    student_update_freq: int = ModeloptField(
        default=5,
        title="Student update frequency",
        description=(
            "One student step for every ``student_update_freq`` fake-score / discriminator steps."
            " Matches FastGen's DMD2 alternation. Not read by DMDPipeline; the training loop is"
            " expected to enforce the alternation."
        ),
    )
    fake_score_pred_type: PredType | None = ModeloptField(
        default="x0",
        title="Fake-score prediction parameterization",
        description=(
            "Parameterization used when training the fake score. If ``None`` falls back to"
            " :attr:`DistillationConfig.pred_type`."
        ),
    )
    gan_loss_weight_gen: float = ModeloptField(
        default=0.0,
        title="Generator GAN weight",
        description="Weight of the GAN generator term in the student loss. ``0`` disables the GAN branch.",
    )
    gan_use_same_t_noise: bool = ModeloptField(
        default=False,
        title="Share t/noise across real and fake",
        description="If True, reuse the same ``t`` and ``eps`` for real and fake samples in the discriminator update.",
    )
    gan_r1_reg_weight: float = ModeloptField(
        default=0.0,
        title="R1 regularization weight",
        description=(
            "Weight of the approximate-R1 regularization term for the discriminator update. ``0`` disables R1."
            " Recommended range when enabled: 100-1000."
        ),
    )
    gan_r1_reg_alpha: float = ModeloptField(
        default=0.1,
        title="R1 regularization noise scale",
        description=(
            "Standard deviation of the perturbation applied to real data when computing the"
            " approximate R1 term."
        ),
    )
    ema: EMAConfig | None = ModeloptField(
        default=None,
        title="Student EMA",
        description=(
            "If set, an exponential moving average of the student is maintained and updated"
            " via ``DMDPipeline.update_ema``."
        ),
    )

    @model_validator(mode="after")
    def _check_gan(self) -> DMDConfig:
        if self.gan_r1_reg_weight > 0 and self.gan_loss_weight_gen <= 0:
            raise ValueError(
                "gan_r1_reg_weight > 0 requires gan_loss_weight_gen > 0 (the discriminator must be enabled)."
            )
        return self

    @classmethod
    def from_yaml(cls, config_file: str | Path) -> DMDConfig:
        """Construct a :class:`DMDConfig` from a YAML file.

        Thin wrapper around :func:`modelopt.torch.fastgen.loader.load_dmd_config`.
        The resolver searches the built-in ``modelopt_recipes/`` package first, then
        the filesystem. Suffixes (``.yml`` / ``.yaml``) may be omitted.
        """
        # Imported lazily to avoid a circular import between this module and
        # ``modelopt.torch.fastgen.loader`` (which imports :class:`DMDConfig`).
        from .loader import load_dmd_config

        return load_dmd_config(config_file)
