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

"""Framework-agnostic diffusion step-distillation losses (FastGen port).

``modelopt.torch.fastgen`` is a loss-computation library. It accepts already-built
``nn.Module`` references (student / teacher / fake-score / optional discriminator) and
returns scalar loss tensors. It does **not** load models, manage optimizers, wrap
anything as a ``DynamicModule``, or register itself in any mode registry.

Typical usage with a YAML-driven config::

    import modelopt.torch.fastgen as mtf

    student, teacher = build_wan_student_and_teacher(...)
    fake_score = mtf.create_fake_score(teacher)

    cfg = mtf.load_dmd_config("general/distillation/dmd2_wan22_5b")

    # If GAN is enabled, expose intermediate teacher features to the discriminator.
    if cfg.gan_loss_weight_gen > 0:
        mtf.plugins.wan22.attach_feature_capture(teacher, feature_indices=[15, 22, 29])

    pipeline = mtf.DMDPipeline(student, teacher, fake_score, cfg, discriminator=disc)

    # Inside the training loop (framework-owned):
    if step % cfg.student_update_freq == 0:
        losses = pipeline.compute_student_loss(
            latents, noise, text_embeds, negative_encoder_hidden_states=neg_embeds
        )
        losses["total"].backward()
        student_opt.step()
        pipeline.update_ema()
    else:
        f = pipeline.compute_fake_score_loss(latents, noise, text_embeds)
        f["total"].backward()
        fake_score_opt.step()
        if disc is not None:
            d = pipeline.compute_discriminator_loss(latents, noise, text_embeds)
            d["total"].backward()
            disc_opt.step()
"""

from . import flow_matching, losses, utils
from .config import DistillationConfig, DMDConfig, EMAConfig, SampleTimestepConfig
from .ema import ExponentialMovingAverage
from .factory import create_fake_score
from .loader import load_config, load_dmd_config
from .methods.dmd import DMDPipeline
from .pipeline import DistillationPipeline

# isort: off
# Plugins must be imported after the core exports so the wan22 hooks can reference
# DMDPipeline if needed in the future; also matches the ordering used by
# modelopt.torch.distill.
from . import plugins

__all__ = [
    "DMDConfig",
    "DMDPipeline",
    "DistillationConfig",
    "DistillationPipeline",
    "EMAConfig",
    "ExponentialMovingAverage",
    "SampleTimestepConfig",
    "create_fake_score",
    "flow_matching",
    "load_config",
    "load_dmd_config",
    "losses",
    "plugins",
    "utils",
]
