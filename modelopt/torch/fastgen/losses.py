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

"""Pure loss functions used by the fastgen distillation pipelines.

All functions in this module are stateless: they take tensors in and return a scalar
loss tensor. They do not touch any ``nn.Module``. Higher-level orchestration (teacher
forward, CFG, noise scheduling) lives in :mod:`modelopt.torch.fastgen.methods.dmd`.

Math ported from ``FastGen/fastgen/methods/common_loss.py`` (lines 12-136) and
``FastGen/fastgen/methods/distribution_matching/dmd2.py`` lines 287-317 (R1).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from .utils import expand_like

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = [
    "dsm_loss",
    "gan_disc_loss",
    "gan_gen_loss",
    "r1_loss",
    "vsd_loss",
]


def dsm_loss(
    pred_type: str,
    net_pred: torch.Tensor,
    *,
    x0: torch.Tensor | None = None,
    eps: torch.Tensor | None = None,
    t: torch.Tensor | None = None,
    alpha_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    sigma_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> torch.Tensor:
    """Denoising score-matching loss for ``x0`` / ``eps`` / ``v`` / ``flow`` predictions.

    The forward process is ``x_t = alpha_t * x_0 + sigma_t * eps``. For
    ``pred_type='v'`` we need ``alpha_t`` and ``sigma_t``, which are supplied as
    callables rather than a full noise-scheduler object so this function stays
    scheduler-agnostic.

    Args:
        pred_type: One of ``"x0"``, ``"eps"``, ``"v"``, ``"flow"``.
        net_pred: The network output; its interpretation is determined by ``pred_type``.
        x0: Clean data. Required for all ``pred_type`` except ``"eps"``.
        eps: Noise used in the forward process. Required for all ``pred_type`` except ``"x0"``.
        t: Timesteps in ``[0, 1]`` (or scheduler convention). Required for ``pred_type='v'``.
        alpha_fn: Callable mapping ``t`` -> ``alpha_t``. Required for ``pred_type='v'``.
        sigma_fn: Callable mapping ``t`` -> ``sigma_t``. Required for ``pred_type='v'``.

    Returns:
        Scalar MSE loss.
    """
    if pred_type == "x0":
        assert x0 is not None, "x0 is required for pred_type='x0'"
        return F.mse_loss(x0, net_pred, reduction="mean")
    if pred_type == "eps":
        assert eps is not None, "eps is required for pred_type='eps'"
        return F.mse_loss(eps, net_pred, reduction="mean")
    if pred_type == "v":
        assert x0 is not None and eps is not None and t is not None, (
            "x0, eps, and t are required for pred_type='v'"
        )
        assert alpha_fn is not None and sigma_fn is not None, (
            "alpha_fn and sigma_fn are required for pred_type='v'"
        )
        alpha_t = expand_like(alpha_fn(t), x0).to(device=x0.device, dtype=x0.dtype)
        sigma_t = expand_like(sigma_fn(t), x0).to(device=x0.device, dtype=x0.dtype)
        v = alpha_t * eps - sigma_t * x0
        return F.mse_loss(v, net_pred, reduction="mean")
    if pred_type == "flow":
        assert x0 is not None and eps is not None, "x0 and eps are required for pred_type='flow'"
        flow_velocity = eps - x0
        return F.mse_loss(flow_velocity, net_pred, reduction="mean")
    raise ValueError(f"Unknown pred_type {pred_type!r}; expected one of 'x0', 'eps', 'v', 'flow'.")


def vsd_loss(
    gen_data: torch.Tensor,
    teacher_x0: torch.Tensor,
    fake_score_x0: torch.Tensor,
    additional_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """Variational score-distillation (VSD) loss used by the DMD student update.

    Implements the FastGen formulation: a per-sample weight
    ``w = 1 / (mean_abs(gen_data - teacher_x0) + 1e-6)`` is computed in fp32 for
    numerical stability, then the gradient ``(fake_score_x0 - teacher_x0) * w`` is
    subtracted from the generated data to form a pseudo-target. The loss is
    ``0.5 * MSE(gen_data, pseudo_target)``.

    Args:
        gen_data: Student-generated clean data ``x_0``.
        teacher_x0: Teacher ``x_0`` prediction (after CFG, if enabled). Detached.
        fake_score_x0: Fake-score ``x_0`` prediction. Detached.
        additional_scale: Optional per-sample scale applied multiplicatively to the weight.

    Returns:
        Scalar VSD loss.
    """
    dims = tuple(range(1, teacher_x0.ndim))

    with torch.no_grad():
        original_dtype = gen_data.dtype
        gen_data_fp32 = gen_data.float()
        teacher_x0_fp32 = teacher_x0.float()

        diff_abs_mean = (gen_data_fp32 - teacher_x0_fp32).abs().mean(dim=dims, keepdim=True)
        w_fp32 = 1.0 / (diff_abs_mean + 1e-6)

        if additional_scale is not None:
            w_fp32 = w_fp32 * expand_like(additional_scale.float(), w_fp32)

        w = w_fp32.to(dtype=original_dtype)
        vsd_grad = (fake_score_x0 - teacher_x0) * w
        pseudo_target = gen_data - vsd_grad

    return 0.5 * F.mse_loss(gen_data, pseudo_target, reduction="mean")


def gan_gen_loss(fake_logits: torch.Tensor) -> torch.Tensor:
    """Softplus GAN generator loss: ``E[softplus(-fake_logits)]``.

    Args:
        fake_logits: Discriminator logits on generated samples. Must be 2D: ``(B, num_heads)``.
    """
    assert fake_logits.ndim == 2, f"fake_logits must be 2D, got shape {tuple(fake_logits.shape)}"
    return F.softplus(-fake_logits).mean()


def gan_disc_loss(real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
    """Softplus GAN discriminator loss: ``E[softplus(fake_logits)] + E[softplus(-real_logits)]``."""
    assert real_logits.ndim == 2, f"real_logits must be 2D, got shape {tuple(real_logits.shape)}"
    assert fake_logits.ndim == 2, f"fake_logits must be 2D, got shape {tuple(fake_logits.shape)}"
    return F.softplus(fake_logits).mean() + F.softplus(-real_logits).mean()


def r1_loss(
    real_logits: torch.Tensor,
    perturbed_real_logits: torch.Tensor,
) -> torch.Tensor:
    """Approximate R1 regularization (APT formulation).

    Penalizes the discriminator for being sensitive to small noise perturbations of
    the real data. The caller is responsible for computing ``perturbed_real_logits``
    by re-running the teacher feature extractor and discriminator on real data that
    has been perturbed with ``alpha * randn_like(real)``; this function only applies
    the final MSE between the two logit sets.

    See ``FastGen/fastgen/methods/distribution_matching/dmd2.py`` lines 287-317.
    """
    assert real_logits.shape == perturbed_real_logits.shape, (
        f"real_logits {tuple(real_logits.shape)} and perturbed_real_logits "
        f"{tuple(perturbed_real_logits.shape)} must have matching shapes"
    )
    return F.mse_loss(real_logits, perturbed_real_logits, reduction="mean")
