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

"""Rectified-flow (RF) helpers: forward process, inversions, timestep sampling.

This module intentionally does **not** define a ``NoiseScheduler`` class. It exposes the
handful of primitives that DMD2 actually needs as plain functions, so callers can plug
fastgen into any training stack without adopting a new scheduler object.

RF convention used throughout: ``alpha_t = 1 - t`` and ``sigma_t = t``, so
``x_t = (1 - t) * x_0 + t * eps`` with ``t in [0, 1]``. Internally all arithmetic is in
``float64`` for numerical stability, and the result is cast back to the input dtype.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from torch.distributions import Normal

from .utils import expand_like

if TYPE_CHECKING:
    from .config import SampleTimestepConfig

__all__ = [
    "add_noise",
    "pred_noise_to_pred_x0",
    "pred_x0_from_flow",
    "rf_alpha",
    "rf_sigma",
    "sample_from_t_list",
    "sample_timesteps",
    "x0_to_eps",
    "x0_to_flow",
]


def rf_alpha(t: torch.Tensor) -> torch.Tensor:
    """Rectified-flow data coefficient ``alpha_t = 1 - t``."""
    return 1.0 - t


def rf_sigma(t: torch.Tensor) -> torch.Tensor:
    """Rectified-flow noise coefficient ``sigma_t = t``."""
    return t


def add_noise(x0: torch.Tensor, eps: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Forward process under rectified flow: ``x_t = (1 - t) * x_0 + t * eps``.

    ``t`` is broadcast across the spatial axes of ``x_0`` via :func:`expand_like`.
    Computation is performed in ``float64`` for numerical stability and the output is
    cast back to ``x_0``'s dtype.
    """
    original_dtype = x0.dtype
    x0_64 = x0.to(torch.float64)
    eps_64 = eps.to(torch.float64)
    t_64 = t.to(torch.float64)
    alpha = expand_like(rf_alpha(t_64), x0_64)
    sigma = expand_like(rf_sigma(t_64), x0_64)
    x_t = x0_64 * alpha + eps_64 * sigma
    return x_t.to(original_dtype)


def pred_noise_to_pred_x0(
    pred_noise: torch.Tensor,
    noisy_latents: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    """Convert an ``eps``-parameterized prediction to an ``x_0`` prediction under RF.

    Solves ``x_t = (1 - t) * x_0 + t * eps`` for ``x_0``:
    ``x_0 = (x_t - t * eps) / (1 - t)``.
    """
    original_dtype = noisy_latents.dtype
    x_t = noisy_latents.to(torch.float64)
    pred_noise_64 = pred_noise.to(torch.float64)
    t_64 = t.to(torch.float64)
    alpha = expand_like(rf_alpha(t_64), x_t)
    sigma = expand_like(rf_sigma(t_64), x_t)
    x0 = (x_t - sigma * pred_noise_64) / alpha.clamp_min(1e-6)
    return x0.to(original_dtype)


def pred_x0_from_flow(
    pred_flow: torch.Tensor,
    noisy_latents: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    """Convert a flow-parameterized prediction (``v = eps - x_0``) to an ``x_0`` prediction.

    Under RF ``x_t = (1 - t) * x_0 + t * eps`` and ``v = eps - x_0`` combine to
    ``x_t = x_0 + t * v``, so ``x_0 = x_t - t * v``.
    """
    original_dtype = noisy_latents.dtype
    x_t = noisy_latents.to(torch.float64)
    v = pred_flow.to(torch.float64)
    t_64 = t.to(torch.float64)
    sigma = expand_like(rf_sigma(t_64), x_t)
    x0 = x_t - sigma * v
    return x0.to(original_dtype)


def x0_to_eps(
    x0: torch.Tensor,
    x_t: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    """Invert the RF forward process: ``eps = (x_t - (1 - t) * x_0) / t``.

    Used when unrolling the student in ODE mode — given the current ``x_t`` and the
    student's ``x_0`` prediction, we can recover the implied ``eps`` deterministically.
    """
    original_dtype = x0.dtype
    x0_64 = x0.to(torch.float64)
    x_t_64 = x_t.to(torch.float64)
    t_64 = t.to(torch.float64)
    alpha = expand_like(rf_alpha(t_64), x0_64)
    sigma = expand_like(rf_sigma(t_64), x0_64)
    eps = (x_t_64 - alpha * x0_64) / sigma.clamp_min(1e-6)
    return eps.to(original_dtype)


def x0_to_flow(
    x0: torch.Tensor,
    x_t: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    """Convert an ``x_0`` prediction back into a flow-parameterized prediction under RF.

    Under RF ``x_t = (1 - t) * x_0 + t * eps`` and ``v = eps - x_0``, so
    ``x_t - x_0 = t * (eps - x_0) = t * v`` and therefore ``v = (x_t - x_0) / t``.

    Used when the fake score is flow-native but the DSM loss is computed in a
    different target parameterization: convert raw flow → x_0 via
    :func:`pred_x0_from_flow`, then back to the loss space (which may coincide
    with flow, in which case the round-trip is identity up to fp64 round-off).
    """
    original_dtype = x0.dtype
    x0_64 = x0.to(torch.float64)
    x_t_64 = x_t.to(torch.float64)
    t_64 = t.to(torch.float64)
    sigma = expand_like(rf_sigma(t_64), x0_64)
    flow = (x_t_64 - x0_64) / sigma.clamp_min(1e-6)
    return flow.to(original_dtype)


# ---------------------------------------------------------------------------- #
#  Timestep sampling                                                           #
# ---------------------------------------------------------------------------- #


def _truncated_lognormal(
    n: int,
    mean: float,
    std: float,
    *,
    min_t: float,
    max_t: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Sample ``n`` values from a lognormal truncated to ``(min_t, max_t)``.

    Implementation ported from ``FastGen/fastgen/networks/noise_schedule.py`` (EDM
    ``_truncated_lognormal_sample``). Uses CDF inversion on the underlying normal for
    exact truncation.
    """
    min_t = max(min_t, 1e-12)
    log_min_t = torch.tensor(math.log(min_t), dtype=torch.float64)
    log_max_t = torch.tensor(math.log(max_t), dtype=torch.float64)
    normal = Normal(
        torch.tensor(mean, dtype=torch.float64),
        torch.tensor(std, dtype=torch.float64),
    )
    cdf_min = normal.cdf(log_min_t)
    cdf_max = normal.cdf(log_max_t)
    u = torch.rand(n, dtype=torch.float64) * (cdf_max - cdf_min) + cdf_min
    t = normal.icdf(u).exp()
    return t.to(device=device, dtype=dtype)


def sample_timesteps(
    n: int,
    cfg: SampleTimestepConfig,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Sample ``n`` training timesteps according to ``cfg``.

    Supports ``uniform``, ``logitnormal``, ``lognormal``, ``shifted``, and ``polynomial``
    distributions. ``polynomial`` for RF degenerates to discrete uniform sampling from a
    ``linspace(min_t, max_t, 1000)`` grid (EDM's polynomial-spaced ``_sigmas`` is
    EDM-specific and not applicable under RF).
    """
    min_t = cfg.min_t
    max_t = cfg.max_t

    if cfg.time_dist_type == "uniform":
        t = torch.rand(n, device=device, dtype=dtype) * (max_t - min_t) + min_t
    elif cfg.time_dist_type == "logitnormal":
        t = (
            torch.sigmoid(torch.randn(n, device=device, dtype=dtype) * cfg.p_std + cfg.p_mean)
            * (max_t - min_t)
            + min_t
        )
    elif cfg.time_dist_type == "lognormal":
        t = _truncated_lognormal(
            n,
            cfg.p_mean,
            cfg.p_std,
            min_t=min_t,
            max_t=max_t,
            device=device,
            dtype=dtype,
        )
    elif cfg.time_dist_type == "shifted":
        t = torch.rand(n, device=device, dtype=dtype) * (max_t - min_t) + min_t
        t = t * cfg.shift / (t * (cfg.shift - 1.0) + 1.0)
    elif cfg.time_dist_type == "polynomial":
        grid = torch.linspace(min_t, max_t, 1000, device=device, dtype=dtype)
        idx = torch.randint(0, grid.numel(), (n,), device=device)
        t = grid[idx]
    else:
        raise ValueError(f"Unsupported time_dist_type={cfg.time_dist_type!r}")

    return t.clamp(min_t, max_t)


def sample_from_t_list(
    n: int,
    t_list: list[float],
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Sample ``n`` starting timesteps uniformly from ``t_list[:-1]``.

    Used for multi-step student training: ``t_list`` encodes the inference trajectory
    (``t_list[-1]`` must be ``0``), and a random intermediate timestep is sampled so the
    student is trained at every rung of the trajectory.
    """
    assert len(t_list) >= 2, "t_list must have at least 2 entries (including the final 0)"
    assert t_list[-1] == 0.0, f"t_list[-1] must be 0.0, got {t_list[-1]}"
    t_tensor = torch.tensor(t_list, device=device, dtype=dtype)
    ids = torch.randint(0, t_tensor.numel() - 1, (n,), device=device)
    return t_tensor[ids]
