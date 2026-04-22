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

"""Small tensor helpers shared across the fastgen subpackage."""

from __future__ import annotations

import torch

__all__ = ["classifier_free_guidance", "expand_like"]


def expand_like(x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Pad ``x`` with trailing singleton dims until it has the same ndim as ``target``.

    Used to broadcast per-sample scalars like ``alpha_t`` / ``sigma_t`` across the
    spatial / temporal axes of a video latent.

    Example::

        x = torch.ones(5)  # shape (5,)
        target = torch.ones(5, 4, 16, 16)
        expand_like(x, target).shape  # (5, 1, 1, 1)
    """
    x = torch.atleast_1d(x)
    while x.ndim < target.ndim:
        x = x[..., None]
    return x


def classifier_free_guidance(
    cond_pred: torch.Tensor,
    uncond_pred: torch.Tensor,
    guidance_scale: float,
) -> torch.Tensor:
    """Combine conditional and unconditional predictions via classifier-free guidance.

    Uses the DMD2 convention ``cond + (scale - 1) * (cond - uncond)``, which is
    mathematically equivalent to the standard CFG formula
    ``uncond + scale * (cond - uncond)``.
    """
    return cond_pred + (guidance_scale - 1.0) * (cond_pred - uncond_pred)
