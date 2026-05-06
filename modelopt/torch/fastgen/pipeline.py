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

"""Base class for diffusion step-distillation pipelines.

:class:`DistillationPipeline` is deliberately minimal: it is **not** an ``nn.Module``,
does not wrap the student or teacher, does not manage optimizers or lifecycle state,
and does not register itself in any mode registry. It exists only to hold references
to the student / teacher and to freeze the teacher in a single place.

Concrete methods — for now :class:`~modelopt.torch.fastgen.methods.dmd.DMDPipeline` —
subclass this and add ``compute_*_loss`` methods.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

from .flow_matching import sample_timesteps

if TYPE_CHECKING:
    from .config import DistillationConfig

__all__ = ["DistillationPipeline"]


class DistillationPipeline:
    """Hold student/teacher references and expose shared utilities.

    Args:
        student: Trainable student module. The pipeline does not wrap it — its lifecycle
            (``train()`` / ``eval()``, ``requires_grad_``, sharding, optimizer) remains
            owned by the caller.
        teacher: Reference module. Frozen here via ``eval()`` + ``requires_grad_(False)``.
        config: A :class:`DistillationConfig` (or subclass).
    """

    def __init__(
        self,
        student: nn.Module,
        teacher: nn.Module,
        config: DistillationConfig,
    ) -> None:
        """Store student / teacher references and freeze the teacher."""
        self.student = student
        self.teacher = teacher.eval().requires_grad_(False)
        self.config = config

    # ------------------------------------------------------------------ #
    #  Device / dtype inferred from the student                          #
    # ------------------------------------------------------------------ #

    @property
    def device(self) -> torch.device:
        """Device of the first student parameter (best-effort; falls back to CPU)."""
        for p in self.student.parameters():
            return p.device
        return torch.device("cpu")

    @property
    def dtype(self) -> torch.dtype:
        """Dtype of the first student parameter (best-effort; falls back to float32)."""
        for p in self.student.parameters():
            return p.dtype
        return torch.float32

    # ------------------------------------------------------------------ #
    #  Shared helpers                                                    #
    # ------------------------------------------------------------------ #

    def sample_timesteps(
        self,
        n: int,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Sample ``n`` training timesteps according to :attr:`config`.``sample_t_cfg``."""
        return sample_timesteps(
            n,
            self.config.sample_t_cfg,
            device=device or self.device,
            dtype=dtype,
        )
