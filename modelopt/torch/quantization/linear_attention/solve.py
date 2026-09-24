# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Explicit triangular inverse policies for differentiable prefill emulation."""

import torch

from .config import LinearAttentionSolveConfig

__all__ = ["triangular_inverse"]


def triangular_inverse(lower: torch.Tensor, policy: LinearAttentionSolveConfig) -> torch.Tensor:
    """Materialize the exact unit-lower triangular inverse for WY operand QDQ."""
    if lower.ndim < 2 or lower.shape[-1] != lower.shape[-2]:
        raise ValueError("lower must contain square matrices")
    with torch.autocast(device_type=lower.device.type, enabled=False):
        identity = torch.eye(lower.shape[-1], device=lower.device, dtype=lower.dtype).expand_as(
            lower
        )
        return torch.linalg.solve_triangular(
            identity + lower, identity, upper=False, unitriangular=True
        )
