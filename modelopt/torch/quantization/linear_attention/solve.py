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

__all__ = ["neumann_inverse_reference", "triangular_inverse"]


def _neumann_binary(lower: torch.Tensor, degree: int) -> torch.Tensor:
    """Compose geometric-series blocks in a fixed binary order."""
    identity = torch.eye(lower.shape[-1], device=lower.device, dtype=lower.dtype).expand_as(lower)
    # Keep the degree-zero polynomial connected, with its correct zero derivative.
    series = identity + lower * 0
    power = -lower
    bits = bin(degree + 1)[3:]
    for index, bit in enumerate(bits):
        series = series + power @ series
        if bit == "1" or index + 1 < len(bits):
            power = power @ power
        if bit == "1":
            series = series + power
            if index + 1 < len(bits):
                power = power @ (-lower)
    return series


def neumann_inverse_reference(lower: torch.Tensor, degree: int) -> torch.Tensor:
    """Independently sum powers 0..degree, including the actual polynomial gradient.

    The strictly lower triangle is used. There is no residual-based fallback.
    Use FP64 to distinguish polynomial error from working-arithmetic error.
    """
    if type(degree) is not int or degree < 0:
        raise ValueError("degree must be a nonnegative integer")
    if lower.ndim < 2 or lower.shape[-1] != lower.shape[-2]:
        raise ValueError("lower must contain square matrices")
    with torch.autocast(device_type=lower.device.type, enabled=False):
        lower = lower.tril(-1)
        identity = torch.eye(lower.shape[-1], device=lower.device, dtype=lower.dtype).expand_as(
            lower
        )
        term = identity + lower * 0
        result = term
        for _ in range(degree):
            term = (-lower) @ term
            result = result + term
        return result


def triangular_inverse(lower: torch.Tensor, policy: LinearAttentionSolveConfig) -> torch.Tensor:
    """Materialize exact or polynomial inverse of I plus the strict lower triangle.

    Materializing the inverse retains the WY operand-QDQ sites. The Triton option
    requires CUDA FP32 chunk-64 matrices and supports first-order gradients only.
    """
    if lower.ndim < 2 or lower.shape[-1] != lower.shape[-2]:
        raise ValueError("lower must contain square matrices")
    with torch.autocast(device_type=lower.device.type, enabled=False):
        if policy.method == "exact":
            identity = torch.eye(lower.shape[-1], device=lower.device, dtype=lower.dtype).expand_as(
                lower
            )
            return torch.linalg.solve_triangular(
                identity + lower, identity, upper=False, unitriangular=True
            )
        if policy.degree is None:
            raise ValueError("Neumann solve requires an explicit polynomial degree")
        lower = lower.tril(-1)
        if policy.implementation == "torch":
            return _neumann_binary(lower, policy.degree)
        from modelopt.torch.kernels.quantization.linear_attention.neumann import neumann_inverse

        return neumann_inverse(lower, policy.degree)
