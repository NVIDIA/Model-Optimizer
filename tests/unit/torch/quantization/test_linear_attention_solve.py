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

import pytest
import torch
from pydantic import ValidationError

from modelopt.torch.quantization.linear_attention import (
    LinearAttentionConfig,
    LinearAttentionSolveConfig,
    neumann_inverse_reference,
    triangular_inverse,
)


@pytest.mark.parametrize("degree", [0, 1, 2, 3, 7, 15, 31, 63])
def test_polynomial_residual_and_gradient(degree):
    torch.manual_seed(82)
    lower = (torch.randn(2, 64, 64, dtype=torch.float64) * 0.04).requires_grad_()
    policy = LinearAttentionSolveConfig(method="neumann", degree=degree)
    actual = triangular_inverse(lower, policy)
    expected = neumann_inverse_reference(lower, degree)
    torch.testing.assert_close(actual, expected, rtol=2e-12, atol=2e-14)
    probe = torch.randn_like(actual)
    gradients = [torch.autograd.grad((a * probe).sum(), lower)[0] for a in (actual, expected)]
    torch.testing.assert_close(*gradients, rtol=2e-11, atol=2e-13)
    matrix = lower.tril(-1)
    identity = torch.eye(64, dtype=lower.dtype)
    residual = identity - (identity + matrix) @ actual
    torch.testing.assert_close(
        residual, torch.linalg.matrix_power(-matrix, degree + 1), rtol=1e-9, atol=2e-14
    )
    if degree == 63:
        exact = triangular_inverse(lower, LinearAttentionSolveConfig())
        torch.testing.assert_close(actual, exact, rtol=2e-12, atol=2e-14)


def test_degree_is_not_replaced_by_exact_fallback():
    lower = torch.diag(torch.full((63,), 1.2, dtype=torch.float64), diagonal=-1)
    approximation = triangular_inverse(
        lower, LinearAttentionSolveConfig(method="neumann", degree=3)
    )
    exact = triangular_inverse(lower, LinearAttentionSolveConfig())
    assert (approximation - exact).abs().amax() > 100
    assert torch.count_nonzero(approximation.tril(-4)) == 0


@pytest.mark.parametrize(
    "solve",
    [
        {"method": "neumann"},
        {"method": "exact", "degree": 3},
        {"method": "exact", "implementation": "triton"},
        {"method": "neumann", "degree": -1},
        {"method": "neumann", "degree": 64},
        {"method": "neumann", "degree": True},
        {"method": "neumann", "degree": 1.5},
    ],
)
def test_invalid_solve_policy(solve):
    with pytest.raises(ValidationError):
        LinearAttentionSolveConfig(**solve)


def test_saved_policy_and_backend_validation():
    policy = LinearAttentionConfig(
        backend="matmul", solve={"method": "neumann", "degree": 7, "implementation": "triton"}
    )
    assert LinearAttentionConfig.model_validate_json(policy.model_dump_json()) == policy
    with pytest.raises(ValidationError, match="backend"):
        LinearAttentionConfig(solve={"method": "neumann", "degree": 7})
    with pytest.raises(ValueError, match="CUDA FP32"):
        triangular_inverse(torch.zeros(64, 64), policy.solve)
