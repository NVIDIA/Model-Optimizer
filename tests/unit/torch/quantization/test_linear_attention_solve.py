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

from modelopt.torch.quantization.linear_attention.config import (
    LinearAttentionConfig,
    LinearAttentionSolveConfig,
)
from modelopt.torch.quantization.linear_attention.solve import (
    neumann_inverse,
    neumann_inverse_reference,
    neumann_tail_bound,
    triangular_inverse,
)


def _exact(lower):
    identity = torch.eye(lower.shape[-1], dtype=lower.dtype).expand_as(lower)
    return torch.linalg.solve_triangular(
        identity + lower.tril(-1), identity, upper=False, unitriangular=True
    )


@pytest.mark.parametrize("degree", [3, 7, 15, 31])
def test_binary_polynomial_matches_independent_power_sum_and_gradient(degree):
    torch.manual_seed(91)
    lower = (torch.randn(2, 16, 16, dtype=torch.float64) * 0.01).requires_grad_()
    actual = neumann_inverse(lower, degree)
    expected = neumann_inverse_reference(lower, degree)
    torch.testing.assert_close(actual, expected, rtol=2e-13, atol=2e-14)
    probe = torch.randn_like(actual)
    gradients = [
        torch.autograd.grad((value * probe).sum(), lower)[0] for value in (actual, expected)
    ]
    torch.testing.assert_close(*gradients, rtol=2e-12, atol=2e-13)


def test_safe_tail_bound_covers_approximation_error():
    torch.manual_seed(29)
    lower = torch.randn(3, 16, 16, dtype=torch.float64).tril(-1) * 0.005
    bound = neumann_tail_bound(lower, 7)
    error = (neumann_inverse(lower, 7) - _exact(lower)).abs().sum(-1).amax(-1)
    assert torch.all(error <= bound + 1e-14)
    assert torch.all(bound < 1e-3)


def test_high_correlation_failure_is_preserved_and_guarded():
    lower = torch.tril(torch.full((64, 64), 0.9), diagonal=-1)
    exact = _exact(lower)
    raw = neumann_inverse_reference(lower, 63)
    relative_error = (raw - exact).norm() / exact.norm()
    assert relative_error > 1e6
    assert torch.isinf(neumann_tail_bound(lower, 31))

    rejecting = LinearAttentionSolveConfig(
        method="guarded_neumann", degree=31, unsafe_action="error"
    )
    with pytest.raises(ValueError, match="truncation bound rejected"):
        triangular_inverse(lower, rejecting)
    fallback = LinearAttentionSolveConfig(method="guarded_neumann", degree=31)
    torch.testing.assert_close(triangular_inverse(lower, fallback), exact, rtol=0, atol=0)


def test_mixed_batch_falls_back_per_matrix_with_exact_gradient():
    torch.manual_seed(112)
    safe = torch.randn(16, 16) * 0.005
    unsafe = torch.tril(torch.full((16, 16), 0.2), diagonal=-1)
    lower = torch.stack((safe, unsafe)).requires_grad_()
    policy = LinearAttentionSolveConfig(method="guarded_neumann", degree=7)
    result = triangular_inverse(lower, policy)
    torch.testing.assert_close(result[1], _exact(lower)[1], rtol=0, atol=0)
    actual_gradient = torch.autograd.grad(result[1].sum(), lower)[0][1]
    expected_gradient = torch.autograd.grad(_exact(lower)[1].sum(), lower)[0][1]
    torch.testing.assert_close(actual_gradient, expected_gradient, rtol=0, atol=0)


@pytest.mark.parametrize(
    "solve",
    [
        {"method": "guarded_neumann"},
        {"method": "guarded_neumann", "degree": 63},
        {"method": "exact", "degree": 7},
        {"method": "guarded_neumann", "degree": 7, "max_tail_bound": 0},
    ],
)
def test_invalid_solve_policy(solve):
    with pytest.raises(ValidationError):
        LinearAttentionSolveConfig(**solve)


def test_approximation_is_gdn_prefill_only_and_saved():
    solve = {"method": "guarded_neumann", "degree": 15, "max_tail_bound": 1e-4}
    policy = LinearAttentionConfig(backend="gdn_prefill", precision="gdn_prefill", solve=solve)
    assert LinearAttentionConfig.model_validate_json(policy.model_dump_json()) == policy
    with pytest.raises(ValidationError, match="restricted"):
        LinearAttentionConfig(backend="serving", solve=solve)
