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

"""First-principles reference math for Parallel Decoding Distillation (PDD).

These deliberately simple CPU oracles are independent of the future PDD implementation.
Production grid, integration, and projection-fusion code should be tested against them.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _reference_shifted_grid(grid_size: int, shift: float) -> torch.Tensor:
    """Build the decreasing shifted rectified-flow grid with scalar arithmetic."""
    values = []
    for index in range(grid_size + 1):
        unshifted = 1.0 - index / grid_size
        values.append(shift * unshifted / (1.0 + (shift - 1.0) * unshifted))
    return torch.tensor(values, dtype=torch.float64)


def _reference_integrate(
    state: torch.Tensor,
    velocities: torch.Tensor,
    grid: torch.Tensor,
    start: int,
    end: int,
) -> torch.Tensor:
    """Integrate exactly the half-open interval-head block ``[start, end)``."""
    result = state.to(torch.float64).clone()
    for index in range(start, end):
        result += (grid[index + 1] - grid[index]) * velocities[index].to(torch.float64)
    return result


def _reference_fused_parameters(
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    grid: torch.Tensor,
    start: int,
    end: int,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Fuse per-interval linear parameters for the block ``[start, end)``."""
    denominator = grid[end] - grid[start]
    fused_weight = torch.zeros_like(weight[0], dtype=torch.float64)
    fused_bias = None if bias is None else torch.zeros_like(bias[0], dtype=torch.float64)

    for index in range(start, end):
        coefficient = (grid[index + 1] - grid[index]) / denominator
        fused_weight += coefficient * weight[index].to(torch.float64)
        if fused_bias is not None:
            fused_bias += coefficient * bias[index].to(torch.float64)

    return fused_weight, fused_bias


def test_shifted_grid_matches_hand_calculated_values_and_preserves_float32_intervals():
    grid = _reference_shifted_grid(grid_size=4, shift=5.0)

    expected = torch.tensor([1.0, 0.9375, 5.0 / 6.0, 0.625, 0.0], dtype=torch.float64)
    torch.testing.assert_close(grid, expected, rtol=0.0, atol=0.0)

    canonical_grid = _reference_shifted_grid(grid_size=128, shift=5.0)
    assert torch.all(torch.diff(canonical_grid.to(torch.float32)) < 0)
    assert canonical_grid.to(torch.bfloat16)[1] == 1.0

    for start in range(0, 128, 32):
        assert canonical_grid[start + 32] - canonical_grid[start] != 0


def test_half_open_integration_uses_only_selected_interval_heads():
    grid = _reference_shifted_grid(grid_size=4, shift=5.0)
    state = torch.tensor([3.0, -2.0])
    velocities = torch.tensor(
        [
            [1000.0, 1000.0],
            [2.0, -1.0],
            [-3.0, 4.0],
            [-1000.0, -1000.0],
        ]
    )

    result = _reference_integrate(state, velocities, grid, start=1, end=3)
    expected = (
        state.to(torch.float64)
        + (grid[2] - grid[1]) * velocities[1].to(torch.float64)
        + (grid[3] - grid[2]) * velocities[2].to(torch.float64)
    )

    torch.testing.assert_close(result, expected, rtol=0.0, atol=1e-15)
    torch.testing.assert_close(
        _reference_integrate(state, velocities, grid, start=2, end=2),
        state.to(torch.float64),
        rtol=0.0,
        atol=0.0,
    )


def test_final_half_open_block_advances_exactly_four_intervals():
    grid = _reference_shifted_grid(grid_size=8, shift=5.0)
    state = torch.tensor([1.25])
    velocities = torch.ones(8, 1)

    result = _reference_integrate(state, velocities, grid, start=4, end=8)
    expected = state.to(torch.float64) + grid[8] - grid[4]

    torch.testing.assert_close(result, expected, rtol=0.0, atol=1e-15)


def test_fused_projection_matches_weighted_sum_and_explicit_block_update():
    grid = _reference_shifted_grid(grid_size=4, shift=5.0)
    inputs = torch.tensor([[2.0, -1.0], [-0.5, 3.0]], dtype=torch.float64)
    weight = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[0.0, 2.0], [1.0, -1.0]],
            [[-1.0, 1.0], [2.0, 0.5]],
            [[3.0, -2.0], [-0.5, 1.5]],
        ],
        dtype=torch.float64,
    )
    bias = torch.tensor(
        [[0.0, 0.5], [1.0, -1.0], [2.0, 0.25], [-1.0, 3.0]],
        dtype=torch.float64,
    )
    original = (inputs.clone(), weight.clone(), bias.clone(), grid.clone())
    start, end = 1, 4

    fused_weight, fused_bias = _reference_fused_parameters(weight, bias, grid, start, end)
    fused_output = F.linear(inputs, fused_weight, fused_bias)

    head_outputs = torch.stack(
        [F.linear(inputs, weight[index], bias[index]) for index in range(weight.shape[0])]
    )
    coefficients = torch.tensor([1.0 / 9.0, 2.0 / 9.0, 2.0 / 3.0], dtype=torch.float64)
    explicit_output = torch.einsum("i,ibo->bo", coefficients, head_outputs[start:end])
    explicit_update = (
        (-5.0 / 48.0) * head_outputs[1]
        + (-5.0 / 24.0) * head_outputs[2]
        + (-5.0 / 8.0) * head_outputs[3]
    )

    torch.testing.assert_close(fused_output, explicit_output, rtol=1e-14, atol=1e-14)
    torch.testing.assert_close(
        (grid[end] - grid[start]) * fused_output,
        explicit_update,
        rtol=1e-14,
        atol=1e-14,
    )
    for value, unchanged in zip((inputs, weight, bias, grid), original):
        assert torch.equal(value, unchanged)
