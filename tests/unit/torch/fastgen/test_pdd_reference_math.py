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

import pytest
import torch
import torch.nn.functional as F

from modelopt.torch.fastgen.flow_matching import (
    fusion_coefficients,
    integrate_interval_velocities,
    make_shifted_flow_grid,
)


def _reference_shifted_grid(grid_size: int, shift: float, max_t: float) -> torch.Tensor:
    """Reproduce the frozen FastGen float64 schedule without production helpers."""
    unshifted = torch.linspace(max_t, 0.0, grid_size + 1, dtype=torch.float64)
    unshifted = unshifted.clamp(max=max_t)
    shifted = shift * unshifted / (1.0 + (shift - 1.0) * unshifted)
    return shifted.clamp(max=max_t)


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
    grid = _reference_shifted_grid(grid_size=4, shift=5.0, max_t=1.0)

    expected = torch.tensor([1.0, 0.9375, 5.0 / 6.0, 0.625, 0.0], dtype=torch.float64)
    torch.testing.assert_close(grid, expected, rtol=0.0, atol=0.0)

    canonical_grid = _reference_shifted_grid(grid_size=128, shift=5.0, max_t=0.999)
    assert torch.all(torch.diff(canonical_grid.to(torch.float32)) < 0)
    assert canonical_grid.to(torch.bfloat16)[1] == 1.0

    for start in range(0, 128, 32):
        assert canonical_grid[start + 32] - canonical_grid[start] != 0


def test_half_open_integration_uses_only_selected_interval_heads():
    grid = _reference_shifted_grid(grid_size=4, shift=5.0, max_t=1.0)
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
    grid = _reference_shifted_grid(grid_size=8, shift=5.0, max_t=1.0)
    state = torch.tensor([1.25])
    velocities = torch.ones(8, 1)

    result = _reference_integrate(state, velocities, grid, start=4, end=8)
    expected = state.to(torch.float64) + grid[8] - grid[4]

    torch.testing.assert_close(result, expected, rtol=0.0, atol=1e-15)


def test_fused_projection_matches_weighted_sum_and_explicit_block_update():
    grid = _reference_shifted_grid(grid_size=4, shift=5.0, max_t=1.0)
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


def test_production_shifted_grid_matches_independent_oracle():
    grid = make_shifted_flow_grid(grid_size=128, shift=5.0, max_t=0.999)
    oracle = _reference_shifted_grid(grid_size=128, shift=5.0, max_t=0.999).to(torch.float32)

    assert grid.dtype == torch.float32
    assert grid.shape == (129,)
    assert grid[0].item() == 0.9990000128746033
    assert grid[-1] == 0.0
    assert torch.all(torch.diff(grid) < 0)
    torch.testing.assert_close(grid, oracle, rtol=0, atol=0)

    direct_fp32 = torch.linspace(0.999, 0.0, 129, dtype=torch.float32)
    upper = torch.tensor(0.999, dtype=torch.float32)
    if upper.item() > 0.999:
        upper = torch.nextafter(upper, torch.tensor(float("-inf")))
    direct_fp32 = direct_fp32.clamp(max=upper)
    direct_fp32 = (5.0 * direct_fp32 / (1.0 + 4.0 * direct_fp32)).clamp(max=upper)
    assert torch.count_nonzero(grid != direct_fp32).item() == 52


def test_production_grid_promotes_low_precision_requests():
    grid = make_shifted_flow_grid(128, 5.0, max_t=0.999, dtype=torch.bfloat16)

    assert grid.dtype == torch.float32
    assert torch.all(torch.diff(grid) < 0)


@pytest.mark.parametrize(
    ("grid_size", "shift", "message"),
    [
        (0, 5.0, "positive integer"),
        (128, 0.5, "finite and >= 1"),
        (128, float("nan"), "finite and >= 1"),
    ],
)
def test_production_grid_rejects_invalid_boundaries(grid_size, shift, message):
    with pytest.raises(ValueError, match=message):
        make_shifted_flow_grid(grid_size, shift, max_t=0.999)


@pytest.mark.parametrize("max_t", [True, 1, 0])
def test_production_grid_rejects_non_float_max_t(max_t):
    with pytest.raises(TypeError, match="max_t must be a float"):
        make_shifted_flow_grid(4, 5.0, max_t=max_t)


@pytest.mark.parametrize("max_t", [float("nan"), float("inf"), float("-inf"), 0.0, -0.1, 1.0001])
def test_production_grid_rejects_invalid_max_t(max_t):
    with pytest.raises(ValueError, match="0 < max_t <= 1"):
        make_shifted_flow_grid(4, 5.0, max_t=max_t)


def test_production_grid_requires_explicit_max_t_and_accepts_one():
    with pytest.raises(TypeError, match="max_t"):
        make_shifted_flow_grid(4, 5.0)

    grid = make_shifted_flow_grid(4, 5.0, max_t=1.0)
    assert grid[0] == 1.0
    assert grid[-1] == 0.0


def test_production_grid_rejects_non_floating_dtype():
    with pytest.raises(TypeError, match="floating-point dtype"):
        make_shifted_flow_grid(128, 5.0, max_t=0.999, dtype=torch.int64)


def test_production_half_open_integration_matches_independent_oracle_per_sample():
    grid = make_shifted_flow_grid(4, 5.0, max_t=0.999, dtype=torch.float64)
    state = torch.tensor([[3.0, -2.0], [1.0, 4.0]], dtype=torch.bfloat16)
    velocities = torch.tensor(
        [
            [[1000.0, 1000.0], [2.0, -1.0], [-3.0, 4.0], [-1000.0, -1000.0]],
            [[1.0, 2.0], [-2.0, 3.0], [4.0, -5.0], [6.0, 7.0]],
        ],
        dtype=torch.bfloat16,
    )
    starts = torch.tensor([1, 2])
    ends = torch.tensor([3, 2])

    actual = integrate_interval_velocities(state, velocities, grid, starts, ends)
    expected = torch.stack(
        [
            _reference_integrate(state[0], velocities[0], grid, start=1, end=3),
            _reference_integrate(state[1], velocities[1], grid, start=2, end=2),
        ]
    )

    assert actual.dtype == torch.float64
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_production_integration_does_not_consume_excluded_nonfinite_heads():
    grid = make_shifted_flow_grid(4, 5.0, max_t=0.999, dtype=torch.float64)
    state = torch.tensor([[3.0, -2.0]], dtype=torch.float64)
    velocities = torch.tensor(
        [[[torch.nan, torch.nan], [2.0, -1.0], [-3.0, 4.0], [torch.inf, -torch.inf]]],
        dtype=torch.float64,
    )

    actual = integrate_interval_velocities(state, velocities, grid, start=1, end=3)
    expected = _reference_integrate(state[0], velocities[0], grid, start=1, end=3)[None]

    assert torch.all(torch.isfinite(actual))
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_production_integration_promotes_bfloat16_math_to_float32():
    grid = make_shifted_flow_grid(4, 5.0, max_t=0.999)
    state = torch.zeros(1, 2, dtype=torch.bfloat16)
    velocities = torch.ones(1, 4, 2, dtype=torch.bfloat16)

    result = integrate_interval_velocities(state, velocities, grid, start=0, end=4)

    assert result.dtype == torch.float32
    torch.testing.assert_close(result, torch.full((1, 2), -0.999))


def test_production_integration_rejects_out_of_range_half_open_block():
    grid = make_shifted_flow_grid(4, 5.0, max_t=0.999)
    state = torch.zeros(1, 2)
    velocities = torch.ones(1, 4, 2)

    with pytest.raises(ValueError, match="0 <= start <= end <= 4"):
        integrate_interval_velocities(state, velocities, grid, start=3, end=5)


def test_production_fusion_coefficients_match_independent_oracle():
    grid = make_shifted_flow_grid(4, 5.0, max_t=0.999, dtype=torch.float64)
    actual = fusion_coefficients(grid, start=1, end=4)
    oracle_grid = _reference_shifted_grid(4, 5.0, 0.999)
    expected = torch.stack(
        [
            (oracle_grid[index + 1] - oracle_grid[index]) / (oracle_grid[4] - oracle_grid[1])
            for index in range(1, 4)
        ]
    )

    assert torch.all(actual > 0)
    torch.testing.assert_close(actual.sum(), torch.tensor(1.0, dtype=torch.float64))
    torch.testing.assert_close(actual, expected, rtol=1e-14, atol=1e-14)


def test_production_fusion_coefficients_reject_empty_block():
    grid = make_shifted_flow_grid(4, 5.0, max_t=0.999)
    with pytest.raises(ValueError, match="0 <= start < end <= 4"):
        fusion_coefficients(grid, start=2, end=2)


def test_production_helpers_do_not_extract_meta_tensor_scalars():
    grid = make_shifted_flow_grid(4, 5.0, max_t=0.999, device="meta")
    state = torch.empty(2, 3, device="meta")
    velocities = torch.empty(2, 4, 3, device="meta")

    integrated = integrate_interval_velocities(
        state,
        velocities,
        grid,
        start=torch.tensor([0, 2], device="meta"),
        end=torch.tensor([2, 4], device="meta"),
    )
    coefficients = fusion_coefficients(grid, start=1, end=4)

    assert integrated.device.type == "meta"
    assert integrated.shape == state.shape
    assert coefficients.device.type == "meta"
    assert coefficients.shape == (3,)
