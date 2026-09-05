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

from modelopt.torch.fastgen.flow_matching import (
    add_noise,
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


def _reference_rf_forward_process(
    data: torch.Tensor,
    noise: torch.Tensor,
    time: torch.Tensor,
) -> torch.Tensor:
    """Reproduce FastGen's float64 RF forward process without production helpers."""
    original_dtype = data.dtype
    data_64 = data.to(torch.float64)
    noise_64 = noise.to(torch.float64)
    time_64 = time.to(torch.float64)
    while time_64.ndim < data_64.ndim:
        time_64 = time_64.unsqueeze(-1)
    state_64 = data_64 * (1.0 - time_64) + noise_64 * time_64
    return state_64.to(original_dtype)


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


def test_production_shifted_grid_matches_independent_oracle():
    grid = make_shifted_flow_grid(grid_size=128, shift=5.0, max_t=0.999)
    oracle = _reference_shifted_grid(grid_size=128, shift=5.0, max_t=0.999).to(torch.float32)

    assert grid.dtype == torch.float32
    assert grid.shape == (129,)
    assert grid[0].item() == 0.9990000128746033
    assert grid[-1] == 0.0
    assert torch.all(torch.diff(grid) < 0)
    torch.testing.assert_close(grid, oracle, rtol=0, atol=0)
    assert make_shifted_flow_grid(128, 5.0, max_t=0.999, dtype=torch.bfloat16).dtype == (
        torch.float32
    )

    direct_fp32 = torch.linspace(0.999, 0.0, 129, dtype=torch.float32)
    upper = torch.tensor(0.999, dtype=torch.float32)
    if upper.item() > 0.999:
        upper = torch.nextafter(upper, torch.tensor(float("-inf")))
    direct_fp32 = direct_fp32.clamp(max=upper)
    direct_fp32 = (5.0 * direct_fp32 / (1.0 + 4.0 * direct_fp32)).clamp(max=upper)
    assert not torch.equal(grid, direct_fp32)


def test_production_rf_forward_process_matches_float64_intermediate_oracle():
    data = torch.tensor(
        [[-0.2654421329498291, 0.5161616802215576, -0.7285917401313782]],
        dtype=torch.float32,
    )
    noise = torch.tensor(
        [[0.3856363296508789, -0.34849217534065247, -0.11881951987743378]],
        dtype=torch.float32,
    )
    grid = make_shifted_flow_grid(128, 5.0, max_t=0.999)
    time = grid[:1]
    original = (data.clone(), noise.clone(), time.clone(), grid.clone())

    expected = _reference_rf_forward_process(data, noise, time)
    actual = add_noise(data, noise, time)
    stale_direct = (1.0 - time[:, None]) * data + time[:, None] * noise

    assert time.item() == 0.9990000128746033
    assert torch.equal(
        stale_direct,
        torch.tensor([[0.3849852383136749, -0.34762755036354065, -0.11942929029464722]]),
    )
    assert torch.equal(
        expected,
        torch.tensor([[0.3849852681159973, -0.34762752056121826, -0.11942928284406662]]),
    )
    assert not torch.equal(stale_direct, expected)
    assert torch.equal(actual, expected)
    assert actual.dtype == torch.float32
    for value, unchanged in zip((data, noise, time, grid), original):
        assert torch.equal(value, unchanged)


@pytest.mark.parametrize(
    ("args", "kwargs", "error"),
    [
        ((0, 5.0), {"max_t": 0.999}, ValueError),
        ((128, 0.5), {"max_t": 0.999}, ValueError),
        ((4, 5.0), {"max_t": 1}, TypeError),
        ((4, 5.0), {"max_t": 0.0}, ValueError),
        ((4, 5.0), {"max_t": 0.999, "dtype": torch.int64}, TypeError),
    ],
)
def test_production_grid_rejects_invalid_inputs(args, kwargs, error):
    with pytest.raises(error):
        make_shifted_flow_grid(*args, **kwargs)


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


def test_production_integration_promotes_bfloat16_math_to_float32():
    grid = make_shifted_flow_grid(4, 5.0, max_t=0.999)
    state = torch.zeros(1, 2, dtype=torch.bfloat16)
    velocities = torch.ones(1, 4, 2, dtype=torch.bfloat16)

    result = integrate_interval_velocities(state, velocities, grid, start=0, end=4)

    assert result.dtype == torch.float32
    torch.testing.assert_close(result, torch.full((1, 2), -0.999))


def test_production_helpers_reject_invalid_blocks():
    grid = make_shifted_flow_grid(4, 5.0, max_t=0.999)
    state = torch.zeros(1, 2)
    velocities = torch.ones(1, 4, 2)

    with pytest.raises(ValueError, match="0 <= start <= end <= 4"):
        integrate_interval_velocities(state, velocities, grid, start=3, end=5)
    with pytest.raises(ValueError, match="0 <= start < end <= 4"):
        fusion_coefficients(grid, start=2, end=2)


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
