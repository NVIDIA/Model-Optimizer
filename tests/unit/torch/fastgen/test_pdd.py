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

"""Tests for framework-neutral PDD configuration, math, projection, loss, and sampling."""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import nn

from modelopt.torch.fastgen import (
    PDDConfig,
    PDDLayerSpec,
    PDDOutputProjection,
    PDDPipeline,
    convert_to_pdd_output_projection,
    load_pdd_config,
)
from modelopt.torch.fastgen.flow_matching import (
    fusion_coefficients,
    integrate_interval_velocities,
    make_shifted_flow_grid,
)


def test_pdd_config_defaults_and_yaml_loading(tmp_path):
    config = PDDConfig()
    assert (
        config.grid_size,
        config.grid_max_t,
        config.flow_shift,
        config.block_size_min,
        config.block_size_max,
        config.inference_blocks,
    ) == (128, 0.999, 5.0, 4, 64, (32, 32, 32, 32))

    path = tmp_path / "pdd.yaml"
    path.write_text("inference_blocks: [64, 64]\nteacher_integrator: midpoint\n")
    loaded = load_pdd_config(path)
    assert loaded.inference_blocks == (64, 64)
    assert loaded.teacher_integrator == "midpoint"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"inference_blocks": [32, 32]},
        {"teacher_integrator": "heun"},
        {"grid_size": 1024, "inference_blocks": [256] * 4},
    ],
)
def test_pdd_config_rejects_invalid_schedules(kwargs):
    with pytest.raises(ValueError):
        PDDConfig(**kwargs)


def _reference_grid(size: int, shift: float, max_t: float) -> torch.Tensor:
    time = torch.linspace(max_t, 0.0, size + 1, dtype=torch.float64).clamp_max(max_t)
    return (shift * time / (1.0 + (shift - 1.0) * time)).clamp_max(max_t)


def test_grid_integration_and_fusion_match_direct_math():
    canonical = make_shifted_flow_grid(128, 5.0, max_t=0.999)
    torch.testing.assert_close(canonical, _reference_grid(128, 5.0, 0.999).float(), rtol=0, atol=0)
    direct = torch.linspace(0.999, 0.0, 129, dtype=torch.float32)
    upper = torch.tensor(0.999, dtype=torch.float32)
    if upper.item() > 0.999:
        upper = torch.nextafter(upper, torch.tensor(float("-inf")))
    direct = direct.clamp_max(upper)
    direct = (5.0 * direct / (1.0 + 4.0 * direct)).clamp_max(upper)
    assert not torch.equal(canonical, direct)

    grid = make_shifted_flow_grid(4, 5.0, max_t=0.999, dtype=torch.float64)
    torch.testing.assert_close(grid, _reference_grid(4, 5.0, 0.999), rtol=0, atol=0)

    state = torch.tensor([[3.0, -2.0], [1.0, 4.0]])
    velocities = torch.arange(16, dtype=torch.float32).reshape(2, 4, 2) / 7
    starts = torch.tensor([1, 2])
    ends = torch.tensor([4, 2])
    expected = state.double()
    for batch in range(2):
        for index in range(int(starts[batch]), int(ends[batch])):
            expected[batch] += (grid[index + 1] - grid[index]) * velocities[batch, index].double()

    actual = integrate_interval_velocities(state, velocities, grid, starts, ends)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    coefficients = fusion_coefficients(grid, 1, 4)
    expected_coefficients = torch.diff(grid[1:5]) / (grid[4] - grid[1])
    torch.testing.assert_close(coefficients, expected_coefficients, rtol=0, atol=0)
    torch.testing.assert_close(coefficients.sum(), torch.tensor(1.0, dtype=torch.float64))


class _ProjectionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(2, 6)
        with torch.no_grad():
            self.proj.weight.copy_(torch.arange(12).reshape(6, 2) / 7)
            self.proj.bias.copy_(torch.arange(6) / 11)


def _projection_spec(layout: str) -> PDDLayerSpec:
    return PDDLayerSpec("proj", layout, output_channels=2 if layout == "patch_major" else None)


def _heads(output: torch.Tensor, layout: str) -> torch.Tensor:
    if layout == "channel_major":
        return output.reshape(output.shape[0], 3, 6)
    return output.reshape(output.shape[0], 3, 3, 2).permute(0, 2, 1, 3).reshape(-1, 3, 6)


@pytest.mark.parametrize("layout", ["channel_major", "patch_major"])
def test_projection_conversion_and_fusion(layout):
    model = _ProjectionModel()
    inputs = torch.tensor([[0.25, -1.0], [2.0, 0.5]])
    base_output = model.proj(inputs)
    spec = _projection_spec(layout)

    projection = convert_to_pdd_output_projection(model, spec, grid_size=3)
    assert isinstance(projection, PDDOutputProjection)
    assert convert_to_pdd_output_projection(model, spec, grid_size=3) is projection
    torch.testing.assert_close(
        _heads(projection(inputs), layout),
        base_output[:, None].expand(-1, 3, -1),
    )

    with torch.no_grad():
        projection.weight.copy_(
            torch.arange(projection.weight.numel()).reshape_as(projection.weight) / 17
        )
        projection.bias.copy_(torch.arange(projection.bias.numel()) / 19)
    all_heads = _heads(projection(inputs), layout)
    grid = torch.tensor([1.0, 0.7, 0.2, 0.0])
    expected = torch.einsum("n,bno->bo", fusion_coefficients(grid, 1, 3), all_heads[:, 1:3])
    torch.testing.assert_close(projection(inputs, fusion=(1, 3, grid)), expected)


class _Student(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.25))
        self.bias = nn.Parameter(torch.arange(8, dtype=torch.float32).reshape(4, 2) / 7)

    def heads(self, state: torch.Tensor) -> torch.Tensor:
        return self.scale * state[:, None] + self.bias[None]


class _Teacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(-0.375))

    def forward(self, state: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        return self.scale * state + time[:, None]


class _Adapter:
    def student_all_heads(
        self,
        model: _Student,
        state: torch.Tensor,
        _time: torch.Tensor,
        **_kwargs: Any,
    ) -> torch.Tensor:
        return model.heads(state)

    def student_fused_block(
        self,
        model: _Student,
        state: torch.Tensor,
        _time: torch.Tensor,
        *,
        start: int,
        end: int,
        grid: torch.Tensor,
        **_kwargs: Any,
    ) -> torch.Tensor:
        return torch.einsum(
            "n,bnd->bd",
            fusion_coefficients(grid, start, end).float(),
            model.heads(state)[:, start:end],
        )

    def teacher_velocity(
        self,
        model: _Teacher,
        state: torch.Tensor,
        time: torch.Tensor,
        **_kwargs: Any,
    ) -> torch.Tensor:
        return model(state, time)


def _pipeline(*, integrator: str = "euler", teacher: bool = True) -> PDDPipeline:
    config = PDDConfig(
        grid_size=4,
        block_size_min=1,
        block_size_max=4,
        inference_blocks=[2, 2],
        teacher_integrator=integrator,
    )
    return PDDPipeline(_Student(), _Teacher() if teacher else None, config, _Adapter())


def _rf_state(data: torch.Tensor, noise: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    time = time.double()
    while time.ndim < data.ndim:
        time = time.unsqueeze(-1)
    return (data.double() * (1 - time) + noise.double() * time).float()


def _integrate(
    state: torch.Tensor,
    heads: torch.Tensor,
    grid: torch.Tensor,
    start: torch.Tensor,
    end: torch.Tensor,
) -> torch.Tensor:
    result = state.clone()
    for batch in range(state.shape[0]):
        for index in range(int(start[batch]), int(end[batch])):
            result[batch] += (grid[index + 1] - grid[index]) * heads[batch, index]
    return result


def test_pdd_loss_matches_direct_objective_and_routes_gradients():
    pipeline = _pipeline()
    data = torch.tensor([[1.0, -2.0], [-1.5, 0.25]])
    noise = torch.tensor([[0.25, 1.5], [2.0, -1.0]])
    n = torch.tensor([0, 2])
    k = torch.tensor([2, 3])

    loss, metrics = pipeline.compute_loss(data, noise=noise, n=n, k=k)
    grid = pipeline.time_grid()
    state = _rf_state(data, noise, grid[n])
    heads = pipeline.student.heads(state)
    state_k = _integrate(state, heads, grid, n, k)
    target = pipeline.teacher(state_k, grid[k])
    expected = (heads[torch.arange(2), k] - target).square().mean()
    torch.testing.assert_close(loss, expected)

    loss.backward()
    assert pipeline.student.scale.grad is not None
    assert torch.count_nonzero(pipeline.student.bias.grad[:2]) == 0
    assert torch.count_nonzero(pipeline.student.bias.grad[2:]) > 0
    assert all(parameter.grad is None for parameter in pipeline.teacher.parameters())
    assert torch.equal(metrics["target_span"], k - n + 1)


def test_data_free_midpoint_loss_and_state_progression_match_direct_math():
    pipeline = _pipeline(integrator="midpoint")
    state = torch.tensor([[0.25, 1.5]])
    n = torch.tensor([0])
    k = torch.tensor([2])

    loss, _, next_state, next_n = pipeline.compute_data_free_loss(state, n=n, k=k)
    grid = pipeline.time_grid()
    heads = pipeline.student.heads(state)
    state_k = _integrate(state, heads, grid, n, k)
    first_velocity = pipeline.teacher(state_k, grid[k])
    delta = grid[k + 1] - grid[k]
    midpoint_target = pipeline.teacher(
        state_k + 0.5 * delta[:, None] * first_velocity,
        grid[k] + 0.5 * delta,
    )
    expected_loss = (heads[:, 2] - midpoint_target).square().mean()
    expected_next = _integrate(state, heads, grid, n, n + 1)

    torch.testing.assert_close(loss, expected_loss)
    torch.testing.assert_close(next_state, expected_next)
    assert torch.equal(next_n, n + 1)
    assert not next_state.requires_grad


def test_fused_sampling_matches_explicit_block_updates_without_teacher():
    pipeline = _pipeline(teacher=False)
    noise = torch.tensor([[-0.21963761746883392, -1.409722924232483]])
    actual = pipeline.sample(noise, blocks=[1, 3])

    grid = pipeline.time_grid()
    fusion_grid = make_shifted_flow_grid(4, 5.0, max_t=0.999, dtype=torch.float64)
    expected = (noise.double() * 0.999).float()
    from_cast_grid = (noise.double() * grid[0].double()).float()
    assert not torch.equal(expected, from_cast_grid)
    start = 0
    for end in (1, 4):
        velocity = torch.einsum(
            "n,bnd->bd",
            fusion_coefficients(fusion_grid, start, end).float(),
            pipeline.student.heads(expected)[:, start:end],
        )
        expected += (grid[end] - grid[start]) * velocity
        start = end

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
