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

"""Analytic tests for the framework-neutral PDD objective and fused sampler."""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import nn

from modelopt.torch.fastgen import PDDConfig, PDDPipeline
from modelopt.torch.fastgen.flow_matching import fusion_coefficients, make_shifted_flow_grid


class _HeadModel(nn.Module):
    """Small state-dependent multi-head velocity model with explicit parameters."""

    def __init__(self, grid_size: int, width: int) -> None:
        super().__init__()
        self.state_scale = nn.Parameter(torch.tensor(0.25))
        self.head_bias = nn.Parameter(
            torch.arange(grid_size * width, dtype=torch.float32).reshape(grid_size, width) / 7
        )

    def all_heads(self, state: torch.Tensor) -> torch.Tensor:
        return self.state_scale * state[:, None] + self.head_bias[None]


class _Teacher(nn.Module):
    """Analytic teacher velocity ``scale * state + time + bias``."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(-0.375))
        self.register_buffer("bias", torch.arange(width, dtype=torch.float32) / 13)

    def forward(self, state: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        return self.scale * state + time[:, None] + self.bias


class _RecordingAdapter:
    """Canonical toy adapter that records every architecture boundary call."""

    def __init__(self) -> None:
        self.student_calls: list[dict[str, Any]] = []
        self.fused_calls: list[dict[str, Any]] = []
        self.teacher_calls: list[dict[str, Any]] = []
        self.bad_student_shape = False
        self.bad_fused_dtype = False
        self.low_precision_outputs = False

    def student_all_heads(
        self,
        model: _HeadModel,
        state: torch.Tensor,
        time: torch.Tensor,
        *,
        condition: Any = None,
        **model_kwargs: Any,
    ) -> torch.Tensor:
        self.student_calls.append(
            {
                "state": state.detach().clone(),
                "time": time.detach().clone(),
                "condition": condition,
                "kwargs": model_kwargs,
            }
        )
        output = model.all_heads(state)
        if self.bad_student_shape:
            output = output[:, :-1]
        return output.to(torch.bfloat16) if self.low_precision_outputs else output

    def student_fused_block(
        self,
        model: _HeadModel,
        state: torch.Tensor,
        time: torch.Tensor,
        *,
        start: int,
        end: int,
        grid: torch.Tensor,
        condition: Any = None,
        **model_kwargs: Any,
    ) -> torch.Tensor:
        self.fused_calls.append(
            {
                "state": state.detach().clone(),
                "time": time.detach().clone(),
                "start": start,
                "end": end,
                "grid": grid.detach().clone(),
                "condition": condition,
                "kwargs": model_kwargs,
            }
        )
        # The real PDD projection derives coefficients from the float64 grid,
        # then casts them to FP32 before fusing FP32 master parameters.
        coefficients = fusion_coefficients(grid, start, end).to(torch.float32)
        heads = model.all_heads(state)[:, start:end]
        output = torch.einsum("n,bnd->bd", coefficients, heads)
        return output.to(torch.int64) if self.bad_fused_dtype else output

    def teacher_velocity(
        self,
        model: _Teacher,
        state: torch.Tensor,
        time: torch.Tensor,
        *,
        condition: Any = None,
        negative_condition: Any = None,
        **model_kwargs: Any,
    ) -> torch.Tensor:
        self.teacher_calls.append(
            {
                "state": state.detach().clone(),
                "time": time.detach().clone(),
                "requires_grad": state.requires_grad,
                "condition": condition,
                "negative_condition": negative_condition,
                "kwargs": model_kwargs,
            }
        )
        output = model(state, time)
        return output.to(torch.bfloat16) if self.low_precision_outputs else output


def _config(*, teacher_integrator: str = "euler") -> PDDConfig:
    return PDDConfig(
        grid_size=8,
        grid_max_t=0.999,
        flow_shift=5.0,
        block_size_min=2,
        block_size_max=4,
        inference_blocks=[4, 4],
        teacher_integrator=teacher_integrator,
    )


def _pipeline(*, teacher_integrator: str = "euler") -> tuple[PDDPipeline, _RecordingAdapter]:
    adapter = _RecordingAdapter()
    pipeline = PDDPipeline(
        _HeadModel(grid_size=8, width=3),
        _Teacher(width=3),
        _config(teacher_integrator=teacher_integrator),
        adapter,
    )
    return pipeline, adapter


def _explicit_integrate_per_sample(
    state: torch.Tensor,
    heads: torch.Tensor,
    grid: torch.Tensor,
    n: torch.Tensor,
    k: torch.Tensor,
) -> torch.Tensor:
    result = state.clone()
    for batch_index in range(state.shape[0]):
        for head_index in range(int(n[batch_index]), int(k[batch_index])):
            result[batch_index] += (grid[head_index + 1] - grid[head_index]) * heads[
                batch_index, head_index
            ]
    return result


def _reference_rf_forward_process(
    data: torch.Tensor,
    noise: torch.Tensor,
    time: torch.Tensor,
) -> torch.Tensor:
    """Reproduce the PDD RF input staging without production helpers."""
    data_64 = data.to(torch.float32).to(torch.float64)
    noise_64 = noise.to(torch.float32).to(torch.float64)
    time_64 = time.to(torch.float32).to(torch.float64)
    while time_64.ndim < data_64.ndim:
        time_64 = time_64.unsqueeze(-1)
    return (data_64 * (1.0 - time_64) + noise_64 * time_64).to(torch.float32)


def test_euler_loss_matches_analytic_empty_and_tail_reconstruction() -> None:
    pipeline, adapter = _pipeline()
    data = torch.tensor([[1.0, -2.0, 0.5], [-1.5, 0.25, 2.0]])
    noise = torch.tensor([[0.25, 1.5, -0.5], [2.0, -1.0, 0.75]])
    n = torch.tensor([0, 6])
    k = torch.tensor([0, 7])

    loss, metrics = pipeline.compute_loss(
        data,
        noise=noise,
        condition="positive",
        negative_condition="negative",
        model_kwargs={"tag": 17},
        n=n,
        k=k,
    )

    grid = pipeline.time_grid()
    x_n = _reference_rf_forward_process(data, noise, grid[n])
    heads = pipeline.student.all_heads(x_n)
    x_bar_k = _explicit_integrate_per_sample(x_n, heads, grid, n, k)
    teacher_target = pipeline.teacher(x_bar_k, grid[k])
    student_target = heads[torch.arange(data.shape[0]), k]
    expected_loss = (student_target - teacher_target).square().mean()

    torch.testing.assert_close(loss, expected_loss)
    torch.testing.assert_close(adapter.student_calls[0]["state"], x_n)
    torch.testing.assert_close(adapter.student_calls[0]["time"], grid[n])
    torch.testing.assert_close(adapter.teacher_calls[0]["state"], x_bar_k)
    torch.testing.assert_close(adapter.teacher_calls[0]["time"], grid[k])
    assert adapter.teacher_calls[0]["requires_grad"] is False
    assert adapter.student_calls[0]["condition"] == "positive"
    assert adapter.student_calls[0]["kwargs"] == {"tag": 17}
    assert adapter.teacher_calls[0]["negative_condition"] == "negative"
    assert len(adapter.student_calls) == len(adapter.teacher_calls) == 1
    assert torch.equal(metrics["n"], n)
    assert torch.equal(metrics["k"], k)
    assert torch.equal(metrics["target_span"], torch.tensor([1, 2]))
    assert metrics["student_target_mse"].shape == (2,)
    assert metrics["all_student_heads_finite"].shape == (2,)
    assert bool(metrics["loss_finite"])
    assert all(not value.requires_grad for value in metrics.values())


def test_midpoint_target_uses_exact_final_interval_midpoint() -> None:
    pipeline, adapter = _pipeline(teacher_integrator="midpoint")
    data = torch.tensor([[1.0, -2.0, 0.5], [-1.5, 0.25, 2.0]])
    noise = torch.tensor([[0.25, 1.5, -0.5], [2.0, -1.0, 0.75]])
    n = torch.tensor([4, 6])
    k = torch.tensor([6, 7])

    loss, _ = pipeline.compute_loss(data, noise=noise, n=n, k=k)

    grid = pipeline.time_grid()
    x_n = _reference_rf_forward_process(data, noise, grid[n])
    heads = pipeline.student.all_heads(x_n)
    x_bar_k = _explicit_integrate_per_sample(x_n, heads, grid, n, k)
    first_velocity = pipeline.teacher(x_bar_k, grid[k])
    delta = grid[k + 1] - grid[k]
    midpoint_state = x_bar_k + 0.5 * delta.reshape(2, 1) * first_velocity
    midpoint_time = grid[k] + 0.5 * delta
    midpoint_target = pipeline.teacher(midpoint_state, midpoint_time)
    expected_loss = (heads[torch.arange(data.shape[0]), k] - midpoint_target).square().mean()

    torch.testing.assert_close(loss, expected_loss)
    assert len(adapter.student_calls) == 1
    assert len(adapter.teacher_calls) == 2
    torch.testing.assert_close(adapter.teacher_calls[0]["state"], x_bar_k)
    torch.testing.assert_close(adapter.teacher_calls[0]["time"], grid[k])
    torch.testing.assert_close(adapter.teacher_calls[1]["state"], midpoint_state)
    torch.testing.assert_close(adapter.teacher_calls[1]["time"], midpoint_time)


def test_data_free_loss_matches_algorithm_3_and_advances_exact_minimum_block() -> None:
    pipeline, adapter = _pipeline(teacher_integrator="midpoint")
    state = torch.tensor([[0.25, 1.5, -0.5], [2.0, -1.0, 0.75]])
    n = torch.tensor([0, 6])
    k = torch.tensor([3, 7])

    loss, metrics, next_state, next_n = pipeline.compute_data_free_loss(
        state,
        n=n,
        k=k,
        condition="positive",
        negative_condition="negative",
    )

    grid = pipeline.time_grid()
    heads = pipeline.student.all_heads(state)
    x_k = _explicit_integrate_per_sample(state, heads, grid, n, k)
    first_velocity = pipeline.teacher(x_k, grid[k])
    delta = grid[k + 1] - grid[k]
    midpoint_state = x_k + 0.5 * delta[:, None] * first_velocity
    midpoint_target = pipeline.teacher(midpoint_state, grid[k] + 0.5 * delta)
    selected = heads[torch.arange(state.shape[0]), k]
    expected_next = _explicit_integrate_per_sample(
        state,
        heads,
        grid,
        n,
        n + pipeline.config.block_size_min,
    )

    torch.testing.assert_close(loss, (selected - midpoint_target).square().mean())
    torch.testing.assert_close(next_state, expected_next)
    assert next_state.requires_grad is False
    assert torch.equal(next_n, torch.tensor([2, 8]))
    assert torch.equal(metrics["n"], n)
    assert torch.equal(metrics["k"], k)
    assert adapter.student_calls[0]["condition"] == "positive"
    assert adapter.teacher_calls[0]["negative_condition"] == "negative"


def test_selected_head_low_precision_outputs_use_float32_mse() -> None:
    pipeline, adapter = _pipeline()
    adapter.low_precision_outputs = True
    data = torch.tensor([[1.0, -2.0, 0.5]])
    noise = torch.tensor([[0.25, 1.5, -0.5]])
    n = torch.tensor([0])
    k = torch.tensor([0])

    loss, _ = pipeline.compute_loss(data, noise=noise, n=n, k=k)

    grid = pipeline.time_grid()
    x_n = _reference_rf_forward_process(data, noise, grid[n])
    selected = pipeline.student.all_heads(x_n)[:, 0].to(torch.bfloat16).float()
    teacher = pipeline.teacher(x_n, grid[k]).to(torch.bfloat16).float()
    expected = (selected - teacher).square().mean()

    assert loss.dtype == torch.float32
    torch.testing.assert_close(loss, expected)


def test_sampled_indices_stay_on_exact_uniform_support() -> None:
    pipeline, _ = _pipeline()
    generator = torch.Generator().manual_seed(1234)

    _, metrics = pipeline.compute_loss(
        torch.ones(4096, 3),
        noise=torch.zeros(4096, 3),
        generator=generator,
    )
    n, k = metrics["n"], metrics["k"]

    assert set(n.tolist()) == {0, 2, 4, 6}
    assert torch.all(n.remainder(2) == 0)
    assert torch.all(k >= n)
    assert torch.all(k < torch.minimum(n + 4, torch.full_like(n, 8)))
    for n_value in (0, 2, 4, 6):
        observed = set(k[n == n_value].tolist())
        assert observed == set(range(n_value, min(n_value + 4, 8)))


@pytest.mark.parametrize("blocks", [None, [2, 2, 2, 2], [1, 7]])
def test_fused_sampler_matches_explicit_block_updates(blocks) -> None:
    pipeline, adapter = _pipeline()
    noise = torch.tensor([[1.0, -2.0, 0.5], [-0.25, 0.75, 1.5]], dtype=torch.bfloat16)

    actual = pipeline.sample(
        noise,
        condition="prompt",
        blocks=blocks,
        model_kwargs={"tag": 23},
    )

    resolved = [4, 4] if blocks is None else blocks
    grid = pipeline.time_grid()
    fusion_grid = make_shifted_flow_grid(
        pipeline.config.grid_size,
        pipeline.config.flow_shift,
        max_t=pipeline.config.grid_max_t,
        dtype=torch.float64,
    )
    expected = (noise.to(torch.float64) * pipeline.config.grid_max_t).to(torch.float32)
    start = 0
    for block in resolved:
        end = start + block
        heads = pipeline.student.all_heads(expected)
        for index in range(start, end):
            expected = expected + (grid[index + 1] - grid[index]) * heads[:, index]
        start = end

    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6)
    assert len(adapter.fused_calls) == len(resolved)
    start = 0
    for call, block in zip(adapter.fused_calls, resolved):
        end = start + block
        assert (call["start"], call["end"]) == (start, end)
        torch.testing.assert_close(call["time"], grid[start].expand(noise.shape[0]))
        torch.testing.assert_close(call["grid"], fusion_grid)
        assert call["grid"].dtype == torch.float64
        assert call["condition"] == "prompt"
        assert call["kwargs"] == {"tag": 23}
        start = end


def test_sampling_does_not_require_a_teacher() -> None:
    pipeline, adapter = _pipeline()
    sampler = PDDPipeline(pipeline.student, None, pipeline.config, adapter)

    result = sampler.sample(torch.ones(1, 3))

    assert result.shape == (1, 3)
    with pytest.raises(RuntimeError, match="training requires a teacher"):
        sampler.compute_loss(torch.ones(1, 3))


def test_fused_sampler_uses_precast_max_time_once_for_raw_noise() -> None:
    pipeline, adapter = _pipeline()
    noise = torch.tensor(
        [[-0.21963761746883392, -1.409722924232483, 1.8951480388641357]],
        dtype=torch.float32,
    )

    pipeline.sample(noise)

    expected = (noise.to(torch.float64) * 0.999).to(torch.float32)
    from_cast_grid = (noise.to(torch.float64) * pipeline.time_grid()[0].to(torch.float64)).to(
        torch.float32
    )
    double_scaled = (expected.to(torch.float64) * 0.999).to(torch.float32)
    assert not torch.equal(expected, from_cast_grid)
    assert not torch.equal(expected, double_scaled)
    assert torch.equal(adapter.fused_calls[0]["state"], expected)


@pytest.mark.parametrize(
    ("n", "k", "message"),
    [
        (torch.tensor([1]), torch.tensor([1]), "n must be aligned"),
        (torch.tensor([8]), torch.tensor([8]), "n must be aligned"),
        (torch.tensor([2]), torch.tensor([1]), "k must satisfy"),
        (torch.tensor([2]), torch.tensor([6]), "k must satisfy"),
        (torch.tensor([6]), torch.tensor([8]), "k must satisfy"),
    ],
)
def test_loss_rejects_indices_outside_trained_support(n, k, message) -> None:
    pipeline, _ = _pipeline()
    with pytest.raises(RuntimeError, match=message):
        pipeline.compute_loss(torch.ones(1, 3), noise=torch.zeros(1, 3), n=n, k=k)


def test_pipeline_rejects_invalid_shapes_dtypes_and_blocks() -> None:
    pipeline, adapter = _pipeline()
    with pytest.raises(TypeError, match="data must be a tensor"):
        pipeline.compute_loss({"state": torch.ones(1, 3)})  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="noise must be a tensor"):
        pipeline.sample({"state": torch.ones(1, 3)})  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="real floating-point"):
        pipeline.compute_loss(torch.ones(1, 3, dtype=torch.int64))
    with pytest.raises(ValueError, match="noise must match"):
        pipeline.compute_loss(torch.ones(1, 3), noise=torch.ones(1, 2))
    with pytest.raises(TypeError, match="n must use an integer dtype"):
        pipeline.compute_loss(
            torch.ones(1, 3),
            noise=torch.zeros(1, 3),
            n=torch.tensor([0.0]),
            k=torch.tensor([0]),
        )
    with pytest.raises(TypeError, match="model_kwargs must be a mapping"):
        pipeline.sample(torch.ones(1, 3), model_kwargs=[])  # type: ignore[arg-type]
    for blocks in ([2, 2], [], [0, 8], [4, 4.0], [2, 2, 2, 4]):
        with pytest.raises(ValueError):
            pipeline.sample(torch.ones(1, 3), blocks=blocks)

    adapter.bad_student_shape = True
    with pytest.raises(ValueError, match="student_all_heads must return shape"):
        pipeline.compute_loss(
            torch.ones(1, 3),
            noise=torch.zeros(1, 3),
            n=torch.tensor([0]),
            k=torch.tensor([0]),
        )
    adapter.bad_fused_dtype = True
    with pytest.raises(TypeError, match="student_fused_block must return a real floating-point"):
        pipeline.sample(torch.ones(1, 3))


def test_pipeline_freezes_teacher_but_not_student() -> None:
    pipeline, _ = _pipeline()

    assert pipeline.teacher.training is False
    assert all(not parameter.requires_grad for parameter in pipeline.teacher.parameters())
    assert all(parameter.requires_grad for parameter in pipeline.student.parameters())


def test_only_selected_head_and_shared_backbone_receive_gradients() -> None:
    pipeline, _ = _pipeline()

    loss, metrics = pipeline.compute_loss(
        torch.tensor([[1.0, -0.5, 0.25]]),
        noise=torch.tensor([[-0.25, 2.0, 0.5]]),
        n=torch.tensor([0]),
        k=torch.tensor([3]),
    )
    loss.backward()

    student = pipeline.student
    assert student.state_scale.grad is not None
    assert torch.isfinite(student.state_scale.grad)
    assert student.head_bias.grad is not None
    assert torch.count_nonzero(student.head_bias.grad[3]) > 0
    assert torch.count_nonzero(student.head_bias.grad[:3]) == 0
    assert torch.count_nonzero(student.head_bias.grad[4:]) == 0
    assert all(parameter.grad is None for parameter in pipeline.teacher.parameters())
    assert all(not value.requires_grad for value in metrics.values())
