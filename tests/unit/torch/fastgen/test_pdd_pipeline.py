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
from modelopt.torch.fastgen.flow_matching import fusion_coefficients


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
        return output[:, :-1] if self.bad_student_shape else output

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
        coefficients = fusion_coefficients(grid, start, end)
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
        return model(state, time)


def _config(*, teacher_integrator: str = "euler") -> PDDConfig:
    return PDDConfig(
        grid_size=8,
        flow_shift=5.0,
        block_size_min=2,
        block_size_max=4,
        inference_blocks=[4, 4],
        student_sample_steps=2,
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
    x_n = (1 - grid[n, None]) * data + grid[n, None] * noise
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
    assert torch.equal(metrics["target_span"], torch.tensor([0, 1]))
    assert metrics["student_target_mse"].shape == (2,)
    assert metrics["all_student_heads_finite"].shape == (2,)
    assert bool(metrics["loss_finite"])
    assert all(not value.requires_grad for value in metrics.values())


def test_midpoint_target_uses_exact_final_interval_midpoint() -> None:
    pipeline, adapter = _pipeline(teacher_integrator="midpoint")
    data = torch.tensor([[1.0, -2.0, 0.5]])
    noise = torch.tensor([[0.25, 1.5, -0.5]])
    n = torch.tensor([6])
    k = torch.tensor([7])

    loss, _ = pipeline.compute_loss(data, noise=noise, n=n, k=k)

    grid = pipeline.time_grid()
    x_n = (1 - grid[n, None]) * data + grid[n, None] * noise
    heads = pipeline.student.all_heads(x_n)
    x_bar_k = x_n + (grid[7] - grid[6]) * heads[:, 6]
    first_velocity = pipeline.teacher(x_bar_k, grid[k])
    delta = grid[8] - grid[7]
    midpoint_state = x_bar_k + 0.5 * delta * first_velocity
    midpoint_time = grid[k] + 0.5 * delta
    midpoint_target = pipeline.teacher(midpoint_state, midpoint_time)
    expected_loss = (heads[:, 7] - midpoint_target).square().mean()

    torch.testing.assert_close(loss, expected_loss)
    assert len(adapter.student_calls) == 1
    assert len(adapter.teacher_calls) == 2
    torch.testing.assert_close(adapter.teacher_calls[0]["state"], x_bar_k)
    torch.testing.assert_close(adapter.teacher_calls[0]["time"], grid[k])
    torch.testing.assert_close(adapter.teacher_calls[1]["state"], midpoint_state)
    torch.testing.assert_close(adapter.teacher_calls[1]["time"], midpoint_time)


def test_small_grid_accepts_exactly_the_trained_index_support() -> None:
    pipeline, _ = _pipeline()
    data = torch.ones(1, 3)
    noise = torch.zeros_like(data)
    expected = {
        (0, 0),
        (0, 1),
        (0, 2),
        (0, 3),
        (2, 2),
        (2, 3),
        (2, 4),
        (2, 5),
        (4, 4),
        (4, 5),
        (4, 6),
        (4, 7),
        (6, 6),
        (6, 7),
    }
    accepted = set()

    for n_value in range(-1, 9):
        for k_value in range(-1, 9):
            pair = (n_value, k_value)
            if pair not in expected:
                with pytest.raises(RuntimeError):
                    pipeline.compute_loss(
                        data,
                        noise=noise,
                        n=torch.tensor([n_value]),
                        k=torch.tensor([k_value]),
                    )
                continue
            pipeline.compute_loss(
                data,
                noise=noise,
                n=torch.tensor([n_value]),
                k=torch.tensor([k_value]),
            )
            accepted.add(pair)

    assert accepted == expected


def test_explicit_k_requires_explicit_n_but_explicit_n_can_sample_k() -> None:
    pipeline, _ = _pipeline()
    data = torch.ones(16, 3)
    noise = torch.zeros_like(data)

    with pytest.raises(ValueError, match="explicit k requires explicit n"):
        pipeline.compute_loss(data, noise=noise, k=torch.zeros(16, dtype=torch.long))

    _, metrics = pipeline.compute_loss(
        data,
        noise=noise,
        n=torch.full((16,), 6, dtype=torch.long),
        generator=torch.Generator().manual_seed(7),
    )
    assert torch.equal(metrics["n"], torch.full((16,), 6, dtype=torch.long))
    assert set(metrics["k"].tolist()) == {6, 7}


def test_sampled_indices_stay_on_exact_uniform_support() -> None:
    pipeline, _ = _pipeline()
    generator = torch.Generator().manual_seed(1234)

    n, k = pipeline._resolve_indices(
        batch_size=4096,
        device=torch.device("cpu"),
        n=None,
        k=None,
        generator=generator,
    )

    assert set(n.tolist()) == {0, 2, 4, 6}
    assert torch.all(n.remainder(2) == 0)
    assert torch.all(k >= n)
    assert torch.all(k < torch.minimum(n + 4, torch.full_like(n, 8)))
    for n_value in (0, 2, 4, 6):
        observed = set(k[n == n_value].tolist())
        assert observed == set(range(n_value, min(n_value + 4, 8)))


@pytest.mark.parametrize("blocks", [None, [2, 2, 2, 2]])
def test_fused_sampler_matches_explicit_block_updates(blocks) -> None:
    pipeline, adapter = _pipeline()
    initial = torch.tensor([[1.0, -2.0, 0.5], [-0.25, 0.75, 1.5]], dtype=torch.bfloat16)

    actual = pipeline.sample(
        initial,
        condition="prompt",
        blocks=blocks,
        model_kwargs={"tag": 23},
    )

    resolved = [4, 4] if blocks is None else blocks
    grid = pipeline.time_grid()
    expected = initial.float()
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
        torch.testing.assert_close(call["time"], grid[start].expand(initial.shape[0]))
        torch.testing.assert_close(call["grid"], grid)
        assert call["condition"] == "prompt"
        assert call["kwargs"] == {"tag": 23}
        start = end


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
    for blocks in ([3, 5], [2, 2], [6, 2], [], [2, 2, 2, 4]):
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
