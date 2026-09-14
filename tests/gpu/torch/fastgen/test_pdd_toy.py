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

"""Single-GPU BF16 proof for the framework-neutral PDD core."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from modelopt.torch.fastgen import (
    PDDConfig,
    PDDLayerSpec,
    PDDOutputProjection,
    PDDPipeline,
    convert_to_pdd_output_projection,
)

_WIDTH = 8
_GRID_SIZE = 4


class _Student(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Linear(_WIDTH, _WIDTH)
        self.projection = nn.Linear(_WIDTH, _WIDTH)

    def forward(
        self,
        state: torch.Tensor,
        *,
        fusion: tuple[int, int, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        return self.projection(torch.tanh(self.backbone(state)), fusion=fusion)


class _Teacher(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(_WIDTH, _WIDTH)

    def forward(self, state: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        return self.projection(state) + 0.125 * time[:, None]


class _Adapter:
    def __init__(self) -> None:
        self.fused_calls = 0

    @staticmethod
    def _model_dtype(model: nn.Module) -> torch.dtype:
        return next(model.parameters()).dtype

    def student_all_heads(
        self,
        model: _Student,
        state: torch.Tensor,
        time: torch.Tensor,
        *,
        condition: Any = None,
        **model_kwargs: Any,
    ) -> torch.Tensor:
        del time, condition, model_kwargs
        output = model(state.to(self._model_dtype(model)))
        return output.reshape(state.shape[0], _GRID_SIZE, _WIDTH)

    def student_fused_block(
        self,
        model: _Student,
        state: torch.Tensor,
        time: torch.Tensor,
        *,
        start: int,
        end: int,
        grid: torch.Tensor,
        condition: Any = None,
        **model_kwargs: Any,
    ) -> torch.Tensor:
        del time, condition, model_kwargs
        projection = model.projection
        assert isinstance(projection, PDDOutputProjection)
        self.fused_calls += 1
        return model(state.to(self._model_dtype(model)), fusion=(start, end, grid))

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
        del condition, negative_condition, model_kwargs
        dtype = self._model_dtype(model)
        return model(state.to(dtype), time.to(dtype))


def _fill_parameters(model: nn.Module, *, offset: float) -> None:
    with torch.no_grad():
        for index, parameter in enumerate(model.parameters()):
            values = torch.linspace(
                -0.2 + offset + 0.01 * index,
                0.2 + offset + 0.01 * index,
                parameter.numel(),
                dtype=torch.float32,
                device=parameter.device,
            )
            parameter.copy_(values.reshape_as(parameter).to(parameter.dtype))


def _build(
    device: torch.device,
) -> tuple[_Student, _Teacher, PDDOutputProjection, PDDPipeline, _Adapter]:
    config = PDDConfig(
        grid_size=_GRID_SIZE,
        grid_max_t=0.999,
        flow_shift=5.0,
        block_size_min=1,
        block_size_max=_GRID_SIZE,
        inference_blocks=[2, 2],
        guidance_scale=None,
    )
    student = _Student().to(device=device, dtype=torch.bfloat16)
    teacher = _Teacher().to(device=device, dtype=torch.bfloat16)
    _fill_parameters(student, offset=0.0)
    _fill_parameters(teacher, offset=0.05)
    projection = convert_to_pdd_output_projection(
        student,
        PDDLayerSpec("projection", "channel_major"),
        config.grid_size,
    )
    adapter = _Adapter()
    pipeline = PDDPipeline(student, teacher, config, adapter)
    return student, teacher, projection, pipeline, adapter


def test_bf16_loss_backward_and_fused_sample() -> None:
    assert torch.cuda.is_available(), "BF16 test requires a real CUDA device"
    device = torch.device("cuda", 0)
    assert torch.cuda.get_device_capability(device)[0] >= 8, "BF16 requires Ampere or newer"

    student, teacher, projection, pipeline, adapter = _build(device)
    assert {parameter.dtype for parameter in student.parameters()} == {torch.bfloat16}
    assert pipeline.time_grid(device).dtype == torch.float32

    data = torch.linspace(-0.75, 0.75, _WIDTH, device=device, dtype=torch.bfloat16).reshape(1, -1)
    noise = torch.linspace(0.5, -0.5, _WIDTH, device=device, dtype=torch.float32).reshape(1, -1)
    n = torch.tensor([0], device=device, dtype=torch.int64)
    k = torch.tensor([2], device=device, dtype=torch.int64)
    loss, metrics = pipeline.compute_loss(data, noise=noise, n=n, k=k)
    assert loss.dtype == torch.float32
    assert torch.isfinite(loss)
    for name in (
        "all_student_heads_finite",
        "student_target_finite",
        "teacher_target_finite",
        "reconstructed_state_finite",
        "loss_finite",
    ):
        assert bool(metrics[name].all()), name
    loss.backward()

    assert all(parameter.grad is None for parameter in teacher.parameters())
    assert projection.weight.grad is not None
    assert student.backbone.weight.grad is not None
    assert torch.isfinite(projection.weight.grad).all()
    assert torch.isfinite(student.backbone.weight.grad).all()

    sampled = pipeline.sample(noise, blocks=[2, 2])
    assert sampled.dtype == torch.float32
    assert torch.isfinite(sampled).all()
    assert adapter.fused_calls == 2
    torch.cuda.synchronize(device)
