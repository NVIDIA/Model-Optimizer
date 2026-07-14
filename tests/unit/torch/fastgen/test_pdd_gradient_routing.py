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

"""Gradient-routing contract for data-dependent PDD targets."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from modelopt.torch.fastgen import PDDConfig, PDDPipeline


class _GradientStudent(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.shared = nn.Parameter(torch.tensor(0.5))
        self.heads = nn.Parameter(torch.arange(16, dtype=torch.float32).reshape(8, 2) / 11)


class _GradientTeacher(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(-0.25))


class _GradientAdapter:
    def __init__(self) -> None:
        self.raw_heads: torch.Tensor | None = None
        self.teacher_query: torch.Tensor | None = None

    def student_all_heads(
        self,
        model: _GradientStudent,
        state: torch.Tensor,
        time: torch.Tensor,
        *,
        condition: Any = None,
        **model_kwargs: Any,
    ) -> torch.Tensor:
        del time, condition, model_kwargs
        output = model.shared * state[:, None] + model.heads[None]
        output.retain_grad()
        self.raw_heads = output
        return output

    def student_fused_block(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        raise AssertionError("sampling is not part of this gradient test")

    def teacher_velocity(
        self,
        model: _GradientTeacher,
        state: torch.Tensor,
        time: torch.Tensor,
        *,
        condition: Any = None,
        negative_condition: Any = None,
        **model_kwargs: Any,
    ) -> torch.Tensor:
        del condition, negative_condition, model_kwargs
        self.teacher_query = state
        return model.scale * state + time[:, None]


def test_only_selected_head_and_shared_backbone_receive_gradients() -> None:
    student = _GradientStudent()
    teacher = _GradientTeacher()
    adapter = _GradientAdapter()
    config = PDDConfig(
        grid_size=8,
        flow_shift=5.0,
        block_size_min=2,
        block_size_max=4,
        inference_blocks=[4, 4],
        student_sample_steps=2,
    )
    pipeline = PDDPipeline(student, teacher, config, adapter)
    data = torch.tensor([[1.0, -0.5]])
    noise = torch.tensor([[-0.25, 2.0]])
    original_heads = student.heads.detach().clone()
    optimizer = torch.optim.SGD(student.parameters(), lr=0.1)

    loss, metrics = pipeline.compute_loss(
        data,
        noise=noise,
        n=torch.tensor([0]),
        k=torch.tensor([3]),
    )
    loss.backward()

    assert student.shared.grad is not None
    assert not torch.equal(student.shared.grad, torch.zeros_like(student.shared.grad))
    assert torch.all(torch.isfinite(student.shared.grad))
    assert student.heads.grad is not None
    assert torch.count_nonzero(student.heads.grad[3]) > 0
    assert torch.count_nonzero(student.heads.grad[:3]) == 0
    assert torch.count_nonzero(student.heads.grad[4:]) == 0
    assert torch.all(torch.isfinite(student.heads.grad))
    assert adapter.raw_heads is not None
    assert adapter.raw_heads.grad is not None
    assert torch.count_nonzero(adapter.raw_heads.grad[:, 3]) > 0
    assert torch.count_nonzero(adapter.raw_heads.grad[:, :3]) == 0
    assert torch.count_nonzero(adapter.raw_heads.grad[:, 4:]) == 0
    assert adapter.teacher_query is not None
    assert adapter.teacher_query.requires_grad is False
    assert teacher.scale.requires_grad is False
    assert teacher.scale.grad is None
    assert loss.requires_grad is True
    assert all(not value.requires_grad for value in metrics.values())

    optimizer.step()
    assert torch.equal(student.heads[:3], original_heads[:3])
    assert not torch.equal(student.heads[3], original_heads[3])
    assert torch.equal(student.heads[4:], original_heads[4:])
