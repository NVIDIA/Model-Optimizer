# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Single-GPU BF16 proof for the framework-neutral PDD core."""

from __future__ import annotations

import importlib.util
import sys
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

from modelopt.torch.fastgen import (
    PDDConfig,
    PDDLayerSpec,
    PDDOutputProjection,
    PDDPipeline,
    convert_to_pdd_output_projection,
)

if TYPE_CHECKING:
    from pathlib import Path

_FORBIDDEN_MODULES = ("diffusers", "fastgen", "nemo_automodel")
_WIDTH = 8
_GRID_SIZE = 4


class _Student(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Linear(_WIDTH, _WIDTH)
        self.projection = nn.Linear(_WIDTH, _WIDTH)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.projection(torch.tanh(self.backbone(state)))


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
        with projection.fuse_block(start, end, grid):
            return model(state.to(self._model_dtype(model)))

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
        student_sample_steps=2,
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


def _assert_optional_frameworks_absent() -> None:
    resolvable = sorted(name for name in _FORBIDDEN_MODULES if importlib.util.find_spec(name))
    assert not resolvable, f"plain PDD GPU environment resolves optional frameworks: {resolvable}"
    imported = sorted(name for name in _FORBIDDEN_MODULES if name in sys.modules)
    assert not imported, f"plain PDD GPU test imported optional frameworks: {imported}"


def test_bf16_loss_gradient_update_reload_and_fused_sample(tmp_path: Path) -> None:
    _assert_optional_frameworks_absent()
    assert torch.cuda.is_available(), "Task-10 BF16 gate requires a real CUDA device"
    device = torch.device("cuda", 0)
    assert torch.cuda.get_device_capability(device)[0] >= 8, "BF16 gate requires Ampere or newer"

    student, teacher, projection, pipeline, adapter = _build(device)
    assert {parameter.dtype for parameter in student.parameters()} == {torch.bfloat16}
    assert pipeline.time_grid(device).dtype == torch.float32

    data = torch.linspace(-0.75, 0.75, _WIDTH, device=device, dtype=torch.bfloat16).reshape(1, -1)
    noise = torch.linspace(0.5, -0.5, _WIDTH, device=device, dtype=torch.float32).reshape(1, -1)
    n = torch.tensor([0], device=device, dtype=torch.int64)
    k = torch.tensor([2], device=device, dtype=torch.int64)
    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=2.0e-3,
        weight_decay=0.0,
        foreach=False,
        fused=False,
    )

    optimizer.zero_grad(set_to_none=True)
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
    weight_grad = projection.weight.grad.reshape(_GRID_SIZE, _WIDTH, _WIDTH)
    bias_grad = projection.bias.grad.reshape(_GRID_SIZE, _WIDTH)
    assert torch.count_nonzero(weight_grad[2]) > 0
    assert torch.count_nonzero(bias_grad[2]) > 0
    assert torch.count_nonzero(weight_grad[[0, 1, 3]]) == 0
    assert torch.count_nonzero(bias_grad[[0, 1, 3]]) == 0
    assert student.backbone.weight.grad is not None
    assert torch.count_nonzero(student.backbone.weight.grad) > 0

    gradients = [
        parameter.grad.float().square().sum()
        for parameter in student.parameters()
        if parameter.grad is not None
    ]
    grad_norm = torch.stack(gradients).sum().sqrt()
    assert torch.isfinite(grad_norm) and grad_norm > 0
    before = {name: parameter.detach().clone() for name, parameter in student.named_parameters()}
    optimizer.step()
    update_norm = (
        torch.stack(
            [
                (parameter.detach() - before[name]).float().square().sum()
                for name, parameter in student.named_parameters()
            ]
        )
        .sum()
        .sqrt()
    )
    assert torch.isfinite(update_norm) and update_norm > 0

    checkpoint = tmp_path / "pdd_bf16_state.pt"
    torch.save(student.state_dict(), checkpoint)
    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    assert saved.keys() == student.state_dict().keys()
    assert all(value.dtype == torch.bfloat16 for value in saved.values())

    restored, _teacher, restored_projection, restored_pipeline, restored_adapter = _build(device)
    incompatible = restored.load_state_dict(saved, strict=True)
    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []
    assert restored_projection.weight.shape == projection.weight.shape
    time = pipeline.time_grid(device)[n]
    expected = adapter.student_all_heads(student, data.float(), time)
    actual = restored_adapter.student_all_heads(restored, data.float(), time)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    sampled = restored_pipeline.sample(noise, blocks=[2, 2])
    assert sampled.dtype == torch.float32
    assert torch.isfinite(sampled).all()
    assert restored_adapter.fused_calls == 2
    torch.cuda.synchronize(device)
    _assert_optional_frameworks_absent()
