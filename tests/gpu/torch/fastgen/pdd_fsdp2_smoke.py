# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Two-rank CUDA FSDP2 optimization and exact-resume proof for plain PDD modules."""

from __future__ import annotations

import gc
import importlib.util
import json
import os
import pathlib
import shutil
import sys
import tempfile
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch import nn
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
from torch.distributed.fsdp import fully_shard

from modelopt.torch.fastgen import (
    PDDConfig,
    PDDLayerSpec,
    PDDOutputProjection,
    PDDPipeline,
    convert_to_pdd_output_projection,
)

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

    def forward(
        self,
        state: torch.Tensor,
        time: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        return self.projection(state) + 0.125 * time[:, None] + 0.05 * condition


class _GuidedAdapter:
    def __init__(self) -> None:
        self.teacher_calls = 0

    @staticmethod
    def _dtype(model: nn.Module) -> torch.dtype:
        return next(model.parameters()).dtype

    def student_all_heads(
        self,
        model: nn.Module,
        state: torch.Tensor,
        time: torch.Tensor,
        *,
        condition: Any = None,
        **model_kwargs: Any,
    ) -> torch.Tensor:
        del time, condition, model_kwargs
        output = model(state.to(self._dtype(model)))
        return output.reshape(state.shape[0], _GRID_SIZE, _WIDTH)

    def student_fused_block(
        self,
        model: nn.Module,
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
        projection = model.get_submodule("projection")
        assert isinstance(projection, PDDOutputProjection)
        with projection.fuse_block(start, end, grid):
            return model(state.to(self._dtype(model)))

    def teacher_velocity(
        self,
        model: nn.Module,
        state: torch.Tensor,
        time: torch.Tensor,
        *,
        condition: Any = None,
        negative_condition: Any = None,
        **model_kwargs: Any,
    ) -> torch.Tensor:
        del model_kwargs
        if not isinstance(condition, torch.Tensor) or not isinstance(
            negative_condition, torch.Tensor
        ):
            raise TypeError("guided toy teacher requires tensor conditions")
        dtype = self._dtype(model)
        state = state.to(dtype)
        time = time.to(dtype)
        conditional = model(state, time, condition.to(dtype))
        unconditional = model(state, time, negative_condition.to(dtype))
        self.teacher_calls += 2
        return conditional + 3.0 * (conditional - unconditional)


@dataclass
class _Lifecycle:
    student: nn.Module
    teacher: nn.Module
    projection: PDDOutputProjection
    pipeline: PDDPipeline
    optimizer: torch.optim.AdamW
    adapter: _GuidedAdapter


def _local(value: torch.Tensor) -> torch.Tensor:
    to_local = getattr(value, "to_local", None)
    return to_local() if callable(to_local) else value


def _fill_parameters(model: nn.Module, *, offset: float) -> None:
    with torch.no_grad():
        for index, parameter in enumerate(model.parameters()):
            values = torch.linspace(
                -0.2 + offset + index * 0.01,
                0.2 + offset + index * 0.01,
                parameter.numel(),
                dtype=torch.float32,
                device=parameter.device,
            )
            parameter.copy_(values.reshape_as(parameter).to(parameter.dtype))


def _config() -> PDDConfig:
    return PDDConfig(
        grid_size=_GRID_SIZE,
        grid_max_t=0.999,
        flow_shift=5.0,
        block_size_min=1,
        block_size_max=_GRID_SIZE,
        inference_blocks=[2, 2],
        student_sample_steps=2,
        guidance_scale=4.0,
    )


def _build(device: torch.device) -> _Lifecycle:
    config = _config()
    student = _Student().to(device=device, dtype=torch.bfloat16)
    teacher = _Teacher().to(device=device, dtype=torch.bfloat16).eval().requires_grad_(False)
    _fill_parameters(student, offset=0.0)
    _fill_parameters(teacher, offset=0.05)
    projection = convert_to_pdd_output_projection(
        student,
        PDDLayerSpec("projection", "channel_major"),
        config.grid_size,
    )
    projection_module_id = id(projection)
    projection_shape = projection.weight.shape
    student = fully_shard(student)
    teacher = fully_shard(teacher)
    assert id(student.get_submodule("projection")) == projection_module_id
    assert student.get_submodule("projection").weight.shape == projection_shape
    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=2.0e-3,
        weight_decay=0.0,
        foreach=False,
        fused=False,
    )
    optimizer_parameters = [
        parameter for group in optimizer.param_groups for parameter in group["params"]
    ]
    assert any(parameter is projection.weight for parameter in optimizer_parameters)
    adapter = _GuidedAdapter()
    pipeline = PDDPipeline(student, teacher, config, adapter)
    return _Lifecycle(student, teacher, projection, pipeline, optimizer, adapter)


def _batch(
    *, rank: int, step: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    base = torch.arange(_WIDTH, device=device, dtype=torch.float32).reshape(1, -1)
    data = (base / 10 + 0.05 * rank + 0.025 * step).to(torch.bfloat16)
    noise = 0.4 - base / 20 + 0.01 * rank
    condition = 0.2 + base / 30 + 0.02 * step
    negative = -0.1 - base / 40 - 0.01 * rank
    n = torch.tensor([0 if step == 1 else 2], device=device, dtype=torch.int64)
    k = torch.tensor([1 if step == 1 else 3], device=device, dtype=torch.int64)
    return data, noise, condition, negative, n, k


def _global_norm(values: list[torch.Tensor], device: torch.device) -> torch.Tensor:
    squared = torch.zeros((), device=device, dtype=torch.float64)
    for value in values:
        squared += _local(value.detach()).float().square().sum(dtype=torch.float64)
    dist.all_reduce(squared, op=dist.ReduceOp.SUM)
    return squared.sqrt()


def _step(lifecycle: _Lifecycle, *, rank: int, step: int, device: torch.device) -> dict[str, float]:
    data, noise, condition, negative, n, k = _batch(rank=rank, step=step, device=device)
    lifecycle.optimizer.zero_grad(set_to_none=True)
    calls_before = lifecycle.adapter.teacher_calls
    loss, metrics = lifecycle.pipeline.compute_loss(
        data,
        noise=noise,
        condition=condition,
        negative_condition=negative,
        n=n,
        k=k,
    )
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
    assert lifecycle.adapter.teacher_calls - calls_before == 2
    assert all(parameter.grad is None for parameter in lifecycle.teacher.parameters())
    gradients = [
        parameter.grad for parameter in lifecycle.student.parameters() if parameter.grad is not None
    ]
    grad_norm = _global_norm(gradients, device)
    assert torch.isfinite(grad_norm) and grad_norm > 0
    before = {
        name: _local(parameter.detach()).clone()
        for name, parameter in lifecycle.student.named_parameters()
    }
    lifecycle.optimizer.step()
    updates = [
        _local(parameter.detach()) - before[name]
        for name, parameter in lifecycle.student.named_parameters()
    ]
    update_norm = _global_norm(updates, device)
    assert torch.isfinite(update_norm) and update_norm > 0
    reduced_loss = loss.detach().double()
    dist.all_reduce(reduced_loss, op=dist.ReduceOp.SUM)
    reduced_loss /= dist.get_world_size()
    return {
        "loss": float(reduced_loss.item()),
        "grad_norm": float(grad_norm.item()),
        "update_norm": float(update_norm.item()),
    }


def _state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: _local(value.detach()).clone()
        for name, value in model.state_dict().items()
        if isinstance(value, torch.Tensor)
    }


def _assert_state_equal(actual: nn.Module, expected: dict[str, torch.Tensor]) -> None:
    actual_state = _state(actual)
    assert actual_state.keys() == expected.keys()
    for name in expected:
        torch.testing.assert_close(actual_state[name], expected[name], rtol=0, atol=0)


def _all_rng_states(device: torch.device) -> dict[str, torch.Tensor]:
    local_state = {
        "cpu": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(device),
    }
    gathered: list[dict[str, torch.Tensor] | None] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, local_state)
    states: dict[str, torch.Tensor] = {}
    for rank, value in enumerate(gathered):
        assert value is not None
        states[f"cpu_rng_rank_{rank}"] = value["cpu"].to(device)
        states[f"cuda_rng_rank_{rank}"] = value["cuda"].to(device)
    return states


def _save_checkpoint(
    root: pathlib.Path,
    lifecycle: _Lifecycle,
    *,
    completed_steps: int,
    device: torch.device,
) -> pathlib.Path:
    staging = root / ".step_00000001.staging"
    final = root / "step_00000001"
    if dist.get_rank() == 0:
        staging.mkdir(parents=True)
    dist.barrier()
    model_state, optimizer_state = get_state_dict(lifecycle.student, lifecycle.optimizer)
    extra = _all_rng_states(device)
    extra["completed_steps"] = torch.tensor([completed_steps], device=device, dtype=torch.int64)
    dcp.save(
        {"model": model_state, "optimizer": optimizer_state, "extra": extra},
        checkpoint_id=staging,
    )
    dist.barrier()
    if dist.get_rank() == 0:
        assert (staging / ".metadata").is_file()
        assert any(path.suffix == ".distcp" for path in staging.iterdir())
        os.replace(staging, final)
        (final / "COMPLETE").write_text(
            json.dumps({"schema_version": 1, "completed_steps": completed_steps}) + "\n"
        )
    dist.barrier()
    assert (final / "COMPLETE").is_file()
    return final


def _load_checkpoint(
    checkpoint: pathlib.Path,
    lifecycle: _Lifecycle,
    *,
    device: torch.device,
) -> int:
    marker = json.loads((checkpoint / "COMPLETE").read_text())
    assert marker == {"schema_version": 1, "completed_steps": 1}
    model_state, optimizer_state = get_state_dict(lifecycle.student, lifecycle.optimizer)
    extra = _all_rng_states(device)
    extra["completed_steps"] = torch.zeros(1, device=device, dtype=torch.int64)
    payload = {"model": model_state, "optimizer": optimizer_state, "extra": extra}
    dcp.load(payload, checkpoint_id=checkpoint)
    incompatible = set_state_dict(
        lifecycle.student,
        lifecycle.optimizer,
        model_state_dict=payload["model"],
        optim_state_dict=payload["optimizer"],
    )
    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []
    rank = dist.get_rank()
    torch.set_rng_state(payload["extra"][f"cpu_rng_rank_{rank}"].cpu())
    torch.cuda.set_rng_state(payload["extra"][f"cuda_rng_rank_{rank}"].cpu(), device)
    return int(payload["extra"]["completed_steps"].item())


def _assert_call_counts(adapter: _GuidedAdapter, expected: int, device: torch.device) -> None:
    value = torch.tensor([adapter.teacher_calls], device=device, dtype=torch.int64)
    gathered = [torch.zeros_like(value) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, value)
    assert [int(item.item()) for item in gathered] == [expected] * dist.get_world_size()


def _assert_optional_frameworks_absent() -> None:
    resolvable = sorted(name for name in _FORBIDDEN_MODULES if importlib.util.find_spec(name))
    assert not resolvable, f"plain PDD FSDP2 environment resolves optional frameworks: {resolvable}"
    imported = sorted(name for name in _FORBIDDEN_MODULES if name in sys.modules)
    assert not imported, f"plain PDD FSDP2 smoke imported optional frameworks: {imported}"


def main() -> None:
    _assert_optional_frameworks_absent()
    assert torch.cuda.is_available(), "Task-10 FSDP2 gate requires CUDA"
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    assert world_size == 2, f"Task-10 FSDP2 gate requires two ranks, got {world_size}"
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    assert torch.cuda.get_device_capability(device)[0] >= 8
    root_payload = [tempfile.mkdtemp(prefix="modelopt-pdd-fsdp2-") if rank == 0 else None]
    dist.broadcast_object_list(root_payload, src=0)
    root = pathlib.Path(root_payload[0])
    try:
        torch.manual_seed(2026 + rank)
        torch.cuda.manual_seed(3026 + rank)
        reference = _build(device)
        resumable = _build(device)
        reference_step1 = _step(reference, rank=rank, step=1, device=device)
        resumable_step1 = _step(resumable, rank=rank, step=1, device=device)
        assert reference_step1 == resumable_step1
        _assert_state_equal(resumable.student, _state(reference.student))
        saved_cpu_rng = torch.get_rng_state().clone()
        saved_cuda_rng = torch.cuda.get_rng_state(device).clone()
        checkpoint = _save_checkpoint(
            root,
            resumable,
            completed_steps=1,
            device=device,
        )

        reference_step2 = _step(reference, rank=rank, step=2, device=device)
        reference_state = _state(reference.student)
        _assert_call_counts(reference.adapter, 4, device)

        del resumable
        gc.collect()
        torch.cuda.empty_cache()
        restored = _build(device)
        assert _load_checkpoint(checkpoint, restored, device=device) == 1
        torch.testing.assert_close(torch.get_rng_state(), saved_cpu_rng, rtol=0, atol=0)
        torch.testing.assert_close(torch.cuda.get_rng_state(device), saved_cuda_rng, rtol=0, atol=0)
        restored_step2 = _step(restored, rank=rank, step=2, device=device)
        assert restored_step2 == reference_step2
        _assert_state_equal(restored.student, reference_state)
        _assert_call_counts(restored.adapter, 2, device)
        _assert_optional_frameworks_absent()
        dist.barrier()
    finally:
        dist.barrier()
        if rank == 0:
            shutil.rmtree(root)
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
