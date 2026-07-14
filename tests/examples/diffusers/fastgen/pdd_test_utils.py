# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small plain-torch objects shared by PDD example lifecycle tests."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import torch
from pdd_training import PDDTrainer, PreparedPDDBatch
from torch import nn

from modelopt.torch.fastgen import (
    PDDConfig,
    PDDLayerSpec,
    PDDMetadata,
    PDDOutputProjection,
    PDDPipeline,
    convert_to_pdd_output_projection,
)


class ToyStudent(nn.Module):
    def __init__(self, width: int = 3) -> None:
        super().__init__()
        self.backbone = nn.Linear(width, width)
        self.projection = nn.Linear(width, width)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.projection(torch.tanh(self.backbone(state)))


class ToyTeacher(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(-0.25))
        self.bias = nn.Parameter(torch.tensor(0.125))

    def forward(self, state: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        return self.scale * state + self.bias + 0.1 * time[:, None]


class ToyAdapter:
    def __init__(self, grid_size: int, *, zero_student_gradient: bool = False) -> None:
        self.grid_size = grid_size
        self.zero_student_gradient = zero_student_gradient

    def student_all_heads(
        self,
        model: ToyStudent,
        state: torch.Tensor,
        time: torch.Tensor,
        *,
        condition: Any = None,
        **model_kwargs: Any,
    ) -> torch.Tensor:
        del time, condition, model_kwargs
        raw = model(state)
        if self.zero_student_gradient:
            raw = raw * 0.0
        return raw.reshape(state.shape[0], self.grid_size, state.shape[1])

    def student_fused_block(
        self,
        model: ToyStudent,
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
        with projection.fuse_block(start, end, grid):
            return model(state)

    def teacher_velocity(
        self,
        model: ToyTeacher,
        state: torch.Tensor,
        time: torch.Tensor,
        *,
        condition: Any = None,
        negative_condition: Any = None,
        **model_kwargs: Any,
    ) -> torch.Tensor:
        del condition, negative_condition, model_kwargs
        return model(state, time)


@dataclass
class ToyLifecycle:
    config: PDDConfig
    student: ToyStudent
    teacher: ToyTeacher
    projection: PDDOutputProjection
    pipeline: PDDPipeline
    optimizer: torch.optim.AdamW
    scheduler: torch.optim.lr_scheduler.LambdaLR
    trainer: PDDTrainer
    metadata: PDDMetadata


def build_toy_lifecycle(
    *,
    seed: int = 17,
    zero_student_gradient: bool = False,
    weight_decay: float = 0.01,
) -> ToyLifecycle:
    torch.manual_seed(seed)
    config = PDDConfig(
        grid_size=4,
        grid_max_t=0.999,
        flow_shift=5.0,
        block_size_min=1,
        block_size_max=4,
        inference_blocks=[2, 2],
        student_sample_steps=2,
        guidance_scale=None,
    )
    student = ToyStudent()
    projection = convert_to_pdd_output_projection(
        student,
        PDDLayerSpec("projection", "channel_major"),
        config.grid_size,
    )
    teacher = ToyTeacher()
    pipeline = PDDPipeline(
        student,
        teacher,
        config,
        ToyAdapter(config.grid_size, zero_student_gradient=zero_student_gradient),
    )
    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=2e-3,
        weight_decay=weight_decay,
        amsgrad=False,
        capturable=False,
        differentiable=False,
        foreach=False,
        fused=False,
        maximize=False,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
    trainer = PDDTrainer(
        pipeline,
        optimizer,
        projection=projection,
        max_grad_norm=0.5,
    )
    return ToyLifecycle(
        config,
        student,
        teacher,
        projection,
        pipeline,
        optimizer,
        scheduler,
        trainer,
        PDDMetadata.from_config(config, projection),
    )


def make_batch(sample_ids: tuple[str, ...], *, offset: float = 0.0) -> PreparedPDDBatch:
    data = torch.stack(
        [
            torch.tensor([0.5 + offset + index / 10, -1.0, 0.25], dtype=torch.float32)
            for index in range(len(sample_ids))
        ]
    )
    condition = (
        torch.zeros((len(sample_ids), 1, 1), dtype=torch.float32),
        torch.ones((len(sample_ids), 1), dtype=torch.long),
    )
    return PreparedPDDBatch(data, condition, None, sample_ids)


class SamplerDataset:
    def __init__(self, sample_ids: tuple[str, ...]) -> None:
        self.metadata = [
            {
                "sample_id": sample_id,
                "bucket_id": "64x64",
                "bucket_resolution": [64, 64],
            }
            for sample_id in sample_ids
        ]
        self.bucket_groups = {
            (64, 64): {
                "indices": list(range(len(sample_ids))),
                "resolution": (64, 64),
                "aspect_name": "square",
            }
        }
        self.sorted_bucket_keys = [(64, 64)]
        self.calculator = None

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, index: int) -> int:
        return index


def ordered_id_sha256(sample_ids: tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    digest.update(b"modelopt-pdd-ordered-train-ids-v1\0")
    for sample_id in sample_ids:
        digest.update(sample_id.encode())
        digest.update(b"\n")
    return digest.hexdigest()
