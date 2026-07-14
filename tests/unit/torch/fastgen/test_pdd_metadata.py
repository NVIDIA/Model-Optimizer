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

"""Plain-torch PDD lifecycle and serialized-metadata reconstruction evidence."""

from __future__ import annotations

import copy
import json
from typing import Any

import pytest
import torch
from torch import nn

from modelopt.torch.fastgen import (
    PDDConfig,
    PDDLayerSpec,
    PDDMetadata,
    PDDOutputProjection,
    PDDPipeline,
    convert_to_pdd_output_projection,
)


class _ToyStudent(nn.Module):
    def __init__(self, width: int = 3) -> None:
        super().__init__()
        self.backbone = nn.Linear(width, width)
        self.projection = nn.Linear(width, width)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.projection(torch.tanh(self.backbone(state)))


class _ToyTeacher(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(-0.25))

    def forward(self, state: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        return self.scale * state + 0.1 * time[:, None]


class _ToyAdapter:
    def __init__(self, grid_size: int) -> None:
        self.grid_size = grid_size

    def _projection(self, model: _ToyStudent) -> PDDOutputProjection:
        projection = model.projection
        assert isinstance(projection, PDDOutputProjection)
        return projection

    def student_all_heads(
        self,
        model: _ToyStudent,
        state: torch.Tensor,
        time: torch.Tensor,
        *,
        condition: Any = None,
        **model_kwargs: Any,
    ) -> torch.Tensor:
        del time, condition, model_kwargs
        raw = model(state)
        return raw.reshape(state.shape[0], self.grid_size, state.shape[1])

    def student_fused_block(
        self,
        model: _ToyStudent,
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
        with self._projection(model).fuse_block(start, end, grid):
            return model(state)

    def teacher_velocity(
        self,
        model: _ToyTeacher,
        state: torch.Tensor,
        time: torch.Tensor,
        *,
        condition: Any = None,
        negative_condition: Any = None,
        **model_kwargs: Any,
    ) -> torch.Tensor:
        del condition, negative_condition, model_kwargs
        return model(state, time)


def _config() -> PDDConfig:
    return PDDConfig(
        grid_size=4,
        flow_shift=5.0,
        block_size_min=1,
        block_size_max=4,
        inference_blocks=[2, 2],
        student_sample_steps=2,
    )


def test_plain_torch_training_sampling_and_strict_metadata_reconstruction(tmp_path) -> None:
    torch.manual_seed(19)
    config = _config()
    layer_spec = PDDLayerSpec("projection", "channel_major")
    student = _ToyStudent()
    projection = convert_to_pdd_output_projection(student, layer_spec, config.grid_size)
    pipeline = PDDPipeline(student, _ToyTeacher(), config, _ToyAdapter(config.grid_size))
    optimizer = torch.optim.SGD(student.parameters(), lr=0.05)
    data = torch.tensor([[0.5, -1.0, 0.25], [-0.75, 0.5, 1.25]])
    noise = torch.tensor([[-0.25, 0.75, 1.0], [0.5, -1.25, 0.0]])
    projection_before = projection.weight.detach().clone()

    loss, _ = pipeline.compute_loss(
        data,
        noise=noise,
        n=torch.tensor([0, 1]),
        k=torch.tensor([2, 3]),
    )
    loss.backward()
    optimizer.step()

    assert torch.isfinite(loss)
    assert not torch.equal(projection.weight, projection_before)
    initial = torch.tensor([[1.0, -0.5, 0.25], [-0.25, 0.75, 1.5]])
    expected_sample = pipeline.sample(initial)

    metadata = PDDMetadata.from_config(config, projection)
    metadata_path = tmp_path / "pdd_metadata.json"
    metadata_path.write_text(json.dumps(metadata.to_dict(), sort_keys=True), encoding="utf-8")
    restored_metadata = PDDMetadata.from_dict(json.loads(metadata_path.read_text(encoding="utf-8")))
    assert restored_metadata == metadata

    restored_config = PDDConfig(
        grid_size=restored_metadata.grid_size,
        flow_shift=restored_metadata.flow_shift,
        block_size_min=restored_metadata.block_size_min,
        block_size_max=restored_metadata.block_size_max,
        inference_blocks=list(restored_metadata.inference_blocks),
        student_sample_steps=len(restored_metadata.inference_blocks),
        teacher_integrator=restored_metadata.teacher_integrator,
    )

    restored_student = _ToyStudent(width=restored_metadata.projection_in_features)
    restored_projection = convert_to_pdd_output_projection(
        restored_student,
        restored_metadata.layer_spec,
        restored_metadata.grid_size,
    )
    load_result = restored_student.load_state_dict(copy.deepcopy(student.state_dict()), strict=True)
    assert load_result.missing_keys == []
    assert load_result.unexpected_keys == []
    assert restored_projection.base_out_features == restored_metadata.projection_out_features
    assert (restored_projection.bias is not None) is restored_metadata.projection_bias

    restored_pipeline = PDDPipeline(
        restored_student,
        copy.deepcopy(pipeline.teacher),
        restored_config,
        _ToyAdapter(restored_metadata.grid_size),
    )
    torch.testing.assert_close(restored_pipeline.sample(initial), expected_sample)


def test_strict_restore_rejects_checkpoint_with_different_projection_grid() -> None:
    config = _config()
    layer_spec = PDDLayerSpec("projection", "channel_major")
    student = _ToyStudent()
    convert_to_pdd_output_projection(student, layer_spec, config.grid_size)
    checkpoint = copy.deepcopy(student.state_dict())
    incompatible = _ToyStudent()
    convert_to_pdd_output_projection(incompatible, layer_spec, grid_size=2)

    with pytest.raises(RuntimeError, match="size mismatch"):
        incompatible.load_state_dict(checkpoint, strict=True)
