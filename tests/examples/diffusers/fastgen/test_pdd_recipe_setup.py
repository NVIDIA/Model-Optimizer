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

"""Focused tests for the PDD loss seam used by AutoModel's diffusion recipe."""

from __future__ import annotations

import pathlib
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]
_FASTGEN_DIR = _REPO_ROOT / "examples" / "diffusers" / "fastgen"
if str(_FASTGEN_DIR) not in sys.path:
    sys.path.insert(0, str(_FASTGEN_DIR))

from pdd.recipe import _validate_prepared_student
from pdd.training import PDDFlowMatchingStepAdapter

from modelopt.torch.fastgen import PDDConfig


class _PreparedStudent(nn.Module):
    def __init__(self, out_features: int) -> None:
        super().__init__()
        self.config = SimpleNamespace(in_channels=4)
        self.proj_out = nn.Linear(3, out_features)


class _LossPipeline:
    def __init__(self, *, guidance_scale: float | None = None) -> None:
        self.config = SimpleNamespace(guidance_scale=guidance_scale)
        self.scale = nn.Parameter(torch.tensor(2.0))
        self.last_call = None

    def compute_loss(self, data, *, condition, negative_condition, collect_metrics):
        self.last_call = (data, condition, negative_condition, collect_metrics)
        per_sample = (data * self.scale).square().flatten(1).mean(1)
        return per_sample.mean(), {"student_target_mse": per_sample}


def _batch() -> dict[str, torch.Tensor]:
    return {
        "image_latents": torch.ones(2, 1, 2, 2),
        "text_embeddings": torch.ones(2, 3, 4),
        "text_embeddings_mask": torch.ones(2, 3, dtype=torch.long),
        "negative_text_embeddings": torch.zeros(2, 3, 4),
        "negative_text_embeddings_mask": torch.ones(2, 3, dtype=torch.long),
    }


def test_prepared_student_width_is_validated_before_training() -> None:
    config = PDDConfig(
        grid_size=8,
        block_size_min=1,
        block_size_max=8,
        inference_blocks=[4, 4],
        student_sample_steps=2,
    )
    _validate_prepared_student(_PreparedStudent(out_features=32), config)

    with pytest.raises(ValueError, match="Prepare the Qwen PDD student"):
        _validate_prepared_student(_PreparedStudent(out_features=4), config)


@pytest.mark.parametrize("guidance_scale", [None, 4.0])
def test_step_adapter_returns_native_tuple_and_preserves_pdd_gradient(guidance_scale) -> None:
    pipeline = _LossPipeline(guidance_scale=guidance_scale)
    adapter = PDDFlowMatchingStepAdapter(pipeline)

    per_sample, loss, prediction, metrics = adapter.step(
        model=nn.Identity(),
        batch=_batch(),
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    loss.backward()

    assert prediction is None
    torch.testing.assert_close(per_sample, torch.full((2,), 4.0))
    assert metrics["student_target_mse"] is per_sample
    torch.testing.assert_close(pipeline.scale.grad, torch.tensor(4.0))
    _, _, negative_condition, collect_metrics = pipeline.last_call
    assert (negative_condition is not None) is (guidance_scale is not None)
    assert collect_metrics is True
