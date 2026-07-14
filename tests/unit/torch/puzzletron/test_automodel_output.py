# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Tests for the AutoModel consolidated score writer.

Validates the legacy-compatible ``rank_*.pth`` round-trip and the single-writer
dedup rule (only one rank per pipeline stage persists). gloo/CPU, no GPU.
"""

import functools
from pathlib import Path

import torch
from _test_utils.torch.distributed.utils import spawn_multiprocess_job
from torch.distributed.device_mesh import init_device_mesh

from modelopt.torch.puzzletron.plugins.automodel.output import write_scores
from modelopt.torch.puzzletron.plugins.automodel.reduction import MeshGroups


class _FakeScorer:
    """Scorer stub whose ``finalize`` returns a fixed score (no collectives)."""

    def __init__(self, name, score):
        self.name = name
        self._score = score

    def finalize(self):
        return {"score": self._score}


def _load_like_pruning(activations_log_dir) -> dict:
    """Replicate pruning_utils._cache_activations_log: glob rank_*.pth and merge by key."""
    return {
        module_name: log
        for path in Path(activations_log_dir).glob("rank_*.pth")
        for module_name, log in torch.load(path).items()
    }


def test_write_scores_roundtrip(tmp_path):
    scorers = [
        _FakeScorer("model.layers.0.mlp.down_proj", torch.tensor([3.0, 1.0, 2.0])),
        _FakeScorer("model.layers.1.mlp.down_proj", torch.tensor([0.5, 0.7])),
    ]
    write_scores(scorers, str(tmp_path), MeshGroups())

    assert [f.name for f in tmp_path.glob("rank_*.pth")] == ["rank_0.pth"]
    merged = _load_like_pruning(tmp_path)
    assert set(merged) == {"model.layers.0.mlp.down_proj", "model.layers.1.mlp.down_proj"}
    assert torch.equal(
        merged["model.layers.0.mlp.down_proj"]["score"], torch.tensor([3.0, 1.0, 2.0])
    )


def _job_single_writer(out_dir, rank, size):
    assert size == 4
    mesh = init_device_mesh("cpu", (2, 2), mesh_dim_names=("dp", "tp"))
    groups = MeshGroups.from_device_mesh(mesh)
    scorers = [
        _FakeScorer("model.layers.0.mlp.down_proj", torch.tensor([1.0, 2.0])),
        _FakeScorer("model.layers.1.mlp.down_proj", torch.tensor([3.0, 4.0])),
    ]
    write_scores(scorers, out_dir, groups)


def test_write_scores_single_writer(tmp_path):
    spawn_multiprocess_job(
        size=4, job=functools.partial(_job_single_writer, str(tmp_path)), backend="gloo"
    )
    # Only the (token=0, tp=0, ep=0) rank of the single pipeline stage writes.
    files = sorted(p.name for p in tmp_path.glob("rank_*.pth"))
    assert files == ["rank_0.pth"], files
    merged = _load_like_pruning(tmp_path)
    assert set(merged) == {"model.layers.0.mlp.down_proj", "model.layers.1.mlp.down_proj"}
