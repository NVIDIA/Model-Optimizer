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

"""Tests for Puzzletron NAS orchestration helpers."""

import json
from pathlib import Path

from omegaconf import OmegaConf
from safetensors.torch import save_file
from torch import tensor

from modelopt.torch.puzzletron.puzzletron_nas_plugin import (
    _force_scoring_revalidation,
    _invalidate_scoring_cache,
    _is_complete_anymodel_checkpoint,
)


def test_complete_anymodel_checkpoint_rejects_config_only(tmp_path: Path):
    checkpoint_dir = tmp_path / "ckpt"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "config.json").write_text("{}")

    assert not _is_complete_anymodel_checkpoint(checkpoint_dir)


def test_complete_anymodel_checkpoint_requires_indexed_weights(tmp_path: Path):
    checkpoint_dir = tmp_path / "ckpt"
    subblocks_dir = checkpoint_dir / "subblocks_safetensors"
    subblocks_dir.mkdir(parents=True)
    weight_path = subblocks_dir / "embeddings.safetensors"
    (checkpoint_dir / "config.json").write_text("{}")
    save_file({"model.embed_tokens.weight": tensor([1.0])}, weight_path)
    (checkpoint_dir / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"format": "pt"},
                "weight_map": {
                    "model.embed_tokens.weight": "subblocks_safetensors/embeddings.safetensors"
                },
            }
        )
    )

    assert _is_complete_anymodel_checkpoint(checkpoint_dir)

    weight_path.unlink()

    assert not _is_complete_anymodel_checkpoint(checkpoint_dir)


def test_invalidate_scoring_cache_removes_validation_jsons(tmp_path: Path):
    output_dir = tmp_path / "single_sequence_replacement_solutions--validation"
    output_dir.mkdir()
    teacher_path = output_dir / "teacher.json"
    solution_path = output_dir / "solution_0.json"
    unrelated_path = output_dir / "notes.json"
    for path in (teacher_path, solution_path, unrelated_path):
        path.write_text("{}")

    cfg = OmegaConf.create({"scoring": {"output_dir": str(output_dir)}})

    _invalidate_scoring_cache(cfg)

    assert not teacher_path.exists()
    assert not solution_path.exists()
    assert unrelated_path.exists()


def test_force_scoring_revalidation_ignores_existing_solutions():
    cfg = OmegaConf.create(
        {"scoring": {"skip_existing_solutions": True, "solutions_to_validate": [0]}}
    )

    _force_scoring_revalidation(cfg)

    assert not cfg.scoring.skip_existing_solutions
    assert cfg.scoring.solutions_to_validate is None
