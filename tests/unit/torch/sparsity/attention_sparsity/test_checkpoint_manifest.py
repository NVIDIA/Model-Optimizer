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

"""Tests for content-addressed checkpoint identity."""

from pathlib import Path

import pytest

from modelopt.torch.sparsity.attention_sparsity.calibration.checkpoint_manifest import (
    CHECKPOINT_MANIFEST_NAME,
    CheckpointManifestError,
    create_checkpoint_manifest,
    verify_checkpoint_manifest,
)


def _toy_checkpoint(path: Path) -> Path:
    path.mkdir()
    (path / "config.json").write_text('{"model_type":"toy"}\n', encoding="utf-8")
    (path / "model.safetensors").write_bytes(b"toy-weights")
    (path / "tokenizer.json").write_text("{}\n", encoding="utf-8")
    return path


def test_manifest_generation_is_deterministic_verified_and_no_clobber(tmp_path):
    first_root = _toy_checkpoint(tmp_path / "first")
    second_root = _toy_checkpoint(tmp_path / "second")

    first = create_checkpoint_manifest(first_root, model="toy-model")
    second = create_checkpoint_manifest(second_root, model="toy-model")

    assert first.sha256 == second.sha256
    assert first.manifest_path.read_bytes() == second.manifest_path.read_bytes()
    assert verify_checkpoint_manifest(first_root, expected_model="toy-model") == first
    with pytest.raises(CheckpointManifestError, match="refusing to overwrite"):
        create_checkpoint_manifest(first_root, model="toy-model")


def test_verifier_rejects_content_mutation_and_symlinks(tmp_path):
    root = _toy_checkpoint(tmp_path / "checkpoint")
    create_checkpoint_manifest(root, model="toy-model")
    (root / "model.safetensors").write_bytes(b"mutated")
    with pytest.raises(CheckpointManifestError, match="does not match"):
        verify_checkpoint_manifest(root)

    other = _toy_checkpoint(tmp_path / "symlinked")
    (other / "alias.bin").symlink_to(other / "model.safetensors")
    with pytest.raises(CheckpointManifestError, match="forbidden symlink"):
        create_checkpoint_manifest(other, model="toy-model")


def test_verifier_rejects_symlink_manifest(tmp_path):
    root = _toy_checkpoint(tmp_path / "checkpoint")
    target = tmp_path / "outside.json"
    target.write_text("{}\n", encoding="utf-8")
    (root / CHECKPOINT_MANIFEST_NAME).symlink_to(target)

    with pytest.raises(CheckpointManifestError, match="without following symlinks"):
        verify_checkpoint_manifest(root)
