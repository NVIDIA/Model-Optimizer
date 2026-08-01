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

from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import pytest

from modelopt.torch.sparsity.attention_sparsity.calibration import checkpoint_manifest
from modelopt.torch.sparsity.attention_sparsity.calibration.checkpoint_manifest import (
    CHECKPOINT_MANIFEST_NAME,
    CheckpointManifestError,
    create_checkpoint_manifest,
    read_stable_file_snapshot,
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


def test_portable_open_fallback_still_rejects_symlink(tmp_path, monkeypatch):
    target = tmp_path / "target.json"
    target.write_text("{}\n", encoding="utf-8")
    alias = tmp_path / "alias.json"
    alias.symlink_to(target)
    monkeypatch.delattr(checkpoint_manifest.os, "O_NOFOLLOW", raising=False)

    with pytest.raises(CheckpointManifestError, match="without following symlinks"):
        read_stable_file_snapshot(alias, label="fallback input")


def test_portable_open_fallback_preserves_exact_binary_bytes(tmp_path, monkeypatch):
    path = tmp_path / "binary.dat"
    payload = b"line-1\r\nline-2\x00\n"
    path.write_bytes(payload)
    monkeypatch.delattr(checkpoint_manifest.os, "O_NOFOLLOW", raising=False)

    snapshot = read_stable_file_snapshot(path, label="binary input")

    assert snapshot.payload == payload
    assert snapshot.sha256 == sha256(payload).hexdigest()


def test_portable_open_uses_binary_flag_when_available(tmp_path, monkeypatch):
    path = tmp_path / "binary.dat"
    path.write_bytes(b"payload")
    binary_flag = 1 << 29
    observed_flags = []
    real_open = checkpoint_manifest.os.open
    monkeypatch.setattr(checkpoint_manifest.os, "O_BINARY", binary_flag, raising=False)

    def recording_open(source, flags, *args, **kwargs):
        observed_flags.append(flags)
        return real_open(source, flags & ~binary_flag, *args, **kwargs)

    monkeypatch.setattr(checkpoint_manifest.os, "open", recording_open)

    snapshot = read_stable_file_snapshot(path, label="binary input")

    assert snapshot.payload == b"payload"
    assert observed_flags and observed_flags[0] & binary_flag


def test_portable_open_rejects_path_swap_before_read(tmp_path, monkeypatch):
    path = tmp_path / "input.dat"
    replacement = tmp_path / "replacement.dat"
    path.write_bytes(b"expected")
    replacement.write_bytes(b"attacker")
    real_open = checkpoint_manifest.os.open
    swapped = False
    monkeypatch.delattr(checkpoint_manifest.os, "O_NOFOLLOW", raising=False)

    def swapping_open(source, flags, *args, **kwargs):
        nonlocal swapped
        if Path(source) == path and not swapped:
            swapped = True
            path.unlink()
            replacement.rename(path)
        return real_open(source, flags, *args, **kwargs)

    monkeypatch.setattr(checkpoint_manifest.os, "open", swapping_open)

    with pytest.raises(CheckpointManifestError, match="stable regular file"):
        read_stable_file_snapshot(path, label="swapped input")


def test_windows_reparse_attribute_is_link_like():
    observed = SimpleNamespace(
        st_mode=0,
        st_file_attributes=checkpoint_manifest._FILE_ATTRIBUTE_REPARSE_POINT,
    )

    assert checkpoint_manifest._is_link_like(observed)


def test_manifest_creation_without_directory_fsync_support(tmp_path, monkeypatch):
    root = _toy_checkpoint(tmp_path / "checkpoint")
    monkeypatch.delattr(checkpoint_manifest.os, "O_DIRECTORY", raising=False)

    created = create_checkpoint_manifest(root, model="toy-model")

    assert verify_checkpoint_manifest(root, expected_model="toy-model") == created


def test_manifest_publication_preserves_destination_created_by_racer(tmp_path, monkeypatch):
    root = _toy_checkpoint(tmp_path / "checkpoint")
    manifest = root / CHECKPOINT_MANIFEST_NAME
    real_link = checkpoint_manifest.os.link

    def racing_link(source, target, **kwargs):
        Path(target).write_text("racer\n", encoding="utf-8")
        return real_link(source, target, **kwargs)

    monkeypatch.setattr(checkpoint_manifest.os, "link", racing_link)

    with pytest.raises(CheckpointManifestError, match="appeared during publication"):
        create_checkpoint_manifest(root, model="toy-model")

    assert manifest.read_text(encoding="utf-8") == "racer\n"
