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

"""Tests for exact, Git-free source-tree witnesses."""

import json
import os
import shutil
import subprocess
import tarfile
from hashlib import sha256
from pathlib import Path

import pytest

from modelopt.torch.sparsity.attention_sparsity.calibration import source_manifest
from modelopt.torch.sparsity.attention_sparsity.calibration.source_manifest import (
    SourceManifestError,
    create_source_manifest_from_git_archive,
    verify_source_manifest,
)

_COMMIT = "a" * 40
_TREE = "b" * 40
_ARCHIVE_SHA256 = "c" * 64


def _toy_source(root: Path) -> Path:
    (root / "flash_attn/cute").mkdir(parents=True)
    (root / "flash_attn/empty-dir").mkdir()
    (root / "flash_attn/cute/interface.py").write_text("# interface\n", encoding="utf-8")
    (root / "flash_attn/cute/block_sparsity.py").write_text("# sparse\n", encoding="utf-8")
    return root


def _write_witness(source: Path, manifest: Path) -> str:
    snapshot = source_manifest._source_tree_snapshot(source)
    raw = {
        "source_manifest_schema_version": 1,
        "source_kind": "flash-attention-4",
        "git_commit": _COMMIT,
        "git_tree": _TREE,
        "git_archive_sha256": _ARCHIVE_SHA256,
        "archive_scope": "flash_attn",
        "directories": list(snapshot.directories),
        "files": list(snapshot.files),
    }
    payload = (
        json.dumps(raw, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"
    ).encode()
    manifest.write_bytes(payload)
    return sha256(payload).hexdigest()


def _verify(source: Path, manifest: Path, digest: str):
    return verify_source_manifest(
        source,
        manifest,
        expected_manifest_sha256=digest,
        expected_commit=_COMMIT,
        expected_source_kind="flash-attention-4",
    )


def _symlink_or_skip(path: Path, target: str) -> None:
    try:
        path.symlink_to(target)
    except OSError as error:
        pytest.skip(f"symbolic links unavailable: {error}")


def test_source_witness_accepts_exact_files_and_directories(tmp_path):
    source = _toy_source(tmp_path / "source")
    manifest = tmp_path / "source-manifest.json"
    digest = _write_witness(source, manifest)

    verified = _verify(source, manifest, digest)

    assert verified.git_commit == _COMMIT
    assert verified.git_tree == _TREE
    assert verified.manifest_sha256 == digest
    assert verified.file_count == 2
    assert verified.directory_count == 3


@pytest.mark.parametrize(
    "change",
    [
        "mutated",
        "missing",
        "extra",
        pytest.param(
            "mode",
            marks=pytest.mark.skipif(os.name == "nt", reason="Windows has no POSIX execute bit"),
        ),
    ],
)
def test_source_witness_rejects_file_tree_changes(tmp_path, change):
    source = _toy_source(tmp_path / "source")
    manifest = tmp_path / "source-manifest.json"
    digest = _write_witness(source, manifest)
    interface = source / "flash_attn/cute/interface.py"
    if change == "mutated":
        interface.write_text("# modified\n", encoding="utf-8")
    elif change == "missing":
        interface.unlink()
    elif change == "extra":
        (source / "flash_attn/cute/shadow.py").write_text("# extra\n", encoding="utf-8")
    else:
        interface.chmod(interface.stat().st_mode | 0o111)

    with pytest.raises(SourceManifestError, match="does not exactly match"):
        _verify(source, manifest, digest)


def test_source_witness_rejects_regular_file_replaced_by_symlink(tmp_path):
    source = _toy_source(tmp_path / "source")
    manifest = tmp_path / "source-manifest.json"
    digest = _write_witness(source, manifest)
    interface = source / "flash_attn/cute/interface.py"
    interface.unlink()
    _symlink_or_skip(interface, "block_sparsity.py")

    with pytest.raises(SourceManifestError, match="must not be a link"):
        _verify(source, manifest, digest)


def test_source_witness_rejects_extra_symlink(tmp_path):
    source = _toy_source(tmp_path / "source")
    manifest = tmp_path / "source-manifest.json"
    digest = _write_witness(source, manifest)
    _symlink_or_skip(source / "flash_attn/unexpected-link", "cute/interface.py")
    with pytest.raises(SourceManifestError, match="must not be a link"):
        _verify(source, manifest, digest)


def test_source_witness_requires_independent_manifest_and_commit_pins(tmp_path):
    source = _toy_source(tmp_path / "source")
    manifest = tmp_path / "source-manifest.json"
    digest = _write_witness(source, manifest)

    with pytest.raises(SourceManifestError, match="expected SHA256"):
        _verify(source, manifest, "0" * 64)
    with pytest.raises(SourceManifestError, match="expected Git commit"):
        verify_source_manifest(
            source,
            manifest,
            expected_manifest_sha256=digest,
            expected_commit="0" * 40,
            expected_source_kind="flash-attention-4",
        )


def test_source_witness_rejects_boolean_schema_version(tmp_path):
    source = _toy_source(tmp_path / "source")
    manifest = tmp_path / "source-manifest.json"
    _write_witness(source, manifest)
    raw = json.loads(manifest.read_bytes())
    raw["source_manifest_schema_version"] = True
    payload = (
        json.dumps(raw, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"
    ).encode()
    manifest.write_bytes(payload)

    with pytest.raises(SourceManifestError, match="schema_version must be 1"):
        _verify(source, manifest, sha256(payload).hexdigest())


def test_source_artifact_temporary_is_cleaned_when_write_setup_fails(tmp_path, monkeypatch):
    destination = tmp_path / "source.tar"
    real_dup = source_manifest.os.dup

    def fail_dup(descriptor):
        assert descriptor >= 0
        raise OSError("injected dup failure")

    monkeypatch.setattr(source_manifest.os, "dup", fail_dup)

    with pytest.raises(OSError, match="injected dup failure"):
        source_manifest._temporary_payload(destination, b"archive")

    monkeypatch.setattr(source_manifest.os, "dup", real_dup)
    assert list(tmp_path.iterdir()) == []


def test_source_artifact_publication_preserves_destination_racer(tmp_path, monkeypatch):
    archive = tmp_path / "source.tar"
    manifest = tmp_path / "source-manifest.json"
    real_link = source_manifest.os.link
    link_count = 0

    def racing_link(source, destination, **kwargs):
        nonlocal link_count
        link_count += 1
        if link_count == 2:
            Path(destination).write_bytes(b"racer")
        return real_link(source, destination, **kwargs)

    monkeypatch.setattr(source_manifest.os, "link", racing_link)

    with pytest.raises(SourceManifestError, match="destination appeared"):
        source_manifest._publish_source_artifacts(archive, b"archive", manifest, b"manifest")

    assert not archive.exists()
    assert manifest.read_bytes() == b"racer"
    assert sorted(path.name for path in tmp_path.iterdir()) == [manifest.name]


def test_source_artifact_publication_preserves_post_publish_replacement(tmp_path, monkeypatch):
    archive = tmp_path / "source.tar"
    manifest = tmp_path / "source-manifest.json"
    real_stable_hash = source_manifest.stable_file_sha256

    def replace_archive_before_rehash(path, *, label):
        if label == "published source archive":
            Path(path).unlink()
            Path(path).write_bytes(b"racer")
        return real_stable_hash(path, label=label)

    monkeypatch.setattr(source_manifest, "stable_file_sha256", replace_archive_before_rehash)

    with pytest.raises(SourceManifestError, match="failed stable rehash"):
        source_manifest._publish_source_artifacts(archive, b"archive", manifest, b"manifest")

    assert archive.read_bytes() == b"racer"
    assert not manifest.exists()
    assert sorted(path.name for path in tmp_path.iterdir()) == [archive.name]


@pytest.mark.skipif(shutil.which("git") is None, reason="Git is required by the generator")
def test_generator_witnesses_the_exact_archive_without_runtime_git(tmp_path):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    subprocess.run(["git", "init", "-q", str(checkout)], check=True)
    subprocess.run(["git", "-C", str(checkout), "config", "user.name", "Test"], check=True)
    subprocess.run(
        ["git", "-C", str(checkout), "config", "user.email", "test@example.com"], check=True
    )
    _toy_source(checkout)
    subprocess.run(["git", "-C", str(checkout), "add", "."], check=True)
    subprocess.run(["git", "-C", str(checkout), "commit", "-qm", "source"], check=True)
    commit = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    archive = tmp_path / "source.tar"
    manifest = tmp_path / "source-manifest.json"

    generated = create_source_manifest_from_git_archive(
        checkout,
        expected_commit=commit,
        source_kind="flash-attention-4",
        archive_output=archive,
        manifest_output=manifest,
    )
    extracted = tmp_path / "extracted"
    extracted.mkdir()
    with tarfile.open(archive, "r:") as handle:
        for member in handle.getmembers():
            path = extracted / member.name
            if member.isdir():
                path.mkdir(parents=True, exist_ok=True)
            else:
                payload = handle.extractfile(member)
                assert payload is not None
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(payload.read())
                path.chmod(member.mode)

    verified = verify_source_manifest(
        extracted,
        manifest,
        expected_manifest_sha256=generated.manifest_sha256,
        expected_commit=commit,
        expected_source_kind="flash-attention-4",
    )
    assert verified.git_archive_sha256 == generated.git_archive_sha256


@pytest.mark.skipif(shutil.which("git") is None, reason="Git is required by the generator")
def test_generator_archive_excludes_ignored_checkout_artifacts(tmp_path):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    subprocess.run(["git", "init", "-q", str(checkout)], check=True)
    subprocess.run(["git", "-C", str(checkout), "config", "user.name", "Test"], check=True)
    subprocess.run(
        ["git", "-C", str(checkout), "config", "user.email", "test@example.com"], check=True
    )
    (checkout / ".gitignore").write_text("*.pyc\n", encoding="utf-8")
    (checkout / "flash_attn").mkdir()
    (checkout / "flash_attn/source.py").write_text("# source\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(checkout), "add", "."], check=True)
    subprocess.run(["git", "-C", str(checkout), "commit", "-qm", "source"], check=True)
    (checkout / "flash_attn/shadow.pyc").write_bytes(b"ignored")
    commit = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    archive = tmp_path / "source.tar"
    manifest = tmp_path / "source-manifest.json"
    generated = create_source_manifest_from_git_archive(
        checkout,
        expected_commit=commit,
        source_kind="flash-attention-4",
        archive_output=archive,
        manifest_output=manifest,
    )
    assert generated.manifest_sha256 == sha256(manifest.read_bytes()).hexdigest()
    with tarfile.open(archive, "r:") as handle:
        assert "flash_attn/shadow.pyc" not in handle.getnames()
