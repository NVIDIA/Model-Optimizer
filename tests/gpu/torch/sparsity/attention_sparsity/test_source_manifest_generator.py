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
import shutil
import subprocess
import tarfile
from hashlib import sha256

import pytest

from modelopt.torch.sparsity.attention_sparsity.calibration import source_manifest
from modelopt.torch.sparsity.attention_sparsity.calibration.source_manifest import (
    SourceManifestError,
    create_source_manifest_from_git_archive,
    verify_source_manifest,
)

_COMMIT = "a" * 40


@pytest.mark.skipif(shutil.which("git") is None, reason="Git is required by the generator")
@pytest.mark.parametrize("source_kind", ["flash-attention-4", "mask-reuse-reference"])
def test_generator_witnesses_the_exact_archive_without_runtime_git(tmp_path, source_kind):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    subprocess.run(["git", "init", "-q", str(checkout)], check=True)
    subprocess.run(["git", "-C", str(checkout), "config", "user.name", "Test"], check=True)
    subprocess.run(
        ["git", "-C", str(checkout), "config", "user.email", "test@example.com"], check=True
    )
    if source_kind == "flash-attention-4":
        (checkout / "flash_attn/cute").mkdir(parents=True)
        (checkout / "flash_attn/empty-dir").mkdir()
        (checkout / "flash_attn/cute/interface.py").write_text("# interface\n")
        (checkout / "flash_attn/cute/block_sparsity.py").write_text("# sparse\n")
    else:
        reference = checkout / "src/attention_backend/src/mask_reuse_vllm/reference"
        reference.mkdir(parents=True)
        (reference / "capture.py").write_text("# reference capture\n")
        (reference / "runtime.py").write_text("# reference runtime\n")
        (checkout / "src/modelopt/.agents").mkdir(parents=True)
        (checkout / "src/modelopt/plugins").mkdir()
        (checkout / "src/modelopt/plugins/skill.txt").write_text("# unrelated skill\n")
        try:
            (checkout / "src/modelopt/.agents/skills").symlink_to("../plugins")
        except OSError as error:
            pytest.skip(f"symbolic links unavailable: {error}")
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
        source_kind=source_kind,
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
        expected_source_kind=source_kind,
    )
    assert verified.git_archive_sha256 == generated.git_archive_sha256
    if source_kind == "mask-reuse-reference":
        raw = json.loads(manifest.read_bytes())
        assert raw["archive_scope"] == "src/attention_backend"
        assert "src" in raw["directories"]
        assert not (extracted / "src/modelopt").exists()
        assert (
            extracted / "src/attention_backend/src/mask_reuse_vllm/reference/capture.py"
        ).is_file()
        assert (checkout / "src/modelopt/.agents/skills").is_symlink()
        # A common string prefix is not scope membership, and an ancestor may
        # appear only as a directory, never as an archived file.
        for outside in ("src/modelopt/shadow.py", "src/attention_backend_extra/shadow.py", "src"):
            changed = {**raw, "files": [{**raw["files"][0], "path": outside}]}
            with pytest.raises(SourceManifestError, match="must be within"):
                source_manifest._parse_manifest(source_manifest._canonical_json_bytes(changed))


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
