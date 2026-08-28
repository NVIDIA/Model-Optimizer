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

"""Tests for immutable per-stage resolved configuration and artifact records."""

from __future__ import annotations

import hashlib
import json
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

import pytest

from examples.puzzletron.main import _completion_is_valid, _mark_completion, _resume_kwargs
from modelopt.torch.puzzletron import execution_record
from modelopt.torch.puzzletron.identity import stable_hash
from modelopt.torch.puzzletron.manifest import (
    StageManifest,
    validate_stage_execution_record,
    write_stage_manifest,
)


def _stage_config(tmp_path: Path, config_path: Path) -> dict:
    return {
        "experiment": {"dir": str(tmp_path)},
        "model": {"source": "example/model", "revision": "revision-1"},
        "convert": {"teacher_dir": str(tmp_path / "ckpts" / "teacher")},
        "_runtime": {
            "config_path": str(config_path),
            "overrides": ["model.revision=revision-1"],
            "gpus_per_node": 4,
        },
    }


def _write_convert_record(
    tmp_path: Path,
    *,
    inputs: dict | None = None,
    outputs: dict | None = None,
    capability_snapshot: dict | None = None,
    implementation_provenance: dict | None = None,
) -> tuple[Path, Path, dict, StageManifest]:
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text("model: {}\n")
    config = _stage_config(tmp_path, config_path)
    manifest = StageManifest(
        stage="convert",
        config=config,
        inputs=inputs or {},
        capability_snapshot=capability_snapshot,
        implementation_provenance=implementation_provenance or {},
        started_at="2026-08-06T10:00:00+00:00",
        ended_at="2026-08-06T10:01:00+00:00",
        status="success",
    )
    if outputs is not None:
        manifest.outputs = outputs
    manifest_path = tmp_path / "manifests" / "convert.json"
    write_stage_manifest(manifest_path, manifest)
    return manifest_path, config_path, config, manifest


def _rewrite_json_record(path: Path, payload: dict) -> str:
    content = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    path.write_text(content)
    return hashlib.sha256(content.encode()).hexdigest()


def test_stage_manifest_persists_path_neutral_resolved_stage_config(
    tmp_path: Path,
) -> None:
    manifest_path, _config_path, config, _manifest = _write_convert_record(tmp_path)

    canonical = json.loads(manifest_path.read_text())
    record = canonical["execution_record"]
    resolved = json.loads((tmp_path / record["resolved_config_path"]).read_text())
    assert "effective_config" not in canonical
    assert resolved["resolved_stage_config"] == {
        "convert": config["convert"],
        "model": config["model"],
    }
    assert resolved["resolved_config_identity"] == stable_hash(
        resolved["resolved_stage_config"], prefix="convert_resolved_cfg"
    )


def test_nonzero_rank_does_not_publish_or_mutate_execution_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text("model: {}\n")
    manifest = StageManifest(stage="convert", config=_stage_config(tmp_path, config_path))
    manifest.complete()
    manifest_path = tmp_path / "manifests" / "convert.json"
    monkeypatch.setenv("RANK", "1")

    write_stage_manifest(manifest_path, manifest)

    assert manifest.version == "1"
    assert manifest.execution_record is None
    assert not manifest_path.exists()
    assert not (tmp_path / "manifests" / "executions").exists()


def test_execution_record_preserves_provenance_and_cross_record_timestamps(
    tmp_path: Path,
) -> None:
    manifest_path, config_path, _config, _manifest = _write_convert_record(
        tmp_path,
        inputs={"descriptor_resolution": {"descriptor": "llama"}},
        capability_snapshot={"backend": "hf"},
        implementation_provenance={"revision": "worker-v1"},
    )
    canonical = json.loads(manifest_path.read_text())
    record = canonical["execution_record"]
    resolved = json.loads((tmp_path / record["resolved_config_path"]).read_text())
    artifact = json.loads((tmp_path / record["artifact_manifest_path"]).read_text())

    assert resolved["started_at"] == artifact["started_at"] == canonical["started_at"]
    assert resolved["ended_at"] == artifact["ended_at"] == canonical["ended_at"]
    assert resolved["authored_config_identity"] == canonical["config_identity"]
    assert resolved["resolved_config_identity"] == record["resolved_config_identity"]
    assert resolved["provenance"] == {
        "authored_config_path": str(config_path),
        "overrides": ["model.revision=revision-1"],
        "implementation": {"revision": "worker-v1"},
        "descriptor_resolution": {"descriptor": "llama"},
        "capability_snapshot": {"backend": "hf"},
    }
    assert artifact["stage_manifest"] == {
        "path": "manifests/convert.json",
        "semantic_identity": canonical["semantic_identity"],
    }


@pytest.mark.parametrize(
    "record_key",
    ["resolved_config_path", "artifact_manifest_path"],
    ids=["resolved-config", "artifact-manifest"],
)
def test_stage_execution_record_is_immutable_and_idempotent(
    tmp_path: Path,
    record_key: str,
) -> None:
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text("model: {}\n")
    manifest = StageManifest(
        stage="convert",
        config=_stage_config(tmp_path, config_path),
        started_at="2026-08-06T10:00:00+00:00",
        ended_at="2026-08-06T10:01:00+00:00",
        status="success",
    )
    manifest_path = tmp_path / "manifests" / "convert.json"
    write_stage_manifest(manifest_path, manifest)
    write_stage_manifest(manifest_path, manifest)

    record_path = tmp_path / manifest.execution_record[record_key]
    record_path.write_text("{}\n")

    with pytest.raises(FileExistsError, match="immutable stage execution record"):
        write_stage_manifest(manifest_path, manifest)


def test_same_config_rerun_creates_a_distinct_execution_record(tmp_path: Path) -> None:
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text("model: {}\n")
    config = _stage_config(tmp_path, config_path)
    original = StageManifest(
        stage="convert",
        config=config,
        started_at="2026-08-06T10:00:00+00:00",
        ended_at="2026-08-06T10:01:00+00:00",
        status="success",
    )
    rerun = StageManifest(
        stage="convert",
        config=config,
        started_at="2026-08-06T11:00:00+00:00",
        ended_at="2026-08-06T11:01:00+00:00",
        status="success",
    )
    manifest_path = tmp_path / "manifests" / "convert.json"

    write_stage_manifest(manifest_path, original)
    write_stage_manifest(manifest_path, rerun)

    assert (
        original.execution_record["execution_identity"]
        != (rerun.execution_record["execution_identity"])
    )
    assert (
        original.execution_record["resolved_config_identity"]
        == (rerun.execution_record["resolved_config_identity"])
    )


def test_version_two_manifest_requires_execution_record_pointer(tmp_path: Path) -> None:
    manifest_path, _config_path, _config, _manifest = _write_convert_record(tmp_path)
    pointer = json.loads(manifest_path.read_text())
    assert pointer["version"] == "2"
    pointer.pop("execution_record")
    manifest_path.write_text(json.dumps(pointer))

    with pytest.raises(ValueError, match="requires an execution record"):
        validate_stage_execution_record(manifest_path, expected_stage="convert")


@pytest.mark.parametrize(
    "record_key",
    ["resolved_config_path", "artifact_manifest_path"],
    ids=["resolved-config", "artifact-manifest"],
)
def test_resume_requires_both_execution_record_files(tmp_path: Path, record_key: str) -> None:
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text("model: {}\n")
    teacher_config = tmp_path / "ckpts" / "teacher" / "config.json"
    teacher_config.parent.mkdir(parents=True)
    teacher_config.write_text("{}\n")
    config = _stage_config(tmp_path, config_path)
    manifest = StageManifest(stage="convert", config=config)
    manifest.complete(outputs={"teacher_dir": str(teacher_config.parent)})

    manifest_path = tmp_path / "manifests" / "convert.json"
    write_stage_manifest(manifest_path, manifest)
    _mark_completion(config, config_path, "convert")
    assert _completion_is_valid(config, config_path, "convert")

    (tmp_path / manifest.execution_record[record_key]).unlink()
    with pytest.raises(ValueError, match=r"invalid .* record"):
        _completion_is_valid(config, config_path, "convert")


@pytest.mark.parametrize(
    "record",
    [None, {}],
    ids=["null-record", "empty-record"],
)
def test_resume_rejects_malformed_execution_record_structure(
    tmp_path: Path,
    record: object,
) -> None:
    manifest_path, config_path, config, _manifest = _write_convert_record(tmp_path)
    payload = json.loads(manifest_path.read_text())
    payload["execution_record"] = record
    manifest_path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="invalid stage execution record"):
        _resume_kwargs(config, config_path, "convert")


@pytest.mark.parametrize(
    "record_key",
    ["resolved_config_path", "artifact_manifest_path"],
    ids=["resolved-config", "artifact-manifest"],
)
@pytest.mark.parametrize("unsafe_path", ["../outside.json", "/tmp/outside.json"])
def test_resume_rejects_unsafe_execution_record_paths(
    tmp_path: Path,
    record_key: str,
    unsafe_path: str,
) -> None:
    manifest_path, config_path, config, _manifest = _write_convert_record(tmp_path)
    payload = json.loads(manifest_path.read_text())
    payload["execution_record"][record_key] = unsafe_path
    manifest_path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="unsafe stage execution record path"):
        _resume_kwargs(config, config_path, "convert")


@pytest.mark.parametrize(
    ("record_key", "message"),
    [
        ("resolved_config_path", "resolved stage configuration SHA256 mismatch"),
        ("artifact_manifest_path", "stage artifact record SHA256 mismatch"),
    ],
    ids=["resolved-config", "artifact-manifest"],
)
def test_resume_rejects_tampered_immutable_stage_record(
    tmp_path: Path,
    record_key: str,
    message: str,
) -> None:
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text("model: {}\n")
    config = _stage_config(tmp_path, config_path)
    manifest = StageManifest(stage="convert", config=config)
    manifest.complete()
    manifest_path = tmp_path / "manifests" / "convert.json"
    write_stage_manifest(manifest_path, manifest)
    record_path = tmp_path / manifest.execution_record[record_key]
    payload = json.loads(record_path.read_text())
    payload["status"] = "tampered"
    record_path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match=message):
        _resume_kwargs(config, config_path, "convert")


@pytest.mark.skipif(
    not (
        hasattr(os, "O_DIRECTORY")
        and hasattr(os, "O_NOFOLLOW")
        and os.open in getattr(os, "supports_dir_fd", set())
    ),
    reason="requires descriptor-rooted no-follow traversal",
)
@pytest.mark.parametrize("swap_location", ["ancestor", "leaf"])
def test_validator_rejects_execution_record_symlink_swap_after_path_check(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    swap_location: str,
) -> None:
    manifest_path, _config_path, _config, manifest = _write_convert_record(tmp_path)
    record_path = tmp_path / manifest.execution_record["resolved_config_path"]
    original_path_without_symlinks = execution_record._path_without_symlinks
    swapped = False

    def swap_after_check(path: Path, *, description: str) -> Path:
        nonlocal swapped
        checked = original_path_without_symlinks(path, description=description)
        if not swapped and description == "resolved stage configuration record":
            if swap_location == "ancestor":
                stage_dir = record_path.parent.parent
                external_stage_dir = stage_dir.with_name("external-convert")
                stage_dir.rename(external_stage_dir)
                stage_dir.symlink_to(external_stage_dir, target_is_directory=True)
            else:
                external_file = tmp_path / "external-resolved-config.json"
                external_file.write_bytes(record_path.read_bytes())
                record_path.unlink()
                record_path.symlink_to(external_file)
            swapped = True
        return checked

    monkeypatch.setattr(execution_record, "_path_without_symlinks", swap_after_check)

    with pytest.raises(ValueError, match="invalid resolved stage configuration record"):
        validate_stage_execution_record(manifest_path, expected_stage="convert")


@pytest.mark.parametrize(
    ("replacement", "message"),
    [
        ('{"score": 2}\n', "stage output evidence SHA256 mismatch"),
        ('{"score": 200}\n', "stage output evidence size mismatch"),
        (None, "invalid stage output evidence file"),
    ],
    ids=["digest", "size", "missing"],
)
def test_resume_rejects_modified_evidence_backed_output(
    tmp_path: Path,
    replacement: str | None,
    message: str,
) -> None:
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text("model: {}\n")
    summary_path = tmp_path / "artifacts" / "summary.json"
    summary_path.parent.mkdir()
    summary_path.write_text('{"score": 1}\n')
    config = _stage_config(tmp_path, config_path)
    manifest = StageManifest(stage="convert", config=config)
    manifest.complete(outputs={"summary_path": str(summary_path)})
    manifest_path = tmp_path / "manifests" / "convert.json"
    write_stage_manifest(manifest_path, manifest)

    if replacement is None:
        summary_path.unlink()
    else:
        summary_path.write_text(replacement)

    with pytest.raises(ValueError, match=message):
        _resume_kwargs(config, config_path, "convert")


@pytest.mark.parametrize("pointer_kind", ["absolute", "relative"])
def test_writer_rejects_evidence_backed_output_outside_campaign_root(
    tmp_path: Path,
    pointer_kind: str,
) -> None:
    campaign_root = tmp_path / "campaign"
    campaign_root.mkdir()
    summary_path = tmp_path / "summary.json"
    summary_path.write_text('{"score": 1}\n')
    output_pointer = str(summary_path) if pointer_kind == "absolute" else "../summary.json"

    with pytest.raises(ValueError, match="outside the campaign root"):
        _write_convert_record(
            campaign_root,
            outputs={"summary_path": output_pointer},
        )


def test_resolved_content_tamper_is_rejected_after_outer_sha_is_updated(
    tmp_path: Path,
) -> None:
    manifest_path, _config_path, _config, manifest = _write_convert_record(tmp_path)
    pointer = json.loads(manifest_path.read_text())
    resolved_path = tmp_path / manifest.execution_record["resolved_config_path"]
    resolved = json.loads(resolved_path.read_text())
    resolved["resolved_stage_config"]["model"]["revision"] = "tampered"
    pointer["execution_record"]["resolved_config_sha256"] = _rewrite_json_record(
        resolved_path, resolved
    )
    _rewrite_json_record(manifest_path, pointer)

    with pytest.raises(ValueError, match="resolved stage configuration identity mismatch"):
        validate_stage_execution_record(manifest_path, expected_stage="convert")


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("resolved_config", "stage artifact resolved-configuration reference mismatch"),
        (
            "canonical_output_pointers",
            "stage artifact canonical output pointers mismatch",
        ),
    ],
    ids=["resolved-config-cross-reference", "canonical-output-pointers"],
)
def test_resealed_artifact_tamper_still_fails_cross_record_validation(
    tmp_path: Path,
    field: str,
    message: str,
) -> None:
    manifest_path, _config_path, _config, manifest = _write_convert_record(tmp_path)
    pointer = json.loads(manifest_path.read_text())
    artifact_path = tmp_path / manifest.execution_record["artifact_manifest_path"]
    artifact = json.loads(artifact_path.read_text())
    if field == "resolved_config":
        artifact[field]["identity"] = "tampered"
    else:
        artifact[field] = {"forged": "output.json"}
    identity_payload = dict(artifact)
    identity_payload.pop("artifact_manifest_identity")
    artifact_identity = stable_hash(identity_payload, prefix="convert_artifacts")
    artifact["artifact_manifest_identity"] = artifact_identity
    pointer["execution_record"]["artifact_manifest_identity"] = artifact_identity
    pointer["execution_record"]["artifact_manifest_sha256"] = _rewrite_json_record(
        artifact_path, artifact
    )
    _rewrite_json_record(manifest_path, pointer)

    with pytest.raises(ValueError, match=message):
        validate_stage_execution_record(manifest_path, expected_stage="convert")
