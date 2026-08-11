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
from typing import TYPE_CHECKING

import pytest

from examples.puzzletron.main import _completion_is_valid, _mark_completion, _resume_kwargs
from modelopt.torch.puzzletron.identity import stable_hash
from modelopt.torch.puzzletron.manifest import (
    StageManifest,
    validate_stage_execution_record,
    write_stage_manifest,
)

if TYPE_CHECKING:
    from pathlib import Path


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


def test_execution_record_binds_terminal_skip_reason(tmp_path: Path) -> None:
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text("model: {}\n")
    config = _stage_config(tmp_path, config_path)
    manifest = StageManifest(stage="tokenize_data", config=config)
    manifest.complete(status="skipped", skip_reason="disabled")
    manifest_path = tmp_path / "manifests" / "tokenize_data.json"

    write_stage_manifest(manifest_path, manifest)

    pointer = json.loads(manifest_path.read_text())
    record = pointer["execution_record"]
    resolved = json.loads((tmp_path / record["resolved_config_path"]).read_text())
    artifact = json.loads((tmp_path / record["artifact_manifest_path"]).read_text())
    assert pointer["skip_reason"] == resolved["skip_reason"] == artifact["skip_reason"]

    pointer["skip_reason"] = "tampered"
    manifest_path.write_text(json.dumps(pointer))
    with pytest.raises(ValueError, match="resolved stage execution skip_reason mismatch"):
        validate_stage_execution_record(manifest_path, expected_stage="tokenize_data")


def test_artifact_manifest_separates_output_pointer_from_immutable_evidence(
    tmp_path: Path,
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

    canonical = json.loads(manifest_path.read_text())
    record = canonical["execution_record"]
    artifacts = json.loads((tmp_path / record["artifact_manifest_path"]).read_text())
    assert artifacts["canonical_output_pointers"] == {"summary_path": str(summary_path)}
    assert artifacts["artifact_contract"] == "stage-manifest-output-pointers/v1"
    assert artifacts["immutable_evidence"]["summary_path"] == {
        "path": str(summary_path),
        "size": len(summary_path.read_bytes()),
        "sha256": hashlib.sha256(summary_path.read_bytes()).hexdigest(),
    }
    assert artifacts["started_at"] == canonical["started_at"]
    assert artifacts["ended_at"] == canonical["ended_at"]
    assert artifacts["artifact_manifest_identity"] == record["artifact_manifest_identity"]
    resolved_bytes = (tmp_path / record["resolved_config_path"]).read_bytes()
    assert artifacts["resolved_config"]["sha256"] == hashlib.sha256(resolved_bytes).hexdigest()
    required_patterns = _resume_kwargs(config, config_path, "convert")["required_patterns"]
    assert record["resolved_config_path"] in required_patterns
    assert record["artifact_manifest_path"] in required_patterns


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


def test_ephemeral_runtime_change_preserves_resolved_config_identity(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text("model: {}\n")
    authored_config = _stage_config(tmp_path, config_path)
    original_effective_config = _stage_config(tmp_path, config_path)
    original = StageManifest(
        stage="convert",
        config=authored_config,
        effective_config=original_effective_config,
        started_at="2026-08-06T10:00:00+00:00",
        ended_at="2026-08-06T10:01:00+00:00",
        status="success",
    )
    changed_effective_config = _stage_config(tmp_path, config_path)
    changed_effective_config["_runtime"]["descriptor"] = "llama"
    changed = StageManifest(
        stage="convert",
        config=authored_config,
        effective_config=changed_effective_config,
        started_at=original.started_at,
        ended_at=original.ended_at,
        status="success",
    )
    manifest_path = tmp_path / "manifests" / "convert.json"

    write_stage_manifest(manifest_path, original)
    write_stage_manifest(manifest_path, changed)

    assert (
        original.execution_record["resolved_config_identity"]
        == (changed.execution_record["resolved_config_identity"])
    )
    assert (tmp_path / original.execution_record["resolved_config_path"]).is_file()
    assert (tmp_path / changed.execution_record["resolved_config_path"]).is_file()


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


def test_implementation_provenance_change_creates_a_distinct_execution_record(
    tmp_path: Path,
) -> None:
    manifest_path, _config_path, config, original = _write_convert_record(
        tmp_path, implementation_provenance={"revision": "worker-v1"}
    )
    changed = StageManifest(
        stage="convert",
        config=config,
        implementation_provenance={"revision": "worker-v2"},
        started_at=original.started_at,
        ended_at=original.ended_at,
        status=original.status,
    )

    write_stage_manifest(manifest_path, changed)

    assert (
        original.execution_record["execution_identity"]
        != (changed.execution_record["execution_identity"])
    )
    assert (
        original.execution_record["resolved_config_identity"]
        == (changed.execution_record["resolved_config_identity"])
    )


def test_effective_config_none_falls_back_but_empty_mapping_is_preserved(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text("model: {}\n")
    authored_config = _stage_config(tmp_path, config_path)
    fallback = StageManifest(
        stage="convert",
        config=authored_config,
        effective_config=None,
        started_at="2026-08-06T11:00:00+00:00",
        ended_at="2026-08-06T11:01:00+00:00",
        status="success",
    )
    empty = StageManifest(
        stage="convert",
        config=authored_config,
        effective_config={},
        started_at="2026-08-06T11:02:00+00:00",
        ended_at="2026-08-06T11:03:00+00:00",
        status="success",
    )
    manifest_path = tmp_path / "manifests" / "convert.json"

    write_stage_manifest(manifest_path, fallback)
    fallback_path = tmp_path / fallback.execution_record["resolved_config_path"]
    write_stage_manifest(manifest_path, empty)
    empty_path = tmp_path / empty.execution_record["resolved_config_path"]

    assert json.loads(fallback_path.read_text())["resolved_stage_config"] == {
        "convert": authored_config["convert"],
        "model": authored_config["model"],
    }
    assert json.loads(empty_path.read_text())["resolved_stage_config"] == {}


def test_artifact_contract_preserves_declared_output_before_it_exists(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text("model: {}\n")
    late_output = tmp_path / "ckpts" / "scoring-parent.json"
    manifest = StageManifest(stage="convert", config=_stage_config(tmp_path, config_path))
    manifest.complete(outputs={"scoring_parent_artifact": str(late_output)})

    write_stage_manifest(tmp_path / "manifests" / "convert.json", manifest)

    artifact_path = tmp_path / manifest.execution_record["artifact_manifest_path"]
    artifact_manifest = json.loads(artifact_path.read_text())
    assert artifact_manifest["artifact_contract"] == "stage-manifest-output-pointers/v1"
    assert artifact_manifest["canonical_output_pointers"] == {
        "scoring_parent_artifact": str(late_output)
    }
    assert artifact_manifest["immutable_evidence"] == {}
    assert not late_output.exists()


def test_resume_remains_compatible_with_historical_stage_manifest(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text("model: {}\n")
    teacher_config = tmp_path / "ckpts" / "teacher" / "config.json"
    teacher_config.parent.mkdir(parents=True)
    teacher_config.write_text("{}\n")
    config = _stage_config(tmp_path, config_path)
    manifest = StageManifest(
        stage="convert",
        status="success",
        outputs={"teacher_dir": str(teacher_config.parent)},
        config=config,
    )
    historical_payload = manifest.to_dict()
    assert "execution_record" not in historical_payload
    manifest_path = tmp_path / "manifests" / "convert.json"
    manifest_path.parent.mkdir()
    manifest_path.write_text(json.dumps(historical_payload))

    _mark_completion(config, config_path, "convert")

    assert _completion_is_valid(config, config_path, "convert")


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
    ("record_key", "value"),
    [
        ("resolved_config_path", None),
        ("resolved_config_path", ""),
        ("resolved_config_path", 1),
        ("artifact_manifest_path", None),
        ("artifact_manifest_path", ""),
        ("artifact_manifest_path", 1),
    ],
    ids=[
        "missing-resolved-path",
        "empty-resolved-path",
        "non-string-resolved-path",
        "missing-artifact-path",
        "empty-artifact-path",
        "non-string-artifact-path",
    ],
)
def test_resume_rejects_malformed_execution_record_paths(
    tmp_path: Path,
    record_key: str,
    value: object,
) -> None:
    manifest_path, config_path, config, _manifest = _write_convert_record(tmp_path)
    pointer = json.loads(manifest_path.read_text())
    if value is None:
        pointer["execution_record"].pop(record_key)
    else:
        pointer["execution_record"][record_key] = value
    manifest_path.write_text(json.dumps(pointer))

    with pytest.raises(ValueError, match="invalid stage execution record path"):
        _resume_kwargs(config, config_path, "convert")


@pytest.mark.parametrize(
    "execution_identity",
    [None, "", 1],
    ids=["missing", "empty", "non-string"],
)
def test_resume_rejects_invalid_execution_identity(
    tmp_path: Path,
    execution_identity: object,
) -> None:
    manifest_path, config_path, config, _manifest = _write_convert_record(tmp_path)
    pointer = json.loads(manifest_path.read_text())
    if execution_identity is None:
        pointer["execution_record"].pop("execution_identity")
    else:
        pointer["execution_record"]["execution_identity"] = execution_identity
    manifest_path.write_text(json.dumps(pointer))

    with pytest.raises(ValueError, match="invalid stage execution identity"):
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


def test_writer_rejects_symlinked_execution_record_ancestor(tmp_path: Path) -> None:
    manifests_dir = tmp_path / "manifests"
    manifests_dir.mkdir()
    external_executions = tmp_path / "external-executions"
    external_executions.mkdir()
    (manifests_dir / "executions").symlink_to(external_executions, target_is_directory=True)
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text("model: {}\n")
    manifest = StageManifest(stage="convert", config=_stage_config(tmp_path, config_path))
    manifest.complete()

    with pytest.raises(ValueError, match="stage execution record path is symlinked"):
        write_stage_manifest(manifests_dir / "convert.json", manifest)


@pytest.mark.parametrize(
    "record_key",
    ["resolved_config_path", "artifact_manifest_path"],
    ids=["resolved-config", "artifact-manifest"],
)
def test_writer_rejects_symlinked_execution_record_leaf(
    tmp_path: Path,
    record_key: str,
) -> None:
    manifest_path, _config_path, _config, manifest = _write_convert_record(tmp_path)
    record_path = tmp_path / manifest.execution_record[record_key]
    external_file = tmp_path / f"external-{record_path.name}"
    external_file.write_bytes(record_path.read_bytes())
    record_path.unlink()
    record_path.symlink_to(external_file)

    with pytest.raises(ValueError, match="stage execution record path is symlinked"):
        write_stage_manifest(manifest_path, manifest)


@pytest.mark.parametrize(
    "location",
    ["stage-ancestor", "resolved-config-leaf", "artifact-manifest-leaf"],
)
def test_validator_rejects_symlinked_execution_record_paths(
    tmp_path: Path,
    location: str,
) -> None:
    manifest_path, _config_path, _config, manifest = _write_convert_record(tmp_path)
    if location == "stage-ancestor":
        stage_dir = (tmp_path / manifest.execution_record["resolved_config_path"]).parent.parent
        external_stage_dir = stage_dir.with_name("external-convert")
        stage_dir.rename(external_stage_dir)
        stage_dir.symlink_to(external_stage_dir, target_is_directory=True)
    else:
        record_key = (
            "resolved_config_path"
            if location == "resolved-config-leaf"
            else "artifact_manifest_path"
        )
        record_path = tmp_path / manifest.execution_record[record_key]
        external_file = tmp_path / f"external-{record_path.name}"
        external_file.write_bytes(record_path.read_bytes())
        record_path.unlink()
        record_path.symlink_to(external_file)

    with pytest.raises(ValueError, match="stage execution record path is symlinked"):
        validate_stage_execution_record(manifest_path, expected_stage="convert")


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


def test_artifact_content_tamper_is_rejected_after_outer_sha_is_updated(
    tmp_path: Path,
) -> None:
    manifest_path, _config_path, _config, manifest = _write_convert_record(tmp_path)
    pointer = json.loads(manifest_path.read_text())
    artifact_path = tmp_path / manifest.execution_record["artifact_manifest_path"]
    artifact = json.loads(artifact_path.read_text())
    artifact["immutable_evidence"]["forged"] = {"sha256": "0" * 64}
    pointer["execution_record"]["artifact_manifest_sha256"] = _rewrite_json_record(
        artifact_path, artifact
    )
    _rewrite_json_record(manifest_path, pointer)

    with pytest.raises(ValueError, match="stage artifact record identity mismatch"):
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


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("config", "stage manifest config identity mismatch"),
        ("semantic_config", "stage manifest semantic config identity mismatch"),
        ("capability_snapshot", "stage manifest semantic identity mismatch"),
        ("inputs", "resolved descriptor input provenance mismatch"),
        ("implementation_provenance", "resolved implementation provenance mismatch"),
    ],
    ids=[
        "authored-config",
        "semantic-config",
        "capability",
        "descriptor-input",
        "implementation-provenance",
    ],
)
def test_resume_recomputes_canonical_pointer_identities(
    tmp_path: Path,
    field: str,
    message: str,
) -> None:
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text("model: {}\n")
    manifest = StageManifest(
        stage="convert",
        config=_stage_config(tmp_path, config_path),
        inputs={"descriptor_resolution": {"descriptor": "llama"}},
        capability_snapshot={"backend": "hf"},
    )
    manifest.complete()
    manifest_path = tmp_path / "manifests" / "convert.json"
    write_stage_manifest(manifest_path, manifest)
    pointer = json.loads(manifest_path.read_text())
    if field in {"config", "semantic_config"}:
        pointer[field]["model"]["revision"] = "tampered"
    elif field == "capability_snapshot":
        pointer[field]["backend"] = "tampered"
    elif field == "inputs":
        pointer[field]["descriptor_resolution"]["descriptor"] = "tampered"
    else:
        pointer[field]["reviewer"] = "tampered"
    manifest_path.write_text(json.dumps(pointer))

    with pytest.raises(ValueError, match=message):
        validate_stage_execution_record(manifest_path, expected_stage="convert")
