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
import sys
from typing import TYPE_CHECKING

import pytest

from examples.puzzletron.main import (
    _completion_is_valid,
    _mark_completion,
    _pipeline_config_from_path,
    _resume_kwargs,
)
from modelopt.torch.puzzletron.identity import stable_hash
from modelopt.torch.puzzletron.manifest import StageManifest, write_stage_manifest
from modelopt.torch.puzzletron.pipeline_config import load_runtime_hydra_config

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


def test_stage_manifest_persists_resolved_config_and_artifact_manifest(tmp_path: Path) -> None:
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text("model:\n  revision: revision-1\n")
    summary_path = tmp_path / "artifacts" / "summary.json"
    summary_path.parent.mkdir()
    summary_path.write_text('{"score": 1}\n')
    config = _stage_config(tmp_path, config_path)
    manifest = StageManifest(
        stage="convert",
        config=config,
        implementation_provenance={"source_identity": "source-v1"},
    )
    manifest.complete(outputs={"summary_path": str(summary_path)})

    manifest_path = tmp_path / "manifests" / "convert.json"
    write_stage_manifest(manifest_path, manifest)

    canonical = json.loads(manifest_path.read_text())
    record = canonical["execution_record"]
    resolved = json.loads((tmp_path / record["resolved_config_path"]).read_text())
    artifacts = json.loads((tmp_path / record["artifact_manifest_path"]).read_text())
    assert resolved["effective_config"] == config
    assert resolved["started_at"] == canonical["started_at"]
    assert resolved["ended_at"] == canonical["ended_at"]
    assert resolved["authored_config_identity"] == canonical["config_identity"]
    assert resolved["resolved_config_identity"] == record["resolved_config_identity"]
    assert resolved["provenance"] == {
        "authored_config_path": str(config_path),
        "overrides": ["model.revision=revision-1"],
        "implementation": {"source_identity": "source-v1"},
        "descriptor_resolution": None,
        "capability_snapshot": None,
    }
    assert resolved["replay"]["argv"] == [
        sys.executable,
        f"{resolved['replay']['working_directory']}/examples/puzzletron/main.py",
        "--config",
        str(tmp_path / record["resolved_config_path"]),
        "--stage",
        "convert",
        "--gpus-per-node",
        "4",
        "--force",
    ]
    assert resolved["resolved_config_identity"] == stable_hash(
        resolved["effective_config"], prefix="convert_resolved_cfg"
    )
    assert artifacts["declared_outputs"] == {"summary_path": str(summary_path)}
    assert artifacts["artifact_contract"] == "stage-manifest-outputs"
    assert artifacts["started_at"] == canonical["started_at"]
    assert artifacts["ended_at"] == canonical["ended_at"]
    assert artifacts["artifact_manifest_identity"] == record["artifact_manifest_identity"]
    resolved_bytes = (tmp_path / record["resolved_config_path"]).read_bytes()
    assert artifacts["resolved_config"]["sha256"] == hashlib.sha256(resolved_bytes).hexdigest()
    required_patterns = _resume_kwargs(config, config_path, "convert")["required_patterns"]
    assert record["resolved_config_path"] in required_patterns
    assert record["artifact_manifest_path"] in required_patterns

    config_path.write_text("model:\n  revision: mutated-after-execution\n")
    replayed = _pipeline_config_from_path(tmp_path / record["resolved_config_path"])
    assert replayed["model"] == config["model"]
    assert replayed["_runtime"]["resolved_config_path"] == str(
        (tmp_path / record["resolved_config_path"]).resolve()
    )
    assert load_runtime_hydra_config(replayed).model.revision == "revision-1"


@pytest.mark.parametrize("record_key", ["resolved_config_path", "artifact_manifest_path"])
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


def test_effective_config_change_creates_a_distinct_execution_record(tmp_path: Path) -> None:
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text("model: {}\n")
    original = StageManifest(
        stage="convert",
        config=_stage_config(tmp_path, config_path),
        started_at="2026-08-06T10:00:00+00:00",
        ended_at="2026-08-06T10:01:00+00:00",
        status="success",
    )
    changed_config = _stage_config(tmp_path, config_path)
    changed_config["model"]["revision"] = "revision-2"
    changed = StageManifest(
        stage="convert",
        config=changed_config,
        started_at=original.started_at,
        ended_at=original.ended_at,
        status="success",
    )
    manifest_path = tmp_path / "manifests" / "convert.json"

    write_stage_manifest(manifest_path, original)
    write_stage_manifest(manifest_path, changed)

    assert (
        original.execution_record["execution_identity"]
        != (changed.execution_record["execution_identity"])
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


def test_effective_config_none_falls_back_but_empty_mapping_is_preserved(tmp_path: Path) -> None:
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

    assert json.loads(fallback_path.read_text())["effective_config"] == authored_config
    assert json.loads(empty_path.read_text())["effective_config"] == {}


def test_artifact_contract_preserves_declared_output_before_it_exists(tmp_path: Path) -> None:
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text("model: {}\n")
    late_output = tmp_path / "ckpts" / "scoring-parent.json"
    manifest = StageManifest(stage="convert", config=_stage_config(tmp_path, config_path))
    manifest.complete(outputs={"scoring_parent_artifact": str(late_output)})

    write_stage_manifest(tmp_path / "manifests" / "convert.json", manifest)

    artifact_path = tmp_path / manifest.execution_record["artifact_manifest_path"]
    artifact_manifest = json.loads(artifact_path.read_text())
    assert artifact_manifest["artifact_contract"] == "stage-manifest-outputs"
    assert artifact_manifest["declared_outputs"] == {"scoring_parent_artifact": str(late_output)}
    assert not late_output.exists()


def test_resolved_config_loader_rejects_tampering(tmp_path: Path) -> None:
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text("model: {}\n")
    manifest = StageManifest(stage="convert", config=_stage_config(tmp_path, config_path))
    manifest.complete()
    write_stage_manifest(tmp_path / "manifests" / "convert.json", manifest)
    resolved_path = tmp_path / manifest.execution_record["resolved_config_path"]
    resolved = json.loads(resolved_path.read_text())
    resolved["effective_config"]["model"]["revision"] = "tampered"
    resolved_path.write_text(json.dumps(resolved))

    with pytest.raises(ValueError, match="identity mismatch"):
        _pipeline_config_from_path(resolved_path)


def test_resume_remains_compatible_with_historical_stage_manifest(tmp_path: Path) -> None:
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text("model: {}\n")
    teacher_config = tmp_path / "ckpts" / "teacher" / "config.json"
    teacher_config.parent.mkdir(parents=True)
    teacher_config.write_text("{}\n")
    config = _stage_config(tmp_path, config_path)
    manifest = StageManifest(stage="convert", config=config)
    manifest.complete(outputs={"teacher_dir": str(teacher_config.parent)})
    historical_payload = manifest.to_dict()
    historical_payload.pop("effective_config")
    historical_payload.pop("execution_record")
    manifest_path = tmp_path / "manifests" / "convert.json"
    manifest_path.parent.mkdir()
    manifest_path.write_text(json.dumps(historical_payload))

    _mark_completion(config, config_path, "convert")

    assert _completion_is_valid(config, config_path, "convert")


def test_resume_requires_execution_record_files(tmp_path: Path) -> None:
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

    (tmp_path / manifest.execution_record["resolved_config_path"]).unlink()
    assert not _completion_is_valid(config, config_path, "convert")
