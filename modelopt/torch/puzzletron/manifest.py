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

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .execution_record import (
    _EXECUTION_RECORD_MANIFEST_VERSION,
    _EXECUTION_RECORD_SCHEMA,
    _file_evidence,
    _path_without_symlinks,
    _portable_relative_path,
    _read_file_bytes,
    _read_mapping,
    _sha256_bytes,
    validate_stage_execution_record,
)
from .identity import canonicalize, stable_hash

__all__ = [
    "StageManifest",
    "read_stage_manifest",
    "semantic_stage_config",
    "validate_stage_execution_record",
    "write_stage_execution_record",
    "write_stage_manifest",
]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return (json.dumps(canonicalize(payload), indent=2, sort_keys=True) + "\n").encode()


def _immutable_record_matches(path: Path, expected: bytes) -> bool:
    try:
        return _read_file_bytes(path, description="stage execution record path") == expected
    except ValueError:
        return False


def _publish_immutable_record(
    record_dir: Path,
    files: dict[str, bytes],
) -> None:
    record_dir = _path_without_symlinks(record_dir, description="stage execution record path")
    for name in files:
        _path_without_symlinks(record_dir / name, description="stage execution record path")
    record_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{record_dir.name}.", dir=record_dir.parent))
    try:
        for name, content in files.items():
            (temporary / name).write_bytes(content)
        try:
            temporary.rename(record_dir)
            return
        except OSError:
            if not record_dir.is_dir():
                raise
        mismatches = [
            name
            for name, content in files.items()
            if not _immutable_record_matches(record_dir / name, content)
        ]
        if mismatches:
            raise FileExistsError(
                f"immutable stage execution record already exists with different content: "
                f"{record_dir} ({', '.join(mismatches)})"
            )
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def _resolved_config_content(config: object) -> Any:
    """Remove path/launch-only runtime facts from resolved semantic identity."""

    if not isinstance(config, Mapping):
        return canonicalize(config)
    return canonicalize({key: value for key, value in config.items() if key != "_runtime"})


def write_stage_execution_record(
    manifest_path: str | Path,
    manifest_payload: dict[str, Any],
) -> dict[str, Any]:
    """Persist immutable resolved config and artifact metadata for one execution."""

    manifest_path = _path_without_symlinks(Path(manifest_path), description="stage manifest path")
    if manifest_path.parent.name != "manifests":
        raise ValueError(
            f"stage manifest must use the campaign manifests directory: {manifest_path}"
        )
    root = manifest_path.parent.parent
    stage = str(manifest_payload.get("stage") or "")
    if not stage or Path(stage).name != stage or stage in {".", ".."}:
        raise ValueError(f"invalid stage identifier for execution record: {stage!r}")

    authored_config = canonicalize(manifest_payload.get("config") or {})
    raw_effective_config = manifest_payload.get("effective_config")
    effective_config = canonicalize(
        authored_config if raw_effective_config is None else raw_effective_config
    )
    authored_config_identity = str(
        manifest_payload.get("config_identity")
        or stable_hash(authored_config, prefix=f"{stage}_cfg")
    )
    resolved_config_content = _resolved_config_content(
        semantic_stage_config(dict(effective_config), stage)
        if isinstance(effective_config, Mapping)
        else effective_config
    )
    resolved_config_identity = stable_hash(resolved_config_content, prefix=f"{stage}_resolved_cfg")
    implementation_provenance = canonicalize(
        manifest_payload.get("implementation_provenance") or {}
    )
    execution_identity = stable_hash(
        {
            "stage": stage,
            "status": manifest_payload.get("status"),
            "skip_reason": manifest_payload.get("skip_reason"),
            "started_at": manifest_payload.get("started_at"),
            "ended_at": manifest_payload.get("ended_at"),
            "authored_config_identity": authored_config_identity,
            "resolved_config_identity": resolved_config_identity,
            "semantic_config_identity": manifest_payload.get("semantic_config_identity"),
            "semantic_identity": manifest_payload.get("semantic_identity"),
            "capability_snapshot": manifest_payload.get("capability_snapshot"),
            "implementation_provenance": implementation_provenance,
        },
        prefix=f"{stage}_execution",
    )
    record_dir = root / "manifests" / "executions" / stage / execution_identity
    resolved_path = record_dir / "resolved_config.json"
    artifact_path = record_dir / "artifact_manifest.json"
    resolved_relative = _portable_relative_path(resolved_path, root)
    artifact_relative = _portable_relative_path(artifact_path, root)

    inputs = manifest_payload.get("inputs") or {}
    runtime = effective_config.get("_runtime") if isinstance(effective_config, dict) else {}
    runtime = runtime if isinstance(runtime, dict) else {}
    resolved_payload = {
        "schema": _EXECUTION_RECORD_SCHEMA,
        "schema_version": 1,
        "stage": stage,
        "status": manifest_payload.get("status"),
        "skip_reason": manifest_payload.get("skip_reason"),
        "started_at": manifest_payload.get("started_at"),
        "ended_at": manifest_payload.get("ended_at"),
        "execution_identity": execution_identity,
        "authored_config_identity": authored_config_identity,
        "resolved_config_identity": resolved_config_identity,
        "semantic_config": manifest_payload.get("semantic_config") or {},
        "semantic_config_identity": manifest_payload.get("semantic_config_identity"),
        "semantic_identity": manifest_payload.get("semantic_identity"),
        "provenance": {
            "authored_config_path": runtime.get("config_path"),
            "overrides": list(runtime.get("overrides") or ()),
            "implementation": implementation_provenance,
            "descriptor_resolution": inputs.get("descriptor_resolution"),
            "capability_snapshot": manifest_payload.get("capability_snapshot"),
        },
        "resolved_stage_config": resolved_config_content,
    }
    resolved_bytes = _json_bytes(resolved_payload)
    resolved_sha256 = _sha256_bytes(resolved_bytes)

    declared_outputs = canonicalize(manifest_payload.get("outputs") or {})
    immutable_evidence = {}
    if isinstance(declared_outputs, Mapping):
        for key, value in declared_outputs.items():
            if not isinstance(value, str):
                continue
            output_path = Path(value)
            if not output_path.is_absolute():
                output_path = root / output_path
            if output_path.is_file():
                safe_output_path = _path_without_symlinks(
                    output_path, description="stage output evidence path"
                )
                try:
                    evidence_path = _portable_relative_path(safe_output_path, root)
                except ValueError as exc:
                    raise ValueError(
                        f"stage output evidence file is outside the campaign root: {output_path}"
                    ) from exc
                immutable_evidence[str(key)] = {
                    "path": evidence_path,
                    **_file_evidence(output_path, description="stage output evidence file"),
                }
    artifact_payload = {
        "schema": _EXECUTION_RECORD_SCHEMA,
        "schema_version": 1,
        "stage": stage,
        "status": manifest_payload.get("status"),
        "skip_reason": manifest_payload.get("skip_reason"),
        "started_at": manifest_payload.get("started_at"),
        "ended_at": manifest_payload.get("ended_at"),
        "execution_identity": execution_identity,
        "resolved_config": {
            "path": resolved_relative,
            "identity": resolved_config_identity,
            "sha256": resolved_sha256,
        },
        "stage_manifest": {
            "path": _portable_relative_path(manifest_path, root),
            "semantic_identity": manifest_payload.get("semantic_identity"),
        },
        "artifact_contract": "stage-manifest-output-pointers/v1",
        "canonical_output_pointers": declared_outputs,
        "immutable_evidence": immutable_evidence,
    }
    artifact_identity = stable_hash(artifact_payload, prefix=f"{stage}_artifacts")
    artifact_payload["artifact_manifest_identity"] = artifact_identity
    artifact_bytes = _json_bytes(artifact_payload)
    artifact_sha256 = _sha256_bytes(artifact_bytes)
    _publish_immutable_record(
        record_dir,
        {
            resolved_path.name: resolved_bytes,
            artifact_path.name: artifact_bytes,
        },
    )
    return {
        "schema": _EXECUTION_RECORD_SCHEMA,
        "execution_identity": execution_identity,
        "resolved_config_path": resolved_relative,
        "resolved_config_identity": resolved_config_identity,
        "resolved_config_sha256": resolved_sha256,
        "artifact_manifest_path": artifact_relative,
        "artifact_manifest_identity": artifact_identity,
        "artifact_manifest_sha256": artifact_sha256,
    }


@dataclass
class StageManifest:
    """Durable metadata and semantic identity for one Puzzletron stage execution."""

    stage: str
    version: str = "1"
    status: str = "pending"
    inputs: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)
    capability_snapshot: dict[str, Any] | None = None
    semantic_config: dict[str, Any] | None = None
    implementation_provenance: dict[str, Any] = field(default_factory=dict)
    skip_reason: str | None = None
    effective_config: dict[str, Any] | None = None
    execution_record: dict[str, Any] | None = None
    stale_reason: str | None = None
    started_at: str = field(default_factory=_now_iso)
    ended_at: str | None = None

    @property
    def config_identity(self) -> str:
        return stable_hash(self.config, prefix=f"{self.stage}_cfg")

    @property
    def semantic_config_identity(self) -> str:
        """Return the identity of configuration relevant to this stage's result."""

        config = self.semantic_config
        if config is None:
            config = semantic_stage_config(self.config, self.stage)
        return stable_hash(config, prefix=f"{self.stage}_semantic_cfg")

    @property
    def semantic_identity(self) -> str:
        """Return the compatibility identity consumed by downstream resume checks."""

        payload = {
            "stage": self.stage,
            "semantic_config_identity": self.semantic_config_identity,
            "capability_snapshot": self.capability_snapshot,
        }
        return stable_hash(payload, prefix=f"{self.stage}_semantic")

    def complete(
        self,
        *,
        outputs: dict[str, Any] | None = None,
        status: str = "success",
        skip_reason: str | None = None,
    ) -> None:
        """Mark the stage complete with its validated outputs and final status."""

        if outputs is not None:
            self.outputs = outputs
        self.status = str(getattr(status, "value", status))
        self.skip_reason = (
            str(getattr(skip_reason, "value", skip_reason)) if skip_reason is not None else None
        )
        self.ended_at = _now_iso()

    def to_dict(self) -> dict[str, Any]:
        """Return the backward-compatible serialized manifest payload."""

        payload = {
            "stage": self.stage,
            "version": self.version,
            "status": self.status,
            "inputs": canonicalize(self.inputs),
            "outputs": canonicalize(self.outputs),
            "config": canonicalize(self.config),
            "config_identity": self.config_identity,
            "semantic_config": canonicalize(
                self.semantic_config
                if self.semantic_config is not None
                else semantic_stage_config(self.config, self.stage)
            ),
            "semantic_config_identity": self.semantic_config_identity,
            "semantic_identity": self.semantic_identity,
            "implementation_provenance": canonicalize(self.implementation_provenance),
            "capability_snapshot": canonicalize(self.capability_snapshot),
            "stale_reason": self.stale_reason,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
        }
        if self.skip_reason is not None:
            payload["skip_reason"] = self.skip_reason
        if self.execution_record is not None:
            payload["execution_record"] = canonicalize(self.execution_record)
        return payload


def write_stage_manifest(path: str | Path, manifest: StageManifest) -> None:
    """Atomically write a stage manifest from rank zero.

    Rank zero publishes the immutable execution record and assigns its reference
    to ``manifest.execution_record`` before writing the manifest. Other ranks
    return without writing and leave ``manifest.execution_record`` unchanged.
    """

    if os.environ.get("RANK") not in (None, "", "0"):
        return
    path = Path(path)
    manifest.version = _EXECUTION_RECORD_MANIFEST_VERSION
    execution_payload = manifest.to_dict()
    execution_payload["effective_config"] = canonicalize(manifest.effective_config)
    manifest.execution_record = write_stage_execution_record(path, execution_payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def read_stage_manifest(path: str | Path) -> dict[str, Any]:
    """Read a stage manifest while preserving compatibility with older schemas."""

    payload, _ = _read_mapping(Path(path), description="stage manifest")
    return payload


# Import after defining the manifest API because the stages package registers
# handlers that import StageManifest while its graph submodule is initialized.
from .stages.graph import semantic_stage_config as semantic_stage_config  # noqa: E402
