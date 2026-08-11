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

"""Dependency-light validation for immutable Puzzletron stage execution records."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path, PurePath
from typing import Any

from .identity import stable_hash

__all__ = ["stage_manifest_uses_execution_record", "validate_stage_execution_record"]

_EXECUTION_RECORD_SCHEMA = "puzzletron.stage-execution/v1"
_EXECUTION_RECORD_MANIFEST_VERSION = "2"


def _safe_relative_path(raw_path: object, *, description: str) -> Path:
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(f"invalid {description}: {raw_path!r}")
    path = Path(raw_path)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe {description}: {raw_path!r}")
    return path


def _path_without_symlinks(path: Path, *, description: str) -> Path:
    absolute = path.expanduser().absolute()
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        if current.is_symlink():
            raise ValueError(f"{description} is symlinked: {current}")
    return absolute


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _portable_relative_path(path: PurePath, root: PurePath) -> str:
    """Serialize a relative path with platform-neutral separators."""

    return path.relative_to(root).as_posix()


def _read_mapping(path: Path, *, description: str) -> tuple[dict[str, Any], bytes]:
    try:
        content = path.read_bytes()
        payload = json.loads(content)
    except (OSError, ValueError) as exc:
        raise ValueError(f"invalid {description}: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"invalid {description}: {path}")
    return payload, content


def stage_manifest_uses_execution_record(manifest: Mapping[str, Any]) -> bool:
    """Return whether a manifest opts into immutable execution-record validation."""

    return (
        manifest.get("version") == _EXECUTION_RECORD_MANIFEST_VERSION
        or "execution_record" in manifest
    )


def validate_stage_execution_record(
    manifest_path: str | Path,
    *,
    expected_stage: str | None = None,
) -> tuple[str, ...]:
    """Validate the canonical pointer and every immutable record it references.

    Historical stage manifests without ``execution_record`` remain readable. Any
    manifest that opts into the execution-record schema is checked fail closed.
    """

    path = _path_without_symlinks(Path(manifest_path), description="stage manifest path")
    if path.parent.name != "manifests":
        raise ValueError(f"stage manifest is outside the expected layout: {path}")
    root = path.parent.parent
    pointer, _ = _read_mapping(path, description="stage manifest")
    stage = pointer.get("stage")
    if not isinstance(stage, str) or Path(stage).name != stage or stage in {".", ".."}:
        raise ValueError(f"invalid stage identifier in stage manifest: {stage!r}")
    if expected_stage is not None and stage != expected_stage:
        raise ValueError(
            f"stage manifest identity mismatch: expected {expected_stage!r}, found {stage!r}"
        )

    pointer_config = pointer.get("config")
    if not isinstance(pointer_config, Mapping):
        raise ValueError("stage manifest config must be a mapping")
    expected_config_identity = stable_hash(pointer_config, prefix=f"{stage}_cfg")
    if pointer.get("config_identity") != expected_config_identity:
        raise ValueError("stage manifest config identity mismatch")

    pointer_semantic_config = pointer.get("semantic_config")
    if not isinstance(pointer_semantic_config, Mapping):
        raise ValueError("stage manifest semantic config must be a mapping")
    expected_semantic_config_identity = stable_hash(
        pointer_semantic_config, prefix=f"{stage}_semantic_cfg"
    )
    if pointer.get("semantic_config_identity") != expected_semantic_config_identity:
        raise ValueError("stage manifest semantic config identity mismatch")
    expected_semantic_identity = stable_hash(
        {
            "stage": stage,
            "semantic_config_identity": expected_semantic_config_identity,
            "capability_snapshot": pointer.get("capability_snapshot"),
        },
        prefix=f"{stage}_semantic",
    )
    if pointer.get("semantic_identity") != expected_semantic_identity:
        raise ValueError("stage manifest semantic identity mismatch")

    if "execution_record" not in pointer:
        if stage_manifest_uses_execution_record(pointer):
            raise ValueError("stage manifest requires an execution record")
        return ()
    record = pointer.get("execution_record")
    if not isinstance(record, Mapping) or record.get("schema") != _EXECUTION_RECORD_SCHEMA:
        raise ValueError(f"invalid stage execution record: {record!r}")
    execution_identity = record.get("execution_identity")
    if not isinstance(execution_identity, str) or not execution_identity:
        raise ValueError("invalid stage execution identity")

    resolved_relative = _safe_relative_path(
        record.get("resolved_config_path"), description="stage execution record path"
    )
    artifact_relative = _safe_relative_path(
        record.get("artifact_manifest_path"), description="stage execution record path"
    )
    expected_parent = Path("manifests") / "executions" / stage / execution_identity
    for relative, filename in (
        (resolved_relative, "resolved_config.json"),
        (artifact_relative, "artifact_manifest.json"),
    ):
        if relative.parent != expected_parent or relative.name != filename:
            raise ValueError(f"stage execution record path does not match identity: {relative}")

    resolved_path = _path_without_symlinks(
        root / resolved_relative, description="stage execution record path"
    )
    artifact_path = _path_without_symlinks(
        root / artifact_relative, description="stage execution record path"
    )
    resolved, resolved_bytes = _read_mapping(
        resolved_path, description="resolved stage configuration record"
    )
    artifact, artifact_bytes = _read_mapping(artifact_path, description="stage artifact record")
    resolved_sha256 = _sha256_bytes(resolved_bytes)
    if record.get("resolved_config_sha256") != resolved_sha256:
        raise ValueError("resolved stage configuration SHA256 mismatch")
    if record.get("artifact_manifest_sha256") != _sha256_bytes(artifact_bytes):
        raise ValueError("stage artifact record SHA256 mismatch")

    for label, payload in (("resolved", resolved), ("artifact", artifact)):
        if payload.get("schema") != _EXECUTION_RECORD_SCHEMA or payload.get("schema_version") != 1:
            raise ValueError(f"invalid {label} stage execution record schema")
        if payload.get("stage") != stage or payload.get("execution_identity") != execution_identity:
            raise ValueError(f"{label} stage execution identity mismatch")
        for key in ("status", "skip_reason", "started_at", "ended_at"):
            if payload.get(key) != pointer.get(key):
                raise ValueError(f"{label} stage execution {key} mismatch")

    resolved_identity = stable_hash(
        resolved.get("resolved_stage_config"), prefix=f"{stage}_resolved_cfg"
    )
    if (
        resolved.get("resolved_config_identity") != resolved_identity
        or record.get("resolved_config_identity") != resolved_identity
    ):
        raise ValueError("resolved stage configuration identity mismatch")
    for key in (
        "authored_config_identity",
        "semantic_config_identity",
        "semantic_identity",
    ):
        pointer_key = "config_identity" if key == "authored_config_identity" else key
        if resolved.get(key) != pointer.get(pointer_key):
            raise ValueError(f"resolved stage {key} mismatch")

    resolved_provenance = resolved.get("provenance") or {}
    if not isinstance(resolved_provenance, Mapping):
        raise ValueError("resolved stage provenance must be a mapping")
    implementation = resolved_provenance.get("implementation") or {}
    pointer_implementation = pointer.get("implementation_provenance") or {}
    if implementation != pointer_implementation:
        raise ValueError("resolved implementation provenance mismatch")
    pointer_inputs = pointer.get("inputs") or {}
    if not isinstance(pointer_inputs, Mapping):
        raise ValueError("stage manifest inputs must be a mapping")
    if resolved.get("semantic_config") != pointer_semantic_config:
        raise ValueError("resolved stage semantic config mismatch")
    if resolved_provenance.get("descriptor_resolution") != pointer_inputs.get(
        "descriptor_resolution"
    ):
        raise ValueError("resolved descriptor input provenance mismatch")
    if resolved_provenance.get("capability_snapshot") != pointer.get("capability_snapshot"):
        raise ValueError("resolved capability provenance mismatch")

    expected_execution_identity = stable_hash(
        {
            "stage": stage,
            "status": resolved.get("status"),
            "skip_reason": resolved.get("skip_reason"),
            "started_at": resolved.get("started_at"),
            "ended_at": resolved.get("ended_at"),
            "authored_config_identity": resolved.get("authored_config_identity"),
            "resolved_config_identity": resolved_identity,
            "semantic_config_identity": resolved.get("semantic_config_identity"),
            "semantic_identity": resolved.get("semantic_identity"),
            "capability_snapshot": resolved_provenance.get("capability_snapshot"),
            "implementation_provenance": implementation,
        },
        prefix=f"{stage}_execution",
    )
    if expected_execution_identity != execution_identity:
        raise ValueError("stage execution identity does not match record content")

    artifact_identity_payload = dict(artifact)
    artifact_identity = artifact_identity_payload.pop("artifact_manifest_identity", None)
    expected_artifact_identity = stable_hash(artifact_identity_payload, prefix=f"{stage}_artifacts")
    if (
        artifact_identity != expected_artifact_identity
        or record.get("artifact_manifest_identity") != expected_artifact_identity
    ):
        raise ValueError("stage artifact record identity mismatch")
    resolved_ref = artifact.get("resolved_config")
    if not isinstance(resolved_ref, Mapping) or resolved_ref != {
        "path": resolved_relative.as_posix(),
        "identity": resolved_identity,
        "sha256": resolved_sha256,
    }:
        raise ValueError("stage artifact resolved-configuration reference mismatch")
    stage_ref = artifact.get("stage_manifest")
    if not isinstance(stage_ref, Mapping) or stage_ref.get("path") != _portable_relative_path(
        path, root
    ):
        raise ValueError("stage artifact canonical-manifest reference mismatch")
    if stage_ref.get("semantic_identity") != pointer.get("semantic_identity"):
        raise ValueError("stage artifact semantic identity mismatch")
    if artifact.get("canonical_output_pointers") != pointer.get("outputs"):
        raise ValueError("stage artifact canonical output pointers mismatch")
    return resolved_relative.as_posix(), artifact_relative.as_posix()
