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

"""Dependency-light validation for receipt-bound imported stage manifests."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any

__all__ = [
    "IMPORT_CAMPAIGN_MANIFEST",
    "canonical_receipt_identity",
    "imported_completion_payload",
    "imported_stage_manifest_is_complete",
]


IMPORT_CAMPAIGN_MANIFEST = Path("manifests/imports/campaign_artifacts.json")


def canonical_receipt_identity(receipt: Mapping[str, Any]) -> str:
    """Return the canonical identity of one version-2 artifact receipt."""

    payload = {key: value for key, value in receipt.items() if key != "receipt_identity"}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def imported_completion_payload(
    *,
    stage_id: str,
    target_config: str,
    receipt_identity: str,
    expected_semantic_config: Mapping[str, Any],
    semantic_identity: str,
    required_artifacts: Mapping[str, Any],
    stable_hash: Callable[..., str],
) -> dict[str, Any]:
    """Build the canonical version-3 completion marker for one imported stage."""

    relevant_stage_config_identity = stable_hash(
        expected_semantic_config,
        prefix=f"{stage_id}_resume_cfg",
    )
    identity_payload = {
        "completion_kind": "imported",
        "mode": stage_id,
        "width": None,
        "depth": None,
        "receipt_identity": receipt_identity,
        "relevant_stage_config_identity": relevant_stage_config_identity,
        "stage_manifest_semantic_identity": semantic_identity,
        "required_artifacts": dict(required_artifacts),
        "upstream_identities": {},
    }
    return {
        "version": 3,
        **identity_payload,
        "config": target_config,
        "implementation_provenance": {"imported": True},
        "completion_identity": stable_hash(identity_payload, prefix=f"{stage_id}_completion"),
    }


def _read_mapping(path: Path) -> Mapping[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, Mapping) else None


def _safe_artifact_path(root: Path, value: Any) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        return None
    artifact = root
    if artifact.is_symlink():
        return None
    for part in relative.parts:
        artifact /= part
        if artifact.is_symlink():
            return None
    return artifact


def _file_identity(path: Path) -> tuple[int, int, int, int, int]:
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


@lru_cache(maxsize=1024)
def _cached_file_digest(path: Path, identity: tuple[int, int, int, int, int]) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    if _file_identity(path) != identity:
        raise OSError("artifact changed while hashing")
    return digest.hexdigest()


def _file_record(root: Path, relative: str) -> dict[str, Any] | None:
    path = _safe_artifact_path(root, relative)
    if path is None:
        return None
    try:
        if not path.is_file():
            return None
        resolved = path.resolve(strict=True)
        identity = _file_identity(resolved)
        digest = _cached_file_digest(resolved, identity)
        if _file_identity(resolved) != identity:
            return None
        return {"path": relative, "size": identity[2], "sha256": digest}
    except OSError:
        return None


def _inventory_matches(root: Path, inventory: list[Any]) -> bool:
    seen: set[str] = set()
    for record in inventory:
        if not isinstance(record, Mapping) or set(record) != {"path", "size", "sha256"}:
            return False
        relative = record.get("path")
        if not isinstance(relative, str) or relative in seen:
            return False
        seen.add(relative)
        if _file_record(root, relative) != record:
            return False
    return bool(seen)


def _receipt_inventory(
    campaign: Mapping[str, Any],
    stage_id: str,
) -> tuple[Mapping[str, Any], ...] | None:
    """Return bound receipt records, an empty legacy marker, or invalid evidence."""

    receipt_version = campaign.get("receipt_version")
    receipt = campaign.get("receipt")
    if receipt_version is None and receipt is None:
        return ()
    if receipt_version != 2 or not isinstance(receipt, Mapping):
        return None
    if (
        receipt.get("version") != 2
        or receipt.get("state") != "complete"
        or receipt.get("campaign_root") != campaign.get("source_campaign")
        or receipt.get("receipt_identity") != campaign.get("receipt_identity")
        or canonical_receipt_identity(receipt) != campaign.get("receipt_identity")
    ):
        return None
    artifacts = receipt.get("artifacts")
    artifact = artifacts.get(stage_id) if isinstance(artifacts, Mapping) else None
    files = artifact.get("files") if isinstance(artifact, Mapping) else None
    if artifact is None or artifact.get("state") != "complete" or not isinstance(files, list):
        return None
    records = []
    for record in files:
        if (
            not isinstance(record, Mapping)
            or set(record) != {"path", "size", "sha256"}
            or not isinstance(record.get("path"), str)
            or not record["path"]
            or not isinstance(record.get("size"), int)
            or record["size"] < 0
            or not isinstance(record.get("sha256"), str)
        ):
            return None
        records.append(record)
    return tuple(records) if records else None


def imported_stage_manifest_is_complete(
    root: Path,
    stage_id: str,
    manifest: Mapping[str, Any],
    *,
    expected_semantic_config: Mapping[str, Any],
    stable_hash: Callable[..., str],
) -> bool:
    """Return whether an imported stage matches the atomic campaign import contract."""

    if (
        manifest.get("stage") != stage_id
        or manifest.get("status") != "imported"
        or manifest.get("semantic_config") != expected_semantic_config
    ):
        return False
    semantic_config_identity = stable_hash(
        expected_semantic_config,
        prefix=f"{stage_id}_semantic_cfg",
    )
    semantic_identity = stable_hash(
        {
            "stage": stage_id,
            "semantic_config_identity": semantic_config_identity,
            "capability_snapshot": manifest.get("capability_snapshot"),
        },
        prefix=f"{stage_id}_semantic",
    )
    if (
        manifest.get("semantic_config_identity") != semantic_config_identity
        or not isinstance(manifest.get("semantic_identity"), str)
        or manifest.get("semantic_identity") != semantic_identity
    ):
        return False
    campaign_path = _safe_artifact_path(root, str(IMPORT_CAMPAIGN_MANIFEST))
    campaign = _read_mapping(campaign_path) if campaign_path is not None else None
    if campaign is None or campaign.get("version") != 1 or campaign.get("status") != "complete":
        return False
    bundles = campaign.get("bundles")
    if (
        not isinstance(bundles, list)
        or not all(isinstance(bundle, str) for bundle in bundles)
        or stage_id not in bundles
        or len(bundles) != len(set(bundles))
        or not isinstance(campaign.get("target_config"), str)
        or not campaign["target_config"]
        or not isinstance(campaign.get("target_config_identity"), str)
        or not campaign["target_config_identity"]
    ):
        return False
    source_campaign = campaign.get("source_campaign")
    receipt_identity = campaign.get("receipt_identity")
    inputs = manifest.get("inputs")
    if (
        not isinstance(source_campaign, str)
        or not source_campaign
        or not isinstance(receipt_identity, str)
        or not receipt_identity.startswith("sha256:")
        or not isinstance(inputs, Mapping)
        or manifest.get("source_campaign") != source_campaign
        or inputs.get("source_campaign") != source_campaign
        or manifest.get("receipt_identity") != receipt_identity
        or inputs.get("receipt_identity") != receipt_identity
    ):
        return False
    inventory = manifest.get("output_inventory")
    outputs = manifest.get("outputs")
    imported_files = outputs.get("imported_files") if isinstance(outputs, Mapping) else None
    if (
        not isinstance(inventory, list)
        or not _inventory_matches(root, inventory)
        or imported_files != [record.get("path") for record in inventory]
    ):
        return False
    receipt_inventory = _receipt_inventory(campaign, stage_id)
    if receipt_inventory is None:
        return False
    if receipt_inventory and inventory != [dict(record) for record in receipt_inventory]:
        return False
    manifest_relative = f"manifests/{stage_id}.json"
    manifest_record = _file_record(root, manifest_relative)
    if manifest_record is None:
        return False
    expected_required_artifacts = {
        record["path"]: [dict(record)] for record in inventory if isinstance(record, Mapping)
    }
    expected_required_artifacts[manifest_relative] = [manifest_record]
    marker_relative = f"manifests/completions/{stage_id}.json"
    marker_path = _safe_artifact_path(root, marker_relative)
    marker = _read_mapping(marker_path) if marker_path is not None else None
    expected_marker = imported_completion_payload(
        stage_id=stage_id,
        target_config=campaign["target_config"],
        receipt_identity=receipt_identity,
        expected_semantic_config=expected_semantic_config,
        semantic_identity=semantic_identity,
        required_artifacts=expected_required_artifacts,
        stable_hash=stable_hash,
    )
    if marker is None or any(marker.get(key) != value for key, value in expected_marker.items()):
        return False
    return True
