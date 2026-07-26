# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve one fingerprinted checkpoint parent for every downstream search stage."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

__all__ = [
    "ScoringParent",
    "ensure_scoring_parent",
    "load_scoring_parent",
    "resolve_scoring_parent",
    "write_scoring_parent",
]


def _checkpoint_identity(path: Path) -> dict[str, Any]:
    # Keep the checkpoint identity implementation canonical while avoiding a
    # module-import cycle: distributed_eval.config itself consumes this parent.
    from .distributed_eval.config import checkpoint_identity

    return checkpoint_identity(path)


@dataclass(frozen=True)
class ScoringParent:
    path: Path
    role: str
    fingerprint: str
    sorted_teacher_path: Path
    sorted_teacher_fingerprint: str
    bypass_manifest_path: Path | None = None
    bypass_manifest_fingerprint: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in ("path", "sorted_teacher_path", "bypass_manifest_path"):
            if payload[key] is not None:
                payload[key] = str(payload[key])
        return payload


def _file_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _require_checkpoint(path: Path, *, label: str) -> Path:
    path = path.resolve()
    if not (path / "config.json").is_file():
        raise FileNotFoundError(f"{label} is not a usable checkpoint: {path}")
    return path


def resolve_scoring_parent(config: dict[str, Any]) -> ScoringParent:
    experiment = config.get("experiment") or {}
    puzzle_dir = Path(config.get("puzzle_dir") or experiment.get("dir"))
    sorted_teacher = _require_checkpoint(
        puzzle_dir / "ckpts" / "sorted_teacher", label="sorted teacher"
    )
    sorted_identity = _checkpoint_identity(sorted_teacher)
    bypass = config.get("bypass") or {}
    use_bypassed = bool(bypass.get("enabled", False)) and bool(
        bypass.get("use_nested_bypassed_checkpoint_for_scoring", False)
    )
    if not use_bypassed:
        return ScoringParent(
            path=sorted_teacher,
            role="sorted_teacher",
            fingerprint=sorted_identity["fingerprint"],
            sorted_teacher_path=sorted_teacher,
            sorted_teacher_fingerprint=sorted_identity["fingerprint"],
        )

    configured = bypass.get("scoring_checkpoint_dir")
    bypassed = _require_checkpoint(
        Path(configured) if configured else puzzle_dir / "ckpts" / "elastic_sorted_teacher",
        label="nested bypass scoring parent",
    )
    manifest_path = puzzle_dir / "manifests" / "bypass.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"nested bypass scoring parent has no successful stage manifest: {manifest_path}"
        )
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "success":
        raise RuntimeError(
            "nested bypass scoring parent manifest is not successful: "
            f"{manifest.get('status')!r}"
        )
    bypass_identity = _checkpoint_identity(bypassed)
    return ScoringParent(
        path=bypassed,
        role="nested_bypassed",
        fingerprint=bypass_identity["fingerprint"],
        sorted_teacher_path=sorted_teacher,
        sorted_teacher_fingerprint=sorted_identity["fingerprint"],
        bypass_manifest_path=manifest_path.resolve(),
        bypass_manifest_fingerprint=_file_fingerprint(manifest_path),
    )


def write_scoring_parent(parent: ScoringParent, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{uuid4().hex}")
    try:
        temporary.write_text(json.dumps(parent.to_dict(), indent=2, sort_keys=True) + "\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def load_scoring_parent(path: str | Path) -> ScoringParent:
    path = Path(path)
    payload = json.loads(path.read_text())
    parent = ScoringParent(
        path=Path(payload["path"]),
        role=str(payload["role"]),
        fingerprint=str(payload["fingerprint"]),
        sorted_teacher_path=Path(payload["sorted_teacher_path"]),
        sorted_teacher_fingerprint=str(payload["sorted_teacher_fingerprint"]),
        bypass_manifest_path=(
            Path(payload["bypass_manifest_path"])
            if payload.get("bypass_manifest_path")
            else None
        ),
        bypass_manifest_fingerprint=payload.get("bypass_manifest_fingerprint"),
    )
    current = _checkpoint_identity(_require_checkpoint(parent.path, label="scoring parent"))
    if current["fingerprint"] != parent.fingerprint:
        raise RuntimeError(
            "stale scoring parent artifact: checkpoint identity changed for "
            f"{parent.path}"
        )
    sorted_current = _checkpoint_identity(
        _require_checkpoint(parent.sorted_teacher_path, label="sorted teacher")
    )
    if sorted_current["fingerprint"] != parent.sorted_teacher_fingerprint:
        raise RuntimeError("stale scoring parent artifact: sorted teacher identity changed")
    if parent.bypass_manifest_path is not None:
        if not parent.bypass_manifest_path.is_file() or (
            _file_fingerprint(parent.bypass_manifest_path)
            != parent.bypass_manifest_fingerprint
        ):
            raise RuntimeError("stale scoring parent artifact: bypass manifest identity changed")
    return parent


def ensure_scoring_parent(
    config: dict[str, Any], *, refresh: bool = False
) -> ScoringParent:
    experiment = config.get("experiment") or {}
    puzzle_dir = Path(config.get("puzzle_dir") or experiment.get("dir"))
    artifact = puzzle_dir / "artifacts" / "scoring_parent.json"
    if artifact.is_file() and not refresh:
        return load_scoring_parent(artifact)
    parent = resolve_scoring_parent(config)
    write_scoring_parent(parent, artifact)
    return parent
