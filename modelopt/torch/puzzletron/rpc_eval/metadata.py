# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from ..identity import cache_key, canonicalize

__all__ = ["EvaluationCacheSlot", "EvaluationMetadataCache"]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _format_time(value: datetime) -> str:
    return value.isoformat()


def _parse_time(value: str | None) -> datetime | None:
    if value is None:
        return None
    return datetime.fromisoformat(value)


def _write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(canonicalize(data), indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


@dataclass(frozen=True, kw_only=True)
class EvaluationCacheSlot:
    """Metadata-only cache slot for teacher hidden states or prefix activations."""

    kind: str
    cache_id: str
    path: Path
    manifest: dict[str, Any]

    @property
    def manifest_path(self) -> Path:
        return self.path / "manifest.json"

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "cache_id": self.cache_id,
            "path": str(self.path),
            "manifest": canonicalize(self.manifest),
        }


class EvaluationMetadataCache:
    """Create manifest-backed cache directories for future tensor-backed reuse.

    These slots intentionally store metadata only. They give Stage 6/7 callers a
    stable identity and directory contract before the real tensor cache exists.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def prepare_teacher_hidden(
        self,
        inputs: Any,
        settings: Any | None = None,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> EvaluationCacheSlot:
        return self._prepare_slot(
            "teacher_hidden",
            inputs,
            settings or {},
            metadata=metadata,
            ttl_seconds=None,
        )

    def prepare_prefix(
        self,
        inputs: Any,
        settings: Any | None = None,
        *,
        metadata: dict[str, Any] | None = None,
        ttl_seconds: int | None = None,
    ) -> EvaluationCacheSlot:
        return self._prepare_slot(
            "prefix",
            inputs,
            settings or {},
            metadata=metadata,
            ttl_seconds=ttl_seconds,
        )

    def evict_expired_prefixes(self) -> list[str]:
        removed: list[str] = []
        now = _utc_now()
        prefix_root = self.root / "prefix"
        if not prefix_root.exists():
            return removed
        for manifest_path in prefix_root.glob("*/manifest.json"):
            manifest = json.loads(manifest_path.read_text())
            expires_at = _parse_time(manifest.get("expires_at"))
            if expires_at is None or expires_at > now:
                continue
            slot_dir = manifest_path.parent
            shutil.rmtree(slot_dir)
            removed.append(manifest["cache_id"])
        return removed

    def _prepare_slot(
        self,
        kind: str,
        inputs: Any,
        settings: Any,
        *,
        metadata: dict[str, Any] | None,
        ttl_seconds: int | None,
    ) -> EvaluationCacheSlot:
        slot_id = cache_key(kind, inputs, settings).value
        slot_dir = self.root / kind / slot_id
        manifest_path = slot_dir / "manifest.json"
        now = _utc_now()
        expires_at = (
            _format_time(now + timedelta(seconds=ttl_seconds))
            if ttl_seconds is not None
            else None
        )
        manifest = {
            "version": 1,
            "kind": kind,
            "cache_id": slot_id,
            "inputs": canonicalize(inputs),
            "settings": canonicalize(settings),
            "metadata": canonicalize(metadata or {}),
            "created_at": _format_time(now),
            "updated_at": _format_time(now),
            "expires_at": expires_at,
            "tensor_cache_ready": False,
        }
        if manifest_path.exists():
            previous = json.loads(manifest_path.read_text())
            manifest["created_at"] = previous.get("created_at", manifest["created_at"])
        _write_json_atomic(manifest_path, manifest)
        return EvaluationCacheSlot(kind=kind, cache_id=slot_id, path=slot_dir, manifest=manifest)
