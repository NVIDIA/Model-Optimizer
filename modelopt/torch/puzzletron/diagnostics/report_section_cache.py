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

"""Disposable section snapshots for incremental Puzzletron campaign reports."""

import hashlib
import json
import os
import tempfile
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


@dataclass(frozen=True)
class SourceIdentity:
    """Stable identity for one canonical or partial report source."""

    path: str
    size: int
    mtime_ns: int
    content_sha256: str | None


@dataclass(frozen=True)
class SectionSnapshot:
    """Cached, validated data and markup for one report section."""

    section_id: str
    schema_version: int
    extractor_version: int
    campaign_identity: str
    input_digest: str
    sources: tuple[SourceIdentity, ...]
    data: dict[str, Any]
    body_html: str
    validation: dict[str, Any]
    telemetry: dict[str, Any]


@dataclass(frozen=True)
class SectionBuildResult:
    """Result of resolving one report section through the cache."""

    snapshot: SectionSnapshot
    snapshot_path: Path
    cache_hit: bool


def _jsonable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_jsonable(item) for item in value), key=_canonical_json)
    if isinstance(value, Path):
        return str(value)
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _jsonable(value),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def stable_digest(value: Any) -> str:
    """Return a deterministic SHA-256 digest for JSON-compatible state."""

    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint_paths(
    root: Path,
    paths: Iterable[Path],
    *,
    hash_contents: bool,
) -> tuple[SourceIdentity, ...]:
    """Inventory source files without rereading content unless requested."""

    root = root.resolve()
    identities = []
    for path in paths:
        resolved = path.resolve()
        stat = resolved.stat()
        try:
            display_path = str(resolved.relative_to(root))
        except ValueError:
            display_path = str(resolved)
        identities.append(
            SourceIdentity(
                path=display_path,
                size=stat.st_size,
                mtime_ns=stat.st_mtime_ns,
                content_sha256=_file_digest(resolved) if hash_contents else None,
            )
        )
    return tuple(sorted(identities, key=lambda identity: identity.path))


def _write_json_temp(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        json.dump(_jsonable(payload), stream, ensure_ascii=False, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
        return Path(stream.name)


def _write_text_temp(path: Path, value: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
        return Path(stream.name)


def _backup_path(path: Path) -> Path | None:
    if not path.exists():
        return None
    backup = path.with_name(f".{path.name}.{uuid4().hex}.backup")
    os.link(path, backup)
    return backup


def _restore_path(path: Path, backup: Path | None) -> None:
    if backup is None:
        path.unlink(missing_ok=True)
    elif backup.exists():
        os.replace(backup, path)


def publish_report_transaction(
    *,
    html_path: Path,
    html: str,
    manifest_path: Path,
    manifest: Mapping[str, Any],
    verifier: Callable[[Path], None],
) -> None:
    """Verify and atomically publish a matching report HTML/manifest pair."""

    html_temp = _write_text_temp(html_path, html)
    manifest_temp = _write_json_temp(manifest_path, manifest)
    html_backup = None
    manifest_backup = None
    try:
        verifier(html_temp)
        html_backup = _backup_path(html_path)
        manifest_backup = _backup_path(manifest_path)
        try:
            os.replace(html_temp, html_path)
            os.replace(manifest_temp, manifest_path)
        except BaseException:
            _restore_path(html_path, html_backup)
            html_backup = None
            _restore_path(manifest_path, manifest_backup)
            manifest_backup = None
            raise
    finally:
        html_temp.unlink(missing_ok=True)
        manifest_temp.unlink(missing_ok=True)
        if html_backup is not None:
            html_backup.unlink(missing_ok=True)
        if manifest_backup is not None:
            manifest_backup.unlink(missing_ok=True)


class ReportSectionCache:
    """Resolve report sections from versioned content-addressed snapshots."""

    def __init__(self, report_dir: Path, *, campaign_identity: str) -> None:
        self.report_dir = Path(report_dir)
        self.campaign_identity = campaign_identity
        self.cache_dir = self.report_dir / "section_cache"
        self.manifest_path = self.report_dir / "report_manifest.json"

    @staticmethod
    def _snapshot_from_payload(payload: Mapping[str, Any]) -> SectionSnapshot:
        return SectionSnapshot(
            section_id=str(payload["section_id"]),
            schema_version=int(payload["schema_version"]),
            extractor_version=int(payload["extractor_version"]),
            campaign_identity=str(payload["campaign_identity"]),
            input_digest=str(payload["input_digest"]),
            sources=tuple(SourceIdentity(**source) for source in payload["sources"]),
            data=dict(payload["data"]),
            body_html=str(payload["body_html"]),
            validation=dict(payload["validation"]),
            telemetry=dict(payload["telemetry"]),
        )

    @staticmethod
    def _snapshot_payload(snapshot: SectionSnapshot) -> dict[str, Any]:
        return asdict(snapshot)

    def _load_snapshot(
        self,
        path: Path,
        *,
        section_id: str,
        schema_version: int,
        extractor_version: int,
        input_digest: str,
        sources: tuple[SourceIdentity, ...],
    ) -> SectionSnapshot | None:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            snapshot = self._snapshot_from_payload(payload)
        except (OSError, ValueError, KeyError, TypeError):
            return None
        expected = (
            snapshot.section_id == section_id
            and snapshot.schema_version == schema_version
            and snapshot.extractor_version == extractor_version
            and snapshot.campaign_identity == self.campaign_identity
            and snapshot.input_digest == input_digest
            and snapshot.sources == sources
            and isinstance(snapshot.data, dict)
            and isinstance(snapshot.validation, dict)
            and isinstance(snapshot.telemetry, dict)
        )
        return snapshot if expected else None

    def load_or_build(
        self,
        *,
        section_id: str,
        schema_version: int,
        extractor_version: int,
        sources: tuple[SourceIdentity, ...],
        config_identity: str,
        dependency_identities: Mapping[str, str],
        builder: Callable[[], tuple[dict[str, Any], str, dict[str, Any]]],
        force: bool = False,
    ) -> SectionBuildResult:
        """Load a matching section snapshot or build and publish a new one."""

        identity = {
            "section_id": section_id,
            "schema_version": schema_version,
            "extractor_version": extractor_version,
            "campaign_identity": self.campaign_identity,
            "sources": sources,
            "config_identity": config_identity,
            "dependency_identities": dependency_identities,
        }
        input_digest = stable_digest(identity)
        snapshot_path = self.cache_dir / section_id / f"{input_digest}.json"
        if not force and snapshot_path.is_file():
            snapshot = self._load_snapshot(
                snapshot_path,
                section_id=section_id,
                schema_version=schema_version,
                extractor_version=extractor_version,
                input_digest=input_digest,
                sources=sources,
            )
            if snapshot is not None:
                return SectionBuildResult(snapshot, snapshot_path, cache_hit=True)

        started = time.monotonic()
        data, body_html, validation = builder()
        if not isinstance(data, dict):
            raise TypeError("section builder data must be a dict")
        if not isinstance(body_html, str):
            raise TypeError("section builder body_html must be a string")
        if not isinstance(validation, dict):
            raise TypeError("section builder validation must be a dict")
        snapshot = SectionSnapshot(
            section_id=section_id,
            schema_version=schema_version,
            extractor_version=extractor_version,
            campaign_identity=self.campaign_identity,
            input_digest=input_digest,
            sources=sources,
            data=data,
            body_html=body_html,
            validation=validation,
            telemetry={
                "built_at": datetime.now(timezone.utc).isoformat(),
                "elapsed_seconds": time.monotonic() - started,
                "source_files": len(sources),
                "source_bytes": sum(source.size for source in sources),
            },
        )
        temp = _write_json_temp(snapshot_path, self._snapshot_payload(snapshot))
        try:
            os.replace(temp, snapshot_path)
        finally:
            temp.unlink(missing_ok=True)
        return SectionBuildResult(snapshot, snapshot_path, cache_hit=False)

    def publish_manifest(self, payload: Mapping[str, Any]) -> Path:
        """Atomically publish a report cache manifest."""

        temp = _write_json_temp(self.manifest_path, payload)
        try:
            os.replace(temp, self.manifest_path)
        finally:
            temp.unlink(missing_ok=True)
        return self.manifest_path
