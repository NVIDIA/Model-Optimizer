#!/usr/bin/env python3
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

"""Content-addressed completion markers for the canonical acceptance driver."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from modelopt.torch.puzzletron.identity import stable_hash

if TYPE_CHECKING:
    from collections.abc import Iterable

ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOTS = (
    ROOT / "modelopt/torch/puzzletron",
    ROOT / "examples/puzzletron",
)

__all__ = [
    "CompletionCheck",
    "build_payload",
    "check_marker",
    "check_marker_details",
    "marker_matches",
    "marker_path",
    "source_identity",
    "write_marker",
]


@dataclass(frozen=True)
class CompletionCheck:
    """Structured result of validating a completion marker."""

    valid: bool
    validation_mode: str
    stale_reasons: tuple[str, ...] = ()


class _UnverifiableMarkerError(ValueError):
    def __init__(self, message: str, *, stale_complete_v3: bool = False) -> None:
        super().__init__(message)
        self.stale_complete_v3 = stale_complete_v3


class _SelectedUpstreamUnverifiable(ValueError):
    def __init__(self, stage: str, *, stale_complete_v3: bool = False) -> None:
        super().__init__(f"selected upstream unverifiable: {stage}")
        self.stage = stage
        self.stale_complete_v3 = stale_complete_v3


def _update_file(digest, path: Path) -> None:
    if not path.is_file():
        digest.update(f"missing:{path}\n".encode())
        return
    digest.update(str(path).encode())
    digest.update(b"\0")
    digest.update(path.read_bytes())
    digest.update(b"\0")


def _update_repository(digest, root: Path) -> None:
    root = Path(root).resolve()
    digest.update(str(root).encode())
    if not root.is_dir():
        digest.update(b"\0missing\0")
        return
    for command in (
        ("git", "-C", str(root), "rev-parse", "HEAD"),
        ("git", "-C", str(root), "status", "--porcelain=v1", "--untracked-files=all"),
        ("git", "-C", str(root), "diff", "--binary", "HEAD", "--"),
    ):
        result = subprocess.run(command, capture_output=True, check=False)
        digest.update(b"\0")
        digest.update(result.stdout)
        digest.update(result.stderr)


def source_identity(
    config: Path,
    *,
    source_roots: Iterable[Path] = SOURCE_ROOTS,
    repository_roots: Iterable[Path] = (),
    extra_files: Iterable[Path] = (),
) -> str:
    """Return broad implementation provenance without defining semantic freshness."""

    digest = hashlib.sha256()
    suffixes = {".py", ".yaml", ".yml", ".sh", ".json"}
    for source_root in source_roots:
        for path in sorted(source_root.rglob("*")):
            if path.is_file() and path.suffix in suffixes and "__pycache__" not in path.parts:
                _update_file(digest, path)
    _update_file(digest, config)
    for path in extra_files:
        _update_file(digest, path)
    for root in repository_roots:
        _update_repository(digest, root)
    return digest.hexdigest()


def marker_path(root: Path, mode: str, width: str | None, depth: str | None) -> Path:
    """Return the deterministic completion-marker path for one stage or scenario."""

    suffix = f"{mode}{'_w' + width if width else ''}{'_d' + depth if depth else ''}.json"
    return root / "manifests" / "completions" / suffix


def _completion_path_without_symlinks(path: Path) -> Path:
    absolute = Path(path).expanduser().absolute()
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        if current.is_symlink():
            raise _UnverifiableMarkerError(f"completion marker path is symlinked: {current}")
    return absolute


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_record(root: Path, path: Path) -> dict:
    stat = path.stat()
    try:
        recorded_path = str(path.relative_to(root))
    except ValueError:
        # Exact absolute required patterns bind configured artifacts outside the
        # campaign root. External globs are deliberately unsupported.
        recorded_path = str(path)
    record = {
        "path": recorded_path,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    # Completion manifests/configs are hashed exactly.  Large checkpoint shards
    # use size+mtime so resume checks do not reread hundreds of GB merely to decide
    # whether a stage is complete; deletion and normal mutations are still detected.
    if stat.st_size <= 16 * 1024 * 1024:
        record["sha256"] = _sha256(path)
    return record


def _immutable_artifact_record(root: Path, path: Path) -> dict:
    return {
        "path": str(path.relative_to(root)),
        "size": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _required_artifacts(root: Path, patterns: Iterable[str]) -> dict[str, list[dict]]:
    result: dict[str, list[dict]] = {}
    for pattern in patterns:
        configured = Path(pattern).expanduser()
        if configured.is_absolute():
            matches = [configured] if configured.is_file() else []
        else:
            matches = sorted(path for path in root.glob(pattern) if path.is_file())
        if not matches:
            raise FileNotFoundError(f"required artifact pattern has no matches: {pattern}")
        result[str(pattern)] = [_artifact_record(root, path) for path in matches]
    return result


def _completion_root(path: Path) -> Path:
    path = _completion_path_without_symlinks(path)
    if path.parent.name != "completions" or path.parent.parent.name != "manifests":
        raise ValueError(f"completion marker is outside the expected manifest layout: {path}")
    return path.parents[2]


def _current_imported_marker_identity(path: Path, payload: dict[str, Any]) -> str:
    """Recompute one imported-v3 identity from immutable current destination bytes."""

    required_fields = {
        "completion_identity",
        "mode",
        "receipt_identity",
        "relevant_stage_config_identity",
        "stage_manifest_semantic_identity",
        "required_artifacts",
        "upstream_identities",
    }
    if payload.get("version") != 3 or payload.get("completion_kind") != "imported":
        raise _UnverifiableMarkerError(f"completion marker is not imported v3 evidence: {path}")
    if not required_fields.issubset(payload):
        raise _UnverifiableMarkerError(f"completion marker has incomplete imported evidence: {path}")
    root = _completion_root(path)
    mode = payload["mode"]
    recorded_artifacts = payload["required_artifacts"]
    if not isinstance(mode, str) or not isinstance(recorded_artifacts, dict):
        raise _UnverifiableMarkerError(f"completion marker has invalid imported evidence: {path}")

    current_artifacts = {}
    for relative_text in sorted(recorded_artifacts):
        relative = Path(relative_text)
        if relative.is_absolute() or ".." in relative.parts:
            raise _UnverifiableMarkerError(
                f"completion marker has unsafe imported evidence: {path}"
            )
        candidate = root
        for part in relative.parts:
            candidate /= part
            if candidate.is_symlink():
                raise _UnverifiableMarkerError(
                    f"completion marker imported evidence is symlinked: {relative_text}",
                    stale_complete_v3=True,
                )
        if not candidate.is_file():
            raise _UnverifiableMarkerError(
                f"completion marker imported evidence is missing: {relative_text}",
                stale_complete_v3=True,
            )
        current_artifacts[relative_text] = [_immutable_artifact_record(root, candidate)]
    if current_artifacts != recorded_artifacts:
        raise _UnverifiableMarkerError(
            f"completion marker imported output identity changed: {path}",
            stale_complete_v3=True,
        )

    current_manifest_identity = _manifest_semantic_identity(root, mode)
    if current_manifest_identity != payload["stage_manifest_semantic_identity"]:
        raise _UnverifiableMarkerError(
            f"completion marker imported semantic identity changed: {path}",
            stale_complete_v3=True,
        )
    current_identity = stable_hash(
        {
            "completion_kind": "imported",
            "mode": mode,
            "width": payload.get("width"),
            "depth": payload.get("depth"),
            "receipt_identity": payload["receipt_identity"],
            "relevant_stage_config_identity": payload["relevant_stage_config_identity"],
            "stage_manifest_semantic_identity": current_manifest_identity,
            "required_artifacts": current_artifacts,
            "upstream_identities": payload["upstream_identities"],
        },
        prefix=f"{mode}_completion",
    )
    if current_identity != payload["completion_identity"]:
        raise _UnverifiableMarkerError(
            f"completion marker imported identity does not match current evidence: {path}",
            stale_complete_v3=True,
        )
    return str(current_identity)


def _current_marker_identity(path: Path, active: frozenset[Path] = frozenset()) -> str:
    """Recompute identity from a complete v3 marker and its current recursive evidence."""

    path = _completion_path_without_symlinks(path)
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise _UnverifiableMarkerError(f"unreadable completion marker: {path}") from error
    if payload.get("completion_kind") == "imported":
        return _current_imported_marker_identity(path, payload)
    provenance_fields = {
        "completion_identity",
        "mode",
        "width",
        "depth",
        "relevant_stage_config_identity",
        "stage_manifest_semantic_identity",
        "required_artifacts",
        "upstream_identities",
    }
    if payload.get("version") != 3 or not provenance_fields.issubset(payload):
        raise _UnverifiableMarkerError(
            f"completion marker is not complete semantic v3 evidence: {path}"
        )
    identity_fields = (
        payload["completion_identity"],
        payload["mode"],
        payload["relevant_stage_config_identity"],
        payload["stage_manifest_semantic_identity"],
    )
    if any(not isinstance(value, str) or not value for value in identity_fields):
        raise _UnverifiableMarkerError(f"completion marker has invalid v3 evidence: {path}")
    if path in active:
        raise _UnverifiableMarkerError(
            f"upstream completion cycle: {path}", stale_complete_v3=True
        )

    root = _completion_root(path)
    mode = str(payload["mode"])
    recorded_artifacts = payload["required_artifacts"]
    recorded_upstream = payload["upstream_identities"]
    if not isinstance(recorded_artifacts, dict) or not isinstance(recorded_upstream, dict):
        raise _UnverifiableMarkerError(f"completion marker has invalid v3 evidence: {path}")
    try:
        current_artifacts: Any = _required_artifacts(root, tuple(recorded_artifacts))
    except (FileNotFoundError, OSError) as error:
        raise _UnverifiableMarkerError(
            f"completion marker required evidence is missing: {path}",
            stale_complete_v3=True,
        ) from error
    current_manifest_identity = _manifest_semantic_identity(root, mode)
    if current_manifest_identity is None:
        raise _UnverifiableMarkerError(
            f"completion marker stage manifest is missing: {path}",
            stale_complete_v3=True,
        )

    next_active = active | {path}
    current_upstream = {}
    for stage in sorted(recorded_upstream):
        upstream_path = marker_path(root, stage, None, None)
        if upstream_path.is_file():
            current_upstream[stage] = _current_marker_identity(upstream_path, next_active)
        else:
            raise _UnverifiableMarkerError(
                f"completion marker upstream evidence is missing: {stage}",
                stale_complete_v3=True,
            )
    current_identity = stable_hash(
        {
            "mode": mode,
            "width": payload.get("width"),
            "depth": payload.get("depth"),
            "relevant_stage_config_identity": payload["relevant_stage_config_identity"],
            "stage_manifest_semantic_identity": current_manifest_identity,
            "required_artifacts": current_artifacts,
            "upstream_identities": current_upstream,
        },
        prefix=f"{mode}_completion",
    )
    if current_identity != payload["completion_identity"]:
        raise _UnverifiableMarkerError(
            f"completion marker identity does not match current evidence: {path}",
            stale_complete_v3=True,
        )
    return str(payload["completion_identity"])


def _named_upstream_markers(
    paths: Iterable[Path] | dict[str, Path],
) -> tuple[tuple[str, Path], ...]:
    if isinstance(paths, dict):
        return tuple(
            (str(name), _completion_path_without_symlinks(Path(path)))
            for name, path in sorted(paths.items())
        )
    return tuple(
        (Path(path).stem, _completion_path_without_symlinks(Path(path))) for path in paths
    )


def _upstream_identities(paths: Iterable[Path] | dict[str, Path]) -> dict[str, str]:
    identities: dict[str, str] = {}
    for name, path in _named_upstream_markers(paths):
        if not path.is_file():
            raise FileNotFoundError(f"required upstream marker is missing: {name}")
        try:
            identities[name] = _current_marker_identity(path)
        except _UnverifiableMarkerError as error:
            raise _SelectedUpstreamUnverifiable(
                name, stale_complete_v3=error.stale_complete_v3
            ) from error
        except (OSError, ValueError) as error:
            raise _SelectedUpstreamUnverifiable(name) from error
    return identities


def _legacy_upstream_identities(paths: Iterable[Path] | dict[str, Path]) -> dict[str, str]:
    identities: dict[str, str] = {}
    for _, path in _named_upstream_markers(paths):
        if not path.is_file():
            raise FileNotFoundError(f"required upstream marker is missing: {path}")
        identities[str(path)] = _sha256(path)
    return identities


def _manifest_semantic_identity(root: Path, mode: str) -> str | None:
    path = root / "manifests" / f"{mode}.json"
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    identity = payload.get("semantic_identity")
    return str(identity) if identity else None


def build_payload(
    *,
    root: Path,
    config: Path,
    mode: str,
    width: str | None,
    depth: str | None,
    required_patterns: Iterable[str] = (),
    upstream_markers: Iterable[Path] | dict[str, Path] = (),
    stage_config: Any = None,
    source_roots: Iterable[Path] = SOURCE_ROOTS,
    repository_roots: Iterable[Path] = (),
    extra_files: Iterable[Path] = (),
) -> dict:
    """Build a version-3 marker with semantic freshness and broad provenance."""

    root = Path(root).resolve()
    config = Path(config).resolve()
    artifacts = _required_artifacts(root, required_patterns)
    upstream_identities = _upstream_identities(upstream_markers)
    relevant_config = (
        stage_config if stage_config is not None else {"config_sha256": _sha256(config)}
    )
    payload = {
        "version": 3,
        "mode": mode,
        "width": width,
        "depth": depth,
        "config": str(config),
        "relevant_stage_config_identity": stable_hash(relevant_config, prefix=f"{mode}_resume_cfg"),
        "stage_manifest_semantic_identity": _manifest_semantic_identity(root, mode),
        "required_artifacts": artifacts,
        "upstream_identities": upstream_identities,
        "implementation_provenance": {
            "source_identity": source_identity(
                config,
                source_roots=source_roots,
                repository_roots=repository_roots,
                extra_files=extra_files,
            )
        },
    }
    payload["completion_identity"] = stable_hash(
        {
            "mode": mode,
            "width": width,
            "depth": depth,
            "relevant_stage_config_identity": payload["relevant_stage_config_identity"],
            "stage_manifest_semantic_identity": payload["stage_manifest_semantic_identity"],
            "required_artifacts": artifacts,
            "upstream_identities": upstream_identities,
        },
        prefix=f"{mode}_completion",
    )
    return payload


def _build_legacy_payload(
    *,
    root: Path,
    config: Path,
    mode: str,
    width: str | None,
    depth: str | None,
    required_patterns: Iterable[str] = (),
    upstream_markers: Iterable[Path] | dict[str, Path] = (),
    source_roots: Iterable[Path] = SOURCE_ROOTS,
    repository_roots: Iterable[Path] = (),
    extra_files: Iterable[Path] = (),
    **_: Any,
) -> dict:
    root = Path(root).resolve()
    config = Path(config).resolve()
    return {
        "version": 2,
        "mode": mode,
        "width": width,
        "depth": depth,
        "source_identity": source_identity(
            config,
            source_roots=source_roots,
            repository_roots=repository_roots,
            extra_files=extra_files,
        ),
        "config": str(config),
        "required_artifacts": _required_artifacts(root, required_patterns),
        "upstream_identities": _legacy_upstream_identities(upstream_markers),
    }


def marker_matches(path: Path, expected: dict) -> bool:
    """Return whether a marker exactly equals an expected payload."""

    try:
        path = _completion_path_without_symlinks(path)
    except _UnverifiableMarkerError:
        return False
    if not path.is_file():
        return False
    try:
        return json.loads(path.read_text()) == expected
    except (OSError, json.JSONDecodeError):
        return False


def check_marker_details(path: Path, **payload_kwargs) -> CompletionCheck:
    """Validate a marker and return deterministic freshness reasons."""

    try:
        safe_path = _completion_path_without_symlinks(Path(path))
        actual = json.loads(safe_path.read_text())
    except (OSError, json.JSONDecodeError, _UnverifiableMarkerError):
        return CompletionCheck(False, "unreadable", ("missing or unreadable completion marker",))
    if actual.get("version") == 2:
        try:
            expected = _build_legacy_payload(**payload_kwargs)
        except (FileNotFoundError, OSError) as error:
            return CompletionCheck(False, "legacy-v2", (str(error),))
        reasons: list[str] = []
        if actual.get("source_identity") != expected["source_identity"]:
            reasons.append("changed implementation/source identity")
        comparable_actual = {
            key: value for key, value in actual.items() if key != "source_identity"
        }
        comparable_expected = {
            key: value for key, value in expected.items() if key != "source_identity"
        }
        if comparable_actual != comparable_expected:
            reasons.append("legacy version-2 marker mismatch")
        return CompletionCheck(not reasons, "legacy-v2", tuple(reasons))
    if actual.get("version") != 3:
        return CompletionCheck(False, "unsupported", ("incompatible semantic identity",))
    if actual.get("completion_kind") == "imported":
        mode = str(payload_kwargs.get("mode") or "")
        if actual.get("mode") != mode:
            return CompletionCheck(False, "imported-v3", ("incompatible semantic identity",))
        stage_config = payload_kwargs.get("stage_config")
        relevant_config = stage_config if stage_config is not None else {}
        expected_config_identity = stable_hash(relevant_config, prefix=f"{mode}_resume_cfg")
        if actual.get("relevant_stage_config_identity") != expected_config_identity:
            return CompletionCheck(False, "imported-v3", ("changed relevant stage config",))
        try:
            _current_imported_marker_identity(safe_path, actual)
        except _UnverifiableMarkerError as error:
            message = str(error)
            reason = (
                "incompatible semantic identity"
                if "semantic identity" in message
                else "changed imported output identity"
            )
            return CompletionCheck(False, "imported-v3", (reason,))
        return CompletionCheck(True, "imported-v3")
    try:
        expected = build_payload(**payload_kwargs)
    except FileNotFoundError as error:
        message = str(error)
        if message.startswith("required artifact pattern has no matches: "):
            pattern = message.removeprefix("required artifact pattern has no matches: ")
            message = f"missing output: {pattern}"
        elif message.startswith("required upstream marker is missing: "):
            stage = message.removeprefix("required upstream marker is missing: ")
            message = f"missing selected upstream completion: {stage}"
        return CompletionCheck(False, "semantic-v3", (message,))
    except _SelectedUpstreamUnverifiable as error:
        actual_upstream = actual.get("upstream_identities") or {}
        if error.stale_complete_v3 and error.stage in actual_upstream:
            reason = f"changed selected upstream identity: {error.stage}"
        else:
            reason = str(error)
        return CompletionCheck(False, "semantic-v3", (reason,))
    except (OSError, ValueError) as error:
        return CompletionCheck(False, "semantic-v3", (str(error),))

    reasons: list[str] = []
    if actual.get("relevant_stage_config_identity") != expected["relevant_stage_config_identity"]:
        reasons.append("changed relevant stage config")
    semantic_changed = (
        actual.get("stage_manifest_semantic_identity")
        != expected["stage_manifest_semantic_identity"]
    )
    if semantic_changed:
        reasons.append("incompatible semantic identity")
    actual_upstream = actual.get("upstream_identities") or {}
    expected_upstream = expected["upstream_identities"]
    for stage in sorted(set(actual_upstream) | set(expected_upstream)):
        if actual_upstream.get(stage) != expected_upstream.get(stage):
            reasons.append(f"changed selected upstream identity: {stage}")
    if not semantic_changed:
        actual_outputs = actual.get("required_artifacts") or {}
        for pattern, records in expected["required_artifacts"].items():
            if actual_outputs.get(pattern) != records:
                reasons.append(f"changed output identity: {pattern}")
    return CompletionCheck(not reasons, "semantic-v3", tuple(reasons))


def check_marker(path: Path, **payload_kwargs) -> bool:
    """Return whether a legacy or semantic completion marker is fresh."""

    return check_marker_details(path, **payload_kwargs).valid


def write_marker(root: Path, mode: str, value: dict) -> Path:
    """Atomically write a completion marker and return its path."""
    path = marker_path(Path(root), mode, value.get("width"), value.get("depth"))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("check", "mark"))
    parser.add_argument("--root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--width")
    parser.add_argument("--depth")
    parser.add_argument("--require", action="append", default=[])
    parser.add_argument("--upstream-marker", action="append", default=[])
    args = parser.parse_args()
    kwargs = {
        "root": Path(args.root),
        "config": Path(args.config),
        "mode": args.mode,
        "width": args.width,
        "depth": args.depth,
        "required_patterns": tuple(args.require),
        "upstream_markers": tuple(Path(value) for value in args.upstream_marker),
    }
    path = marker_path(Path(args.root), args.mode, args.width, args.depth)
    if args.action == "check":
        raise SystemExit(0 if check_marker(path, **kwargs) else 1)

    expected = build_payload(**kwargs)
    write_marker(Path(args.root), args.mode, expected)


if __name__ == "__main__":
    main()
