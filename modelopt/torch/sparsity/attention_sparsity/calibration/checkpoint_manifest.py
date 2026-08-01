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

"""Strict, content-addressed checkpoint identity for mask-reuse calibration."""

from __future__ import annotations

import json
import os
import stat
import tempfile
import unicodedata
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path, PurePosixPath

__all__ = [
    "CHECKPOINT_MANIFEST_NAME",
    "CheckpointManifestError",
    "StableFileSnapshot",
    "VerifiedCheckpointManifest",
    "create_checkpoint_manifest",
    "read_stable_file_snapshot",
    "verify_checkpoint_manifest",
]

CHECKPOINT_MANIFEST_NAME = "checkpoint_manifest.json"
_MANIFEST_FIELDS = frozenset({"checkpoint_manifest_schema_version", "model", "files"})
_FILE_FIELDS = frozenset({"path", "size_bytes", "sha256"})
_WEIGHT_SUFFIXES = frozenset({".bin", ".pt", ".safetensors"})


class CheckpointManifestError(ValueError):
    """Raised when a checkpoint cannot be bound to its exact file contents."""


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise CheckpointManifestError(f"checkpoint manifest repeats JSON key {key!r}")
        result[key] = value
    return result


def _exact_fields(raw: dict[str, object], expected: frozenset[str], label: str) -> None:
    missing = expected - raw.keys()
    extra = raw.keys() - expected
    if missing or extra:
        raise CheckpointManifestError(
            f"{label} fields do not match the schema; "
            f"missing={sorted(missing)}, extra={sorted(extra)}"
        )


def _text(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or unicodedata.normalize("NFC", value) != value
        or any(ord(character) < 32 for character in value)
    ):
        raise CheckpointManifestError(f"{label} must be non-empty canonical NFC text")
    return value


def _sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise CheckpointManifestError(f"{label} must be a lowercase SHA256")
    return value


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"
    ).encode()


def _open_stable_regular(path: Path, label: str) -> tuple[int, os.stat_result]:
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    except OSError as error:
        raise CheckpointManifestError(
            f"could not open {label} without following symlinks"
        ) from error
    opened = os.fstat(descriptor)
    try:
        named = path.stat(follow_symlinks=False)
    except OSError:
        os.close(descriptor)
        raise
    if (
        not stat.S_ISREG(opened.st_mode)
        or stat.S_ISLNK(named.st_mode)
        or (opened.st_dev, opened.st_ino) != (named.st_dev, named.st_ino)
    ):
        os.close(descriptor)
        raise CheckpointManifestError(f"{label} must be one stable regular file, not a symlink")
    return descriptor, opened


def _hash_stable_regular(
    path: Path, label: str, *, capture_payload: bool = False
) -> tuple[int, str, bytes | None]:
    descriptor, before = _open_stable_regular(path, label)
    digest = sha256()
    payload = bytearray() if capture_payload else None
    observed_size = 0
    try:
        with os.fdopen(descriptor, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                observed_size += len(chunk)
                digest.update(chunk)
                if payload is not None:
                    payload.extend(chunk)
            after = os.fstat(handle.fileno())
        named_after = path.stat(follow_symlinks=False)
    except OSError as error:
        raise CheckpointManifestError(f"could not hash stable {label}") from error
    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    identity_named = (
        named_after.st_dev,
        named_after.st_ino,
        named_after.st_size,
        named_after.st_mtime_ns,
    )
    if identity_before != identity_after or identity_after != identity_named:
        raise CheckpointManifestError(f"{label} changed while it was being hashed")
    return observed_size, digest.hexdigest(), None if payload is None else bytes(payload)


@dataclass(frozen=True, slots=True)
class StableFileSnapshot:
    """Exact bytes and SHA256 read from one stable no-follow descriptor."""

    path: Path
    payload: bytes
    sha256: str


def read_stable_file_snapshot(path: str | Path, *, label: str) -> StableFileSnapshot:
    """Read and hash identical bytes from one stable regular file."""
    source = Path(path)
    _, digest, payload = _hash_stable_regular(source, label, capture_payload=True)
    assert payload is not None
    return StableFileSnapshot(source, payload, digest)


def _checkpoint_files(root: Path, manifest_path: Path) -> set[str]:
    files: set[str] = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise CheckpointManifestError(
                f"checkpoint contains forbidden symlink {path.relative_to(root).as_posix()!r}"
            )
        if path.is_file() and path != manifest_path:
            files.add(path.relative_to(root).as_posix())
    return files


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def create_checkpoint_manifest(checkpoint: str | Path, *, model: str) -> VerifiedCheckpointManifest:
    """Create the deterministic checkpoint manifest without replacing any file."""
    root = Path(checkpoint).expanduser().resolve()
    if not root.is_dir():
        raise CheckpointManifestError("checkpoint must be a local directory")
    manifest_path = root / CHECKPOINT_MANIFEST_NAME
    if os.path.lexists(manifest_path):
        raise CheckpointManifestError(
            f"{CHECKPOINT_MANIFEST_NAME} already exists; refusing to overwrite it"
        )
    files = _checkpoint_files(root, manifest_path)
    entries = []
    for relative in sorted(files):
        size, digest, _ = _hash_stable_regular(root / relative, f"checkpoint file {relative!r}")
        entries.append({"path": relative, "size_bytes": size, "sha256": digest})
    if _checkpoint_files(root, manifest_path) != files:
        raise CheckpointManifestError("checkpoint file set changed while building manifest")
    payload = _canonical_json_bytes(
        {
            "checkpoint_manifest_schema_version": 1,
            "model": _text(model, "model"),
            "files": entries,
        }
    )
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=root,
            prefix=f".{CHECKPOINT_MANIFEST_NAME}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        observed = temporary.stat(follow_symlinks=False)
        identity = observed.st_dev, observed.st_ino
        os.link(temporary, manifest_path, follow_symlinks=False)
        try:
            temporary.unlink()
            temporary = None
            _fsync_directory(root)
        except BaseException:
            try:
                published = manifest_path.stat(follow_symlinks=False)
                if (published.st_dev, published.st_ino) == identity:
                    manifest_path.unlink()
                    _fsync_directory(root)
            finally:
                raise
    except FileExistsError as error:
        raise CheckpointManifestError(
            f"{CHECKPOINT_MANIFEST_NAME} appeared during publication; refusing to overwrite it"
        ) from error
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return verify_checkpoint_manifest(root, expected_model=model)


def _relative_path(value: object, label: str) -> str:
    text = _text(value, label)
    path = PurePosixPath(text)
    if (
        path.is_absolute()
        or text != path.as_posix()
        or "\\" in text
        or any(part in {"", ".", ".."} for part in path.parts)
        or text == CHECKPOINT_MANIFEST_NAME
    ):
        raise CheckpointManifestError(f"{label} must be a canonical relative POSIX path")
    return text


@dataclass(frozen=True, slots=True)
class VerifiedCheckpointManifest:
    """Identity of a checkpoint whose complete file set was SHA256-verified."""

    checkpoint_root: Path
    manifest_path: Path
    model: str
    sha256: str
    file_count: int
    total_size_bytes: int


def verify_checkpoint_manifest(
    checkpoint: str | Path, *, expected_model: str | None = None
) -> VerifiedCheckpointManifest:
    """Verify the fixed manifest under ``checkpoint`` and every declared file.

    The manifest must enumerate every regular file below the loaded checkpoint
    directory except itself. This prevents a manifest that binds only a subset
    of weights or remote-code/tokenizer inputs from naming the checkpoint.
    """
    root = Path(checkpoint).expanduser().resolve()
    if not root.is_dir():
        raise CheckpointManifestError("checkpoint must be a local directory")
    manifest_path = root / CHECKPOINT_MANIFEST_NAME
    _, manifest_digest, payload = _hash_stable_regular(
        manifest_path, "checkpoint manifest", capture_payload=True
    )
    assert payload is not None
    try:
        raw = json.loads(payload, object_pairs_hook=_strict_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CheckpointManifestError("checkpoint manifest is not strict UTF-8 JSON") from error
    if not isinstance(raw, dict):
        raise CheckpointManifestError("checkpoint manifest must be a JSON object")
    _exact_fields(raw, _MANIFEST_FIELDS, "checkpoint manifest")
    if raw["checkpoint_manifest_schema_version"] != 1:
        raise CheckpointManifestError("checkpoint_manifest_schema_version must be 1")
    if payload != _canonical_json_bytes(raw):
        raise CheckpointManifestError("checkpoint manifest bytes are not canonical JSON")
    model = _text(raw["model"], "checkpoint manifest.model")
    if expected_model is not None and model != expected_model:
        raise CheckpointManifestError(
            f"checkpoint manifest model {model!r} does not match requested model {expected_model!r}"
        )
    raw_files = raw["files"]
    if not isinstance(raw_files, list) or not raw_files:
        raise CheckpointManifestError("checkpoint manifest.files must be a non-empty list")

    declared: dict[str, tuple[int, str]] = {}
    for index, item in enumerate(raw_files):
        label = f"checkpoint manifest.files[{index}]"
        if not isinstance(item, dict):
            raise CheckpointManifestError(f"{label} must be an object")
        _exact_fields(item, _FILE_FIELDS, label)
        relative = _relative_path(item["path"], f"{label}.path")
        size = item["size_bytes"]
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise CheckpointManifestError(f"{label}.size_bytes must be an integer >= 0")
        digest = _sha256(item["sha256"], f"{label}.sha256")
        if relative in declared:
            raise CheckpointManifestError(f"checkpoint manifest repeats file {relative!r}")
        declared[relative] = (size, digest)
    if list(declared) != sorted(declared):
        raise CheckpointManifestError("checkpoint manifest files must be sorted by path")
    if "config.json" not in declared or not any(
        Path(relative).suffix in _WEIGHT_SUFFIXES for relative in declared
    ):
        raise CheckpointManifestError(
            "checkpoint manifest must bind config.json and at least one model weight file"
        )

    actual = _checkpoint_files(root, manifest_path)
    if actual != set(declared):
        raise CheckpointManifestError(
            "checkpoint manifest does not exactly cover checkpoint files; "
            f"missing={sorted(actual - set(declared))}, extra={sorted(set(declared) - actual)}"
        )
    total_size = 0
    for relative, (expected_size, expected_digest) in declared.items():
        path = root / relative
        observed_size, observed_digest, _ = _hash_stable_regular(
            path, f"checkpoint file {relative!r}"
        )
        if observed_size != expected_size or observed_digest != expected_digest:
            raise CheckpointManifestError(
                f"checkpoint file {relative!r} does not match its size/SHA256 manifest entry"
            )
        total_size += observed_size
    if _checkpoint_files(root, manifest_path) != actual:
        raise CheckpointManifestError("checkpoint file set changed during verification")
    return VerifiedCheckpointManifest(
        checkpoint_root=root,
        manifest_path=manifest_path,
        model=model,
        sha256=manifest_digest,
        file_count=len(declared),
        total_size_bytes=total_size,
    )
