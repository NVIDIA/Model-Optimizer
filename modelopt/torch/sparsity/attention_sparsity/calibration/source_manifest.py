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

"""Git-free verification of runtime source extracted from an exact Git archive."""

from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess  # nosec B404 - used only by the explicit outside-container generator
import tarfile
import tempfile
import unicodedata
from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import cast

from .checkpoint_manifest import (
    CheckpointManifestError,
    read_stable_file_snapshot,
    stable_file_sha256,
)

__all__ = [
    "GeneratedSourceManifest",
    "SourceManifestError",
    "VerifiedSourceManifest",
    "create_source_manifest_from_git_archive",
    "verify_source_manifest",
]

_ARCHIVE_SCOPE = "flash_attn"
_MANIFEST_FIELDS = frozenset(
    {
        "source_manifest_schema_version",
        "source_kind",
        "git_commit",
        "git_tree",
        "git_archive_sha256",
        "archive_scope",
        "directories",
        "files",
    }
)
_FILE_FIELDS = frozenset({"path", "mode", "size_bytes", "sha256"})
_FILE_ATTRIBUTE_REPARSE_POINT = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)


class SourceManifestError(ValueError):
    """Raised when source bytes do not match their sealed Git-archive witness."""


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise SourceManifestError(f"source manifest repeats JSON key {key!r}")
        result[key] = value
    return result


def _exact_fields(raw: dict[str, object], expected: frozenset[str], label: str) -> None:
    missing = expected - raw.keys()
    extra = raw.keys() - expected
    if missing or extra:
        raise SourceManifestError(
            f"{label} fields do not match the schema; "
            f"missing={sorted(missing)}, extra={sorted(extra)}"
        )


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"
    ).encode()


def _canonical_text(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or unicodedata.normalize("NFC", value) != value
        or any(ord(character) < 32 for character in value)
    ):
        raise SourceManifestError(f"{label} must be non-empty canonical NFC text")
    return value


def _hex_digest(value: object, length: int, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != length
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise SourceManifestError(f"{label} must be {length} lowercase hexadecimal characters")
    return value


def _relative_path(value: object, label: str) -> str:
    text = _canonical_text(value, label)
    path = PurePosixPath(text)
    if (
        path.is_absolute()
        or text != path.as_posix()
        or "\\" in text
        or any(part in {"", ".", ".."} for part in path.parts)
        or path.parts[0] == ".git"
    ):
        raise SourceManifestError(f"{label} must be a canonical non-.git relative POSIX path")
    return text


def _scoped_path(value: object, label: str) -> str:
    text = _relative_path(value, label)
    if PurePosixPath(text).parts[0] != _ARCHIVE_SCOPE:
        raise SourceManifestError(f"{label} must be within {_ARCHIVE_SCOPE!r}")
    return text


def _is_reparse_point(value: os.stat_result) -> bool:
    return bool(getattr(value, "st_file_attributes", 0) & _FILE_ATTRIBUTE_REPARSE_POINT)


def _file_mode(value: os.stat_result) -> str:
    return "100755" if value.st_mode & 0o111 else "100644"


@dataclass(frozen=True, slots=True)
class _TreeSnapshot:
    directories: tuple[str, ...]
    files: tuple[dict[str, object], ...]


def _source_tree_snapshot(root: Path) -> _TreeSnapshot:
    directories: set[str] = set()
    files: dict[str, dict[str, object]] = {}
    pending: list[tuple[Path, str]] = [(root, "")]
    while pending:
        directory, directory_relative = pending.pop()
        try:
            observed_directory = directory.stat(follow_symlinks=False)
            if not stat.S_ISDIR(observed_directory.st_mode) or _is_reparse_point(
                observed_directory
            ):
                raise SourceManifestError(
                    f"source directory {directory_relative or '.'!r} is not stable"
                )
            with os.scandir(directory) as iterator:
                children = sorted(iterator, key=lambda entry: entry.name)
        except OSError as error:
            raise SourceManifestError(
                f"could not traverse source directory {directory_relative or '.'!r}"
            ) from error
        for child in children:
            if not directory_relative and child.name == ".git":
                continue
            relative = _relative_path(
                f"{directory_relative}/{child.name}".lstrip("/"), "source path"
            )
            path = Path(child.path)
            try:
                observed = child.stat(follow_symlinks=False)
            except OSError as error:
                raise SourceManifestError(f"could not inspect source path {relative!r}") from error
            if stat.S_ISLNK(observed.st_mode) or _is_reparse_point(observed):
                raise SourceManifestError(f"source path {relative!r} must not be a link")
            if stat.S_ISDIR(observed.st_mode):
                directories.add(relative)
                pending.append((path, relative))
            elif stat.S_ISREG(observed.st_mode):
                try:
                    digest = stable_file_sha256(path, label=f"source file {relative!r}")
                except CheckpointManifestError as error:
                    raise SourceManifestError(
                        f"could not hash stable source file {relative!r}"
                    ) from error
                files[relative] = {
                    "path": relative,
                    "mode": _file_mode(observed),
                    "size_bytes": observed.st_size,
                    "sha256": digest,
                }
            else:
                raise SourceManifestError(f"source path {relative!r} is not regular")
    return _TreeSnapshot(
        directories=tuple(sorted(directories)),
        files=tuple(files[path] for path in sorted(files)),
    )


def _parse_manifest(payload: bytes) -> dict[str, object]:
    try:
        raw = json.loads(payload, object_pairs_hook=_strict_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SourceManifestError("source manifest is not strict UTF-8 JSON") from error
    if not isinstance(raw, dict):
        raise SourceManifestError("source manifest must be a JSON object")
    _exact_fields(raw, _MANIFEST_FIELDS, "source manifest")
    if (
        type(raw["source_manifest_schema_version"]) is not int
        or raw["source_manifest_schema_version"] != 1
    ):
        raise SourceManifestError("source_manifest_schema_version must be 1")
    if payload != _canonical_json_bytes(raw):
        raise SourceManifestError("source manifest bytes are not canonical JSON")
    _canonical_text(raw["source_kind"], "source manifest.source_kind")
    _hex_digest(raw["git_commit"], 40, "source manifest.git_commit")
    _hex_digest(raw["git_tree"], 40, "source manifest.git_tree")
    _hex_digest(raw["git_archive_sha256"], 64, "source manifest.git_archive_sha256")
    if raw["archive_scope"] != _ARCHIVE_SCOPE:
        raise SourceManifestError(f"source manifest.archive_scope must be {_ARCHIVE_SCOPE!r}")

    raw_directories = raw["directories"]
    if not isinstance(raw_directories, list):
        raise SourceManifestError("source manifest.directories must be a list")
    directories = [
        _scoped_path(value, f"source manifest.directories[{index}]")
        for index, value in enumerate(raw_directories)
    ]
    if directories != sorted(set(directories)):
        raise SourceManifestError("source manifest directories must be unique and sorted")

    raw_files = raw["files"]
    if not isinstance(raw_files, list) or not raw_files:
        raise SourceManifestError("source manifest.files must be a non-empty list")
    paths: list[str] = []
    for index, item in enumerate(raw_files):
        label = f"source manifest.files[{index}]"
        if not isinstance(item, dict):
            raise SourceManifestError(f"{label} must be an object")
        _exact_fields(item, _FILE_FIELDS, label)
        paths.append(_scoped_path(item["path"], f"{label}.path"))
        if item["mode"] not in {"100644", "100755"}:
            raise SourceManifestError(f"{label}.mode must be 100644 or 100755")
        size = item["size_bytes"]
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise SourceManifestError(f"{label}.size_bytes must be an integer >= 0")
        _hex_digest(item["sha256"], 64, f"{label}.sha256")
    if paths != sorted(set(paths)):
        raise SourceManifestError("source manifest file paths must be unique and sorted")
    if set(paths) & set(directories):
        raise SourceManifestError("source manifest paths cannot be both files and directories")
    return raw


@dataclass(frozen=True, slots=True)
class VerifiedSourceManifest:
    """Identity of an exact, fully enumerated runtime source tree."""

    source_root: Path
    manifest_path: Path
    source_kind: str
    git_commit: str
    git_tree: str
    git_archive_sha256: str
    manifest_sha256: str
    directory_count: int
    file_count: int
    total_size_bytes: int


def verify_source_manifest(
    source: str | Path,
    manifest: str | Path,
    *,
    expected_manifest_sha256: str,
    expected_commit: str,
    expected_source_kind: str,
) -> VerifiedSourceManifest:
    """Verify every non-``.git`` source path without invoking Git."""
    root = Path(source).expanduser().resolve()
    if not root.is_dir():
        raise SourceManifestError("source root must be a local directory")
    expected_digest = _hex_digest(expected_manifest_sha256, 64, "expected source manifest SHA256")
    expected_git_commit = _hex_digest(expected_commit, 40, "expected source Git commit")
    expected_kind = _canonical_text(expected_source_kind, "expected source kind")
    try:
        manifest_snapshot = read_stable_file_snapshot(manifest, label="source manifest")
    except CheckpointManifestError as error:
        raise SourceManifestError("could not read stable source manifest") from error
    if manifest_snapshot.sha256 != expected_digest:
        raise SourceManifestError("source manifest does not match its expected SHA256")
    raw = _parse_manifest(manifest_snapshot.payload)
    if raw["source_kind"] != expected_kind:
        raise SourceManifestError("source manifest kind does not match the expected source kind")
    if raw["git_commit"] != expected_git_commit:
        raise SourceManifestError("source manifest commit does not match the expected Git commit")

    actual = _source_tree_snapshot(root)
    if list(actual.directories) != raw["directories"] or list(actual.files) != raw["files"]:
        raise SourceManifestError("source tree does not exactly match its sealed manifest")
    if _source_tree_snapshot(root) != actual:
        raise SourceManifestError("source tree changed during verification")
    return VerifiedSourceManifest(
        source_root=root,
        manifest_path=manifest_snapshot.path.resolve(),
        source_kind=expected_kind,
        git_commit=expected_git_commit,
        git_tree=str(raw["git_tree"]),
        git_archive_sha256=str(raw["git_archive_sha256"]),
        manifest_sha256=manifest_snapshot.sha256,
        directory_count=len(actual.directories),
        file_count=len(actual.files),
        total_size_bytes=sum(cast("int", item["size_bytes"]) for item in actual.files),
    )


@dataclass(frozen=True, slots=True)
class GeneratedSourceManifest:
    """Hashes emitted by the outside-container Git-archive generator."""

    git_commit: str
    git_tree: str
    git_archive_sha256: str
    manifest_sha256: str


def _run_git(checkout: Path, arguments: list[str]) -> bytes:
    executable = shutil.which("git")
    if executable is None:
        raise SourceManifestError("Git is required by the outside-container generator")
    try:
        return subprocess.run(
            [executable, "-C", str(checkout), *arguments],  # nosec B603
            check=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise SourceManifestError(f"Git command failed: {' '.join(arguments)}") from error


def _fsync_directory(path: Path) -> None:
    directory_flag = getattr(os, "O_DIRECTORY", None)
    if directory_flag is None:
        return
    descriptor = os.open(path, os.O_RDONLY | directory_flag)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _unlink_if_identity(path: Path, identity: tuple[int, int]) -> None:
    if identity[1] == 0:
        return
    try:
        observed = path.stat(follow_symlinks=False)
    except FileNotFoundError:
        return
    if (
        observed.st_ino != 0
        and stat.S_ISREG(observed.st_mode)
        and not _is_reparse_point(observed)
        and (observed.st_dev, observed.st_ino) == identity
    ):
        path.unlink()
        _fsync_directory(path.parent)


def _temporary_payload(destination: Path, payload: bytes) -> tuple[Path, tuple[int, int], int]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    temporary = Path(name)
    identity: tuple[int, int] | None = None
    try:
        with os.fdopen(os.dup(descriptor), "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        opened = os.fstat(descriptor)
        observed = temporary.stat(follow_symlinks=False)
        identity = (opened.st_dev, opened.st_ino)
        if (
            opened.st_ino == 0
            or not stat.S_ISREG(opened.st_mode)
            or _is_reparse_point(observed)
            or (observed.st_dev, observed.st_ino) != identity
        ):
            raise SourceManifestError("source artifact temporary file has no stable identity")
        return temporary, identity, descriptor
    except BaseException:
        if identity is None:
            try:
                opened = os.fstat(descriptor)
                if opened.st_ino != 0 and stat.S_ISREG(opened.st_mode):
                    identity = (opened.st_dev, opened.st_ino)
            except OSError:
                pass
        if os.name == "nt":
            os.close(descriptor)
        if identity is not None:
            _unlink_if_identity(temporary, identity)
        if os.name != "nt":
            os.close(descriptor)
        raise


def _publish_no_clobber(
    temporary: Path, destination: Path, identity: tuple[int, int], descriptor: int
) -> None:
    observed = temporary.stat(follow_symlinks=False)
    opened = os.fstat(descriptor)
    if (
        observed.st_ino == 0
        or (observed.st_dev, observed.st_ino) != identity
        or (opened.st_dev, opened.st_ino) != identity
    ):
        raise SourceManifestError("source artifact temporary file changed before publication")
    os.link(temporary, destination, follow_symlinks=False)
    published = destination.stat(follow_symlinks=False)
    opened_after = os.fstat(descriptor)
    if (
        published.st_ino == 0
        or not stat.S_ISREG(published.st_mode)
        or _is_reparse_point(published)
        or (published.st_dev, published.st_ino) != identity
        or (opened_after.st_dev, opened_after.st_ino) != identity
    ):
        raise SourceManifestError("source artifact destination changed during publication")


def _publish_source_artifacts(
    archive_path: Path,
    archive: bytes,
    manifest_path: Path,
    manifest: bytes,
) -> None:
    archive_temporary_path, archive_temporary_identity, archive_descriptor = _temporary_payload(
        archive_path, archive
    )
    archive_temporary: Path | None = archive_temporary_path
    manifest_temporary: Path | None = None
    manifest_temporary_identity: tuple[int, int] | None = None
    manifest_descriptor: int | None = None
    archive_identity: tuple[int, int] | None = None
    manifest_identity: tuple[int, int] | None = None
    try:
        manifest_temporary, manifest_temporary_identity, manifest_descriptor = _temporary_payload(
            manifest_path, manifest
        )
        assert archive_temporary is not None
        archive_identity = archive_temporary_identity
        _publish_no_clobber(
            archive_temporary, archive_path, archive_temporary_identity, archive_descriptor
        )
        if os.name == "nt":
            os.close(archive_descriptor)
            archive_descriptor = None
        _unlink_if_identity(archive_temporary, archive_temporary_identity)
        _fsync_directory(archive_path.parent)
        archive_temporary = None
        manifest_identity = manifest_temporary_identity
        _publish_no_clobber(
            manifest_temporary,
            manifest_path,
            manifest_temporary_identity,
            manifest_descriptor,
        )
        if os.name == "nt":
            os.close(manifest_descriptor)
            manifest_descriptor = None
        _unlink_if_identity(manifest_temporary, manifest_temporary_identity)
        _fsync_directory(manifest_path.parent)
        manifest_temporary = None
        if (
            stable_file_sha256(archive_path, label="published source archive")
            != sha256(archive).hexdigest()
            or stable_file_sha256(manifest_path, label="published source manifest")
            != sha256(manifest).hexdigest()
        ):
            raise SourceManifestError("published source artifacts failed stable rehash")
    except BaseException as error:
        if os.name == "nt" and manifest_descriptor is not None:
            os.close(manifest_descriptor)
            manifest_descriptor = None
        if os.name == "nt" and archive_descriptor is not None:
            os.close(archive_descriptor)
            archive_descriptor = None
        if manifest_identity is not None:
            _unlink_if_identity(manifest_path, manifest_identity)
        if archive_identity is not None:
            _unlink_if_identity(archive_path, archive_identity)
        if manifest_temporary is not None and manifest_temporary_identity is not None:
            _unlink_if_identity(manifest_temporary, manifest_temporary_identity)
        if archive_temporary is not None:
            _unlink_if_identity(archive_temporary, archive_temporary_identity)
        if isinstance(error, FileExistsError):
            raise SourceManifestError(
                "source artifact destination appeared during publication"
            ) from error
        if isinstance(error, CheckpointManifestError):
            raise SourceManifestError("could not rehash published source artifacts") from error
        raise
    finally:
        if manifest_descriptor is not None:
            os.close(manifest_descriptor)
        if archive_descriptor is not None:
            os.close(archive_descriptor)


def _manifest_from_git_archive(
    archive: bytes, *, source_kind: str, git_commit: str, git_tree: str
) -> bytes:
    directories: set[str] = set()
    files: dict[str, dict[str, object]] = {}
    try:
        with tarfile.open(fileobj=BytesIO(archive), mode="r:") as handle:
            for member in handle.getmembers():
                relative = _relative_path(member.name.rstrip("/"), "Git archive path")
                if member.isdir():
                    directories.add(relative)
                elif member.isfile():
                    extracted = handle.extractfile(member)
                    if extracted is None:
                        raise SourceManifestError(f"could not read Git archive file {relative!r}")
                    payload = extracted.read()
                    if len(payload) != member.size:
                        raise SourceManifestError(f"Git archive file {relative!r} is truncated")
                    files[relative] = {
                        "path": relative,
                        "mode": "100755" if member.mode & 0o111 else "100644",
                        "size_bytes": len(payload),
                        "sha256": sha256(payload).hexdigest(),
                    }
                else:
                    raise SourceManifestError(
                        f"Git archive path {relative!r} is not a regular file or directory"
                    )
    except (tarfile.TarError, OSError) as error:
        raise SourceManifestError("could not parse exact Git archive") from error
    if not files:
        raise SourceManifestError("Git archive contains no source files")
    return _canonical_json_bytes(
        {
            "source_manifest_schema_version": 1,
            "source_kind": _canonical_text(source_kind, "source kind"),
            "git_commit": _hex_digest(git_commit, 40, "Git commit"),
            "git_tree": _hex_digest(git_tree, 40, "Git tree"),
            "git_archive_sha256": sha256(archive).hexdigest(),
            "archive_scope": _ARCHIVE_SCOPE,
            "directories": sorted(directories),
            "files": [files[path] for path in sorted(files)],
        }
    )


def create_source_manifest_from_git_archive(
    checkout: str | Path,
    *,
    expected_commit: str,
    source_kind: str,
    archive_output: str | Path,
    manifest_output: str | Path,
) -> GeneratedSourceManifest:
    """Create a sealed runtime-source witness and its exact scoped Git archive."""
    root = Path(checkout).expanduser().resolve()
    if not root.is_dir():
        raise SourceManifestError("Git checkout must be a local directory")
    commit = _hex_digest(expected_commit, 40, "expected Git commit")
    top_level = _run_git(root, ["rev-parse", "--show-toplevel"]).decode().strip()
    if Path(top_level).resolve() != root:
        raise SourceManifestError("Git checkout must be the repository root")
    if _run_git(root, ["rev-parse", "HEAD"]).decode().strip() != commit:
        raise SourceManifestError("Git checkout HEAD does not match the expected commit")
    status_arguments = ["status", "--porcelain=v1", "--untracked-files=all"]
    if _run_git(root, status_arguments).strip():
        raise SourceManifestError("Git checkout has tracked or untracked modifications")
    tree = _hex_digest(
        _run_git(root, ["rev-parse", "HEAD^{tree}"]).decode().strip(), 40, "Git tree"
    )
    archive = _run_git(root, ["archive", "--format=tar", commit, "--", _ARCHIVE_SCOPE])
    manifest = _manifest_from_git_archive(
        archive, source_kind=source_kind, git_commit=commit, git_tree=tree
    )
    if (
        _run_git(root, ["rev-parse", "HEAD"]).decode().strip() != commit
        or _run_git(root, ["rev-parse", "HEAD^{tree}"]).decode().strip() != tree
        or _run_git(root, status_arguments).strip()
    ):
        raise SourceManifestError("Git checkout changed while its archive was generated")

    archive_path = Path(archive_output).expanduser().resolve()
    manifest_path = Path(manifest_output).expanduser().resolve()
    if archive_path == manifest_path:
        raise SourceManifestError("archive and source-manifest outputs must differ")
    _publish_source_artifacts(archive_path, archive, manifest_path, manifest)
    return GeneratedSourceManifest(
        git_commit=commit,
        git_tree=tree,
        git_archive_sha256=sha256(archive).hexdigest(),
        manifest_sha256=sha256(manifest).hexdigest(),
    )
