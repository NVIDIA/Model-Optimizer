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

"""Download and safely prepare pinned VLM benchmark data."""

from __future__ import annotations

import argparse
import ctypes
import errno
import fcntl
import hashlib
import json
import os
import shutil
import stat
import sys
import tarfile
import tempfile
import zipfile
from contextlib import contextmanager, suppress
from pathlib import Path, PurePosixPath
from typing import IO, TYPE_CHECKING, cast
from urllib.parse import urlparse

from huggingface_hub import HfApi, get_hf_file_metadata, hf_hub_url, snapshot_download
from huggingface_hub.file_download import http_get
from huggingface_hub.utils import build_hf_headers

REPOSITORY_ROOT = Path(__file__).absolute().parents[5]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from examples.puzzletron.evaluation import checkpoint  # noqa: E402
from examples.puzzletron.evaluation.vlm import profile  # noqa: E402

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

_MARKER_NAME = ".modelopt_vlm_benchmark_preparation.json"
_RANGE_MARKER_NAME = ".modelopt_vlm_benchmark_range_download.json"
_SNAPSHOT_MARKER_PREFIX = ".modelopt_vlm_benchmark_snapshot_"
_TEMPORARY_DIR_NAME = ".modelopt_vlm_benchmark_temporary"
_LOCK_DIRECTORY_NAME = ".modelopt_vlm_benchmark_locks"
_COPY_CHUNK_BYTES = 8 * 1024 * 1024
_AT_FDCWD = -100
_RENAME_EXCHANGE = 2


DATASETS = profile.VLM_BENCHMARK_DATASETS

_MVBENCH_MEDIA_ROOTS = (
    "FunQA_test",
    "Moments_in_Time_Raw",
    "clevrer",
    "nturgbd_convert",
    "perception",
    "scene_qa",
    "ssv2_video_mp4",
    "sta",
    "star",
    "tvqa",
    "vlnqa",
)


def benchmark_catalog_contract(tasks: Iterable[str]) -> dict[str, dict[str, object]]:
    """Return the authoritative preparation contract for selected benchmark tasks."""

    return {
        task: {
            "repository": DATASETS[task].repository,
            "revision": DATASETS[task].revision,
            "requires_media": DATASETS[task].preparation_dir is not None,
            "preparation_dir": DATASETS[task].preparation_dir,
        }
        for task in tasks
    }


def _task_selection(value: str) -> tuple[str, ...]:
    tasks = tuple(dict.fromkeys(part.strip() for part in value.split(",") if part.strip()))
    unknown = sorted(set(tasks) - set(DATASETS))
    if not tasks:
        raise argparse.ArgumentTypeError("at least one task is required")
    if unknown:
        raise argparse.ArgumentTypeError(f"unsupported VLM benchmark data tasks: {unknown}")
    return tasks


def _path_traverses_symlink(path: Path) -> bool:
    return any(candidate.is_symlink() for candidate in (path, *path.parents))


def _hub_snapshot(hf_home: Path, task: str) -> Path:
    spec = DATASETS[task]
    repository_cache = f"datasets--{spec.repository.replace('/', '--')}"
    return hf_home / "hub" / repository_cache / "snapshots" / spec.revision


def _download(hf_home: Path, task: str, *, max_workers: int) -> Path:
    spec = DATASETS[task]
    snapshot_download(
        repo_id=spec.repository,
        repo_type="dataset",
        revision=spec.revision,
        cache_dir=hf_home / "hub",
        max_workers=max_workers,
    )
    snapshot = _hub_snapshot(hf_home, task)
    if not snapshot.is_dir():
        raise RuntimeError(f"download did not produce the pinned snapshot: {snapshot}")
    return snapshot


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(_COPY_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _inventory_sha256(entries: list[dict[str, object]]) -> str:
    payload = json.dumps(entries, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


def _inventory_summary_is_valid(payload: dict[str, object], entries: object) -> bool:
    if not isinstance(entries, list) or not entries:
        return False
    if any(
        not isinstance(entry, dict) or not isinstance(entry.get("bytes"), int) for entry in entries
    ):
        return False
    recorded_count = payload.get("file_count") if "file_count" in payload else payload.get("files")
    return (
        payload.get("inventory_sha256") == _inventory_sha256(entries)
        and recorded_count == len(entries)
        and payload.get("bytes") == sum(entry["bytes"] for entry in entries)
    )


def _inventory(root: Path, *, repository_cache: Path | None = None) -> list[dict[str, object]]:
    entries = []
    for path in sorted(root.rglob("*")):
        if path.name == _MARKER_NAME:
            continue
        if path.is_dir() and not path.is_symlink():
            continue
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            if repository_cache is None:
                raise ValueError(f"prepared media inventory contains a symlink: {path}")
            resolved = path.resolve(strict=True)
            if not resolved.is_relative_to(repository_cache) or not resolved.is_file():
                raise ValueError(f"snapshot inventory path escapes its repository cache: {path}")
            stat_result = resolved.stat()
            target = resolved.relative_to(repository_cache).as_posix()
            blob_name = resolved.name
            content_sha256 = _sha256(resolved)
            if (
                len(blob_name) == 64
                and all(character in "0123456789abcdef" for character in blob_name.lower())
                and content_sha256 != blob_name.lower()
            ):
                raise ValueError(f"Hub blob content differs from its SHA-256 identity: {resolved}")
            entries.append(
                {
                    "path": relative,
                    "kind": "hub_blob_symlink",
                    "target": target,
                    "bytes": stat_result.st_size,
                    "mtime_ns": stat_result.st_mtime_ns,
                    "ctime_ns": stat_result.st_ctime_ns,
                    "sha256": content_sha256,
                }
            )
            continue
        if not path.is_file():
            raise ValueError(f"inventory path is not a regular file: {path}")
        stat_result = path.stat()
        entries.append(
            {
                "path": relative,
                "kind": "file",
                "bytes": stat_result.st_size,
                "mtime_ns": stat_result.st_mtime_ns,
                "ctime_ns": stat_result.st_ctime_ns,
                "sha256": _sha256(path),
            }
        )
    if not entries:
        raise ValueError(f"inventory root contains no files: {root}")
    return entries


def _inventory_is_current(
    root: Path,
    entries: object,
    *,
    repository_cache: Path | None = None,
) -> bool:
    if root.is_symlink() or not root.is_dir() or not isinstance(entries, list) or not entries:
        return False
    expected_paths = []
    for entry in entries:
        if not isinstance(entry, dict):
            return False
        relative = entry.get("path")
        if not isinstance(relative, str):
            return False
        try:
            safe_relative = _safe_relative_path(relative)
        except ValueError:
            return False
        path = root.joinpath(*safe_relative.parts)
        expected_paths.append(relative)
        if entry.get("kind") == "hub_blob_symlink":
            if repository_cache is None or not path.is_symlink():
                return False
            try:
                resolved = path.resolve(strict=True)
            except OSError:
                return False
            if (
                not resolved.is_relative_to(repository_cache)
                or not resolved.is_file()
                or resolved.relative_to(repository_cache).as_posix() != entry.get("target")
            ):
                return False
            inspected = resolved
        elif entry.get("kind") == "file":
            if path.is_symlink() or not path.is_file():
                return False
            inspected = path
        else:
            return False
        stat_result = inspected.stat()
        if stat_result.st_size != entry.get("bytes"):
            return False
        expected_sha256 = entry.get("sha256")
        if not isinstance(expected_sha256, str) or _sha256(inspected) != expected_sha256:
            return False
    observed_paths = sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if not (path.is_dir() and not path.is_symlink()) and path.name != _MARKER_NAME
    )
    return sorted(expected_paths) == observed_paths


def _lfs_sha256(entry: object) -> str | None:
    lfs = getattr(entry, "lfs", None)
    if lfs is None:
        return None
    if isinstance(lfs, dict):
        value = lfs.get("sha256")
    else:
        value = getattr(lfs, "sha256", None)
    return value if isinstance(value, str) else None


def _range_download_marker(hf_home: Path, task: str) -> Path:
    spec = DATASETS[task]
    repository_cache = f"datasets--{spec.repository.replace('/', '--')}"
    return hf_home / "hub" / repository_cache / _RANGE_MARKER_NAME


def _snapshot_inventory_marker(hf_home: Path, task: str) -> Path:
    snapshot = _hub_snapshot(hf_home, task)
    return snapshot.parent.parent / f"{_SNAPSHOT_MARKER_PREFIX}{DATASETS[task].revision}.json"


def _snapshot_inventory_report(hf_home: Path, task: str, snapshot: Path) -> dict[str, object]:
    repository_cache = snapshot.parent.parent.resolve()
    entries = _inventory(snapshot, repository_cache=repository_cache)
    marker = _snapshot_inventory_marker(hf_home, task)
    report = {
        "schema": "modelopt.vlm-benchmark-snapshot-inventory/v1",
        "task": task,
        "repository": DATASETS[task].repository,
        "revision": DATASETS[task].revision,
        "snapshot": str(snapshot),
        "files": entries,
        "file_count": len(entries),
        "bytes": sum(cast("int", entry["bytes"]) for entry in entries),
        "inventory_sha256": _inventory_sha256(entries),
    }
    _write_json_atomic(marker, report)
    return {**report, "manifest": str(marker)}


def _snapshot_inventory_is_current(report: object) -> bool:
    if not isinstance(report, dict):
        return False
    snapshot_value = report.get("snapshot")
    manifest_value = report.get("manifest")
    entries = report.get("files")
    if not isinstance(snapshot_value, str) or not isinstance(manifest_value, str):
        return False
    snapshot = Path(snapshot_value)
    repository_cache = snapshot.parent.parent.resolve()
    marker = Path(manifest_value)
    try:
        recorded = json.loads(marker.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    expected = {key: value for key, value in report.items() if key != "manifest"}
    return (
        recorded == expected
        and _inventory_summary_is_valid(report, entries)
        and _inventory_is_current(snapshot, entries, repository_cache=repository_cache)
    )


def _ensure_directory_path(root: Path, directory: Path) -> None:
    if not directory.is_relative_to(root):
        raise ValueError(f"directory escapes its owned root: {directory}")
    relative = directory.relative_to(root)
    if ".." in relative.parts:
        raise ValueError(f"directory escapes its owned root: {directory}")
    if _path_traverses_symlink(root) or not root.is_dir():
        raise ValueError(f"owned root must be a regular directory: {root}")
    current = root
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            raise ValueError(f"directory path traverses a symlink: {directory}")
        if current.exists():
            if not current.is_dir():
                raise ValueError(f"directory path collides with a file: {directory}")
        else:
            current.mkdir()


def _prepare_range_download(hf_home: Path, task: str) -> Path:
    marker = _range_download_marker(hf_home, task)
    _ensure_directory_path(hf_home, marker.parent)
    expected = {
        "schema": "modelopt.vlm-benchmark-range-download/v1",
        "repository": DATASETS[task].repository,
        "revision": DATASETS[task].revision,
    }
    if marker.exists():
        try:
            observed = json.loads(marker.read_text())
        except json.JSONDecodeError as error:
            raise ValueError(f"range-download marker is invalid: {marker}") from error
        for key, value in expected.items():
            if observed.get(key) != value:
                raise ValueError(f"range-download marker differs for {task}: {key}")
    else:
        _write_json_atomic(marker, {**expected, "status": "in_progress"})
    return marker


def _download_range_file(
    *,
    repository: str,
    revision: str,
    entry: object,
    snapshot: Path,
    destination: Path,
) -> dict[str, object]:
    filename = getattr(entry, "path", None)
    expected_size = getattr(entry, "size", None)
    if not isinstance(filename, str) or not isinstance(expected_size, int):
        raise TypeError(f"repository file metadata is incomplete: {entry!r}")
    expected_sha256 = _lfs_sha256(entry)
    partial = destination.with_name(f".{destination.name}.modelopt-part")
    _ensure_directory_path(snapshot, destination.parent)
    if partial.is_symlink():
        raise ValueError(f"range-download path must not be a symlink: {partial}")
    if destination.is_symlink():
        try:
            resolved = destination.resolve(strict=True)
        except OSError as error:
            raise ValueError(f"range-download symlink is invalid: {destination}") from error
        repository_cache = snapshot.parent.parent.resolve()
        if not resolved.is_relative_to(repository_cache) or not resolved.is_file():
            raise ValueError(f"range-download path escapes its repository cache: {destination}")
    if destination.exists():
        if not destination.is_file() or destination.stat().st_size != expected_size:
            raise ValueError(f"downloaded file size differs from pinned metadata: {destination}")
        if expected_sha256 is not None and _sha256(destination) != expected_sha256:
            raise ValueError(f"downloaded file hash differs from pinned metadata: {destination}")
        return {"path": filename, "bytes": expected_size, "status": "reused"}
    resume_size = partial.stat().st_size if partial.exists() else 0
    if resume_size > expected_size:
        raise ValueError(f"partial download exceeds the pinned file size: {partial}")

    source_url = hf_hub_url(
        repository,
        filename=filename,
        repo_type="dataset",
        revision=revision,
    )
    # Public benchmark repositories must remain downloadable on clean workers.
    # ``token=None`` reuses configured credentials when present and otherwise
    # falls back to anonymous access; ``token=True`` rejects the latter before
    # making the metadata request.
    metadata = get_hf_file_metadata(source_url, token=None)
    if metadata.commit_hash != revision or metadata.size != expected_size:
        raise ValueError(f"remote metadata differs from the pinned file: {filename}")
    headers = build_hf_headers(token=None)
    download_url = source_url
    if metadata.xet_file_data is None and source_url != metadata.location:
        download_url = metadata.location
        if urlparse(source_url).netloc != urlparse(download_url).netloc:
            headers.pop("authorization", None)
    with partial.open("ab") as stream:
        http_get(
            download_url,
            stream,
            resume_size=resume_size,
            headers=headers,
            expected_size=expected_size,
            displayed_filename=filename,
        )
    if partial.stat().st_size != expected_size:
        raise RuntimeError(f"range download did not reach the pinned file size: {partial}")
    if expected_sha256 is not None and _sha256(partial) != expected_sha256:
        raise ValueError(f"range download hash differs from pinned metadata: {partial}")
    os.replace(partial, destination)
    return {"path": filename, "bytes": expected_size, "status": "downloaded"}


def _range_download(hf_home: Path, task: str) -> Path:
    spec = DATASETS[task]
    marker = _prepare_range_download(hf_home, task)
    snapshot = _hub_snapshot(hf_home, task)
    _ensure_directory_path(hf_home, snapshot)
    entries = [
        entry
        for entry in HfApi().list_repo_tree(
            spec.repository,
            repo_type="dataset",
            revision=spec.revision,
            recursive=True,
        )
        if getattr(entry, "size", None) is not None
    ]
    reports = []
    for entry in entries:
        relative = _safe_relative_path(entry.path)
        reports.append(
            _download_range_file(
                repository=spec.repository,
                revision=spec.revision,
                entry=entry,
                snapshot=snapshot,
                destination=snapshot.joinpath(*relative.parts),
            )
        )
    _write_json_atomic(
        marker,
        {
            "schema": "modelopt.vlm-benchmark-range-download/v1",
            "repository": spec.repository,
            "revision": spec.revision,
            "status": "complete",
            "files": reports,
        },
    )
    return snapshot


def _safe_relative_path(name: str) -> PurePosixPath:
    relative = PurePosixPath(name)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise ValueError(f"unsafe archive member: {name}")
    return relative


def _marker_payload(task: str, *, status: str) -> dict[str, object]:
    spec = DATASETS[task]
    return {
        "schema": "modelopt.vlm-benchmark-data-preparation/v1",
        "task": task,
        "repository": spec.repository,
        "revision": spec.revision,
        "requires_media": spec.preparation_dir is not None,
        "preparation_dir": spec.preparation_dir,
        "status": status,
    }


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w") as stream:
            stream.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_marker(target: Path, payload: dict[str, object]) -> None:
    _write_json_atomic(target / _MARKER_NAME, payload)


@contextmanager
def _task_lock(hf_home: Path, task: str) -> Iterator[None]:
    lock_root = hf_home / _LOCK_DIRECTORY_NAME
    _ensure_directory_path(hf_home, lock_root)
    lock_path = lock_root / f"{task}.lock"
    if lock_path.is_symlink():
        raise ValueError(f"benchmark preparation lock must not be a symlink: {lock_path}")
    flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(lock_path, flags, 0o600)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError(f"benchmark preparation lock must be a regular file: {lock_path}")
        stream = os.fdopen(descriptor, "r+")
    except BaseException:
        os.close(descriptor)
        raise
    with stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _cleanup_temporary_directory(target: Path) -> None:
    temporary = target / _TEMPORARY_DIR_NAME
    if not temporary.exists() and not temporary.is_symlink():
        return
    if temporary.is_symlink() or not temporary.is_dir():
        raise ValueError(f"managed temporary path is unsafe: {temporary}")
    for entry in temporary.iterdir():
        if entry.is_symlink() or not entry.is_file():
            raise ValueError(f"managed temporary entry is unsafe: {entry}")
        entry.unlink()
    temporary.rmdir()


def _remove_owned_media_root(hf_home: Path, target: Path) -> None:
    if (
        target.is_symlink()
        or not target.is_dir()
        or not target.is_relative_to(hf_home)
        or target == hf_home
    ):
        raise ValueError(f"prepared media root is unsafe to repair: {target}")
    for entry in target.rglob("*"):
        if entry.is_symlink():
            raise ValueError(f"prepared media repair refuses a symlink: {entry}")
    shutil.rmtree(target)


def _atomic_exchange_directories(first: Path, second: Path) -> bool:
    """Atomically exchange two directories when the host provides renameat2."""

    renameat2 = getattr(ctypes.CDLL(None, use_errno=True), "renameat2", None)
    if renameat2 is None:
        return False
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        _AT_FDCWD,
        os.fsencode(first),
        _AT_FDCWD,
        os.fsencode(second),
        _RENAME_EXCHANGE,
    )
    if result == 0:
        return True
    error_number = ctypes.get_errno()
    if error_number in {errno.ENOSYS, errno.EINVAL, errno.ENOTSUP}:
        return False
    raise OSError(error_number, os.strerror(error_number))


def _media_marker_is_current(target: Path, task: str, observed: object) -> bool:
    if not isinstance(observed, dict):
        return False
    expected = _marker_payload(task, status="complete")
    if any(observed.get(key) != value for key, value in expected.items()):
        return False
    inventory = observed.get("inventory")
    return _inventory_summary_is_valid(observed, inventory) and _inventory_is_current(
        target, inventory
    )


def _inspect_prepare_target(hf_home: Path, task: str) -> tuple[Path, dict[str, object] | None]:
    preparation_dir = DATASETS[task].preparation_dir
    if preparation_dir is None:
        raise AssertionError(f"video dataset has no preparation directory: {task}")
    target = hf_home / preparation_dir
    marker = target / _MARKER_NAME
    expected = _marker_payload(task, status="in_progress")
    _ensure_directory_path(hf_home, target.parent)
    if target.exists():
        if target.is_symlink():
            raise ValueError(f"media root must not be a symlink: {target}")
        try:
            observed = json.loads(marker.read_text())
        except (FileNotFoundError, json.JSONDecodeError) as error:
            raise FileExistsError(
                f"refusing to repair a media root without a readable ownership marker: {target}"
            ) from error
        if _media_marker_is_current(target, task, observed):
            return target, observed
        if not isinstance(observed, dict):
            raise FileExistsError(f"media root ownership marker must be an object: {marker}")
        for key, value in expected.items():
            if key == "status":
                continue
            if observed.get(key) != value:
                raise ValueError(f"refusing to repair media with mismatched ownership: {key}")
        if observed.get("status") not in {"in_progress", "complete"}:
            raise ValueError("refusing to repair media with an invalid ownership-marker status")
        for entry in target.rglob("*"):
            if entry.is_symlink():
                raise ValueError(f"prepared media repair refuses a symlink: {entry}")
    return target, None


def _copy_member(source: IO[bytes], destination: Path, expected_size: int, root: Path) -> bool:
    if not destination.is_relative_to(root):
        raise ValueError(f"archive destination escapes the media root: {destination}")
    if destination.relative_to(root).parts[0] == _TEMPORARY_DIR_NAME:
        raise ValueError(f"archive destination uses a reserved path: {destination}")
    parent = destination.parent
    while True:
        if parent.is_symlink():
            raise ValueError(f"archive destination traverses a symlink: {destination}")
        if parent == root:
            break
        parent = parent.parent
    if destination.exists():
        if (
            not destination.is_symlink()
            and destination.is_file()
            and destination.stat().st_size == expected_size
        ):
            with destination.open("rb") as observed:
                while True:
                    expected_chunk = source.read(_COPY_CHUNK_BYTES)
                    observed_chunk = observed.read(_COPY_CHUNK_BYTES)
                    if expected_chunk != observed_chunk:
                        raise ValueError(f"prepared file differs from the archive: {destination}")
                    if not expected_chunk:
                        return False
        raise FileExistsError(f"archive member collides with prepared data: {destination}")
    _ensure_directory_path(root, destination.parent)
    temporary_root = root / _TEMPORARY_DIR_NAME
    _ensure_directory_path(root, temporary_root)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f"{destination.name}.", suffix=".part", dir=temporary_root
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            shutil.copyfileobj(source, output, length=_COPY_CHUNK_BYTES)
            output.flush()
            os.fsync(output.fileno())
        if temporary.stat().st_size != expected_size:
            raise RuntimeError(f"archive member size differs after extraction: {destination}")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
        with suppress(OSError):
            temporary_root.rmdir()
    return True


def _extract_zip(
    archive: Path,
    target: Path,
    *,
    allowed_roots: set[str] | None = None,
    video_only: bool = False,
) -> dict[str, object]:
    extracted_files = 0
    extracted_bytes = 0
    with zipfile.ZipFile(archive) as stream:
        for member in stream.infolist():
            relative = _safe_relative_path(member.filename)
            if stat.S_ISLNK(member.external_attr >> 16):
                raise ValueError(f"archive symlink is not allowed: {member.filename}")
            if member.is_dir() or relative.parts[0] == "__MACOSX":
                continue
            if allowed_roots is not None and relative.parts[0] not in allowed_roots:
                raise ValueError(f"archive root differs from the loader layout: {member.filename}")
            if video_only and relative.suffix.lower() != ".mp4":
                continue
            destination = target.joinpath(*relative.parts)
            with stream.open(member) as source:
                if _copy_member(source, destination, member.file_size, target):
                    extracted_files += 1
                    extracted_bytes += member.file_size
    return {
        "archive": str(archive),
        "new_files": extracted_files,
        "new_bytes": extracted_bytes,
    }


class _ConcatenatedReader:
    def __init__(self, paths: Iterable[Path]):
        self._paths = iter(paths)
        self._current: IO[bytes] | None = None

    def read(self, size: int = -1) -> bytes:
        if size == 0:
            return b""
        chunks = []
        remaining = size
        while remaining != 0:
            if self._current is None:
                try:
                    self._current = next(self._paths).open("rb")
                except StopIteration:
                    break
            chunk = self._current.read(remaining)
            if chunk:
                chunks.append(chunk)
                if remaining > 0:
                    remaining -= len(chunk)
                continue
            self._current.close()
            self._current = None
        return b"".join(chunks)

    def close(self) -> None:
        if self._current is not None:
            self._current.close()
            self._current = None


def _extract_tar(stream: IO[bytes], target: Path, *, label: str) -> dict[str, object]:
    extracted_files = 0
    extracted_bytes = 0
    with tarfile.open(fileobj=stream, mode="r|*") as archive:
        for member in archive:
            relative = _safe_relative_path(member.name)
            if member.isdir():
                continue
            if not member.isfile():
                raise ValueError(f"archive link or special file is not allowed: {member.name}")
            source = archive.extractfile(member)
            if source is None:
                raise RuntimeError(f"tar member has no file payload: {member.name}")
            with source:
                destination = target.joinpath(*relative.parts)
                if _copy_member(source, destination, member.size, target):
                    extracted_files += 1
                    extracted_bytes += member.size
    return {
        "archive": label,
        "new_files": extracted_files,
        "new_bytes": extracted_bytes,
    }


def _materialize_snapshot_root(snapshot: Path, root_name: str, target: Path) -> dict[str, object]:
    source_root = snapshot / root_name
    if source_root.is_symlink() or not source_root.is_dir():
        raise FileNotFoundError(f"snapshot media root is missing or unsafe: {source_root}")
    repository_cache = snapshot.parents[1].resolve()
    copied_files = 0
    copied_bytes = 0
    for source in source_root.rglob("*"):
        if source.is_symlink() and source.resolve(strict=True).is_dir():
            raise ValueError(f"snapshot media path is a directory symlink: {source}")
        if source.is_dir():
            continue
        resolved = source.resolve(strict=True)
        if not resolved.is_relative_to(repository_cache) or not resolved.is_file():
            raise ValueError(f"snapshot media path escapes its repository cache: {source}")
        relative = _safe_relative_path(source.relative_to(snapshot).as_posix())
        size = resolved.stat().st_size
        with resolved.open("rb") as stream:
            if _copy_member(stream, target.joinpath(*relative.parts), size, target):
                copied_files += 1
                copied_bytes += size
    return {
        "archive": f"snapshot:{root_name}",
        "new_files": copied_files,
        "new_bytes": copied_bytes,
    }


def _extract(task: str, snapshot: Path, target: Path) -> list[dict[str, object]]:
    if task == "mmvu_val":
        return [
            _extract_zip(
                snapshot / "videos.zip",
                target,
                allowed_roots={"videos"},
                video_only=True,
            )
        ]
    if task == "video_mmmu":
        archives = tuple(
            snapshot / name
            for name in (
                "Art.zip",
                "Business.zip",
                "Engineering.zip",
                "Humanities.zip",
                "Medicine.zip",
                "Science.zip",
            )
        )
        return [
            _extract_zip(
                archive,
                target,
                allowed_roots={archive.stem},
                video_only=True,
            )
            for archive in archives
        ]
    if task == "mvbench":
        return [
            _materialize_snapshot_root(snapshot, root_name, target)
            for root_name in _MVBENCH_MEDIA_ROOTS
        ]
    if task == "videomme":
        archives = (
            snapshot / "subtitle.zip",
            *(snapshot / f"videos_chunked_{index:02d}.zip" for index in range(1, 21)),
        )
        missing = [archive.name for archive in archives if not archive.is_file()]
        if missing:
            raise FileNotFoundError(f"Video-MME archives are missing from {snapshot}: {missing}")
        return [
            _extract_zip(
                archive,
                target,
                allowed_roots={"subtitle" if archive.name == "subtitle.zip" else "data"},
            )
            for archive in archives
        ]
    if task == "mlvu_dev":
        archives = tuple(snapshot / f"video_part_{index}.zip" for index in range(1, 9))
        missing = [archive.name for archive in archives if not archive.is_file()]
        if missing:
            raise FileNotFoundError(f"MLVU archives are missing from {snapshot}: {missing}")
        return [_extract_zip(archive, target) for archive in archives]
    if task == "longvideobench_val_v":
        reports = []
        subtitles = snapshot / "subtitles.tar"
        with subtitles.open("rb") as stream:
            reports.append(_extract_tar(stream, target, label=str(subtitles)))
        parts = tuple(sorted(snapshot.glob("videos.tar.part.*")))
        if not parts:
            raise FileNotFoundError(
                f"LongVideoBench multipart video archive is missing: {snapshot}"
            )
        stream = _ConcatenatedReader(parts)
        try:
            reports.append(
                _extract_tar(stream, target, label=f"multipart:{parts[0].name}..{parts[-1].name}")
            )
        finally:
            stream.close()
        return reports
    if task == "perceptiontest_val_mc":
        archives = tuple(snapshot / f"videos_chunked_{index:02d}.zip" for index in range(1, 3))
        missing = [archive.name for archive in archives if not archive.is_file()]
        if missing:
            raise FileNotFoundError(
                f"PerceptionTest archives are missing from {snapshot}: {missing}"
            )
        return [
            _extract_zip(
                archive,
                target,
                allowed_roots={"videos"},
                video_only=True,
            )
            for archive in archives
        ]
    raise ValueError(f"unsupported VLM benchmark data task: {task}")


def _prepare(hf_home: Path, task: str, snapshot: Path) -> dict[str, object]:
    with _task_lock(hf_home, task):
        target, complete = _inspect_prepare_target(hf_home, task)
        if complete is not None:
            return complete
        staging = Path(
            tempfile.mkdtemp(prefix=f".{target.name}.modelopt-staging.", dir=target.parent)
        )
        staging.chmod(0o755)
        try:
            _write_marker(staging, _marker_payload(task, status="in_progress"))
            archives = _extract(task, snapshot, staging)
            _cleanup_temporary_directory(staging)
            inventory = _inventory(staging)
            report = {
                **_marker_payload(task, status="complete"),
                "snapshot": str(snapshot),
                "media_root": str(target),
                "archives": archives,
                "files": len(inventory),
                "bytes": sum(cast("int", entry["bytes"]) for entry in inventory),
                "inventory": inventory,
                "inventory_sha256": _inventory_sha256(inventory),
            }
            _write_marker(staging, report)
            if target.exists():
                if not _atomic_exchange_directories(staging, target):
                    raise RuntimeError(
                        "atomic media-directory exchange is unavailable; "
                        "the existing prepared root was preserved"
                    )
                _remove_owned_media_root(hf_home, staging)
            else:
                os.replace(staging, target)
            return report
        finally:
            if staging.exists():
                _remove_owned_media_root(hf_home, staging)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-home", required=True, type=Path)
    parser.add_argument(
        "--tasks",
        required=True,
        type=_task_selection,
        help=f"Comma-separated tasks: {','.join(DATASETS)}",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--download-only", action="store_true")
    mode.add_argument("--extract-only", action="store_true")
    parser.add_argument("--max-workers", type=checkpoint.positive_int, default=8)
    parser.add_argument(
        "--range-resume",
        action="store_true",
        help="Use a deterministic single-writer HTTP range download that resumes across jobs",
    )
    return parser


def prepare_benchmark_datasets(
    hf_home: Path,
    tasks: Iterable[str],
    *,
    max_workers: int = 8,
    range_resume: bool = False,
    expected_catalog: dict[str, object] | None = None,
) -> list[dict[str, object]]:
    """Download pinned benchmark snapshots and prepare media when required."""

    if max_workers <= 0:
        raise ValueError("max_workers must be positive")
    if isinstance(tasks, str):
        raise TypeError("tasks must be an iterable of task names, not a string")
    hf_home = hf_home.expanduser().absolute()
    if _path_traverses_symlink(hf_home):
        raise ValueError(f"HF home must not be a symlink: {hf_home}")
    hf_home.mkdir(parents=True, exist_ok=True)
    raw_tasks = tuple(tasks)
    if any(not isinstance(task, str) or not task.strip() for task in raw_tasks):
        raise TypeError("tasks must contain non-empty task names")
    selected = tuple(task.strip() for task in raw_tasks)
    if len(set(selected)) != len(selected):
        raise ValueError("VLM benchmark data tasks must be unique")
    unknown = sorted(set(selected) - set(DATASETS))
    if not selected:
        raise ValueError("at least one VLM benchmark data task is required")
    if unknown:
        raise ValueError(f"unsupported VLM benchmark data tasks: {unknown}")
    catalog = benchmark_catalog_contract(selected)
    if expected_catalog is not None and expected_catalog != catalog:
        raise ValueError("configured VLM benchmark catalog differs from the authoritative catalog")

    reports = []
    for task in selected:
        snapshot = (
            _range_download(hf_home, task)
            if range_resume
            else _download(hf_home, task, max_workers=max_workers)
        )
        if not snapshot.is_dir():
            raise FileNotFoundError(f"pinned dataset snapshot is missing: {snapshot}")
        spec = DATASETS[task]
        snapshot_inventory = _snapshot_inventory_report(hf_home, task, snapshot)
        report: dict[str, object] = {
            "task": task,
            "repository": spec.repository,
            "revision": spec.revision,
            "snapshot": str(snapshot),
            "requires_media": spec.preparation_dir is not None,
            "preparation_dir": spec.preparation_dir,
            "snapshot_inventory": snapshot_inventory,
            "status": "downloaded",
        }
        if spec.preparation_dir is not None:
            report = {**report, **_prepare(hf_home, task, snapshot)}
        reports.append(report)
    return reports


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    hf_home = args.hf_home.expanduser().absolute()
    if _path_traverses_symlink(hf_home):
        raise ValueError(f"HF home must not be a symlink: {hf_home}")
    hf_home.mkdir(parents=True, exist_ok=True)
    if args.extract_only:
        reports = []
        for task in args.tasks:
            snapshot = _hub_snapshot(hf_home, task)
            if not snapshot.is_dir():
                raise FileNotFoundError(f"pinned dataset snapshot is missing: {snapshot}")
            spec = DATASETS[task]
            report: dict[str, object] = {
                "task": task,
                "repository": spec.repository,
                "revision": spec.revision,
                "snapshot": str(snapshot),
                "requires_media": spec.preparation_dir is not None,
                "status": "downloaded",
            }
            if spec.preparation_dir is not None:
                report = {**report, **_prepare(hf_home, task, snapshot)}
            reports.append(report)
    elif args.download_only:
        reports = []
        for task in args.tasks:
            snapshot = (
                _range_download(hf_home, task)
                if args.range_resume
                else _download(hf_home, task, max_workers=args.max_workers)
            )
            spec = DATASETS[task]
            reports.append(
                {
                    "task": task,
                    "repository": spec.repository,
                    "revision": spec.revision,
                    "snapshot": str(snapshot),
                    "requires_media": spec.preparation_dir is not None,
                    "status": "downloaded",
                }
            )
    else:
        reports = prepare_benchmark_datasets(
            hf_home,
            args.tasks,
            max_workers=args.max_workers,
            range_resume=args.range_resume,
        )
    print(json.dumps({"hf_home": str(hf_home), "tasks": reports}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
