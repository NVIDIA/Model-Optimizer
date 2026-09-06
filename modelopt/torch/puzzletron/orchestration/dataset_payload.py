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

"""Dependency-light validation for stage-owned file inventories."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

__all__ = ["file_inventories_are_complete", "record_file_inventory"]

_SCHEMA = "modelopt.puzzletron.file-inventories/v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def record_file_inventory(
    root: str | Path,
    *,
    ignored_names: tuple[str, ...] = (),
    allowed_symlink_root: str | Path | None = None,
) -> dict[str, Any]:
    """Record the regular files below ``root`` for later completion checks."""

    root = Path(root).expanduser().absolute()
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"inventory root must be a regular directory: {root}")
    symlink_root = (
        Path(allowed_symlink_root).expanduser().resolve()
        if allowed_symlink_root is not None
        else None
    )
    files = []
    for path in sorted(root.rglob("*")):
        if path.name in ignored_names or (path.is_dir() and not path.is_symlink()):
            continue
        if path.is_symlink():
            try:
                inspected = path.resolve(strict=True)
            except OSError as error:
                raise ValueError(f"inventory symlink is invalid: {path}") from error
            if symlink_root is None or not inspected.is_relative_to(symlink_root):
                raise ValueError(f"inventory symlink escapes its allowed root: {path}")
            kind = "symlink"
            target = inspected.relative_to(symlink_root).as_posix()
        elif path.is_file():
            inspected = path
            kind = "file"
            target = None
        else:
            raise ValueError(f"inventory path must be a regular file: {path}")
        stat = inspected.stat()
        entry = {
            "path": path.relative_to(root).as_posix(),
            "kind": kind,
            "bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "ctime_ns": stat.st_ctime_ns,
            "sha256": _sha256(inspected),
        }
        if target is not None:
            entry["target"] = target
        files.append(entry)
    if not files:
        raise ValueError(f"inventory root contains no files: {root}")
    return {
        "root": str(root),
        "ignored_names": list(ignored_names),
        "allowed_symlink_root": str(symlink_root) if symlink_root is not None else None,
        "files": files,
    }


def _inventory_is_complete(inventory: object, *, verify_content: bool = False) -> bool:
    if not isinstance(inventory, Mapping):
        return False
    root_value = inventory.get("root")
    entries = inventory.get("files")
    ignored_names = inventory.get("ignored_names", [])
    symlink_root_value = inventory.get("allowed_symlink_root")
    if (
        not isinstance(root_value, str)
        or not isinstance(entries, list)
        or not entries
        or not isinstance(ignored_names, list)
        or any(not isinstance(name, str) for name in ignored_names)
    ):
        return False
    root = Path(root_value).expanduser().absolute()
    symlink_root = (
        Path(symlink_root_value).expanduser().resolve()
        if isinstance(symlink_root_value, str)
        else None
    )
    if root.is_symlink() or not root.is_dir():
        return False

    expected_paths = []
    files_to_hash = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            return False
        relative_value = entry.get("path")
        digest = entry.get("sha256")
        if not isinstance(relative_value, str) or not isinstance(digest, str):
            return False
        relative = Path(relative_value)
        if relative.is_absolute() or not relative.parts or ".." in relative.parts:
            return False
        path = root / relative
        try:
            if entry.get("kind") == "symlink":
                if symlink_root is None or not path.is_symlink():
                    return False
                inspected = path.resolve(strict=True)
                if not inspected.is_relative_to(symlink_root) or inspected.relative_to(
                    symlink_root
                ).as_posix() != entry.get("target"):
                    return False
            elif entry.get("kind") == "file":
                if path.is_symlink() or not path.is_file():
                    return False
                inspected = path
            else:
                return False
            stat = inspected.stat()
            if stat.st_size != entry.get("bytes"):
                return False
        except OSError:
            return False
        expected_paths.append(relative.as_posix())
        metadata_matches = stat.st_mtime_ns == entry.get(
            "mtime_ns"
        ) and stat.st_ctime_ns == entry.get("ctime_ns")
        if verify_content or not metadata_matches:
            files_to_hash.append((inspected, digest))

    observed_paths = sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if not (path.is_dir() and not path.is_symlink()) and path.name not in ignored_names
    )
    if sorted(expected_paths) != observed_paths:
        return False
    if any(_sha256(path) != digest for path, digest in files_to_hash):
        return False
    return True


def file_inventories_are_complete(payload: object, *, verify_content: bool = False) -> bool:
    """Return whether every file inventory emitted by a stage is still current."""

    if not isinstance(payload, Mapping) or payload.get("schema") != _SCHEMA:
        return False
    inventories = payload.get("inventories")
    return (
        isinstance(inventories, list)
        and bool(inventories)
        and all(
            _inventory_is_complete(inventory, verify_content=verify_content)
            for inventory in inventories
        )
    )
