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

"""Portable cache paths, identities, and metadata loading for FastGen examples."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

CACHE_SCHEMA_VERSION = 1
DATASET_CACHE_ENV = "MODELOPT_FASTGEN_DATASET_CACHE_DIR"
SAMPLE_ID_DOMAIN = "modelopt-fastgen-sample-v1"
_BANNED_PATH_KEYS = {
    "cache_dir",
    "image_path",
    "output_dir",
    "source_dir",
    "source_path",
    "video_path",
}
_PATH_FIELD_NAMES = {
    "cache_file",
    "directory",
    "directories",
    "file",
    "files",
    "path",
    "paths",
    "source_ref",
}


def resolve_cache_root(configured_root: str | os.PathLike[str]) -> Path:
    """Resolve the YAML root unless a valid absolute environment override is set."""
    override = os.environ.get(DATASET_CACHE_ENV)
    selected = Path(override) if override else Path(configured_root)
    if override and not selected.is_absolute():
        raise ValueError(f"{DATASET_CACHE_ENV} must be an absolute path, got {override!r}.")
    try:
        resolved = selected.expanduser().resolve(strict=True)
    except FileNotFoundError as error:
        source = DATASET_CACHE_ENV if override else "configured cache_dir"
        raise FileNotFoundError(f"{source} does not exist: {selected}") from error
    if not resolved.is_dir():
        raise NotADirectoryError(f"dataset cache root is not a directory: {resolved}")
    return resolved


def validate_relative_reference(value: str | os.PathLike[str], *, label: str) -> Path:
    """Validate one portable, normalized relative reference without resolving it."""
    if not isinstance(value, str | os.PathLike):
        raise TypeError(f"{label} must be a relative path string, got {type(value).__name__}.")
    raw = os.fspath(value)
    if not raw or raw == ".":
        raise ValueError(f"{label} must be a non-empty relative path.")
    if "\0" in raw:
        raise ValueError(f"{label} must not contain NUL bytes.")
    if "\\" in raw:
        raise ValueError(f"{label} must use portable '/' separators, got {raw!r}.")
    windows_path = PureWindowsPath(raw)
    if PurePosixPath(raw).is_absolute() or windows_path.is_absolute() or windows_path.drive:
        raise ValueError(f"{label} must be relative, got absolute path {raw!r}.")
    path = Path(raw)
    if any(part in ("", ".", "..") for part in PurePosixPath(raw).parts):
        raise ValueError(f"{label} contains traversal or non-normalized components: {raw!r}.")
    return path


def resolve_cache_asset(
    root: Path,
    reference: str | os.PathLike[str],
    *,
    label: str,
    kind: str = "file",
) -> Path:
    """Resolve an existing relative asset and reject traversal or symlink escape."""
    relative = validate_relative_reference(reference, label=label)
    root = root.resolve(strict=True)
    try:
        resolved = (root / relative).resolve(strict=True)
    except FileNotFoundError as error:
        raise FileNotFoundError(f"{label} does not exist beneath cache root: {relative}") from error
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} resolves outside cache root: {relative}") from error
    if kind == "file" and not resolved.is_file():
        raise ValueError(f"{label} is not a file: {relative}")
    if kind == "directory" and not resolved.is_dir():
        raise ValueError(f"{label} is not a directory: {relative}")
    if kind not in ("file", "directory", "any"):
        raise ValueError(f"unsupported asset kind {kind!r}.")
    return resolved


def resolve_negative_embedding(root: Path, reference: str | os.PathLike[str]) -> Path:
    """Resolve a negative embedding and require it to stay beneath the cache root.

    Portable configs use a relative reference. An absolute reference is accepted only for
    compatibility with existing launch overrides and only when it resolves beneath the same
    effective root.
    """
    raw = os.fspath(reference)
    path = Path(raw).expanduser()
    windows_path = PureWindowsPath(raw)
    if windows_path.drive and not PurePosixPath(raw).is_absolute():
        raise ValueError("negative_prompt_embedding_path must use the host path syntax")
    if not (PurePosixPath(raw).is_absolute() or windows_path.is_absolute()):
        return resolve_cache_asset(root, raw, label="negative_prompt_embedding_path")

    root = root.resolve(strict=True)
    try:
        resolved = path.resolve(strict=True)
    except FileNotFoundError as error:
        raise FileNotFoundError(f"negative prompt embedding does not exist: {path}") from error
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(
            "negative prompt embedding resolves outside the effective cache root"
        ) from error
    if not resolved.is_file():
        raise ValueError(f"negative prompt embedding is not a file: {resolved}")
    return resolved


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Return the hexadecimal SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def stable_sample_id(*, source_ref: str, resolution: Sequence[int], model_type: str) -> str:
    """Derive a root-independent sample ID from a logical source and processing identity."""
    logical = validate_relative_reference(source_ref, label="source_ref").as_posix()
    if len(resolution) != 2 or any(type(value) is not int or value <= 0 for value in resolution):
        raise ValueError(f"resolution must contain two positive integers, got {resolution!r}.")
    if not isinstance(model_type, str) or not model_type:
        raise ValueError("model_type must be a non-empty string.")
    identity = (
        f"{SAMPLE_ID_DOMAIN}\0{model_type}\0{logical}\0{resolution[0]}x{resolution[1]}"
    ).encode()
    return hashlib.sha256(identity).hexdigest()


def _is_absolute_path(value: str) -> bool:
    windows_path = PureWindowsPath(value)
    return (
        PurePosixPath(value).is_absolute()
        or windows_path.is_absolute()
        or bool(windows_path.drive)
        or value.lower().startswith("file://")
    )


def _is_path_field(key: str) -> bool:
    lowered = key.lower()
    return lowered in _PATH_FIELD_NAMES or lowered.endswith(
        ("_path", "_paths", "_dir", "_dirs", "_file", "_files")
    )


def audit_no_absolute_paths(
    value: Any,
    *,
    context: str = "value",
    _path_context: bool = False,
) -> None:
    """Reject absolute locations in path fields while leaving ordinary text untouched."""
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{context} contains non-string key {key!r}.")
            if _is_absolute_path(key):
                raise ValueError(f"{context} contains absolute path key {key!r}.")
            if key.lower() in _BANNED_PATH_KEYS:
                raise ValueError(f"{context} contains forbidden personal-path key {key!r}.")
            audit_no_absolute_paths(
                nested,
                context=f"{context}.{key}",
                _path_context=_path_context or _is_path_field(key),
            )
        return
    if isinstance(value, list | tuple | set | frozenset):
        for index, nested in enumerate(value):
            audit_no_absolute_paths(
                nested,
                context=f"{context}[{index}]",
                _path_context=_path_context,
            )
        return
    if _path_context and isinstance(value, str) and _is_absolute_path(value):
        raise ValueError(f"{context} contains absolute path {value!r}.")


def _load_json(path: Path, *, expected_type: type, label: str):
    try:
        with path.open(encoding="utf-8") as stream:
            value = json.load(stream)
    except json.JSONDecodeError as error:
        raise ValueError(f"{label} is not valid JSON: {path}") from error
    if not isinstance(value, expected_type):
        raise ValueError(
            f"{label} must contain {expected_type.__name__}, got {type(value).__name__}."
        )
    return value


def load_portable_metadata(
    root: Path,
    metadata_index: str = "metadata.json",
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Load and validate one split index, filtering IDs before bucket grouping."""
    index_path = resolve_cache_asset(root, metadata_index, label="metadata_index")
    index = _load_json(index_path, expected_type=dict, label="metadata_index")
    audit_no_absolute_paths(index, context="metadata_index")
    if index.get("schema_version") != CACHE_SCHEMA_VERSION:
        raise ValueError(
            f"metadata_index schema_version must be {CACHE_SCHEMA_VERSION}, "
            f"got {index.get('schema_version')!r}."
        )
    shards = index.get("shards")
    sample_ids = index.get("sample_ids")
    if not isinstance(shards, list) or not shards or any(not isinstance(v, str) for v in shards):
        raise ValueError("metadata_index.shards must be a non-empty list of relative paths.")
    if (
        not isinstance(sample_ids, list)
        or not sample_ids
        or any(not isinstance(v, str) or not v for v in sample_ids)
    ):
        raise ValueError("metadata_index.sample_ids must be a non-empty list of strings.")
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("metadata_index.sample_ids contains duplicates.")
    if len(shards) != len(set(shards)):
        raise ValueError("metadata_index.shards contains duplicates.")
    if "num_shards" in index and index["num_shards"] != len(shards):
        raise ValueError("metadata_index.num_shards must equal len(shards).")
    if index.get("total_items") != len(sample_ids):
        raise ValueError("metadata_index.total_items must equal len(sample_ids).")

    entries_by_id: dict[str, dict[str, Any]] = {}
    for shard_number, shard_ref in enumerate(shards):
        shard_path = resolve_cache_asset(root, shard_ref, label=f"shards[{shard_number}]")
        entries = _load_json(shard_path, expected_type=list, label=f"shards[{shard_number}]")
        for entry_number, entry in enumerate(entries):
            label = f"shards[{shard_number}][{entry_number}]"
            if not isinstance(entry, dict):
                raise ValueError(f"{label} must be an object.")
            audit_no_absolute_paths(entry, context=label)
            sample_id = entry.get("sample_id")
            if not isinstance(sample_id, str) or not sample_id:
                raise ValueError(f"{label}.sample_id must be a non-empty string.")
            if sample_id in entries_by_id:
                raise ValueError(f"duplicate sample_id in shards: {sample_id}")
            resolve_cache_asset(root, entry.get("cache_file"), label=f"{label}.cache_file")
            payload_sha256 = entry.get("payload_sha256")
            if not isinstance(payload_sha256, str) or len(payload_sha256) != 64:
                raise ValueError(f"{label}.payload_sha256 must be a hexadecimal SHA-256 digest.")
            try:
                int(payload_sha256, 16)
            except ValueError as error:
                raise ValueError(
                    f"{label}.payload_sha256 must be a hexadecimal SHA-256 digest."
                ) from error
            if "source_ref" in entry:
                validate_relative_reference(entry["source_ref"], label=f"{label}.source_ref")
            entries_by_id[sample_id] = entry

    missing = [sample_id for sample_id in sample_ids if sample_id not in entries_by_id]
    if missing:
        raise ValueError(f"metadata_index references missing sample_ids: {missing[:5]}")
    return index, [entries_by_id[sample_id] for sample_id in sample_ids]
