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
import stat
import struct
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

PREPROCESS_STAGING_SCHEMA_VERSION = 1
PORTABLE_SNAPSHOT_SCHEMA_VERSION = 2
APPROVED_IDS_SCHEMA_VERSION = 1
SPLIT_POLICY_SCHEMA_VERSION = 1
DATASET_CACHE_ENV = "MODELOPT_FASTGEN_DATASET_CACHE_DIR"
SAMPLE_ID_DOMAIN = "modelopt-fastgen-sample-v1"
PDD_HOLDOUT_DOMAIN = "modelopt-pdd-holdout-v1"
_ORDERED_IDS_DOMAIN = b"modelopt-fastgen-ordered-sample-ids-v1"
_SPLIT_POLICY_KEYS = {
    "algorithm",
    "approved_ordered_ids_sha256",
    "domain",
    "heldout_count",
    "schema_version",
}
_SHA256_HEX_LENGTH = 64
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

__all__ = [
    "APPROVED_IDS_SCHEMA_VERSION",
    "DATASET_CACHE_ENV",
    "PDD_HOLDOUT_DOMAIN",
    "PORTABLE_SNAPSHOT_SCHEMA_VERSION",
    "PREPROCESS_STAGING_SCHEMA_VERSION",
    "SAMPLE_ID_DOMAIN",
    "SPLIT_POLICY_SCHEMA_VERSION",
    "audit_no_absolute_paths",
    "load_approved_sample_ids",
    "load_portable_metadata",
    "load_strict_json",
    "ordered_sample_ids_sha256",
    "resolve_cache_asset",
    "resolve_cache_root",
    "resolve_negative_embedding",
    "select_pdd_holdout_ids",
    "sha256_file",
    "stable_sample_id",
    "validate_relative_reference",
]


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


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, nested in pairs:
        if key in value:
            raise ValueError(f"JSON object contains duplicate key {key!r}.")
        value[key] = nested
    return value


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"JSON contains non-standard constant {value!r}.")


def load_strict_json(path: Path, *, label: str) -> Any:
    """Load one regular, non-symlink UTF-8 JSON file with strict object syntax."""
    path = Path(path)
    if path.is_symlink():
        raise ValueError(f"{label} must not be a symlink: {path}")
    mode = path.stat().st_mode
    if not stat.S_ISREG(mode):
        raise ValueError(f"{label} must be a regular file: {path}")
    try:
        with path.open(encoding="utf-8") as stream:
            return json.load(
                stream,
                object_pairs_hook=_reject_duplicate_json_keys,
                parse_constant=_reject_json_constant,
            )
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError(f"{label} is not valid UTF-8 JSON: {path}") from error


def _validate_sha256(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != _SHA256_HEX_LENGTH
        or value.lower() != value
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a 64-character lowercase hexadecimal SHA-256 digest.")
    return value


def _validate_ordered_sample_ids(sample_ids: Sequence[str], *, label: str) -> tuple[str, ...]:
    if isinstance(sample_ids, str | bytes) or not isinstance(sample_ids, Sequence):
        raise TypeError(f"{label} must be a sequence of strings.")
    resolved = tuple(sample_ids)
    if not resolved:
        raise ValueError(f"{label} must be non-empty.")
    for index, sample_id in enumerate(resolved):
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError(f"{label}[{index}] must be a non-empty string.")
        try:
            sample_id.encode("utf-8")
        except UnicodeEncodeError as error:
            raise ValueError(f"{label}[{index}] must be UTF-8 encodable.") from error
    if len(resolved) != len(set(resolved)):
        raise ValueError(f"{label} contains duplicates.")
    return resolved


def ordered_sample_ids_sha256(sample_ids: Sequence[str]) -> str:
    """Hash an ordered logical-ID sequence with count and byte-length framing."""
    resolved = _validate_ordered_sample_ids(sample_ids, label="sample_ids")
    digest = hashlib.sha256()
    digest.update(_ORDERED_IDS_DOMAIN)
    digest.update(b"\0")
    digest.update(struct.pack(">Q", len(resolved)))
    for sample_id in resolved:
        encoded = sample_id.encode("utf-8")
        digest.update(struct.pack(">Q", len(encoded)))
        digest.update(encoded)
    return digest.hexdigest()


def load_approved_sample_ids(
    path: Path,
    *,
    expected_sha256: str | None = None,
) -> tuple[tuple[str, ...], str]:
    """Load and authenticate an ordered post-filter sample-ID artifact."""
    value = load_strict_json(path, label="approved_ids_manifest")
    if not isinstance(value, dict):
        raise ValueError("approved_ids_manifest must contain an object.")
    required = {
        "ordered_sample_ids",
        "ordered_sample_ids_sha256",
        "schema_version",
    }
    if set(value) != required:
        raise ValueError(
            "approved_ids_manifest keys mismatch: "
            f"expected={sorted(required)}, actual={sorted(value)}."
        )
    if (
        type(value["schema_version"]) is not int
        or value["schema_version"] != APPROVED_IDS_SCHEMA_VERSION
    ):
        raise ValueError(
            f"approved_ids_manifest.schema_version must be {APPROVED_IDS_SCHEMA_VERSION}."
        )
    sample_ids = _validate_ordered_sample_ids(
        value["ordered_sample_ids"],
        label="approved_ids_manifest.ordered_sample_ids",
    )
    computed = ordered_sample_ids_sha256(sample_ids)
    declared = _validate_sha256(
        value["ordered_sample_ids_sha256"],
        label="approved_ids_manifest.ordered_sample_ids_sha256",
    )
    if declared != computed:
        raise ValueError("approved_ids_manifest ordered sample-ID SHA-256 mismatch.")
    if expected_sha256 is not None:
        expected = _validate_sha256(expected_sha256, label="expected_approved_ids_sha256")
        if expected != computed:
            raise ValueError("approved_ids_manifest does not match expected approved-ID SHA-256.")
    return sample_ids, computed


def select_pdd_holdout_ids(
    sample_ids: Sequence[str], heldout_count: int
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Select the frozen seedless PDD holdout membership while preserving input order."""
    resolved = _validate_ordered_sample_ids(sample_ids, label="sample_ids")
    if type(heldout_count) is not int or not 0 < heldout_count < len(resolved):
        raise ValueError(
            "heldout_count must be an integer strictly between zero and len(sample_ids)."
        )
    ranked = sorted(
        resolved,
        key=lambda sample_id: (
            hashlib.sha256(f"{PDD_HOLDOUT_DOMAIN}\0{sample_id}".encode()).digest(),
            sample_id.encode("utf-8"),
        ),
    )
    heldout_members = set(ranked[:heldout_count])
    train = tuple(sample_id for sample_id in resolved if sample_id not in heldout_members)
    heldout = tuple(sample_id for sample_id in resolved if sample_id in heldout_members)
    return train, heldout


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
    value = load_strict_json(path, label=label)
    if not isinstance(value, expected_type):
        raise ValueError(
            f"{label} must contain {expected_type.__name__}, got {type(value).__name__}."
        )
    return value


def _resolve_strict_json_asset(root: Path, reference: str, *, label: str) -> Path:
    relative = validate_relative_reference(reference, label=label)
    candidate = root.resolve(strict=True) / relative
    if candidate.is_symlink():
        raise ValueError(f"{label} must not be a symlink: {relative}")
    return resolve_cache_asset(root, reference, label=label)


def _validate_split_policy(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != _SPLIT_POLICY_KEYS:
        actual = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise ValueError(
            "metadata_index.split_policy keys mismatch: "
            f"expected={sorted(_SPLIT_POLICY_KEYS)}, actual={actual}."
        )
    if (
        type(value["schema_version"]) is not int
        or value["schema_version"] != SPLIT_POLICY_SCHEMA_VERSION
    ):
        raise ValueError(
            f"metadata_index.split_policy.schema_version must be {SPLIT_POLICY_SCHEMA_VERSION}."
        )
    if value["algorithm"] != "sha256-domain-ranked":
        raise ValueError("metadata_index.split_policy.algorithm is unsupported.")
    if value["domain"] != PDD_HOLDOUT_DOMAIN:
        raise ValueError("metadata_index.split_policy.domain is unsupported.")
    heldout_count = value["heldout_count"]
    if type(heldout_count) is not int or heldout_count <= 0:
        raise ValueError("metadata_index.split_policy.heldout_count must be a positive integer.")
    _validate_sha256(
        value["approved_ordered_ids_sha256"],
        label="metadata_index.split_policy.approved_ordered_ids_sha256",
    )
    return value


def load_portable_metadata(
    root: Path,
    metadata_index: str = "metadata.json",
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Load and validate one split index, filtering IDs before bucket grouping."""
    index_path = _resolve_strict_json_asset(root, metadata_index, label="metadata_index")
    index = _load_json(index_path, expected_type=dict, label="metadata_index")
    audit_no_absolute_paths(index, context="metadata_index")
    if (
        type(index.get("schema_version")) is not int
        or index["schema_version"] != PORTABLE_SNAPSHOT_SCHEMA_VERSION
    ):
        raise ValueError(
            "metadata_index is not an authenticated portable snapshot: "
            f"schema_version must be {PORTABLE_SNAPSHOT_SCHEMA_VERSION}, got "
            f"{index.get('schema_version')!r}; run migrate_cache_manifest.py."
        )
    split_policy = _validate_split_policy(index.get("split_policy"))
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
    if "num_shards" in index and (
        type(index["num_shards"]) is not int or index["num_shards"] != len(shards)
    ):
        raise ValueError("metadata_index.num_shards must be an integer equal to len(shards).")
    if type(index.get("total_items")) is not int or index["total_items"] != len(sample_ids):
        raise ValueError("metadata_index.total_items must equal len(sample_ids).")
    computed_ordered_hash = ordered_sample_ids_sha256(sample_ids)
    declared_ordered_hash = _validate_sha256(
        index.get("ordered_sample_ids_sha256"),
        label="metadata_index.ordered_sample_ids_sha256",
    )
    if declared_ordered_hash != computed_ordered_hash:
        raise ValueError("metadata_index ordered sample-ID SHA-256 mismatch.")
    if index.get("split") == "all" and (
        split_policy["approved_ordered_ids_sha256"] != computed_ordered_hash
    ):
        raise ValueError("all metadata index does not match the approved ordered sample-ID hash.")
    negative = index.get("negative_prompt_embedding")
    if negative is not None:
        if not isinstance(negative, dict) or set(negative) != {"path", "sha256"}:
            raise ValueError(
                "metadata_index.negative_prompt_embedding must contain path and sha256."
            )
        validate_relative_reference(
            negative["path"],
            label="metadata_index.negative_prompt_embedding.path",
        )
        _validate_sha256(
            negative["sha256"],
            label="metadata_index.negative_prompt_embedding.sha256",
        )

    entries_by_id: dict[str, dict[str, Any]] = {}
    for shard_number, shard_ref in enumerate(shards):
        shard_path = _resolve_strict_json_asset(
            root,
            shard_ref,
            label=f"shards[{shard_number}]",
        )
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
            _validate_sha256(entry.get("payload_sha256"), label=f"{label}.payload_sha256")
            if "source_ref" in entry:
                validate_relative_reference(entry["source_ref"], label=f"{label}.source_ref")
            entries_by_id[sample_id] = entry

    missing = [sample_id for sample_id in sample_ids if sample_id not in entries_by_id]
    if missing:
        raise ValueError(f"metadata_index references missing sample_ids: {missing[:5]}")
    return index, [entries_by_id[sample_id] for sample_id in sample_ids]
