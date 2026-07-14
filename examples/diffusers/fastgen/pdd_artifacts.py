# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Strict canonical-JSON and relative-artifact helpers for the PDD example."""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any


def sha256_file(path: Path) -> str:
    """Hash one regular file without following a symlink."""
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"PDD artifact is not a regular file: {path}.")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_sha256(value: Any, *, name: str) -> str:
    """Validate and normalize a hexadecimal SHA-256 value."""
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} must be a 64-character SHA-256 digest.")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{name} must be hexadecimal.") from error
    return value.lower()


def _validate_json_value(value: Any, *, name: str = "JSON") -> None:
    if value is None or isinstance(value, str | bool | int):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{name} contains a non-finite number.")
        return
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError(f"{name} object keys must be strings.")
        for key, item in value.items():
            _validate_json_value(item, name=f"{name}.{key}")
        return
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        for index, item in enumerate(value):
            _validate_json_value(item, name=f"{name}[{index}]")
        return
    raise TypeError(f"{name} contains unsupported type {type(value).__name__}.")


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize finite JSON deterministically with a trailing newline."""
    _validate_json_value(value)
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"canonical JSON contains duplicate key {key!r}.")
        value[key] = item
    return value


def _reject_json_constant(token: str) -> None:
    raise ValueError(f"canonical JSON contains {token}.")


def load_canonical_json(path: Path) -> Any:
    """Load canonical JSON, rejecting duplicates, NaN/Inf, and noncanonical bytes."""
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"PDD JSON artifact is not a regular file: {path}.")
    raw = path.read_bytes()
    try:
        value = json.loads(
            raw,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise RuntimeError(f"cannot parse canonical PDD JSON {path}.") from error
    try:
        expected = canonical_json_bytes(value)
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"invalid canonical PDD JSON {path}.") from error
    if raw != expected:
        raise RuntimeError(f"PDD JSON is not in canonical form: {path}.")
    return value


def write_canonical_json(path: Path, value: Any) -> None:
    """Create one canonical JSON file and fsync its contents."""
    data = canonical_json_bytes(value)
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def resolve_relative_artifact(root: Path, reference: str) -> Path:
    """Resolve a normalized POSIX reference beneath root with no symlink component."""
    if not isinstance(reference, str) or not reference:
        raise ValueError("artifact reference must be a non-empty string.")
    if "\\" in reference:
        raise ValueError(f"artifact reference must use POSIX separators: {reference!r}.")
    pure = PurePosixPath(reference)
    if pure.is_absolute() or any(part in ("", ".", "..") for part in pure.parts):
        raise ValueError(f"artifact reference must be normalized and relative: {reference!r}.")
    root = root.resolve()
    if not root.is_dir() or root.is_symlink():
        raise RuntimeError(f"PDD artifact root is not a regular directory: {root}.")
    candidate = root
    for part in pure.parts:
        candidate = candidate / part
        if candidate.is_symlink():
            raise RuntimeError(f"PDD artifact reference traverses a symlink: {reference!r}.")
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"artifact reference escapes its root: {reference!r}.") from error
    if not resolved.is_file():
        raise FileNotFoundError(f"PDD artifact is missing: {reference!r}.")
    return resolved


def validate_artifact_reference(root: Path, value: Any, *, name: str) -> Path:
    """Validate an exact path/hash reference and return the verified regular file."""
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise ValueError(f"{name} must contain exactly path and sha256.")
    path = resolve_relative_artifact(root, value["path"])
    expected = require_sha256(value["sha256"], name=f"{name}.sha256")
    if sha256_file(path) != expected:
        raise RuntimeError(f"{name} SHA-256 does not match {value['path']!r}.")
    return path
