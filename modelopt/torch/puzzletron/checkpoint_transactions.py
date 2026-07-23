# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Transactional helpers for Puzzletron HF checkpoint realizations."""

from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path
from typing import Mapping

__all__ = [
    "REALIZATION_MANIFEST",
    "REALIZATION_TMP_SUFFIX",
    "invalidate_realization",
    "prepare_realization_retry",
    "quarantine_incomplete_realization",
    "realization_is_complete",
    "remove_realization_temp_dir",
]

REALIZATION_MANIFEST = "puzzletron_realization.json"
REALIZATION_TMP_SUFFIX = ".puzzletron-tmp"


def _read_manifest(checkpoint_dir: Path) -> dict | None:
    manifest_path = checkpoint_dir / REALIZATION_MANIFEST
    if not manifest_path.is_file():
        return None
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _indexed_shards_complete(checkpoint_dir: Path) -> bool:
    if (checkpoint_dir / "model.safetensors").is_file() or (
        checkpoint_dir / "pytorch_model.bin"
    ).is_file():
        return (checkpoint_dir / "config.json").is_file()
    for index_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index_path = checkpoint_dir / index_name
        if not index_path.is_file():
            continue
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        shards = {str(shard) for shard in (index.get("weight_map") or {}).values()}
        if shards and all((checkpoint_dir / shard).is_file() for shard in shards):
            return (checkpoint_dir / "config.json").is_file()
    return False


def realization_is_complete(checkpoint_dir: str | Path) -> bool:
    """Return whether a realization directory is complete and loadable."""

    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.is_dir():
        return False
    manifest = _read_manifest(checkpoint_dir)
    if manifest is None or manifest.get("status") != "complete":
        return False
    return _indexed_shards_complete(checkpoint_dir)


def quarantine_incomplete_realization(checkpoint_dir: str | Path) -> Path | None:
    """Move an incomplete or corrupt realization aside before a clean retry."""

    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return None
    if realization_is_complete(checkpoint_dir):
        raise FileExistsError(
            f"refusing to quarantine completed realization checkpoint: {checkpoint_dir}"
        )
    quarantine = checkpoint_dir.with_name(
        f".{checkpoint_dir.name}.realization_quarantine.{uuid.uuid4().hex}"
    )
    checkpoint_dir.replace(quarantine)
    return quarantine


def invalidate_realization(checkpoint_dir: str | Path) -> Path | None:
    """Quarantine a published realization after its first load proves it invalid.

    Unlike :func:`quarantine_incomplete_realization`, this deliberately permits
    moving a checkpoint whose manifest and shard inventory look complete.  It
    must therefore only be called from the first-load failure path.
    """

    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return None
    quarantine = checkpoint_dir.with_name(
        f".{checkpoint_dir.name}.realization_quarantine.{uuid.uuid4().hex}"
    )
    checkpoint_dir.replace(quarantine)
    remove_realization_temp_dir(checkpoint_dir)
    return quarantine


def remove_realization_temp_dir(checkpoint_dir: str | Path) -> None:
    """Remove a stale materialization transaction directory if present."""

    tmp_dir = Path(checkpoint_dir).with_name(
        f"{Path(checkpoint_dir).name}{REALIZATION_TMP_SUFFIX}"
    )
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)


def prepare_realization_retry(
    output_dir: str | Path,
    *,
    expected_identity: Mapping[str, str] | None = None,
) -> bool:
    """Quarantine stale partial realizations so a retry can rebuild cleanly.

    Returns True when the destination was quarantined or absent and a rebuild
    should proceed. Returns False when a matching complete realization exists.
    """

    output_dir = Path(output_dir)
    if not output_dir.exists():
        remove_realization_temp_dir(output_dir)
        return True

    manifest = _read_manifest(output_dir)
    if manifest is not None and manifest.get("status") == "complete":
        if expected_identity is not None and not all(
            manifest.get(key) == value for key, value in expected_identity.items()
        ):
            raise FileExistsError(
                f"realization destination exists with a different identity: {output_dir}"
            )
        if realization_is_complete(output_dir):
            return False
        quarantine_incomplete_realization(output_dir)
        remove_realization_temp_dir(output_dir)
        return True

    quarantine_incomplete_realization(output_dir)
    remove_realization_temp_dir(output_dir)
    return True
