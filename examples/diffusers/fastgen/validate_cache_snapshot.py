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

"""Read-only integrity validation for a portable FastGen cache snapshot."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from portable_cache import (
    audit_no_absolute_paths,
    load_portable_metadata,
    resolve_cache_asset,
    sha256_file,
    validate_relative_reference,
)


def _validate_payload(root: Path, entry: dict[str, Any]) -> Path:
    payload_path = resolve_cache_asset(root, entry["cache_file"], label="cache_file")
    actual_digest = sha256_file(payload_path)
    if actual_digest != entry["payload_sha256"]:
        raise ValueError(
            f"payload SHA-256 mismatch for {entry['sample_id']}: "
            f"expected {entry['payload_sha256']}, got {actual_digest}"
        )
    payload = torch.load(payload_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise TypeError(f"cache payload must be a dict: {entry['cache_file']}")
    audit_no_absolute_paths(payload, context=f"payload[{entry['sample_id']}]")
    if payload.get("sample_id") != entry["sample_id"]:
        raise ValueError(f"payload/manifest sample_id mismatch for {entry['cache_file']}")
    if payload.get("source_ref") != entry.get("source_ref"):
        raise ValueError(f"payload/manifest source_ref mismatch for {entry['cache_file']}")
    return payload_path


def validate_snapshot(
    cache_root: str | Path,
    *,
    all_index: str = "metadata.json",
    train_index: str = "metadata_train.json",
    heldout_index: str = "metadata_heldout.json",
    reject_orphans: bool = True,
) -> dict[str, Any]:
    """Validate manifests, payload hashes, splits, and declared snapshot inventory.

    The function performs no writes. All, train, and held-out indices are required so inventory,
    split-disjointness, and split-union checks cannot be skipped accidentally.
    """
    root = Path(cache_root).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(f"cache_root is not a directory: {root}")

    expected_indexes = {
        "all": validate_relative_reference(all_index, label="all_index").as_posix(),
        "train": validate_relative_reference(train_index, label="train_index").as_posix(),
        "heldout": validate_relative_reference(heldout_index, label="heldout_index").as_posix(),
    }
    missing_indexes = [name for name in expected_indexes.values() if not (root / name).is_file()]
    if missing_indexes:
        raise FileNotFoundError(f"required metadata indices do not exist: {missing_indexes}")

    split_ids: dict[str, set[str]] = {}
    entries_by_id: dict[str, dict[str, Any]] = {}
    declared_files = {
        resolve_cache_asset(root, name, label="metadata_index")
        for name in expected_indexes.values()
    }
    negative_declarations: set[tuple[str, str]] = set()

    for expected_split, index_name in expected_indexes.items():
        index, entries = load_portable_metadata(root, index_name)
        split_name = index.get("split")
        if split_name != expected_split:
            raise ValueError(f"{index_name}.split must be {expected_split!r}, got {split_name!r}")
        if split_name in split_ids:
            raise ValueError(f"duplicate split declaration: {split_name}")
        split_ids[split_name] = {entry["sample_id"] for entry in entries}

        for shard_ref in index["shards"]:
            declared_files.add(resolve_cache_asset(root, shard_ref, label="metadata shard"))
        negative = index.get("negative_prompt_embedding")
        if negative is not None:
            if not isinstance(negative, dict) or set(negative) != {"path", "sha256"}:
                raise ValueError(
                    f"{index_name}.negative_prompt_embedding must contain path and sha256"
                )
            negative_path = resolve_cache_asset(
                root,
                negative["path"],
                label=f"{index_name}.negative_prompt_embedding.path",
            )
            if sha256_file(negative_path) != negative["sha256"]:
                raise ValueError(f"negative prompt embedding SHA-256 mismatch in {index_name}")
            negative_payload = torch.load(negative_path, map_location="cpu", weights_only=True)
            audit_no_absolute_paths(negative_payload, context="negative_prompt_embedding")
            negative_declarations.add((negative["path"], negative["sha256"]))
            declared_files.add(negative_path)

        for entry in entries:
            sample_id = entry["sample_id"]
            previous = entries_by_id.get(sample_id)
            if previous is not None and previous != entry:
                raise ValueError(f"inconsistent manifest entry for sample_id {sample_id}")
            entries_by_id[sample_id] = entry

    if len(negative_declarations) > 1:
        raise ValueError("metadata indices disagree on the negative prompt embedding")

    train_ids = split_ids.get("train")
    heldout_ids = split_ids.get("heldout")
    if train_ids is not None and heldout_ids is not None:
        overlap = train_ids & heldout_ids
        if overlap:
            raise ValueError(f"train and heldout splits overlap: {sorted(overlap)[:5]}")
        all_ids = split_ids.get("all")
        if all_ids is not None and train_ids | heldout_ids != all_ids:
            raise ValueError("train and heldout split union does not equal the all split")

    payload_files = {_validate_payload(root, entry) for entry in entries_by_id.values()}
    declared_files.update(payload_files)

    if reject_orphans:
        actual_files = set()
        for path in root.rglob("*"):
            if path.is_symlink():
                resolved = path.resolve(strict=True)
                try:
                    resolved.relative_to(root)
                except ValueError as error:
                    raise ValueError(
                        f"snapshot symlink resolves outside cache root: {path}"
                    ) from error
                raise ValueError(f"snapshot contains unsupported symlink: {path}")
            if not path.is_file():
                continue
            resolved = path.resolve(strict=True)
            try:
                resolved.relative_to(root)
            except ValueError as error:
                raise ValueError(f"snapshot file resolves outside cache root: {path}") from error
            actual_files.add(resolved)
        undeclared = sorted(
            path.relative_to(root).as_posix() for path in actual_files - declared_files
        )
        if undeclared:
            raise ValueError(f"snapshot contains undeclared files: {undeclared[:5]}")

    return {
        "root": str(root),
        "indexes": list(expected_indexes.values()),
        "splits": {name: len(ids) for name, ids in split_ids.items()},
        "unique_payloads": len(payload_files),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--all-index", default="metadata.json")
    parser.add_argument("--train-index", default="metadata_train.json")
    parser.add_argument("--heldout-index", default="metadata_heldout.json")
    parser.add_argument("--allow-orphans", action="store_true")
    args = parser.parse_args()
    report = validate_snapshot(
        args.cache_root,
        all_index=args.all_index,
        train_index=args.train_index,
        heldout_index=args.heldout_index,
        reject_orphans=not args.allow_orphans,
    )
    print(report)


if __name__ == "__main__":
    main()
