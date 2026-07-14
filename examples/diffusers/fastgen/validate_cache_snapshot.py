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
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from portable_cache import (
    PORTABLE_SNAPSHOT_SCHEMA_VERSION,
    audit_no_absolute_paths,
    load_portable_metadata,
    ordered_sample_ids_sha256,
    resolve_cache_asset,
    select_pdd_holdout_ids,
    sha256_file,
    validate_relative_reference,
)


def _validate_payload(root: Path, entry: dict[str, Any]) -> tuple[Path, str]:
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
    return payload_path, actual_digest


def validate_snapshot(
    cache_root: str | Path,
    *,
    all_index: str = "metadata.json",
    train_index: str = "metadata_train.json",
    heldout_index: str = "metadata_heldout.json",
    reject_orphans: bool = True,
    expected_approved_ids_sha256: str | None = None,
    expected_heldout_count: int | None = None,
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

    split_ids: dict[str, tuple[str, ...]] = {}
    split_policies: dict[str, dict[str, Any]] = {}
    ordered_hashes: dict[str, str] = {}
    index_hashes: dict[str, str] = {}
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
        split_ids[split_name] = tuple(entry["sample_id"] for entry in entries)
        split_policies[split_name] = dict(index["split_policy"])
        ordered_hashes[split_name] = ordered_sample_ids_sha256(split_ids[split_name])
        if ordered_hashes[split_name] != index["ordered_sample_ids_sha256"]:
            raise ValueError(f"{index_name} ordered sample-ID SHA-256 mismatch")
        index_hashes[split_name] = sha256_file(root / index_name)

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

    policies = tuple(split_policies.values())
    if any(policy != policies[0] for policy in policies[1:]):
        raise ValueError("metadata indices disagree on the split policy")
    split_policy = split_policies["all"]

    train_ids = split_ids["train"]
    heldout_ids = split_ids["heldout"]
    all_ids = split_ids["all"]
    train_members = set(train_ids)
    heldout_members = set(heldout_ids)
    overlap = train_members & heldout_members
    if overlap:
        raise ValueError(f"train and heldout splits overlap: {sorted(overlap)[:5]}")
    if train_members | heldout_members != set(all_ids):
        raise ValueError("train and heldout split union does not equal the all split")
    expected_train, expected_heldout = select_pdd_holdout_ids(
        all_ids,
        split_policy["heldout_count"],
    )
    if train_ids != expected_train:
        raise ValueError("train split membership/order does not match the frozen PDD policy")
    if heldout_ids != expected_heldout:
        raise ValueError("heldout split membership/order does not match the frozen PDD policy")
    if split_policy["approved_ordered_ids_sha256"] != ordered_hashes["all"]:
        raise ValueError("split policy approved ordered-ID hash does not match the all index")
    if expected_approved_ids_sha256 is not None:
        if (
            not isinstance(expected_approved_ids_sha256, str)
            or len(expected_approved_ids_sha256) != 64
            or expected_approved_ids_sha256.lower() != expected_approved_ids_sha256
            or any(
                character not in "0123456789abcdef" for character in expected_approved_ids_sha256
            )
        ):
            raise ValueError("expected_approved_ids_sha256 must be lowercase hexadecimal SHA-256")
        if expected_approved_ids_sha256 != ordered_hashes["all"]:
            raise ValueError("snapshot does not match the expected approved ordered-ID hash")
    if expected_heldout_count is not None:
        if type(expected_heldout_count) is not int or expected_heldout_count <= 0:
            raise ValueError("expected_heldout_count must be a positive integer")
        if split_policy["heldout_count"] != expected_heldout_count:
            raise ValueError("snapshot heldout count does not match expected_heldout_count")
    if len(heldout_ids) != split_policy["heldout_count"]:
        raise ValueError("heldout split length does not match split policy")

    payload_hashes = dict(_validate_payload(root, entry) for entry in entries_by_id.values())
    payload_files = set(payload_hashes)
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

    declared_hashes = {
        path.relative_to(root).as_posix(): (
            payload_hashes[path] if path in payload_hashes else sha256_file(path)
        )
        for path in declared_files
    }
    snapshot_digest = hashlib.sha256()
    snapshot_digest.update(b"modelopt-fastgen-cache-snapshot-v1\0")
    for relative, file_digest in sorted(declared_hashes.items()):
        snapshot_digest.update(relative.encode())
        snapshot_digest.update(b"\0")
        snapshot_digest.update(file_digest.encode())
        snapshot_digest.update(b"\n")

    negative_report = None
    if negative_declarations:
        negative_path, negative_sha256 = next(iter(negative_declarations))
        negative_report = {"path": negative_path, "sha256": negative_sha256}

    return {
        "schema_version": 1,
        "record_type": "modelopt_fastgen_portable_snapshot_validation",
        "snapshot_schema_version": PORTABLE_SNAPSHOT_SCHEMA_VERSION,
        "indexes": expected_indexes,
        "split_policy": split_policy,
        "splits": {name: len(ids) for name, ids in split_ids.items()},
        "ordered_sample_ids_sha256": ordered_hashes,
        "index_sha256": index_hashes,
        "negative_prompt_embedding": negative_report,
        "unique_payloads": len(payload_files),
        "declared_files": len(declared_hashes),
        "snapshot_sha256": snapshot_digest.hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--all-index", default="metadata.json")
    parser.add_argument("--train-index", default="metadata_train.json")
    parser.add_argument("--heldout-index", default="metadata_heldout.json")
    parser.add_argument("--allow-orphans", action="store_true")
    parser.add_argument("--expected-approved-ids-sha256")
    parser.add_argument("--expected-heldout-count", type=int)
    args = parser.parse_args()
    report = validate_snapshot(
        args.cache_root,
        all_index=args.all_index,
        train_index=args.train_index,
        heldout_index=args.heldout_index,
        reject_orphans=not args.allow_orphans,
        expected_approved_ids_sha256=args.expected_approved_ids_sha256,
        expected_heldout_count=args.expected_heldout_count,
    )
    print(json.dumps(report, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
