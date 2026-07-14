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

"""Migrate a legacy absolute-path FastGen cache into an immutable portable snapshot."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from portable_cache import (
    PDD_HOLDOUT_DOMAIN,
    PORTABLE_SNAPSHOT_SCHEMA_VERSION,
    PREPROCESS_STAGING_SCHEMA_VERSION,
    SPLIT_POLICY_SCHEMA_VERSION,
    audit_no_absolute_paths,
    load_approved_sample_ids,
    load_strict_json,
    ordered_sample_ids_sha256,
    resolve_cache_asset,
    select_pdd_holdout_ids,
    sha256_file,
    stable_sample_id,
    validate_relative_reference,
)
from validate_cache_snapshot import validate_snapshot

if TYPE_CHECKING:
    from collections.abc import Sequence

_REMOVED_PATH_KEYS = {
    "cache_dir",
    "cache_file",
    "image_path",
    "output_dir",
    "source_dir",
    "source_path",
    "video_path",
}


@dataclass(frozen=True)
class MigrationRecord:
    """Frozen mapping produced by read-only pass 1 and consumed by pass 2."""

    sample_id: str
    source_ref: str
    source_payload: Path
    source_sha256: str
    destination_ref: str
    manifest_fields: dict[str, Any]


def _load_json(path: Path, expected_type: type, label: str):
    value = load_strict_json(path, label=label)
    if not isinstance(value, expected_type):
        raise ValueError(f"{label} must contain {expected_type.__name__}")
    return value


def _load_source_json(
    root: Path,
    reference: str,
    *,
    expected_type: type,
    label: str,
):
    relative = validate_relative_reference(reference, label=label)
    candidate = root / relative
    if candidate.is_symlink():
        raise ValueError(f"{label} must not be a symlink: {relative}")
    path = resolve_cache_asset(root, reference, label=label)
    return _load_json(path, expected_type, label)


def _relative_to_legacy_prefix(raw: str, prefix: Path, *, label: str) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        return validate_relative_reference(raw, label=label)
    try:
        relative = path.relative_to(prefix)
    except ValueError as error:
        raise ValueError(f"{label} is outside the declared legacy prefix: {raw}") from error
    return validate_relative_reference(relative.as_posix(), label=label)


def _resolve_legacy_payload(
    source_root: Path,
    cache_file: Any,
    legacy_cache_root: Path,
    *,
    label: str,
) -> Path:
    if not isinstance(cache_file, str):
        raise TypeError(f"{label} must be a string")
    relative = _relative_to_legacy_prefix(cache_file, legacy_cache_root, label=label)
    return resolve_cache_asset(source_root, relative.as_posix(), label=label)


def _legacy_source_ref(
    entry: dict[str, Any], legacy_source_root: Path | None, *, label: str
) -> str:
    if "source_ref" in entry:
        return validate_relative_reference(
            entry["source_ref"], label=f"{label}.source_ref"
        ).as_posix()
    image_path = entry.get("image_path")
    if not isinstance(image_path, str):
        raise ValueError(f"{label} needs source_ref or legacy image_path")
    if Path(image_path).is_absolute() and legacy_source_root is None:
        raise ValueError("--legacy-source-root is required for absolute legacy image_path values")
    prefix = legacy_source_root or Path(".")
    return _relative_to_legacy_prefix(image_path, prefix, label=f"{label}.image_path").as_posix()


def _sanitize_payload(value: Any) -> Any:
    if isinstance(value, dict):
        sanitized = {}
        for key, nested in value.items():
            if not isinstance(key, str):
                raise ValueError(f"payload contains non-string key {key!r}")
            if key.lower() not in _REMOVED_PATH_KEYS:
                sanitized[key] = _sanitize_payload(nested)
        return sanitized
    if isinstance(value, list):
        return [_sanitize_payload(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_sanitize_payload(item) for item in value)
    if isinstance(value, set):
        return {_sanitize_payload(item) for item in value}
    if isinstance(value, frozenset):
        return frozenset(_sanitize_payload(item) for item in value)
    return value


def _portable_manifest_fields(entry: dict[str, Any], *, label: str) -> dict[str, Any]:
    required = ("bucket_resolution", "original_resolution", "prompt", "bucket_id", "aspect_ratio")
    missing = [name for name in required if name not in entry]
    if missing:
        raise ValueError(f"{label} is missing required fields: {missing}")
    fields = {name: entry[name] for name in required}
    for name in ("pixels", "model_type", "crop_resolution"):
        if name in entry:
            fields[name] = entry[name]
    audit_no_absolute_paths(fields, context=label)
    return fields


def plan_migration(
    source_root: str | Path,
    *,
    source_index: str | Sequence[str] = "metadata.json",
    legacy_cache_root: str | Path | None = None,
    legacy_source_root: str | Path | None = None,
) -> tuple[MigrationRecord, ...]:
    """Perform read-only pass 1 and return a deterministic frozen mapping."""
    root = Path(source_root).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(f"source_root is not a directory: {root}")
    cache_prefix = Path(legacy_cache_root).expanduser() if legacy_cache_root else root
    source_prefix = Path(legacy_source_root).expanduser() if legacy_source_root else None
    if not cache_prefix.is_absolute():
        raise ValueError("legacy_cache_root must be absolute")
    if source_prefix is not None and not source_prefix.is_absolute():
        raise ValueError("legacy_source_root must be absolute")

    source_indexes = (source_index,) if isinstance(source_index, str) else tuple(source_index)
    if not source_indexes or any(not isinstance(name, str) or not name for name in source_indexes):
        raise ValueError("source_index must contain one or more index paths")
    if len(source_indexes) != len(set(source_indexes)):
        raise ValueError("source_index contains duplicates")

    parsed_indexes = []
    all_shards: set[str] = set()
    for index_number, index_ref in enumerate(source_indexes):
        label = f"source_index[{index_number}]"
        index = _load_source_json(root, index_ref, expected_type=dict, label=label)
        if "schema_version" in index and (
            type(index["schema_version"]) is not int
            or index["schema_version"] != PREPROCESS_STAGING_SCHEMA_VERSION
        ):
            raise ValueError(f"{label}.schema_version is unsupported")
        shards = index.get("shards")
        if (
            not isinstance(shards, list)
            or not shards
            or any(not isinstance(item, str) for item in shards)
        ):
            raise ValueError(f"{label}.shards must be a non-empty list of relative paths")
        if len(shards) != len(set(shards)):
            raise ValueError(f"{label}.shards contains duplicates")
        if "num_shards" in index and (
            type(index["num_shards"]) is not int or index["num_shards"] != len(shards)
        ):
            raise ValueError(f"{label}.num_shards does not match len(shards)")
        overlap = all_shards.intersection(shards)
        if overlap:
            raise ValueError(f"source indices share metadata shards: {sorted(overlap)}")
        all_shards.update(shards)
        parsed_indexes.append((label, index, shards))

    ranked_indexes = [
        index for _, index, _ in parsed_indexes if "shard_world" in index or "shard_rank" in index
    ]
    if ranked_indexes:
        if len(ranked_indexes) != len(parsed_indexes):
            raise ValueError("all source indices must declare shard_rank and shard_world")
        if any(
            type(index.get(field)) is not int
            for index in ranked_indexes
            for field in ("shard_rank", "shard_world")
        ):
            raise ValueError("source shard_rank and shard_world must be integers")
        worlds = {index.get("shard_world") for index in ranked_indexes}
        ranks = {index.get("shard_rank") for index in ranked_indexes}
        if worlds != {len(parsed_indexes)} or ranks != set(range(len(parsed_indexes))):
            raise ValueError("source rank indices are incomplete or inconsistent")

    records: list[MigrationRecord] = []
    seen_ids: set[str] = set()
    for index_label, index, shards in parsed_indexes:
        index_record_count = 0
        source_sample_ids = []
        for shard_number, shard_ref in enumerate(shards):
            shard_label = f"{index_label}.shards[{shard_number}]"
            entries = _load_source_json(
                root,
                shard_ref,
                expected_type=list,
                label=shard_label,
            )
            index_record_count += len(entries)
            for entry_number, entry in enumerate(entries):
                label = f"{shard_label}[{entry_number}]"
                if not isinstance(entry, dict):
                    raise ValueError(f"{label} must be an object")
                source_sample_ids.append(entry.get("sample_id"))
                source_ref = _legacy_source_ref(entry, source_prefix, label=label)
                source_payload = _resolve_legacy_payload(
                    root,
                    entry.get("cache_file"),
                    cache_prefix,
                    label=f"{label}.cache_file",
                )
                resolution = entry.get("bucket_resolution", entry.get("crop_resolution"))
                model_type = entry.get("model_type")
                if not isinstance(model_type, str) or not model_type:
                    raise ValueError(f"{label}.model_type must be a non-empty string")
                sample_id = stable_sample_id(
                    source_ref=source_ref,
                    resolution=resolution,
                    model_type=model_type,
                )
                if sample_id in seen_ids:
                    raise ValueError(f"duplicate migrated sample_id: {sample_id}")
                seen_ids.add(sample_id)

                source_digest = sha256_file(source_payload)
                payload = torch.load(source_payload, map_location="cpu", weights_only=True)
                if not isinstance(payload, dict):
                    raise TypeError(f"{label} payload must be a dict")
                sanitized = _sanitize_payload(payload)
                sanitized["sample_id"] = sample_id
                sanitized["source_ref"] = source_ref
                audit_no_absolute_paths(sanitized, context=f"payload[{sample_id}]")

                records.append(
                    MigrationRecord(
                        sample_id=sample_id,
                        source_ref=source_ref,
                        source_payload=source_payload,
                        source_sha256=source_digest,
                        destination_ref=f"payloads/{sample_id}.pt",
                        manifest_fields=_portable_manifest_fields(entry, label=label),
                    )
                )
        if "total_items" in index and (
            type(index["total_items"]) is not int or index["total_items"] != index_record_count
        ):
            raise ValueError(f"{index_label}.total_items does not match its loaded entry count")
        if "sample_ids" in index and index["sample_ids"] != source_sample_ids:
            raise ValueError(f"{index_label}.sample_ids does not match its loaded entries")

    if not records:
        raise ValueError("legacy source index contains no entries")
    return tuple(sorted(records, key=lambda record: record.sample_id))


def _write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def migrate_cache(
    source_root: str | Path,
    output_root: str | Path,
    *,
    approved_ids_manifest: str | Path,
    heldout_count: int,
    expected_approved_ids_sha256: str | None = None,
    source_index: str | Sequence[str] = "metadata.json",
    legacy_cache_root: str | Path | None = None,
    legacy_source_root: str | Path | None = None,
    negative_embedding: str | None = None,
    shard_size: int = 10000,
) -> dict[str, Any]:
    """Run two-pass migration and atomically publish the validated destination."""
    destination = Path(output_root).expanduser()
    if destination.exists():
        raise FileExistsError(f"output_root already exists: {destination}")
    if not destination.parent.exists():
        raise FileNotFoundError(f"output_root parent does not exist: {destination.parent}")
    if type(shard_size) is not int or shard_size <= 0:
        raise ValueError("shard_size must be a positive integer")

    # Pass 1 is intentionally complete before any destination or staging path is created.
    records = plan_migration(
        source_root,
        source_index=source_index,
        legacy_cache_root=legacy_cache_root,
        legacy_source_root=legacy_source_root,
    )
    approved_ids, approved_digest = load_approved_sample_ids(
        Path(approved_ids_manifest).expanduser(),
        expected_sha256=expected_approved_ids_sha256,
    )
    records_by_id = {record.sample_id: record for record in records}
    unknown_ids = [sample_id for sample_id in approved_ids if sample_id not in records_by_id]
    if unknown_ids:
        raise ValueError(f"approved_ids_manifest references unknown sample IDs: {unknown_ids[:5]}")
    selected_records = tuple(records_by_id[sample_id] for sample_id in approved_ids)
    train_ids, heldout_ids = select_pdd_holdout_ids(approved_ids, heldout_count)

    source = Path(source_root).expanduser().resolve(strict=True)
    negative_source = None
    negative_source_sha256 = None
    if negative_embedding is not None:
        negative_source = _resolve_legacy_payload(
            source,
            negative_embedding,
            Path(legacy_cache_root).expanduser() if legacy_cache_root else source,
            label="negative_embedding",
        )
        negative_source_sha256 = sha256_file(negative_source)
        negative_payload = torch.load(negative_source, map_location="cpu", weights_only=True)
        audit_no_absolute_paths(negative_payload, context="negative_prompt_embedding")

    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent))
    try:
        (staging / "payloads").mkdir()
        portable_entries = []
        for record in selected_records:
            if sha256_file(record.source_payload) != record.source_sha256:
                raise RuntimeError(f"source payload changed after pass 1: {record.source_payload}")
            payload = torch.load(record.source_payload, map_location="cpu", weights_only=True)
            sanitized = _sanitize_payload(payload)
            sanitized["sample_id"] = record.sample_id
            sanitized["source_ref"] = record.source_ref
            audit_no_absolute_paths(sanitized, context=f"payload[{record.sample_id}]")

            destination_path = staging / record.destination_ref
            torch.save(sanitized, destination_path)
            portable_entries.append(
                {
                    "sample_id": record.sample_id,
                    "source_ref": record.source_ref,
                    "cache_file": record.destination_ref,
                    "payload_sha256": sha256_file(destination_path),
                    **record.manifest_fields,
                }
            )

        shard_names = []
        for offset in range(0, len(portable_entries), shard_size):
            name = f"metadata_shard_s{offset // shard_size:04d}.json"
            _write_json(staging / name, portable_entries[offset : offset + shard_size])
            shard_names.append(name)

        negative_declaration = None
        if negative_source is not None:
            if sha256_file(negative_source) != negative_source_sha256:
                raise RuntimeError("negative prompt embedding changed after pass 1")
            negative_name = "negative_prompt_embedding.pt"
            shutil.copyfile(negative_source, staging / negative_name)
            negative_declaration = {
                "path": negative_name,
                "sha256": sha256_file(staging / negative_name),
            }

        common = {
            "schema_version": PORTABLE_SNAPSHOT_SCHEMA_VERSION,
            "shards": shard_names,
            "num_shards": len(shard_names),
            "split_policy": {
                "schema_version": SPLIT_POLICY_SCHEMA_VERSION,
                "algorithm": "sha256-domain-ranked",
                "domain": PDD_HOLDOUT_DOMAIN,
                "heldout_count": heldout_count,
                "approved_ordered_ids_sha256": approved_digest,
            },
        }
        if negative_declaration is not None:
            common["negative_prompt_embedding"] = negative_declaration
        split_specs = {
            "metadata.json": ("all", approved_ids),
            "metadata_train.json": ("train", train_ids),
            "metadata_heldout.json": ("heldout", heldout_ids),
        }
        for name, (split, sample_ids) in split_specs.items():
            index = {
                **common,
                "split": split,
                "sample_ids": list(sample_ids),
                "ordered_sample_ids_sha256": ordered_sample_ids_sha256(sample_ids),
                "total_items": len(sample_ids),
            }
            audit_no_absolute_paths(index, context=name)
            _write_json(staging / name, index)

        validation = validate_snapshot(
            staging,
            expected_approved_ids_sha256=expected_approved_ids_sha256,
            expected_heldout_count=heldout_count,
        )
        os.replace(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    return {
        "schema_version": 1,
        "record_type": "modelopt_fastgen_cache_migration",
        "output_root": str(destination.resolve()),
        "counts": {
            "source": len(records),
            "approved": len(selected_records),
            "filtered": len(records) - len(selected_records),
        },
        "validation": validation,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--source-index",
        action="append",
        dest="source_indexes",
        help="Relative source index; repeat to finalize all rank-local preprocessing indices.",
    )
    parser.add_argument("--legacy-cache-root")
    parser.add_argument("--legacy-source-root")
    parser.add_argument("--negative-embedding")
    parser.add_argument("--approved-ids-manifest", required=True)
    parser.add_argument("--expected-approved-ids-sha256")
    parser.add_argument("--heldout-count", required=True, type=int)
    parser.add_argument("--shard-size", default=10000, type=int)
    args = parser.parse_args()
    report = migrate_cache(
        args.source_root,
        args.output_root,
        approved_ids_manifest=args.approved_ids_manifest,
        heldout_count=args.heldout_count,
        expected_approved_ids_sha256=args.expected_approved_ids_sha256,
        source_index=tuple(args.source_indexes) if args.source_indexes else "metadata.json",
        legacy_cache_root=args.legacy_cache_root,
        legacy_source_root=args.legacy_source_root,
        negative_embedding=args.negative_embedding,
        shard_size=args.shard_size,
    )
    print(json.dumps(report, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
