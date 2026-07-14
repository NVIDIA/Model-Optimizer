# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
import pathlib
import shutil
import sys
import types

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("nemo_automodel")

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]
_FASTGEN_DIR = _REPO_ROOT / "examples" / "diffusers" / "fastgen"
if str(_FASTGEN_DIR) not in sys.path:
    sys.path.insert(0, str(_FASTGEN_DIR))

from fastgen_data import (
    TextToImageDataset,
    build_text_to_image_multiresolution_dataloader,
    collate_fn_text_to_image,
)
from portable_cache import (
    DATASET_CACHE_ENV,
    PORTABLE_SNAPSHOT_SCHEMA_VERSION,
    PREPROCESS_STAGING_SCHEMA_VERSION,
    audit_no_absolute_paths,
    load_approved_sample_ids,
    load_portable_metadata,
    ordered_sample_ids_sha256,
    resolve_cache_root,
    resolve_negative_embedding,
    select_pdd_holdout_ids,
    sha256_file,
)
from validate_cache_snapshot import validate_snapshot


def _sample_id(source_ref: str, resolution: tuple[int, int]) -> str:
    identity = (
        f"modelopt-fastgen-sample-v1\0qwen_image\0{source_ref}\0{resolution[0]}x{resolution[1]}"
    )
    return hashlib.sha256(identity.encode()).hexdigest()


def _write_json(path: pathlib.Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _write_approved_manifest(path: pathlib.Path, sample_ids: list[str]) -> str:
    digest = ordered_sample_ids_sha256(sample_ids)
    _write_json(
        path,
        {
            "schema_version": 1,
            "ordered_sample_ids": sample_ids,
            "ordered_sample_ids_sha256": digest,
        },
    )
    return digest


def _make_id_only_snapshot(
    root: pathlib.Path,
    sample_ids: tuple[str, ...],
    heldout_count: int,
    *,
    heldout_override: tuple[str, ...] | None = None,
) -> None:
    root.mkdir()
    (root / "payload.pt").write_bytes(b"placeholder-not-read-before-split-gates")
    entries = [
        {
            "sample_id": sample_id,
            "cache_file": "payload.pt",
            "payload_sha256": "0" * 64,
        }
        for sample_id in sample_ids
    ]
    _write_json(root / "metadata_shard_s0000.json", entries)
    if heldout_override is None:
        train_ids, heldout_ids = select_pdd_holdout_ids(sample_ids, heldout_count)
    else:
        heldout_ids = heldout_override
        heldout_members = set(heldout_ids)
        train_ids = tuple(sample_id for sample_id in sample_ids if sample_id not in heldout_members)
    approved_digest = ordered_sample_ids_sha256(sample_ids)
    policy = {
        "schema_version": 1,
        "algorithm": "sha256-domain-ranked",
        "domain": "modelopt-pdd-holdout-v1",
        "heldout_count": heldout_count,
        "approved_ordered_ids_sha256": approved_digest,
    }
    for name, split, ids in (
        ("metadata.json", "all", sample_ids),
        ("metadata_train.json", "train", train_ids),
        ("metadata_heldout.json", "heldout", heldout_ids),
    ):
        _write_json(
            root / name,
            {
                "schema_version": 2,
                "split": split,
                "total_items": len(ids),
                "num_shards": 1,
                "shards": ["metadata_shard_s0000.json"],
                "sample_ids": list(ids),
                "ordered_sample_ids_sha256": ordered_sample_ids_sha256(ids),
                "split_policy": policy,
            },
        )


def _make_snapshot(root: pathlib.Path) -> dict[str, list[str]]:
    root.mkdir()
    entries = []
    for index, resolution in enumerate(((64, 64), (64, 64), (128, 64), (128, 64))):
        source_ref = f"class/{chr(ord('a') + index)}.png"
        sample_id = _sample_id(source_ref, resolution)
        cache_ref = f"payloads/{sample_id}.pt"
        payload_path = root / cache_ref
        payload_path.parent.mkdir(exist_ok=True)
        torch.save(
            {
                "latent": torch.full((2, 2, 2), index, dtype=torch.float32),
                "prompt_embeds": torch.full((3, 4), index, dtype=torch.float32),
                "prompt_embeds_mask": torch.ones(3, dtype=torch.long),
                "crop_offset": (0, 0),
                "prompt": f"prompt {index}",
                "sample_id": sample_id,
                "source_ref": source_ref,
            },
            payload_path,
        )
        entries.append(
            {
                "sample_id": sample_id,
                "source_ref": source_ref,
                "cache_file": cache_ref,
                "payload_sha256": sha256_file(payload_path),
                "bucket_resolution": list(resolution),
                "original_resolution": list(resolution),
                "bucket_id": f"bucket-{resolution[0]}",
                "aspect_ratio": resolution[0] / resolution[1],
            }
        )

    _write_json(root / "metadata_shard_s0000.json", entries)
    all_ids = [entry["sample_id"] for entry in entries]
    train_ids, heldout_ids = select_pdd_holdout_ids(all_ids, 1)
    splits = {"all": all_ids, "train": list(train_ids), "heldout": list(heldout_ids)}
    approved_digest = ordered_sample_ids_sha256(all_ids)
    split_policy = {
        "schema_version": 1,
        "algorithm": "sha256-domain-ranked",
        "domain": "modelopt-pdd-holdout-v1",
        "heldout_count": 1,
        "approved_ordered_ids_sha256": approved_digest,
    }
    for split, ids in splits.items():
        name = "metadata.json" if split == "all" else f"metadata_{split}.json"
        _write_json(
            root / name,
            {
                "schema_version": 2,
                "split": split,
                "total_items": len(ids),
                "num_shards": 1,
                "shards": ["metadata_shard_s0000.json"],
                "sample_ids": ids,
                "ordered_sample_ids_sha256": ordered_sample_ids_sha256(ids),
                "split_policy": split_policy,
            },
        )
    negative_path = root / "negative_prompt_embedding.pt"
    torch.save({"embed": torch.arange(12).reshape(3, 4)}, negative_path)
    negative_declaration = {
        "path": negative_path.name,
        "sha256": sha256_file(negative_path),
    }
    for name in ("metadata.json", "metadata_train.json", "metadata_heldout.json"):
        index = json.loads((root / name).read_text())
        index["negative_prompt_embedding"] = negative_declaration
        _write_json(root / name, index)
    return splits


def _batch_signature(dataset: TextToImageDataset) -> list[tuple[str, float]]:
    return [
        (dataset[index]["sample_id"], dataset[index]["latent"].sum().item())
        for index in range(len(dataset))
    ]


def test_authenticated_cache_constants_hash_framing_and_seedless_split() -> None:
    assert PREPROCESS_STAGING_SCHEMA_VERSION == 1
    assert PORTABLE_SNAPSHOT_SCHEMA_VERSION == 2
    expected = {
        ("a\nb",): "41e07cc133e8a85fc4a08e60a38c223f3c24dbca80312d106f251e533254eedf",
        ("a", "b"): "8cf774af4e8509811c2d4bc2adec6b852e4c614f9d8d833924502ead7c0689d7",
        ("ab", "c"): "6df9e72da4c55f09b4c0320337d6a5d46396271ac61b7a60e1ee8146ce49709e",
        ("a", "bc"): "7cedefc9d46613683a89c3081c3a743b66164861975cf100346d59c13cf31d26",
        tuple(
            str(index) for index in range(16)
        ): "b157f73e9710fe1eb2c4f8d94286f304d5c2a9de2b09b31d2b1f5eee15448e69",
    }
    for sample_ids, digest in expected.items():
        assert ordered_sample_ids_sha256(sample_ids) == digest
    assert len(set(expected.values())) == len(expected)

    train, heldout = select_pdd_holdout_ids(tuple(str(index) for index in range(16)), 4)
    assert heldout == ("4", "7", "10", "13")
    assert train == tuple(str(index) for index in range(16) if str(index) not in heldout)
    assert heldout not in (("0", "2", "4", "11"), ("2", "7", "8", "9"))


def test_approved_id_artifact_is_strict_and_externally_authenticatable(tmp_path) -> None:
    path = tmp_path / "approved.json"
    digest = _write_approved_manifest(path, ["second", "first"])
    assert load_approved_sample_ids(path, expected_sha256=digest) == (
        ("second", "first"),
        digest,
    )

    for value, message in (
        ({"schema_version": 1}, "keys mismatch"),
        (
            {
                "schema_version": 1,
                "ordered_sample_ids": ["first", "first"],
                "ordered_sample_ids_sha256": digest,
            },
            "duplicates",
        ),
    ):
        _write_json(path, value)
        with pytest.raises(ValueError, match=message):
            load_approved_sample_ids(path)

    path.write_text('{"schema_version":1,"schema_version":1}')
    with pytest.raises(ValueError, match="duplicate key"):
        load_approved_sample_ids(path)
    path.write_text('{"schema_version": NaN}')
    with pytest.raises(ValueError, match="non-standard constant"):
        load_approved_sample_ids(path)
    path.write_bytes(b"\xff")
    with pytest.raises(ValueError, match="UTF-8 JSON"):
        load_approved_sample_ids(path)
    path.write_text("{")
    with pytest.raises(ValueError, match="UTF-8 JSON"):
        load_approved_sample_ids(path)

    real = tmp_path / "real.json"
    _write_approved_manifest(real, ["first", "second"])
    path.unlink()
    path.symlink_to(real)
    with pytest.raises(ValueError, match="symlink"):
        load_approved_sample_ids(path)
    with pytest.raises(ValueError, match="regular file"):
        load_approved_sample_ids(tmp_path)
    with pytest.raises(ValueError, match="expected approved-ID"):
        load_approved_sample_ids(real, expected_sha256="f" * 64)


@pytest.mark.parametrize(
    "old_heldout",
    [("0", "2", "4", "11"), ("2", "7", "8", "9")],
)
def test_self_consistent_old_seed_partitions_are_rejected(tmp_path, old_heldout) -> None:
    root = tmp_path / "cache"
    sample_ids = tuple(str(index) for index in range(16))
    _make_id_only_snapshot(root, sample_ids, 4, heldout_override=old_heldout)
    with pytest.raises(ValueError, match="frozen PDD policy"):
        validate_snapshot(root, reject_orphans=False)


def test_self_consistent_all_list_rewrite_fails_external_hash(tmp_path) -> None:
    approved_ids = tuple(str(index) for index in range(16))
    rewritten_ids = approved_ids[:-1]
    root = tmp_path / "cache"
    _make_id_only_snapshot(root, rewritten_ids, 4)
    with pytest.raises(ValueError, match="expected approved ordered-ID hash"):
        validate_snapshot(
            root,
            expected_approved_ids_sha256=ordered_sample_ids_sha256(approved_ids),
            reject_orphans=False,
        )


@pytest.mark.parametrize(
    ("policy_update", "message"),
    [
        ({"domain": "modelopt-fastgen-split-v1"}, "domain"),
        ({"algorithm": "other"}, "algorithm"),
        ({"schema_version": 2}, "schema_version"),
        ({"heldout_count": True}, "heldout_count"),
        ({"extra": "field"}, "keys mismatch"),
    ],
)
def test_split_policy_tampering_is_rejected(tmp_path, policy_update, message) -> None:
    root = tmp_path / "cache"
    _make_snapshot(root)
    for name in ("metadata.json", "metadata_train.json", "metadata_heldout.json"):
        index = json.loads((root / name).read_text())
        index["split_policy"].update(policy_update)
        _write_json(root / name, index)
    with pytest.raises(ValueError, match=message):
        validate_snapshot(root)


def test_strict_index_and_shard_json_reject_duplicates_and_nonfinite(tmp_path) -> None:
    root = tmp_path / "cache"
    _make_snapshot(root)
    index_path = root / "metadata.json"
    index_path.write_text('{"schema_version":2,"schema_version":2}')
    with pytest.raises(ValueError, match="duplicate key"):
        load_portable_metadata(root)

    _make_snapshot(tmp_path / "second")
    shard_path = tmp_path / "second" / "metadata_shard_s0000.json"
    shard_path.write_text('[{"sample_id":"a","value":NaN}]')
    with pytest.raises(ValueError, match="non-standard constant"):
        load_portable_metadata(tmp_path / "second")


@pytest.mark.parametrize("actual_heldout_count", [1999, 2001])
def test_validator_rejects_noncanonical_actual_holdout_counts(
    tmp_path, actual_heldout_count
) -> None:
    root = tmp_path / "cache"
    sample_ids = tuple(str(index) for index in range(actual_heldout_count + 1))
    _make_id_only_snapshot(root, sample_ids, actual_heldout_count)
    with pytest.raises(ValueError, match="expected_heldout_count"):
        validate_snapshot(
            root,
            expected_approved_ids_sha256=ordered_sample_ids_sha256(sample_ids),
            expected_heldout_count=2000,
            reject_orphans=False,
        )


def test_cache_root_environment_precedence(monkeypatch, tmp_path):
    configured = tmp_path / "configured"
    override = tmp_path / "override"
    configured.mkdir()
    override.mkdir()

    monkeypatch.delenv(DATASET_CACHE_ENV, raising=False)
    assert resolve_cache_root(configured) == configured.resolve()
    monkeypatch.setenv(DATASET_CACHE_ENV, "")
    assert resolve_cache_root(configured) == configured.resolve()
    monkeypatch.setenv(DATASET_CACHE_ENV, str(override))
    assert resolve_cache_root(configured) == override.resolve()
    monkeypatch.setenv(DATASET_CACHE_ENV, "relative/cache")
    with pytest.raises(ValueError, match="absolute"):
        resolve_cache_root(configured)
    monkeypatch.setenv(DATASET_CACHE_ENV, str(tmp_path / "missing"))
    with pytest.raises(FileNotFoundError):
        resolve_cache_root(configured)
    not_a_directory = tmp_path / "cache-file"
    not_a_directory.write_text("not a directory")
    monkeypatch.setenv(DATASET_CACHE_ENV, str(not_a_directory))
    with pytest.raises(NotADirectoryError):
        resolve_cache_root(configured)


def test_relocation_preserves_order_payloads_and_buckets(monkeypatch, tmp_path):
    first = tmp_path / "alice" / "cache"
    second = tmp_path / "bob" / "cache"
    first.parent.mkdir()
    splits = _make_snapshot(first)
    second.parent.mkdir()
    shutil.copytree(first, second)

    monkeypatch.delenv(DATASET_CACHE_ENV, raising=False)
    first_dataset = TextToImageDataset(str(first), metadata_index="metadata_train.json")
    second_dataset = TextToImageDataset(str(second), metadata_index="metadata_train.json")
    assert _batch_signature(first_dataset) == _batch_signature(second_dataset)
    assert [entry["sample_id"] for entry in second_dataset.metadata] == splits["train"]
    assert first_dataset.bucket_groups == second_dataset.bucket_groups
    first_report = validate_snapshot(first)
    second_report = validate_snapshot(second)
    assert first_report["snapshot_sha256"] == second_report["snapshot_sha256"]
    assert first_report["declared_files"] == second_report["declared_files"]

    monkeypatch.setenv(DATASET_CACHE_ENV, str(second))
    overridden = TextToImageDataset(str(first), metadata_index="metadata_train.json")
    assert overridden.cache_dir == second.resolve()
    assert _batch_signature(overridden) == _batch_signature(first_dataset)


def test_pdd_loader_iterator_does_not_advance_training_rng(monkeypatch, tmp_path):
    root = tmp_path / "cache"
    _make_snapshot(root)
    monkeypatch.delenv(DATASET_CACHE_ENV, raising=False)
    loader, _ = build_text_to_image_multiresolution_dataloader(
        cache_dir=str(root),
        metadata_index="metadata_train.json",
        batch_size=1,
        base_resolution=(64, 64),
        num_workers=0,
        exact_resume=True,
        sampler_seed=17,
        loader_seed=17,
    )
    before = torch.get_rng_state().clone()

    next(iter(loader))

    assert torch.equal(torch.get_rng_state(), before)


def test_split_filters_before_inherited_bucket_grouping(monkeypatch, tmp_path):
    root = tmp_path / "cache"
    splits = _make_snapshot(root)
    shard_path = root / "metadata_shard_s0000.json"
    entries = json.loads(shard_path.read_text())
    heldout_entry = next(entry for entry in entries if entry["sample_id"] in splits["heldout"])
    del heldout_entry["bucket_resolution"]
    _write_json(shard_path, entries)

    monkeypatch.delenv(DATASET_CACHE_ENV, raising=False)
    dataset = TextToImageDataset(str(root), metadata_index="metadata_train.json")
    assert [entry["sample_id"] for entry in dataset.metadata] == splits["train"]
    grouped = [index for bucket in dataset.bucket_groups.values() for index in bucket["indices"]]
    assert sorted(grouped) == list(range(len(splits["train"])))


@pytest.mark.parametrize(
    "reference",
    ["/etc/passwd", "C:/secret.pt", "C:secret.pt", "../escape.pt", "a/../b.pt"],
)
def test_manifest_rejects_nonportable_payload_references(monkeypatch, tmp_path, reference):
    root = tmp_path / "cache"
    _make_snapshot(root)
    shard_path = root / "metadata_shard_s0000.json"
    entries = json.loads(shard_path.read_text())
    entries[0]["cache_file"] = reference
    _write_json(shard_path, entries)

    monkeypatch.delenv(DATASET_CACHE_ENV, raising=False)
    with pytest.raises((ValueError, FileNotFoundError)):
        load_portable_metadata(root, "metadata.json")


def test_manifest_rejects_missing_and_symlink_escape(monkeypatch, tmp_path):
    root = tmp_path / "cache"
    _make_snapshot(root)
    outside = tmp_path / "outside.pt"
    torch.save({}, outside)
    shard_path = root / "metadata_shard_s0000.json"
    original = json.loads(shard_path.read_text())

    monkeypatch.delenv(DATASET_CACHE_ENV, raising=False)
    for reference in ("payloads/missing.pt", "payloads/escape.pt"):
        entries = json.loads(json.dumps(original))
        entries[0]["cache_file"] = reference
        if reference.endswith("escape.pt"):
            (root / reference).symlink_to(outside)
        _write_json(shard_path, entries)
        with pytest.raises((ValueError, FileNotFoundError)):
            load_portable_metadata(root, "metadata.json")


def test_negative_embedding_is_resolved_under_effective_root(tmp_path):
    root = tmp_path / "cache"
    _make_snapshot(root)
    assert (
        resolve_negative_embedding(root, "negative_prompt_embedding.pt")
        == (root / "negative_prompt_embedding.pt").resolve()
    )
    assert resolve_negative_embedding(root, root / "negative_prompt_embedding.pt").is_file()

    outside = tmp_path / "outside.pt"
    torch.save({}, outside)
    with pytest.raises(ValueError, match="outside"):
        resolve_negative_embedding(root, outside)
    (root / "negative_escape.pt").symlink_to(outside)
    with pytest.raises(ValueError, match="outside"):
        resolve_negative_embedding(root, "negative_escape.pt")


def test_collate_emits_logical_identity_without_source_paths(tmp_path):
    root = tmp_path / "cache"
    _make_snapshot(root)
    dataset = TextToImageDataset(str(root), metadata_index="metadata_train.json")
    same_resolution = next(
        group["indices"] for group in dataset.bucket_groups.values() if len(group["indices"]) >= 2
    )
    samples = [dataset[index] for index in same_resolution[:2]]
    output = collate_fn_text_to_image(samples)
    assert output["metadata"]["sample_ids"] == [item["sample_id"] for item in samples]
    assert output["metadata"]["source_refs"] == [item["source_ref"] for item in samples]
    assert "image_paths" not in output["metadata"]
    assert str(root) not in repr(output)


def test_validator_detects_hash_split_and_orphan_failures(tmp_path):
    root = tmp_path / "cache"
    splits = _make_snapshot(root)
    assert validate_snapshot(root)["unique_payloads"] == 4

    orphan = root / "payloads" / "orphan.pt"
    torch.save({"image_path": "/home/alice/private.png"}, orphan)
    with pytest.raises(ValueError, match="undeclared"):
        validate_snapshot(root)
    orphan.unlink()

    heldout_path = root / "metadata_heldout.json"
    heldout = json.loads(heldout_path.read_text())
    heldout["sample_ids"] = [splits["train"][0]]
    heldout["ordered_sample_ids_sha256"] = ordered_sample_ids_sha256(heldout["sample_ids"])
    _write_json(heldout_path, heldout)
    with pytest.raises(ValueError, match="overlap"):
        validate_snapshot(root)


def test_validator_requires_complete_splits_and_rejects_directory_symlink(tmp_path):
    root = tmp_path / "cache"
    _make_snapshot(root)
    (root / "metadata_heldout.json").unlink()
    with pytest.raises(FileNotFoundError, match="required metadata indices"):
        validate_snapshot(root)

    _make_snapshot(tmp_path / "complete")
    complete = tmp_path / "complete"
    outside = tmp_path / "outside-directory"
    outside.mkdir()
    (complete / "escaped-directory").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink resolves outside"):
        validate_snapshot(complete)


def test_recursive_path_audit_is_schema_aware_and_covers_containers():
    audit_no_absolute_paths({"prompt": "/imagine a cat"})
    with pytest.raises(ValueError, match="absolute path"):
        audit_no_absolute_paths({"paths": {"/home/alice/private.png"}})
    with pytest.raises(ValueError, match="absolute path"):
        audit_no_absolute_paths({"paths": {"primary": "/home/alice/private.png"}})
    with pytest.raises(ValueError, match="absolute path"):
        audit_no_absolute_paths({"source_file": {"value": "/lustre/private.pt"}})
    with pytest.raises(ValueError, match="absolute path key"):
        audit_no_absolute_paths({"/home/alice/private.png": "value"})


def test_validator_detects_payload_hash_mismatch_and_is_read_only(tmp_path):
    root = tmp_path / "cache"
    _make_snapshot(root)
    before = {
        path.relative_to(root).as_posix(): (sha256_file(path), path.stat().st_mtime_ns)
        for path in root.rglob("*")
        if path.is_file()
    }
    validate_snapshot(root)
    after = {
        path.relative_to(root).as_posix(): (sha256_file(path), path.stat().st_mtime_ns)
        for path in root.rglob("*")
        if path.is_file()
    }
    assert after == before

    first_payload = next((root / "payloads").glob("*.pt"))
    first_payload.write_bytes(first_payload.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        validate_snapshot(root)


def test_empty_split_is_rejected_before_bucket_grouping(tmp_path):
    root = tmp_path / "cache"
    _make_snapshot(root)
    _write_json(
        root / "metadata_empty.json",
        {
            "schema_version": 2,
            "split": "empty",
            "total_items": 0,
            "shards": ["metadata_shard_s0000.json"],
            "sample_ids": [],
            "ordered_sample_ids_sha256": "0" * 64,
            "split_policy": json.loads((root / "metadata.json").read_text())["split_policy"],
        },
    )
    with pytest.raises(ValueError, match="non-empty"):
        TextToImageDataset(str(root), metadata_index="metadata_empty.json")


def test_qwen_preprocessor_payload_is_sanitized():
    from preprocess.processors.qwen_image import QwenImageProcessor

    processor = QwenImageProcessor()
    metadata = {
        "original_resolution": (64, 64),
        "bucket_resolution": (64, 64),
        "crop_offset": (0, 0),
        "prompt": "portable",
        "sample_id": "b" * 64,
        "source_ref": "class/b.png",
        "bucket_id": "square-64",
        "aspect_ratio": 1.0,
    }
    payload = processor.get_cache_data(
        torch.zeros(2, 2, 2),
        {"prompt_embeds": torch.zeros(1, 3, 4)},
        metadata,
    )
    assert payload["sample_id"] == metadata["sample_id"]
    assert payload["source_ref"] == metadata["source_ref"]
    assert "image_path" not in payload


def test_portable_index_writer_is_deterministic(monkeypatch, tmp_path):
    try:
        import cv2  # noqa: F401
    except ImportError:
        monkeypatch.setitem(sys.modules, "cv2", types.ModuleType("cv2"))
    from migrate_cache_manifest import migrate_cache
    from preprocess.preprocessing_multiprocess import _save_metadata_shards

    staging = tmp_path / "staging"
    (staging / "payloads").mkdir(parents=True)
    entries = []
    for character in ("b", "a"):
        source_ref = f"class/{character}.png"
        sample_id = _sample_id(source_ref, (64, 64))
        payload_ref = f"payloads/{sample_id}.pt"
        payload_path = staging / payload_ref
        torch.save(
            {
                "latent": torch.zeros(2, 2),
                "crop_offset": (0, 0),
                "prompt": character,
                "sample_id": sample_id,
                "source_ref": source_ref,
            },
            payload_path,
        )
        entries.append(
            {
                "sample_id": sample_id,
                "source_ref": source_ref,
                "cache_file": payload_ref,
                "payload_sha256": sha256_file(payload_path),
                "bucket_resolution": [64, 64],
                "original_resolution": [64, 64],
                "prompt": character,
                "bucket_id": "square-64",
                "aspect_ratio": 1.0,
                "model_type": "qwen_image",
            }
        )
    _save_metadata_shards(
        entries,
        staging,
        "qwen_image",
        "Qwen/Qwen-Image",
        "qwen_image",
        10,
        {},
        portable=True,
    )
    index = json.loads((staging / "metadata.json").read_text())
    shard = json.loads((staging / index["shards"][0]).read_text())
    assert index["schema_version"] == PREPROCESS_STAGING_SCHEMA_VERSION
    expected_ids = sorted(entry["sample_id"] for entry in entries)
    assert index["sample_ids"] == expected_ids
    assert [entry["sample_id"] for entry in shard] == expected_ids
    assert not (staging / "metadata_train.json").exists()
    assert str(staging) not in json.dumps([index, shard])
    with pytest.raises(ValueError, match=r"migrate_cache_manifest\.py"):
        load_portable_metadata(staging, "metadata.json")

    finalized = tmp_path / "finalized"
    approved = tmp_path / "approved.json"
    _write_approved_manifest(approved, expected_ids)
    migrate_cache(
        staging,
        finalized,
        approved_ids_manifest=approved,
        heldout_count=1,
    )
    assert validate_snapshot(finalized)["splits"] == {"all": 2, "train": 1, "heldout": 1}
