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
    audit_no_absolute_paths,
    load_portable_metadata,
    resolve_cache_root,
    resolve_negative_embedding,
    sha256_file,
)
from validate_cache_snapshot import validate_snapshot


def _sample_id(source_ref: str, resolution: tuple[int, int]) -> str:
    identity = (
        f"modelopt-fastgen-sample-v1\0qwen_image\0{source_ref}\0{resolution[0]}x{resolution[1]}"
    )
    return hashlib.sha256(identity.encode()).hexdigest()


def _write_json(path: pathlib.Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


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
    splits = {
        "all": all_ids,
        "train": [all_ids[2], all_ids[0], all_ids[1]],
        "heldout": [all_ids[3]],
    }
    for split, ids in splits.items():
        name = "metadata.json" if split == "all" else f"metadata_{split}.json"
        _write_json(
            root / name,
            {
                "schema_version": 1,
                "split": split,
                "total_items": len(ids),
                "num_shards": 1,
                "shards": ["metadata_shard_s0000.json"],
                "sample_ids": ids,
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
    samples = [dataset[1], dataset[2]]
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
            "schema_version": 1,
            "split": "empty",
            "total_items": 0,
            "shards": ["metadata_shard_s0000.json"],
            "sample_ids": [],
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
    expected_ids = sorted(entry["sample_id"] for entry in entries)
    assert index["sample_ids"] == expected_ids
    assert [entry["sample_id"] for entry in shard] == expected_ids
    assert not (staging / "metadata_train.json").exists()
    assert str(staging) not in json.dumps([index, shard])

    finalized = tmp_path / "finalized"
    migrate_cache(staging, finalized, heldout_count=1)
    assert validate_snapshot(finalized)["splits"] == {"all": 2, "train": 1, "heldout": 1}
