# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import pathlib
import shutil
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]
_FASTGEN_DIR = _REPO_ROOT / "examples" / "diffusers" / "fastgen"
if str(_FASTGEN_DIR) not in sys.path:
    sys.path.insert(0, str(_FASTGEN_DIR))

import migrate_cache_manifest as migration
from portable_cache import load_portable_metadata, ordered_sample_ids_sha256
from validate_cache_snapshot import validate_snapshot


def _write_json(path: pathlib.Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _write_approved(path: pathlib.Path, sample_ids: list[str] | tuple[str, ...]) -> str:
    digest = ordered_sample_ids_sha256(sample_ids)
    _write_json(
        path,
        {
            "schema_version": 1,
            "ordered_sample_ids": list(sample_ids),
            "ordered_sample_ids_sha256": digest,
        },
    )
    return digest


def _make_legacy_cache(
    root: pathlib.Path,
    *,
    stored_cache_root: pathlib.Path,
    stored_source_root: pathlib.Path,
    reverse_shards: bool = False,
) -> None:
    root.mkdir()
    (root / "legacy_payloads").mkdir()
    entries = []
    for index in range(4):
        relative_payload = pathlib.Path("legacy_payloads") / f"item-{index}.pt"
        source_path = stored_source_root / "class" / f"item-{index}.png"
        torch.save(
            {
                "latent": torch.full((2, 2), index, dtype=torch.float32),
                "prompt_embeds": torch.full((3, 4), index, dtype=torch.float32),
                "crop_offset": (0, 0),
                "prompt": f"prompt {index}",
                "image_path": str(source_path),
                "nested": {"source_path": str(source_path)},
            },
            root / relative_payload,
        )
        entries.append(
            {
                "cache_file": str(stored_cache_root / relative_payload),
                "image_path": str(source_path),
                "bucket_resolution": [64, 64],
                "original_resolution": [64, 64],
                "prompt": f"prompt {index}",
                "bucket_id": "square-64",
                "aspect_ratio": 1.0,
                "pixels": 4096,
                "model_type": "qwen_image",
            }
        )

    shards = [entries[:2], entries[2:]]
    if reverse_shards:
        shards.reverse()
    shard_names = []
    for index, shard in enumerate(shards):
        name = f"legacy-shard-{index}.json"
        _write_json(root / name, shard)
        shard_names.append(name)
    _write_json(root / "metadata.json", {"shards": shard_names, "total_items": 4})
    torch.save({"embed": torch.arange(12).reshape(3, 4)}, root / "legacy-negative.pt")


def _manifest_signature(root: pathlib.Path):
    index, entries = load_portable_metadata(root, "metadata.json")
    return index["sample_ids"], [
        (entry["sample_id"], entry["source_ref"], entry["cache_file"]) for entry in entries
    ]


def test_migration_is_path_independent_and_relocatable(tmp_path):
    alice = tmp_path / "alice-legacy"
    bob = tmp_path / "bob-legacy"
    alice_cache_prefix = pathlib.Path("/legacy/alice/cache")
    bob_cache_prefix = pathlib.Path("/different/bob/cache")
    alice_source_prefix = pathlib.Path("/datasets/alice/images")
    bob_source_prefix = pathlib.Path("/mnt/bob/source")
    _make_legacy_cache(
        alice,
        stored_cache_root=alice_cache_prefix,
        stored_source_root=alice_source_prefix,
    )
    _make_legacy_cache(
        bob,
        stored_cache_root=bob_cache_prefix,
        stored_source_root=bob_source_prefix,
        reverse_shards=True,
    )

    alice_output = tmp_path / "alice-portable"
    bob_output = tmp_path / "bob-portable"
    planned_ids = [
        record.sample_id
        for record in migration.plan_migration(
            alice,
            legacy_cache_root=alice_cache_prefix,
            legacy_source_root=alice_source_prefix,
        )
    ]
    approved_ids = [planned_ids[index] for index in (2, 0, 3)]
    approved = tmp_path / "approved.json"
    approved_digest = _write_approved(approved, approved_ids)
    alice_result = migration.migrate_cache(
        alice,
        alice_output,
        approved_ids_manifest=approved,
        heldout_count=1,
        expected_approved_ids_sha256=approved_digest,
        legacy_cache_root=alice_cache_prefix,
        legacy_source_root=alice_source_prefix,
        negative_embedding="legacy-negative.pt",
        shard_size=2,
    )
    bob_result = migration.migrate_cache(
        bob,
        bob_output,
        approved_ids_manifest=approved,
        heldout_count=1,
        legacy_cache_root=bob_cache_prefix,
        legacy_source_root=bob_source_prefix,
        negative_embedding="legacy-negative.pt",
        shard_size=2,
    )

    assert _manifest_signature(alice_output) == _manifest_signature(bob_output)
    alice_train = load_portable_metadata(alice_output, "metadata_train.json")[0]["sample_ids"]
    bob_train = load_portable_metadata(bob_output, "metadata_train.json")[0]["sample_ids"]
    assert alice_train == bob_train
    alice_report = validate_snapshot(alice_output)
    bob_report = validate_snapshot(bob_output)
    assert alice_report["splits"] == {"all": 3, "train": 2, "heldout": 1}
    assert bob_report["splits"] == {"all": 3, "train": 2, "heldout": 1}
    assert alice_report["snapshot_sha256"] == bob_report["snapshot_sha256"]
    assert alice_report["ordered_sample_ids_sha256"]["all"] == approved_digest
    assert alice_report["split_policy"]["approved_ordered_ids_sha256"] == approved_digest
    assert alice_result["validation"] == bob_result["validation"]
    assert alice_result["counts"] == {"source": 4, "approved": 3, "filtered": 1}
    assert set(alice_result) == {
        "schema_version",
        "record_type",
        "output_root",
        "counts",
        "validation",
    }
    assert set(alice_result["validation"]) == {
        "schema_version",
        "record_type",
        "snapshot_schema_version",
        "indexes",
        "split_policy",
        "splits",
        "ordered_sample_ids_sha256",
        "index_sha256",
        "negative_prompt_embedding",
        "unique_payloads",
        "declared_files",
        "snapshot_sha256",
    }
    validation_text = json.dumps(alice_result["validation"], sort_keys=True)
    assert str(alice_output) not in validation_text
    assert ".staging-" not in validation_text
    assert alice_result["output_root"] != bob_result["output_root"]

    cli = subprocess.run(
        [
            sys.executable,
            str(_FASTGEN_DIR / "validate_cache_snapshot.py"),
            "--cache-root",
            str(alice_output),
            "--train-index",
            "metadata_train.json",
            "--heldout-index",
            "metadata_heldout.json",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert cli.returncode == 0, cli.stderr

    portable_text = "".join(path.read_text() for path in alice_output.glob("*.json"))
    assert str(alice_cache_prefix) not in portable_text
    assert str(alice_source_prefix) not in portable_text
    for payload_path in alice_output.glob("payloads/*.pt"):
        payload = torch.load(payload_path, map_location="cpu", weights_only=True)
        assert "image_path" not in payload
        assert "source_path" not in payload["nested"]

    relocated = tmp_path / "relocated" / "cache"
    relocated.parent.mkdir()
    shutil.copytree(alice_output, relocated)
    assert _manifest_signature(relocated) == _manifest_signature(alice_output)
    assert validate_snapshot(relocated)["snapshot_sha256"] == alice_report["snapshot_sha256"]


def test_incomplete_pass_one_publishes_nothing(monkeypatch, tmp_path):
    legacy = tmp_path / "legacy"
    cache_prefix = pathlib.Path("/legacy/cache")
    source_prefix = pathlib.Path("/legacy/images")
    _make_legacy_cache(
        legacy,
        stored_cache_root=cache_prefix,
        stored_source_root=source_prefix,
    )
    (legacy / "legacy_payloads" / "item-3.pt").unlink()
    output = tmp_path / "portable"
    save_calls = []
    monkeypatch.setattr(migration.torch, "save", lambda *args, **kwargs: save_calls.append(args))

    with pytest.raises(FileNotFoundError):
        migration.migrate_cache(
            legacy,
            output,
            approved_ids_manifest=tmp_path / "unused-approved.json",
            heldout_count=1,
            legacy_cache_root=cache_prefix,
            legacy_source_root=source_prefix,
        )
    assert not output.exists()
    assert not list(tmp_path.glob(".portable.staging-*"))
    assert save_calls == []


def test_approved_artifact_failures_and_final_validation_publish_nothing(
    monkeypatch, tmp_path
) -> None:
    legacy = tmp_path / "legacy"
    cache_prefix = pathlib.Path("/legacy/cache")
    source_prefix = pathlib.Path("/legacy/images")
    _make_legacy_cache(
        legacy,
        stored_cache_root=cache_prefix,
        stored_source_root=source_prefix,
    )
    records = migration.plan_migration(
        legacy,
        legacy_cache_root=cache_prefix,
        legacy_source_root=source_prefix,
    )
    sample_ids = [record.sample_id for record in records]
    approved = tmp_path / "approved.json"
    digest = _write_approved(approved, sample_ids)

    with pytest.raises(ValueError, match="expected approved-ID"):
        migration.migrate_cache(
            legacy,
            tmp_path / "bad-external",
            approved_ids_manifest=approved,
            expected_approved_ids_sha256="f" * 64,
            heldout_count=1,
            legacy_cache_root=cache_prefix,
            legacy_source_root=source_prefix,
        )
    unknown = tmp_path / "unknown.json"
    _write_approved(unknown, [*sample_ids, "unknown"])
    with pytest.raises(ValueError, match="unknown sample IDs"):
        migration.migrate_cache(
            legacy,
            tmp_path / "unknown-output",
            approved_ids_manifest=unknown,
            heldout_count=1,
            legacy_cache_root=cache_prefix,
            legacy_source_root=source_prefix,
        )
    symlink = tmp_path / "approved-symlink.json"
    symlink.symlink_to(approved)
    with pytest.raises(ValueError, match="symlink"):
        migration.migrate_cache(
            legacy,
            tmp_path / "symlink-output",
            approved_ids_manifest=symlink,
            heldout_count=1,
            legacy_cache_root=cache_prefix,
            legacy_source_root=source_prefix,
        )

    monkeypatch.setattr(
        migration,
        "validate_snapshot",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("injected final validation")),
    )
    output = tmp_path / "validation-output"
    with pytest.raises(ValueError, match="injected final validation"):
        migration.migrate_cache(
            legacy,
            output,
            approved_ids_manifest=approved,
            expected_approved_ids_sha256=digest,
            heldout_count=1,
            legacy_cache_root=cache_prefix,
            legacy_source_root=source_prefix,
        )
    assert not output.exists()
    assert not list(tmp_path.glob(".validation-output.staging-*"))


def test_migration_cli_removes_seed_and_requires_approved_manifest() -> None:
    cli = subprocess.run(
        [sys.executable, str(_FASTGEN_DIR / "migrate_cache_manifest.py"), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert cli.returncode == 0, cli.stderr
    assert "--approved-ids-manifest" in cli.stdout
    assert "--split-seed" not in cli.stdout


def test_migration_rejects_finalized_and_non_strict_source_json(tmp_path) -> None:
    cache_prefix = pathlib.Path("/legacy/cache")
    source_prefix = pathlib.Path("/legacy/images")

    finalized = tmp_path / "finalized-source"
    _make_legacy_cache(
        finalized,
        stored_cache_root=cache_prefix,
        stored_source_root=source_prefix,
    )
    index_path = finalized / "metadata.json"
    index = json.loads(index_path.read_text())
    index["schema_version"] = 2
    _write_json(index_path, index)
    with pytest.raises(ValueError, match="schema_version is unsupported"):
        migration.plan_migration(
            finalized,
            legacy_cache_root=cache_prefix,
            legacy_source_root=source_prefix,
        )

    duplicate = tmp_path / "duplicate-source"
    _make_legacy_cache(
        duplicate,
        stored_cache_root=cache_prefix,
        stored_source_root=source_prefix,
    )
    (duplicate / "metadata.json").write_text(
        '{"shards":["legacy-shard-0.json"],"shards":["legacy-shard-1.json"]}'
    )
    with pytest.raises(ValueError, match="duplicate key"):
        migration.plan_migration(
            duplicate,
            legacy_cache_root=cache_prefix,
            legacy_source_root=source_prefix,
        )

    nonfinite = tmp_path / "nonfinite-source"
    _make_legacy_cache(
        nonfinite,
        stored_cache_root=cache_prefix,
        stored_source_root=source_prefix,
    )
    (nonfinite / "metadata.json").write_text('{"shards":[],"total_items":NaN}')
    with pytest.raises(ValueError, match="non-standard constant"):
        migration.plan_migration(
            nonfinite,
            legacy_cache_root=cache_prefix,
            legacy_source_root=source_prefix,
        )


def test_changed_source_after_frozen_plan_cleans_staging(monkeypatch, tmp_path):
    legacy = tmp_path / "legacy"
    cache_prefix = pathlib.Path("/legacy/cache")
    source_prefix = pathlib.Path("/legacy/images")
    _make_legacy_cache(
        legacy,
        stored_cache_root=cache_prefix,
        stored_source_root=source_prefix,
    )
    frozen = migration.plan_migration(
        legacy,
        legacy_cache_root=cache_prefix,
        legacy_source_root=source_prefix,
    )
    changed = frozen[-1].source_payload
    original_plan = migration.plan_migration

    def _return_frozen(*args, **kwargs):
        changed.write_bytes(b"changed after pass one")
        return frozen

    monkeypatch.setattr(migration, "plan_migration", _return_frozen)
    output = tmp_path / "portable"
    approved = tmp_path / "approved.json"
    _write_approved(approved, [record.sample_id for record in frozen])
    with pytest.raises(RuntimeError, match="changed after pass 1"):
        migration.migrate_cache(
            legacy,
            output,
            approved_ids_manifest=approved,
            heldout_count=1,
            legacy_cache_root=cache_prefix,
            legacy_source_root=source_prefix,
        )
    monkeypatch.setattr(migration, "plan_migration", original_plan)
    assert not output.exists()
    assert not list(tmp_path.glob(".portable.staging-*"))


def test_invalid_legacy_reference_fails_before_publish(tmp_path):
    legacy = tmp_path / "legacy"
    cache_prefix = pathlib.Path("/legacy/cache")
    source_prefix = pathlib.Path("/legacy/images")
    _make_legacy_cache(
        legacy,
        stored_cache_root=cache_prefix,
        stored_source_root=source_prefix,
    )
    shard_path = legacy / "legacy-shard-0.json"
    shard = json.loads(shard_path.read_text())
    shard[0]["cache_file"] = "/other/private/cache.pt"
    _write_json(shard_path, shard)

    output = tmp_path / "portable"
    with pytest.raises(ValueError, match="outside the declared legacy prefix"):
        migration.migrate_cache(
            legacy,
            output,
            approved_ids_manifest=tmp_path / "unused-approved.json",
            heldout_count=1,
            legacy_cache_root=cache_prefix,
            legacy_source_root=source_prefix,
        )
    assert not output.exists()


def test_incomplete_source_counts_and_rank_indices_publish_nothing(tmp_path):
    legacy = tmp_path / "legacy"
    cache_prefix = pathlib.Path("/legacy/cache")
    source_prefix = pathlib.Path("/legacy/images")
    _make_legacy_cache(
        legacy,
        stored_cache_root=cache_prefix,
        stored_source_root=source_prefix,
    )
    index_path = legacy / "metadata.json"
    index = json.loads(index_path.read_text())
    index["num_shards"] = 3
    _write_json(index_path, index)
    output = tmp_path / "invalid-count-output"
    with pytest.raises(ValueError, match="num_shards"):
        migration.migrate_cache(
            legacy,
            output,
            approved_ids_manifest=tmp_path / "unused-approved.json",
            heldout_count=1,
            legacy_cache_root=cache_prefix,
            legacy_source_root=source_prefix,
        )
    assert not output.exists()
    assert not list(tmp_path.glob(".invalid-count-output.staging-*"))

    shards = index["shards"]
    for rank, shard in enumerate(shards):
        _write_json(
            legacy / f"metadata_r{rank:02d}.json",
            {
                "shards": [shard],
                "num_shards": 1,
                "total_items": 2,
                "shard_rank": rank,
                "shard_world": 2,
            },
        )
    incomplete_output = tmp_path / "incomplete-ranks-output"
    with pytest.raises(ValueError, match="incomplete or inconsistent"):
        migration.migrate_cache(
            legacy,
            incomplete_output,
            approved_ids_manifest=tmp_path / "unused-approved.json",
            source_index="metadata_r00.json",
            heldout_count=1,
            legacy_cache_root=cache_prefix,
            legacy_source_root=source_prefix,
        )
    assert not incomplete_output.exists()

    complete_output = tmp_path / "complete-ranks-output"
    complete_records = migration.plan_migration(
        legacy,
        source_index=("metadata_r00.json", "metadata_r01.json"),
        legacy_cache_root=cache_prefix,
        legacy_source_root=source_prefix,
    )
    complete_approved = tmp_path / "complete-approved.json"
    _write_approved(complete_approved, [record.sample_id for record in complete_records])
    migration.migrate_cache(
        legacy,
        complete_output,
        approved_ids_manifest=complete_approved,
        source_index=("metadata_r00.json", "metadata_r01.json"),
        heldout_count=1,
        legacy_cache_root=cache_prefix,
        legacy_source_root=source_prefix,
    )
    assert validate_snapshot(complete_output)["splits"] == {"all": 4, "train": 3, "heldout": 1}


def test_migration_path_audit_allows_prompt_commands_but_rejects_set_paths(tmp_path):
    legacy = tmp_path / "legacy"
    cache_prefix = pathlib.Path("/legacy/cache")
    source_prefix = pathlib.Path("/legacy/images")
    _make_legacy_cache(
        legacy,
        stored_cache_root=cache_prefix,
        stored_source_root=source_prefix,
    )
    shard_path = legacy / "legacy-shard-0.json"
    shard = json.loads(shard_path.read_text())
    shard[0]["prompt"] = "/imagine a cat"
    _write_json(shard_path, shard)
    payload_path = legacy / "legacy_payloads" / "item-0.pt"
    payload = torch.load(payload_path, map_location="cpu", weights_only=True)
    payload["prompt"] = "/imagine a cat"
    torch.save(payload, payload_path)
    migration.plan_migration(
        legacy,
        legacy_cache_root=cache_prefix,
        legacy_source_root=source_prefix,
    )

    payload["paths"] = {"/home/alice/private.png"}
    torch.save(payload, payload_path)
    output = tmp_path / "portable"
    with pytest.raises(ValueError, match="absolute path"):
        migration.migrate_cache(
            legacy,
            output,
            approved_ids_manifest=tmp_path / "unused-approved.json",
            heldout_count=1,
            legacy_cache_root=cache_prefix,
            legacy_source_root=source_prefix,
        )
    assert not output.exists()
