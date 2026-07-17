# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import os
from pathlib import Path

import pytest

from examples.puzzletron.import_campaign_artifacts import main
from examples.puzzletron.main import _completion_is_valid
from examples.puzzletron.inventory_campaign_artifacts import inventory_campaign_artifacts, write_inventory
from modelopt.torch.puzzletron.artifact_import import ArtifactImportError, import_campaign_artifacts
from modelopt.torch.puzzletron.identity import stable_hash
from modelopt.torch.puzzletron.manifest import StageManifest


_SOURCE_CONFIG = {
    "model": {"source": "/campaign/sorted-teacher", "family": "qwen"},
    "data": {"identity": "data-v1"},
    "search_space": {"identity": "search-v1"},
    "activation": {},
    "depth": {},
    "vllm_stats": {},
    "scoring": {},
}


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _score_index_entry(result_path: Path, request_id: str) -> dict:
    return {
        "request_id": request_id,
        "source_result_path": str(result_path),
        "metrics": {"loss": 1.0},
        "metadata": {
            "result_kind": "replace_one_block",
            "candidate_score_identity": f"score:{request_id}",
        },
    }


def _complete_campaign(root: Path) -> None:
    for stage in ("activation", "depth", "vllm_stats", "scoring", "build_library"):
        _write_json(
            root / f"manifests/{stage}.json",
            StageManifest(stage=stage, status="success", config=_SOURCE_CONFIG).to_dict(),
        )

    activation = root / "pruning/pruning_scores/automodel/full"
    _write_json(activation / "activation_passes_manifest.json", {"passes": ["attention", "ffn"]})
    for name in ("attention", "ffn"):
        path = activation / name / "rank_0000.pth"
        path.parent.mkdir(parents=True)
        path.write_bytes(f"activation-{name}".encode())

    selected = [{"layer_idx": 0, "kind": "attention"}]
    scenarios = []
    for length in range(2):
        scenario = {
            "parent_checkpoint_identity": "parent-v1",
            "hidden_width": 4096,
            "removals": selected[:length],
            "data_identity": "data-v1",
            "evaluator_revision": "depth-v1",
            "metric": "lm_loss",
            "granularity": "subblock",
        }
        scenarios.append(
            {**scenario, "scenario_id": stable_hash(scenario, prefix="depth_scenario")}
        )
    _write_json(
        root / "depth/iterative/trajectory.json",
        {
            "status": "complete",
            "max_removals": 1,
            "selected": selected,
            "scenarios": scenarios,
            "source_checkpoint_dir": "/campaign/sorted-teacher",
            "data_identity": "data-v1",
            "model_identity": "model-v1",
            "granularity": "subblock",
        },
    )

    for index in range(2):
        _write_json(
            root / f"runtime_cache/shards/runtime-spec/shard_{index:04d}.json",
            {
                "spec_identity": "runtime-spec",
                "shard_index": index,
                "shard_count": 2,
                "results": {str(index): {"total_ms": index + 1.0, "prefill_ms": 0.5}},
            },
        )
        _write_json(
            root / f"runtime_cache/shards/runtime-spec/shard_{index:04d}.done",
            {"count": 1},
        )
    _write_json(
        root / "subblock_stats.json",
        {
            "runtime_decomposition": {"method": "exact"},
            "block_runtime_records": [{"index": 0}, {"index": 1}],
            "spec_identity": "runtime-spec",
            "source_checkpoint_dir": "/campaign/sorted-teacher",
            "data_identity": "data-v1",
            "model_identity": "model-v1",
            "granularity": "block",
        },
    )

    _write_json(
        root / "subblock_replacement_manifest.json",
        {
            "mode": "replace_one_subblock",
            "subblock_solution_count": 2,
            "teacher_dir": "/campaign/sorted-teacher",
            "data_identity": "data-v1",
            "model_identity": "model-v1",
            "full_search_space_preserved": True,
        },
    )
    for index in range(2):
        _write_json(
            root / f"single_subblock_replacement_solutions--validation/solution_{index}.json",
            {"i_solution": index},
        )
    _write_json(
        root / "rpc_eval/replace_one_block/score_index.json",
        {
            "scores": [
                _score_index_entry(
                    root / "single_subblock_replacement_solutions--validation/solution_0.json",
                    "score-0",
                ),
                _score_index_entry(
                    root / "single_subblock_replacement_solutions--validation/solution_1.json",
                    "score-1",
                ),
            ]
        },
    )
    observations = root / "artifacts/bypass/dp_observations.jsonl"
    observations.parent.mkdir(parents=True)
    observations.write_text('{"step": 1}\n', encoding="utf-8")


def _receipt(source: Path, path: Path) -> Path:
    write_inventory(inventory_campaign_artifacts(source), path)
    newest_source = max(item.stat().st_mtime_ns for item in source.rglob("*") if item.is_file())
    os.utime(path, ns=(newest_source + 1_000_000, newest_source + 1_000_000))
    return path


def _setup(tmp_path: Path) -> tuple[Path, Path, Path]:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    receipt = tmp_path / "receipt.json"
    _complete_campaign(source)
    return source, destination, _receipt(source, receipt)


def _target_config(destination: Path, path: Path) -> tuple[dict, Path]:
    config = {
        "puzzle_dir": str(destination),
        "experiment": {"dir": str(destination)},
        "model": {"source": "/campaign/sorted-teacher", "family": "qwen"},
        "data": {"identity": "data-v1"},
        "search_space": {"identity": "search-v1"},
        "activation": {},
        "depth": {},
        "vllm_stats": {},
        "scoring": {},
    }
    _write_json(path, config)
    return config, path


def _relocated_semantic_setup(tmp_path: Path) -> tuple[Path, Path, Path, dict, Path]:
    source = tmp_path / "campaigns/full_pipeline"
    destination = tmp_path / "campaigns/sanity_check"
    external_dataset = Path.cwd() / "tests/fixtures/puzzletron/dataset.jsonl"
    source_config = {
        "model": {"source": "org/model", "revision": "main"},
        "data": {
            "path": str(external_dataset),
            "sequence_length": 16384,
            "calibration": {
                "path": str(external_dataset),
                "num_samples": 65536,
                "micro_batch_size": 1,
            },
            "scoring": {"num_samples": 128, "micro_batch_size": 1},
        },
        "search_space": {"axes": {"hidden_width": {"values": [4096, 3584]}}},
        "embedding_pruning": {
            "enabled": False,
            "widths": [4096, 3840, 3584],
            "alignment": 256,
            "cycle_widths": True,
            "ranking": {"method": "activation", "tie_break": "stable"},
        },
        "granularity": "subblock",
        "activation": {
            "enabled": True,
            "output_dir": str(source / "pruning/pruning_scores/automodel/full"),
            "micro_batch_size": 1,
            "automodel": {
                "parallel": {"tp": 1, "cp": 4, "pp": 2, "ep": 1}
            },
            "runtime": {
                "execution": "distributed",
                "sharding": {"world_size": 8},
                "topology": {"nodes": 1, "gpus_per_node": 8},
            },
        },
        "depth": {},
        "vllm_stats": {},
        "scoring": {},
    }
    _complete_campaign(source)
    for stage in ("activation", "depth", "vllm_stats", "scoring", "build_library"):
        _write_json(
            source / f"manifests/{stage}.json",
            StageManifest(stage=stage, status="success", config=source_config).to_dict(),
        )
    receipt = _receipt(source, tmp_path / "receipt.json")

    target_config = json.loads(json.dumps(source_config))
    target_config["puzzle_dir"] = str(destination)
    target_config["data"]["path"] = "tests/fixtures/puzzletron/dataset.jsonl"
    target_config["data"]["calibration"]["path"] = "tests/fixtures/puzzletron/dataset.jsonl"
    target_config["data"]["calibration"]["micro_batch_size"] = 8
    target_config["data"]["scoring"]["micro_batch_size"] = 8
    target_config["embedding_pruning"] = {
        "widths": [4096, 3840, 3584],
        "alignment": 256,
    }
    target_config["activation"] = {
        "enabled": False,
        "output_dir": str(destination / "pruning/pruning_scores/automodel/full"),
        "micro_batch_size": 8,
        "automodel": {
            "parallel": {
                "tp": 1,
                "cp": 1,
                "pp": 1,
                "ep": 1,
                "dp_shard": 1,
                "dp_replicate": 8,
            }
        },
        "runtime": {
            "execution": "inline",
            "sharding": {"world_size": 1},
            "topology": {"nodes": 1, "gpus_per_node": 1},
        },
    }
    config_path = tmp_path / "target-config.json"
    _write_json(config_path, target_config)
    return source, destination, receipt, target_config, config_path


def test_successful_dry_run_validates_and_writes_nothing(tmp_path):
    source, destination, receipt = _setup(tmp_path)
    source_before = {path: path.read_bytes() for path in source.rglob("*") if path.is_file()}

    result = import_campaign_artifacts(source, destination, receipt, dry_run=True)

    assert result["status"] == "planned"
    assert tuple(result["bundles"]) == (
        "activation",
        "depth",
        "vllm_stats",
        "scoring",
        "bypass_evidence",
    )
    assert [item["destination"] for item in result["bundles"]["vllm_stats"]["files"]] == [
        "runtime_cache/shards/runtime-spec/shard_0000.done",
        "runtime_cache/shards/runtime-spec/shard_0000.json",
        "runtime_cache/shards/runtime-spec/shard_0001.done",
        "runtime_cache/shards/runtime-spec/shard_0001.json",
        "subblock_stats.json",
    ]
    assert not destination.exists()
    assert {path: path.read_bytes() for path in source.rglob("*") if path.is_file()} == source_before


def test_successful_import_preserves_canonical_files_and_publishes_imported_manifests(tmp_path):
    source, destination, receipt = _setup(tmp_path)

    result = import_campaign_artifacts(source, destination, receipt)

    assert result["status"] == "imported"
    canonical = (
        "pruning/pruning_scores/automodel/full/attention/rank_0000.pth",
        "depth/iterative/trajectory.json",
        "runtime_cache/shards/runtime-spec/shard_0001.json",
        "subblock_stats.json",
        "subblock_replacement_manifest.json",
        "single_subblock_replacement_solutions--validation/solution_1.json",
        "rpc_eval/replace_one_block/score_index.json",
        "artifacts/bypass/dp_observations.jsonl",
    )
    for relative in canonical:
        imported = destination / relative
        assert imported.read_bytes() == (source / relative).read_bytes()
        assert not imported.is_symlink()
    for stage in ("activation", "depth", "vllm_stats", "scoring"):
        manifest = json.loads((destination / f"manifests/{stage}.json").read_text())
        assert manifest["status"] == "imported"
        assert manifest["source_campaign"] == str(source)
        assert manifest["receipt_identity"].startswith("sha256:")
        assert manifest["compatibility"]
        assert manifest["counts"]["observed"] == manifest["counts"]["expected"]
        assert manifest["output_inventory"]
        assert all(item["sha256"] for item in manifest["output_inventory"])


@pytest.mark.parametrize("state", ["partial", "incompatible"])
def test_incomplete_or_incompatible_receipt_is_rejected(tmp_path, state):
    source, destination, receipt = _setup(tmp_path)
    payload = json.loads(receipt.read_text())
    payload["state"] = state
    _write_json(receipt, payload)

    with pytest.raises(ArtifactImportError, match="receipt must be complete"):
        import_campaign_artifacts(source, destination, receipt)

    assert not destination.exists()


def test_source_mutation_after_receipt_is_rejected(tmp_path):
    source, destination, receipt = _setup(tmp_path)
    mutated = source / "depth/iterative/trajectory.json"
    payload = json.loads(mutated.read_text())
    payload["scenarios"][0]["scenario_id"] = "mutated"
    _write_json(mutated, payload)
    receipt_time = receipt.stat().st_mtime_ns
    os.utime(mutated, ns=(receipt_time + 1_000_000, receipt_time + 1_000_000))

    with pytest.raises(ArtifactImportError, match="source changed after receipt"):
        import_campaign_artifacts(source, destination, receipt)

    assert not destination.exists()


def test_interrupted_temporary_copy_never_appears_complete_and_is_retryable(tmp_path, monkeypatch):
    source, destination, receipt = _setup(tmp_path)
    import modelopt.torch.puzzletron.artifact_import as artifact_import

    original = artifact_import.shutil.copy2
    calls = 0

    def interrupt(source_path, destination_path):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated interruption")
        return original(source_path, destination_path)

    monkeypatch.setattr(artifact_import.shutil, "copy2", interrupt)
    with pytest.raises(ArtifactImportError, match="copy failed"):
        import_campaign_artifacts(source, destination, receipt, bundles=("activation",))

    assert not destination.exists()

    monkeypatch.setattr(artifact_import.shutil, "copy2", original)
    result = import_campaign_artifacts(source, destination, receipt, bundles=("activation",))
    assert result["status"] == "imported"
    assert json.loads((destination / "manifests/activation.json").read_text())["status"] == "imported"


def test_whole_campaign_copy_interruption_leaks_no_canonical_root(tmp_path, monkeypatch):
    source, destination, receipt = _setup(tmp_path)
    import modelopt.torch.puzzletron.artifact_import as artifact_import

    original = artifact_import.shutil.copy2
    calls = 0

    def interrupt(source_path, destination_path):
        nonlocal calls
        calls += 1
        if calls == 4:
            raise OSError("simulated whole-campaign copy interruption")
        return original(source_path, destination_path)

    monkeypatch.setattr(artifact_import.shutil, "copy2", interrupt)

    with pytest.raises(ArtifactImportError, match="copy interruption|copy failed"):
        import_campaign_artifacts(source, destination, receipt)

    assert not destination.exists()


def test_whole_campaign_publication_interruption_leaks_no_canonical_root(tmp_path, monkeypatch):
    source, destination, receipt = _setup(tmp_path)
    import modelopt.torch.puzzletron.artifact_import as artifact_import

    def interrupt(*_args, **_kwargs):
        raise OSError("simulated publication interruption")

    monkeypatch.setattr(artifact_import, "_rename_noreplace", interrupt)

    with pytest.raises(ArtifactImportError, match="publication interruption|publish failed"):
        import_campaign_artifacts(source, destination, receipt)

    assert not destination.exists()


def test_identical_retry_is_a_noop(tmp_path):
    source, destination, receipt = _setup(tmp_path)
    import_campaign_artifacts(source, destination, receipt)
    before = {path: path.stat().st_mtime_ns for path in destination.rglob("*") if path.is_file()}

    result = import_campaign_artifacts(source, destination, receipt)

    assert result["status"] == "noop"
    assert {path: path.stat().st_mtime_ns for path in destination.rglob("*") if path.is_file()} == before


@pytest.mark.parametrize("symlink_kind", ("manifest", "ancestor"))
def test_identical_retry_rejects_symlinked_import_manifest_boundary(
    tmp_path, symlink_kind
):
    source, destination, receipt = _setup(tmp_path)
    import_campaign_artifacts(source, destination, receipt, bundles=("activation",))
    manifest = destination / "manifests/imports/campaign_artifacts.json"

    if symlink_kind == "manifest":
        external = tmp_path / "external-campaign-artifacts.json"
        manifest.rename(external)
        manifest.symlink_to(external)
    else:
        imports_dir = manifest.parent
        external = tmp_path / "external-imports"
        imports_dir.rename(external)
        imports_dir.symlink_to(external, target_is_directory=True)

    with pytest.raises(ArtifactImportError, match="symlink"):
        import_campaign_artifacts(source, destination, receipt, bundles=("activation",))


def test_conflicting_destination_is_rejected(tmp_path):
    source, destination, receipt = _setup(tmp_path)
    conflict = destination / "depth/iterative/trajectory.json"
    conflict.parent.mkdir(parents=True)
    conflict.write_text("conflict\n", encoding="utf-8")

    with pytest.raises(ArtifactImportError, match="conflicting destination"):
        import_campaign_artifacts(source, destination, receipt, bundles=("depth",))

    assert conflict.read_text() == "conflict\n"
    assert not (destination / "manifests/depth.json").exists()


def test_preexisting_empty_destination_is_a_conflict(tmp_path):
    source, destination, receipt = _setup(tmp_path)
    destination.mkdir()

    with pytest.raises(ArtifactImportError, match="conflicting destination"):
        import_campaign_artifacts(source, destination, receipt)

    assert not any(destination.iterdir())


def test_destination_creation_race_is_rejected_without_overwrite(tmp_path, monkeypatch):
    source, destination, receipt = _setup(tmp_path)
    import modelopt.torch.puzzletron.artifact_import as artifact_import

    original = artifact_import._rename_noreplace

    def race(source_path, destination_path):
        raced = Path(destination_path)
        raced.mkdir(parents=True)
        (raced / "racer.txt").write_text("owned by racer\n", encoding="utf-8")
        return original(source_path, destination_path)

    monkeypatch.setattr(artifact_import, "_rename_noreplace", race)

    with pytest.raises(ArtifactImportError, match="conflicting destination|destination race"):
        import_campaign_artifacts(source, destination, receipt)

    assert (destination / "racer.txt").read_text(encoding="utf-8") == "owned by racer\n"
    assert not (destination / "manifests/activation.json").exists()


def test_empty_destination_creation_race_is_rejected_without_replacement(tmp_path, monkeypatch):
    source, destination, receipt = _setup(tmp_path)
    import modelopt.torch.puzzletron.artifact_import as artifact_import

    original = artifact_import._rename_noreplace

    def race(source_path, destination_path):
        Path(destination_path).mkdir(parents=True)
        return original(source_path, destination_path)

    monkeypatch.setattr(artifact_import, "_rename_noreplace", race)

    with pytest.raises(ArtifactImportError, match="conflicting destination|destination race"):
        import_campaign_artifacts(source, destination, receipt)

    assert destination.is_dir()
    assert not any(destination.iterdir())


def test_atomic_root_rename_is_the_last_fallible_operation(tmp_path, monkeypatch):
    source, destination, receipt = _setup(tmp_path)
    import modelopt.torch.puzzletron.artifact_import as artifact_import

    events = []
    original_validate = artifact_import._validate_payload_files
    original_publish = artifact_import._publish_campaign_root

    def validate(*args, **kwargs):
        events.append("validate")
        return original_validate(*args, **kwargs)

    def publish(*args, **kwargs):
        events.append("publish")
        return original_publish(*args, **kwargs)

    monkeypatch.setattr(artifact_import, "_validate_payload_files", validate)
    monkeypatch.setattr(artifact_import, "_publish_campaign_root", publish)

    import_campaign_artifacts(source, destination, receipt)

    assert events[-1] == "publish"


def test_destination_symlink_is_rejected_even_when_its_target_is_identical(tmp_path):
    source, destination, receipt = _setup(tmp_path)
    destination_file = destination / "depth/iterative/trajectory.json"
    destination_file.parent.mkdir(parents=True)
    destination_file.symlink_to(source / "depth/iterative/trajectory.json")

    with pytest.raises(ArtifactImportError, match="conflicting destination"):
        import_campaign_artifacts(source, destination, receipt, bundles=("depth",))

    assert destination_file.is_symlink()
    assert not (destination / "manifests/depth.json").exists()


def test_source_root_symlink_ancestor_is_rejected(tmp_path):
    source, destination, receipt = _setup(tmp_path)
    source_link = tmp_path / "source-link"
    source_link.symlink_to(source, target_is_directory=True)

    with pytest.raises(ArtifactImportError, match="symlink"):
        import_campaign_artifacts(source_link, destination, receipt)

    assert not destination.exists()


def test_source_artifact_symlink_ancestor_is_rejected(tmp_path):
    source, destination, receipt = _setup(tmp_path)
    original_depth = source / "depth"
    relocated_depth = source / "relocated-depth"
    original_depth.rename(relocated_depth)
    original_depth.symlink_to(relocated_depth, target_is_directory=True)

    with pytest.raises(ArtifactImportError, match="symlink"):
        import_campaign_artifacts(source, destination, receipt, bundles=("depth",))

    assert not destination.exists()


def test_destination_root_symlink_ancestor_is_rejected(tmp_path):
    source, _, receipt = _setup(tmp_path)
    real_parent = tmp_path / "real-destinations"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked-destinations"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    destination = linked_parent / "campaign"

    with pytest.raises(ArtifactImportError, match="symlink"):
        import_campaign_artifacts(source, destination, receipt)

    assert not (real_parent / "campaign").exists()


def test_source_duplicate_is_recomputed_instead_of_trusting_receipt_counts(tmp_path):
    source, destination, receipt = _setup(tmp_path)
    duplicate = source / "single_subblock_replacement_solutions--validation/solution_9999.json"
    _write_json(duplicate, {"i_solution": 0})
    receipt_time = receipt.stat().st_mtime_ns
    os.utime(duplicate, ns=(receipt_time - 1_000_000, receipt_time - 1_000_000))

    with pytest.raises(ArtifactImportError, match="duplicate"):
        import_campaign_artifacts(source, destination, receipt, bundles=("scoring",))

    assert not destination.exists()


def test_source_semantics_are_recomputed_instead_of_trusting_receipt_metadata(tmp_path):
    source, destination, receipt = _setup(tmp_path)
    aggregate = source / "subblock_stats.json"
    payload = json.loads(aggregate.read_text())
    payload["source_checkpoint_dir"] = "/campaign/incompatible-teacher"
    _write_json(aggregate, payload)
    receipt_time = receipt.stat().st_mtime_ns
    os.utime(aggregate, ns=(receipt_time - 1_000_000, receipt_time - 1_000_000))

    with pytest.raises(ArtifactImportError, match="semantic|current source inventory"):
        import_campaign_artifacts(source, destination, receipt, bundles=("vllm_stats",))

    assert not destination.exists()


def test_touched_receipt_does_not_change_its_content_identity(tmp_path):
    source, destination, receipt = _setup(tmp_path)
    identity = json.loads(receipt.read_text())["receipt_identity"]
    touched = receipt.stat().st_mtime_ns + 10_000_000
    os.utime(receipt, ns=(touched, touched))

    result = import_campaign_artifacts(source, destination, receipt, dry_run=True)

    assert result["receipt_identity"] == identity
    assert not destination.exists()


def test_touched_receipt_cannot_hide_backdated_source_mutation(tmp_path):
    source, destination, receipt = _setup(tmp_path)
    trajectory = source / "depth/iterative/trajectory.json"
    timestamps = (trajectory.stat().st_atime_ns, trajectory.stat().st_mtime_ns)
    payload = json.loads(trajectory.read_text())
    payload["scenarios"][0]["scenario_id"] = "same-mtime-mutation"
    _write_json(trajectory, payload)
    os.utime(trajectory, ns=timestamps)
    touched = receipt.stat().st_mtime_ns + 10_000_000
    os.utime(receipt, ns=(touched, touched))

    with pytest.raises(ArtifactImportError, match="receipt identity|source inventory|source mutation"):
        import_campaign_artifacts(source, destination, receipt, bundles=("depth",))

    assert not destination.exists()


def test_bypass_evidence_does_not_complete_the_bypass_execution_node(tmp_path):
    source, destination, receipt = _setup(tmp_path)

    import_campaign_artifacts(source, destination, receipt, bundles=("bypass_evidence",))

    evidence = json.loads((destination / "manifests/imports/bypass_evidence.json").read_text())
    assert evidence["status"] == "evidence"
    assert evidence["report_only"] is True
    assert not (destination / "manifests/bypass.json").exists()
    assert not (destination / "manifests/completions/bypass.json").exists()


def test_imported_vllm_stats_is_recognized_complete_by_main(tmp_path):
    stage = "vllm_stats"
    source, destination, receipt = _setup(tmp_path)
    config, config_path = _target_config(destination, tmp_path / "target-config.json")

    import_campaign_artifacts(
        source,
        destination,
        receipt,
        bundles=(stage,),
        target_config_path=config_path,
    )

    assert _completion_is_valid(config, config_path, stage)
    marker = json.loads((destination / f"manifests/completions/{stage}.json").read_text())
    assert marker["completion_kind"] == "imported"
    assert marker["receipt_identity"] == json.loads(receipt.read_text())["receipt_identity"]
    changed_config = {**config, stage: {"semantic-change": True}}
    assert not _completion_is_valid(changed_config, config_path, stage)


def test_deleted_imported_vllm_aggregate_stales_completion(tmp_path):
    source, destination, receipt = _setup(tmp_path)
    config, config_path = _target_config(destination, tmp_path / "target-config.json")

    import_campaign_artifacts(
        source,
        destination,
        receipt,
        bundles=("vllm_stats",),
        target_config_path=config_path,
    )

    assert _completion_is_valid(config, config_path, "vllm_stats")
    (destination / "subblock_stats.json").unlink()
    assert not _completion_is_valid(config, config_path, "vllm_stats")


@pytest.mark.parametrize("section", ("model", "data", "search_space"))
def test_target_semantic_config_must_match_source_before_completion_publication(
    tmp_path, section
):
    source, destination, receipt = _setup(tmp_path)
    config, config_path = _target_config(destination, tmp_path / "target-config.json")
    config[section] = {"identity": f"different-{section}"}
    _write_json(config_path, config)

    with pytest.raises(ArtifactImportError, match="semantic config|compatibility"):
        import_campaign_artifacts(
            source,
            destination,
            receipt,
            bundles=("activation",),
            target_config_path=config_path,
        )

    assert not destination.exists()


def test_relocated_sibling_accepts_normalized_import_compatible_semantics(tmp_path):
    source, destination, receipt, _, config_path = _relocated_semantic_setup(tmp_path)

    result = import_campaign_artifacts(
        source,
        destination,
        receipt,
        bundles=("activation",),
        target_config_path=config_path,
        dry_run=True,
    )

    assert result["status"] == "planned"
    assert not destination.exists()


@pytest.mark.parametrize(
    ("mutation", "expected"),
    (
        (lambda config: config["model"].update(revision="different"), "model"),
        (
            lambda config: config["data"].update(path="tests/fixtures/puzzletron/other.jsonl"),
            "data",
        ),
        (
            lambda config: config["search_space"]["axes"]["hidden_width"].update(
                values=[4096, 3072]
            ),
            "search_space",
        ),
        (lambda config: config.update(granularity="block"), "semantic config"),
        (
            lambda config: config["embedding_pruning"].update(widths=[4096, 3072]),
            "semantic config",
        ),
        (
            lambda config: config["embedding_pruning"].update(alignment=128),
            "semantic config",
        ),
    ),
)
def test_normalization_rejects_true_semantic_mismatches(tmp_path, mutation, expected):
    source, destination, receipt, target_config, config_path = _relocated_semantic_setup(tmp_path)
    mutation(target_config)
    _write_json(config_path, target_config)

    with pytest.raises(ArtifactImportError, match=expected):
        import_campaign_artifacts(
            source,
            destination,
            receipt,
            bundles=("activation",),
            target_config_path=config_path,
            dry_run=True,
        )

    assert not destination.exists()


@pytest.mark.parametrize("stage", ("vllm_stats", "scoring"))
def test_folded_stage_target_projection_mismatch_is_rejected(tmp_path, stage):
    source, destination, receipt = _setup(tmp_path)
    (source / f"manifests/{stage}.json").unlink()
    _receipt(source, receipt)
    config, config_path = _target_config(destination, tmp_path / "target-config.json")
    config[stage] = {"semantic-change": True}
    _write_json(config_path, config)

    with pytest.raises(ArtifactImportError, match="semantic config|compatibility"):
        import_campaign_artifacts(
            source,
            destination,
            receipt,
            bundles=(stage,),
            target_config_path=config_path,
        )

    assert not destination.exists()


def test_imported_bypass_evidence_is_not_recognized_complete_by_main(tmp_path):
    source, destination, receipt = _setup(tmp_path)
    config, config_path = _target_config(destination, tmp_path / "target-config.json")

    import_campaign_artifacts(
        source,
        destination,
        receipt,
        bundles=("bypass_evidence",),
    )

    assert not _completion_is_valid(config, config_path, "bypass")
    assert not (destination / "manifests/completions/bypass.json").exists()


def test_final_payload_hashes_match_receipt_and_files_are_read_only(tmp_path):
    source, destination, receipt = _setup(tmp_path)

    import_campaign_artifacts(source, destination, receipt)

    receipt_payload = json.loads(receipt.read_text())
    for bundle in ("activation", "depth", "vllm_stats", "scoring", "bypass_evidence"):
        for item in receipt_payload["artifacts"][bundle]["files"]:
            path = destination / item["path"]
            assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]
            assert path.stat().st_mode & 0o222 == 0


def test_cli_requires_all_paths_and_dry_run_prints_exact_plan_without_writes(tmp_path, capsys):
    source, destination, receipt = _setup(tmp_path)

    assert main(
        [
            "--source-root",
            str(source),
            "--destination-root",
            str(destination),
            "--receipt",
            str(receipt),
            "--dry-run",
        ]
    ) == 0

    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "planned"
    assert not destination.exists()
