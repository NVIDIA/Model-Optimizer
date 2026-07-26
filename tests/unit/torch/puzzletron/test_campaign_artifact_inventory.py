# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import os
from pathlib import Path

import pytest

from modelopt.torch.puzzletron.artifact_inventory import (
    inventory_campaign_artifacts,
    write_inventory,
)
from modelopt.torch.puzzletron.identity import stable_hash
from modelopt.torch.puzzletron.manifest import StageManifest, semantic_stage_config

_SOURCE_CONFIG = {
    "model": {"source": "/campaign/sorted-teacher", "family": "qwen"},
    "data": {"identity": "data-v1"},
    "search_space": {"identity": "search-v1"},
    "activation": {},
    "depth": {},
    "vllm_stats": {},
    "scoring": {},
}


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


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


def _complete_campaign(root):
    for stage in ("activation", "depth", "vllm_stats", "scoring", "build_library"):
        _write_json(
            root / f"manifests/{stage}.json",
            StageManifest(stage=stage, status="success", config=_SOURCE_CONFIG).to_dict(),
        )

    activation = root / "pruning/pruning_scores/automodel/full"
    _write_json(activation / "activation_passes_manifest.json", {"passes": ["attention", "ffn"]})
    for name in ("attention", "ffn"):
        (activation / name).mkdir(parents=True, exist_ok=True)
        (activation / name / "rank_0000.pth").write_bytes(b"activation")

    selected = [
        {"layer_idx": 0, "kind": "attention"},
        {"layer_idx": 1, "kind": "ffn"},
    ]
    scenarios = []
    for length in range(3):
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
            "max_removals": 2,
            "selected": selected,
            "scenarios": scenarios,
            "source_checkpoint_dir": "/campaign/sorted-teacher",
            "granularity": "subblock",
        },
    )

    shard_dir = root / "runtime_cache/shards/runtime-spec"
    for index in range(2):
        _write_json(
            shard_dir / f"shard_{index:04d}.json",
            {
                "spec_identity": "runtime-spec",
                "shard_index": index,
                "shard_count": 2,
                "results": {str(index): {"total_ms": index + 1.0, "prefill_ms": 0.5}},
            },
        )
        _write_json(shard_dir / f"shard_{index:04d}.done", {"count": 1})
    _write_json(
        root / "subblock_stats.json",
        {
            "runtime_decomposition": {"method": "exact"},
            "block_runtime_records": [{"index": 0}, {"index": 1}],
            "spec_identity": "runtime-spec",
            "granularity": "block",
        },
    )

    _write_json(
        root / "subblock_replacement_manifest.json",
        {
            "mode": "replace_one_subblock",
            "subblock_solution_count": 3,
            "teacher_dir": "/campaign/sorted-teacher",
            "full_search_space_preserved": True,
        },
    )
    score_dir = root / "single_subblock_replacement_solutions--validation"
    for index in range(3):
        _write_json(score_dir / f"solution_{index}.json", {"i_solution": index})
    _write_json(
        root / "rpc_eval/replace_one_block/score_index.json",
        {
            "scores": [
                _score_index_entry(score_dir / f"solution_{index}.json", f"score-{index}")
                for index in range(3)
            ]
        },
    )
    observations = root / "artifacts/bypass/dp_observations.jsonl"
    observations.parent.mkdir(parents=True, exist_ok=True)
    observations.write_text('{"step": 1}\n')


def test_complete_inventory_is_deterministic_and_writes_only_the_requested_receipt(tmp_path):
    _complete_campaign(tmp_path)
    before = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*") if path.is_file())

    first = inventory_campaign_artifacts(tmp_path)
    second = inventory_campaign_artifacts(tmp_path)
    output = tmp_path.parent / "receipt.json"
    write_inventory(first, output)

    assert first == second
    assert first["state"] == "complete"
    assert first["artifacts"]["activation"]["counts"] == {"expected": 2, "observed": 2}
    assert first["artifacts"]["depth"]["counts"] == {"expected": 3, "observed": 3}
    assert first["artifacts"]["vllm_stats"]["counts"] == {"expected": 2, "observed": 2}
    assert first["artifacts"]["scoring"]["counts"] == {"expected": 3, "observed": 3}
    assert first["artifacts"]["bypass_evidence"]["state"] == "complete"
    assert first["compatibility"]["depth"]["granularity"] == "subblock"
    assert first["compatibility"]["scoring"]["teacher_dir"] == "/campaign/sorted-teacher"
    for stage in ("activation", "depth", "vllm_stats", "scoring"):
        expected = stable_hash(
            semantic_stage_config(_SOURCE_CONFIG, stage),
            prefix=f"{stage}_semantic_cfg",
        )
        assert first["compatibility"][stage]["source_semantic_config_identity"] == expected
    assert json.loads(output.read_text()) == first
    assert sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*") if path.is_file()) == before


def test_inventory_reports_missing_and_partial_artifacts(tmp_path):
    _complete_campaign(tmp_path)
    (tmp_path / "pruning/pruning_scores/automodel/full/ffn/rank_0000.pth").unlink()
    (tmp_path / "artifacts/bypass/dp_observations.jsonl").unlink()

    inventory = inventory_campaign_artifacts(tmp_path)

    assert inventory["state"] == "partial"
    assert inventory["artifacts"]["activation"]["state"] == "partial"
    assert inventory["artifacts"]["activation"]["counts"] == {"expected": 2, "observed": 1}
    assert inventory["artifacts"]["bypass_evidence"]["state"] == "missing"


def test_inventory_reports_duplicate_conflicting_scores(tmp_path):
    _complete_campaign(tmp_path)
    score_dir = tmp_path / "single_subblock_replacement_solutions--validation"
    _write_json(
        tmp_path / "rpc_eval/replace_one_block/score_index.json",
        {
            "scores": [
                _score_index_entry(score_dir / "solution_0.json", "score-0"),
                _score_index_entry(score_dir / "solution_1.json", "score-0"),
            ]
        },
    )

    inventory = inventory_campaign_artifacts(tmp_path)

    assert inventory["state"] == "duplicate_conflicting"
    assert inventory["artifacts"]["scoring"]["state"] == "duplicate_conflicting"
    assert inventory["artifacts"]["scoring"]["duplicates"] == ["score-0"]


def test_inventory_reports_incompatible_runtime_shards(tmp_path):
    _complete_campaign(tmp_path)
    _write_json(
        tmp_path / "runtime_cache/shards/runtime-spec/shard_0001.json",
        {
            "spec_identity": "other-spec",
            "shard_index": 1,
            "shard_count": 2,
            "results": {"1": {"total_ms": 2.0, "prefill_ms": 0.5}},
        },
    )

    inventory = inventory_campaign_artifacts(tmp_path)

    assert inventory["state"] == "incompatible"
    assert inventory["artifacts"]["vllm_stats"]["state"] == "incompatible"
    assert "spec_identity" in inventory["artifacts"]["vllm_stats"]["incompatibilities"]


def test_inventory_paths_are_overridable(tmp_path):
    _complete_campaign(tmp_path)
    custom_activation = tmp_path / "custom-activation"
    custom_activation.mkdir()
    (custom_activation / "rank_0000.pth").write_bytes(b"activation")

    inventory = inventory_campaign_artifacts(
        tmp_path,
        artifact_paths={"activation": "custom-activation"},
    )

    assert inventory["artifacts"]["activation"]["path"] == "custom-activation"
    assert inventory["artifacts"]["activation"]["counts"] == {"expected": None, "observed": 1}


def test_activation_discovery_uses_the_stage_manifest_output_path(tmp_path):
    _complete_campaign(tmp_path)
    activation = tmp_path / "pruning/pruning_scores/automodel/full"
    discovered = tmp_path / "pruning/pruning_scores/automodel/campaign-identity"
    discovered.parent.mkdir(parents=True, exist_ok=True)
    activation.rename(discovered)
    manifest_path = tmp_path / "manifests/activation.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["outputs"] = {"activations_log_dir": str(discovered)}
    _write_json(manifest_path, manifest)

    inventory = inventory_campaign_artifacts(tmp_path)

    assert inventory["artifacts"]["activation"]["state"] == "complete"
    assert inventory["artifacts"]["activation"]["path"] == str(discovered.relative_to(tmp_path))


def test_bypass_evidence_accepts_any_positive_observation_count(tmp_path):
    _complete_campaign(tmp_path)
    observations = tmp_path / "artifacts/bypass/dp_observations.jsonl"
    observations.write_text("".join('{"step": 1}\n' for _ in range(80)))

    inventory = inventory_campaign_artifacts(tmp_path)

    assert inventory["artifacts"]["bypass_evidence"]["state"] == "complete"
    assert inventory["artifacts"]["bypass_evidence"]["counts"] == {"expected": None, "observed": 80}


def test_scoring_is_complete_without_optional_rpc_index(tmp_path):
    _complete_campaign(tmp_path)
    (tmp_path / "rpc_eval/replace_one_block/score_index.json").unlink()

    inventory = inventory_campaign_artifacts(tmp_path)

    assert inventory["state"] == "complete"
    assert inventory["artifacts"]["scoring"]["state"] == "complete"


@pytest.mark.parametrize("stage", ("activation", "depth"))
def test_missing_non_folded_dedicated_stage_manifest_is_incompatible(tmp_path, stage):
    _complete_campaign(tmp_path)
    (tmp_path / f"manifests/{stage}.json").unlink()

    inventory = inventory_campaign_artifacts(tmp_path)

    assert inventory["artifacts"][stage]["state"] == "incompatible"
    assert "source_stage_manifest" in inventory["artifacts"][stage]["incompatibilities"]


@pytest.mark.parametrize("stage", ("activation", "depth"))
def test_historical_dedicated_manifest_derives_identity_from_embedded_config(tmp_path, stage):
    _complete_campaign(tmp_path)
    manifest_path = tmp_path / f"manifests/{stage}.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.pop("semantic_config", None)
    manifest.pop("semantic_config_identity", None)
    _write_json(manifest_path, manifest)

    inventory = inventory_campaign_artifacts(tmp_path)

    expected = stable_hash(
        semantic_stage_config(_SOURCE_CONFIG, stage),
        prefix=f"{stage}_semantic_cfg",
    )
    assert inventory["artifacts"][stage]["state"] == "complete"
    assert inventory["compatibility"][stage]["source_semantic_config_identity"] == expected


@pytest.mark.parametrize("stage", ("vllm_stats", "scoring"))
def test_build_library_manifest_supplies_missing_stage_projection(tmp_path, stage):
    _complete_campaign(tmp_path)
    for manifest_path in (tmp_path / "manifests").glob("*.json"):
        if manifest_path.name != "build_library.json":
            manifest_path.unlink()

    inventory = inventory_campaign_artifacts(tmp_path)

    expected = stable_hash(
        semantic_stage_config(_SOURCE_CONFIG, stage),
        prefix=f"{stage}_semantic_cfg",
    )
    assert inventory["artifacts"][stage]["state"] == "complete"
    assert inventory["compatibility"][stage]["source_semantic_config_identity"] == expected


def test_folded_stage_provider_ignores_legitimate_other_manifest_config_evolution(tmp_path):
    _complete_campaign(tmp_path)
    for stage in ("vllm_stats", "scoring"):
        (tmp_path / f"manifests/{stage}.json").unlink()

    for stage, section, value in (
        ("activation", "model", {"source": "/later/model"}),
        ("depth", "data", {"identity": "later-data"}),
    ):
        path = tmp_path / f"manifests/{stage}.json"
        manifest = json.loads(path.read_text())
        manifest["config"][section] = value
        manifest["semantic_config"] = semantic_stage_config(manifest["config"], stage)
        manifest["semantic_config_identity"] = stable_hash(
            manifest["semantic_config"], prefix=f"{stage}_semantic_cfg"
        )
        _write_json(path, manifest)

    provider_path = tmp_path / "manifests/build_library.json"
    provider = json.loads(provider_path.read_text())
    provider.pop("semantic_config", None)
    provider.pop("semantic_config_identity", None)
    _write_json(provider_path, provider)

    inventory = inventory_campaign_artifacts(tmp_path)

    for stage in ("vllm_stats", "scoring"):
        expected = stable_hash(
            semantic_stage_config(_SOURCE_CONFIG, stage),
            prefix=f"{stage}_semantic_cfg",
        )
        assert inventory["artifacts"][stage]["state"] == "complete"
        assert inventory["compatibility"][stage]["source_semantic_config_identity"] == expected


def test_missing_all_successful_embedded_campaign_configs_is_incompatible(tmp_path):
    _complete_campaign(tmp_path)
    for manifest_path in (tmp_path / "manifests").glob("*.json"):
        manifest_path.unlink()

    inventory = inventory_campaign_artifacts(tmp_path)

    assert inventory["state"] == "incompatible"
    assert "source_stage_manifest" in inventory["artifacts"]["activation"]["incompatibilities"]


def test_tampered_folded_stage_provider_identity_is_incompatible(tmp_path):
    _complete_campaign(tmp_path)
    (tmp_path / "manifests/scoring.json").unlink()
    build_path = tmp_path / "manifests/build_library.json"
    build = json.loads(build_path.read_text())
    build["semantic_config_identity"] = "build_library_semantic_cfg:tampered"
    _write_json(build_path, build)

    inventory = inventory_campaign_artifacts(tmp_path)

    assert inventory["artifacts"]["scoring"]["state"] == "incompatible"
    assert "source_semantic_config_identity" in inventory["artifacts"]["scoring"]["incompatibilities"]


def test_mislabeled_dedicated_stage_manifest_is_incompatible(tmp_path):
    _complete_campaign(tmp_path)
    path = tmp_path / "manifests/activation.json"
    manifest = json.loads(path.read_text())
    manifest["stage"] = "depth"
    _write_json(path, manifest)

    inventory = inventory_campaign_artifacts(tmp_path)

    assert inventory["artifacts"]["activation"]["state"] == "incompatible"
    assert "source_stage_identity" in inventory["artifacts"]["activation"]["incompatibilities"]


def test_mislabeled_folded_stage_provider_is_incompatible(tmp_path):
    _complete_campaign(tmp_path)
    (tmp_path / "manifests/scoring.json").unlink()
    path = tmp_path / "manifests/build_library.json"
    manifest = json.loads(path.read_text())
    manifest["stage"] = "scoring"
    _write_json(path, manifest)

    inventory = inventory_campaign_artifacts(tmp_path)

    assert inventory["artifacts"]["scoring"]["state"] == "incompatible"
    assert "source_stage_identity" in inventory["artifacts"]["scoring"]["incompatibilities"]


@pytest.mark.parametrize("symlink_kind", ("manifest", "ancestor"))
def test_dedicated_stage_manifest_symlink_boundary_is_incompatible(tmp_path, symlink_kind):
    _complete_campaign(tmp_path)
    path = tmp_path / "manifests/activation.json"
    if symlink_kind == "manifest":
        external = tmp_path / "external-activation.json"
        path.rename(external)
        path.symlink_to(external)
    else:
        manifests = path.parent
        external = tmp_path / "external-manifests"
        manifests.rename(external)
        manifests.symlink_to(external, target_is_directory=True)

    inventory = inventory_campaign_artifacts(tmp_path)

    assert inventory["artifacts"]["activation"]["state"] == "incompatible"
    assert "source_stage_manifest_symlink" in inventory["artifacts"]["activation"]["incompatibilities"]


@pytest.mark.parametrize("symlink_kind", ("manifest", "ancestor"))
def test_folded_provider_manifest_symlink_boundary_is_incompatible(tmp_path, symlink_kind):
    _complete_campaign(tmp_path)
    (tmp_path / "manifests/scoring.json").unlink()
    path = tmp_path / "manifests/build_library.json"
    if symlink_kind == "manifest":
        external = tmp_path / "external-build-library.json"
        path.rename(external)
        path.symlink_to(external)
    else:
        manifests = path.parent
        external = tmp_path / "external-manifests"
        manifests.rename(external)
        manifests.symlink_to(external, target_is_directory=True)

    inventory = inventory_campaign_artifacts(tmp_path)

    assert inventory["artifacts"]["scoring"]["state"] == "incompatible"
    assert "source_stage_manifest_symlink" in inventory["artifacts"]["scoring"]["incompatibilities"]


@pytest.mark.parametrize("provider_defect", ("unsuccessful", "malformed"))
def test_unusable_folded_stage_provider_is_incompatible(tmp_path, provider_defect):
    _complete_campaign(tmp_path)
    (tmp_path / "manifests/scoring.json").unlink()
    provider_path = tmp_path / "manifests/build_library.json"
    provider = json.loads(provider_path.read_text())
    if provider_defect == "unsuccessful":
        provider["status"] = "failed"
    else:
        provider["config"] = None
    _write_json(provider_path, provider)

    inventory = inventory_campaign_artifacts(tmp_path)

    assert inventory["artifacts"]["scoring"]["state"] == "incompatible"
    assert "source_stage_manifest" in inventory["artifacts"]["scoring"]["incompatibilities"]


def test_tampered_source_stage_semantic_identity_is_incompatible(tmp_path):
    _complete_campaign(tmp_path)
    manifest_path = tmp_path / "manifests/activation.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["semantic_config_identity"] = "activation_semantic_cfg:tampered"
    _write_json(manifest_path, manifest)

    inventory = inventory_campaign_artifacts(tmp_path)

    assert inventory["state"] == "incompatible"
    assert "source_semantic_config_identity" in inventory["artifacts"]["activation"]["incompatibilities"]


@pytest.mark.parametrize("mutation", ("selected", "prefix", "scenario_identity"))
def test_depth_requires_selected_prefix_and_scenario_identity_consistency(tmp_path, mutation):
    _complete_campaign(tmp_path)
    path = tmp_path / "depth/iterative/trajectory.json"
    trajectory = json.loads(path.read_text())
    if mutation == "selected":
        trajectory["selected"][1] = trajectory["selected"][0]
    elif mutation == "prefix":
        trajectory["scenarios"][2]["removals"] = trajectory["selected"][:1]
    else:
        trajectory["scenarios"][1]["scenario_id"] = "depth_scenario:tampered"
    _write_json(path, trajectory)

    inventory = inventory_campaign_artifacts(tmp_path)

    assert inventory["artifacts"]["depth"]["state"] == "incompatible"
    assert "depth_identity" in inventory["artifacts"]["depth"]["incompatibilities"]


def test_vllm_aggregate_must_be_nonempty_and_match_shard_identity(tmp_path):
    _complete_campaign(tmp_path)
    _write_json(tmp_path / "subblock_stats.json", {})

    empty = inventory_campaign_artifacts(tmp_path)

    assert empty["artifacts"]["vllm_stats"]["state"] == "incompatible"
    assert "aggregate_empty" in empty["artifacts"]["vllm_stats"]["incompatibilities"]

    _complete_campaign(tmp_path)
    aggregate_path = tmp_path / "subblock_stats.json"
    aggregate = json.loads(aggregate_path.read_text())
    aggregate["spec_identity"] = "different-runtime-spec"
    _write_json(aggregate_path, aggregate)

    mismatched = inventory_campaign_artifacts(tmp_path)

    assert mismatched["artifacts"]["vllm_stats"]["state"] == "incompatible"
    assert "aggregate_shard_identity" in mismatched["artifacts"]["vllm_stats"]["incompatibilities"]


def test_vllm_aggregate_rejects_multiple_shard_spec_sets(tmp_path):
    _complete_campaign(tmp_path)
    stale = tmp_path / "runtime_cache/shards/stale-spec"
    _write_json(
        stale / "shard_0000.json",
        {
            "spec_identity": "stale-spec",
            "shard_index": 0,
            "shard_count": 1,
            "results": {"9": {"total_ms": 9.0, "prefill_ms": 0.5}},
        },
    )
    _write_json(stale / "shard_0000.done", {"count": 1})

    inventory = inventory_campaign_artifacts(tmp_path)

    assert inventory["artifacts"]["vllm_stats"]["state"] == "incompatible"
    assert "spec_identity" in inventory["artifacts"]["vllm_stats"]["incompatibilities"]


def test_scoring_solution_filenames_and_index_form_exact_expected_set(tmp_path):
    _complete_campaign(tmp_path)
    result_dir = tmp_path / "single_subblock_replacement_solutions--validation"
    (result_dir / "solution_1.json").rename(result_dir / "solution_01.json")

    bad_filename = inventory_campaign_artifacts(tmp_path)

    assert bad_filename["artifacts"]["scoring"]["state"] == "incompatible"
    assert "solution_set" in bad_filename["artifacts"]["scoring"]["incompatibilities"]

    (result_dir / "solution_01.json").unlink()
    _complete_campaign(tmp_path)
    _write_json(
        tmp_path / "rpc_eval/replace_one_block/score_index.json",
        {
            "scores": [
                _score_index_entry(result_dir / "solution_0.json", "score-0"),
                _score_index_entry(result_dir / "solution_1.json", "score-1"),
                _score_index_entry(result_dir / "solution_99.json", "score-2"),
            ]
        },
    )

    bad_index = inventory_campaign_artifacts(tmp_path)

    assert bad_index["artifacts"]["scoring"]["state"] == "incompatible"
    assert "score_index_set" in bad_index["artifacts"]["scoring"]["incompatibilities"]


def test_scoring_accepts_production_index_schema_without_i_solution(tmp_path):
    _complete_campaign(tmp_path)

    inventory = inventory_campaign_artifacts(tmp_path)

    assert inventory["artifacts"]["scoring"]["state"] == "complete"


def test_scoring_rejects_duplicate_source_result_identities(tmp_path):
    _complete_campaign(tmp_path)
    result_dir = tmp_path / "single_subblock_replacement_solutions--validation"
    _write_json(
        tmp_path / "rpc_eval/replace_one_block/score_index.json",
        {
            "scores": [
                _score_index_entry(result_dir / "solution_0.json", "score-0"),
                _score_index_entry(result_dir / "solution_0.json", "score-1"),
                _score_index_entry(result_dir / "solution_2.json", "score-2"),
            ]
        },
    )

    inventory = inventory_campaign_artifacts(tmp_path)

    assert inventory["artifacts"]["scoring"]["state"] == "duplicate_conflicting"
    assert inventory["artifacts"]["scoring"]["duplicate_index_solution_identities"] == [0]


def test_receipt_output_inside_the_source_campaign_is_rejected(tmp_path):
    _complete_campaign(tmp_path)

    with pytest.raises(ValueError, match="source campaign"):
        write_inventory(inventory_campaign_artifacts(tmp_path), tmp_path / "receipt.json")


def test_missing_campaign_and_mixed_inventory_states_are_distinguished(tmp_path):
    assert inventory_campaign_artifacts(tmp_path)["state"] == "missing"

    _complete_campaign(tmp_path)
    (tmp_path / "artifacts/bypass/dp_observations.jsonl").unlink()
    (tmp_path / "runtime_cache").rename(tmp_path / "removed-runtime-cache")
    (tmp_path / "subblock_stats.json").unlink()
    inventory = inventory_campaign_artifacts(tmp_path)

    assert inventory["state"] == "partial"
    assert inventory["artifacts"]["vllm_stats"]["state"] == "missing"


def test_conflicting_source_checkpoint_identities_are_incompatible(tmp_path):
    _complete_campaign(tmp_path)
    manifest = tmp_path / "subblock_replacement_manifest.json"
    payload = json.loads(manifest.read_text())
    payload["teacher_dir"] = "/other/sorted-teacher"
    _write_json(manifest, payload)

    inventory = inventory_campaign_artifacts(tmp_path)

    assert inventory["state"] == "incompatible"
    assert "source_checkpoint" in inventory["compatibility"]["contradictions"]


def test_different_artifact_granularities_remain_distinct(tmp_path):
    _complete_campaign(tmp_path)

    inventory = inventory_campaign_artifacts(tmp_path)

    assert inventory["state"] == "complete"
    assert inventory["compatibility"]["granularities"] == {
        "depth": "subblock",
        "scoring": "subblock",
        "vllm_stats": "block",
    }


def test_non_object_json_is_incompatible(tmp_path):
    _complete_campaign(tmp_path)
    (tmp_path / "subblock_replacement_manifest.json").write_text("[]\n")

    inventory = inventory_campaign_artifacts(tmp_path)

    assert inventory["artifacts"]["scoring"]["state"] == "incompatible"


def test_duplicate_activation_pass_names_are_conflicting(tmp_path):
    _complete_campaign(tmp_path)
    _write_json(
        tmp_path / "pruning/pruning_scores/automodel/full/activation_passes_manifest.json",
        {"passes": ["attention", "attention", "ffn"]},
    )

    inventory = inventory_campaign_artifacts(tmp_path)

    assert inventory["artifacts"]["activation"]["state"] == "duplicate_conflicting"
    assert inventory["artifacts"]["activation"]["duplicates"] == ["attention"]


def test_duplicate_scoring_solution_identities_are_conflicting(tmp_path):
    _complete_campaign(tmp_path)
    _write_json(
        tmp_path / "single_subblock_replacement_solutions--validation/solution_0001.json",
        {"i_solution": 0},
    )

    inventory = inventory_campaign_artifacts(tmp_path)

    assert inventory["artifacts"]["scoring"]["state"] == "duplicate_conflicting"
    assert inventory["artifacts"]["scoring"]["duplicate_solution_identities"] == [0]


def test_bundle_overrides_cover_aggregate_results_and_optional_index(tmp_path):
    _complete_campaign(tmp_path)
    alternate = tmp_path / "alternate"
    shard_dir = alternate / "runtime/shards/runtime-spec"
    for index in range(2):
        _write_json(
            shard_dir / f"shard_{index:04d}.json",
            {
                "spec_identity": "runtime-spec",
                "shard_index": index,
                "shard_count": 2,
                "results": {str(index): {"total_ms": index + 1.0, "prefill_ms": 0.5}},
            },
        )
        _write_json(shard_dir / f"shard_{index:04d}.done", {"count": 1})
    _write_json(
        alternate / "runtime-aggregate.json",
        {"granularity": "block", "spec_identity": "runtime-spec", "records": [1, 2]},
    )
    _write_json(
        alternate / "scoring-manifest.json",
        {"subblock_solution_count": 2, "mode": "replace_one_subblock"},
    )
    for index in range(2):
        _write_json(alternate / "score-results" / f"solution_{index}.json", {"i_solution": index})
    _write_json(
        alternate / "score-index.json",
        {
            "scores": [
                _score_index_entry(alternate / "score-results/solution_0.json", "one"),
                _score_index_entry(alternate / "score-results/solution_1.json", "two"),
            ]
        },
    )

    inventory = inventory_campaign_artifacts(
        tmp_path,
        artifact_paths={
            "vllm_stats": "alternate/runtime",
            "vllm_aggregate": "alternate/runtime-aggregate.json",
            "scoring": "alternate/scoring-manifest.json",
            "scoring_results": "alternate/score-results",
            "score_index": "alternate/score-index.json",
        },
    )

    assert inventory["artifacts"]["vllm_stats"]["state"] == "complete"
    assert inventory["artifacts"]["scoring"]["state"] == "complete"


def test_invalid_bypass_observation_evidence_is_incompatible(tmp_path):
    _complete_campaign(tmp_path)
    (tmp_path / "artifacts/bypass/dp_observations.jsonl").write_text("not-json\n")

    inventory = inventory_campaign_artifacts(tmp_path)

    assert inventory["artifacts"]["bypass_evidence"]["state"] == "incompatible"


def test_list_form_vllm_aggregate_is_valid(tmp_path):
    _complete_campaign(tmp_path)
    _write_json(tmp_path / "subblock_stats.json", [{"runtime_ms": 1.0}])

    inventory = inventory_campaign_artifacts(tmp_path)

    assert inventory["state"] == "complete"
    assert inventory["artifacts"]["vllm_stats"]["state"] == "complete"


def test_scalar_vllm_aggregate_is_incompatible(tmp_path):
    _complete_campaign(tmp_path)
    (tmp_path / "subblock_stats.json").write_text("1\n")

    inventory = inventory_campaign_artifacts(tmp_path)

    assert inventory["state"] == "incompatible"
    assert inventory["artifacts"]["vllm_stats"]["state"] == "incompatible"


def test_receipt_binds_every_bundle_to_canonical_file_identities(tmp_path):
    _complete_campaign(tmp_path)

    receipt = inventory_campaign_artifacts(tmp_path)

    assert receipt["version"] == 2
    for bundle in ("activation", "depth", "vllm_stats", "scoring", "bypass_evidence"):
        files = receipt["artifacts"][bundle]["files"]
        assert files == sorted(files, key=lambda item: item["path"])
        assert files
        for item in files:
            path = tmp_path / item["path"]
            assert not Path(item["path"]).is_absolute()
            assert item == {
                "path": item["path"],
                "size": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }

    identity_payload = {key: value for key, value in receipt.items() if key != "receipt_identity"}
    expected = hashlib.sha256(
        json.dumps(identity_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert receipt["receipt_identity"] == f"sha256:{expected}"


def test_receipt_identity_changes_for_backdated_byte_mutation(tmp_path):
    _complete_campaign(tmp_path)
    original = inventory_campaign_artifacts(tmp_path)
    trajectory = tmp_path / "depth/iterative/trajectory.json"
    timestamps = (trajectory.stat().st_atime_ns, trajectory.stat().st_mtime_ns)
    payload = json.loads(trajectory.read_text())
    payload["scenarios"][0]["hidden_width"] += 1
    scenario = payload["scenarios"][0]
    scenario["scenario_id"] = stable_hash(
        {key: value for key, value in scenario.items() if key != "scenario_id"},
        prefix="depth_scenario",
    )
    _write_json(trajectory, payload)
    os.utime(trajectory, ns=timestamps)

    changed = inventory_campaign_artifacts(tmp_path)

    assert changed["state"] == "complete"
    assert changed["receipt_identity"] != original["receipt_identity"]
