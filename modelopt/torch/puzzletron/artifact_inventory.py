# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Create deterministic, read-only receipts for Puzzletron campaign artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

from .identity import stable_hash
from .manifest import semantic_stage_config

__all__ = ["DEFAULT_ARTIFACT_PATHS", "inventory_campaign_artifacts", "main", "write_inventory"]


DEFAULT_ARTIFACT_PATHS = {
    "activation": "pruning/pruning_scores/automodel/full",
    "depth": "depth/iterative/trajectory.json",
    "vllm_stats": "runtime_cache",
    "vllm_aggregate": "subblock_stats.json",
    "scoring": "subblock_replacement_manifest.json",
    "scoring_results": "single_subblock_replacement_solutions--validation",
    "score_index": "rpc_eval/replace_one_block/score_index.json",
    "bypass_evidence": "artifacts/bypass/dp_observations.jsonl",
}
_EXECUTION_BUNDLES = ("activation", "depth", "vllm_stats", "scoring")
_LEGACY_STAGE_CONFIG_PROVIDERS = {
    "vllm_stats": "build_library",
    "scoring": "build_library",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _canonical_file_inventory(root: Path, paths: tuple[Path, ...]) -> list[dict[str, Any]]:
    files: set[Path] = set()
    for path in paths:
        if path.is_file():
            files.add(path)
        elif path.is_dir():
            files.update(item for item in path.rglob("*") if item.is_file())
    inventory = []
    for path in sorted(files, key=lambda item: str(item)):
        try:
            relative = path.relative_to(root)
        except ValueError as error:
            raise ValueError(f"artifact file is outside the campaign root: {path}") from error
        inventory.append(
            {
                "path": str(relative),
                "size": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return inventory


def _read_json(path: Path, *, object_only: bool = True) -> tuple[Any, str | None]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return {}, str(error)
    if object_only and not isinstance(payload, dict):
        return {}, "top-level JSON must be an object"
    return payload, None


def _artifact_path(
    root: Path, artifact_paths: Mapping[str, str | Path] | None, name: str
) -> tuple[Path, str]:
    relative = Path((artifact_paths or {}).get(name, DEFAULT_ARTIFACT_PATHS[name]))
    path = relative if relative.is_absolute() else root / relative
    return path, str(relative)


def _receipt_path(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _has_symlink_component(path: Path) -> bool:
    absolute = path.expanduser()
    if not absolute.is_absolute():
        absolute = Path.cwd() / absolute
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        if current.is_symlink():
            return True
    return False


def _duplicates(values: list[Any]) -> list[Any]:
    return sorted({value for value in values if values.count(value) > 1}, key=repr)


def _activation_path(
    root: Path, artifact_paths: Mapping[str, str | Path] | None
) -> tuple[Path, str]:
    if artifact_paths and "activation" in artifact_paths:
        return _artifact_path(root, artifact_paths, "activation")
    manifest_path = root / "manifests" / "activation.json"
    manifest = {}
    if not _has_symlink_component(manifest_path):
        manifest, _ = _read_json(manifest_path)
    output = (manifest.get("outputs") or {}).get("activations_log_dir")
    if isinstance(output, str):
        path = Path(output)
        path = path if path.is_absolute() else root / path
        return path, _receipt_path(root, path)
    return _artifact_path(root, artifact_paths, "activation")


def _count_state(expected: int | None, observed: int) -> str:
    if observed == 0:
        return "missing"
    if expected is None or observed == expected:
        return "complete"
    return "partial"


def _activation_inventory(path: Path, relative_path: str) -> tuple[dict[str, Any], dict[str, Any]]:
    if not path.is_dir():
        return {"state": "missing", "path": relative_path, "counts": {"expected": None, "observed": 0}}, {}
    manifest_path = path / "activation_passes_manifest.json"
    manifest, error = _read_json(manifest_path) if manifest_path.is_file() else ({}, None)
    passes = manifest.get("passes") if error is None else None
    duplicates = _duplicates(passes) if isinstance(passes, list) else []
    if isinstance(passes, list) and all(isinstance(name, str) for name in passes):
        expected = len(passes)
        observed = sum(bool(list((path / name).glob("rank_*.pth"))) for name in passes)
    else:
        expected = None
        observed = len(list(path.rglob("rank_*.pth")))
    inventory = {
        "state": "duplicate_conflicting" if duplicates else "incompatible" if error else _count_state(expected, observed),
        "path": relative_path,
        "counts": {"expected": expected, "observed": observed},
    }
    if error:
        inventory["incompatibilities"] = ["activation_passes_manifest"]
    if duplicates:
        inventory["duplicates"] = duplicates
    return inventory, {"passes": sorted(passes) if isinstance(passes, list) else []}


def _depth_inventory(path: Path, relative_path: str) -> tuple[dict[str, Any], dict[str, Any]]:
    if not path.is_file():
        return {"state": "missing", "path": relative_path, "counts": {"expected": None, "observed": 0}}, {}
    payload, error = _read_json(path)
    max_removals = payload.get("max_removals")
    scenarios = payload.get("scenarios")
    expected = int(max_removals) + 1 if isinstance(max_removals, int) else None
    observed = len(scenarios) if isinstance(scenarios, list) else 0
    incompatibilities = []
    selected = payload.get("selected")
    if error or payload.get("status") not in (None, "complete"):
        incompatibilities.append("trajectory")
    if not isinstance(selected, list) or len(selected) != max_removals:
        incompatibilities.append("depth_identity")
        selected = []
    removal_keys = []
    for removal in selected:
        if (
            not isinstance(removal, dict)
            or not isinstance(removal.get("layer_idx"), int)
            or removal["layer_idx"] < 0
            or removal.get("kind") not in {"block", "attention", "mamba", "ffn", "moe"}
        ):
            incompatibilities.append("depth_identity")
            continue
        removal_keys.append((removal["layer_idx"], removal["kind"]))
    if len(set(removal_keys)) != len(removal_keys):
        incompatibilities.append("depth_identity")
    scenario_identities = []
    if isinstance(scenarios, list):
        for index, scenario in enumerate(scenarios):
            if not isinstance(scenario, dict) or scenario.get("removals") != selected[:index]:
                incompatibilities.append("depth_identity")
                continue
            identity_payload = {key: value for key, value in scenario.items() if key != "scenario_id"}
            expected_identity = stable_hash(identity_payload, prefix="depth_scenario")
            if scenario.get("scenario_id") != expected_identity:
                incompatibilities.append("depth_identity")
            scenario_identities.append(scenario.get("scenario_id"))
    else:
        incompatibilities.append("depth_identity")
    if len(set(scenario_identities)) != len(scenario_identities):
        incompatibilities.append("depth_identity")
    state = "incompatible" if incompatibilities else _count_state(expected, observed)
    inventory = {"state": state, "path": relative_path, "counts": {"expected": expected, "observed": observed}}
    if incompatibilities:
        inventory["incompatibilities"] = sorted(set(incompatibilities))
    compatibility = {
        key: payload[key]
        for key in ("source_checkpoint_dir", "parent_checkpoint_identity", "data_identity", "evaluator_revision", "granularity")
        if key in payload
    }
    if "source_checkpoint_dir" in compatibility:
        compatibility["source_checkpoint"] = compatibility["source_checkpoint_dir"]
    if selected:
        compatibility.update(
            selected_identity=stable_hash(selected, prefix="depth_selected"),
            removal_prefix_identities=[
                stable_hash(selected[:length], prefix="depth_removal_prefix")
                for length in range(len(selected) + 1)
            ],
            scenario_identities=scenario_identities,
        )
    return inventory, compatibility


def _vllm_inventory(
    path: Path,
    relative_path: str,
    aggregate: Path,
    aggregate_relative_path: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    shard_root = path / "shards"
    shard_files = sorted(shard_root.glob("*/shard_*.json")) if shard_root.is_dir() else []
    expected = 0
    observed = 0
    incompatibilities: set[str] = set()
    identities: list[str] = []
    result_indices: set[int] = set()
    groups: dict[Path, list[Path]] = {}
    for shard in shard_files:
        groups.setdefault(shard.parent, []).append(shard)
    for directory, shards in sorted(groups.items(), key=lambda item: str(item[0])):
        metadata = []
        for shard in shards:
            payload, error = _read_json(shard)
            if error:
                incompatibilities.add("shard_json")
                continue
            metadata.append((shard, payload))
        declared_counts = {payload.get("shard_count") for _, payload in metadata}
        if len(declared_counts) != 1 or not declared_counts or not isinstance(next(iter(declared_counts)), int):
            incompatibilities.add("shard_count")
            continue
        shard_count = next(iter(declared_counts))
        expected += shard_count
        identities.append(directory.name)
        for shard, payload in metadata:
            index = payload.get("shard_index")
            if payload.get("spec_identity") != directory.name:
                incompatibilities.add("spec_identity")
            if not isinstance(index, int) or index < 0 or index >= shard_count or shard.stem != f"shard_{index:04d}":
                incompatibilities.add("shard_index")
            results = payload.get("results")
            if results is not None:
                if not isinstance(results, dict) or not results:
                    incompatibilities.add("shard_results")
                else:
                    for raw_index in results:
                        try:
                            result_index = int(raw_index)
                        except (TypeError, ValueError):
                            incompatibilities.add("shard_results")
                            continue
                        if result_index < 0 or result_index in result_indices:
                            incompatibilities.add("shard_results")
                        result_indices.add(result_index)
            done = shard.with_suffix(".done")
            if done.is_file():
                observed += 1
    if len(set(identities)) > 1:
        incompatibilities.add("spec_identity")
    aggregate_payload, aggregate_error = (
        _read_json(aggregate, object_only=False) if aggregate.is_file() else ({}, None)
    )
    if aggregate_error is None and not isinstance(aggregate_payload, (dict, list)):
        aggregate_error = "top-level JSON must be an object or array"
    if not shard_files and not aggregate.is_file():
        state = "missing"
    else:
        if not aggregate.is_file():
            incompatibilities.add("aggregate")
        if not shard_files:
            incompatibilities.add("shards")
        if aggregate_error:
            incompatibilities.add("aggregate_json")
        if isinstance(aggregate_payload, (dict, list)) and not aggregate_payload:
            incompatibilities.add("aggregate_empty")
        if isinstance(aggregate_payload, dict):
            aggregate_spec = aggregate_payload.get("spec_identity")
            if aggregate_spec is not None and aggregate_spec not in set(identities):
                incompatibilities.add("aggregate_shard_identity")
            aggregate_count = aggregate_payload.get("shard_count")
            if aggregate_count is not None and aggregate_count != expected:
                incompatibilities.add("aggregate_shard_count")
            records = aggregate_payload.get("block_runtime_records")
            if result_indices and isinstance(records, list) and len(records) != len(result_indices):
                incompatibilities.add("aggregate_shard_results")
        state = "incompatible" if incompatibilities else _count_state(expected or None, observed)
    inventory = {
        "state": state,
        "path": relative_path,
        "counts": {"expected": expected or None, "observed": observed},
    }
    if incompatibilities:
        inventory["incompatibilities"] = sorted(incompatibilities)
    compatibility = {
        "spec_identities": sorted(identities),
        "aggregate": aggregate_relative_path,
        **{
            key: aggregate_payload[key]
            for key in ("source_checkpoint_dir", "teacher_dir", "data_identity", "model_identity", "granularity")
            if isinstance(aggregate_payload, dict) and key in aggregate_payload
        },
    }
    if "source_checkpoint_dir" in compatibility:
        compatibility["source_checkpoint"] = compatibility["source_checkpoint_dir"]
    elif "teacher_dir" in compatibility:
        compatibility["source_checkpoint"] = compatibility["teacher_dir"]
    return inventory, compatibility


def _scoring_inventory(
    path: Path,
    relative_path: str,
    result_dir: Path,
    index_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not path.is_file():
        return {"state": "missing", "path": relative_path, "counts": {"expected": None, "observed": 0}}, {}
    manifest, error = _read_json(path)
    expected = manifest.get("subblock_solution_count")
    expected = int(expected) if isinstance(expected, int) else None
    solution_files = sorted(result_dir.glob("solution_*.json")) if result_dir.is_dir() else []
    observed = len(solution_files)
    solution_ids = []
    incompatibilities: set[str] = set()
    for solution_path in solution_files:
        solution, solution_error_text = _read_json(solution_path)
        if solution_error_text:
            incompatibilities.add("solution_set")
            continue
        solution_id = solution.get("i_solution")
        solution_ids.append(solution_id)
        if not isinstance(solution_id, int) or solution_path.name != f"solution_{solution_id}.json":
            incompatibilities.add("solution_set")
    duplicate_solution_ids = _duplicates(solution_ids)
    index, index_error = _read_json(index_path) if index_path.is_file() else ({}, None)
    entries = index.get("scores") if index_error is None and index_path.is_file() else None
    request_ids = [
        entry.get("request_id")
        for entry in entries or []
        if isinstance(entry, dict) and entry.get("request_id")
    ]
    duplicates = _duplicates(request_ids)
    expected_ids = set(range(expected)) if expected is not None else None
    if expected_ids is not None and set(solution_ids) != expected_ids:
        incompatibilities.add("solution_set")
    if index_path.is_file():
        if index_error or not isinstance(entries, list):
            incompatibilities.add("score_index_set")
        else:
            index_solution_ids = []
            for entry in entries:
                if (
                    not isinstance(entry, dict)
                    or not isinstance(entry.get("request_id"), str)
                    or not isinstance(entry.get("source_result_path"), str)
                    or not isinstance(entry.get("metrics"), dict)
                    or not isinstance(entry.get("metadata"), dict)
                ):
                    incompatibilities.add("score_index_set")
                    continue
                source_result_path = Path(entry["source_result_path"])
                matching_files = [
                    solution_path
                    for solution_path in solution_files
                    if solution_path.name == source_result_path.name
                    and (
                        not source_result_path.is_absolute()
                        or source_result_path.absolute() == solution_path.absolute()
                    )
                ]
                if len(matching_files) != 1:
                    incompatibilities.add("score_index_set")
                    continue
                index_solution_ids.append(
                    solution_ids[solution_files.index(matching_files[0])]
                )
            if expected_ids is not None and set(index_solution_ids) != expected_ids:
                incompatibilities.add("score_index_set")
            duplicate_index_solution_ids = _duplicates(index_solution_ids)
            if duplicate_index_solution_ids:
                incompatibilities.add("score_index_set")
        if not isinstance(entries, list):
            duplicate_index_solution_ids = []
    else:
        duplicate_index_solution_ids = []
    state = (
        "duplicate_conflicting"
        if duplicates or duplicate_solution_ids or duplicate_index_solution_ids
        else "incompatible"
        if error or incompatibilities
        else _count_state(expected, observed)
    )
    inventory = {"state": state, "path": relative_path, "counts": {"expected": expected, "observed": observed}}
    if duplicates:
        inventory["duplicates"] = duplicates
    if duplicate_solution_ids:
        inventory["duplicate_solution_identities"] = duplicate_solution_ids
    if duplicate_index_solution_ids:
        inventory["duplicate_index_solution_identities"] = duplicate_index_solution_ids
    if state == "incompatible":
        if error:
            incompatibilities.add("manifest_or_index")
        inventory["incompatibilities"] = sorted(incompatibilities)
    compatibility = {
        key: manifest[key]
        for key in ("mode", "teacher_dir", "full_search_space_preserved", "identity", "granularity")
        if key in manifest
    }
    if "granularity" not in compatibility and manifest.get("mode") == "replace_one_subblock":
        compatibility["granularity"] = "subblock"
    for key in ("source_checkpoint_dir", "data_identity", "model_identity"):
        if key in manifest:
            compatibility[key] = manifest[key]
    if "teacher_dir" in compatibility:
        compatibility["source_checkpoint"] = compatibility["teacher_dir"]
    elif "source_checkpoint_dir" in compatibility:
        compatibility["source_checkpoint"] = compatibility["source_checkpoint_dir"]
    return inventory, compatibility


def _bypass_inventory(path: Path, relative_path: str) -> tuple[dict[str, Any], dict[str, Any]]:
    if not path.is_file():
        return {"state": "missing", "path": relative_path, "counts": {"expected": None, "observed": 0}}, {}
    try:
        observations = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        valid = all(isinstance(json.loads(line), dict) for line in observations)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        valid = False
        observations = []
    return {
        "state": _count_state(None, len(observations)) if valid else "incompatible",
        "path": relative_path,
        "counts": {"expected": None, "observed": len(observations)},
    }, {}


def _overall_state(artifacts: Mapping[str, Mapping[str, Any]]) -> str:
    states = [str(artifact["state"]) for artifact in artifacts.values()]
    if all(state == "missing" for state in states):
        return "missing"
    if "duplicate_conflicting" in states:
        return "duplicate_conflicting"
    if "incompatible" in states:
        return "incompatible"
    if "partial" in states or "missing" in states:
        return "partial"
    return "complete"


def _identity_value(value: Any, *, path: bool = False) -> str:
    if path and isinstance(value, str):
        return os.path.normpath(value)
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _semantic_compatibility(artifacts: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    granularities = {
        name: artifact["granularity"]
        for name, artifact in artifacts.items()
        if "granularity" in artifact
    }
    contradictions = {}
    for name, path in (("source_checkpoint", True), ("data_identity", False), ("model_identity", False)):
        values = {
            _identity_value(artifact[name], path=path)
            for artifact in artifacts.values()
            if name in artifact
        }
        if len(values) > 1:
            contradictions[name] = sorted(values)
    return {"granularities": granularities, "contradictions": contradictions}


def _successful_manifest_config(
    path: Path, expected_stage: str
) -> tuple[dict[str, Any] | None, list[str]]:
    if _has_symlink_component(path):
        return None, ["source_stage_manifest_symlink"]
    manifest, error = _read_json(path)
    if error or manifest.get("status") != "success":
        return None, []
    config = manifest.get("config")
    if not isinstance(config, dict):
        return None, ["source_stage_manifest"]
    manifest_stage = manifest.get("stage")
    if manifest_stage is not None and manifest_stage != expected_stage:
        return None, ["source_stage_identity"]
    if manifest_stage is None and path.stem != expected_stage:
        return None, ["source_stage_identity"]
    projection = semantic_stage_config(config, expected_stage)
    identity = stable_hash(projection, prefix=f"{expected_stage}_semantic_cfg")
    incompatibilities = []
    if "semantic_config" in manifest and manifest["semantic_config"] != projection:
        incompatibilities.append("source_semantic_config")
    if (
        "semantic_config_identity" in manifest
        and manifest["semantic_config_identity"] != identity
    ):
        incompatibilities.append("source_semantic_config_identity")
    return config, incompatibilities


def _source_stage_compatibility(root: Path, stage: str) -> tuple[dict[str, Any], list[str]]:
    dedicated = root / "manifests" / f"{stage}.json"
    if _has_symlink_component(dedicated):
        return {}, ["source_stage_manifest_symlink"]
    if dedicated.is_file():
        config, incompatibilities = _successful_manifest_config(dedicated, stage)
        if incompatibilities:
            return {}, incompatibilities
        if config is None:
            return {}, ["source_stage_manifest"]
    else:
        provider = _LEGACY_STAGE_CONFIG_PROVIDERS.get(stage)
        if provider is None:
            return {}, ["source_stage_manifest"]
        config, incompatibilities = _successful_manifest_config(
            root / "manifests" / f"{provider}.json", provider
        )
        if incompatibilities:
            return {}, incompatibilities
        if config is None:
            return {}, ["source_stage_manifest"]

    projection = semantic_stage_config(config, stage)
    identity = stable_hash(projection, prefix=f"{stage}_semantic_cfg")
    compatibility = {
        "source_semantic_config": projection,
        "source_semantic_config_identity": identity,
    }
    for section in ("model", "data", "dataset", "search_space"):
        if section in projection:
            compatibility[f"source_{section}_compatibility"] = projection[section]
    return compatibility, incompatibilities


def inventory_campaign_artifacts(
    campaign_root: str | Path,
    *,
    artifact_paths: Mapping[str, str | Path] | None = None,
) -> dict[str, Any]:
    """Return a deterministic, read-only inventory receipt for a campaign root."""

    root = Path(campaign_root)
    activation_path, activation_relative = _activation_path(root, artifact_paths)
    depth_path, depth_relative = _artifact_path(root, artifact_paths, "depth")
    vllm_path, vllm_relative = _artifact_path(root, artifact_paths, "vllm_stats")
    vllm_aggregate_path, vllm_aggregate_relative = _artifact_path(root, artifact_paths, "vllm_aggregate")
    scoring_path, scoring_relative = _artifact_path(root, artifact_paths, "scoring")
    scoring_results_path, scoring_results_relative = _artifact_path(
        root, artifact_paths, "scoring_results"
    )
    score_index_path, score_index_relative = _artifact_path(root, artifact_paths, "score_index")
    bypass_path, bypass_relative = _artifact_path(root, artifact_paths, "bypass_evidence")
    activation, activation_compatibility = _activation_inventory(activation_path, activation_relative)
    depth, depth_compatibility = _depth_inventory(depth_path, depth_relative)
    vllm_stats, vllm_compatibility = _vllm_inventory(
        vllm_path, vllm_relative, vllm_aggregate_path, vllm_aggregate_relative
    )
    scoring, scoring_compatibility = _scoring_inventory(
        scoring_path, scoring_relative, scoring_results_path, score_index_path
    )
    bypass_evidence, bypass_compatibility = _bypass_inventory(bypass_path, bypass_relative)
    artifacts = {
        "activation": activation,
        "depth": depth,
        "vllm_stats": vllm_stats,
        "scoring": scoring,
        "bypass_evidence": bypass_evidence,
    }
    bundle_paths = {
        "activation": (activation_path,),
        "depth": (depth_path,),
        "vllm_stats": (vllm_path, vllm_aggregate_path),
        "scoring": (scoring_path, scoring_results_path, score_index_path),
        "bypass_evidence": (bypass_path,),
    }
    for name, paths in bundle_paths.items():
        artifacts[name]["files"] = _canonical_file_inventory(root, paths)
    compatibility = {
        "activation": activation_compatibility,
        "depth": depth_compatibility,
        "vllm_stats": vllm_compatibility,
        "scoring": scoring_compatibility,
        "bypass_evidence": bypass_compatibility,
    }
    for stage in _EXECUTION_BUNDLES:
        if artifacts[stage]["state"] == "missing":
            continue
        stage_compatibility, incompatibilities = _source_stage_compatibility(root, stage)
        compatibility[stage].update(stage_compatibility)
        if incompatibilities:
            artifacts[stage]["state"] = "incompatible"
            existing = set(artifacts[stage].get("incompatibilities") or ())
            artifacts[stage]["incompatibilities"] = sorted(existing | set(incompatibilities))
    compatibility.update(_semantic_compatibility(compatibility))
    receipt = {
        "version": 2,
        "campaign_root": str(root),
        "state": "incompatible" if compatibility["contradictions"] else _overall_state(artifacts),
        "artifact_paths": {
            "activation": activation_relative,
            "depth": depth_relative,
            "vllm_stats": vllm_relative,
            "vllm_aggregate": vllm_aggregate_relative,
            "scoring": scoring_relative,
            "scoring_results": scoring_results_relative,
            "score_index": score_index_relative,
            "bypass_evidence": bypass_relative,
        },
        "artifacts": artifacts,
        "compatibility": compatibility,
    }
    receipt["receipt_identity"] = _canonical_json_hash(receipt)
    return receipt


def write_inventory(inventory: Mapping[str, Any], output_path: str | Path) -> None:
    """Write an inventory receipt without changing any campaign artifact."""

    output = Path(output_path)
    campaign_root = inventory.get("campaign_root")
    if campaign_root and output.resolve().is_relative_to(Path(str(campaign_root)).resolve()):
        raise ValueError("receipt output path must be outside the source campaign")
    output.write_text(json.dumps(inventory, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    """Run the receipt CLI for a campaign root and optional JSON output path."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign_root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    inventory = inventory_campaign_artifacts(args.campaign_root)
    if args.output:
        write_inventory(inventory, args.output)
    else:
        print(json.dumps(inventory, indent=2, sort_keys=True))
    return 0
