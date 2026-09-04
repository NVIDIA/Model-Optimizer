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

"""Compatibility adapter for canonical single-stage Puzzletron execution."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

from ..dataset_payload import vlm_materialization_is_complete
from ..identity import (
    artifact_snapshot_identity,
    hash_payload,
    mip_input_artifact_paths,
    stable_hash,
)
from ..import_contract import imported_stage_manifest_is_complete
from ..schema import (
    AttemptSpec,
    CampaignPlan,
    CommandSpec,
    ExecutionStrategy,
    StagePlanNode,
    TaskLauncher,
    TaskTopology,
    ValidatedResult,
    WorkItem,
    WorkPlan,
)
from ..stages import StageStatus, semantic_stage_config, stage_spec, stage_terminal_state
from ..token_caches import resolve_tokenize_caches
from ..vllm_measurements import normalize_vllm_measurements
from .base import WorkAdapter

if __package__.startswith("puzzletron_orchestrator."):
    from puzzletron_orchestrator.execution_record import (
        stage_manifest_uses_execution_record,
        validate_stage_execution_record,
    )
else:
    from ...execution_record import (
        stage_manifest_uses_execution_record,
        validate_stage_execution_record,
    )

__all__ = [
    "StageCompatAdapter",
    "post_mip_summary_is_current",
    "stage_output_patterns",
    "stage_is_complete",
]


def _completion_pattern(root: Path, path: Any) -> str:
    """Return a campaign-relative pattern or an exact absolute external path."""

    resolved = _normalized_path(path)
    try:
        return str(resolved.relative_to(root))
    except ValueError:
        # Acceptance markers treat absolute patterns as exact files. This binds
        # configured caches outside the campaign without globbing external trees.
        return str(resolved)


def stage_output_patterns(config: Mapping[str, Any], stage_id: str) -> tuple[str, ...]:
    """Return completion artifact patterns for one stage."""

    if stage_id.startswith("post."):
        node_id = stage_id.split(".", 2)[-1]
        return (f"artifacts/post_mip/nodes/{node_id}/summary.json",)

    if stage_id == "prepare_dataset":
        output = (config.get("prepare_dataset") or {}).get("output") or config.get("dataset_path")
        configured_root = config.get("puzzle_dir") or (config.get("experiment") or {}).get("dir")
        if output is None or configured_root is None:
            return ()
        puzzle_dir = _normalized_path(configured_root)
        return (
            _completion_pattern(
                puzzle_dir,
                _normalized_path(output) / "puzzletron_acquisition.json",
            ),
        )
    if stage_id == "convert":
        patterns = ["ckpts/teacher/config.json"]
        if bool((config.get("vllm_stats") or {}).get("enabled", False)):
            patterns.append("subblock_library.json")
        return tuple(patterns)
    if stage_id == "tokenize_data":
        try:
            configured_caches = resolve_tokenize_caches(config)
        except (TypeError, ValueError):
            return ()
        if not configured_caches:
            return ()
        configured_root = config.get("puzzle_dir") or (config.get("experiment") or {}).get("dir")
        if configured_root is None:
            return ()
        puzzle_dir = _normalized_path(configured_root)
        patterns = []
        try:
            for cache in configured_caches:
                output = _normalized_path(cache["output"])
                patterns.extend(
                    (
                        _completion_pattern(puzzle_dir, output),
                        _completion_pattern(
                            puzzle_dir,
                            output.with_suffix(output.suffix + ".json"),
                        ),
                    )
                )
        except (KeyError, TypeError, ValueError):
            return ()
        return tuple(dict.fromkeys(patterns))
    if stage_id == "slicing_sanity":
        slicing = config.get("slicing_sanity") or {}
        if slicing.get("backend") == "distributed_parent_sweep":
            return ("artifacts/slicing_sanity/summary.json",)
        return (
            "artifacts/width_slice_equivalence/manifest.json",
            "artifacts/width_slice_equivalence/summary.json",
            "artifacts/width_slice_equivalence/cases/**/*.json",
            "artifacts/width_slice_equivalence/comparisons/*.safetensors",
        )
    if stage_id == "build_library":
        stats_name = (config.get("vllm_stats") or {}).get(
            "subblock_stats_filename", "subblock_stats.json"
        )
        patterns = ["replacement_library.json", "candidate_library.json", stats_name]
        embedding = config.get("embedding_pruning") or {}
        if bool(embedding.get("enabled", False)):
            patterns.append("scenarios/width_scenarios.json")
            for configured_width in embedding.get("widths", ()):
                scenario = f"scenarios/width-{int(configured_width):04d}/depth-00"
                patterns.extend(
                    (
                        f"{scenario}/scenario_manifest.json",
                        f"{scenario}/replacement_library.json",
                        f"{scenario}/candidate_library.json",
                        f"{scenario}/{stats_name}",
                        f"{scenario}/manifests/build_library.json",
                    )
                )
        return tuple(patterns)
    if stage_id == "vllm_stats":
        stats_name = (config.get("vllm_stats") or {}).get(
            "subblock_stats_filename", "subblock_stats.json"
        )
        measurements = normalize_vllm_measurements(config)
        patterns = ["artifacts/vllm_stats/summary.json", stats_name]
        if not (len(measurements) == 1 and next(iter(measurements.values())).legacy):
            patterns.append("artifacts/vllm_stats/measurements/index.json")
            patterns.extend(str(item.relative_stats_path) for item in measurements.values())
        return tuple(patterns)
    if stage_id == "bypass":
        patterns = ["artifacts/bypass/local_kd_loss_history.json"]
        if bool((config.get("bypass") or {}).get("elastic", False)):
            patterns.append("artifacts/bypass/dp_observations.jsonl")
        return tuple(patterns)
    if stage_id == "zero_shot_evaluation":
        return ("artifacts/zero_shot_evaluation/**/evaluation_summary.json",)
    spec = stage_spec(stage_id)
    return spec.completion_artifacts


def _vllm_stats_are_complete(config: Mapping[str, Any], puzzle_dir: Path) -> bool:
    stats_name = (config.get("vllm_stats") or {}).get(
        "subblock_stats_filename", "subblock_stats.json"
    )
    try:
        payload = json.loads((puzzle_dir / stats_name).read_text())
    except (OSError, ValueError):
        return False
    if not isinstance(payload, list) or not payload:
        return False
    measured_widths = {
        int(entry["args"]["n_embd"])
        for entry in payload
        if isinstance(entry, Mapping)
        and isinstance(entry.get("args"), Mapping)
        and entry["args"].get("runtime_stats") is True
        and entry["args"].get("n_embd") is not None
    }
    expected_widths = {
        int(width) for width in (config.get("embedding_pruning") or {}).get("widths", ())
    }
    if not expected_widths.issubset(measured_widths):
        return False
    measurements = normalize_vllm_measurements(config)
    if len(measurements) == 1 and next(iter(measurements.values())).legacy:
        return True
    index = _read_mapping(puzzle_dir / "artifacts" / "vllm_stats" / "measurements" / "index.json")
    if index is None:
        return False
    recorded = index.get("measurements")
    if not isinstance(recorded, Mapping) or set(recorded) != set(measurements):
        return False
    return all(
        (puzzle_dir / measurement.relative_stats_path).is_file()
        and (puzzle_dir / measurement.relative_stats_path).stat().st_size > 0
        for measurement in measurements.values()
    )


def _read_mapping(path: Path) -> Mapping[str, Any] | None:
    try:
        payload = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, Mapping) else None


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _recorded_inventory_is_current(
    root: Path,
    entries: object,
    *,
    repository_cache: Path | None = None,
    ignored_names: tuple[str, ...] = (),
) -> bool:
    if root.is_symlink() or not root.is_dir() or not isinstance(entries, list) or not entries:
        return False
    expected_paths = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            return False
        relative = entry.get("path")
        if not isinstance(relative, str):
            return False
        relative_path = Path(relative)
        if relative_path.is_absolute() or not relative_path.parts or ".." in relative_path.parts:
            return False
        path = root / relative_path
        expected_paths.append(relative_path.as_posix())
        if entry.get("kind") == "hub_blob_symlink":
            if repository_cache is None or not path.is_symlink():
                return False
            try:
                inspected = path.resolve(strict=True)
            except OSError:
                return False
            if (
                not inspected.is_relative_to(repository_cache)
                or not inspected.is_file()
                or inspected.relative_to(repository_cache).as_posix() != entry.get("target")
            ):
                return False
        elif entry.get("kind") == "file":
            if path.is_symlink() or not path.is_file():
                return False
            inspected = path
        else:
            return False
        stat_result = inspected.stat()
        if stat_result.st_size != entry.get("bytes"):
            return False
        metadata_matches = stat_result.st_mtime_ns == entry.get(
            "mtime_ns"
        ) and stat_result.st_ctime_ns == entry.get("ctime_ns")
        if not metadata_matches and _file_sha256(inspected) != entry.get("sha256"):
            return False
    observed_paths = sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if not (path.is_dir() and not path.is_symlink()) and path.name not in ignored_names
    )
    return sorted(expected_paths) == observed_paths


def _recorded_inventory_summary_matches(
    entries: object, *, count: object, size: object, digest: object
) -> bool:
    if not isinstance(entries, list) or not entries:
        return False
    if any(
        not isinstance(entry, Mapping) or not isinstance(entry.get("bytes"), int)
        for entry in entries
    ):
        return False
    encoded = json.dumps(entries, separators=(",", ":"), sort_keys=True).encode()
    return (
        count == len(entries)
        and size == sum(entry["bytes"] for entry in entries)
        and digest == hashlib.sha256(encoded).hexdigest()
    )


def _successful_manifest_is_current(
    config: Mapping[str, Any],
    stage_id: str,
    manifest: Mapping[str, Any],
) -> bool:
    """Return whether a successful manifest carries the current semantic identity."""

    recorded_config = manifest.get("semantic_config")
    expected_config = semantic_stage_config(config, stage_id)
    expected_config_identity = stable_hash(expected_config, prefix=f"{stage_id}_semantic_cfg")
    if (
        stable_hash(recorded_config, prefix=f"{stage_id}_semantic_cfg") != expected_config_identity
        or manifest.get("semantic_config_identity") != expected_config_identity
    ):
        return False
    expected_semantic_identity = stable_hash(
        {
            "stage": stage_id,
            "semantic_config_identity": expected_config_identity,
            "capability_snapshot": manifest.get("capability_snapshot"),
        },
        prefix=f"{stage_id}_semantic",
    )
    return manifest.get("semantic_identity") == expected_semantic_identity


def _normalized_path(path: Any) -> Path:
    return Path(str(path)).expanduser().resolve()


def _path_traverses_symlink(path: Path) -> bool:
    return any(candidate.is_symlink() for candidate in (path, *path.parents) if candidate.exists())


def _token_cache_metadata_is_complete(
    config: Mapping[str, Any],
    stage_config: Mapping[str, Any],
    cache: Mapping[str, Any],
    output: Path,
    metadata_path: Path,
) -> bool:
    metadata = _read_mapping(metadata_path)
    try:
        num_samples = int(cache["num_samples"])
        seq_length = int(cache["seq_length"])
        shuffle_seed = int(cache["shuffle_seed"])
        expected_bytes = num_samples * (seq_length + 1) * 4
        expected_metadata = {
            "status": "complete",
            "version": 1,
            "dataset_path": str(_normalized_path(config["dataset_path"])),
            "tokenizer_path": str(_normalized_path((config.get("convert") or {})["teacher_dir"])),
            "split": str(cache["split"]),
            "content_field": str(stage_config.get("content_field", "messages")),
            "num_samples": num_samples,
            "seq_length": seq_length,
            "shuffle_seed": shuffle_seed,
            "trust_remote_code": bool((config.get("model") or {}).get("trust_remote_code", False)),
            "dtype": "uint32",
            "bytes": expected_bytes,
        }
    except (KeyError, TypeError, ValueError):
        return False
    if metadata is None or any(
        metadata.get(key) != value for key, value in expected_metadata.items()
    ):
        return False
    try:
        return output.is_file() and output.stat().st_size == expected_bytes
    except OSError:
        return False


def _token_caches_are_complete(config: Mapping[str, Any], manifest: Mapping[str, Any]) -> bool:
    """Validate configured token caches against their manifest receipts and metadata."""

    stage_config = config.get("tokenize_data") or {}
    try:
        configured = resolve_tokenize_caches(config)
    except (TypeError, ValueError):
        return False
    outputs = manifest.get("outputs")
    recorded = outputs.get("caches") if isinstance(outputs, Mapping) else None
    if not isinstance(recorded, (list, tuple)) or len(recorded) != len(configured):
        return False
    if not configured:
        return False

    expected_by_path: dict[Path, tuple[Mapping[str, Any], Path]] = {}
    for cache in configured:
        if not isinstance(cache, Mapping) or "output" not in cache:
            return False
        output = _normalized_path(cache["output"])
        metadata_path = output.with_suffix(output.suffix + ".json")
        if output in expected_by_path:
            return False
        expected_by_path[output] = (cache, metadata_path)

    recorded_by_path: dict[Path, tuple[Mapping[str, Any], Path]] = {}
    for receipt in recorded:
        if not isinstance(receipt, Mapping):
            return False
        try:
            output = _normalized_path(receipt["path"])
            metadata_path = _normalized_path(receipt["metadata"])
        except (KeyError, TypeError, ValueError):
            return False
        if output in recorded_by_path:
            return False
        recorded_by_path[output] = (receipt, metadata_path)

    if set(recorded_by_path) != set(expected_by_path):
        return False
    for output, (cache, expected_metadata_path) in expected_by_path.items():
        receipt, recorded_metadata_path = recorded_by_path[output]
        if (
            recorded_metadata_path != expected_metadata_path
            or receipt.get("split") != str(cache.get("split"))
            or not _token_cache_metadata_is_complete(
                config,
                stage_config,
                cache,
                output,
                expected_metadata_path,
            )
        ):
            return False
    return True


def _stage_manifest_succeeded(puzzle_dir: Path, stage_id: str) -> Mapping[str, Any] | None:
    payload = _read_mapping(puzzle_dir / "manifests" / f"{stage_id}.json")
    state = stage_terminal_state(payload, expected_stage=stage_id)
    if state is None or state.status not in {StageStatus.SUCCESS, StageStatus.IMPORTED}:
        return None
    return payload


def _build_library_is_complete(config: Mapping[str, Any], puzzle_dir: Path) -> bool:
    if not _patterns_present(puzzle_dir, stage_output_patterns(config, "build_library")):
        return False
    embedding = config.get("embedding_pruning") or {}
    if not bool(embedding.get("enabled", False)):
        return True
    for configured_width in embedding.get("widths", ()):
        scenario = puzzle_dir / "scenarios" / f"width-{int(configured_width):04d}" / "depth-00"
        scenario_manifest = _read_mapping(scenario / "scenario_manifest.json")
        build_manifest = _read_mapping(scenario / "manifests" / "build_library.json")
        if (
            scenario_manifest is None
            or scenario_manifest.get("status") != "complete"
            or build_manifest is None
            or build_manifest.get("status") != "success"
        ):
            return False
    return True


def _hf_checkpoint_is_complete(path: Path) -> bool:
    if not (path / "config.json").is_file():
        return False
    if (path / "model.safetensors").is_file() or (path / "pytorch_model.bin").is_file():
        return True
    for index_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index = _read_mapping(path / index_name)
        if index is None:
            continue
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, Mapping):
            continue
        shards = {str(shard) for shard in weight_map.values()}
        if shards and all((path / shard).is_file() for shard in shards):
            return True
    return False


def _sort_is_complete(puzzle_dir: Path) -> bool:
    stage_manifest_path = puzzle_dir / "manifests" / "sort.json"
    if _stage_manifest_succeeded(puzzle_dir, "sort") is None:
        return False
    width_manifest = puzzle_dir / "manifests" / "width_importance.json"
    if (
        width_manifest.is_file()
        and width_manifest.stat().st_mtime > stage_manifest_path.stat().st_mtime
    ):
        return False
    sorted_teacher = puzzle_dir / "ckpts" / "sorted_teacher"
    sort_manifest = _read_mapping(sorted_teacher / "parallel_sort_manifest.json")
    permutations = _read_mapping(sorted_teacher / "sorted_permutations.json")
    return (
        sort_manifest is not None
        and sort_manifest.get("status") == "complete"
        and bool(permutations)
        and _hf_checkpoint_is_complete(sorted_teacher)
    )


def _width_importance_is_complete(puzzle_dir: Path) -> bool:
    stage_manifest = _stage_manifest_succeeded(puzzle_dir, "width_importance")
    if stage_manifest is None:
        return False
    outputs = stage_manifest.get("outputs")
    if not isinstance(outputs, Mapping) or not outputs.get("activations_log_dir"):
        return False
    output_dir = Path(str(outputs["activations_log_dir"]))
    if not output_dir.is_absolute():
        output_dir = puzzle_dir / output_dir
    pass_manifest = _read_mapping(output_dir / "activation_passes_manifest.json")
    passes = pass_manifest.get("passes") if pass_manifest is not None else None
    if not isinstance(passes, list) or not passes or len(passes) != len(set(passes)):
        return False
    return all(
        isinstance(pass_name, str) and (output_dir / pass_name / "args.json").is_file()
        for pass_name in passes
    )


def _depth_trajectory_is_complete(config: Mapping[str, Any], puzzle_dir: Path) -> bool:
    depth = config.get("depth_importance") or {}
    target = int(depth.get("max_removals", depth.get("max_subblocks_to_remove", 10)))
    configured_output = depth.get("output_dir")
    output_dir = (
        Path(str(configured_output)) if configured_output else puzzle_dir / "depth" / "iterative"
    )
    try:
        payload = json.loads((output_dir / "trajectory.json").read_text())
    except (OSError, ValueError):
        return False
    selected = payload.get("selected")
    return (
        payload.get("status") == "complete"
        and int(payload.get("max_removals", -1)) == target
        and isinstance(selected, list)
        and len(selected) == target
    )


def _patterns_present(puzzle_dir: Path, patterns: tuple[str, ...]) -> bool:
    if not patterns:
        return False
    return all(
        Path(pattern).is_file()
        if Path(pattern).is_absolute()
        else any(path.is_file() for path in puzzle_dir.glob(pattern))
        for pattern in patterns
    )


def post_mip_summary_is_current(
    config: Mapping[str, Any], puzzle_dir: Path, stage_id: str, summary: Mapping[str, Any]
) -> bool:
    """Validate a node summary without importing torch."""

    try:
        if (__package__ or "").startswith("puzzletron_orchestrator."):
            from puzzletron_orchestrator.post_mip.identity import (
                expected_post_mip_execution_identity,
            )
        else:
            from ...post_mip.identity import expected_post_mip_execution_identity

        effective_config = dict(config)
        effective_config["puzzle_dir"] = str(puzzle_dir)
        return summary.get("execution_identity") == expected_post_mip_execution_identity(
            effective_config,
            stage_id,
        )
    except (KeyError, OSError, RuntimeError, TypeError, ValueError):
        return False


def _mip_profiles_are_complete(config: Mapping[str, Any], puzzle_dir: Path) -> bool:
    runs = (config.get("mip") or {}).get("runs") or {}
    if not isinstance(runs, Mapping) or not runs:
        return _patterns_present(puzzle_dir, stage_output_patterns(config, "mip"))
    manifest = _read_mapping(puzzle_dir / "mip" / "active_profiles.json")
    if manifest is None or manifest.get("status") != "success":
        return False
    depth = config.get("depth_importance") or config.get("depth") or {}
    max_depth = int(depth.get("max_subblocks_to_remove", depth.get("max_removals", 0)))
    selected = []
    if max_depth:
        trajectory = _read_mapping(puzzle_dir / "depth" / "iterative" / "trajectory.json")
        if trajectory is None:
            return False
        selected = list(trajectory.get("selected") or ())[:max_depth]
        if len(selected) != max_depth:
            return False
    widths = [int(value) for value in (config.get("embedding_pruning") or {}).get("widths", ())]
    score_granularity = str((config.get("mip") or {}).get("score_granularity", "block"))
    input_artifact_identity = artifact_snapshot_identity(
        mip_input_artifact_paths(puzzle_dir, widths, score_granularity)
    )
    execution_payload = {
        "mip_config": config.get("mip") or {},
        "widths": widths,
        "max_depth": max_depth,
        "depth_trajectory": selected,
        "solve_only": True,
        "input_artifact_identity": input_artifact_identity,
    }
    expected_execution = f"mip_execution_{hash_payload(execution_payload)[:16]}"
    if (
        manifest.get("execution_identity") != expected_execution
        or manifest.get("input_artifact_identity") != input_artifact_identity
    ):
        return False
    profile_ids = list(manifest.get("profile_ids") or ())
    identities = dict(manifest.get("profile_identities") or {})
    if not profile_ids or set(profile_ids) != set(identities):
        return False
    for profile_id in profile_ids:
        grid = _read_mapping(puzzle_dir / "mip" / "profiles" / str(profile_id) / "mip_grid.json")
        if (
            grid is None
            or grid.get("status") != "success"
            or grid.get("execution_identity") != expected_execution
            or grid.get("profile_identity") != identities[profile_id]
        ):
            return False
        scenarios = list(grid.get("scenarios") or ())
        if len(scenarios) != int(grid.get("expected_scenario_count", -1)):
            return False
        solve_only = bool(grid.get("solve_only", False))
        for scenario in scenarios:
            path = Path(str(scenario.get("solution_path") or ""))
            try:
                raw_solutions = json.loads(path.read_text())
            except (OSError, ValueError):
                return False
            solutions = list(scenario.get("solutions") or ())
            if not isinstance(raw_solutions, list) or len(raw_solutions) != len(solutions):
                return False
            if int(scenario.get("solution_count", -1)) != len(solutions):
                return False
            if (scenario.get("status") == "feasible") != bool(raw_solutions):
                return False
            homogeneous = list(scenario.get("homogeneous_solutions") or ())
            if int(scenario.get("homogeneous_solution_count", 0)) != len(homogeneous):
                return False
            homogeneous_path = scenario.get("homogeneous_solution_path")
            if homogeneous_path:
                try:
                    homogeneous_raw = json.loads(Path(str(homogeneous_path)).read_text())
                except (OSError, ValueError):
                    return False
                if not isinstance(homogeneous_raw, list) or len(homogeneous_raw) != len(
                    homogeneous
                ):
                    return False
            elif homogeneous:
                return False
            if not solve_only:
                for result in [*solutions, *homogeneous]:
                    checkpoint = Path(str(result.get("checkpoint") or ""))
                    if not _hf_checkpoint_is_complete(checkpoint):
                        return False
    return True


def _zero_shot_profiles_are_complete(config: Mapping[str, Any], puzzle_dir: Path) -> bool:
    profile_ids = (config.get("zero_shot_evaluation") or {}).get("profile_ids") or ()
    if not profile_ids:
        return _patterns_present(puzzle_dir, stage_output_patterns(config, "zero_shot_evaluation"))
    return all(
        bool(
            list(
                puzzle_dir.glob(
                    "artifacts/zero_shot_evaluation/"
                    f"profiles/{profile_id}/**/evaluation_summary.json"
                )
            )
        )
        for profile_id in profile_ids
    )


def _prepared_dataset_is_complete(config: Mapping[str, Any], manifest: Mapping[str, Any]) -> bool:
    stage_config = config.get("prepare_dataset") or {}
    output = stage_config.get("output") or config.get("dataset_path")
    if output is None:
        return False
    dataset_root = _normalized_path(output)
    canonical = _read_mapping(dataset_root / "manifest.json")
    compatibility = _read_mapping(dataset_root / "puzzletron_acquisition.json")
    if canonical is None or compatibility != canonical:
        return False
    if stage_config.get("adapter", "nemotron_vlm_v2") == (
        "nemotron_vlm_v2"
    ) and not vlm_materialization_is_complete(
        dataset_root,
        canonical,
        cache_success=True,
    ):
        return False

    evaluation_tasks = stage_config.get("evaluation_tasks") or ()
    if not evaluation_tasks:
        return True
    if isinstance(evaluation_tasks, str) or not isinstance(evaluation_tasks, (list, tuple)):
        return False
    expected_tasks = tuple(str(task) for task in evaluation_tasks)
    catalog = stage_config.get("evaluation_datasets")
    configured_hf_home = stage_config.get("evaluation_hf_home")
    if not isinstance(catalog, Mapping) or not isinstance(configured_hf_home, str):
        return False
    outputs = manifest.get("outputs")
    if not isinstance(outputs, Mapping):
        return False
    hf_home_value = outputs.get("evaluation_hf_home")
    rows = outputs.get("evaluation_data")
    if not isinstance(hf_home_value, str) or not isinstance(rows, list):
        return False
    hf_home_path = Path(hf_home_value).expanduser().absolute()
    expected_hf_home_path = Path(configured_hf_home).expanduser().absolute()
    runner_hf_home = os.environ.get("HF_HOME")
    if (
        hf_home_path != expected_hf_home_path
        or _path_traverses_symlink(hf_home_path)
        or not hf_home_path.is_dir()
        or runner_hf_home is None
        or hf_home_path != Path(runner_hf_home).expanduser().absolute()
        or len(rows) != len(expected_tasks)
    ):
        return False
    hf_home = hf_home_path.resolve()

    observed_tasks: list[str] = []
    for row in rows:
        if not isinstance(row, Mapping):
            return False
        task = row.get("task")
        repository = row.get("repository")
        revision = row.get("revision")
        snapshot_value = row.get("snapshot")
        requires_media = row.get("requires_media")
        preparation_dir = row.get("preparation_dir")
        status = row.get("status")
        contract = catalog.get(task) if isinstance(task, str) else None
        if (
            not isinstance(task, str)
            or not isinstance(repository, str)
            or not isinstance(revision, str)
            or not isinstance(snapshot_value, str)
            or not isinstance(requires_media, bool)
            or not isinstance(contract, Mapping)
            or contract.get("repository") != repository
            or contract.get("revision") != revision
            or contract.get("requires_media") != requires_media
            or contract.get("preparation_dir") != preparation_dir
            or status != ("complete" if requires_media else "downloaded")
        ):
            return False
        snapshot_path = Path(snapshot_value).expanduser().absolute()
        expected_snapshot = (
            hf_home / "hub" / f"datasets--{repository.replace('/', '--')}" / "snapshots" / revision
        )
        if _path_traverses_symlink(snapshot_path) or snapshot_path.resolve() != expected_snapshot:
            return False
        snapshot = snapshot_path.resolve()
        snapshot_report = row.get("snapshot_inventory")
        if not isinstance(snapshot_report, Mapping):
            return False
        snapshot_manifest_value = snapshot_report.get("manifest")
        snapshot_entries = snapshot_report.get("files")
        if not isinstance(snapshot_manifest_value, str) or not isinstance(snapshot_entries, list):
            return False
        snapshot_manifest_path = Path(snapshot_manifest_value).expanduser().absolute()
        repository_cache = expected_snapshot.parent.parent
        expected_snapshot_manifest = (
            repository_cache / f".modelopt_vlm_benchmark_snapshot_{revision}.json"
        )
        recorded_snapshot = _read_mapping(snapshot_manifest_path)
        expected_snapshot_report = {
            key: value for key, value in snapshot_report.items() if key != "manifest"
        }
        if (
            _path_traverses_symlink(snapshot_manifest_path)
            or snapshot_manifest_path.resolve() != expected_snapshot_manifest
            or recorded_snapshot != expected_snapshot_report
            or snapshot_report.get("task") != task
            or snapshot_report.get("repository") != repository
            or snapshot_report.get("revision") != revision
            or snapshot_report.get("snapshot") != str(snapshot_path)
            or not _recorded_inventory_summary_matches(
                snapshot_entries,
                count=snapshot_report.get("file_count"),
                size=snapshot_report.get("bytes"),
                digest=snapshot_report.get("inventory_sha256"),
            )
            or not _recorded_inventory_is_current(
                snapshot, snapshot_entries, repository_cache=repository_cache
            )
        ):
            return False
        observed_tasks.append(task)
        if requires_media:
            media_root_value = row.get("media_root")
            if not isinstance(media_root_value, str) or not isinstance(preparation_dir, str):
                return False
            relative_media = Path(preparation_dir)
            if relative_media.is_absolute() or ".." in relative_media.parts:
                return False
            media_root_path = Path(media_root_value).expanduser().absolute()
            expected_media_root = hf_home / relative_media
            if (
                _path_traverses_symlink(media_root_path)
                or media_root_path.resolve() != expected_media_root
            ):
                return False
            media_root = media_root_path.resolve()
            marker = _read_mapping(media_root / ".modelopt_vlm_benchmark_preparation.json")
            media_entries = marker.get("inventory") if marker is not None else None
            if (
                not media_root.is_dir()
                or marker is None
                or marker.get("schema") != "modelopt.vlm-benchmark-data-preparation/v1"
                or marker.get("status") != "complete"
                or marker.get("task") != task
                or marker.get("repository") != repository
                or marker.get("revision") != revision
                or marker.get("requires_media") is not True
                or marker.get("preparation_dir") != preparation_dir
                or _normalized_path(marker.get("snapshot")) != snapshot
                or any(
                    marker.get(key) != row.get(key)
                    for key in ("media_root", "files", "bytes", "inventory", "inventory_sha256")
                )
                or not _recorded_inventory_summary_matches(
                    media_entries,
                    count=marker.get("files"),
                    size=marker.get("bytes"),
                    digest=marker.get("inventory_sha256"),
                )
                or not _recorded_inventory_is_current(
                    media_root,
                    media_entries,
                    ignored_names=(".modelopt_vlm_benchmark_preparation.json",),
                )
            ):
                return False
    return tuple(observed_tasks) == expected_tasks and len(set(observed_tasks)) == len(
        observed_tasks
    )


def stage_is_complete(config: Mapping[str, Any], stage_id: str) -> bool:
    puzzle_dir = Path(config.get("puzzle_dir") or (config.get("experiment") or {}).get("dir", "."))
    if stage_id.startswith("post."):
        node_id = stage_id.split(".", 2)[-1]
        summary = _read_mapping(
            puzzle_dir / "artifacts" / "post_mip" / "nodes" / node_id / "summary.json"
        )
        if summary is None or summary.get("status") != "success":
            return False
        return post_mip_summary_is_current(config, puzzle_dir, stage_id, summary) and all(
            _hf_checkpoint_is_complete(Path(str(checkpoint)))
            for checkpoint in summary.get("checkpoints") or ()
        )
    manifest = _read_mapping(puzzle_dir / "manifests" / f"{stage_id}.json")
    if manifest is None:
        return False
    if stage_manifest_uses_execution_record(manifest):
        try:
            validate_stage_execution_record(
                puzzle_dir / "manifests" / f"{stage_id}.json",
                expected_stage=stage_id,
            )
        except ValueError:
            return False
    state = stage_terminal_state(manifest, expected_stage=stage_id)
    if state is None or not state.allows_completion(stage_id, config):
        return False
    if state.status is StageStatus.SKIPPED:
        return True
    if state.status is StageStatus.IMPORTED:
        return imported_stage_manifest_is_complete(
            puzzle_dir,
            stage_id,
            manifest,
            expected_semantic_config=semantic_stage_config(config, stage_id),
            stable_hash=stable_hash,
        )
    if not _successful_manifest_is_current(config, stage_id, manifest):
        return False
    if stage_id == "prepare_dataset":
        return _prepared_dataset_is_complete(config, manifest)
    if stage_id == "tokenize_data":
        return _token_caches_are_complete(config, manifest)
    if stage_id == "depth_importance":
        return _depth_trajectory_is_complete(config, puzzle_dir)
    if stage_id == "width_importance":
        return _width_importance_is_complete(puzzle_dir)
    if stage_id == "sort":
        return _sort_is_complete(puzzle_dir)
    if stage_id == "vllm_stats":
        return _vllm_stats_are_complete(config, puzzle_dir)
    if stage_id == "build_library":
        return _build_library_is_complete(config, puzzle_dir)
    if stage_id == "mip":
        return _mip_profiles_are_complete(config, puzzle_dir)
    if stage_id == "zero_shot_evaluation":
        return _zero_shot_profiles_are_complete(config, puzzle_dir)
    return _patterns_present(puzzle_dir, stage_output_patterns(config, stage_id))


class StageCompatAdapter(WorkAdapter):
    """Run one coordinated stage through examples/puzzletron/main.py."""

    strategy = ExecutionStrategy.SINGLE

    def plan(self, plan: CampaignPlan, node: StagePlanNode) -> WorkPlan:
        item = WorkItem(
            work_id=f"{node.stage_id}:0",
            stage_id=node.stage_id,
            shard_index=0,
            shard_count=1,
            gpus_per_instance=node.gpus_per_instance,
        )
        return WorkPlan(stage_id=node.stage_id, strategy=self.strategy, items=(item,))

    def command(
        self,
        *,
        plan: CampaignPlan,
        node: StagePlanNode,
        item: WorkItem,
        attempt_id: str,
        runner,
        overrides: list[str] | None = None,
    ) -> AttemptSpec:
        repo = Path(runner.contract.repository)
        main_py = repo / "examples" / "puzzletron" / "main.py"
        log_dir = plan.log_dir
        log_path = str(log_dir / f"{node.stage_id}_{attempt_id}.log")
        argv: list[str] = [
            "python",
            str(main_py),
            "--config",
            plan.experiment_config_path,
            "--worker-stage",
            node.stage_id,
        ]
        if node.resource != "cpu":
            argv.extend(("--gpus-per-node", str(node.gpus_per_node)))
        for override in overrides or []:
            argv.extend(["--override", override])
        if node.distributed:
            if node.gpus_per_instance % node.nodes:
                raise ValueError(
                    f"stage {node.stage_id} uses {node.gpus_per_instance} GPUs across "
                    f"{node.nodes} nodes; GPUs must divide evenly across tasks"
                )
            topology = TaskTopology(
                task_count=node.nodes,
                gpus_per_task=node.gpus_per_instance // node.nodes,
                tasks_per_group=node.nodes,
                launcher=TaskLauncher.TORCHRUN,
            )
        else:
            topology = TaskTopology(gpus_per_task=node.gpus_per_instance)
        env = {}
        if node.stage_id in {"build_library", "mip"}:
            # These stages do not use ModelOpt's vLLM quantization
            # integration. Avoid loading vLLM in the root worker and in any
            # subprocesses that inherit its environment.
            env["MODELOPT_SKIP_VLLM_PLUGIN"] = "1"
        return AttemptSpec(
            attempt_id=attempt_id,
            work_id=item.work_id,
            stage_id=node.stage_id,
            command=CommandSpec(argv=tuple(argv), cwd=str(repo), env=env, log_path=log_path),
            allocation_nodes=node.nodes,
            allocation_gpus=node.total_gpus,
            exclusive=node.exclusive,
            contract_hash=plan.contract_hash,
            metadata={
                "gpus_per_node": node.gpus_per_node,
                **({"partition": node.partition} if node.partition else {}),
            },
            task_topology=topology,
        )

    def validate(self, *, plan: CampaignPlan, node: StagePlanNode) -> ValidatedResult:
        if stage_is_complete(plan.experiment_config, node.stage_id):
            artifacts = stage_output_patterns(plan.experiment_config, node.stage_id)
            return ValidatedResult(valid=True, reason="stage outputs present", artifacts=artifacts)
        return ValidatedResult(valid=False, reason="stage outputs missing")
