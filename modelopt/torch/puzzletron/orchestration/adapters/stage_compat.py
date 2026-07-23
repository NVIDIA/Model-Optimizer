# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compatibility adapter for canonical single-stage Puzzletron execution."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from ..identity import artifact_snapshot_identity, hash_payload, mip_input_artifact_paths
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
from ..stages import stage_spec
from .base import WorkAdapter

__all__ = [
    "StageCompatAdapter",
    "post_mip_summary_is_current",
    "stage_output_patterns",
    "stage_is_complete",
]


def stage_output_patterns(config: Mapping[str, Any], stage_id: str) -> tuple[str, ...]:
    """Return completion artifact patterns for one stage."""

    if stage_id.startswith("post."):
        node_id = stage_id.split(".", 2)[-1]
        return (f"artifacts/post_mip/nodes/{node_id}/summary.json",)

    if stage_id == "convert":
        patterns = ["ckpts/teacher/config.json"]
        if bool((config.get("vllm_stats") or {}).get("enabled", False)):
            patterns.append("subblock_library.json")
        return tuple(patterns)
    if stage_id == "tokenize_data":
        return ("dataset_cache/*.tokens", "dataset_cache/*.tokens.json")
    if stage_id == "sort":
        return ("ckpts/sorted_teacher/config.json",)
    if stage_id == "depth_importance":
        return ("depth/iterative/trajectory.json",)
    if stage_id == "build_library":
        patterns = ["replacement_library.json", "candidate_library.json"]
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
                        f"{scenario}/subblock_stats.json",
                        f"{scenario}/manifests/build_library.json",
                    )
                )
        return tuple(patterns)
    if stage_id == "vllm_stats":
        stats_name = (config.get("vllm_stats") or {}).get(
            "subblock_stats_filename", "subblock_stats.json"
        )
        return ("artifacts/vllm_stats/summary.json", stats_name)
    if stage_id == "replacement_scoring":
        return ("artifacts/replacement_scoring/summary.json",)
    if stage_id == "bypass":
        patterns = ["artifacts/bypass/local_kd_loss_history.json"]
        if bool((config.get("bypass") or {}).get("elastic", False)):
            patterns.append("artifacts/bypass/dp_observations.jsonl")
        return tuple(patterns)
    if stage_id == "mip":
        return ("mip/profiles/*/mip_grid.json",)
    if stage_id == "zero_shot_evaluation":
        return ("artifacts/zero_shot_evaluation/**/evaluation_summary.json",)
    if stage_id == "aiperf":
        return ("artifacts/aiperf/**/aiperf_results.json",)
    if stage_id == "global_distillation_sanity":
        return (
            "artifacts/global_distillation_sanity/**/global_distillation_sanity_summary.json",
        )
    if stage_id == "global_distillation":
        return ("artifacts/global_distillation/**/global_distillation_summary.json",)
    if stage_id == "post_distillation_evaluation":
        return ("artifacts/post_distillation_evaluation/**/evaluation_summary.json",)
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
    return expected_widths.issubset(measured_widths)


def _read_mapping(path: Path) -> Mapping[str, Any] | None:
    try:
        payload = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, Mapping) else None


def _stage_manifest_succeeded(puzzle_dir: Path, stage_id: str) -> Mapping[str, Any] | None:
    payload = _read_mapping(puzzle_dir / "manifests" / f"{stage_id}.json")
    if payload is None or payload.get("status") != "success":
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
    if width_manifest.is_file() and width_manifest.stat().st_mtime > stage_manifest_path.stat().st_mtime:
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
        Path(str(configured_output))
        if configured_output
        else puzzle_dir / "depth" / "iterative"
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
    return all(bool(list(puzzle_dir.glob(pattern))) for pattern in patterns)


def _prefixed_hash(prefix: str, payload: Mapping[str, Any]) -> str:
    return f"{prefix}_{hash_payload(payload)[:16]}"


def _post_input_candidate_set(
    config: Mapping[str, Any], puzzle_dir: Path, stage_id: str
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    _prefix, flow_id, node_id = stage_id.split(".", 2)
    flow = config["post_mip"]["flows"][flow_id]
    node = flow["nodes"][node_id]
    input_id = str(node.get("input", "source"))
    registry = _read_mapping(
        puzzle_dir / "artifacts" / "post_mip" / "candidate_registry.json"
    )
    if registry is None:
        raise RuntimeError("post-MIP candidate registry is unavailable")
    if input_id != "source":
        current = _read_mapping(
            puzzle_dir
            / "artifacts"
            / "post_mip"
            / "nodes"
            / input_id
            / "current.json"
        )
        if current is None:
            raise RuntimeError(f"post-MIP input node {input_id!r} has no current execution")
        candidate_set = _read_mapping(
            puzzle_dir
            / "artifacts"
            / "post_mip"
            / "nodes"
            / input_id
            / "executions"
            / str(current["execution_identity"])
            / "candidate_set.json"
        )
        if candidate_set is None:
            raise RuntimeError(f"post-MIP input node {input_id!r} has no candidate set")
        identity_payload = {
            key: candidate_set[key]
            for key in (
                "flow_id",
                "node_id",
                "revision_ids",
                "producer_execution_identity",
            )
        }
        if candidate_set.get("identity") != _prefixed_hash(
            "candidate_set", identity_payload
        ):
            raise RuntimeError(f"post-MIP input node {input_id!r} has an invalid candidate set")
        return candidate_set, registry

    active = _read_mapping(puzzle_dir / "mip" / "active_profiles.json")
    if active is None or active.get("status") != "success":
        raise RuntimeError("active MIP profile manifest is unavailable")
    active_execution = str(active["execution_identity"])
    active_profiles = {str(value) for value in active.get("profile_ids") or ()}
    if (
        registry.get("active_mip_execution_identity") != active_execution
        or set(registry.get("active_profile_ids") or ()) != active_profiles
    ):
        raise RuntimeError("post-MIP registry does not reflect the active MIP execution")
    source = flow["source"]
    variants = source.get("variants", "all")
    objectives = source.get("objectives", "all")
    if isinstance(variants, str) and variants != "all":
        variants = [variants]
    if isinstance(objectives, str) and objectives != "all":
        objectives = [objectives]
    revision_ids = []
    for architecture in dict(registry.get("architectures") or {}).values():
        origins = [
            origin
            for origin in architecture.get("origins") or ()
            if origin.get("profile_id") in active_profiles
            and origin.get("mip_execution_identity") == active_execution
            and origin.get("run_id") == source["run"]
            and (variants == "all" or origin.get("variant_id") in variants)
            and (
                objectives == "all"
                or (origin.get("objective") or {}).get("metric") in objectives
            )
        ]
        if origins:
            origins.sort(
                key=lambda origin: (
                    str(origin.get("profile_id")),
                    str(origin.get("kind")),
                    int(origin.get("rank", 0)),
                )
            )
            revision_ids.append(str(origins[0]["revision_id"]))
    revision_ids = sorted(dict.fromkeys(revision_ids))
    payload = {
        "flow_id": flow_id,
        "node_id": "source",
        "revision_ids": revision_ids,
        "producer_execution_identity": active_execution,
    }
    return {
        **payload,
        "identity": _prefixed_hash("candidate_set", payload),
    }, registry


def post_mip_summary_is_current(
    config: Mapping[str, Any], puzzle_dir: Path, stage_id: str, summary: Mapping[str, Any]
) -> bool:
    """Validate a node summary without importing the PyTorch-backed worker package."""

    try:
        _prefix, flow_id, node_id = stage_id.split(".", 2)
        node = dict(config["post_mip"]["flows"][flow_id]["nodes"][node_id])
        candidate_set, registry = _post_input_candidate_set(config, puzzle_dir, stage_id)
        owners = set()
        if node.get("type") == "filter":
            if node.get("mode") in {"top_k", "threshold"}:
                references = [node["metric"]]
            else:
                references = [entry["metric"] for entry in node.get("metrics") or ()]
            owners.update(
                str(reference).partition(".")[0]
                for reference in references
                if not str(reference).startswith("mip.")
            )
        model_source = str(node.get("model_source", "latest"))
        if model_source not in {"latest", "origin"}:
            owners.add(model_source)
        dependency_executions = {}
        for owner in sorted(owners):
            current = _read_mapping(
                puzzle_dir
                / "artifacts"
                / "post_mip"
                / "nodes"
                / owner
                / "current.json"
            )
            if current is None:
                return False
            dependency_executions[owner] = current["execution_identity"]
        revision_ids = [str(value) for value in candidate_set.get("revision_ids") or ()]
        revisions = dict(registry.get("revisions") or {})
        if model_source == "latest":
            source_revisions = {value: value for value in revision_ids}
        elif model_source == "origin":
            source_revisions = {}
            for value in revision_ids:
                current = value
                while revisions[current].get("parent_revision_id") is not None:
                    current = str(revisions[current]["parent_revision_id"])
                source_revisions[value] = current
        else:
            recorded = (summary.get("execution_contract") or {}).get(
                "source_revisions"
            ) or {}
            if set(recorded) != set(revision_ids):
                return False
            source_revisions = dict(recorded)
        contract = {
            "candidate_set": candidate_set["identity"],
            "node": node,
            "dependency_executions": dependency_executions,
            "source_revisions": source_revisions,
        }
        return summary.get("execution_identity") == _prefixed_hash(
            "post_mip_execution", contract
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
        grid = _read_mapping(
            puzzle_dir / "mip" / "profiles" / str(profile_id) / "mip_grid.json"
        )
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
        return _patterns_present(
            puzzle_dir, stage_output_patterns(config, "zero_shot_evaluation")
        )
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


def stage_is_complete(config: Mapping[str, Any], stage_id: str) -> bool:
    puzzle_dir = Path(
        config.get("puzzle_dir") or (config.get("experiment") or {}).get("dir", ".")
    )
    if stage_id.startswith("post."):
        node_id = stage_id.split(".", 2)[-1]
        summary = _read_mapping(
            puzzle_dir / "artifacts" / "post_mip" / "nodes" / node_id / "summary.json"
        )
        if summary is None or summary.get("status") != "success":
            return False
        return post_mip_summary_is_current(
            config, puzzle_dir, stage_id, summary
        ) and all(
            _hf_checkpoint_is_complete(Path(str(checkpoint)))
            for checkpoint in summary.get("checkpoints") or ()
        )
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
        log_dir = plan.puzzle_dir / "logs"
        log_path = str(log_dir / f"{node.stage_id}_{attempt_id}.log")
        argv: list[str] = [
            "python",
            str(main_py),
            "--config",
            plan.experiment_config_path,
            "--worker-stage",
            node.stage_id,
            "--gpus-per-node",
            str(node.gpus_per_node),
        ]
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
