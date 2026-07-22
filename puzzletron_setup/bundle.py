# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render and validate portable Puzzletron setup bundles."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from puzzletron_orchestrator.compiler import (
    compile_campaign_plan,
    load_execution_config,
    load_runner_config,
)
from puzzletron_orchestrator.controller import dry_run_plan

from . import SetupError

__all__ = [
    "BundleResult",
    "BundleValidation",
    "build_bundles",
    "dry_run_bundle",
    "render_execution",
    "render_experiment",
    "render_runner",
    "validate_bundle",
]

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = REPOSITORY_ROOT / "examples/puzzletron/configs/base.yaml"


@dataclass(frozen=True)
class BundleValidation:
    """Dependency-light validation result for one generated bundle."""

    valid: bool
    stage_count: int = 0
    submission_count: int = 0
    error: str | None = None


@dataclass(frozen=True)
class BundleResult:
    """Locations and independent validation results for generated bundles."""

    campaign_dir: Path
    smoke: BundleValidation
    production: BundleValidation


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _deep_merge(base: Mapping[str, Any], update: Mapping[str, Any]) -> dict[str, Any]:
    merged = deepcopy(dict(base))
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _load_fragment(path: Path) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text()) or {}
    except (OSError, yaml.YAMLError) as error:
        raise SetupError(f"Cannot load canonical config fragment {path}: {error}") from error
    if not isinstance(payload, dict):
        raise SetupError(f"Canonical config fragment must be a mapping: {path}")
    payload.pop("defaults", None)
    return payload


def _atomic_yaml(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("w") as stream:
            yaml.safe_dump(_plain(value), stream, sort_keys=False, width=100)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except OSError as error:
        raise SetupError(f"Cannot write generated config {path}: {error}") from error


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("w") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except OSError as error:
        raise SetupError(f"Cannot write generated file {path}: {error}") from error


def _answers(state: Mapping[str, Any], section: str) -> dict[str, Any]:
    return _mapping(_mapping(state.get("answers")).get(section))


def _parallel(mesh: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "tp": int(mesh.get("tp", 1)),
        "cp": int(mesh.get("cp", 1)),
        "pp": int(mesh.get("pp", 1)),
        "ep": int(mesh.get("ep", 1)),
        "dp_shard": int(mesh.get("dp_shard", 1)),
        "dp_replicate": int(mesh.get("dp_replicate", 1)),
        "sequence_parallel": False,
    }


def _model_info(state: Mapping[str, Any]) -> dict[str, Any]:
    model = _mapping(state.get("model"))
    inventory = _mapping(state.get("inventory"))
    config = _mapping(model.get("config"))
    text_config = _mapping(config.get("text_config")) or config
    facts = _mapping(inventory.get("facts"))
    result = {
        "hf_repo": model.get("source"),
        "hf_revision": model.get("resolved_revision"),
        "model_type": inventory.get("model_type"),
        "architectures": list(inventory.get("architectures") or ()),
        "num_hidden_layers": int(inventory.get("num_layers", 0)),
        "layer_counts": _mapping(inventory.get("layer_counts")),
    }
    result.update(facts)
    for key in (
        "max_position_embeddings",
        "hybrid_override_pattern",
        "layers_block_type",
        "mtp_num_hidden_layers",
    ):
        if key in text_config:
            result[key] = text_config[key]
    return {key: value for key, value in result.items() if value is not None}


def _axis_config(pruning: Mapping[str, Any]) -> dict[str, Any]:
    axes = {}
    for axis_id, raw in _mapping(pruning.get("axes")).items():
        axis = _mapping(raw)
        teacher = int(axis["teacher_value"])
        values = [int(value) for value in axis.get("values") or () if int(value) != teacher]
        axes[axis_id] = {
            "enabled": bool(axis.get("enabled", True)),
            "teacher_value": teacher,
            "values": values,
        }
    return axes


def _attention_heads(axes: Mapping[str, Any]) -> list[list[int]]:
    kv_axis = _mapping(axes.get("kv_groups"))
    q_axis = _mapping(axes.get("q_heads_per_group"))
    if not kv_axis or not q_axis:
        return []
    kv_values = [kv_axis["teacher_value"], *kv_axis.get("values", ())]
    q_values = [q_axis["teacher_value"], *q_axis.get("values", ())]
    pairs = {(int(kv) * int(q), int(kv)) for kv in kv_values for q in q_values}
    return [list(pair) for pair in sorted(pairs)]


def _mip_runs(state: Mapping[str, Any], *, smoke: bool) -> dict[str, Any]:
    runs = deepcopy(_mapping(_answers(state, "mip").get("runs")))
    if smoke:
        for run in runs.values():
            solver = _mapping(run.get("solver"))
            solver["num_solutions"] = min(int(solver.get("num_solutions", 2000)), 64)
            solver["max_seconds_per_solution"] = min(
                int(solver.get("max_seconds_per_solution", 120)), 30
            )
            run["solver"] = solver
            homogeneous = _mapping(run.get("homogeneous"))
            if isinstance(homogeneous.get("keep"), int):
                homogeneous["keep"] = min(int(homogeneous["keep"]), 16)
            run["homogeneous"] = homogeneous
    return runs


def _post_mip_flows(
    state: Mapping[str, Any],
    *,
    smoke: bool,
    common_mesh: Mapping[str, Any],
    global_kd_mesh: Mapping[str, Any],
) -> dict[str, Any]:
    flows = deepcopy(_mapping(_answers(state, "post_mip").get("flows")))
    for flow in flows.values():
        for node in _mapping(flow.get("nodes")).values():
            node_type = node.get("type")
            config = _mapping(node.get("config"))
            if node_type in {"evaluation", "materialize"}:
                config.setdefault("automodel", {})["parallel"] = _parallel(common_mesh)
            elif node_type == "aiperf":
                config["topology"] = {
                    "tensor_parallel_size": int(common_mesh.get("tp", 1)),
                    "pipeline_parallel_size": int(common_mesh.get("pp", 1)),
                    "prefill_context_parallel_size": int(common_mesh.get("cp", 1)),
                    "decode_context_parallel_size": int(common_mesh.get("cp", 1)),
                    "distributed_executor_backend": "mp",
                    "gpu_group_size": int(common_mesh.get("tp", 1)),
                }
                if smoke:
                    config["minimum_request_count"] = 4
                    config["requests_per_concurrency"] = 1
            elif node_type == "global_kd":
                config.setdefault("automodel", {})["parallel"] = _parallel(global_kd_mesh)
            if smoke and node_type == "evaluation":
                config["eval_samples"] = min(int(config.get("eval_samples", 128)), 16)
            if smoke and node_type == "global_kd":
                config["max_steps"] = min(int(config.get("max_steps", 128)), 8)
            if config or "config" in node:
                node["config"] = config
    return flows


def render_experiment(state: Mapping[str, Any], budget: str) -> dict[str, Any]:
    """Render one fully composed experiment mapping for smoke or production."""
    if budget not in {"smoke", "production"}:
        raise ValueError(f"Unknown bundle budget: {budget}")
    smoke = budget == "smoke"
    inventory = _mapping(state.get("inventory"))
    profile_path = REPOSITORY_ROOT / str(inventory["family_config"])
    base = _load_fragment(BASE_CONFIG)
    family = _load_fragment(profile_path)
    config = _deep_merge(base, family)

    model = _mapping(state.get("model"))
    data = _answers(state, "data")
    pruning = _answers(state, "pruning")
    runtime = _answers(state, "runtime")
    infrastructure = _answers(state, "infrastructure")
    output = _answers(state, "output")
    meshes = _mapping(infrastructure.get("meshes"))
    common_mesh = _mapping(meshes.get("common"))
    bypass_mesh = _mapping(meshes.get("bypass"))
    global_kd_mesh = _mapping(meshes.get("global_kd"))
    axes = _axis_config(pruning)
    sequence_length = int(data["sequence_length"])
    width_samples = int(pruning["width_importance_samples"])
    replacement_samples = int(pruning["replacement_samples"])
    if smoke:
        width_samples = min(width_samples, 128)
        replacement_samples = min(replacement_samples, 16)
    result_root = str(output["result_root"]).rstrip("/")
    puzzle_dir = f"{result_root}/{budget}"

    activation_passes = list(_mapping(config.get("pruning")).get("activation_passes") or ())
    if "hidden_width" in axes and not any(
        entry.get("name") == "hidden_width" for entry in activation_passes
    ):
        activation_passes.insert(
            0,
            {
                "name": "hidden_width",
                "activation_hooks_kwargs": {"method": "minitron_hidden_width"},
            },
        )
    model_parallel = _parallel(common_mesh)
    bypass = _mapping(pruning.get("bypass"))
    bypass_samples = int(bypass.get("samples", 4096))
    if smoke:
        bypass_samples = min(bypass_samples, 64)

    train_cache = f"{puzzle_dir}/dataset_cache/train.tokens"
    validation_cache = f"{puzzle_dir}/dataset_cache/validation.tokens"
    overlay = {
        "defaults": ["_self_"],
        "clean_config_root": str(Path(puzzle_dir) / "resolved_config"),
        "puzzle_dir": puzzle_dir,
        "dataset_path": data["source"],
        "input_hf_model_path": model["source"],
        "model_info": _model_info(state),
        "model": {
            "source": model["source"],
            "revision": model.get("resolved_revision") or model.get("requested_revision"),
            "descriptor_override": inventory["descriptor"],
            "trust_remote_code": inventory.get("family") == "nemotron3",
            "force_hf": False,
        },
        "capability_validation": {"require_complete_pipeline": True},
        "data": {
            "modality": data["modality"],
            "layout": data["layout"],
            "max_sample_length": sequence_length,
            "path": data["source"],
            "packing": (
                {
                    "pack_size": sequence_length,
                    "packing_ratio": 0.9,
                    "drop_long_samples": True,
                }
                if data["layout"] == "packed_varlen"
                else None
            ),
            "calibration": {
                "path": data["source"],
                "num_samples": width_samples,
                "micro_batch_size": 1,
                "seq_len": sequence_length,
            },
            "replacement_scoring": {
                "num_samples": replacement_samples,
                "micro_batch_size": 1,
            },
        },
        "train_token_cache_path": train_cache,
        "validation_token_cache_path": validation_cache,
        "tokenize_data": {
            "enabled": True,
            "workers": 16 if smoke else 64,
            "tokenize_batch_size": 64,
            "content_field": "messages",
            "caches": [
                {
                    "output": train_cache,
                    "split": "train",
                    "num_samples": width_samples,
                    "seq_length": sequence_length,
                    "shuffle_seed": 444,
                },
                {
                    "output": validation_cache,
                    "split": "validation",
                    "num_samples": replacement_samples,
                    "seq_length": sequence_length,
                    "shuffle_seed": 445,
                },
            ],
        },
        "embedding_pruning": {
            "enabled": bool(_mapping(axes.get("hidden_width")).get("values")),
            "widths": list(_mapping(axes.get("hidden_width")).get("values") or ()),
            "alignment": int(
                _mapping(_mapping(pruning.get("axes")).get("hidden_width")).get("alignment", 1)
            ),
            "cycle_widths": True,
        },
        "pruning": {
            "eval_samples": width_samples,
            "micro_batch_size": 1,
            "block_size": sequence_length,
            "dataset_path": data["source"],
            "activation_passes": activation_passes,
            "automodel": {"parallel": model_parallel},
            "intermediate_size_list": list(
                _mapping(axes.get("ffn_intermediate")).get("values") or ()
            ),
            "attn_heads_list": _attention_heads(axes),
        },
        "search_space": {"axes": axes},
        "sort_sanity": {
            "enabled": bool(pruning.get("sort_sanity", False)),
            "eval_samples": replacement_samples,
            "block_size": sequence_length,
            "packed_token_cache_path": validation_cache,
            "automodel": {"parallel": model_parallel},
        },
        "width_sanity": {"enabled": False},
        "slicing_sanity": {"enabled": False},
        "bypass_sanity": {
            "enabled": bool(bypass.get("sanity", False)),
            "steps": min(128, bypass_samples),
        },
        "bypass": {
            "enabled": bool(bypass.get("enabled", False)),
            "granularity": bypass.get("granularity", "subblock"),
            "automodel": {"parallel": _parallel(bypass_mesh)},
            "iter_num": max(1, bypass_samples // max(1, int(bypass.get("batch_size", 8)))),
            "step_num": max(1, bypass_samples // max(1, int(bypass.get("batch_size", 8)))),
            "data": {
                "block_size": int(bypass.get("sequence_length", sequence_length)),
                "max_eval_samples": bypass_samples,
                "packed_token_cache_path": train_cache,
            },
            "training": {"micro_batch_size": int(bypass.get("batch_size", 8))},
        },
        "depth_importance": {
            "enabled": int(pruning.get("depth_remove", 0)) > 0,
            "granularity": pruning.get("depth_granularity", "subblock"),
            "source_checkpoint_dir": "${teacher_dir}",
            "output_dir": "${puzzle_dir}/depth/iterative",
            "expected_initial_sublayers": int(inventory.get("num_sublayers", 0)),
            "max_removals": int(pruning.get("depth_remove", 0)),
            "max_subblocks_to_remove": int(pruning.get("depth_remove", 0)),
            "metric": "lm_loss",
            "eval_samples": replacement_samples,
            "micro_batch_size": 1,
            "block_size": sequence_length,
            "packed_token_cache_path": validation_cache,
            "automodel": {"parallel": model_parallel},
        },
        "build_library": {
            "enabled": True,
            "include_bypass": bool(bypass.get("enabled", False)),
        },
        "replacement_scoring": {
            "enabled": True,
            "granularity": pruning.get("replacement_granularity", "subblock"),
            "eval_samples": replacement_samples,
            "block_size": sequence_length,
            "dataset_path": data["source"],
            "packed_token_cache_path": validation_cache,
            "automodel": {"parallel": model_parallel},
        },
        "vllm_stats": {
            "enabled": bool(runtime.get("vllm_enabled", False)),
            "batch_sizes": [int(runtime.get("concurrency", 1))],
            "prefill_seq_len": int(runtime.get("isl", sequence_length)),
            "generation_seq_len": int(runtime.get("osl", 1024)),
            "runtime_stats": {
                "enabled": bool(runtime.get("vllm_enabled", False)),
                "max_num_seqs": int(runtime.get("concurrency", 1)),
                "topology": {
                    "tensor_parallel_size": int(common_mesh.get("tp", 1)),
                    "pipeline_parallel_size": int(common_mesh.get("pp", 1)),
                    "prefill_context_parallel_size": int(common_mesh.get("cp", 1)),
                    "decode_context_parallel_size": int(common_mesh.get("cp", 1)),
                    "gpu_group_size": int(common_mesh.get("tp", 1)),
                },
            },
        },
        "mip": {
            "enabled": True,
            "score_granularity": pruning.get("replacement_granularity", "subblock"),
            "workloads": {
                runtime.get("workload_id", "serving-default"): {
                    "isl": int(runtime.get("isl", sequence_length)),
                    "osl": int(runtime.get("osl", 1024)),
                    "batch_size": int(runtime.get("concurrency", 1)),
                    "concurrency": int(runtime.get("concurrency", 1)),
                }
            },
            "runs": _mip_runs(state, smoke=smoke),
        },
        "post_mip": {
            "flows": _post_mip_flows(
                state,
                smoke=smoke,
                common_mesh=common_mesh,
                global_kd_mesh=global_kd_mesh,
            )
        },
        "zero_shot_evaluation": {"enabled": False},
        "aiperf": {"enabled": False},
        "global_distillation_sanity": {"enabled": False},
        "global_distillation": {
            "enabled": False,
            "automodel": {"parallel": _parallel(global_kd_mesh)},
        },
        "post_distillation_evaluation": {"enabled": False},
    }
    return _deep_merge(config, overlay)


def render_runner(state: Mapping[str, Any], budget: str) -> dict[str, Any]:
    """Render one existing-schema Slurm or SSH bare-metal runner mapping."""
    del budget
    infrastructure = _answers(state, "infrastructure")
    runner_answers = _mapping(infrastructure.get("runner"))
    runner = {
        "kind": runner_answers["kind"],
        "execution_contract": deepcopy(infrastructure["execution_contract"]),
    }
    if runner["kind"] == "slurm":
        slurm = deepcopy(_mapping(runner_answers.get("slurm")))
        slurm["partition"] = slurm.get("partition_batch", "batch")
        runner["slurm"] = slurm
    elif runner["kind"] == "baremetal":
        runner["inventory"] = deepcopy(_mapping(runner_answers.get("inventory")))
    else:
        raise SetupError(f"Unsupported generated runner kind: {runner['kind']}")
    return {"runner": runner}


def _dynamic_stage_entries(
    experiment: Mapping[str, Any],
    workers: Mapping[str, Any],
    gpus_per_node: int,
) -> dict[str, Any]:
    entries = {}
    for flow_id, flow in _mapping(_mapping(experiment.get("post_mip")).get("flows")).items():
        for node_id, node in _mapping(flow.get("nodes")).items():
            node_type = str(node.get("type"))
            selector = node_type in {"filter", "manual_filter"}
            entries[f"post.{flow_id}.{node_id}"] = {
                "strategy": "single" if selector else "sharded",
                "instances": 1 if selector else int(workers.get("sharded", 1)),
                "gpus_per_node": gpus_per_node,
            }
    return entries


def render_execution(
    state: Mapping[str, Any], experiment: Mapping[str, Any], budget: str
) -> dict[str, Any]:
    """Render scheduler-neutral execution semantics and dynamic node resources."""
    del budget
    infrastructure = _answers(state, "infrastructure")
    workers = _mapping(infrastructure.get("workers"))
    meshes = _mapping(infrastructure.get("meshes"))
    common = _parallel(_mapping(meshes.get("common")))
    bypass = _parallel(_mapping(meshes.get("bypass")))
    global_kd = _parallel(_mapping(meshes.get("global_kd")))
    gpus_per_node = int(infrastructure.get("gpus_per_node", 8))
    pool_workers = int(workers.get("pool", 1))
    sharded_workers = int(workers.get("sharded", 1))
    embedding_widths = list(_mapping(experiment.get("embedding_pruning")).get("widths") or ())
    stages = {
        "convert": {"strategy": "single", "instances": 1, "parallel": common},
        "tokenize_data": {"strategy": "single", "instances": 1},
        "vllm_stats": {
            "strategy": "sharded",
            "instances": sharded_workers,
            "parallel": common,
        },
        "depth_importance": {
            "strategy": "persistent_pool",
            "instances": pool_workers,
            "parallel": common,
        },
        "width_importance": {"strategy": "single", "instances": 1, "parallel": common},
        "sort": {"strategy": "single", "instances": 1, "parallel": common},
        "sort_sanity": {"strategy": "single", "instances": 1, "parallel": common},
        "bypass_sanity": {"strategy": "single", "instances": 1, "parallel": bypass},
        "bypass": {
            "strategy": "persistent_pool",
            "instances": pool_workers,
            "parallel": bypass,
        },
        "build_library": {"strategy": "single", "instances": 1, "parallel": common},
        "replacement_scoring": {
            "strategy": "sharded" if len(embedding_widths) > 1 else "persistent_pool",
            "instances": sharded_workers if len(embedding_widths) > 1 else pool_workers,
            "parallel": common,
        },
        "mip": {"strategy": "single", "instances": 1},
    }
    stages.update(_dynamic_stage_entries(experiment, workers, gpus_per_node))
    for stage_id, stage in stages.items():
        stage.setdefault("gpus_per_node", gpus_per_node)
        if "short_kd" in stage_id or "global_kd" in stage_id:
            stage["parallel"] = global_kd
    return {
        "execution": {
            "defaults": {
                "failure_policy": "strict",
                "halt_policy": "drain",
                "gpus_per_node": gpus_per_node,
            },
            "stages": stages,
        }
    }


def _bundle_paths(bundle_dir: Path) -> tuple[Path, Path, Path]:
    return (
        bundle_dir / "experiment.yaml",
        bundle_dir / "runner.yaml",
        bundle_dir / "execution.yaml",
    )


def _compile_bundle(bundle_dir: Path):
    experiment_path, runner_path, execution_path = _bundle_paths(bundle_dir)
    runner = load_runner_config(runner_path)
    execution = load_execution_config(execution_path)
    return compile_campaign_plan(
        experiment_config_path=experiment_path,
        runner=runner,
        execution=execution,
        stage_filter="full",
    )


def _submission_payload(submission) -> dict[str, Any]:
    return {
        "stage_id": submission.stage_id,
        "work_id": submission.work_id,
        "attempt_id": submission.attempt_id,
        "nodes": submission.nodes,
        "gpus": submission.gpus,
        "gpus_per_node": submission.gpus_per_node,
        "task_count": submission.task_count,
        "gpus_per_task": submission.gpus_per_task,
        "tasks_per_group": submission.tasks_per_group,
        "group_count": submission.group_count,
        "task_capacity": submission.task_capacity,
        "unused_gpus": submission.unused_gpus,
        "launcher": submission.launcher,
        "exclusive": submission.exclusive,
        "argv": list(submission.argv),
    }


def dry_run_bundle(bundle_dir: Path) -> str:
    """Compile a bundle and serialize its no-submission execution plan."""
    plan = _compile_bundle(bundle_dir)
    submissions = dry_run_plan(plan)
    lines = [
        "Puzzletron orchestration dry run — no jobs submitted",
        f"Stages: {len(plan.stages)}",
        f"Submissions: {len(submissions)}",
        "",
    ]
    for stage in plan.stages:
        count = sum(item.stage_id == stage.stage_id for item in submissions)
        lines.append(
            f"{stage.stage_id}: {count} submission(s), strategy={stage.strategy.value}, "
            f"gpus_per_instance={stage.gpus_per_instance}, nodes={stage.nodes}"
        )
    lines.extend(["", json.dumps([_submission_payload(item) for item in submissions], indent=2)])
    return "\n".join(lines) + "\n"


def validate_bundle(bundle_dir: Path) -> BundleValidation:
    """Validate and dry-run one bundle without constructing an executor."""
    try:
        plan = _compile_bundle(bundle_dir)
        submissions = dry_run_plan(plan)
        return BundleValidation(
            valid=True,
            stage_count=len(plan.stages),
            submission_count=len(submissions),
        )
    except Exception as error:
        return BundleValidation(valid=False, error=f"{type(error).__name__}: {error}")


def _readme(
    campaign_dir: Path,
    state: Mapping[str, Any],
    validations: Mapping[str, BundleValidation],
) -> str:
    model = _mapping(state.get("model"))
    inventory = _mapping(state.get("inventory"))
    data = _answers(state, "data")
    mip = _answers(state, "mip")
    infrastructure = _answers(state, "infrastructure")
    repository = str(_mapping(infrastructure.get("execution_contract")).get("repository", "."))
    orchestrator = Path(repository) / "examples/puzzletron/orchestrate.py"
    rows = infrastructure.get("resource_rows") or ()
    lines = [
        "# Generated Puzzletron campaign",
        "",
        f"- Model: `{model.get('source')}`",
        f"- Resolved revision: `{model.get('resolved_revision') or 'local'}`",
        f"- Profile: `{inventory.get('family')} / {inventory.get('descriptor')}`",
        f"- Dataset modality: `{data.get('modality')}` ({data.get('modality_evidence')})",
        f"- MIP runs: `{', '.join(_mapping(mip.get('runs')))}`",
        "",
        "Run the commands below from this campaign directory.",
        "Smoke and production are independent. Running smoke is recommended but never gates production.",
        "The setup wizard did not submit any jobs.",
        "",
        "## Resource summary",
        "",
        "| Stage group | Instances | GPUs/instance | Nodes |",
        "| --- | ---: | ---: | ---: |",
    ]
    lines.extend(
        (f"| {row['stage']} | {row['instances']} | {row['gpus_per_instance']} | {row['nodes']} |")
        for row in rows
    )
    for budget in ("smoke", "production"):
        validation = validations[budget]
        status = (
            f"valid ({validation.stage_count} stages, {validation.submission_count} submissions)"
            if validation.valid
            else f"invalid: {validation.error}"
        )
        bundle = Path(budget)
        lines.extend(
            [
                "",
                f"## {budget.title()}",
                "",
                f"Validation: **{status}**",
                "",
                "```bash",
                f"python {orchestrator} \\",
                f"  --experiment {bundle / 'experiment.yaml'} \\",
                f"  --runner {bundle / 'runner.yaml'} \\",
                f"  --execution {bundle / 'execution.yaml'} --dry-run",
                "",
                f"python {orchestrator} \\",
                f"  --experiment {bundle / 'experiment.yaml'} \\",
                f"  --runner {bundle / 'runner.yaml'} \\",
                f"  --execution {bundle / 'execution.yaml'}",
                "```",
            ]
        )
    lines.extend(
        [
            "",
            "## Resume setup",
            "",
            "```bash",
            f"python {Path(repository) / 'examples/puzzletron/puzzletron_setup.py'} --resume .",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def build_bundles(campaign_dir: Path, state: Mapping[str, Any]) -> BundleResult:
    """Generate, independently validate, and dry-run smoke and production bundles."""
    campaign_dir = campaign_dir.expanduser().resolve()
    for budget in ("smoke", "production"):
        bundle_dir = campaign_dir / budget
        experiment = render_experiment(state, budget)
        runner = render_runner(state, budget)
        execution = render_execution(state, experiment, budget)
        experiment_path, runner_path, execution_path = _bundle_paths(bundle_dir)
        _atomic_yaml(experiment_path, experiment)
        _atomic_yaml(runner_path, runner)
        _atomic_yaml(execution_path, execution)

    validations = {
        budget: validate_bundle(campaign_dir / budget) for budget in ("smoke", "production")
    }
    for budget, validation in validations.items():
        plan_path = campaign_dir / budget / "dry-run-plan.txt"
        if validation.valid:
            _atomic_text(plan_path, dry_run_bundle(campaign_dir / budget))
            print(
                f"{budget.title()} bundle valid: {validation.stage_count} stages, "
                f"{validation.submission_count} submissions."
            )
        else:
            _atomic_text(
                plan_path,
                "Puzzletron orchestration dry run failed — no jobs submitted\n"
                f"{validation.error}\n",
            )
            print(f"{budget.title()} bundle validation failed: {validation.error}")
    _atomic_text(campaign_dir / "README.md", _readme(campaign_dir, state, validations))
    return BundleResult(
        campaign_dir=campaign_dir,
        smoke=validations["smoke"],
        production=validations["production"],
    )
