#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Solve a profile-scoped width/depth MIP grid, with optional realization."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf
from safetensors import safe_open

from modelopt.torch.puzzletron.anymodel.registry import resolve_descriptor_from_pretrained
from modelopt.torch.puzzletron.granularity import resolve_granularity
from modelopt.torch.puzzletron.mip.run_puzzle import (
    _add_block_stats_to_gathered_metrics,
    filter_subblock_stats_by_args,
    gather_composed_subblock_puzzle_metrics,
    gather_multi_layer_puzzle_metrics,
    run_puzzle,
)
from modelopt.torch.puzzletron.mip.profiles import (
    DepthSelection,
    compile_profile_constraints,
    normalize_mip_profiles,
)
from modelopt.torch.puzzletron.pipeline_config import pipeline_config_from_path
from modelopt.torch.puzzletron.replacement_library.library import ReplacementLibrary
from modelopt.torch.puzzletron.replacement_library.replacement_utils import (
    parse_layer_replacement,
)


def _atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _depth_stage_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return canonical depth-importance settings with a legacy fallback."""

    return dict(config.get("depth_importance") or config.get("depth") or {})


def _checkpoint_parameter_count(checkpoint: Path) -> int:
    index_path = checkpoint / "model.safetensors.index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text())
        files = sorted(set(index["weight_map"].values()))
    elif (checkpoint / "model.safetensors").is_file():
        files = ["model.safetensors"]
    else:
        raise FileNotFoundError(f"realized checkpoint has no safetensors: {checkpoint}")
    total = 0
    for relative in files:
        with safe_open(str(checkpoint / relative), framework="pt") as handle:
            for key in handle.keys():
                total += math.prod(int(dim) for dim in handle.get_slice(key).get_shape())
    return int(total)


def _one_solution(path: Path) -> dict:
    raw = json.loads(path.read_text())
    if len(raw) != 1:
        raise RuntimeError(f"expected one MIP solution in {path}, found {len(raw)}")
    return raw[0]


def _requested_depths(selected: list, *, max_depth: int) -> tuple[int, ...]:
    """Return every realizable depth from the parent through the configured cap."""

    return tuple(range(min(int(max_depth), len(selected)) + 1))


def _forced_removals_for_depth(
    selected: list[dict[str, Any]], depth_selection: DepthSelection
) -> list[dict[str, Any]]:
    """Resolve one total or typed selection against the global depth ranking."""

    counts = depth_selection.as_dict()
    if "total" in counts:
        requested = counts["total"]
        if requested > len(selected):
            raise ValueError(
                f"depth selection requests total={requested} but trajectory contains "
                f"{len(selected)} removals"
            )
        return list(selected[:requested])

    available = Counter(str(item["kind"]) for item in selected)
    chosen = Counter()
    removals = []
    for item in selected:
        kind = str(item["kind"])
        if chosen[kind] < counts.get(kind, 0):
            removals.append(item)
            chosen[kind] += 1
    missing = [
        (kind, requested, available[kind])
        for kind, requested in counts.items()
        if chosen[kind] != requested
    ]
    if missing:
        kind, requested, present = missing[0]
        raise ValueError(
            f"depth selection requests {kind}={requested} but trajectory contains {present}"
        )
    return removals


def _expected_scenario_count(widths: list[int], *, max_depth: int) -> int:
    return len(widths) * (int(max_depth) + 1)


def _replacement_score_paths(
    base_dir: Path, granularity: str
) -> tuple[Path, Path]:
    """Resolve width-local score inputs without changing full-block identities."""

    granularity = str(granularity).lower()
    validation_names = {
        "block": "single_sequence_replacement_solutions--validation",
        "subblock": "single_subblock_replacement_solutions--validation",
    }
    if granularity not in validation_names:
        raise ValueError(f"unsupported MIP score_granularity={granularity!r}")
    return (
        base_dir / validation_names[granularity],
        base_dir / "single_sequence_replacement_solutions.json",
    )


def _profile_id(parameter_ratio: float) -> str:
    return _constraint_profile_id("parameter_ratio", parameter_ratio)


def _constraint_profile_id(constraint_type: str, ratio: float) -> str:
    prefixes = {"parameter_ratio": "params", "latency_ratio": "latency"}
    if constraint_type not in prefixes:
        raise ValueError(f"unsupported ratio constraint type {constraint_type!r}")
    percent = ratio * 100
    text = f"{percent:.4f}".rstrip("0").rstrip(".").replace(".", "p")
    return f"{prefixes[constraint_type]}-{text.zfill(3)}"


def _load_completed_scenario(
    scenario_root: Path,
    *,
    profile_id: str,
    width: int,
    depth_selection: DepthSelection,
    constraint_type: str,
    solve_only: bool,
) -> dict[str, Any] | None:
    """Return an atomically completed matching scenario, otherwise rerun it."""
    path = scenario_root / "scenario_manifest.json"
    if not path.is_file():
        return None
    scenario = json.loads(path.read_text())
    recorded_selection = scenario.get("depth_selection")
    if recorded_selection is None:
        recorded_selection = {"total": int(scenario.get("removed_sublayers", -1))}
    if not isinstance(recorded_selection, dict):
        return None
    identity = (
        str(scenario.get("profile_id")),
        int(scenario.get("hidden_width", -1)),
        {str(kind): int(count) for kind, count in recorded_selection.items()},
        str(scenario.get("constraint_type")),
    )
    expected = (profile_id, int(width), depth_selection.as_dict(), constraint_type)
    if identity != expected or scenario.get("status") not in {"feasible", "infeasible"}:
        return None
    if scenario["status"] == "feasible" and not solve_only:
        checkpoint_value = scenario.get("checkpoint")
        if not checkpoint_value:
            return None
        checkpoint = Path(str(checkpoint_value))
        if not (checkpoint / "config.json").is_file():
            return None
    return scenario


def _stats_profile(stats_path: Path, *, runtime_stats: bool) -> dict[str, Any]:
    payload = json.loads(stats_path.read_text())
    profiles = [
        row
        for row in payload
        if isinstance(row, dict)
        and bool((row.get("args") or {}).get("runtime_stats", False)) is runtime_stats
    ]
    if not profiles:
        raise RuntimeError(
            f"expected a {'measured runtime' if runtime_stats else 'static'} profile "
            f"in {stats_path}, found none"
        )
    if len(profiles) == 1:
        return profiles[0]
    if runtime_stats:
        raise RuntimeError(
            f"expected exactly one measured runtime profile in {stats_path}, "
            f"found {len(profiles)}"
        )

    def parameter_inventory(profile: dict[str, Any]) -> tuple[Any, tuple[Any, ...]]:
        return (
            (profile.get("non_block") or {}).get("num_params"),
            tuple(
                row.get("num_params")
                for row in profile.get("subblocks", ())
                if isinstance(row, dict)
            ),
        )

    inventories = {parameter_inventory(profile) for profile in profiles}
    if len(inventories) != 1:
        raise RuntimeError(
            f"static profiles in {stats_path} have conflicting parameter inventories"
        )
    return profiles[0]


def _report_costs(runtime_profile: dict[str, Any]) -> list[str]:
    fields = {
        key
        for row in runtime_profile.get("subblocks", ())
        if isinstance(row, dict)
        for key, value in row.items()
        if key != "parent_layer_index"
        and isinstance(value, (int, float))
        and not isinstance(value, bool)
    }
    fields.update(
        {
            "has_attention",
            "has_mamba",
            "has_ffn",
            "has_moe",
            "not_no_op",
            "num_kv_heads",
            "num_query_heads",
            "num_experts",
            "top_k",
        }
    )
    return [f"stats.{field}" for field in sorted(fields)]


def _teacher_costs(
    scoring_dir: Path,
    stats_path: Path,
    runtime_args: dict[str, Any],
    *,
    score_granularity: str,
    canonical_solutions_path: Path,
) -> dict[str, float]:
    if score_granularity == "subblock":
        gathered = gather_composed_subblock_puzzle_metrics(
            canonical_solutions_path,
            scoring_dir,
        )
    else:
        gathered = gather_multi_layer_puzzle_metrics(scoring_dir)
    stats = filter_subblock_stats_by_args(
        json.loads(stats_path.read_text()), runtime_args
    )
    _add_block_stats_to_gathered_metrics(gathered, stats)
    totals: dict[str, float] = {
        key: float(value)
        for key, value in (stats.get("non_block") or {}).items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }
    teachers = [
        row
        for row in gathered.values()
        if isinstance(row, dict) and row.get("is_teacher") is True
    ]
    if not teachers:
        raise RuntimeError(f"no teacher blocks were reconstructed from {scoring_dir}")
    for row in teachers:
        for key, value in (row.get("stats") or {}).items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                totals[key] = totals.get(key, 0.0) + float(value)
    return {f"stats.{key}": value for key, value in totals.items()}


def _teacher_summary_costs(
    base_costs: dict[str, float],
    workload_costs: dict[str, dict[str, float]],
) -> dict[str, float]:
    """Persist base and named-workload teacher denominators in one inventory."""

    summary = {str(key): float(value) for key, value in base_costs.items()}
    for workload, costs in workload_costs.items():
        summary.update(
            {f"{key}@{workload}": float(value) for key, value in costs.items()}
        )
    return summary


def _solution_score(solution: dict[str, Any], objective: str) -> float | None:
    total = solution.get("total_value")
    if isinstance(total, dict):
        value = total.get(objective)
    else:
        value = total
    return float(value) if isinstance(value, (int, float)) else None


def _verify_direct_constraints(
    total_costs: dict[str, float],
    constraints: dict[str, Any],
    *,
    context: str,
) -> None:
    for key, bound in constraints.items():
        if isinstance(bound, (list, tuple)):
            minimum, maximum = bound
        else:
            minimum, maximum = None, bound
        value = total_costs[key]
        if minimum is not None and value < minimum and not math.isclose(value, minimum):
            raise RuntimeError(
                f"MIP constraint violated for {context}: {key}={value} < {minimum}"
            )
        if maximum is not None and value > maximum and not math.isclose(value, maximum):
            raise RuntimeError(
                f"MIP constraint violated for {context}: {key}={value} > {maximum}"
            )


def _sliced_teacher_baseline(scoring_dir: Path, objective: str) -> float:
    metric = objective.removeprefix("metrics.")
    paths = sorted(scoring_dir.glob("solution_*.json"))
    if not paths:
        raise FileNotFoundError(f"no replacement scores found in {scoring_dir}")
    payload = json.loads(paths[0].read_text())
    value = (payload.get("sliced_teacher_baseline") or {}).get(metric)
    if isinstance(value, dict):
        value = value.get("avg")
    if not isinstance(value, (int, float)):
        raise KeyError(f"sliced teacher baseline has no numeric {metric!r}: {paths[0]}")
    return float(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--parameter-ratio",
        type=float,
        action="append",
        help="Repeat to solve multiple profiles; defaults to mip.constraint_profiles or 0.85.",
    )
    parser.add_argument(
        "--latency-ratio",
        type=float,
        action="append",
        help=(
            "Repeat to solve latency profiles against the original full-width teacher; "
            "defaults to mip.latency_constraint_profiles when configured."
        ),
    )
    parser.add_argument(
        "--solve-only",
        action="store_true",
        help="Write MIP architectures without physically materializing checkpoints.",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        help="Override the configured depth limit; use 0 to select only depth-00 scenarios.",
    )
    args = parser.parse_args()
    cfg = pipeline_config_from_path(args.config)
    puzzle_dir = Path(cfg["puzzle_dir"])
    widths = [int(width) for width in cfg["embedding_pruning"]["widths"]]
    depth_config = _depth_stage_config(cfg)
    configured_max_depth = int(
        depth_config.get(
            "max_subblocks_to_remove",
            depth_config.get("max_removals", 0),
        )
    )
    max_depth = configured_max_depth if args.max_depth is None else int(args.max_depth)
    if max_depth < 0:
        raise ValueError(f"--max-depth must be non-negative, got {max_depth}")
    objective = "metrics.raw_replacement_loss"
    mip_config = dict(cfg.get("mip") or {})
    score_granularity = str(mip_config.get("score_granularity", "block")).lower()
    depth_granularity = resolve_granularity("depth", depth_config)
    selected: list[dict[str, Any]] = []
    if max_depth:
        trajectory_path = puzzle_dir / "depth" / "iterative" / "trajectory.json"
        trajectory = json.loads(trajectory_path.read_text())
        trajectory_granularity = str(trajectory.get("granularity", depth_granularity))
        if trajectory_granularity != depth_granularity:
            raise RuntimeError(
                "depth trajectory granularity does not match the campaign config: "
                f"artifact={trajectory_granularity} config={depth_granularity}"
            )
        selected = list(trajectory.get("selected") or [])
        if len(selected) < max_depth:
            raise RuntimeError(
                f"depth trajectory has only {len(selected)} removals but {max_depth} "
                f"were requested: {trajectory_path}"
            )
        selected = selected[:max_depth]
    available_depth_counts = Counter(str(item["kind"]) for item in selected)
    named_profiles = normalize_mip_profiles(
        mip_config,
        available_depths=range(max_depth + 1),
        available_embeddings=widths,
        available_depth_counts=available_depth_counts,
        depth_granularity=depth_granularity,
    )
    if named_profiles and (args.parameter_ratio or args.latency_ratio):
        raise ValueError(
            "--parameter-ratio/--latency-ratio cannot be combined with mip.profiles"
        )
    parameter_ratios = list(
        args.parameter_ratio
        or ([] if named_profiles else mip_config.get("constraint_profiles", []))
        or []
    )
    latency_ratios = list(
        args.latency_ratio
        or ([] if named_profiles else mip_config.get("latency_constraint_profiles", []))
        or []
    )
    if not named_profiles and not parameter_ratios and not latency_ratios:
        parameter_ratios = [0.85]
    legacy_profiles = [
        (constraint_type, float(ratio))
        for constraint_type, values in (
            ("parameter_ratio", parameter_ratios),
            ("latency_ratio", latency_ratios),
        )
        for ratio in values
    ]
    invalid = [ratio for _, ratio in legacy_profiles if not 0 < ratio <= 1]
    if invalid:
        raise ValueError(f"constraint ratios must be in (0, 1], got {invalid}")

    uses_runtime = any(kind == "latency_ratio" for kind, _ in legacy_profiles)
    if uses_runtime and any(kind == "parameter_ratio" for kind, _ in legacy_profiles):
        raise ValueError("parameter and latency profiles must be solved in separate invocations")

    teacher_width = max(widths)
    descriptor_checkpoint = (
        puzzle_dir
        / "scenarios"
        / f"width-{teacher_width:04d}"
        / "depth-00"
        / "ckpts"
        / "sorted_teacher"
    )
    actual_teacher_params = _checkpoint_parameter_count(descriptor_checkpoint)
    width_inputs: dict[int, dict[str, Any]] = {}
    for width in widths:
        base_dir = puzzle_dir / "scenarios" / f"width-{width:04d}" / "depth-00"
        stats_path = base_dir / "subblock_stats.json"
        scoring_dir, canonical_solutions_path = _replacement_score_paths(
            base_dir,
            score_granularity,
        )
        stats_payload = json.loads(stats_path.read_text())
        stats_profile = _stats_profile(
            stats_path,
            runtime_stats=uses_runtime if not named_profiles else False,
        )
        teacher_costs = _teacher_costs(
            scoring_dir,
            stats_path,
            stats_profile["args"],
            score_granularity=score_granularity,
            canonical_solutions_path=canonical_solutions_path,
        )
        workload_profiles = {}
        workload_teacher_costs = {}
        for workload_name, workload_args in dict(mip_config.get("workloads") or {}).items():
            aliases = {
                "isl": "prefill_seq_len",
                "osl": "generation_seq_len",
                "concurrency": "max_num_seqs",
            }
            normalized_args = {
                aliases.get(str(key), str(key)): value
                for key, value in dict(workload_args).items()
            }
            normalized_args["n_embd"] = width
            workload_profile = filter_subblock_stats_by_args(
                stats_payload,
                normalized_args,
                convert_dicts_to_dataclasses=False,
            )
            workload_profiles[str(workload_name)] = workload_profile
            workload_teacher_costs[str(workload_name)] = _teacher_costs(
                scoring_dir,
                stats_path,
                workload_profile["args"],
                score_granularity=score_granularity,
                canonical_solutions_path=canonical_solutions_path,
            )
        width_inputs[width] = {
            "base_dir": base_dir,
            "stats_path": stats_path,
            "scoring_dir": scoring_dir,
            "canonical_solutions_path": canonical_solutions_path,
            "stats_profile": stats_profile,
            "teacher_costs": teacher_costs,
            "workload_profiles": workload_profiles,
            "workload_teacher_costs": workload_teacher_costs,
            "report_costs": _report_costs(stats_profile),
            "sliced_teacher_baseline": _sliced_teacher_baseline(
                scoring_dir, objective
            ),
        }

    formula_teacher_params = int(
        round(width_inputs[teacher_width]["teacher_costs"]["stats.num_params"])
    )
    if formula_teacher_params != actual_teacher_params:
        raise RuntimeError(
            "full-width teacher parameter formula/checkpoint mismatch: "
            f"formula={formula_teacher_params} actual={actual_teacher_params}"
        )
    full_teacher_runtime_ms = (
        float(width_inputs[teacher_width]["teacher_costs"]["stats.runtime_ms"])
        if uses_runtime
        else None
    )

    descriptor = None
    libraries: dict[int, ReplacementLibrary] = {}
    if not args.solve_only:
        model_cfg = cfg.get("model") or {}
        descriptor = resolve_descriptor_from_pretrained(
            str(descriptor_checkpoint),
            trust_remote_code=bool(model_cfg.get("trust_remote_code", False)),
            descriptor_override=model_cfg.get("descriptor_override"),
        ).descriptor
        libraries = {
            width: ReplacementLibrary(
                width_inputs[width]["base_dir"] / "replacement_library.json", descriptor
            )
            for width in widths
        }

    run_profiles = []
    teacher_inputs = width_inputs[teacher_width]
    teacher_totals = {
        None: {
            key.removeprefix("stats."): value
            for key, value in teacher_inputs["teacher_costs"].items()
        }
    }
    teacher_totals.update(
        {
            workload: {
                key.removeprefix("stats."): value for key, value in costs.items()
            }
            for workload, costs in teacher_inputs["workload_teacher_costs"].items()
        }
    )
    for profile in named_profiles:
        direct_constraints = compile_profile_constraints(
            profile,
            teacher_totals=teacher_totals,
        )
        run_profiles.append(
            {
                "profile_id": profile.profile_id,
                "constraint_type": "named_profile",
                "constraint_label": profile.profile_id,
                "constraint_fields": {
                    "base_profile_id": profile.base_profile_id,
                    "constraints": direct_constraints,
                },
                "human_constraints": None,
                "mip_constraints": direct_constraints,
                "widths": profile.embedding_widths,
                "depth_selections": profile.depth_selections,
                "axes_default": profile.axes_default,
                "axis_options": profile.axis_options,
                "num_homogeneous_solutions": profile.num_homogeneous_solutions,
                "required_workloads": profile.required_workloads,
            }
        )
    for constraint_type, ratio in legacy_profiles:
        profile_id = _constraint_profile_id(constraint_type, ratio)
        parameter_limit = int(math.floor(formula_teacher_params * ratio))
        latency_limit_ms = full_teacher_runtime_ms * ratio if uses_runtime else None
        if constraint_type == "parameter_ratio":
            constraint_label = f"{ratio * 100:g}% params"
            human_constraints = {"num_params": parameter_limit}
            constraint_fields = {
                "parameter_ratio": ratio,
                "parameter_limit": parameter_limit,
                "parameter_denominator": formula_teacher_params,
                "parameter_denominator_scope": "original_full_width_teacher",
            }
        else:
            constraint_label = f"{ratio * 100:g}% latency"
            human_constraints = {"target_latency_seconds": latency_limit_ms / 1000.0}
            constraint_fields = {
                "latency_ratio": ratio,
                "latency_limit_ms": latency_limit_ms,
                "latency_denominator_ms": full_teacher_runtime_ms,
                "latency_denominator_scope": "original_full_width_teacher",
            }
        run_profiles.append(
            {
                "profile_id": profile_id,
                "constraint_type": constraint_type,
                "constraint_label": constraint_label,
                "constraint_fields": constraint_fields,
                "human_constraints": human_constraints,
                "mip_constraints": None,
                "widths": tuple(widths),
                "depth_selections": tuple(
                    DepthSelection.total_prefix(depth)
                    for depth in _requested_depths(selected, max_depth=max_depth)
                ),
                "axes_default": "all",
                "axis_options": {},
                "num_homogeneous_solutions": 0,
                "required_workloads": (),
            }
        )

    profile_summaries = []
    for run_profile in run_profiles:
        profile_id = run_profile["profile_id"]
        constraint_type = run_profile["constraint_type"]
        constraint_label = run_profile["constraint_label"]
        constraint_fields = run_profile["constraint_fields"]
        human_constraints = run_profile["human_constraints"]
        profile_root = puzzle_dir / "mip" / "profiles" / profile_id
        summary = {
            "version": 1,
            "profile": {
                "id": profile_id,
                "label": constraint_label,
                "constraint_type": constraint_type,
                **constraint_fields,
            },
            "teacher": {
                "label": "Teacher",
                "hidden_width": teacher_width,
                "removed_sublayers": 0,
                "status": "reference",
                "score": width_inputs[teacher_width]["sliced_teacher_baseline"],
                "sliced_teacher_baseline": width_inputs[teacher_width][
                    "sliced_teacher_baseline"
                ],
                "total_costs": _teacher_summary_costs(
                    width_inputs[teacher_width]["teacher_costs"],
                    width_inputs[teacher_width]["workload_teacher_costs"],
                ),
                "parameter_ratio": 1.0,
                "checkpoint": str(descriptor_checkpoint),
            },
            "runtime_profile": {
                workload: teacher_inputs["workload_profiles"][workload]["args"]
                for workload in run_profile["required_workloads"]
            }
            or teacher_inputs["stats_profile"]["args"],
            "selection_evidence": {
                "kind": "named_multi_constraint"
                if constraint_type == "named_profile"
                else "measured_vllm_latency"
                if uses_runtime
                else "static_parameter_budget",
                "verified": bool(run_profile["required_workloads"]) or uses_runtime,
            },
            "solve_only": bool(args.solve_only),
            "expected_scenario_count": len(run_profile["widths"])
            * len(run_profile["depth_selections"]),
            "scenarios": [],
        }

        for width in run_profile["widths"]:
            inputs = width_inputs[width]
            for depth_selection in run_profile["depth_selections"]:
                depth = depth_selection.total
                forced_removals = _forced_removals_for_depth(selected, depth_selection)
                scenario_root = (
                    profile_root
                    / "scenarios"
                    / f"width-{width:04d}"
                    / depth_selection.slug
                )
                completed = _load_completed_scenario(
                    scenario_root,
                    profile_id=profile_id,
                    width=width,
                    depth_selection=depth_selection,
                    constraint_type=constraint_type,
                    solve_only=bool(args.solve_only),
                )
                if completed is not None:
                    summary["scenarios"].append(completed)
                    continue
                mip_cfg = OmegaConf.create(dict(cfg["mip"]))
                OmegaConf.set_struct(mip_cfg, False)
                mip_cfg.puzzle_profile = None
                mip_cfg.gathered_metrics_path = None
                mip_cfg.single_block_replacement_validation_dir = inputs["scoring_dir"]
                mip_cfg.canonical_solutions_path = inputs["canonical_solutions_path"]
                mip_cfg.subblock_stats_path = inputs["stats_path"]
                mip_cfg.output_path = scenario_root / "puzzle_solutions"
                mip_cfg.objective = objective
                mip_cfg.bigger_is_better = False
                if run_profile["mip_constraints"] is None:
                    mip_cfg.human_constraints = human_constraints
                    mip_cfg.pop("mip_constraints", None)
                else:
                    mip_cfg.human_constraints = None
                    mip_cfg.mip_constraints = dict(run_profile["mip_constraints"])
                mip_cfg.forced_removals = forced_removals
                mip_cfg.materialization_tp = int((cfg.get("parallel") or {}).get("tp", 1))
                mip_cfg.subblock_stats_args = dict(inputs["stats_profile"]["args"])
                mip_cfg.workload_stats_args = {
                    workload: dict(inputs["workload_profiles"][workload]["args"])
                    for workload in run_profile["required_workloads"]
                }
                mip_cfg.axes_default = run_profile["axes_default"]
                mip_cfg.axis_options = dict(run_profile["axis_options"])
                mip_cfg.num_homogeneous_solutions = run_profile[
                    "num_homogeneous_solutions"
                ]
                mip_cfg.report_additional_costs = sorted(
                    {
                        *inputs["report_costs"],
                        *(run_profile["mip_constraints"] or {}),
                    }
                )
                paths = run_puzzle(mip_cfg)
                if len(paths) != 1:
                    raise RuntimeError(
                        f"profile={profile_id} width={width} "
                        f"depth={depth_selection.slug} produced "
                        f"{len(paths)} solution files"
                    )
                solution_path = Path(paths[0])
                raw_solutions = json.loads(solution_path.read_text())
                scenario: dict[str, Any] = {
                    "profile_id": profile_id,
                    "hidden_width": width,
                    "removed_sublayers": depth,
                    "depth_selection": depth_selection.as_dict(),
                    "forced_removals": forced_removals,
                    "solution_path": str(solution_path),
                    "constraint_type": constraint_type,
                    **constraint_fields,
                    "status": "infeasible" if not raw_solutions else "feasible",
                }
                if raw_solutions:
                    if len(raw_solutions) != 1:
                        raise RuntimeError(
                            f"expected one solution in {solution_path}, found {len(raw_solutions)}"
                        )
                    solution = raw_solutions[0]
                    total_costs = {
                        str(key): float(value)
                        for key, value in (solution.get("total_costs") or {}).items()
                        if isinstance(value, (int, float)) and not isinstance(value, bool)
                    }
                    parameter_count = int(round(total_costs["stats.num_params"]))
                    chosen_count = len(solution.get("chosen_replacements") or [])
                    solver_objective_sum = _solution_score(solution, objective)
                    baseline = float(inputs["sliced_teacher_baseline"])
                    score = (
                        solver_objective_sum - (chosen_count - 1) * baseline
                        if solver_objective_sum is not None
                        else None
                    )
                    scenario.update(
                        {
                            "score": score,
                            "solver_objective_sum": solver_objective_sum,
                            "sliced_teacher_baseline": baseline,
                            "total_costs": total_costs,
                            "parameter_count": parameter_count,
                            "parameter_ratio": parameter_count / formula_teacher_params,
                            "chosen_replacement_count": chosen_count,
                            "solution_repr": solution.get("solution_repr"),
                        }
                    )
                    if constraint_type == "parameter_ratio" and parameter_count > int(
                        constraint_fields["parameter_limit"]
                    ):
                        raise RuntimeError(
                            f"MIP parameter constraint violated for {profile_id} "
                            f"width={width} depth={depth_selection.slug}: {parameter_count}>"
                            f"{constraint_fields['parameter_limit']}"
                        )
                    if constraint_type == "latency_ratio" and float(
                        total_costs["stats.runtime_ms"]
                    ) > float(constraint_fields["latency_limit_ms"]):
                        raise RuntimeError(
                            f"MIP latency constraint violated for {profile_id} "
                            f"width={width} depth={depth_selection.slug}: "
                            f"{total_costs['stats.runtime_ms']}>"
                            f"{constraint_fields['latency_limit_ms']}"
                        )
                    if constraint_type == "named_profile":
                        _verify_direct_constraints(
                            total_costs,
                            run_profile["mip_constraints"],
                            context=(
                                f"{profile_id} width={width} depth={depth_selection.slug}"
                            ),
                        )
                    if not args.solve_only:
                        replacements = [
                            parse_layer_replacement(item["layer_replacement"])
                            for item in solution["chosen_replacements"]
                        ]
                        library = libraries[width]
                        model_config = library.create_model_config(replacements)
                        checkpoint = scenario_root / "checkpoints" / "solution_0"
                        library.materialize_checkpoint(
                            replacements,
                            checkpoint,
                            model_config=model_config,
                            solution_identity=(
                                f"{profile_id}-width-{width:04d}-{depth_selection.slug}"
                            ),
                        )
                        actual_params = _checkpoint_parameter_count(checkpoint)
                        if actual_params != parameter_count:
                            raise RuntimeError(
                                "parameter formula/materialization mismatch for "
                                f"profile={profile_id} width={width} "
                                f"depth={depth_selection.slug}: "
                                f"formula={parameter_count} actual={actual_params}"
                            )
                        scenario["checkpoint"] = str(checkpoint)
                homogeneous_path = solution_path.with_name(
                    "homogeneous_solutions.json"
                )
                if homogeneous_path.is_file():
                    homogeneous_records = []
                    for index, homogeneous in enumerate(
                        json.loads(homogeneous_path.read_text())
                    ):
                        homogeneous_costs = {
                            str(key): float(value)
                            for key, value in homogeneous["total_costs"].items()
                        }
                        if run_profile["mip_constraints"] is not None:
                            _verify_direct_constraints(
                                homogeneous_costs,
                                run_profile["mip_constraints"],
                                context=(
                                    f"homogeneous {profile_id} width={width} "
                                    f"depth={depth_selection.slug} rank={index}"
                                ),
                            )
                        chosen_count = len(homogeneous["chosen_replacements"])
                        objective_sum = _solution_score(homogeneous, objective)
                        baseline = float(inputs["sliced_teacher_baseline"])
                        record = {
                            "rank": index,
                            "score": objective_sum
                            - (chosen_count - 1) * baseline,
                            "solver_objective_sum": objective_sum,
                            "total_costs": homogeneous_costs,
                            "parameter_count": int(
                                round(homogeneous_costs["stats.num_params"])
                            ),
                            "homogeneous_assignment": homogeneous[
                                "homogeneous_assignment"
                            ],
                            "solution_repr": homogeneous.get("solution_repr"),
                        }
                        if not args.solve_only:
                            replacements = [
                                parse_layer_replacement(item["layer_replacement"])
                                for item in homogeneous["chosen_replacements"]
                            ]
                            library = libraries[width]
                            model_config = library.create_model_config(replacements)
                            checkpoint = (
                                scenario_root
                                / "checkpoints"
                                / f"homogeneous_{index}"
                            )
                            library.materialize_checkpoint(
                                replacements,
                                checkpoint,
                                model_config=model_config,
                                solution_identity=(
                                    f"{profile_id}-width-{width:04d}-"
                                    f"{depth_selection.slug}-homogeneous-{index}"
                                ),
                            )
                            actual_params = _checkpoint_parameter_count(checkpoint)
                            if actual_params != record["parameter_count"]:
                                raise RuntimeError(
                                    "homogeneous parameter formula/materialization "
                                    f"mismatch: formula={record['parameter_count']} "
                                    f"actual={actual_params}"
                                )
                            record["checkpoint"] = str(checkpoint)
                        homogeneous_records.append(record)
                    scenario["homogeneous_solution_path"] = str(homogeneous_path)
                    scenario["homogeneous_solutions"] = homogeneous_records
                _atomic_json(scenario_root / "scenario_manifest.json", scenario)
                summary["scenarios"].append(scenario)

        expected_scenarios = len(run_profile["widths"]) * len(
            run_profile["depth_selections"]
        )
        if len(summary["scenarios"]) != expected_scenarios:
            raise RuntimeError(
                f"expected {expected_scenarios} width/depth scenarios, "
                f"found {len(summary['scenarios'])}"
            )
        _atomic_json(profile_root / "mip_grid.json", summary)
        profile_summaries.append(
            {
                "id": profile_id,
                "label": summary["profile"]["label"],
                "path": str(profile_root / "mip_grid.json"),
                "scenario_count": len(summary["scenarios"]),
                "feasible_count": sum(
                    row.get("status") == "feasible" for row in summary["scenarios"]
                ),
            }
        )

    index_path = puzzle_dir / "mip" / "profiles" / "index.json"
    existing = json.loads(index_path.read_text()) if index_path.is_file() else {"profiles": []}
    by_id = {row["id"]: row for row in existing.get("profiles", [])}
    by_id.update({row["id"]: row for row in profile_summaries})
    _atomic_json(
        index_path,
        {"version": 1, "profiles": sorted(by_id.values(), key=lambda x: x["id"])},
    )
    print(json.dumps(profile_summaries, indent=2))


if __name__ == "__main__":
    main()
