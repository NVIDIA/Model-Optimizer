#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare deduplicated MIP architectures for resident-parent online evaluation."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _solutions(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, list):
        raise RuntimeError(f"expected a list of MIP solutions in {path}")
    return [dict(row) for row in payload]


def _layer_replacement(row: dict[str, Any]) -> dict[str, Any]:
    replacement = row.get("layer_replacement", row)
    return {
        "parent_layer_indices": [int(value) for value in replacement["parent_layer_indices"]],
        "child_block_configs": replacement["child_block_configs"],
    }


def _architecture_id(solution: dict[str, Any], width: int) -> str:
    replacements = sorted(
        (_layer_replacement(row) for row in solution["chosen_replacements"]),
        key=lambda row: tuple(row["parent_layer_indices"]),
    )
    canonical = json.dumps(
        {"hidden_width": int(width), "replacements": replacements},
        sort_keys=True,
        separators=(",", ":"),
    )
    return "arch-" + hashlib.sha256(canonical.encode()).hexdigest()[:20]


def _depth_slug(scenario: dict[str, Any]) -> str:
    selection = scenario.get("depth_selection") or {
        "total": int(scenario.get("removed_sublayers", 0))
    }
    return "-".join(f"{key}-{int(value):02d}" for key, value in sorted(selection.items()))


def online_execution_contract(puzzle_dir: Path, width: int) -> dict[str, Any]:
    """Describe the immutable checkpoint roles used by one online worker."""

    scenario = puzzle_dir / "scenarios" / f"width-{int(width):04d}" / "depth-00"
    manifest = json.loads((scenario / "scenario_manifest.json").read_text())
    parent = Path(str(manifest["parent_checkpoint"]))
    bypass = manifest.get("bypass_checkpoint")
    return {
        "mode": "resident_sorted_teacher_online",
        "materialized_solution_checkpoints": False,
        "model_loads_per_worker": 1,
        "checkpoint_roles": {
            "source": str(parent),
            "target": str(parent),
            "bypass_overlay": str(bypass) if bypass is not None else None,
        },
    }


def build_online_evaluation_plan(
    puzzle_dir: Path,
    profile_ids: tuple[str, ...] | list[str],
) -> dict[str, Any]:
    """Collect every logical profile solution and deduplicate exact architectures."""

    architectures: dict[int, dict[str, dict[str, Any]]] = {}
    logical_solution_count = 0
    for profile_id in profile_ids:
        grid_path = puzzle_dir / "mip" / "profiles" / profile_id / "mip_grid.json"
        grid = json.loads(grid_path.read_text())
        for scenario in grid.get("scenarios", ()):
            if scenario.get("status") != "feasible":
                continue
            width = int(scenario["hidden_width"])
            depth_slug = _depth_slug(scenario)
            mixed = _solutions(Path(scenario["solution_path"]))
            mixed_records = list(scenario.get("solutions") or ())
            if len(mixed) != len(mixed_records):
                raise RuntimeError(
                    "MIP solution/metadata cardinality mismatch for "
                    f"{profile_id} width={width} depth={depth_slug}: "
                    f"{len(mixed)} != {len(mixed_records)}"
                )
            candidates: list[tuple[str, int | None, dict[str, Any], dict[str, Any]]] = [
                ("mixed", int(record.get("rank", index)), dict(solution), record)
                for index, (solution, record) in enumerate(zip(mixed, mixed_records))
            ]
            homogeneous_path = scenario.get("homogeneous_solution_path")
            if homogeneous_path:
                homogeneous = json.loads(Path(homogeneous_path).read_text())
                records = list(scenario.get("homogeneous_solutions") or ())
                if len(homogeneous) != len(records):
                    raise RuntimeError(
                        "homogeneous solution/metadata cardinality mismatch for "
                        f"{profile_id} width={width} depth={depth_slug}: "
                        f"{len(homogeneous)} != {len(records)}"
                    )
                candidates.extend(
                    ("homogeneous", int(record.get("rank", index)), dict(solution), record)
                    for index, (solution, record) in enumerate(zip(homogeneous, records))
                )

            by_id = architectures.setdefault(width, {})
            for kind, rank, solution, metadata in candidates:
                architecture_id = _architecture_id(solution, width)
                rank_suffix = "" if rank is None else f"-{rank:04d}"
                solution_id = f"w{width:04d}-{depth_slug}-{kind}{rank_suffix}"
                alias = {
                    "profile_id": str(profile_id),
                    "solution_id": solution_id,
                    "architecture_id": architecture_id,
                    "kind": kind,
                    "hidden_width": width,
                    "removed_sublayers": int(scenario.get("removed_sublayers", 0)),
                    "depth_selection": scenario.get("depth_selection")
                    or {"total": int(scenario.get("removed_sublayers", 0))},
                    "score": metadata.get("score"),
                    "parameter_count": metadata.get("parameter_count"),
                    "total_costs": metadata.get("total_costs"),
                    "homogeneous_rank": rank,
                    "homogeneous_assignment": metadata.get("homogeneous_assignment"),
                }
                logical_solution_count += 1
                if architecture_id not in by_id:
                    payload = copy.deepcopy(solution)
                    payload["hidden_width"] = width
                    payload["online_evaluation"] = {"architecture_id": architecture_id}
                    by_id[architecture_id] = {
                        "architecture_id": architecture_id,
                        "hidden_width": width,
                        "puzzle_solution": payload,
                        "aliases": [],
                    }
                by_id[architecture_id]["aliases"].append(alias)

    architectures_by_width = {
        width: sorted(rows.values(), key=lambda row: row["architecture_id"])
        for width, rows in sorted(architectures.items())
    }
    return {
        "version": 1,
        "profile_ids": list(profile_ids),
        "logical_solution_count": logical_solution_count,
        "unique_architecture_count": sum(len(rows) for rows in architectures_by_width.values()),
        "architectures_by_width": architectures_by_width,
    }


def write_online_evaluation_plan(puzzle_dir: Path, plan: dict[str, Any]) -> Path:
    root = puzzle_dir / "artifacts" / "zero_shot_evaluation" / "online_plan"
    compact = {key: value for key, value in plan.items() if key != "architectures_by_width"}
    compact["execution"] = {
        "mode": "resident_sorted_teacher_online",
        "materialized_solution_checkpoints": False,
        "model_loads_per_worker": 1,
    }
    compact["widths"] = {}
    for raw_width, architectures in plan["architectures_by_width"].items():
        width = int(raw_width)
        width_root = root / f"width-{width:04d}"
        solutions = [row["puzzle_solution"] for row in architectures]
        index = [
            {key: value for key, value in row.items() if key != "puzzle_solution"}
            | {"solution_index": index}
            for index, row in enumerate(architectures)
        ]
        _atomic_json(width_root / "solutions.json", solutions)
        _atomic_json(width_root / "index.json", index)
        compact["widths"][str(width)] = {
            "unique_architecture_count": len(architectures),
            "solutions_path": str(width_root / "solutions.json"),
            "index_path": str(width_root / "index.json"),
            "execution": online_execution_contract(puzzle_dir, width),
        }
    output = root / "index.json"
    _atomic_json(output, compact)
    return output


def shard_solution_indices(total: int, shard_index: int, shard_count: int) -> list[int]:
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError(f"invalid shard index/count: index={shard_index} count={shard_count}")
    return list(range(int(shard_index), int(total), int(shard_count)))


def run_online_evaluation_shard(
    puzzle_dir: Path,
    *,
    config_path: Path,
    width: int,
    shard_index: int,
    shard_count: int,
    eval_samples: int,
    block_size: int,
    micro_batch_size: int,
    solution_indices: tuple[int, ...] = (),
) -> Path:
    """Load one width-specific sorted parent and score one architecture shard."""

    from omegaconf import OmegaConf

    from modelopt.torch.puzzletron.pipeline_config import (
        load_runtime_hydra_config,
        pipeline_config_from_path,
    )
    from modelopt.torch.puzzletron.plugins.automodel.solution_launch import (
        launch_score_solutions_automodel,
    )
    from modelopt.torch.puzzletron.stages.pipeline import _distributed

    plan_root = puzzle_dir / "artifacts" / "zero_shot_evaluation" / "online_plan"
    width_root = plan_root / f"width-{int(width):04d}"
    architectures = json.loads((width_root / "index.json").read_text())
    indices = list(solution_indices) or shard_solution_indices(
        len(architectures),
        shard_index,
        shard_count,
    )
    if any(not 0 <= index < len(architectures) for index in indices):
        raise ValueError(f"solution index is outside [0, {len(architectures)}): {indices}")
    output_dir = plan_root / "raw" / f"width-{int(width):04d}" / f"shard-{int(shard_index):02d}"
    if not indices:
        _atomic_json(
            output_dir / "worker.json",
            {"status": "empty", "shard_index": shard_index, "shard_count": shard_count},
        )
        return output_dir / "worker.json"

    config = dict(pipeline_config_from_path(config_path))
    hydra_cfg = load_runtime_hydra_config(config)
    OmegaConf.set_struct(hydra_cfg, False)
    execution = online_execution_contract(puzzle_dir, width)
    roles = execution["checkpoint_roles"]
    scenario = puzzle_dir / "scenarios" / f"width-{int(width):04d}" / "depth-00"
    parent = Path(roles["source"])
    bypass = Path(roles["bypass_overlay"]) if roles["bypass_overlay"] is not None else None
    if roles["source"] != roles["target"]:
        raise RuntimeError(f"online source and target must be the same sorted teacher: {roles}")
    checkpoints = [("parent", parent)]
    if bypass is not None:
        checkpoints.append(("bypass", bypass))
    for role, checkpoint in checkpoints:
        if not (checkpoint / "config.json").is_file():
            raise FileNotFoundError(f"online evaluation {role} is missing: {checkpoint}")
    hydra_cfg.puzzle_dir = str(scenario)
    hydra_cfg.scoring.teacher_dir = str(parent)
    hydra_cfg.scoring.target_teacher_dir = str(parent)
    hydra_cfg.scoring.source_checkpoint_dir = str(parent)
    hydra_cfg.scoring.bypass_checkpoint_dir = str(bypass) if bypass is not None else None
    hydra_cfg.scoring.solutions_path = str(width_root / "solutions.json")
    hydra_cfg.scoring.output_dir = str(output_dir)
    hydra_cfg.scoring.solutions_to_validate = indices
    hydra_cfg.scoring.skip_existing_solutions = True
    hydra_cfg.scoring.score_source_baseline = True
    hydra_cfg.scoring.eval_samples = int(eval_samples)
    hydra_cfg.scoring.micro_batch_size = int(micro_batch_size)
    hydra_cfg.scoring.block_size = int(block_size)
    hydra_cfg.scoring.packed_token_cache_path = str(
        puzzle_dir / "dataset_cache" / f"validation_{int(eval_samples)}x{int(block_size)}.tokens"
    )

    with _distributed(hydra_cfg):
        launch_score_solutions_automodel(hydra_cfg)

    if int(__import__("os").environ.get("RANK", "0")) == 0:
        missing = [
            index for index in indices if not (output_dir / f"solution_{index}.json").is_file()
        ]
        if missing:
            raise RuntimeError(f"online evaluation shard is missing results: {missing[:20]}")
        manifest = {
            "status": "success",
            "width": int(width),
            "shard_index": int(shard_index),
            "shard_count": int(shard_count),
            "eval_samples": int(eval_samples),
            "block_size": int(block_size),
            "micro_batch_size": int(micro_batch_size),
            "solution_indices": indices,
            "result_count": len(indices),
            "execution": execution,
        }
        _atomic_json(output_dir / "worker.json", manifest)
    return output_dir / "worker.json"


def _finite_metrics(payload: dict[str, Any]) -> dict[str, float]:
    excluded = {
        "hidden_width",
        "i_solution",
        "args",
        "puzzle_solution",
        "sliced_teacher_baseline",
        "observability",
        "distributed_evaluation",
    }
    metrics = {}
    for key, value in payload.items():
        if key in excluded:
            continue
        average = value.get("avg") if isinstance(value, dict) else value
        if (
            isinstance(average, (int, float))
            and not isinstance(average, bool)
            and math.isfinite(float(average))
        ):
            metrics[str(key)] = float(average)
    if "lm_loss" not in metrics:
        raise RuntimeError("online evaluation result has no finite lm_loss")
    return metrics


def merge_online_evaluation(
    puzzle_dir: Path,
    *,
    eval_samples: int,
    block_size: int,
) -> Path:
    """Fan each deduplicated architecture result back to every logical profile row."""

    root = puzzle_dir / "artifacts" / "zero_shot_evaluation"
    plan_root = root / "online_plan"
    plan = json.loads((plan_root / "index.json").read_text())
    by_profile: dict[str, list[dict[str, Any]]] = {
        str(profile_id): [] for profile_id in plan["profile_ids"]
    }
    teacher_candidates: list[tuple[int, Path, dict[str, float]]] = []
    unique_results = []
    for raw_width, width_plan in plan["widths"].items():
        width = int(raw_width)
        architectures = json.loads(Path(width_plan["index_path"]).read_text())
        raw_root = plan_root / "raw" / f"width-{width:04d}"
        baselines = sorted(raw_root.glob("shard-*/sliced_teacher.json"))
        if not baselines:
            raise FileNotFoundError(f"missing width-{width} source baseline under {raw_root}")
        baseline_payload = json.loads(baselines[0].read_text())
        teacher_candidates.append((width, baselines[0], _finite_metrics(baseline_payload)))
        for architecture in architectures:
            index = int(architecture["solution_index"])
            paths = sorted(raw_root.glob(f"shard-*/solution_{index}.json"))
            if not paths:
                raise FileNotFoundError(
                    f"missing online result width={width} solution_index={index}"
                )
            payload = json.loads(paths[0].read_text())
            metrics = _finite_metrics(payload)
            for duplicate in paths[1:]:
                duplicate_metrics = _finite_metrics(json.loads(duplicate.read_text()))
                if duplicate_metrics != metrics:
                    raise RuntimeError(
                        f"conflicting duplicate online results for width={width} index={index}"
                    )
            unique_results.append(
                {
                    "architecture_id": architecture["architecture_id"],
                    "hidden_width": width,
                    "solution_index": index,
                    "metrics": metrics,
                    "result_path": str(paths[0]),
                }
            )
            for alias in architecture["aliases"]:
                row = copy.deepcopy(alias)
                row.update(
                    {
                        "metrics": metrics,
                        "result_path": str(paths[0]),
                        "architecture_solution_index": index,
                        "architecture_solutions_path": width_plan["solutions_path"],
                    }
                )
                by_profile[str(alias["profile_id"])].append(row)

    teacher_width, teacher_path, teacher_metrics = max(
        teacher_candidates,
        key=lambda item: item[0],
    )
    teacher = {
        "solution_id": "teacher",
        "label": "Teacher",
        "hidden_width": teacher_width,
        "checkpoint": str(puzzle_dir / "ckpts" / "teacher"),
        "metrics": teacher_metrics,
        "result_path": str(teacher_path),
    }
    profile_outputs = {}
    workload_id = f"text-s{int(eval_samples)}-l{int(block_size)}"
    for profile_id, rows in by_profile.items():
        rows.sort(key=lambda row: (float(row["metrics"]["lm_loss"]), row["solution_id"]))
        for rank, row in enumerate(rows, start=1):
            row["rank"] = rank
        mip_grid = json.loads(
            (puzzle_dir / "mip" / "profiles" / profile_id / "mip_grid.json").read_text()
        )
        grid_teacher = mip_grid.get("teacher") or {}
        profile_teacher = {
            **teacher,
            "parameter_count": grid_teacher.get("parameter_count")
            or (grid_teacher.get("total_costs") or {}).get("stats.num_params"),
            "parameter_ratio": float(grid_teacher.get("parameter_ratio", 1.0)),
            "score": grid_teacher.get("score", 0.0),
            "total_costs": grid_teacher.get("total_costs") or {},
            "source_identity": {"profile_id": profile_id, "kind": "teacher"},
        }
        summary = {
            "version": 1,
            "profile_id": profile_id,
            "mode": "online_solutions",
            "eval_samples": int(eval_samples),
            "block_size": int(block_size),
            "teacher": profile_teacher,
            "solutions": rows,
        }
        output = root / "profiles" / profile_id / workload_id / "evaluation_summary.json"
        _atomic_json(output, summary)
        profile_outputs[profile_id] = {
            "summary_path": str(output),
            "solution_count": len(rows),
            "best_solution_id": rows[0]["solution_id"] if rows else None,
            "best_lm_loss": rows[0]["metrics"]["lm_loss"] if rows else None,
        }

    output = root / "online_evaluation_summary.json"
    _atomic_json(
        output,
        {
            "version": 1,
            "mode": "online_solutions",
            "eval_samples": int(eval_samples),
            "block_size": int(block_size),
            "logical_solution_count": int(plan["logical_solution_count"]),
            "unique_architecture_count": int(plan["unique_architecture_count"]),
            "unique_results": unique_results,
            "teacher": teacher,
            "profiles": profile_outputs,
        },
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--puzzle-dir", type=Path, required=True)
    parser.add_argument("--profile-id", action="append", default=[])
    parser.add_argument("--config", type=Path)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--run-shard", action="store_true")
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--width", type=int)
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--solution-index", type=int, action="append", default=[])
    parser.add_argument("--eval-samples", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=8192)
    parser.add_argument("--micro-batch-size", type=int, default=4)
    args = parser.parse_args()
    shard_index = (
        int(os.environ.get("PUZZLETRON_GROUP_INDEX", os.environ.get("SLURM_PROCID", "0")))
        if args.shard_index is None or args.shard_index < 0
        else args.shard_index
    )
    selected_modes = sum((args.prepare, args.run_shard, args.merge))
    if selected_modes > 1:
        parser.error("choose only one of --prepare, --run-shard, or --merge")
    if args.run_shard:
        if args.config is None or args.width is None:
            parser.error("--run-shard requires --config and --width")
        output = run_online_evaluation_shard(
            args.puzzle_dir,
            config_path=args.config,
            width=args.width,
            shard_index=shard_index,
            shard_count=args.shard_count,
            solution_indices=tuple(args.solution_index),
            eval_samples=args.eval_samples,
            block_size=args.block_size,
            micro_batch_size=args.micro_batch_size,
        )
    elif args.merge:
        output = merge_online_evaluation(
            args.puzzle_dir,
            eval_samples=args.eval_samples,
            block_size=args.block_size,
        )
    else:
        if not args.profile_id:
            parser.error("--prepare requires at least one --profile-id")
        plan = build_online_evaluation_plan(args.puzzle_dir, tuple(args.profile_id))
        output = write_online_evaluation_plan(args.puzzle_dir, plan)
    print(output)


if __name__ == "__main__":
    main()
