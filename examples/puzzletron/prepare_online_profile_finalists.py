#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Realize the best online-evaluated profile architectures for AIPerf and KD."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from safetensors import safe_open

from modelopt.torch.puzzletron.anymodel.registry import resolve_descriptor_from_pretrained
from modelopt.torch.puzzletron.pipeline_config import pipeline_config_from_path
from modelopt.torch.puzzletron.replacement_library.library import ReplacementLibrary
from modelopt.torch.puzzletron.replacement_library.replacement_utils import parse_layer_replacement
from modelopt.torch.puzzletron.solution_registry import write_solution_registry

_COLORS = ("#4f8cff", "#22d3ee", "#ff6577")


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _parameter_count(checkpoint: Path) -> int:
    index_path = checkpoint / "model.safetensors.index.json"
    if index_path.is_file():
        files = sorted(set(json.loads(index_path.read_text())["weight_map"].values()))
    elif (checkpoint / "model.safetensors").is_file():
        files = ["model.safetensors"]
    else:
        raise FileNotFoundError(f"checkpoint has no safetensors: {checkpoint}")
    total = 0
    for relative in files:
        with safe_open(str(checkpoint / relative), framework="pt") as handle:
            for key in handle.keys():
                total += math.prod(int(value) for value in handle.get_slice(key).get_shape())
    return int(total)


def select_online_finalists(summary: dict[str, Any], count: int = 3) -> list[dict[str, Any]]:
    """Return the lowest-finite-LM-loss candidates; the teacher is not a candidate."""

    candidates = []
    for row in summary.get("solutions", ()):
        lm_loss = (row.get("metrics") or {}).get("lm_loss")
        if isinstance(lm_loss, (int, float)) and math.isfinite(float(lm_loss)):
            candidates.append(dict(row))
    candidates.sort(key=lambda row: (float(row["metrics"]["lm_loss"]), row["solution_id"]))
    if len(candidates) < int(count):
        raise RuntimeError(f"expected at least {count} finite online candidates, found {len(candidates)}")
    return candidates[: int(count)]


def build_finalist_registry(
    *,
    profile_id: str,
    summary: dict[str, Any],
    mip_grid: dict[str, Any],
    finalists: list[dict[str, Any]],
    checkpoints: dict[str, Path],
) -> dict[str, Any]:
    """Build the selected-model registry shared by AIPerf and global KD."""

    teacher = dict(summary["teacher"])
    teacher_grid = dict(mip_grid["teacher"])
    teacher_parameter_count = int(
        teacher_grid.get("parameter_count")
        or (teacher_grid.get("total_costs") or {})["stats.num_params"]
    )
    records = [
        {
            "solution_id": "teacher",
            "label": "Teacher",
            "hidden_width": int(teacher["hidden_width"]),
            "removed_sublayers": 0,
            "checkpoint": str(teacher["checkpoint"]),
            "color": "#f5c451",
            "marker": "star",
            "always_enabled": True,
            "parameter_ratio": 1.0,
            "parameter_count": teacher_parameter_count,
            "metrics": teacher.get("metrics") or {},
            "source_identity": {"profile_id": profile_id, "kind": "teacher"},
        }
    ]
    for rank, (color, row) in enumerate(zip(_COLORS, finalists), start=1):
        solution_id = str(row["solution_id"])
        checkpoint = checkpoints[solution_id]
        records.append(
            {
                "solution_id": solution_id,
                "label": f"Rank {rank}: {solution_id}",
                "rank": rank,
                "hidden_width": int(row["hidden_width"]),
                "removed_sublayers": int(row.get("removed_sublayers", 0)),
                "depth_selection": row.get("depth_selection") or {},
                "kind": row.get("kind", "mixed"),
                "checkpoint": str(checkpoint),
                "color": color,
                "marker": "circle",
                "always_enabled": False,
                "parameter_count": int(row["parameter_count"]),
                "parameter_ratio": float(row["parameter_count"])
                / float(teacher_parameter_count),
                "score": row.get("score"),
                "total_costs": row.get("total_costs") or {},
                "metrics": row["metrics"],
                "source_identity": {
                    "profile_id": profile_id,
                    "architecture_id": row["architecture_id"],
                    "architecture_solution_index": row["architecture_solution_index"],
                    "architecture_solutions_path": row["architecture_solutions_path"],
                },
            }
        )
    return {
        "version": 1,
        "profile_id": profile_id,
        "profile": mip_grid.get("profile") or {"id": profile_id},
        "selection_metric": "lm_loss",
        "absolute_best_solution_id": str(finalists[0]["solution_id"]),
        "solutions": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--puzzle-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--profile-id", default="runtime-075")
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    profile_root = args.puzzle_dir / "mip" / "profiles" / args.profile_id
    summaries = sorted(
        (args.puzzle_dir / "artifacts" / "zero_shot_evaluation" / "profiles" / args.profile_id).glob(
            "*/evaluation_summary.json"
        )
    )
    if len(summaries) != 1:
        raise RuntimeError(f"expected one online evaluation summary, found {summaries}")
    summary = json.loads(summaries[0].read_text())
    grid = json.loads((profile_root / "mip_grid.json").read_text())
    finalists = select_online_finalists(summary, args.count)

    config = dict(pipeline_config_from_path(args.config))
    model_cfg = dict(config.get("model") or {})
    teacher = args.puzzle_dir / "ckpts" / "teacher"
    descriptor = resolve_descriptor_from_pretrained(
        str(teacher),
        trust_remote_code=bool(model_cfg.get("trust_remote_code", False)),
        descriptor_override=model_cfg.get("descriptor_override"),
    ).descriptor
    libraries: dict[int, ReplacementLibrary] = {}
    checkpoints = {}
    for row in finalists:
        solution_id = str(row["solution_id"])
        width = int(row["hidden_width"])
        checkpoint = (
            args.puzzle_dir
            / "artifacts"
            / "realized_finalists"
            / "profiles"
            / args.profile_id
            / solution_id
            / "checkpoint"
        )
        expected = int(row["parameter_count"])
        if (checkpoint / "config.json").is_file() and not args.overwrite:
            actual = _parameter_count(checkpoint)
        else:
            solutions = json.loads(Path(row["architecture_solutions_path"]).read_text())
            solution = solutions[int(row["architecture_solution_index"])]
            replacements = [
                parse_layer_replacement(item["layer_replacement"])
                for item in solution["chosen_replacements"]
            ]
            library = libraries.get(width)
            if library is None:
                scenario = args.puzzle_dir / "scenarios" / f"width-{width:04d}" / "depth-00"
                library = ReplacementLibrary(scenario / "replacement_library.json", descriptor)
                libraries[width] = library
            model_config = library.create_model_config(replacements)
            library.materialize_checkpoint(
                replacements,
                checkpoint,
                model_config=model_config,
                overwrite=args.overwrite,
                solution_identity=f"{args.profile_id}-{solution_id}",
            )
            actual = _parameter_count(checkpoint)
        if actual != expected:
            raise RuntimeError(
                f"parameter formula/materialization mismatch for {solution_id}: {expected} != {actual}"
            )
        checkpoints[solution_id] = checkpoint

    registry = build_finalist_registry(
        profile_id=args.profile_id,
        summary=summary,
        mip_grid=grid,
        finalists=finalists,
        checkpoints=checkpoints,
    )
    output = write_solution_registry(profile_root / "selected_solutions.json", registry)
    by_id = {str(row["solution_id"]): row for row in summary["solutions"]}
    for solution_id, checkpoint in checkpoints.items():
        by_id[solution_id]["checkpoint"] = str(checkpoint)
    summary["absolute_best_solution_id"] = registry["absolute_best_solution_id"]
    summary["selected_solution_ids"] = [
        str(row["solution_id"]) for row in registry["solutions"] if row["solution_id"] != "teacher"
    ]
    _atomic_json(summaries[0], summary)
    print(output)


if __name__ == "__main__":
    main()
