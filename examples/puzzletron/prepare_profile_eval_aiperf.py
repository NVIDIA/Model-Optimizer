#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Realize a selected MIP profile subset and publish its comparison registry."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from typing import Any

from safetensors import safe_open

from modelopt.torch.puzzletron.anymodel.registry import resolve_descriptor_from_pretrained
from modelopt.torch.puzzletron.pipeline_config import pipeline_config_from_path
from modelopt.torch.puzzletron.replacement_library.library import ReplacementLibrary
from modelopt.torch.puzzletron.replacement_library.replacement_utils import (
    parse_layer_replacement,
)
from modelopt.torch.puzzletron.solution_registry import (
    build_profile_solution_registry,
    write_solution_registry,
)


DEFAULT_SELECTIONS = ((1024, 0), (1024, 1), (512, 0), (512, 1))


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


def _selection(value: str) -> tuple[int, int]:
    try:
        width, depth = value.split(":", maxsplit=1)
        parsed = int(width), int(depth)
    except (ValueError, TypeError) as error:
        raise argparse.ArgumentTypeError("candidate must use WIDTH:DEPTH") from error
    if parsed[0] <= 0 or parsed[1] < 0:
        raise argparse.ArgumentTypeError("candidate width must be positive and depth nonnegative")
    return parsed


def _resolved_config(puzzle_dir: Path, config_path: Path | None = None) -> dict[str, Any]:
    if config_path is not None:
        return dict(pipeline_config_from_path(config_path))
    manifest = puzzle_dir / "manifests" / "build_library.json"
    if not manifest.is_file():
        raise FileNotFoundError(f"missing build-library manifest: {manifest}")
    return dict(json.loads(manifest.read_text()).get("config") or {})


def _scenario(grid: dict[str, Any], width: int, depth: int) -> dict[str, Any]:
    rows = [
        row
        for row in grid.get("scenarios", ())
        if row.get("status") == "feasible"
        and int(row.get("hidden_width", -1)) == width
        and int(row.get("removed_sublayers", -1)) == depth
    ]
    if len(rows) != 1:
        raise RuntimeError(
            f"expected one feasible scenario for width={width}, depth={depth}; found {len(rows)}"
        )
    return rows[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--puzzle-dir", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        help="Resolved campaign config source; required when build_library.json is unavailable.",
    )
    parser.add_argument("--profile-id", default="params-080")
    parser.add_argument(
        "--candidate",
        action="append",
        type=_selection,
        help="Repeat WIDTH:DEPTH; defaults to the four approved 80%% comparisons.",
    )
    parser.add_argument(
        "--all-feasible",
        action="store_true",
        help="Select every feasible width/depth scenario in the profile grid.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    puzzle_dir = args.puzzle_dir
    profile_root = puzzle_dir / "mip" / "profiles" / args.profile_id
    grid_path = profile_root / "mip_grid.json"
    grid = json.loads(grid_path.read_text())
    if args.all_feasible and args.candidate:
        parser.error("--all-feasible cannot be combined with --candidate")
    selections = (
        tuple(
            sorted(
                {
                    (int(row["hidden_width"]), int(row["removed_sublayers"]))
                    for row in grid.get("scenarios", ())
                    if row.get("status") == "feasible"
                }
            )
        )
        if args.all_feasible
        else tuple(args.candidate or DEFAULT_SELECTIONS)
    )
    if not selections:
        raise RuntimeError(f"profile contains no feasible scenarios: {grid_path}")
    if len(selections) != len(set(selections)):
        raise ValueError(f"duplicate candidate selections: {selections}")

    config = _resolved_config(puzzle_dir, args.config)
    model_cfg = dict(config.get("model") or {})
    teacher = Path(grid["teacher"]["checkpoint"])
    descriptor = resolve_descriptor_from_pretrained(
        str(teacher),
        trust_remote_code=bool(model_cfg.get("trust_remote_code", False)),
        descriptor_override=model_cfg.get("descriptor_override"),
    ).descriptor
    libraries: dict[int, ReplacementLibrary] = {}

    for width, depth in selections:
        row = _scenario(grid, width, depth)
        scenario_root = profile_root / "scenarios" / f"width-{width:04d}" / f"depth-{depth:02d}"
        published_checkpoint = Path(str(row.get("checkpoint") or ""))
        checkpoint = (
            published_checkpoint
            if (published_checkpoint / "config.json").is_file()
            else scenario_root / "realized" / "solution_0"
        )
        expected = int(row["parameter_count"])
        if (checkpoint / "config.json").is_file() and not args.overwrite:
            actual = _parameter_count(checkpoint)
            if actual != expected:
                raise RuntimeError(
                    f"stale realized checkpoint {checkpoint}: expected {expected}, found {actual}"
                )
        else:
            library = libraries.get(width)
            if library is None:
                base = puzzle_dir / "scenarios" / f"width-{width:04d}" / "depth-00"
                library = ReplacementLibrary(base / "replacement_library.json", descriptor)
                libraries[width] = library
            raw = json.loads(Path(row["solution_path"]).read_text())
            if len(raw) != 1:
                raise RuntimeError(f"expected one architecture in {row['solution_path']}")
            replacements = [
                parse_layer_replacement(item["layer_replacement"])
                for item in raw[0]["chosen_replacements"]
            ]
            model_config = library.create_model_config(replacements)
            library.materialize_checkpoint(
                replacements,
                checkpoint,
                model_config=model_config,
                overwrite=args.overwrite,
                solution_identity=f"{args.profile_id}-width-{width:04d}-depth-{depth:02d}",
            )
            actual = _parameter_count(checkpoint)
            if actual != expected:
                raise RuntimeError(
                    "parameter formula/materialization mismatch for "
                    f"width={width}, depth={depth}: {expected} != {actual}"
                )
        row["checkpoint"] = str(checkpoint)
        manifest_path = scenario_root / "scenario_manifest.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else dict(row)
        manifest["checkpoint"] = str(checkpoint)
        manifest["parameter_count_verified"] = expected
        _atomic_json(manifest_path, manifest)
        _atomic_json(grid_path, grid)

    registry_grid = copy.deepcopy(grid)
    original_teacher = puzzle_dir / "ckpts" / "teacher"
    if not (original_teacher / "config.json").is_file():
        raise FileNotFoundError(f"missing original teacher checkpoint: {original_teacher}")
    registry_grid["teacher"]["checkpoint"] = str(original_teacher)
    registry = build_profile_solution_registry(registry_grid, selections=selections)
    output = write_solution_registry(profile_root / "selected_solutions.json", registry)
    print(json.dumps({"registry": str(output), "solutions": registry["solutions"]}, indent=2))


if __name__ == "__main__":
    main()
