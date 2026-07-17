#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evaluate one registry solution on one GPU, or merge completed worker rows."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from modelopt.torch.puzzletron.anymodel.registry import resolve_descriptor_from_pretrained
from modelopt.torch.puzzletron.identity import canonicalize, stable_hash
from modelopt.torch.puzzletron.pipeline_config import (
    load_runtime_hydra_config,
    pipeline_config_from_path,
)


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _single_gpu_parallelism() -> dict[str, int | bool]:
    """Return the stage-local AutoModel mesh for one-GPU profile evaluation."""

    return {
        "tp": 1,
        "cp": 1,
        "pp": 1,
        "ep": 1,
        "dp_shard": 1,
        "dp_replicate": 1,
        "sequence_parallel": False,
    }


def _evaluation_root(
    puzzle_dir: Path, *, profile_id: str, eval_samples: int, block_size: int
) -> Path:
    """Return the canonical artifact root for one zero-shot evaluation workload."""

    return (
        puzzle_dir
        / "artifacts"
        / "zero_shot_evaluation"
        / "profiles"
        / profile_id
        / f"text-s{eval_samples}-l{block_size}"
    )


def _registry(puzzle_dir: Path, profile_id: str) -> tuple[Path, dict[str, Any]]:
    path = puzzle_dir / "mip" / "profiles" / profile_id / "selected_solutions.json"
    return path, json.loads(path.read_text())


def _config(puzzle_dir: Path, config_path: Path | None = None) -> dict[str, Any]:
    if config_path is not None:
        return dict(pipeline_config_from_path(config_path))
    manifest = json.loads((puzzle_dir / "manifests" / "build_library.json").read_text())
    return dict(manifest["config"])


def _solution(registry: dict[str, Any], solution_id: str) -> dict[str, Any]:
    matches = [row for row in registry["solutions"] if row["solution_id"] == solution_id]
    if len(matches) != 1:
        raise ValueError(f"expected one registry solution {solution_id!r}, found {len(matches)}")
    return matches[0]


def _identity_solution(checkpoint: Path, descriptor) -> dict[str, Any]:
    from modelopt.torch.puzzletron.block_config import maybe_cast_block_configs
    from modelopt.torch.puzzletron.tools.checkpoint_utils import load_model_config

    checkpoint_config = load_model_config(
        checkpoint, trust_remote_code=descriptor.requires_trust_remote_code()
    )
    lm = descriptor.get_language_model_config(checkpoint_config)
    blocks = list(maybe_cast_block_configs(checkpoint_config.block_configs))
    replacements = [
        {
            "weight_paths": [],
            "parent_layer_indices": [index],
            "child_block_configs": [block.to_dict()],
        }
        for index, block in enumerate(blocks)
    ]
    identity = json.loads(json.dumps(canonicalize(replacements[0])))
    identity["diagnostic"] = {"num_changed_layers": 0}
    return {
        "single_sequence_replacement": identity,
        "chosen_replacements": replacements,
        "block_configs": [block.to_dict() for block in blocks],
        "hidden_width": int(lm.hidden_size),
    }


def evaluate_one(
    puzzle_dir: Path,
    *,
    profile_id: str,
    solution_id: str,
    eval_samples: int,
    block_size: int,
    checkpoint_override: Path | None = None,
    output_dir_override: Path | None = None,
    config_path: Path | None = None,
) -> Path:
    from modelopt.torch.puzzletron.plugins.automodel.solution_launch import (
        launch_score_solutions_automodel,
    )
    from modelopt.torch.puzzletron.stages.pipeline import _distributed

    _, registry = _registry(puzzle_dir, profile_id)
    row = _solution(registry, solution_id)
    row = dict(row)
    checkpoint = Path(checkpoint_override or row["checkpoint"])
    if checkpoint_override is not None:
        row.update(
            {
                "checkpoint": str(checkpoint),
                "solution_id": f"{solution_id}-post-kd",
                "label": f"{row.get('label', solution_id)} post-KD",
            }
        )
    config = _config(puzzle_dir, config_path)
    model_cfg = dict(config.get("model") or {})
    resolution = resolve_descriptor_from_pretrained(
        str(checkpoint),
        trust_remote_code=bool(model_cfg.get("trust_remote_code", False)),
        descriptor_override=model_cfg.get("descriptor_override"),
    )
    descriptor = resolution.descriptor
    config.setdefault("_runtime", {})["descriptor"] = resolution.name
    hydra_cfg = load_runtime_hydra_config(config)
    OmegaConf.set_struct(hydra_cfg, False)
    hydra_cfg.descriptor = resolution.name

    output_dir = output_dir_override or (
        _evaluation_root(
            puzzle_dir,
            profile_id=profile_id,
            eval_samples=eval_samples,
            block_size=block_size,
        )
        / solution_id
    )
    hydra_cfg.scoring.automodel.parallel = _single_gpu_parallelism()
    hydra_cfg.scoring.automodel.force_hf = False
    hydra_cfg.scoring.teacher_dir = str(puzzle_dir / "ckpts" / "teacher")
    hydra_cfg.scoring.target_teacher_dir = str(puzzle_dir / "ckpts" / "teacher")
    hydra_cfg.scoring.source_checkpoint_dir = str(checkpoint)
    hydra_cfg.scoring.output_dir = str(output_dir / "raw")
    hydra_cfg.scoring.eval_samples = int(eval_samples)
    hydra_cfg.scoring.micro_batch_size = 1
    hydra_cfg.scoring.block_size = int(block_size)
    hydra_cfg.scoring.skip_existing_solutions = False
    solutions_path = output_dir / "identity_solution.json"
    solutions_path.parent.mkdir(parents=True, exist_ok=True)
    solutions_path.write_text(
        json.dumps([_identity_solution(checkpoint, descriptor)], indent=2, sort_keys=True) + "\n"
    )
    hydra_cfg.scoring.solutions_path = str(solutions_path)
    hydra_cfg.scoring.solutions_to_validate = None

    with _distributed(hydra_cfg):
        launch_score_solutions_automodel(hydra_cfg)

    raw_path = output_dir / "raw" / "sliced_teacher.json"
    raw = json.loads(raw_path.read_text())
    metrics = {
        key: float(value["avg"])
        for key, value in raw.items()
        if isinstance(value, dict)
        and isinstance(value.get("avg"), (int, float))
        and math.isfinite(float(value["avg"]))
    }
    if not metrics or "lm_loss" not in metrics:
        raise RuntimeError(f"evaluation produced no finite LM metrics: {raw_path}")
    data_fingerprint = stable_hash(
        {
            "data": config.get("data"),
            "scoring": {
                "eval_samples": eval_samples,
                "block_size": block_size,
                "seed": (config.get("scoring") or {}).get("seed", 42),
                "shuffle_seed": (config.get("scoring") or {}).get("shuffle_seed", 444),
            },
        },
        prefix="profile_evaluation_data",
    )
    result = {
        **row,
        "profile_id": profile_id,
        "metrics": metrics,
        "data_fingerprint": data_fingerprint,
        "eval_samples": int(eval_samples),
        "block_size": int(block_size),
        "result_path": str(raw_path),
        "observability": raw.get("observability") or {},
    }
    output = output_dir / "result.json"
    _atomic_json(output, result)
    return output


def merge_results(
    puzzle_dir: Path, *, profile_id: str, eval_samples: int, block_size: int
) -> Path:
    _, registry = _registry(puzzle_dir, profile_id)
    root = _evaluation_root(
        puzzle_dir,
        profile_id=profile_id,
        eval_samples=eval_samples,
        block_size=block_size,
    )
    rows = []
    for solution in registry["solutions"]:
        path = root / solution["solution_id"] / "result.json"
        rows.append(json.loads(path.read_text()))
    fingerprints = {row["data_fingerprint"] for row in rows}
    if len(fingerprints) != 1:
        raise RuntimeError(f"evaluation data fingerprints do not match: {fingerprints}")
    if len({row["solution_id"] for row in rows}) != len(rows):
        raise RuntimeError("evaluation results contain duplicate solution IDs")
    summary = {
        "version": 1,
        "profile_id": profile_id,
        "data_fingerprint": next(iter(fingerprints)),
        "eval_samples": int(eval_samples),
        "block_size": int(block_size),
        "solutions": rows,
    }
    output = root / "evaluation_summary.json"
    _atomic_json(output, summary)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--puzzle-dir", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        help="Resolved campaign config source; required when build_library.json is unavailable.",
    )
    parser.add_argument("--profile-id", default="params-080")
    parser.add_argument("--solution-id")
    parser.add_argument("--eval-samples", type=int, default=1024)
    parser.add_argument("--block-size", type=int, default=2048)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--merge", action="store_true")
    args = parser.parse_args()
    if args.merge:
        output = merge_results(
            args.puzzle_dir,
            profile_id=args.profile_id,
            eval_samples=args.eval_samples,
            block_size=args.block_size,
        )
    else:
        if not args.solution_id:
            parser.error("--solution-id is required unless --merge is used")
        output = evaluate_one(
            args.puzzle_dir,
            profile_id=args.profile_id,
            solution_id=args.solution_id,
            eval_samples=args.eval_samples,
            block_size=args.block_size,
            checkpoint_override=args.checkpoint,
            output_dir_override=args.output_dir,
            config_path=args.config,
        )
    print(output)


if __name__ == "__main__":
    main()
