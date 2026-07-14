# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exact-LM finalist selection and sequential native global KD."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

import modelopt.torch.utils.distributed as dist

from ..plugins.automodel.validation import validate_realized_checkpoints_automodel
from .global_automodel import build_global_kd_config, run_global_kd

__all__ = ["run_global_kd_tournament"]


def _metric(path: Path, key: str) -> float:
    raw = json.loads(path.read_text())
    value = raw[key]
    if isinstance(value, dict):
        value = value.get("avg")
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError(f"non-finite tournament metric {key}={result} in {path}")
    return result


def _consolidated_checkpoint(output_dir: Path) -> Path:
    candidates = list(output_dir.glob("checkpoints/**/model/consolidated/config.json"))
    candidates.extend(output_dir.glob("checkpoints/**/consolidated/config.json"))
    candidates.extend(output_dir.glob("checkpoints/**/config.json"))
    if not candidates:
        raise FileNotFoundError(f"global KD wrote no consolidated HF checkpoint under {output_dir}")
    path = max(candidates, key=lambda item: item.stat().st_mtime_ns).parent
    if not (path / "config.json").is_file():
        raise RuntimeError(f"invalid consolidated checkpoint selection: {path}")
    return path


def run_global_kd_tournament(
    config: dict[str, Any],
    hydra_cfg: DictConfig,
    *,
    recipe_runner=None,
) -> dict[str, Any]:
    tournament = dict((config.get("distillation") or {}).get("tournament") or {})
    puzzle_dir = Path((config.get("experiment") or {})["dir"])
    evaluation_dir = Path(
        tournament.get(
            "pre_kd_evaluation_dir",
            puzzle_dir / "mip" / "puzzle_solutions" / "depth_tournament" / "exact_lm_evaluation",
        )
    )
    checkpoints_dir = Path(
        tournament.get(
            "solution_checkpoints_dir",
            puzzle_dir / "mip" / "puzzle_solutions" / "depth_tournament" / "solutions--checkpoints",
        )
    )
    metric = str(tournament.get("metric", "lm_loss"))
    top_k = int(tournament.get("top_k", 3))
    rows = []
    for result_path in sorted(evaluation_dir.glob("solution_*.json")):
        solution_id = int(result_path.stem.removeprefix("solution_"))
        checkpoint = checkpoints_dir / f"solution_{solution_id}"
        if not (checkpoint / "config.json").is_file():
            raise FileNotFoundError(f"missing realized finalist checkpoint: {checkpoint}")
        rows.append(
            {
                "solution_id": solution_id,
                "pre_kd_lm_loss": _metric(result_path, metric),
                "checkpoint_dir": str(checkpoint),
                "result_path": str(result_path),
            }
        )
    expected = int(tournament.get("expected_solution_count", 11))
    if len(rows) != expected:
        raise RuntimeError(f"expected {expected} pre-KD solutions, found {len(rows)}")
    rows.sort(key=lambda row: (row["pre_kd_lm_loss"], row["solution_id"]))
    requested_ids = tournament.get("solution_ids")
    if requested_ids is None:
        finalists = rows[:top_k]
    else:
        requested_ids = [int(solution_id) for solution_id in requested_ids]
        if not requested_ids or len(requested_ids) != len(set(requested_ids)):
            raise ValueError(
                "distillation.tournament.solution_ids must be a non-empty list of unique IDs"
            )
        by_id = {row["solution_id"]: row for row in rows}
        missing = [solution_id for solution_id in requested_ids if solution_id not in by_id]
        if missing:
            raise ValueError(f"unknown tournament solution IDs: {missing}")
        finalists = [by_id[solution_id] for solution_id in requested_ids]

    for rank, finalist in enumerate(finalists):
        candidate_config = copy.deepcopy(config)
        kd = dict(candidate_config.get("distillation") or {})
        output_dir = puzzle_dir / "global_kd_tournament" / f"rank_{rank}_solution_{finalist['solution_id']}"
        kd["student_dir"] = finalist["checkpoint_dir"]
        kd["output_dir"] = str(output_dir)
        kd.pop("tournament", None)
        candidate_config["distillation"] = kd
        result = run_global_kd(
            build_global_kd_config(candidate_config),
            recipe_runner=recipe_runner,
        )
        finalist["kd_result"] = result.to_dict()
        finalist["post_kd_checkpoint_dir"] = str(_consolidated_checkpoint(output_dir))

    validation_args = OmegaConf.create(
        OmegaConf.to_container(hydra_cfg.realize_model, resolve=True)
    )
    validation_args.teacher_dir = str(
        (config.get("convert") or {}).get("teacher_dir")
    )
    validation_args.output_dir = str(puzzle_dir / "global_kd_tournament" / "post_kd_evaluation")
    validate_realized_checkpoints_automodel(
        hydra_cfg,
        validation_args,
        [
            (
                int(finalist["solution_id"]),
                Path(finalist["post_kd_checkpoint_dir"]),
                {"tournament_rank": rank, "pre_kd": finalist},
            )
            for rank, finalist in enumerate(finalists)
        ],
        validation_args.output_dir,
    )
    dist.barrier()
    if dist.is_master():
        post_dir = Path(validation_args.output_dir)
        for finalist in finalists:
            result_path = post_dir / f"solution_{finalist['solution_id']}.json"
            finalist["post_kd_result_path"] = str(result_path)
            finalist["post_kd_lm_loss"] = _metric(result_path, metric)
            finalist["lm_loss_delta"] = (
                finalist["post_kd_lm_loss"] - finalist["pre_kd_lm_loss"]
            )
        summary_path = puzzle_dir / "global_kd_tournament" / "summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps({"metric": metric, "all_solutions": rows, "finalists": finalists}, indent=2, sort_keys=True, default=str) + "\n"
        )
    else:
        summary_path = puzzle_dir / "global_kd_tournament" / "summary.json"
    dist.barrier()
    return {"summary_path": str(summary_path), "finalists": finalists}
