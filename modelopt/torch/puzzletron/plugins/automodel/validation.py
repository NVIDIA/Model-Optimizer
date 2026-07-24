# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AutoModel-native validation for realized Puzzletron checkpoints."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from omegaconf import DictConfig

import modelopt.torch.utils.distributed as dist

from ...tools.hydra_utils import clone_hydra_config
from ...tools.validation_utils import write_results
from .config import build_solution_recipe_config, solution_scoring_params
from .launch import _free_scoring_memory
from .patch import apply_patch
from .solution_launch import _extract_teacher_targets, _run_recipe, _score_candidate
from .teacher_cache import TeacherTargetCache

__all__ = ["validate_realized_checkpoints_automodel"]


def _validation_config(hydra_cfg: DictConfig, args: DictConfig) -> DictConfig:
    cfg = clone_hydra_config(hydra_cfg)
    scoring = clone_hydra_config(args)
    if "automodel" not in scoring:
        scoring.automodel = clone_hydra_config(
            hydra_cfg.scoring.automodel
            if hydra_cfg.get("scoring", None) is not None
            else {}
        )
    scoring.backend = "automodel"
    scoring.teacher_dir = str(args.teacher_dir)
    scoring.source_checkpoint_dir = str(args.teacher_dir)
    scoring.target_teacher_dir = str(args.teacher_dir)
    scoring.use_puzzletron_dataloader = bool(
        scoring.automodel.get("use_puzzletron_dataloader", True)
    )
    cfg.scoring = scoring
    return cfg


def validate_realized_checkpoints_automodel(
    hydra_cfg: DictConfig,
    args: DictConfig,
    checkpoints: Iterable[tuple[int, Path, dict[str, Any]]],
    output_dir: str | Path,
) -> None:
    """Compare realized checkpoints with one cached teacher, loading each once."""
    apply_patch()
    cfg = _validation_config(hydra_cfg, args)
    scoring = cfg.scoring
    params = solution_scoring_params(cfg)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    teacher_dir = Path(args.teacher_dir)
    if not (teacher_dir / "config.json").is_file():
        raise FileNotFoundError(f"validation teacher is not an HF checkpoint: {teacher_dir}")

    cache = TeacherTargetCache(device=params["teacher_cache_device"])
    teacher_recipe = _run_recipe(
        build_solution_recipe_config(cfg, teacher_dir),
        scoring,
        params["eval_iters"],
        params["use_puzzletron_dataloader"],
    )
    try:
        teacher_scores = _extract_teacher_targets(teacher_recipe, cache, params)
        if teacher_scores is not None and bool(
            getattr(teacher_recipe, "_puzzletron_output_writer", False)
        ):
            write_results(output_dir, "teacher", scoring, teacher_scores)
        dist.barrier()
    finally:
        teacher_recipe.teardown_capture()
        _free_scoring_memory(teacher_recipe)

    for solution_id, checkpoint_dir, solution in checkpoints:
        checkpoint_dir = Path(checkpoint_dir)
        if not (checkpoint_dir / "config.json").is_file():
            raise FileNotFoundError(
                f"realized solution {solution_id} has no config.json: {checkpoint_dir}"
            )
        recipe = _run_recipe(
            build_solution_recipe_config(cfg, checkpoint_dir),
            scoring,
            params["eval_iters"],
            params["use_puzzletron_dataloader"],
        )
        try:
            _score_candidate(
                recipe,
                cache,
                params,
                output_dir,
                scoring,
                name=f"solution_{solution_id}",
                payload={"i_solution": solution_id, "puzzle_solution": solution},
                prune_target=None,
            )
            dist.barrier()
        finally:
            recipe.teardown_capture()
            _free_scoring_memory(recipe)
