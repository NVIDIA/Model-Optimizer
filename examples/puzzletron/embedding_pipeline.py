# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Embedding-width fan-out used by the one-command Puzzletron runner."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

__all__ = ["run_embedding_stage", "scenario_worker_commands"]


def _scenario_dir(puzzle_dir: Path, width: int) -> Path:
    return puzzle_dir / "scenarios" / f"width-{width:04d}" / "depth-00"


def _scenario_overrides(config: dict, scenario: Path) -> tuple[str, ...]:
    teacher = scenario / "ckpts" / "sorted_teacher"
    scoring_output = scenario / "single_sequence_replacement_solutions--validation"
    return (
        f"puzzle_dir={scenario}",
        f"experiment.dir={scenario}",
        f"teacher_dir={teacher}",
        f"convert.teacher_dir={teacher}",
        "bypass.enabled=false",
        f"replacement_library_path={scenario / 'replacement_library.json'}",
        f"build_replacement_library.source_checkpoint_dir={teacher}",
        "calc_subblock_stats.runtime_stats.execution=inline",
        f"scoring.teacher_dir={teacher}",
        f"scoring.source_checkpoint_dir={teacher}",
        f"scoring.target_teacher_dir={teacher}",
        f"scoring.solutions_path={scenario / 'single_sequence_replacement_solutions.json'}",
        f"scoring.output_dir={scoring_output}",
        f"vllm_stats_diagnostic.stats_path={scenario / 'subblock_stats.json'}",
        f"vllm_stats_diagnostic.output_dir={scenario / 'artifacts/vllm_stats_diagnostic'}",
        f"scoring_diagnostic.scores_dir={scoring_output}",
        f"scoring_diagnostic.output_dir={scenario / 'artifacts/scoring_diagnostic'}",
    )


def scenario_worker_commands(
    *,
    config_path: str | Path,
    config: dict,
    stage: str,
    gpus_per_node: int,
) -> tuple[tuple[str, ...], ...]:
    """Return one isolated worker command per configured embedding width."""

    puzzle_dir = Path(config["puzzle_dir"])
    widths = tuple(
        int(width) for width in (config.get("embedding_pruning") or {}).get("widths", ())
    )
    commands = []
    for width in widths:
        command = [sys.executable]
        if stage == "scoring":
            command.extend(
                (
                    "-m",
                    "torch.distributed.run",
                    "--standalone",
                    f"--nproc_per_node={gpus_per_node}",
                )
            )
        command.extend(
            (
                str(Path(__file__).resolve().parent / "main.py"),
                "--config",
                str(config_path),
                "--worker-stage",
                stage,
                "--scenario-child",
            )
        )
        for override in _scenario_overrides(config, _scenario_dir(puzzle_dir, width)):
            command.extend(("--override", override))
        commands.append(tuple(command))
    return tuple(commands)


def _run_commands(commands: tuple[tuple[str, ...], ...]) -> None:
    for command in commands:
        subprocess.run(command, check=True)


def run_embedding_stage(
    *,
    config_path: str | Path,
    config: dict,
    stage: str,
    gpus_per_node: int,
) -> dict:
    """Run the width-aware portion of a composite pipeline stage."""

    puzzle_dir = Path(config["puzzle_dir"])
    examples = Path(__file__).resolve().parent
    if stage == "build_library":
        subprocess.run(
            (
                sys.executable,
                str(examples / "prepare_width_scenarios.py"),
                "--config",
                str(config_path),
            ),
            check=True,
        )
        _run_commands(
            scenario_worker_commands(
                config_path=config_path,
                config=config,
                stage="build_library",
                gpus_per_node=gpus_per_node,
            )
        )
    elif stage in {"vllm_stats_diagnostic", "scoring", "scoring_diagnostic"}:
        _run_commands(
            scenario_worker_commands(
                config_path=config_path,
                config=config,
                stage=stage,
                gpus_per_node=gpus_per_node,
            )
        )
    elif stage == "mip":
        subprocess.run(
            (
                sys.executable,
                str(examples / "run_width_depth_mips.py"),
                "--config",
                str(config_path),
            ),
            check=True,
        )
    else:
        raise ValueError(f"unsupported embedding composite stage: {stage}")
    return {
        "widths": [
            int(width) for width in (config.get("embedding_pruning") or {}).get("widths", ())
        ],
        "scenarios_root": str(puzzle_dir / "scenarios"),
        "stage": stage,
    }
