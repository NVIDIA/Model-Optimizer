# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Embedding-width fan-out used by the one-command Puzzletron runner."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from modelopt.torch.puzzletron.diagnostics import generate_replace_block_report
from modelopt.torch.puzzletron.stages.graph import distributed_stage_ids

__all__ = [
    "finalize_replacement_scoring_diagnostics",
    "run_embedding_stage",
    "scenario_preparation_commands",
    "scenario_worker_commands",
]


def _scenario_dir(puzzle_dir: Path, width: int) -> Path:
    return puzzle_dir / "scenarios" / f"width-{width:04d}" / "depth-00"


def _visible_gpu_count(fallback: int) -> int:
    visible = tuple(
        value.strip()
        for value in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
        if value.strip()
    )
    return len(visible) if visible else int(fallback)


def _project_vllm_stats_to_scenarios(config: dict) -> dict[int, Path]:
    """Publish the root vLLM aggregate entries owned by each hidden-width scenario."""

    puzzle_dir = Path(config["puzzle_dir"])
    root_stats_path = puzzle_dir / "subblock_stats.json"
    root_stats = json.loads(root_stats_path.read_text())
    if not isinstance(root_stats, list):
        raise TypeError(f"vLLM aggregate must be a list: {root_stats_path}")

    outputs = {}
    widths = (config.get("embedding_pruning") or {}).get("widths", ())
    for configured_width in widths:
        width = int(configured_width)
        width_stats = [
            entry
            for entry in root_stats
            if int((entry.get("args") or {}).get("n_embd", -1)) == width
        ]
        if not width_stats:
            raise ValueError(
                f"vLLM aggregate {root_stats_path} has no entries for hidden width {width}"
            )
        output = _scenario_dir(puzzle_dir, width) / "subblock_stats.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(json.dumps(width_stats, indent=2) + "\n")
        temporary.replace(output)
        outputs[width] = output
    return outputs


def finalize_replacement_scoring_diagnostics(config: dict) -> dict:
    """Publish per-width reports and one root summary for embedding campaigns."""

    puzzle_dir = Path(config["puzzle_dir"])
    scoring = config.get("replacement_scoring") or {}
    granularity = str(scoring.get("granularity", "block"))
    scores_name = (
        "single_subblock_replacement_solutions--validation"
        if granularity == "subblock"
        else "single_sequence_replacement_solutions--validation"
    )
    widths = [int(width) for width in (config.get("embedding_pruning") or {}).get("widths", ())]
    children = []
    for width in widths:
        scenario = _scenario_dir(puzzle_dir, width)
        children.append(
            generate_replace_block_report(
                scenario,
                scores_dir=scenario / scores_name,
                output_dir=scenario / "artifacts" / "replacement_scoring",
                granularity=granularity,
                default_metric=str(
                    scoring.get("default_metric", "normalized_mse_loss_hidden_states")
                ),
                default_layer_count=int(scoring.get("default_layer_count", 5)),
                anchor_count=int(scoring.get("anchor_count", 3)),
                trend_relative_tolerance=float(scoring.get("trend_relative_tolerance", 0.02)),
            )
        )

    summary = {
        "version": 1,
        "kind": "replacement_scoring",
        "granularity": granularity,
        "widths": widths,
        "scenario_count": len(children),
        "record_count": sum(int(child.get("record_count", 0)) for child in children),
        "warning_count": sum(int(child.get("warning_count", 0)) for child in children),
        "axes": list(dict.fromkeys(axis for child in children for axis in child.get("axes", ()))),
        "metrics": list(
            dict.fromkeys(metric for child in children for metric in child.get("metrics", ()))
        ),
        "children": children,
    }
    output = puzzle_dir / "artifacts" / "replacement_scoring" / "summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    temporary.replace(output)
    return summary


def _scenario_overrides(config: dict, scenario: Path) -> tuple[str, ...]:
    teacher = scenario / "ckpts" / "sorted_teacher"
    scenario_manifest = json.loads((scenario / "scenario_manifest.json").read_text())
    subblock = str((config.get("replacement_scoring") or {}).get("granularity")) == "subblock"
    stem = (
        "single_subblock_replacement_solutions"
        if subblock
        else "single_sequence_replacement_solutions"
    )
    scoring_solutions = scenario / f"{stem}.json"
    scoring_output = scenario / f"{stem}--validation"
    overrides = [
        f"puzzle_dir={scenario}",
        f"experiment.dir={scenario}",
        f"teacher_dir={teacher}",
        f"convert.teacher_dir={teacher}",
        "bypass.enabled=false",
        f"replacement_library_path={scenario / 'replacement_library.json'}",
        f"build_replacement_library.source_checkpoint_dir={teacher}",
        "calc_subblock_stats.runtime_stats.execution=inline",
        f"replacement_scoring.teacher_dir={teacher}",
        f"replacement_scoring.source_checkpoint_dir={teacher}",
        f"replacement_scoring.target_teacher_dir={teacher}",
        f"replacement_scoring.solutions_path={scoring_solutions}",
        f"replacement_scoring.output_dir={scoring_output}",
        f"vllm_stats_diagnostic.stats_path={scenario / 'subblock_stats.json'}",
        f"vllm_stats_diagnostic.output_dir={scenario / 'artifacts/vllm_stats_diagnostic'}",
        f"scoring_diagnostic.scores_dir={scoring_output}",
        f"scoring_diagnostic.output_dir={scenario / 'artifacts/scoring_diagnostic'}",
    ]
    if scenario_manifest.get("bypass_checkpoint") is not None:
        overrides.append(
            f"replacement_scoring.bypass_checkpoint_dir={scenario / 'ckpts' / 'bypass_overlay'}"
        )
    return tuple(overrides)


def scenario_preparation_commands(*, config: dict, stage: str) -> tuple[tuple[str, ...], ...]:
    """Return width-local input preparation commands for a composite stage."""

    replacement = config.get("replacement_scoring") or {}
    if stage != "replacement_scoring" or replacement.get("granularity") != "subblock":
        return ()
    script = Path(__file__).resolve().parent / "prepare_subblock_replacement_scoring.py"
    puzzle_dir = Path(config["puzzle_dir"])
    trust_remote_code = bool((config.get("model") or {}).get("trust_remote_code", False))
    commands = []
    for width in (config.get("embedding_pruning") or {}).get("widths", ()):
        command = [
            sys.executable,
            str(script),
            "--puzzle-dir",
            str(_scenario_dir(puzzle_dir, int(width))),
        ]
        if trust_remote_code:
            command.append("--trust-remote-code")
        commands.append(tuple(command))
    return tuple(commands)


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
        if stage in distributed_stage_ids():
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
    outputs = {
        "widths": [
            int(width) for width in (config.get("embedding_pruning") or {}).get("widths", ())
        ],
        "scenarios_root": str(puzzle_dir / "scenarios"),
        "stage": stage,
    }
    # A distributed root stage invokes this function in every torchrun
    # process.  Width preparation and nested workers own shared artifacts and
    # GPUs, so only the global rank-zero process may fan them out.
    if int(os.environ.get("RANK", "0")) != 0:
        outputs["skipped_nonzero_rank"] = True
        return outputs

    examples = Path(__file__).resolve().parent
    worker_gpus = _visible_gpu_count(gpus_per_node)
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
                gpus_per_node=worker_gpus,
            )
        )
        # Scenario workers rewrite local subblock_stats.json with static
        # parameter inventories. Re-project the root vLLM aggregate afterward
        # so MIP sees runtime_stats=True identities.
        _project_vllm_stats_to_scenarios(config)
    elif stage in {
        "vllm_stats_diagnostic",
        "replacement_scoring",
        "scoring_diagnostic",
    }:
        _run_commands(scenario_preparation_commands(config=config, stage=stage))
        _run_commands(
            scenario_worker_commands(
                config_path=config_path,
                config=config,
                stage=stage,
                gpus_per_node=worker_gpus,
            )
        )
        if stage == "replacement_scoring":
            finalize_replacement_scoring_diagnostics(config)
    elif stage == "mip":
        subprocess.run(
            (
                sys.executable,
                str(examples / "run_width_depth_mips.py"),
                "--config",
                str(config_path),
                "--solve-only",
            ),
            check=True,
        )
    else:
        raise ValueError(f"unsupported embedding composite stage: {stage}")
    return outputs
