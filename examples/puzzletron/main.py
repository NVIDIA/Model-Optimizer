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

# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One-command Puzzletron pipeline and single-stage runner.

The public process is an orchestrator. It launches a fresh worker process for
each stage so distributed state and GPU memory cannot leak between stages.
Workers are selected through the private ``--worker-stage`` option.
"""

from __future__ import annotations

import argparse
import faulthandler
import json
import os
import signal
import subprocess
import sys
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

import modelopt.torch.puzzletron as mtpz
from modelopt.torch.puzzletron.manifest import (
    StageManifest,
    semantic_stage_config,
    write_stage_manifest,
)
from modelopt.torch.puzzletron.orchestration.adapters.stage_compat import (
    stage_is_complete as artifacts_are_complete,
)
from modelopt.torch.puzzletron.orchestration.adapters.stage_compat import (
    stage_output_patterns as canonical_stage_output_patterns,
)
from modelopt.torch.puzzletron.stages.graph import (
    StageStatus,
    configured_parent_stage_ids,
    distributed_stage_ids,
    enabled_stage_ids,
    required_stage_ids,
    stage_ids,
    stage_is_enabled,
    stage_terminal_state,
    topological_stage_ids,
)

if __package__:
    from .acceptance_resume import build_payload, check_marker, marker_path, write_marker
else:
    from acceptance_resume import build_payload, check_marker, marker_path, write_marker

STAGES = stage_ids()
PIPELINE_STAGE_ORDER = topological_stage_ids()
REQUIRED_STAGES = frozenset(required_stage_ids())
DISTRIBUTED_STAGES = frozenset(distributed_stage_ids())
REQUIRED_OUTPUT_PATTERNS = {
    "convert": ("ckpts/teacher/config.json",),
    "tokenize_data": ("dataset_cache/*.tokens", "dataset_cache/*.tokens.json"),
    "sort": ("ckpts/sorted_teacher/config.json",),
    "slicing_sanity": (
        "artifacts/width_slice_equivalence/manifest.json",
        "artifacts/width_slice_equivalence/summary.json",
        "artifacts/width_slice_equivalence/cases/**/*.json",
        "artifacts/width_slice_equivalence/comparisons/*.safetensors",
    ),
    "depth_importance": ("depth/iterative/trajectory.json",),
    "build_library": (
        "replacement_library.json",
        "candidate_library.json",
        "subblock_stats.json",
    ),
    "vllm_stats": ("artifacts/vllm_stats/summary.json",),
    "replacement_scoring": ("artifacts/replacement_scoring/summary.json",),
    "mip": ("mip/**/*.json",),
    "zero_shot_evaluation": ("artifacts/**/evaluation_summary.json",),
    "aiperf": ("artifacts/aiperf/**/*.json",),
    "global_distillation_sanity": ("artifacts/global_distillation_sanity/**/*.json",),
    "global_distillation": ("artifacts/global_distillation/**/*.json",),
}


def _register_faulthandler() -> None:
    if not hasattr(signal, "SIGUSR1"):
        return
    stack_log = None
    if os.environ.get("SLURM_JOB_ID") and os.environ.get("RANK") is not None:
        stack_path = Path("puzzle_runs/logs") / (
            f"faulthandler_{os.environ['SLURM_JOB_ID']}_rank{os.environ['RANK']}.log"
        )
        stack_path.parent.mkdir(parents=True, exist_ok=True)
        stack_log = stack_path.open("a")
    faulthandler.register(signal.SIGUSR1, file=stack_log, all_threads=True)


_register_faulthandler()


def _stage_enabled(config: dict, stage: str) -> bool:
    return stage_is_enabled(stage, config)


def stage_sequence(stage: str | None, config: dict) -> tuple[str, ...]:
    """Return one explicit stage or every configured stage in dependency order."""

    if stage is not None:
        if stage == "full":
            return enabled_stage_ids(config)
        return (stage,)
    return enabled_stage_ids(config)


def _is_externally_launched() -> bool:
    """Return whether torchrun has already established this process group."""

    # Slurm's PMIx environment also exports RANK/WORLD_SIZE/LOCAL_RANK for a
    # single srun task.  torchrun uniquely provides this rendezvous identifier,
    # so it distinguishes a real worker from a shell that must self-launch.
    return os.environ.get("TORCHELASTIC_RUN_ID") is not None


def build_worker_command(
    *,
    config_path: str | Path,
    stage: str,
    overrides: Sequence[str],
    gpus_per_node: int,
    force_single: bool = False,
) -> tuple[str, ...]:
    """Build the isolated worker command for one stage."""

    command = [sys.executable]
    if stage in DISTRIBUTED_STAGES and not force_single:
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
            str(Path(__file__).resolve()),
            "--config",
            str(config_path),
            "--worker-stage",
            stage,
            "--gpus-per-node",
            str(gpus_per_node),
        )
    )
    for override in overrides:
        command.extend(("--override", str(override)))
    return tuple(command)


def refresh_campaign_report(config: dict, running_stage: str | None = None) -> None:
    """Refresh the stable campaign report from rank zero.

    Report generation must never fail a completed stage: the installed
    ``nvidia-modelopt`` wheel can shadow the local checkout and omit newer
    diagnostics modules.
    """

    if int(os.environ.get("RANK", "0")) != 0:
        return
    puzzle_dir = config.get("puzzle_dir") or (config.get("experiment") or {}).get("dir")
    if not puzzle_dir:
        return
    try:
        from modelopt.torch.puzzletron.diagnostics.campaign_progress_report import (
            generate_campaign_progress_report,
        )
    except Exception as exc:
        print(
            f"warning: skipping campaign progress report ({type(exc).__name__}: {exc})",
            flush=True,
        )
        return
    try:
        generate_campaign_progress_report(
            puzzle_dir,
            model_name=_report_model_name(config),
            running_stage=running_stage,
        )
    except Exception as exc:
        print(
            f"warning: campaign progress report failed ({type(exc).__name__}: {exc})",
            flush=True,
        )


def _report_model_name(config: dict) -> str:
    """Return a stable human identity rather than a resolved cache path."""

    model = config.get("model") or {}
    model_info = config.get("model_info") or {}
    return str(
        config.get("display_name")
        or model_info.get("hf_repo")
        or model.get("display_name")
        or model.get("name")
        or model.get("source")
        or "Puzzletron model"
    )


def _manifest_is_complete(config: dict, stage: str) -> bool:
    state = _manifest_terminal_state(config, stage)
    return state is not None and state.produced_artifacts and artifacts_are_complete(config, stage)


def _manifest_terminal_state(config: dict, stage: str):
    puzzle_dir = Path(config.get("puzzle_dir") or (config.get("experiment") or {})["dir"])
    path = puzzle_dir / "manifests" / f"{stage}.json"
    try:
        payload = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    state = stage_terminal_state(payload, expected_stage=stage)
    if state is None or not state.allows_completion(stage, config):
        return None
    return state


def _runtime_stats_filename(config: dict) -> str:
    stats = config.get("vllm_stats") or {}
    return str(stats.get("subblock_stats_filename", "subblock_stats.json"))


def _stage_output_patterns(config: dict, stage: str) -> tuple[str, ...]:
    if stage == "vllm_stats":
        return (_runtime_stats_filename(config),)
    if stage == "slicing_sanity":
        slicing_cfg = config.get("slicing_sanity") or {}
        if slicing_cfg.get("backend") == "distributed_parent_sweep":
            return ("artifacts/slicing_sanity/summary.json",)
    patterns = REQUIRED_OUTPUT_PATTERNS.get(stage, ())
    if stage == "build_library":
        resolved = [
            _runtime_stats_filename(config) if pattern == "subblock_stats.json" else pattern
            for pattern in patterns
        ]
        embedding = config.get("embedding_pruning") or {}
        if bool(embedding.get("enabled", False)):
            resolved.append("scenarios/width_scenarios.json")
            for configured_width in embedding.get("widths", ()):
                scenario = f"scenarios/width-{int(configured_width):04d}/depth-00"
                resolved.extend(
                    (
                        f"{scenario}/scenario_manifest.json",
                        f"{scenario}/replacement_library.json",
                        f"{scenario}/candidate_library.json",
                        f"{scenario}/{_runtime_stats_filename(config)}",
                        f"{scenario}/manifests/build_library.json",
                    )
                )
        return tuple(resolved)
    return patterns


def _resume_kwargs(config: dict, config_path: str | Path, stage: str) -> dict:
    puzzle_dir = Path(config.get("puzzle_dir") or (config.get("experiment") or {})["dir"])
    upstream = {
        parent: marker_path(puzzle_dir, parent, None, None)
        for parent in configured_parent_stage_ids(stage, config)
    }
    paths = config.get("paths") or {}
    repositories = tuple(
        Path(paths[key]) for key in ("automodel_root", "vllm_root", "aiperf_root") if paths.get(key)
    )
    return {
        "root": puzzle_dir,
        "config": Path(config_path),
        "mode": stage,
        "width": None,
        "depth": None,
        "required_patterns": (
            f"manifests/{stage}.json",
            *_stage_output_patterns(config, stage),
        ),
        "upstream_markers": upstream,
        "stage_config": semantic_stage_config(config, stage),
        "repository_roots": repositories,
    }


def _completion_is_valid(config: dict, config_path: str | Path, stage: str) -> bool:
    state = _manifest_terminal_state(config, stage)
    if state is None:
        return False
    if state.status is StageStatus.SKIPPED:
        return True
    if not artifacts_are_complete(config, stage):
        return False
    kwargs = _resume_kwargs(config, config_path, stage)
    return check_marker(marker_path(kwargs["root"], stage, None, None), **kwargs)


def _mark_completion(config: dict, config_path: str | Path, stage: str) -> None:
    state = _manifest_terminal_state(config, stage)
    if state is None:
        raise RuntimeError(f"stage {stage!r} did not write an accepted terminal manifest")
    if state.status is StageStatus.SKIPPED:
        return
    if not artifacts_are_complete(config, stage):
        raise RuntimeError(f"stage {stage!r} failed canonical artifact validation")
    kwargs = _resume_kwargs(config, config_path, stage)
    write_marker(kwargs["root"], stage, build_payload(**kwargs))


def _validate_worker_result(config: dict, result, *, expected_stage: str | None = None) -> None:
    """Fail the worker unless its result, manifest, and required artifacts agree."""

    expected_stage = expected_stage or result.stage
    if result.stage != expected_stage:
        raise RuntimeError(
            f"worker stage {expected_stage!r} returned result for stage {result.stage!r}"
        )
    puzzle_dir = Path(config.get("puzzle_dir") or (config.get("experiment") or {})["dir"])
    expected_manifest_path = puzzle_dir / "manifests" / f"{expected_stage}.json"
    if Path(result.manifest_path).resolve() != expected_manifest_path.resolve():
        raise RuntimeError(
            f"stage {expected_stage!r} returned manifest path {result.manifest_path!s}; "
            f"expected {expected_manifest_path!s}"
        )
    try:
        payload = json.loads(expected_manifest_path.read_text())
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"stage {expected_stage!r} wrote an unreadable manifest") from exc
    if payload.get("stage") != expected_stage:
        raise RuntimeError(
            f"stage {expected_stage!r} manifest identifies stage {payload.get('stage')!r}"
        )
    state = stage_terminal_state(payload, expected_stage=expected_stage)
    if state is None or not state.allows_completion(expected_stage, config):
        raise RuntimeError(f"stage {expected_stage!r} wrote an invalid terminal manifest")
    if result.status != state.status.value:
        raise RuntimeError(
            f"stage {expected_stage!r} result status {result.status!r} disagrees with "
            f"manifest status {state.status.value!r}"
        )
    expected_reason = state.skip_reason.value if state.skip_reason is not None else None
    if result.skip_reason != expected_reason:
        raise RuntimeError(
            f"stage {expected_stage!r} result skip reason {result.skip_reason!r} disagrees with "
            f"manifest skip reason {expected_reason!r}"
        )
    if not state.produced_artifacts:
        return
    if not artifacts_are_complete(config, expected_stage):
        expected = canonical_stage_output_patterns(config, expected_stage)
        raise RuntimeError(
            f"stage {expected_stage!r} failed canonical artifact validation; expected: "
            + (", ".join(expected) or "stage-specific outputs")
        )


def run_pipeline(
    *,
    config_path: str | Path,
    config: dict,
    stages: Sequence[str],
    overrides: Sequence[str],
    gpus_per_node: int,
    force: bool,
    is_complete: Callable[[str], bool],
    mark_complete: Callable[[str], None],
    refresh_report: Callable[[str | None], None],
    command_runner: Callable[[Sequence[str]], subprocess.CompletedProcess] = subprocess.run,
) -> None:
    """Run stage workers sequentially, stopping on the first failed worker."""

    refresh_report(None)
    for stage in stages:
        if not force and is_complete(stage):
            continue
        refresh_report(stage)
        command = build_worker_command(
            config_path=config_path,
            stage=stage,
            overrides=overrides,
            gpus_per_node=gpus_per_node,
            force_single=bool((config.get("embedding_pruning") or {}).get("enabled", False))
            and stage == "replacement_scoring",
        )
        result = command_runner(command)
        refresh_report(None)
        if result.returncode:
            raise subprocess.CalledProcessError(result.returncode, command)
        mark_complete(stage)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Puzzletron compression pipeline.")
    parser.add_argument("--config", required=True, help="Hydra YAML entrypoint.")
    parser.add_argument("--stage", choices=("full", *STAGES))
    parser.add_argument("--force", action="store_true", help="Rerun the selected stage(s).")
    parser.add_argument("--gpus-per-node", type=int, default=None)
    parser.add_argument("--override", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--worker-stage", choices=STAGES, help=argparse.SUPPRESS)
    parser.add_argument("--scenario-child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.gpus_per_node is not None and args.gpus_per_node < 1:
        parser.error("--gpus-per-node must be >= 1")
    if args.force and args.worker_stage:
        parser.error("--force is an orchestrator option")
    return args


def _embedding_followup_stage(stage: str) -> bool:
    """Return whether a completed root stage fans out into width scenarios."""
    return stage == "build_library"


def _run_worker(args: argparse.Namespace) -> None:
    cfg = mtpz.pipeline_config.pipeline_config_from_path(
        args.config,
        overrides=args.override,
    )
    embedding_root = (
        bool((cfg.get("embedding_pruning") or {}).get("enabled", False)) and not args.scenario_child
    )
    gpus_per_node = int(args.gpus_per_node or (cfg.get("execution") or {}).get("gpus_per_node", 8))
    composite_only = {"replacement_scoring", "mip"}
    if not _stage_enabled(cfg, args.worker_stage):
        result = mtpz.stage_runner.run_stage(cfg, args.worker_stage, handlers={})
    elif args.worker_stage == "tokenize_data":
        if __package__:
            from .tokenize_data import tokenize_data_stage
        else:
            from tokenize_data import tokenize_data_stage

        result = tokenize_data_stage(cfg)
    elif embedding_root and args.worker_stage in composite_only:
        if __package__:
            from .embedding_pipeline import run_embedding_stage
        else:
            from embedding_pipeline import run_embedding_stage

        outputs = run_embedding_stage(
            config_path=args.config,
            config=cfg,
            stage=args.worker_stage,
            gpus_per_node=gpus_per_node,
        )
        result = _complete_composite_stage(cfg, args.worker_stage, outputs)
    else:
        result = mtpz.stage_runner.run_stage(cfg, args.worker_stage)
        if embedding_root and _embedding_followup_stage(args.worker_stage):
            if __package__:
                from .embedding_pipeline import run_embedding_stage
            else:
                from embedding_pipeline import run_embedding_stage

            outputs = run_embedding_stage(
                config_path=args.config,
                config=cfg,
                stage=args.worker_stage,
                gpus_per_node=gpus_per_node,
            )
            outputs["base_manifest"] = str(result.manifest_path)
            result = _complete_composite_stage(cfg, args.worker_stage, outputs)
    if int(os.environ.get("RANK", "0")) == 0:
        _validate_worker_result(cfg, result, expected_stage=args.worker_stage)
    refresh_campaign_report(cfg)
    mtpz.tools.mprint(
        f"Puzzletron stage {result.stage!r} finished with status {result.status}: "
        f"{result.manifest_path}"
    )


def _complete_composite_stage(config: dict, stage: str, outputs: dict):
    puzzle_dir = Path(config.get("puzzle_dir") or (config.get("experiment") or {})["dir"])
    manifest_path = puzzle_dir / "manifests" / f"{stage}.json"
    manifest = StageManifest(stage=stage, inputs={"config": config}, config=config)
    manifest.complete(outputs=outputs)
    write_stage_manifest(manifest_path, manifest)
    return mtpz.stage_runner.StageResult(
        stage=stage,
        status="success",
        manifest_path=manifest_path,
        message=f"Completed embedding-width composite stage {stage!r}.",
    )


def main() -> None:
    args = _parse_args()
    if args.worker_stage:
        _run_worker(args)
        return

    cfg = mtpz.pipeline_config.pipeline_config_from_path(args.config, overrides=args.override)
    execution = cfg.get("execution") or {}
    gpus_per_node = int(args.gpus_per_node or execution.get("gpus_per_node", 8))
    stages = stage_sequence(args.stage, cfg)
    if _is_externally_launched():
        if len(stages) != 1 or stages[0] not in DISTRIBUTED_STAGES:
            raise RuntimeError(
                "An externally launched distributed job must run exactly one distributed stage."
            )
        stage = stages[0]
        if args.force or not _completion_is_valid(cfg, args.config, stage):
            args.worker_stage = stage
            _run_worker(args)
            if int(os.environ["RANK"]) == 0:
                _mark_completion(cfg, args.config, stage)
        refresh_campaign_report(cfg)
        return
    run_pipeline(
        config_path=args.config,
        config=cfg,
        stages=stages,
        overrides=tuple(args.override),
        gpus_per_node=gpus_per_node,
        force=args.force,
        is_complete=partial(_completion_is_valid, cfg, args.config),
        mark_complete=partial(_mark_completion, cfg, args.config),
        refresh_report=lambda running: refresh_campaign_report(cfg, running),
    )


if __name__ == "__main__":
    main()
