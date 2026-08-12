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

from __future__ import annotations

import copy
import json
import math
import os
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import asdict
from pathlib import Path
from queue import Queue
from typing import TYPE_CHECKING, Any, Sequence

from ..anymodel.model_descriptor import ModelDescriptorFactory
from ..anymodel.registry import resolve_descriptor_from_pretrained
from ..distillation.global_automodel import (
    GlobalKDConfig,
    GlobalKDResult,
    build_global_kd_config,
    run_global_kd,
)
from .common import complete_stage
from .graph import StageSkipReason

if TYPE_CHECKING:
    from ..manifest import StageManifest

__all__ = [
    "aiperf_stage",
    "evaluation_stage",
    "post_distillation_evaluation_stage",
    "distillation_overfit_stage",
    "distillation_stage",
]


def _distributed_barrier(context: str) -> None:
    """Synchronize initialized ranks and preserve collective failures."""
    import torch.distributed as torch_dist

    if not (torch_dist.is_available() and torch_dist.is_initialized()):
        return
    try:
        torch_dist.barrier()
    except RuntimeError as error:
        raise RuntimeError(f"{context} barrier failed: {error}") from error


def _resolve_evaluation_descriptor(config: dict[str, Any], checkpoint: Path):
    """Resolve the evaluation descriptor without requiring a public config override."""
    descriptor_name = config.get("descriptor")
    if descriptor_name:
        return ModelDescriptorFactory.get(str(descriptor_name))
    model_cfg = dict(config.get("model") or {})
    descriptor_name = model_cfg.get("descriptor_override")
    if descriptor_name:
        return ModelDescriptorFactory.get(str(descriptor_name))
    return resolve_descriptor_from_pretrained(
        str(checkpoint),
        trust_remote_code=bool(model_cfg.get("trust_remote_code", False)),
    ).descriptor


def _scenario_grid_kd_configs(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand the realized width/depth grid into isolated global-KD configs."""
    puzzle_dir = Path((config.get("experiment") or {})["dir"])
    checkpoints = sorted(
        path.parent
        for path in (puzzle_dir / "scenarios").glob(
            "width-*/depth-*/checkpoints/solution_0/config.json"
        )
    )
    if not checkpoints:
        raise FileNotFoundError("scenario-grid global KD found no realized checkpoints")
    model_cfg = dict(config.get("model") or {})
    teacher_dir = str((config.get("convert") or {})["teacher_dir"])
    candidates = []
    for checkpoint in checkpoints:
        relative = checkpoint.relative_to(puzzle_dir / "scenarios")
        scenario = relative.parents[1]
        candidate = copy.deepcopy(config)
        kd = dict(candidate.get("distillation") or {})
        kd["student_dir"] = str(checkpoint)
        kd["teacher_dir"] = teacher_dir
        kd["output_dir"] = str(puzzle_dir / "artifacts" / "global_kd" / "scenarios" / scenario)
        kd.pop("tournament", None)
        candidate["distillation"] = kd
        resolution = resolve_descriptor_from_pretrained(
            str(checkpoint),
            trust_remote_code=bool(model_cfg.get("trust_remote_code", False)),
        )
        runtime = dict(candidate.get("_runtime") or {})
        runtime["descriptor"] = resolution.name
        candidate["_runtime"] = runtime
        candidates.append(candidate)
    return candidates


def _scenario_grid_global_kd_checkpoints(
    puzzle_dir: Path,
) -> list[tuple[str, Path]]:
    """Return all published consolidated global-distillation checkpoints."""
    scenario_root = puzzle_dir / "artifacts" / "global_kd" / "scenarios"
    checkpoints = []
    for scenario in sorted(scenario_root.glob("width-*/depth-*")):
        candidates = []
        for config_path in scenario.glob(
            "checkpoints/epoch_*_step_*/model/consolidated/config.json"
        ):
            checkpoint_dir = config_path.parent
            step_dir = checkpoint_dir.parents[1].name
            try:
                step = int(step_dir.rsplit("_step_", 1)[1])
            except (IndexError, ValueError):
                continue
            candidates.append((step, checkpoint_dir))
        if not candidates:
            continue
        _, checkpoint = max(candidates, key=lambda item: item[0])
        relative = scenario.relative_to(scenario_root)
        checkpoints.append(("__".join(relative.parts), checkpoint))

    canonical_root = puzzle_dir / "artifacts" / "global_distillation"
    for summary_path in sorted(canonical_root.glob("**/global_distillation_summary.json")):
        try:
            summary = json.loads(summary_path.read_text())
            checkpoint = Path(summary["post_kd_checkpoint"])
        except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
        if not (checkpoint / "config.json").is_file():
            continue
        profile_id = str(summary.get("profile_id", "unknown"))
        solution_id = str(summary.get("solution_id", checkpoint.parent.name))
        checkpoints.append((f"{profile_id}__{solution_id}", checkpoint))

    unique = {}
    for name, checkpoint in checkpoints:
        unique[str(checkpoint.resolve())] = (name, checkpoint)
    return sorted(unique.values(), key=lambda item: item[0])


def _selection_reasons(row: dict[str, Any]) -> list[str]:
    """Return one stable copy of a candidate's selection provenance."""
    return list(dict.fromkeys(str(reason) for reason in row.get("selection_reasons", ())))


def _select_evaluated_candidates(
    rows: list[dict[str, Any]] | tuple[dict[str, Any], ...], *, num_best: int
) -> list[dict[str, Any]]:
    """Select the best finite-LM-loss candidates with stable deduplication."""
    candidates: dict[str, tuple[float, dict[str, Any]]] = {}
    for row in rows:
        checkpoint = row.get("checkpoint")
        metrics = row.get("metrics") or {}
        try:
            loss = float(metrics["lm_loss"])
        except (KeyError, TypeError, ValueError):
            continue
        if not checkpoint or not math.isfinite(loss):
            continue
        key = str(checkpoint)
        normalized = dict(row)
        normalized["selection_reasons"] = _selection_reasons(row)
        current = candidates.get(key)
        if current is None or (loss, str(row.get("solution_id", ""))) < (
            current[0],
            str(current[1].get("solution_id", "")),
        ):
            if current is not None:
                normalized["selection_reasons"] = list(
                    dict.fromkeys(normalized["selection_reasons"] + current[1]["selection_reasons"])
                )
            candidates[key] = (loss, normalized)
        else:
            current[1]["selection_reasons"] = list(
                dict.fromkeys(current[1]["selection_reasons"] + normalized["selection_reasons"])
            )
    selected = sorted(
        candidates.values(),
        key=lambda item: (item[0], str(item[1]["checkpoint"]), str(item[1].get("solution_id", ""))),
    )
    return [row for _, row in selected[:num_best]]


def _with_teacher_checkpoint(
    teacher_dir: str | Path | None, candidates: Sequence[tuple[str, str | Path]]
) -> list[tuple[str, str | Path]]:
    """Prepend the configured teacher while keeping downstream checkpoints unique."""
    if teacher_dir is None:
        return list(candidates)
    teacher_key = str(teacher_dir)
    return [("teacher", teacher_dir)] + [
        (name, checkpoint)
        for name, checkpoint in candidates
        if name != "teacher" and str(checkpoint) != teacher_key
    ]


def _profile_solution_checkpoints(
    puzzle_dir: Path, profile_id: str | None = None
) -> list[tuple[str, Path]]:
    """Return the selected MIP profile checkpoints when one profile is unambiguous."""

    profiles_root = puzzle_dir / "mip" / "profiles"
    if profile_id is None:
        index_path = profiles_root / "index.json"
        if index_path.is_file():
            index = json.loads(index_path.read_text())
            profile_ids = [
                str(row["id"])
                for row in index.get("profiles", ())
                if (profiles_root / str(row.get("id")) / "selected_solutions.json").is_file()
            ]
        else:
            profile_ids = [
                path.parent.name for path in profiles_root.glob("*/selected_solutions.json")
            ]
        if len(profile_ids) != 1:
            return []
        profile_id = profile_ids[0]

    registry_path = profiles_root / str(profile_id) / "selected_solutions.json"
    if not registry_path.is_file():
        return []
    registry = json.loads(registry_path.read_text())
    checkpoints = [
        (str(row["solution_id"]), Path(str(row["checkpoint"])))
        for row in registry.get("solutions", ())
    ]
    if len({name for name, _ in checkpoints}) != len(checkpoints):
        raise ValueError(f"profile registry has duplicate solution IDs: {registry_path}")
    return checkpoints


def _aiperf_executable(stage_cfg: dict[str, Any]) -> str:
    """Resolve an explicit client first, then the shared environment client."""
    return str(stage_cfg.get("executable") or os.environ.get("AIPERF_EXECUTABLE", "aiperf"))


def _bounded_map(function, items, *, max_workers: int):
    """Map without eagerly queuing work that would survive an early failure."""
    iterator = iter(items)
    results = []
    executor = ThreadPoolExecutor(max_workers=max_workers)
    active = {}
    try:
        for _ in range(max_workers):
            try:
                item = next(iterator)
            except StopIteration:
                break
            active[executor.submit(function, item)] = item
        while active:
            completed, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in completed:
                active.pop(future)
                results.append(future.result())
                try:
                    item = next(iterator)
                except StopIteration:
                    continue
                active[executor.submit(function, item)] = item
    except BaseException:
        for future in active:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)
        raise
    executor.shutdown(wait=True)
    return results


def _aiperf_checkpoint_work(checkpoints, concurrencies):
    """Keep one checkpoint's compile identity serial across concurrency sweeps."""
    values = tuple(int(value) for value in concurrencies)
    return [(name, checkpoint, values) for name, checkpoint in checkpoints]


def aiperf_stage(config: dict[str, Any], manifest: StageManifest):
    stage_cfg = dict(config.get("aiperf") or {})
    if not bool(stage_cfg.get("enabled", False)):
        return complete_stage(
            config,
            manifest,
            outputs={"skipped": True},
            status="skipped",
            skip_reason=StageSkipReason.DISABLED,
            message="AIPerf is disabled.",
        )
    from ..benchmarks import run_aiperf_sweep, write_aiperf_report

    experiment_dir = (config.get("experiment") or {}).get("dir")
    if experiment_dir is None:
        raise ValueError("AIPerf requires experiment.dir")
    puzzle_dir = Path(experiment_dir)
    teacher_dir = (config.get("convert") or {}).get("teacher_dir")
    checkpoint_root = Path(
        stage_cfg.get(
            "solution_checkpoints_dir",
            puzzle_dir / "mip" / "puzzle_solutions" / "depth_tournament" / "solutions--checkpoints",
        )
    )
    checkpoints: list[tuple[str, str | Path]] = []
    if stage_cfg.get("checkpoint_source") == "global_kd":
        checkpoints.extend(_scenario_grid_global_kd_checkpoints(puzzle_dir))
    elif stage_cfg.get("checkpoint_source") == "scenario_grid":
        checkpoints.extend(
            (name, checkpoint)
            for name, checkpoint in _profile_solution_checkpoints(
                puzzle_dir, stage_cfg.get("profile_id")
            )
            if name != "teacher"
        )
        if not checkpoints:
            checkpoints.extend(
                (
                    "-".join(path.parts[-5:-3]),
                    path.parent,
                )
                for path in sorted(
                    (puzzle_dir / "scenarios").glob(
                        "width-*/depth-*/checkpoints/solution_0/config.json"
                    )
                )
            )
    else:
        checkpoints.extend(
            (path.name, path)
            for path in sorted(checkpoint_root.glob("solution_*"))
            if (path / "config.json").is_file()
        )
    evaluation_summary_path = Path(
        stage_cfg.get("evaluation_summary_path")
        or puzzle_dir / "artifacts" / "zero_shot_evaluation" / "evaluation_summary.json"
    )
    if checkpoints:
        rows = json.loads(evaluation_summary_path.read_text())
        selected = _select_evaluated_candidates(
            [row for row in rows if str(row.get("checkpoint")) != str(teacher_dir)],
            num_best=int(stage_cfg.get("num_best_to_eval", 1)),
        )
        selected_paths = {str(row["checkpoint"]) for row in selected}
        checkpoints = [
            (name, checkpoint)
            for name, checkpoint in checkpoints
            if str(checkpoint) in selected_paths
        ]
    checkpoints = _with_teacher_checkpoint(teacher_dir, checkpoints)
    expected = stage_cfg.get("expected_solution_count")
    if expected is not None and len(checkpoints) > int(expected) + 1:
        raise RuntimeError(
            f"AIPerf expected at most teacher + {expected} solutions, found {len(checkpoints)}"
        )
    if len(checkpoints) == 1:
        raise FileNotFoundError("AIPerf found no realized candidate checkpoints")
    topology = dict(stage_cfg.get("topology") or {})
    group_size = int(topology.get("gpu_group_size", 4))
    visible = [
        item.strip()
        for item in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
        if item.strip()
    ]
    if not visible:
        visible = [str(index) for index in range(group_size)]
    if len(visible) < group_size or len(visible) % group_size:
        raise RuntimeError(
            f"AIPerf visible GPU count {len(visible)} must be a positive multiple "
            f"of gpu_group_size={group_size}"
        )
    gpu_groups = [
        ",".join(visible[index : index + group_size])
        for index in range(0, len(visible), group_size)
    ]
    output_dir = Path(stage_cfg.get("output_dir", puzzle_dir / "artifacts" / "aiperf"))
    model_cfg = dict(config.get("model") or {})
    trust_remote_code = bool(
        stage_cfg.get("trust_remote_code", model_cfg.get("trust_remote_code", False))
    )
    allow_aiperf_v011_online_tokenizer_resolution = bool(
        stage_cfg.get("allow_aiperf_v011_online_tokenizer_resolution", False)
    )
    work = _aiperf_checkpoint_work(checkpoints, list(stage_cfg.get("concurrency", [1, 2, 4, 8])))
    pool: Queue[str] = Queue()
    for gpu_group in gpu_groups:
        pool.put(gpu_group)

    def _run_checkpoint(item):
        name, checkpoint, concurrencies = item
        gpu_ids = pool.get()
        try:
            request_counts = {
                concurrency: max(
                    int(stage_cfg.get("minimum_request_count", 4)),
                    int(stage_cfg.get("requests_per_concurrency", 2)) * concurrency,
                )
                for concurrency in concurrencies
            }
            return run_aiperf_sweep(
                checkpoint,
                artifact_dir=output_dir / name,
                concurrencies=concurrencies,
                input_tokens=int(stage_cfg.get("input_tokens", 122880)),
                output_tokens=int(stage_cfg.get("output_tokens", 8192)),
                gpu_ids=gpu_ids,
                topology=topology,
                request_counts=request_counts,
                solution_id=name,
                profile_id=str(stage_cfg.get("profile_id", "unknown")),
                executable=_aiperf_executable(stage_cfg),
                endpoint_type=str(stage_cfg.get("endpoint_type", "chat")),
                extra_inputs=dict(stage_cfg.get("extra_inputs") or {}),
                use_server_token_count=bool(stage_cfg.get("use_server_token_count", True)),
                seed=int(stage_cfg.get("seed", 42)),
                trust_remote_code=trust_remote_code,
                allow_aiperf_v011_online_tokenizer_resolution=(
                    allow_aiperf_v011_online_tokenizer_resolution
                ),
            )
        finally:
            pool.put(gpu_ids)

    nested_results = _bounded_map(_run_checkpoint, work, max_workers=len(gpu_groups))
    results = [result for checkpoint_results in nested_results for result in checkpoint_results]
    reports = write_aiperf_report(results, output_dir)
    return complete_stage(
        config,
        manifest,
        outputs={"result_count": len(results), "reports": reports},
    )


def evaluation_stage(config: dict[str, Any], manifest: StageManifest):
    stage_cfg = dict(config.get("zero_shot_evaluation") or {})
    if not bool(stage_cfg.get("enabled", False)):
        return complete_stage(
            config,
            manifest,
            outputs={"skipped": True},
            status="skipped",
            skip_reason=StageSkipReason.DISABLED,
            message="Zero-shot evaluation is disabled.",
        )
    from omegaconf import OmegaConf

    import modelopt.torch.utils.distributed as dist

    from ..block_config import maybe_cast_block_configs
    from ..identity import canonicalize
    from ..pipeline_config import load_runtime_hydra_config
    from ..plugins.automodel.solution_launch import launch_score_solutions_automodel
    from ..tools.checkpoint_utils import load_model_config
    from ..tools.hydra_utils import clone_hydra_config
    from .pipeline import _distributed

    puzzle_dir = Path((config.get("experiment") or {})["dir"])
    configured = stage_cfg.get("checkpoints")
    raw_checkpoint_entries: list[tuple[str, str | Path]]
    if configured:
        raw_checkpoint_entries = [(Path(path).name, Path(path)) for path in configured]
    elif stage_cfg.get("checkpoint_source") == "global_kd":
        raw_checkpoint_entries = _scenario_grid_global_kd_checkpoints(puzzle_dir)
    else:
        raw_checkpoint_entries = [
            (name, checkpoint)
            for name, checkpoint in _profile_solution_checkpoints(
                puzzle_dir, stage_cfg.get("profile_id")
            )
            if name != "teacher"
        ]
        if not raw_checkpoint_entries:
            raw_checkpoint_entries = [
                (path.parent.name, path.parent)
                for path in sorted(
                    (puzzle_dir / "scenarios").glob(
                        "width-*/depth-*/checkpoints/solution_0/config.json"
                    )
                )
            ]
    teacher_dir = (config.get("convert") or {}).get("teacher_dir")
    checkpoint_entries = [
        (name, Path(checkpoint))
        for name, checkpoint in _with_teacher_checkpoint(teacher_dir, raw_checkpoint_entries)
    ]
    if len(checkpoint_entries) == 1 and teacher_dir is None:
        raise FileNotFoundError("exact evaluation found no scenario checkpoints")
    if teacher_dir is None:
        raise ValueError("exact evaluation requires convert.teacher_dir")
    teacher_dir = Path(teacher_dir)
    descriptor = _resolve_evaluation_descriptor(config, checkpoint_entries[0][1])
    hydra_cfg = load_runtime_hydra_config(config)
    root = Path(stage_cfg.get("output_dir", puzzle_dir / "artifacts" / "zero_shot_evaluation"))
    summaries = []

    with _distributed(hydra_cfg):
        for solution_id, checkpoint in checkpoint_entries:
            checkpoint_config = load_model_config(
                checkpoint,
                trust_remote_code=descriptor.requires_trust_remote_code(),
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
            solution = {
                "single_sequence_replacement": identity,
                "chosen_replacements": replacements,
                "block_configs": [block.to_dict() for block in blocks],
                "hidden_width": int(lm.hidden_size),
            }
            relative = (
                checkpoint.relative_to(puzzle_dir)
                if checkpoint.is_relative_to(puzzle_dir)
                else Path(checkpoint.name)
            )
            output_dir = root / relative.parent
            solutions_path = output_dir / "identity_solution.json"
            if dist.is_master():
                output_dir.mkdir(parents=True, exist_ok=True)
                solutions_path.write_text(json.dumps([solution], indent=2, sort_keys=True) + "\n")
            dist.barrier()
            cfg = clone_hydra_config(hydra_cfg)
            OmegaConf.set_struct(cfg, False)
            cfg.scoring.teacher_dir = str(teacher_dir)
            cfg.scoring.target_teacher_dir = str(teacher_dir)
            cfg.scoring.source_checkpoint_dir = str(checkpoint)
            cfg.scoring.solutions_path = str(solutions_path)
            cfg.scoring.output_dir = str(output_dir)
            cfg.scoring.solutions_to_validate = None
            cfg.scoring.skip_existing_solutions = True
            cfg.scoring.eval_samples = int(stage_cfg.get("eval_samples", 16))
            cfg.scoring.micro_batch_size = int(stage_cfg.get("micro_batch_size", 1))
            cfg.scoring.block_size = int(stage_cfg.get("block_size", 2048))
            launch_score_solutions_automodel(cfg)
            dist.barrier()
            if dist.is_master():
                result_path = output_dir / "sliced_teacher.json"
                result = json.loads(result_path.read_text())
                summaries.append(
                    {
                        "checkpoint": str(checkpoint),
                        "solution_id": solution_id,
                        "hidden_width": int(lm.hidden_size),
                        "result_path": str(result_path),
                        "metrics": {
                            key: value["avg"]
                            for key, value in result.items()
                            if isinstance(value, dict) and "avg" in value
                        },
                        "observability": result.get("observability", {}),
                    }
                )
        if dist.is_master():
            summary_path = root / "evaluation_summary.json"
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary_path.write_text(json.dumps(summaries, indent=2, sort_keys=True) + "\n")
        dist.barrier()
    return complete_stage(
        config,
        manifest,
        outputs={
            "checkpoint_count": len(checkpoint_entries),
            "summary_path": str(root / "evaluation_summary.json"),
        },
    )


def post_distillation_evaluation_stage(config: dict[str, Any], manifest: StageManifest):
    """Evaluate globally distilled checkpoints through the shared evaluator."""

    candidate = copy.deepcopy(config)
    stage_cfg = dict(candidate.get("post_distillation_evaluation") or {})
    puzzle_dir = Path((candidate.get("experiment") or {})["dir"])
    stage_cfg.setdefault("checkpoint_source", "global_kd")
    stage_cfg.setdefault(
        "output_dir", str(puzzle_dir / "artifacts" / "post_distillation_evaluation")
    )
    candidate["zero_shot_evaluation"] = stage_cfg
    return evaluation_stage(candidate, manifest)


def _select_best_evaluated_checkpoint(summary_path: str | Path) -> Path:
    """Return the realized candidate with the lowest finite LM loss."""

    payload = json.loads(Path(summary_path).read_text())
    rows = payload.get("solutions", ()) if isinstance(payload, dict) else payload
    candidates = _select_evaluated_candidates(rows, num_best=1)
    if not candidates:
        raise RuntimeError(f"global KD found no finite evaluated candidates in {summary_path}")
    return Path(candidates[0]["checkpoint"])


def _write_global_distillation_summary(kd_config: GlobalKDConfig, result: GlobalKDResult) -> Path:
    """Publish the canonical final summary from the durable training log."""

    records_by_step: dict[int, dict[str, Any]] = {}
    training_log = kd_config.output_dir / "checkpoints" / "training.jsonl"
    if training_log.is_file():
        for line in training_log.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                step = int(record.get("step", record.get("global_step")))
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
            records_by_step[step] = record
    records = [records_by_step[step] for step in sorted(records_by_step)]
    consolidated = []
    for config_path in (kd_config.output_dir / "checkpoints").glob(
        "epoch_*_step_*/model/consolidated/config.json"
    ):
        checkpoint = config_path.parent
        step_dir = checkpoint.parents[1]
        if not (step_dir / "saving_completed").is_file():
            continue
        try:
            step = int(step_dir.name.rsplit("_step_", 1)[1])
        except (IndexError, ValueError):
            continue
        consolidated.append((step, checkpoint))
    post_kd_checkpoint = max(consolidated, key=lambda item: item[0])[1] if consolidated else None
    consolidation_requested = str(kd_config.save_consolidated).strip().lower() not in {
        "false",
        "0",
        "none",
    }
    if consolidation_requested and post_kd_checkpoint is None:
        raise RuntimeError(
            "global KD requested a consolidated export but published no durable checkpoint "
            f"under {kd_config.output_dir / 'checkpoints'}"
        )
    domain = kd_config.domain if kd_config.domain in {"llm", "vlm"} else "llm"
    dataset = dict((kd_config.metadata.get(domain) or {}).get("dataset") or {})
    output_dir = kd_config.output_dir
    profile_id = output_dir.parents[1].name if len(output_dir.parents) >= 2 else "default"
    payload = {
        "version": 1,
        "profile_id": profile_id,
        "run_id": output_dir.parent.name,
        "solution_id": output_dir.name,
        "kd_id": result.kd_id,
        "max_steps": kd_config.max_steps,
        "sequence_length": dataset.get("seq_length"),
        "sample_count": dataset.get("num_samples"),
        "global_batch_size": kd_config.global_batch_size,
        "local_batch_size": kd_config.local_batch_size,
        "records": records,
        "metrics": result.metrics,
    }
    if post_kd_checkpoint is not None:
        payload["post_kd_checkpoint"] = str(post_kd_checkpoint)
    summary_path = output_dir / "global_distillation_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = summary_path.with_name(f".{summary_path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, summary_path)
    return summary_path


def _promote_global_distillation_config(config: dict[str, Any]) -> dict[str, Any]:
    """Expose the public stage namespace to the shared Global KD builder."""

    candidate = copy.deepcopy(config)
    candidate["distillation"] = dict(
        candidate.get("global_distillation") or candidate.get("distillation") or {}
    )
    return candidate


def distillation_stage(config: dict[str, Any], manifest: StageManifest):
    distillation = dict(config.get("global_distillation") or {})
    if distillation.get("selection") == "best_evaluation":
        puzzle_dir = Path((config.get("experiment") or {})["dir"])
        summary_path = Path(
            distillation.get("evaluation_summary_path")
            or puzzle_dir / "artifacts" / "zero_shot_evaluation" / "evaluation_summary.json"
        )
        selected = _select_best_evaluated_checkpoint(summary_path)
        config = copy.deepcopy(config)
        distillation["student_dir"] = str(selected)
        distillation.setdefault(
            "output_dir",
            str(puzzle_dir / "artifacts" / "global_distillation" / selected.parent.name),
        )
        config["global_distillation"] = distillation
    config = _promote_global_distillation_config(config)
    if bool(distillation.get("scenario_grid", False)):
        import modelopt.torch.utils.distributed as dist

        results = []
        for candidate in _scenario_grid_kd_configs(config):
            kd_config = build_global_kd_config(candidate)
            result = run_global_kd(
                kd_config,
                recipe_runner=config.get("_global_kd_runner"),
            )
            results.append(result.to_dict())
        summary_path = (
            Path((config.get("experiment") or {})["dir"])
            / "artifacts"
            / "global_kd"
            / "summary.json"
        )
        if dist.is_master():
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
        dist.barrier()
        return complete_stage(
            config,
            manifest,
            outputs={"scenario_count": len(results), "summary_path": str(summary_path)},
        )
    tournament = dict((config.get("global_distillation") or {}).get("tournament") or {})
    if bool(tournament.get("enabled", False)):
        from ..distillation.tournament import run_global_kd_tournament
        from ..pipeline_config import load_runtime_hydra_config

        hydra_cfg = load_runtime_hydra_config(config)
        result = run_global_kd_tournament(
            config,
            hydra_cfg,
            recipe_runner=config.get("_global_kd_runner"),
        )
        return complete_stage(config, manifest, outputs=result)
    kd_config = build_global_kd_config(config)
    result = run_global_kd(kd_config, recipe_runner=config.get("_global_kd_runner"))
    summary_path = kd_config.output_dir / "global_distillation_summary.json"
    if int(os.environ.get("RANK", "0")) == 0:
        summary_path = _write_global_distillation_summary(kd_config, result)
    _distributed_barrier("global distillation publication")
    return complete_stage(
        config,
        manifest,
        outputs={**result.to_dict(), "summary_path": str(summary_path)},
    )


def _distillation_dataset_source(
    stage_cfg: dict[str, Any], config: dict[str, Any]
) -> tuple[str, str]:
    """Resolve one usable global-KD sanity dataset source."""

    dataset_path = str(
        stage_cfg.get("dataset_path")
        or (config.get("replacement_scoring") or {}).get("dataset_path")
        or (config.get("calibration") or {}).get("dataset_path")
        or ""
    )
    packed_token_cache_path = str(stage_cfg.get("packed_token_cache_path") or "")
    if not dataset_path and not packed_token_cache_path:
        raise ValueError("distillation overfit requires dataset_path or packed_token_cache_path")
    return dataset_path, packed_token_cache_path


def distillation_overfit_stage(config: dict[str, Any], manifest: StageManifest):
    """Replay one frozen minibatch while globally distilling selected solutions."""
    stage_cfg = dict(config.get("global_distillation_sanity") or {})
    if not bool(stage_cfg.get("enabled", False)):
        return complete_stage(
            config,
            manifest,
            outputs={"enabled": False},
            status="skipped",
            skip_reason=StageSkipReason.DISABLED,
            message="global_distillation_sanity.enabled is false",
        )

    puzzle_dir = Path((config.get("experiment") or {})["dir"])
    profile_id = str(stage_cfg.get("profile_id", "params-080"))
    registry_path = Path(
        stage_cfg.get("registry_path")
        or puzzle_dir / "mip" / "profiles" / profile_id / "selected_solutions.json"
    )
    registry = json.loads(registry_path.read_text())
    configured_ids = set(stage_cfg.get("solution_ids") or ())
    solutions = sorted(
        (
            row
            for row in registry.get("solutions", ())
            if row.get("solution_id") != "teacher"
            and (not configured_ids or row.get("solution_id") in configured_ids)
        ),
        key=lambda row: (str(row.get("solution_id", "")), str(row.get("checkpoint", ""))),
    )[: int((config.get("global_distillation") or {}).get("num_best_to_distill", 1))]
    if not solutions:
        raise ValueError(f"distillation overfit selected no candidates from {registry_path}")

    teacher = next(
        (row for row in registry.get("solutions", ()) if row.get("solution_id") == "teacher"),
        None,
    )
    if teacher is None:
        raise ValueError(f"solution registry has no teacher: {registry_path}")

    sample_count = int(stage_cfg.get("sample_count", 128))
    sequence_length = int(stage_cfg.get("sequence_length", 128))
    max_steps = int(stage_cfg.get("max_steps", 64))
    local_batch_size = int(stage_cfg.get("local_batch_size", 4))
    seed = int(stage_cfg.get("seed", 444))
    dataset_path, packed_token_cache_path = _distillation_dataset_source(stage_cfg, config)

    root = (
        puzzle_dir
        / "artifacts"
        / "global_distillation_sanity"
        / "profiles"
        / profile_id
        / (f"text-n{sample_count}-l{sequence_length}-s{max_steps}-b{local_batch_size}-seed{seed}")
    )
    summaries = []
    global_rank = int(os.environ.get("RANK", "0"))
    model_cfg = dict(config.get("model") or {})
    for solution in solutions:
        solution_id = str(solution["solution_id"])
        output_dir = root / solution_id
        result_path = output_dir / "global_distillation_sanity_result.json"
        if result_path.is_file():
            cached = json.loads(result_path.read_text())
            if len(cached.get("records") or ()) == max_steps:
                summaries.append(cached)
                continue

        candidate = copy.deepcopy(config)
        student_dir = Path(solution["checkpoint"])
        student_resolution = resolve_descriptor_from_pretrained(
            str(student_dir),
            trust_remote_code=bool(model_cfg.get("trust_remote_code", False)),
        )
        teacher_resolution = resolve_descriptor_from_pretrained(
            str(teacher["checkpoint"]),
            trust_remote_code=bool(model_cfg.get("trust_remote_code", False)),
        )
        dataset_config = {
            "_target_": (
                "modelopt.torch.puzzletron.distillation.dataset.make_puzzletron_llm_overfit_dataset"
            ),
            "dataset_path": dataset_path,
            "split": str(stage_cfg.get("dataset_split", "train")),
            "num_samples": sample_count,
            "seq_length": sequence_length,
            "seed": seed,
        }
        if packed_token_cache_path:
            dataset_config["packed_token_cache_path"] = packed_token_cache_path

        metadata = {
            "llm": {
                "dataset": dataset_config,
                "dataloader": {
                    "_target_": "torchdata.stateful_dataloader.StatefulDataLoader",
                    "collate_fn": (
                        "modelopt.torch.puzzletron.distillation.dataset."
                        "collate_puzzletron_llm_batch"
                    ),
                    "shuffle": False,
                    # AutoModel supplies local_batch_size automatically for map-style
                    # datasets, but iterable datasets must state it explicitly.
                    "batch_size": int(stage_cfg.get("local_batch_size", 4)),
                    "num_workers": 0,
                    "pin_memory": True,
                },
            },
            "recipe_overrides": {
                "step_scheduler": {
                    "ckpt_every_steps": max_steps,
                    "save_checkpoint_every_epoch": False,
                    "log_remote_every_steps": 1,
                    # Iterable datasets do not always expose an epoch length to
                    # AutoModel. Avoid its fallback of ten epochs and let the
                    # explicit max_steps terminate the replay run.
                    "num_epochs": max_steps,
                }
            },
        }
        kd = {
            "teacher_dir": str(teacher["checkpoint"]),
            "student_dir": str(student_dir),
            "output_dir": str(output_dir),
            "teacher_descriptor": teacher_resolution.name,
            "student_descriptor": student_resolution.name,
            "teacher_model_kwargs": dict(stage_cfg.get("teacher_model_kwargs") or {}),
            "student_model_kwargs": dict(stage_cfg.get("student_model_kwargs") or {}),
            "teacher_force_hf": False,
            "student_force_hf": False,
            "force_hf": False,
            "domain": "llm",
            "automodel": dict(stage_cfg.get("automodel") or {}),
            "global_batch_size": sample_count,
            "local_batch_size": local_batch_size,
            "max_steps": max_steps,
            "lr": float(stage_cfg.get("lr", 1.0e-4)),
            "weight_decay": 0.0,
            "seed": seed,
            "validation_enabled": False,
            "resume": True,
            "freeze_policy": "projector_and_language",
            "objective": {
                "main_ce": {"weight": float(stage_cfg.get("main_ce_weight", 1.0))},
                "main_kd": {"weight": float(stage_cfg.get("main_kd_weight", 1.0))},
                # MTP is opt-in: architectures such as Nemotron Nano have no
                # next-token prediction head, and enabling it would make every
                # otherwise-valid global-KD run fail at the first loss step.
                "mtp_ce": {"weight": float(stage_cfg.get("mtp_ce_weight", 0.0))},
                "mtp_kd": {"weight": float(stage_cfg.get("mtp_kd_weight", 0.0))},
            },
            "metadata": metadata,
        }
        candidate["distillation"] = kd
        result = run_global_kd(build_global_kd_config(candidate))
        training_log = output_dir / "checkpoints" / "training.jsonl"
        records = [
            json.loads(line) for line in training_log.read_text().splitlines() if line.strip()
        ]
        if len(records) != max_steps:
            raise RuntimeError(
                f"distillation overfit {solution_id} expected {max_steps} records, "
                f"found {len(records)} in {training_log}"
            )
        invalid_losses = [
            (index, record.get("loss"))
            for index, record in enumerate(records)
            if not isinstance(record.get("loss"), (int, float))
            or not math.isfinite(float(record["loss"]))
        ]
        if invalid_losses:
            raise RuntimeError(
                f"distillation overfit {solution_id} has missing or non-finite loss records: "
                f"{invalid_losses}"
            )
        from ..diagnostics.campaign_findings import loss_trend_findings

        findings = [
            asdict(finding)
            for finding in loss_trend_findings(
                stage="global_distillation_sanity",
                records=({**record, "solution_id": solution_id} for record in records),
                group_key="solution_id",
                window=int(stage_cfg.get("trend_window", 4)),
            )
        ]
        summary = {
            "solution_id": solution_id,
            "label": solution.get("label", solution_id),
            "color": solution.get("color", "#4f8cff"),
            "checkpoint": str(student_dir),
            "selection_reasons": _selection_reasons(solution),
            "teacher_checkpoint": str(teacher["checkpoint"]),
            "sample_count": sample_count,
            "sequence_length": sequence_length,
            "max_steps": max_steps,
            "dataset_path": dataset_path,
            "frozen_minibatch": True,
            "result": result.to_dict(),
            "records": records,
            "passed": not findings,
            "findings": findings,
            "verdict": "passed" if not findings else "warning",
        }
        if global_rank == 0:
            output_dir.mkdir(parents=True, exist_ok=True)
            temporary = result_path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
            temporary.replace(result_path)
        _distributed_barrier("global distillation sanity publication")
        summaries.append(summary)

    summary_path = root / "global_distillation_sanity_summary.json"
    findings = [
        finding
        for solution_summary in summaries
        for finding in solution_summary.get("findings", ())
    ]
    if global_rank == 0:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = summary_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(
                {
                    "profile_id": profile_id,
                    "sample_count": sample_count,
                    "sequence_length": sequence_length,
                    "max_steps": max_steps,
                    "frozen_minibatch": True,
                    "passed": not findings,
                    "findings": findings,
                    "verdict": "passed" if not findings else "warning",
                    "solutions": summaries,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        temporary.replace(summary_path)
    from ..diagnostics.sanity_verdict import SanityVerdict, complete_sanity_stage

    return complete_sanity_stage(
        config,
        manifest,
        outputs={
            "profile_id": profile_id,
            "solution_count": len(summaries),
            "summary_path": str(summary_path),
            "frozen_minibatch": True,
        },
        verdict=SanityVerdict(passed=not findings, findings=findings),
    )
