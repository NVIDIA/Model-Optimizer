#!/usr/bin/env python3
"""Run or finalize one independently distributed width-diagnostic axis."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

from modelopt.torch.puzzletron.diagnostics.campaign_findings import MetricSpec
from modelopt.torch.puzzletron.diagnostics.campaign_progress_report import (
    generate_campaign_progress_report,
)
from modelopt.torch.puzzletron.diagnostics.width_sanity import aggregate_width_sanity
from modelopt.torch.puzzletron.manifest import StageManifest, write_stage_manifest
from modelopt.torch.puzzletron.pipeline_config import (
    load_runtime_hydra_config,
    pipeline_config_from_path,
)
from modelopt.torch.puzzletron.stage_runner import run_stage
from modelopt.torch.puzzletron.stages.diagnostics import _PRIMARY_METRICS
from modelopt.torch.puzzletron.tools.checkpoint_utils import load_model_config


def _axes(config: dict) -> list[str]:
    search_axes = (config.get("search_space") or {}).get("axes") or {}
    non_sortable = set(
        str(axis)
        for axis in (config.get("width_sanity") or {}).get("non_sortable_axes", ())
    )
    enabled = [
        str(axis)
        for axis, axis_cfg in search_axes.items()
        if (
            isinstance(axis_cfg, dict)
            and bool(axis_cfg.get("enabled", False))
            and str(axis) not in non_sortable
        )
    ]
    return [*enabled, "hidden_width"]


def _safe_axis(axis: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", axis)


def _puzzle_dir(config: dict) -> Path:
    return Path(config.get("puzzle_dir") or (config.get("experiment") or {})["dir"])


def _hidden_widths(config: dict) -> list[int]:
    teacher = Path(
        config.get("teacher_dir")
        or (config.get("convert") or {}).get("teacher_dir")
        or _puzzle_dir(config) / "ckpts" / "teacher"
    )
    checkpoint_config = load_model_config(teacher, trust_remote_code=True)
    language = getattr(checkpoint_config, "text_config", checkpoint_config)
    full = int(language.hidden_size)
    alignment = int((config.get("embedding_pruning") or {}).get("alignment", 1) or 1)
    from modelopt.torch.puzzletron.stages.diagnostics import _ratio_aligned_hidden_widths

    return _ratio_aligned_hidden_widths(full, (7 / 8, 3 / 4), alignment=alignment)


def _worker_config(config: dict, axis: str, config_path: Path) -> dict:
    puzzle_dir = _puzzle_dir(config).resolve()
    safe = _safe_axis(axis)
    config["experiment"] = dict(config.get("experiment") or {})
    config["experiment"]["dir"] = str(puzzle_dir / ".axis_workers" / safe)
    runtime = dict(config.get("_runtime") or {})
    runtime_overrides = list(runtime.get("overrides") or ())
    diagnostic = dict(config.get("width_sanity") or {})
    diagnostic_automodel = dict(diagnostic.get("automodel") or {})
    diagnostic.update(
        {
            "enabled": True,
            "single_load_parent_sweep": True,
            "axes": [axis],
            "experiment_id": f"axis_{safe}",
            "one_case_per_axis": False,
            "target_count_per_axis": int(diagnostic.get("target_count_per_axis", 2)),
            "layer_count": int(diagnostic.get("layer_count", 3)),
            "layer_selection": "random",
            "ratios": [0.875, 0.75],
            "overwrite": False,
            "force_rescore": False,
            "cleanup_reverse_on_success": False,
            "cleanup_physical_checkpoints": True,
            "reuse_sort_equivalence": True,
            "reverse_checkpoint_dir": str(puzzle_dir / "ckpts" / "reverse_sorted_teacher"),
            "reverse_activation_logs_dir": str(
                puzzle_dir / "pruning" / "pruning_scores" / "automodel" / "reverse_all_axes"
            ),
            "automodel": diagnostic_automodel,
        }
    )
    if axis == "hidden_width":
        widths = _hidden_widths(config)
        embedding = dict(config.get("embedding_pruning") or {})
        embedding.update(enabled=True, widths=widths)
        config["embedding_pruning"] = embedding
        runtime_overrides.extend(
            (
                "embedding_pruning.enabled=true",
                f"embedding_pruning.widths={widths}",
            )
        )
        diagnostic.update(
            hidden_width_diagnostic=True,
            hidden_width_targets=widths,
            physical_realization=False,
        )
    else:
        diagnostic.update(hidden_width_diagnostic=False, physical_realization=True)
    config["width_sanity"] = diagnostic
    runtime["overrides"] = runtime_overrides
    config["_runtime"] = runtime
    return config


def _validate_worker_topology(config: dict, axis: str) -> None:
    diagnostic = config.get("width_sanity") or {}
    parallel = dict((diagnostic.get("automodel") or {}).get("parallel") or {})
    if not parallel:
        raise ValueError("axis diagnostic worker requires width_sanity.automodel.parallel")
    sizes = {
        name: int(parallel.get(name, 1) or 1)
        for name in ("tp", "cp", "pp", "ep", "dp_shard", "dp_replicate")
    }
    invalid = {name: value for name, value in sizes.items() if value < 1}
    if invalid:
        raise ValueError(f"axis diagnostic parallel axes must be positive integers: {invalid}")
    if sizes["dp_shard"] % sizes["ep"]:
        raise ValueError(
            "axis diagnostic dp_shard must be divisible by ep because EP is overlaid "
            f"on FSDP shards: parallel={parallel}"
        )
    expected = (
        sizes["tp"]
        * sizes["cp"]
        * sizes["pp"]
        * sizes["dp_shard"]
        * sizes["dp_replicate"]
    )
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if expected != world_size:
        raise ValueError(
            "axis diagnostic worker must use exactly the stage-owned "
            "width_sanity.automodel.parallel mesh: "
            f"parallel={parallel} expected_world_size={expected} world_size={world_size}"
        )
    hydra_config = load_runtime_hydra_config(config)
    runtime_parallel = dict(hydra_config.width_sanity.automodel.parallel)
    runtime_sizes = {
        name: int(runtime_parallel.get(name, 1) or 1)
        for name in ("tp", "cp", "pp", "ep", "dp_shard", "dp_replicate")
    }
    if runtime_sizes != sizes:
        raise ValueError(
            "axis diagnostic parallelism did not survive Hydra reconstruction: "
            f"expected={parallel} actual={runtime_parallel}"
        )
    if axis == "hidden_width" and not bool(hydra_config.embedding_pruning.enabled):
        raise ValueError("hidden-width diagnostic is disabled after Hydra reconstruction")


def _run(config_path: Path, axis_index: int) -> None:
    config = pipeline_config_from_path(config_path)
    axes = _axes(config)
    if not 0 <= axis_index < len(axes):
        raise ValueError(f"axis index {axis_index} is outside 0..{len(axes) - 1}")
    axis = axes[axis_index]
    config = _worker_config(config, axis, config_path)
    manifest_path = Path(config["experiment"]["dir"]) / "manifests" / "width_sanity.json"
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text())
        if existing.get("status") == "success":
            print(f"axis diagnostic already complete: axis={axis} manifest={manifest_path}")
            return
    _validate_worker_topology(config, axis)
    result = run_stage(config, "width_sanity")
    if result.status != "success":
        raise RuntimeError(f"axis {axis} finished with status {result.status}")


def _finalize(config_path: Path) -> None:
    config = pipeline_config_from_path(config_path)
    puzzle_dir = _puzzle_dir(config).resolve()
    axes = _axes(config)
    axis_summaries = {}
    worker_manifests = {}
    for axis in axes:
        safe = _safe_axis(axis)
        manifest_path = (
            puzzle_dir / ".axis_workers" / safe / "manifests" / "width_sanity.json"
        )
        artifact_dir = puzzle_dir / "artifacts" / f"activation_diagnostic_axis_{safe}"
        summary_path = artifact_dir / "activation_diagnostic_summary.json"
        if manifest_path.is_file():
            payload = json.loads(manifest_path.read_text())
            if payload.get("status") != "success":
                raise RuntimeError(f"axis worker is incomplete: axis={axis} manifest={payload}")
            worker_manifests[axis] = str(manifest_path)
        if axis == "hidden_width":
            hidden_path = artifact_dir / "hidden_width_diagnostic_summary.json"
            if summary_path.is_file():
                summary = json.loads(summary_path.read_text())
                axis_summaries[axis] = summary.get("hidden_width") or json.loads(
                    hidden_path.read_text()
                )
            elif hidden_path.is_file():
                axis_summaries[axis] = json.loads(hidden_path.read_text())
            else:
                raise RuntimeError(f"hidden-width summary is missing: {hidden_path}")
        else:
            if not summary_path.is_file():
                raise RuntimeError(f"axis summary is missing: axis={axis} path={summary_path}")
            summary = json.loads(summary_path.read_text())
            axis_summaries[axis] = summary

    diagnostic = config.get("width_sanity") or {}
    tolerance = float(diagnostic.get("comparison_tolerance", 0.0))
    metric_specs = {
        metric: MetricSpec(
            name=metric,
            direction="higher" if metric.startswith("token_accuracy_") else "lower",
            abs_tolerance=tolerance,
        )
        for metric in _PRIMARY_METRICS
    }
    width_summary, slicing_summary = aggregate_width_sanity(
        axis_summaries, metric_specs=metric_specs
    )
    parallel = dict(
        ((config.get("width_sanity") or {}).get("automodel") or {}).get("parallel") or {}
    )
    sizes = {
        name: int(parallel.get(name, 1) or 1)
        for name in ("tp", "cp", "pp", "ep", "dp_shard", "dp_replicate")
    }
    parallel_execution = {
        "workers": len(axes),
        "gpus_per_worker": (
            sizes["tp"]
            * sizes["cp"]
            * sizes["pp"]
            * sizes["dp_shard"]
            * sizes["dp_replicate"]
        ),
        **sizes,
    }
    for stage, summary in (
        ("width_sanity", width_summary),
        ("slicing_sanity", slicing_summary),
    ):
        summary.update(
            {
                "axis_summaries": axis_summaries,
                "parallel_execution": parallel_execution,
            }
        )
        artifacts_dir = puzzle_dir / "artifacts" / stage
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        summary_path = artifacts_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        manifest = StageManifest(
            stage=stage,
            inputs={"config": config, "worker_manifests": worker_manifests},
            config=config,
        )
        manifest.complete(
            outputs={
                "summary_path": str(summary_path),
                "axes": axes,
                "worker_manifests": worker_manifests,
            }
        )
        write_stage_manifest(puzzle_dir / "manifests" / f"{stage}.json", manifest)
    model = config.get("model") or {}
    generate_campaign_progress_report(
        puzzle_dir,
        model_name=str(config.get("display_name") or model.get("source") or "Puzzletron model"),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--axis-index", type=int)
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args()
    if args.finalize:
        if int(os.environ.get("RANK", "0")) == 0:
            _finalize(args.config)
        return
    if args.axis_index is None:
        parser.error("--axis-index is required unless --finalize is set")
    _run(args.config, args.axis_index)


if __name__ == "__main__":
    main()
