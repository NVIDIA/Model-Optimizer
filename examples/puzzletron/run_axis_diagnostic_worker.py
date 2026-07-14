#!/usr/bin/env python3
"""Run or finalize one independently distributed width-diagnostic axis."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

from omegaconf import OmegaConf

from modelopt.torch.puzzletron.diagnostics.campaign_progress_report import (
    generate_campaign_progress_report,
)
from modelopt.torch.puzzletron.diagnostics.campaign_findings import MetricSpec
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
    enabled = [
        str(axis)
        for axis, axis_cfg in search_axes.items()
        if isinstance(axis_cfg, dict) and bool(axis_cfg.get("enabled", False))
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

    return _ratio_aligned_hidden_widths(full, (7 / 8, 1 / 4), alignment=alignment)


def _worker_config(config: dict, axis: str, config_path: Path) -> dict:
    puzzle_dir = _puzzle_dir(config).resolve()
    safe = _safe_axis(axis)
    config["experiment"] = dict(config.get("experiment") or {})
    config["experiment"]["dir"] = str(puzzle_dir / ".axis_workers" / safe)
    clean_root = Path(config.get("clean_config_root") or config_path.resolve().parents[3])
    recipe_path = str(clean_root / "recipes" / "automodel_pp2_dp1.yaml")
    config["recipe_path"] = recipe_path
    runtime = dict(config.get("_runtime") or {})
    runtime_overrides = list(runtime.get("overrides") or ())
    runtime_overrides.extend(
        (
            f"recipe_path={recipe_path}",
            f"++width_sanity.automodel.recipe_path={recipe_path}",
        )
    )
    diagnostic = dict(config.get("width_sanity") or {})
    diagnostic_automodel = dict(diagnostic.get("automodel") or {})
    diagnostic_automodel["recipe_path"] = recipe_path
    diagnostic.update(
        {
            "enabled": True,
            "single_load_parent_sweep": True,
            "axes": [axis],
            "experiment_id": f"axis_{safe}",
            "one_case_per_axis": False,
            "target_count_per_axis": 2,
            "layer_count": 2,
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
    recipe_path = ((diagnostic.get("automodel") or {}).get("recipe_path"))
    if not recipe_path:
        raise ValueError("axis diagnostic worker requires width_sanity.automodel.recipe_path")
    recipe = OmegaConf.to_container(OmegaConf.load(recipe_path), resolve=True)
    distributed = recipe.get("distributed") or {}
    sizes = {
        name: int(distributed.get(f"{name}_size", 1) or 1)
        for name in ("tp", "cp", "pp", "ep", "dp")
    }
    expected = sizes["tp"] * sizes["cp"] * sizes["pp"] * sizes["ep"] * sizes["dp"]
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if expected != world_size or sizes != {"tp": 1, "cp": 1, "pp": 2, "ep": 1, "dp": 1}:
        raise ValueError(
            "axis diagnostic topology must be PP=2 with TP=CP=EP=DP=1: "
            f"recipe={recipe_path} sizes={sizes} world_size={world_size}"
        )
    hydra_config = load_runtime_hydra_config(config)
    if str(hydra_config.recipe_path) != str(recipe_path):
        raise ValueError(
            "axis diagnostic recipe did not survive Hydra reconstruction: "
            f"expected={recipe_path} actual={hydra_config.recipe_path}"
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
    parallel_execution = {
        "workers": len(axes),
        "gpus_per_worker": 2,
        "pp": 2,
        "dp": 1,
        "cp": 1,
        "tp": 1,
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
