#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the profile-selected frozen-minibatch global-KD overfit stage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from modelopt.torch.puzzletron.stage_runner import run_stage


def _ensure_sanity_automodel_parallel(stage_cfg: dict) -> None:
    """Translate a persisted legacy sanity mesh when no stage-local mesh exists."""
    automodel = dict(stage_cfg.get("automodel") or {})
    if (automodel.get("parallel") or {}) or not any(
        key in stage_cfg for key in ("tp", "cp", "pp", "dp", "ep")
    ):
        return
    ep = int(stage_cfg.get("ep", 1) or 1)
    dp_shard = int(stage_cfg.get("dp_shard", stage_cfg.get("dp", 1)) or 1)
    if "dp_shard" not in stage_cfg:
        dp_shard *= ep
    automodel["parallel"] = {
        "tp": int(stage_cfg.get("tp", 1) or 1),
        "cp": int(stage_cfg.get("cp", 1) or 1),
        "pp": int(stage_cfg.get("pp", 1) or 1),
        "ep": ep,
        "dp_shard": dp_shard,
        "dp_replicate": int(stage_cfg.get("dp_replicate", 1) or 1),
        "sequence_parallel": bool(stage_cfg.get("sequence_parallel", False)),
        "pipeline_schedule": str(stage_cfg.get("pipeline_schedule", "1f1b")),
    }
    stage_cfg["automodel"] = automodel


def _apply_sanity_automodel_parallel_overrides(stage_cfg: dict, args: argparse.Namespace) -> None:
    """Apply explicitly requested stage-local mesh dimensions."""
    overrides = {
        "tp": args.tp,
        "cp": args.cp,
        "pp": args.pp,
        "ep": args.ep,
        "dp_shard": args.dp_shard,
        "dp_replicate": args.dp_replicate,
    }
    if not any(value is not None for value in overrides.values()):
        return
    _ensure_sanity_automodel_parallel(stage_cfg)
    automodel = dict(stage_cfg.get("automodel") or {})
    parallel = dict(automodel.get("parallel") or {})
    parallel.update({key: value for key, value in overrides.items() if value is not None})
    automodel["parallel"] = parallel
    stage_cfg["automodel"] = automodel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--puzzle-dir", type=Path, required=True)
    parser.add_argument("--profile-id", default="params-080")
    parser.add_argument("--sample-count", type=int, default=128)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--max-steps", type=int, default=64)
    parser.add_argument("--local-batch-size", type=int, default=4)
    parser.add_argument("--solution-id", action="append", default=[])
    parser.add_argument("--registry-path", type=Path)
    parser.add_argument("--student-torch-dtype")
    parser.add_argument("--teacher-torch-dtype")
    parser.add_argument("--tp", type=int)
    parser.add_argument("--cp", type=int)
    parser.add_argument("--pp", type=int)
    parser.add_argument("--ep", type=int)
    parser.add_argument("--dp-shard", type=int)
    parser.add_argument("--dp-replicate", type=int)
    args = parser.parse_args()

    manifest_path = args.puzzle_dir / "manifests" / "build_library.json"
    manifest = json.loads(manifest_path.read_text())
    config = dict(manifest.get("config") or (manifest.get("inputs") or {}).get("config") or {})
    if not config:
        raise ValueError(f"manifest contains no resolved config: {manifest_path}")
    config.pop("parallel", None)
    stage_cfg = dict(config.get("global_distillation_sanity") or {})
    stage_cfg.update(
        {
            "enabled": True,
            "profile_id": args.profile_id,
            "sample_count": args.sample_count,
            "sequence_length": args.sequence_length,
            "max_steps": args.max_steps,
            "local_batch_size": args.local_batch_size,
        }
    )
    if args.solution_id:
        stage_cfg["solution_ids"] = args.solution_id
    if args.registry_path is not None:
        stage_cfg["registry_path"] = str(args.registry_path)
    if args.student_torch_dtype:
        student_model_kwargs = dict(stage_cfg.get("student_model_kwargs") or {})
        student_model_kwargs["torch_dtype"] = args.student_torch_dtype
        stage_cfg["student_model_kwargs"] = student_model_kwargs
    if args.teacher_torch_dtype:
        teacher_model_kwargs = dict(stage_cfg.get("teacher_model_kwargs") or {})
        teacher_model_kwargs["torch_dtype"] = args.teacher_torch_dtype
        stage_cfg["teacher_model_kwargs"] = teacher_model_kwargs
    if not stage_cfg.get("dataset_path") and not stage_cfg.get("packed_token_cache_path"):
        dataset_path = (config.get("scoring") or {}).get("dataset_path") or config.get(
            "dataset_path"
        )
        if dataset_path:
            stage_cfg["dataset_path"] = str(dataset_path)
    _ensure_sanity_automodel_parallel(stage_cfg)
    _apply_sanity_automodel_parallel_overrides(stage_cfg, args)
    config["global_distillation_sanity"] = stage_cfg
    result = run_stage(config, "global_distillation_sanity")
    if int(__import__("os").environ.get("RANK", "0")) == 0:
        print(result.manifest_path)


if __name__ == "__main__":
    main()
