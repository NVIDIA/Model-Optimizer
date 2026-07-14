#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run proper profile-selected global KD on fresh data rather than a replayed batch."""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
from typing import Any

from modelopt.torch.puzzletron.anymodel.registry import resolve_descriptor_from_pretrained
from modelopt.torch.puzzletron.stage_runner import run_stage


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def build_profile_distillation_config(
    config: dict[str, Any],
    *,
    teacher_dir: Path,
    student_dir: Path,
    output_dir: Path,
    descriptor: str,
    sequence_length: int,
    global_batch_size: int,
    local_batch_size: int,
    max_steps: int,
    learning_rate: float,
    seed: int,
    checkpoint_every: int,
    tp: int = 2,
    cp: int = 2,
    pp: int = 2,
    dp: int = 2,
    ep: int = 1,
) -> dict[str, Any]:
    """Build the canonical native-AutoModel text+MTP distillation config."""
    candidate = copy.deepcopy(config)
    dataset_path = str(
        (candidate.get("scoring") or {}).get("dataset_path")
        or (candidate.get("calibration") or {}).get("dataset_path")
        or ""
    )
    if not dataset_path:
        raise ValueError("profile distillation requires scoring.dataset_path")
    sample_count = int(global_batch_size) * int(max_steps)
    candidate["distillation"] = {
        "teacher_dir": str(teacher_dir),
        "student_dir": str(student_dir),
        "output_dir": str(output_dir),
        "teacher_descriptor": descriptor,
        "student_descriptor": descriptor,
        "teacher_force_hf": False,
        "student_force_hf": False,
        "force_hf": False,
        "domain": "llm",
        "tp": int(tp),
        "cp": int(cp),
        "pp": int(pp),
        "dp": int(dp),
        "ep": int(ep),
        "sequence_parallel": True,
        "global_batch_size": int(global_batch_size),
        "local_batch_size": int(local_batch_size),
        "max_steps": int(max_steps),
        "lr": float(learning_rate),
        "weight_decay": 0.0,
        "seed": int(seed),
        "validation_enabled": False,
        "resume": True,
        "freeze_policy": "projector_and_language",
        "objective": {
            "main_ce": {"weight": 1.0},
            "main_kd": {"weight": 1.0, "metric": "kld", "chunk_size": 128},
            "mtp_ce": {"weight": 1.0},
            "mtp_kd": {"weight": 1.0, "metric": "kld", "chunk_size": 128},
        },
        "metadata": {
            "llm": {
                "dataset": {
                    "_target_": (
                        "modelopt.torch.puzzletron.distillation.dataset."
                        "make_puzzletron_llm_dataset"
                    ),
                    "dataset_path": dataset_path,
                    "split": "train",
                    "num_samples": sample_count,
                    "seq_length": int(sequence_length),
                    "seed": int(seed),
                },
                "dataloader": {
                    "_target_": "torchdata.stateful_dataloader.StatefulDataLoader",
                    "collate_fn": (
                        "modelopt.torch.puzzletron.distillation.dataset."
                        "collate_puzzletron_llm_batch"
                    ),
                    "shuffle": False,
                    "batch_size": int(local_batch_size),
                    "num_workers": 0,
                    "pin_memory": True,
                },
            },
            "recipe_overrides": {
                "step_scheduler": {
                    "ckpt_every_steps": int(checkpoint_every),
                    "save_checkpoint_every_epoch": False,
                    "log_remote_every_steps": 1,
                    "num_epochs": int(max_steps),
                }
            },
        },
    }
    return candidate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--puzzle-dir", type=Path, required=True)
    parser.add_argument("--profile-id", default="params-080")
    parser.add_argument("--solution-id", default="h1024-d0")
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--global-batch-size", type=int, default=32)
    parser.add_argument("--local-batch-size", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=5.0e-5)
    parser.add_argument("--seed", type=int, default=445)
    parser.add_argument("--checkpoint-every", type=int, default=64)
    parser.add_argument("--tp", type=int, default=2)
    parser.add_argument("--cp", type=int, default=2)
    parser.add_argument("--pp", type=int, default=2)
    parser.add_argument("--dp", type=int, default=2)
    parser.add_argument("--ep", type=int, default=1)
    args = parser.parse_args()

    manifest_path = args.puzzle_dir / "manifests" / "build_library.json"
    manifest = json.loads(manifest_path.read_text())
    config = dict(manifest.get("config") or (manifest.get("inputs") or {}).get("config") or {})
    if not config:
        raise ValueError(f"manifest contains no resolved config: {manifest_path}")
    registry_path = (
        args.puzzle_dir
        / "mip"
        / "profiles"
        / args.profile_id
        / "selected_solutions.json"
    )
    registry = json.loads(registry_path.read_text())
    solutions = {row["solution_id"]: row for row in registry["solutions"]}
    if args.solution_id not in solutions or "teacher" not in solutions:
        raise ValueError(
            f"registry must contain teacher and {args.solution_id!r}: {registry_path}"
        )
    student = solutions[args.solution_id]
    teacher = solutions["teacher"]
    model_cfg = dict(config.get("model") or {})
    resolution = resolve_descriptor_from_pretrained(
        student["checkpoint"],
        trust_remote_code=bool(model_cfg.get("trust_remote_code", False)),
        descriptor_override=model_cfg.get("descriptor_override"),
    )
    run_id = (
        f"text-l{args.sequence_length}-g{args.global_batch_size}"
        f"-b{args.local_batch_size}-s{args.max_steps}-seed{args.seed}"
    )
    output_dir = (
        args.puzzle_dir
        / "artifacts"
        / "distillation"
        / "profiles"
        / args.profile_id
        / run_id
        / args.solution_id
    )
    candidate = build_profile_distillation_config(
        config,
        teacher_dir=Path(teacher["checkpoint"]),
        student_dir=Path(student["checkpoint"]),
        output_dir=output_dir,
        descriptor=resolution.name,
        sequence_length=args.sequence_length,
        global_batch_size=args.global_batch_size,
        local_batch_size=args.local_batch_size,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        seed=args.seed,
        checkpoint_every=args.checkpoint_every,
        tp=args.tp,
        cp=args.cp,
        pp=args.pp,
        dp=args.dp,
        ep=args.ep,
    )
    result = run_stage(candidate, "distillation")
    if int(os.environ.get("RANK", "0")) == 0:
        training_path = output_dir / "checkpoints" / "training.jsonl"
        records = [
            json.loads(line)
            for line in training_path.read_text().splitlines()
            if line.strip()
        ]
        checkpoint_dirs = sorted(
            (output_dir / "checkpoints").glob("epoch_*_step_*"),
            key=lambda path: int(path.name.rsplit("_", 1)[-1]),
        )
        if not checkpoint_dirs:
            raise RuntimeError(f"distillation produced no checkpoint under {output_dir}")
        distilled_checkpoint = checkpoint_dirs[-1] / "model" / "consolidated"
        if not (distilled_checkpoint / "config.json").is_file():
            raise RuntimeError(
                f"final distillation checkpoint is not consolidated: {distilled_checkpoint}"
            )
        summary = {
            "version": 1,
            "profile_id": args.profile_id,
            "solution_id": args.solution_id,
            "label": student.get("label", args.solution_id),
            "color": student.get("color", "#4f8cff"),
            "teacher_checkpoint": teacher["checkpoint"],
            "student_checkpoint": student["checkpoint"],
            "distilled_checkpoint": str(distilled_checkpoint),
            "sequence_length": args.sequence_length,
            "global_batch_size": args.global_batch_size,
            "local_batch_size": args.local_batch_size,
            "max_steps": args.max_steps,
            "learning_rate": args.learning_rate,
            "seed": args.seed,
            "records": records,
            "manifest": str(result.manifest_path),
        }
        _atomic_json(output_dir / "distillation_summary.json", summary)
        print(output_dir / "distillation_summary.json")


if __name__ == "__main__":
    main()
