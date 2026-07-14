#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the profile-selected frozen-minibatch global-KD overfit stage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from modelopt.torch.puzzletron.stage_runner import run_stage


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
    config["distillation_overfit"] = {
        "enabled": True,
        "profile_id": args.profile_id,
        "solution_ids": args.solution_id,
        "sample_count": args.sample_count,
        "sequence_length": args.sequence_length,
        "max_steps": args.max_steps,
        "local_batch_size": args.local_batch_size,
        "tp": int(args.tp),
        "cp": int(args.cp),
        "pp": int(args.pp),
        "dp": int(args.dp),
        "ep": int(args.ep),
        "sequence_parallel": True,
        "lr": 1.0e-4,
        "dataset_path": str(
            (config.get("scoring") or {}).get("dataset_path")
            or config.get("dataset_path")
        ),
    }
    if args.registry_path is not None:
        config["distillation_overfit"]["registry_path"] = str(args.registry_path)
    result = run_stage(config, "distillation_overfit")
    if int(__import__("os").environ.get("RANK", "0")) == 0:
        print(result.manifest_path)


if __name__ == "__main__":
    main()
