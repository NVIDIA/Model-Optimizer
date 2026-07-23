#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run several one-GPU vLLM runtime-stats shards on one multi-GPU node."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from modelopt.torch.puzzletron.pipeline_config import pipeline_config_from_path
from modelopt.torch.puzzletron.stages.pipeline import finalize_vllm_stats_report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--shard-indices",
        required=True,
        help="Comma-separated logical shard indices to run on this node.",
    )
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()

    shard_indices = [int(part.strip()) for part in args.shard_indices.split(",") if part.strip()]
    if not shard_indices:
        raise SystemExit("no shard indices provided")
    if len(shard_indices) != len(set(shard_indices)):
        raise SystemExit(f"duplicate shard indices: {shard_indices}")

    repo = Path(__file__).resolve().parents[2]
    worker = Path(__file__).resolve().parent / "run_runtime_stats_shard.py"
    processes: list[subprocess.Popen[str]] = []
    for local_gpu, shard_index in enumerate(shard_indices):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(local_gpu)
        env["SLURM_LOCALID"] = str(local_gpu)
        env["SLURM_PROCID"] = str(shard_index)
        env["SLURM_NTASKS"] = str(args.shard_count)
        env["PUZZLETRON_RUNTIME_SHARD_INDEX"] = str(shard_index)
        env["PUZZLETRON_RUNTIME_SHARD_COUNT"] = str(args.shard_count)
        argv = [sys.executable, str(worker), "--config", args.config]
        for override in args.override:
            argv.extend(["--override", override])
        processes.append(
            subprocess.Popen(
                argv,
                cwd=str(repo),
                env=env,
            )
        )

    exit_code = 0
    for process in processes:
        code = process.wait()
        if code:
            exit_code = code
    if exit_code == 0 and 0 in shard_indices:
        config = pipeline_config_from_path(args.config, overrides=args.override)
        finalize_vllm_stats_report(config)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
