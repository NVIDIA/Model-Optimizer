#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run one shared-cache shard of Puzzletron vLLM runtime statistics."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from modelopt.torch.puzzletron.anymodel.registry import resolve_descriptor_from_pretrained
from modelopt.torch.puzzletron.pipeline_config import (
    load_runtime_hydra_config,
    pipeline_config_from_path,
)
from modelopt.torch.puzzletron.stages.common import experiment_dir
from modelopt.torch.puzzletron.stages.pipeline import (
    finalize_vllm_stats_report,
    prepare_vllm_width_checkpoints,
)
from modelopt.torch.puzzletron.subblock_stats.calc_subblock_stats import launch_calc_subblock_stats


def _inject_runtime_descriptor(config: dict) -> None:
    """Mirror stage_runner's descriptor inference for this standalone worker."""

    model = config.get("model") or {}
    source = (
        (config.get("build_replacement_library") or {}).get("source_checkpoint_dir")
        or config.get("teacher_dir")
        or model.get("source")
    )
    if source is None:
        raise ValueError(
            "runtime stats needs a checkpoint or model source for descriptor inference"
        )
    resolution = resolve_descriptor_from_pretrained(
        str(Path(str(source)).resolve()) if Path(str(source)).exists() else str(source),
        trust_remote_code=bool(model.get("trust_remote_code", False)),
        descriptor_override=model.get("descriptor_override"),
    )
    runtime = config.setdefault("_runtime", {})
    runtime.update(
        {
            "descriptor": resolution.name,
            "descriptor_reason": resolution.reason,
            "descriptor_confidence": resolution.confidence,
        }
    )


def _require_runtime_subblock_library(config: dict) -> Path:
    """Fail fast unless convert already emitted the runtime candidate library."""

    path = experiment_dir(config) / "subblock_library.json"
    if path.is_file() and path.stat().st_size > 0:
        return path
    raise FileNotFoundError(
        f"missing runtime subblock library {path}; convert must emit it when vllm_stats is enabled"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()
    shard_index = os.environ.get(
        "PUZZLETRON_RUNTIME_SHARD_INDEX",
        os.environ.get("PUZZLETRON_GROUP_INDEX", os.environ.get("SLURM_PROCID", "0")),
    )
    shard_count = os.environ.get(
        "PUZZLETRON_RUNTIME_SHARD_COUNT", os.environ.get("SLURM_NTASKS", "1")
    )
    os.environ["PUZZLETRON_RUNTIME_SHARD_INDEX"] = shard_index
    os.environ["PUZZLETRON_RUNTIME_SHARD_COUNT"] = shard_count
    plain = pipeline_config_from_path(args.config, overrides=args.override)
    _inject_runtime_descriptor(plain)
    _require_runtime_subblock_library(plain)
    cfg = load_runtime_hydra_config(plain)
    prepare_vllm_width_checkpoints(plain, cfg)
    launch_calc_subblock_stats(cfg)
    if int(shard_index) == 0:
        finalize_vllm_stats_report(plain)


if __name__ == "__main__":
    main()
