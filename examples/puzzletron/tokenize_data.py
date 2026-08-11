# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optionally materialize fixed-token caches ahead of Puzzletron stages."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from modelopt.torch.puzzletron.manifest import StageManifest, write_stage_manifest
from modelopt.torch.puzzletron.stage_runner import StageResult
from modelopt.torch.puzzletron.stages.graph import StageSkipReason, stage_is_enabled

__all__ = ["tokenize_data_stage"]


def tokenize_data_stage(config: dict) -> StageResult:
    """Materialize every configured fixed-token cache ahead of its consumers."""

    stage_config = config.get("tokenize_data") or {}
    puzzle_dir = Path(config.get("puzzle_dir") or (config.get("experiment") or {})["dir"])
    manifest_path = puzzle_dir / "manifests" / "tokenize_data.json"
    manifest = StageManifest(stage="tokenize_data", inputs={"config": config}, config=config)
    if not stage_is_enabled("tokenize_data", config):
        skip_reason = StageSkipReason.DISABLED
        manifest.complete(
            outputs={"enabled": False},
            status="skipped",
            skip_reason=skip_reason,
        )
        write_stage_manifest(manifest_path, manifest)
        return StageResult(
            "tokenize_data",
            "skipped",
            manifest_path,
            "Ahead-of-time fixed-token cache materialization is disabled.",
            skip_reason.value,
        )

    tool = Path(__file__).resolve().parent / "tools" / "build_packed_token_memmap.py"
    teacher_dir = Path((config.get("convert") or {})["teacher_dir"])
    dataset_path = str(config["dataset_path"])
    outputs = []
    for cache in stage_config.get("caches") or ():
        output = Path(cache["output"])
        command = (
            sys.executable,
            str(tool),
            "--dataset-path",
            dataset_path,
            "--tokenizer-path",
            str(teacher_dir),
            "--output",
            str(output),
            "--split",
            str(cache["split"]),
            "--content-field",
            str(stage_config.get("content_field", "messages")),
            "--num-samples",
            str(cache["num_samples"]),
            "--seq-length",
            str(cache["seq_length"]),
            "--workers",
            str(stage_config.get("workers", 64)),
            "--tokenize-batch-size",
            str(stage_config.get("tokenize_batch_size", 64)),
            "--shuffle-seed",
            str(cache["shuffle_seed"]),
            "--trust-remote-code",
        )
        subprocess.run(command, check=True)
        outputs.append(
            {
                "path": str(output),
                "metadata": str(output.with_suffix(output.suffix + ".json")),
                "split": str(cache["split"]),
            }
        )

    manifest.complete(outputs={"caches": outputs})
    write_stage_manifest(manifest_path, manifest)
    return StageResult(
        "tokenize_data",
        "success",
        manifest_path,
        f"Prepared {len(outputs)} token caches.",
    )
