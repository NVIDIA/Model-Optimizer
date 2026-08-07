# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tokenize the fixed data caches consumed by Puzzletron stages."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

from modelopt.torch.puzzletron.manifest import StageManifest, write_stage_manifest
from modelopt.torch.puzzletron.stage_runner import StageResult

__all__ = ["resolve_tokenize_caches", "tokenize_data_stage"]


def resolve_tokenize_caches(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return explicit tokenize caches, or defaults from campaign token paths."""

    stage_config = config.get("tokenize_data") or {}
    caches = [dict(cache) for cache in stage_config.get("caches") or ()]
    if caches:
        return caches

    data_cfg = config.get("data") or {}
    calibration = data_cfg.get("calibration") or {}
    train_samples = int(calibration.get("num_samples") or 32768)
    train_seq = int(
        calibration.get("seq_len")
        or data_cfg.get("max_sample_length")
        or 4096
    )
    scoring = data_cfg.get("replacement_scoring") or {}
    val_samples = int(scoring.get("num_samples") or 128)
    train_seed = int((config.get("pruning") or {}).get("shuffle_seed") or 444)

    defaults: list[dict[str, Any]] = []
    train_path = config.get("train_token_cache_path")
    if train_path:
        defaults.append(
            {
                "output": str(train_path),
                "split": "train",
                "num_samples": train_samples,
                "seq_length": train_seq,
                "shuffle_seed": train_seed,
            }
        )
    validation_path = config.get("validation_token_cache_path")
    if validation_path:
        defaults.append(
            {
                "output": str(validation_path),
                "split": "validation",
                "num_samples": val_samples,
                "seq_length": train_seq,
                "shuffle_seed": train_seed + 1,
            }
        )
    return defaults


def tokenize_data_stage(config: dict) -> StageResult:
    """Build every configured cache and record a normal stage manifest."""

    stage_config = config.get("tokenize_data") or {}
    puzzle_dir = Path(config.get("puzzle_dir") or (config.get("experiment") or {})["dir"])
    manifest_path = puzzle_dir / "manifests" / "tokenize_data.json"
    manifest = StageManifest(stage="tokenize_data", inputs={"config": config}, config=config)
    if not bool(stage_config.get("enabled", False)):
        manifest.complete(outputs={"enabled": False}, status="skipped")
        write_stage_manifest(manifest_path, manifest)
        return StageResult(
            "tokenize_data", "skipped", manifest_path, "Data tokenization is disabled."
        )

    caches = resolve_tokenize_caches(config)
    if not caches:
        raise ValueError(
            "tokenize_data.enabled is true but no caches are configured. "
            "Set tokenize_data.caches, or set train_token_cache_path / "
            "validation_token_cache_path so defaults can be derived."
        )

    tool = Path(__file__).resolve().parent / "tools" / "build_packed_token_memmap.py"
    teacher_dir = Path((config.get("convert") or {})["teacher_dir"])
    dataset_path = str(config["dataset_path"])
    outputs = []
    for cache in caches:
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
