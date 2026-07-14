# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..manifest import StageManifest, write_stage_manifest
from ..stage_runner import StageResult

__all__ = ["experiment_dir", "stage_manifest_path", "complete_stage"]


def experiment_dir(config: dict[str, Any]) -> Path:
    exp = ((config.get("experiment") or {}).get("dir"))
    if exp is None:
        raise ValueError("Puzzletron config must define experiment.dir")
    return Path(exp)


def stage_manifest_path(config: dict[str, Any], stage: str) -> Path:
    return experiment_dir(config) / "manifests" / f"{stage}.json"


def complete_stage(
    config: dict[str, Any],
    manifest: StageManifest,
    *,
    outputs: dict[str, Any] | None = None,
    status: str = "success",
    message: str | None = None,
) -> StageResult:
    manifest.complete(outputs=outputs or {}, status=status)
    path = stage_manifest_path(config, manifest.stage)
    write_stage_manifest(path, manifest)
    return StageResult(
        stage=manifest.stage,
        status=status,
        manifest_path=path,
        message=message or f"Stage '{manifest.stage}' completed.",
    )
