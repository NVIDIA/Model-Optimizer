# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..manifest import StageManifest, write_stage_manifest
from ..stage_runner import StageResult

if TYPE_CHECKING:
    from .graph import StageSkipReason, StageStatus

__all__ = ["experiment_dir", "stage_manifest_path", "complete_stage"]


def experiment_dir(config: dict[str, Any]) -> Path:
    exp = (config.get("experiment") or {}).get("dir")
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
    status: str | StageStatus = "success",
    skip_reason: str | StageSkipReason | None = None,
    message: str | None = None,
) -> StageResult:
    """Persist terminal manifest state and return its normalized stage result.

    ``status`` and ``skip_reason`` are normalized to their string values in the
    manifest. The returned result reports those same persisted values and the
    manifest path, including the skip reason for skipped stages.
    """

    manifest.complete(outputs=outputs or {}, status=status, skip_reason=skip_reason)
    path = stage_manifest_path(config, manifest.stage)
    write_stage_manifest(path, manifest)
    return StageResult(
        stage=manifest.stage,
        status=manifest.status,
        manifest_path=path,
        message=message or f"Stage '{manifest.stage}' completed.",
        skip_reason=manifest.skip_reason,
    )
