#!/usr/bin/env python3
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

"""Publish a terminal stage manifest for distributed depth importance."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from modelopt.torch.puzzletron.manifest import stage_manifest_from_config, write_stage_manifest
from modelopt.torch.puzzletron.pipeline_config import pipeline_config_from_path

__all__ = ["finalize_depth_importance", "main"]


def _validated_trajectory(path: Path, *, expected_removals: int) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, ValueError) as error:
        raise RuntimeError(f"depth trajectory is unavailable or invalid: {path}") from error
    selected = payload.get("selected")
    if (
        payload.get("status") != "complete"
        or int(payload.get("max_removals", -1)) != expected_removals
        or not isinstance(selected, list)
        or len(selected) != expected_removals
    ):
        raise RuntimeError(
            f"depth trajectory is incomplete: expected {expected_removals} removals in {path}"
        )
    return payload


def finalize_depth_importance(
    config_path: str | Path,
    puzzle_dir: str | Path,
    output_dir: str | Path,
    *,
    overrides: list[str] | None = None,
) -> dict[str, Any]:
    """Validate the durable trajectory and publish the canonical stage manifest."""

    config = pipeline_config_from_path(config_path, overrides=overrides)
    puzzle_dir = Path(puzzle_dir)
    config["puzzle_dir"] = str(puzzle_dir)
    config.setdefault("experiment", {})["dir"] = str(puzzle_dir)
    depth = config.get("depth_importance") or {}
    expected_removals = int(depth.get("max_removals", depth.get("max_subblocks_to_remove", 10)))
    trajectory_path = Path(output_dir) / "trajectory.json"
    trajectory = _validated_trajectory(
        trajectory_path,
        expected_removals=expected_removals,
    )

    outputs = {
        "trajectory_path": str(trajectory_path),
        "scenario_count": len(trajectory.get("scenarios") or ()),
        "selected": trajectory["selected"],
    }
    manifest = stage_manifest_from_config("depth_importance", config)
    manifest.complete(outputs=outputs)
    write_stage_manifest(
        puzzle_dir / "manifests" / "depth_importance.json",
        manifest,
    )
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--puzzle-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    overrides = [
        override
        for override in os.environ.get("DISTRIBUTED_EVAL_OVERRIDES", "").splitlines()
        if override
    ]
    finalize_depth_importance(
        args.config,
        args.puzzle_dir,
        args.output_dir,
        overrides=overrides,
    )


if __name__ == "__main__":
    main()
