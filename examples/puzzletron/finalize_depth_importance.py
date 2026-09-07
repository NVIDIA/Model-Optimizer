#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Publish a terminal manifest for a completed distributed depth run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from modelopt.torch.puzzletron.manifest import stage_manifest_from_config
from modelopt.torch.puzzletron.pipeline_config import pipeline_config_from_path
from modelopt.torch.puzzletron.stages.common import complete_stage


def _trajectory_path(config: dict[str, Any]) -> Path:
    depth = config.get("depth_importance") or {}
    configured = depth.get("output_dir")
    if configured:
        return Path(str(configured)) / "trajectory.json"
    return Path(str(config["puzzle_dir"])) / "depth" / "iterative" / "trajectory.json"


def finalize_depth_importance(config: dict[str, Any]) -> None:
    """Validate the durable trajectory and publish the depth terminal contract."""

    trajectory_path = _trajectory_path(config)
    try:
        trajectory = json.loads(trajectory_path.read_text())
    except (OSError, ValueError) as exc:
        raise ValueError(f"unable to read depth trajectory: {trajectory_path}") from exc

    depth = config.get("depth_importance") or {}
    expected_removals = int(
        depth.get("max_removals", depth.get("max_subblocks_to_remove", 10))
    )
    selected = trajectory.get("selected")
    if (
        trajectory.get("status") != "complete"
        or int(trajectory.get("max_removals", -1)) != expected_removals
        or not isinstance(selected, list)
        or len(selected) != expected_removals
    ):
        raise ValueError(f"depth trajectory is not complete: {trajectory_path}")

    manifest = stage_manifest_from_config("depth_importance", config, effective_config=config)
    complete_stage(
        config,
        manifest,
        outputs={
            "trajectory_path": str(trajectory_path),
            "selected": selected,
            "scenario_count": len(trajectory.get("scenarios") or ()),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()
    finalize_depth_importance(pipeline_config_from_path(args.config, overrides=args.override))


if __name__ == "__main__":
    main()
