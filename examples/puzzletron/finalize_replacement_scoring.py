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

"""Publish replacement-scoring reports after distributed evaluation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

if __package__:
    from .embedding_pipeline import finalize_replacement_scoring_diagnostics
else:
    from embedding_pipeline import finalize_replacement_scoring_diagnostics

from modelopt.torch.puzzletron.diagnostics import generate_replace_block_report
from modelopt.torch.puzzletron.manifest import stage_manifest_from_config, write_stage_manifest
from modelopt.torch.puzzletron.pipeline_config import pipeline_config_from_path


def _successful_manifest_identity(manifest_path: str | Path) -> str | None:
    try:
        manifest = json.loads(Path(manifest_path).read_text())
    except (OSError, ValueError):
        return None
    if manifest.get("stage") != "replacement_scoring" or manifest.get("status") != "success":
        return None
    identity = manifest.get("semantic_identity")
    return str(identity) if identity else None


def finalization_marker_is_current(
    marker_path: str | Path,
    manifest_path: str | Path,
    summary_path: str | Path,
) -> bool:
    """Return whether a pool marker names the currently published result."""

    try:
        marker_identity = Path(marker_path).read_text().strip()
        summary = json.loads(Path(summary_path).read_text())
        manifest = json.loads(Path(manifest_path).read_text())
    except OSError:
        return False
    except ValueError:
        return False
    return bool(
        marker_identity
        and marker_identity == _successful_manifest_identity(manifest_path)
        and summary == (manifest.get("outputs") or {}).get("report")
    )


def write_finalization_marker(marker_path: str | Path, manifest_path: str | Path) -> None:
    """Atomically bind a pool marker to the published manifest identity."""

    identity = _successful_manifest_identity(manifest_path)
    if identity is None:
        raise RuntimeError(f"replacement-scoring manifest is not successful: {manifest_path}")
    marker = Path(marker_path)
    temporary = marker.with_suffix(marker.suffix + ".tmp")
    temporary.write_text(identity + "\n")
    temporary.replace(marker)


def finalize_replacement_scoring(
    config_path: str | Path,
    puzzle_dir: str | Path,
    *,
    overrides: list[str] | None = None,
) -> dict:
    """Publish replacement reports and their canonical terminal manifest."""

    config = pipeline_config_from_path(config_path, overrides=overrides)
    config["puzzle_dir"] = str(puzzle_dir)
    embedding = config.get("embedding_pruning") or {}
    if bool(embedding.get("enabled", False)):
        report = finalize_replacement_scoring_diagnostics(config)
    else:
        puzzle_dir = Path(puzzle_dir)
        scoring = config.get("replacement_scoring") or {}
        granularity = str(scoring.get("granularity", "block"))
        stem = (
            "single_subblock_replacement_solutions"
            if granularity == "subblock"
            else "single_sequence_replacement_solutions"
        )
        report = generate_replace_block_report(
            puzzle_dir,
            scores_dir=puzzle_dir / f"{stem}--validation",
            output_dir=puzzle_dir / "artifacts" / "replacement_scoring",
            granularity=granularity,
            default_metric=str(scoring.get("default_metric", "normalized_mse_loss_hidden_states")),
            default_layer_count=int(scoring.get("default_layer_count", 5)),
            anchor_count=int(scoring.get("anchor_count", 3)),
            trend_relative_tolerance=float(scoring.get("trend_relative_tolerance", 0.02)),
        )

    manifest = stage_manifest_from_config("replacement_scoring", config)
    manifest.complete(outputs={"report": report})
    write_stage_manifest(
        Path(puzzle_dir) / "manifests" / "replacement_scoring.json",
        manifest,
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--puzzle-dir", required=True)
    args = parser.parse_args()

    overrides = [
        override for override in os.environ.get("FINALIZE_OVERRIDES", "").splitlines() if override
    ]
    finalize_replacement_scoring(args.config, args.puzzle_dir, overrides=overrides)


if __name__ == "__main__":
    main()
