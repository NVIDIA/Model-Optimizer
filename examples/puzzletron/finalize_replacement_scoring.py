#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Publish replacement-scoring reports after distributed evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

from modelopt.torch.puzzletron.diagnostics import generate_replace_block_report
from modelopt.torch.puzzletron.pipeline_config import pipeline_config_from_path

from embedding_pipeline import finalize_replacement_scoring_diagnostics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--puzzle-dir", required=True)
    args = parser.parse_args()

    config = pipeline_config_from_path(args.config)
    config["puzzle_dir"] = args.puzzle_dir
    embedding = config.get("embedding_pruning") or {}
    if bool(embedding.get("enabled", False)):
        finalize_replacement_scoring_diagnostics(config)
        return

    puzzle_dir = Path(args.puzzle_dir)
    scoring = config.get("replacement_scoring") or {}
    granularity = str(scoring.get("granularity", "block"))
    stem = (
        "single_subblock_replacement_solutions"
        if granularity == "subblock"
        else "single_sequence_replacement_solutions"
    )
    generate_replace_block_report(
        puzzle_dir,
        scores_dir=puzzle_dir / f"{stem}--validation",
        output_dir=puzzle_dir / "artifacts" / "replacement_scoring",
        granularity=granularity,
        default_metric=str(
            scoring.get("default_metric", "normalized_mse_loss_hidden_states")
        ),
        default_layer_count=int(scoring.get("default_layer_count", 5)),
        anchor_count=int(scoring.get("anchor_count", 3)),
        trend_relative_tolerance=float(scoring.get("trend_relative_tolerance", 0.02)),
    )


if __name__ == "__main__":
    main()
