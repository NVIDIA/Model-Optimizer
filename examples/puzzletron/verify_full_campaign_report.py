#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Generate and verify the complete Puzzletron campaign report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from modelopt.torch.puzzletron.diagnostics.campaign_progress_report import (
    generate_campaign_progress_report,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--puzzle-dir", type=Path, required=True)
    parser.add_argument("--model-name", default="Qwen3.5-9B")
    args = parser.parse_args()
    root = args.puzzle_dir

    report = generate_campaign_progress_report(root, model_name=args.model_name)
    html = Path(report["html"])
    text = html.read_text()
    required = (
        "Sort diagnosis",
        "Activation ranking diagnosis",
        "Nested bypass",
        "Iterative depth pruning",
        "Granularity and artifact coverage",
        "vLLM statistics",
        "Replace-one-subblock scoring",
        "MIP solutions",
        "Exact evaluation",
        "AIPerf",
    )
    missing = [name for name in required if name not in text]
    if missing:
        raise RuntimeError(f"report is missing sections: {missing}")

    coverage = root / "artifacts" / "mip" / "artifact_coverage.json"
    tournament = root / "mip" / "puzzle_solutions" / "depth_tournament" / "solutions.json"
    evaluation = (
        root
        / "artifacts"
        / "evaluation"
        / "profiles"
        / "depth-tournament"
        / "text-s128-l16384"
        / "evaluation_summary.json"
    )
    observations = root / "artifacts" / "bypass" / "dp_observations.jsonl"
    aiperf = list(
        (root / "artifacts" / "aiperf" / "profiles" / "depth-finalists").glob(
            "*/aiperf_results.json"
        )
    )
    if not coverage.is_file():
        raise FileNotFoundError(coverage)
    if len(json.loads(tournament.read_text())) != 7:
        raise RuntimeError(f"expected seven depth solutions: {tournament}")
    if not evaluation.is_file():
        raise FileNotFoundError(evaluation)
    if len(aiperf) != 1:
        raise RuntimeError(f"expected one merged AIPerf matrix, found {aiperf}")
    if not observations.is_file() or observations.stat().st_size == 0:
        raise RuntimeError(f"bypass observations are missing or empty: {observations}")

    print(f"FINAL_REPORT={html}")
    print("DEPTH_SOLUTIONS=7")
    print(f"EVALUATION={evaluation}")
    print(f"AIPERF={aiperf[0]}")
    print(f"BYPASS_OBSERVATIONS={observations}")
    print(f"COVERAGE={coverage}")


if __name__ == "__main__":
    main()
