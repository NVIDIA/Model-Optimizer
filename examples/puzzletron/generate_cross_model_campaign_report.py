#!/usr/bin/env python3
"""Generate one strict, self-contained report for a cross-model campaign member."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from modelopt.torch.puzzletron.diagnostics.campaign_report import generate_campaign_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--puzzle-dir", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--expected-kd-scenarios", type=int, required=True)
    args = parser.parse_args()
    report = generate_campaign_report(
        args.puzzle_dir,
        model_name=args.model_name,
        expected_kd_scenarios=args.expected_kd_scenarios,
    )
    print(json.dumps(report["reports"], indent=2))


if __name__ == "__main__":
    main()
