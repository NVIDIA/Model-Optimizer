# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Command-line entry point for the Puzzletron setup wizard."""

from __future__ import annotations

import argparse
from pathlib import Path

from . import SetupError

__all__ = ["main"]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a Puzzletron pruning campaign.")
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Ask advanced solver, pipeline, and per-stage resource questions.",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        help="Resume from a campaign directory or answers.yaml (fresh by default).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the setup wizard and return a process exit code."""
    args = _parser().parse_args(argv)
    # Setup dependencies are intentionally optional for the rest of ModelOpt;
    # defer their import so ``--help`` remains available before installation.
    from .wizard import run_wizard

    try:
        campaign = run_wizard(detailed=args.detailed, resume=args.resume)
    except KeyboardInterrupt:
        print("\nSetup interrupted. Re-run with --resume <campaign> to continue.")
        return 130
    except SetupError as error:
        print(f"Setup stopped: {error}")
        return 2
    print(f"Campaign written to {campaign}")
    return 0
