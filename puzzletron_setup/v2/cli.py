# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dependency-light CLI for Puzzletron setup v2."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

from puzzletron_setup import SetupError

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["main"]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a navigable, schema-driven Puzzletron pruning campaign."
    )
    parser.add_argument(
        "--resume",
        type=Path,
        help="Resume from a campaign directory or answers_v2.yaml.",
    )
    parser.add_argument(
        "--defaults",
        type=Path,
        help="Explicit versioned defaults YAML; never discovered automatically.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help=(
            "Expose every advanced section and nested setting. "
            "Without this flag, setup uses a guided profile."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the setup-v2 command-line interface."""
    args = _parser().parse_args(argv)
    # Keep heavyweight model inspection out of --help and argument-error paths.
    from .wizard import run_wizard_v2

    try:
        campaign = run_wizard_v2(
            resume=args.resume,
            defaults_path=args.defaults,
            full=args.full,
        )
    except KeyboardInterrupt:
        target = args.resume or "<campaign>"
        print(
            "\nSetup interrupted. Resume with: "
            f"python examples/puzzletron/puzzletron_setup_v2.py --resume {target}"
        )
        return 130
    except SetupError as error:
        print(f"Setup stopped: {error}")
        return 2
    print(f"Campaign written to {campaign}")
    return 0
