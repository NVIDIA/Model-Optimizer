# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dependency-light CLI for Puzzletron setup v2."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

from puzzletron_setup import SetupError

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
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    from .wizard import run_wizard_v2

    try:
        campaign = run_wizard_v2(
            resume=args.resume,
            defaults_path=args.defaults,
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
