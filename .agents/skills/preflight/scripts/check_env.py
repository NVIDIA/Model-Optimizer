#!/usr/bin/env python3
"""Report whether named environment variables are set without printing values."""

from __future__ import annotations

import argparse
import os
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def check(names: Sequence[str]) -> tuple[list[str], list[str]]:
    """Return set and missing variable names, preserving input order."""
    present: list[str] = []
    missing: list[str] = []
    for name in dict.fromkeys(names):
        if not ENV_NAME.fullmatch(name):
            raise ValueError(f"Invalid environment variable name: {name!r}")
        (present if os.environ.get(name) else missing).append(name)
    return present, missing


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="+", metavar="NAME")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Check requested names and return nonzero when any are missing."""
    args = build_parser().parse_args(argv)
    try:
        present, missing = check(args.names)
    except ValueError as error:
        print(f"ERROR: {error}")
        return 2

    status = dict.fromkeys(present, "set") | dict.fromkeys(missing, "missing")
    for name in dict.fromkeys(args.names):
        print(f"{name}: {status[name]}")
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
