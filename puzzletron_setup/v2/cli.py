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

"""Dependency-light CLI for Puzzletron setup v2."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

from puzzletron_setup import SetupError

from .presets import QUICK_SETUP_PRESETS
from .prompts import NonInteractiveBackend

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
        "--campaign-dir",
        type=Path,
        help="Campaign directory for a new setup; bypasses that interactive prompt.",
    )
    parser.add_argument(
        "--profile",
        choices=tuple(preset.name for preset in QUICK_SETUP_PRESETS),
        default="balanced",
        help="Guided setup profile for a new campaign (default: balanced).",
    )
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Accept resolved defaults and fail if any required answer has no default.",
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
    parser = _parser()
    args = parser.parse_args(argv)
    if args.resume is not None and args.campaign_dir is not None:
        parser.error("--campaign-dir cannot be combined with --resume")
    if args.non_interactive and args.resume is None:
        if args.campaign_dir is None:
            parser.error("--non-interactive requires --campaign-dir for a new campaign")
        if args.defaults is None:
            parser.error("--non-interactive requires --defaults for a new campaign")
    # Keep heavyweight model inspection out of --help and argument-error paths.
    from .wizard import run_wizard_v2

    try:
        campaign = run_wizard_v2(
            resume=args.resume,
            defaults_path=args.defaults,
            full=args.full,
            campaign_dir=args.campaign_dir,
            setup_profile=args.profile,
            backend=NonInteractiveBackend() if args.non_interactive else None,
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
