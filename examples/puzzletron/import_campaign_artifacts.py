#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Safely import immutable artifact bundles between Puzzletron campaigns."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from modelopt.torch.puzzletron.artifact_import import DEFAULT_BUNDLES, import_campaign_artifacts

__all__ = ["main"]


def main(argv: list[str] | None = None) -> int:
    """Run the validated campaign-artifact import CLI."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--destination-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--target-config", type=Path)
    parser.add_argument("--artifact", action="append", choices=DEFAULT_BUNDLES, dest="bundles")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    result = import_campaign_artifacts(
        args.source_root,
        args.destination_root,
        args.receipt,
        bundles=args.bundles,
        dry_run=args.dry_run,
        target_config_path=args.target_config,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
