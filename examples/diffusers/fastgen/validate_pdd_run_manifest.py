# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Authenticate a complete PDD effectiveness evidence bundle."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.dont_write_bytecode = True

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parents[2]
for path in (_REPO_ROOT, _THIS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    from pdd_evaluation import validate_effectiveness_bundle

    validated = validate_effectiveness_bundle(args.manifest)
    print(validated["manifest_sha256"])


if __name__ == "__main__":
    main()
