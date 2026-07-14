# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Create and verify the detached SHA-256 for a canonical PDD run manifest."""

from __future__ import annotations

import argparse
import os
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
    from pdd_artifacts import load_canonical_json, sha256_file
    from pdd_evaluation import validate_effectiveness_bundle

    manifest = args.manifest.resolve()
    load_canonical_json(manifest)
    detached = manifest.with_suffix(manifest.suffix + ".sha256")
    with detached.open("xb") as stream:
        stream.write((sha256_file(manifest) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())
    try:
        validate_effectiveness_bundle(manifest)
    except BaseException:
        detached.unlink(missing_ok=True)
        raise
    print(detached)


if __name__ == "__main__":
    main()
