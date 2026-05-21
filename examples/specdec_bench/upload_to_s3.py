#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Upload specdec_bench results to S3.

Handles both flat and sweep directory layouts:

  Flat:   LOCAL_DIR/run_name/{configuration,timing,...}.json
  Sweep:  LOCAL_DIR/sweep_name/run_name/{configuration,timing,...}.json

LOCAL_DIR's name is preserved under the S3 prefix:

  s3://bucket/prefix/LOCAL_DIR_NAME/[sweep_name/]run_name/

Usage examples:

  # Upload a sweep output directory
  python upload_to_s3.py /data/sweep_outputs/my_sweep s3://team-specdec-workgroup/results

  # Upload a single run
  python upload_to_s3.py /data/my_single_run s3://team-specdec-workgroup/results

  # Skip already-uploaded runs instead of failing
  python upload_to_s3.py /data/sweep_outputs/my_sweep s3://team-specdec-workgroup/results --skip-existing
"""

import argparse
import os
import sys
from pathlib import Path

from specdec_bench.s3_utils import (
    S3_DEFAULT_ENDPOINT,
    S3_DEFAULT_KEY_ID,
    S3_DEFAULT_SECRET,
    make_s3_client,
    parse_s3_path,
    upload_run_dir,
)

_RUN_SENTINELS = ("configuration.json", "timing.json", "aa_timing.json", "acceptance_rate.json")


def _is_run_dir(d: Path) -> bool:
    return any((d / f).exists() for f in _RUN_SENTINELS)


def _discover_runs(local_root: Path, s3_prefix_base: str) -> list[tuple[Path, str]]:
    """Return list of (local_run_dir, s3_key) pairs to upload.

    local_root's name is appended to s3_prefix_base, then contents mirrored:
      local_root/run_name/            → s3_prefix_base/local_root.name/run_name/
      local_root/sweep_name/run_name/ → s3_prefix_base/local_root.name/sweep_name/run_name/
    """
    base = f"{s3_prefix_base}/{local_root.name}".lstrip("/")
    queue: list[tuple[Path, str]] = []

    if _is_run_dir(local_root):
        # The directory itself is a single run
        queue.append((local_root, base))
        return queue

    for child in sorted(local_root.iterdir()):
        if not child.is_dir():
            continue
        if _is_run_dir(child):
            # Flat layout: local_root/run_name/
            queue.append((child, f"{base}/{child.name}"))
        else:
            # Sweep layout: local_root/sweep_name/run_name/
            queue.extend(
                (grandchild, f"{base}/{child.name}/{grandchild.name}")
                for grandchild in sorted(child.iterdir())
                if grandchild.is_dir() and _is_run_dir(grandchild)
            )

    return queue


def main():
    parser = argparse.ArgumentParser(
        description="Upload specdec_bench results to S3.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Usage examples:")[1] if "Usage examples:" in __doc__ else "",
    )
    parser.add_argument("local_dir", help="Local results directory to upload")
    parser.add_argument(
        "s3_dest", help="S3 destination prefix, e.g. s3://team-specdec-workgroup/results"
    )
    parser.add_argument(
        "--endpoint",
        default=os.environ.get("S3_ENDPOINT", S3_DEFAULT_ENDPOINT),
        help="S3 endpoint URL",
    )
    parser.add_argument(
        "--key-id",
        default=os.environ.get("S3_KEY_ID", S3_DEFAULT_KEY_ID),
        dest="key_id",
        help="S3 access key ID",
    )
    parser.add_argument(
        "--secret",
        default=os.environ.get("S3_SECRET", S3_DEFAULT_SECRET),
        help="S3 secret access key",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip runs that already exist in S3 instead of failing",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be uploaded without actually uploading",
    )
    args = parser.parse_args()

    if not args.s3_dest.startswith("s3://"):
        sys.exit("Error: s3_dest must start with s3://")

    local_root = Path(args.local_dir).resolve()
    if not local_root.is_dir():
        sys.exit(f"Error: {local_root} is not a directory")

    bucket, s3_prefix_base = parse_s3_path(args.s3_dest)
    queue = _discover_runs(local_root, s3_prefix_base)

    if not queue:
        sys.exit("No run directories found to upload.")

    print(f"Found {len(queue)} run(s) to upload → s3://{bucket}/")

    if args.dry_run:
        for local_run_dir, s3_key in queue:
            print(f"  {local_run_dir} → s3://{bucket}/{s3_key}")
        return

    s3 = make_s3_client(args.endpoint, args.key_id, args.secret)

    errors = 0
    skipped = 0
    uploaded = 0
    for local_run_dir, s3_key in queue:
        print(f"\n{local_run_dir.name} → s3://{bucket}/{s3_key}")
        try:
            upload_run_dir(s3, local_run_dir, bucket, s3_key)
            uploaded += 1
        except ValueError as exc:
            if args.skip_existing:
                print(f"  Skipped: {exc}")
                skipped += 1
            else:
                print(f"  Error: {exc}")
                errors += 1

    print(f"\nDone: {uploaded} uploaded, {skipped} skipped, {errors} failed.")
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
