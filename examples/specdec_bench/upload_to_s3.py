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
import json
import os
import sys
import traceback
from pathlib import Path

_RUN_SENTINELS = ("configuration.json", "timing.json", "acceptance_rate.json")

# Provenance fields that must be non-null for a run to count as strictly
# reproducible from S3. `container_image` carries the torch / CUDA / NCCL
# versions which dump_env can't otherwise see; without it the row in the
# visualizer has no path back to the binary environment.
_REQUIRED_PROVENANCE_FIELDS = ("container_image",)


def _check_provenance(run_dir: Path) -> list[str]:
    """Return a list of missing required provenance fields in run_dir/configuration.json.

    Empty list means the run is acceptable for upload. A run with no
    configuration.json at all is reported as missing every field.
    """
    cfg_path = run_dir / "configuration.json"
    if not cfg_path.is_file():
        return list(_REQUIRED_PROVENANCE_FIELDS)
    try:
        with open(cfg_path) as f:
            cfg = json.load(f)
    except (OSError, json.JSONDecodeError):
        return list(_REQUIRED_PROVENANCE_FIELDS)
    return [k for k in _REQUIRED_PROVENANCE_FIELDS if not cfg.get(k)]


# ── S3 helpers ────────────────────────────────────────────────────────────────
# Endpoint, key id, and secret default to empty and are taken from --endpoint /
# --key-id / --secret (or the corresponding S3_ENDPOINT / S3_KEY_ID / S3_SECRET
# env vars).


def parse_s3_path(path: str) -> tuple[str, str]:
    """'s3://bucket/prefix' → (bucket, prefix).  prefix may be empty."""
    without_scheme = path[5:]  # strip "s3://"
    parts = without_scheme.split("/", 1)
    bucket = parts[0]
    prefix = parts[1].strip("/") if len(parts) > 1 else ""
    return bucket, prefix


def make_s3_client(endpoint: str, key_id: str, secret: str):
    import boto3
    from botocore.config import Config

    return boto3.client(
        "s3",
        endpoint_url=endpoint or None,
        aws_access_key_id=key_id,
        aws_secret_access_key=secret,
        region_name="us-east-1",
        config=Config(s3={"addressing_style": "path"}),
    )


def s3_prefix_exists(s3, bucket: str, prefix: str) -> bool:
    resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix.rstrip("/") + "/", MaxKeys=1)
    return bool(resp.get("Contents"))


def _upload_files(s3, local_dir: Path, bucket: str, s3_prefix: str) -> None:
    for file_path in sorted(local_dir.rglob("*")):
        if not file_path.is_file():
            continue
        rel = file_path.relative_to(local_dir).as_posix()
        key = f"{s3_prefix}/{rel}"
        s3.upload_file(str(file_path), bucket, key)
        print(f"  Uploaded: s3://{bucket}/{key}")


def upload_run_dir(s3, local_dir: Path, bucket: str, s3_prefix: str) -> None:
    """Upload a single run directory to s3://bucket/s3_prefix/.

    Raises ValueError if the destination prefix already has any objects.
    """
    s3_prefix = s3_prefix.rstrip("/")
    if s3_prefix_exists(s3, bucket, s3_prefix):
        raise ValueError(
            f"S3 destination already exists: s3://{bucket}/{s3_prefix} — refusing to overwrite"
        )
    _upload_files(s3, local_dir, bucket, s3_prefix)


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
        default=os.environ.get("S3_ENDPOINT", ""),
        help="S3 endpoint URL",
    )
    parser.add_argument(
        "--key-id",
        default=os.environ.get("S3_KEY_ID", ""),
        dest="key_id",
        help="S3 access key ID",
    )
    parser.add_argument(
        "--secret",
        default=os.environ.get("S3_SECRET", ""),
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
    parser.add_argument(
        "--allow-incomplete-provenance",
        action="store_true",
        help=(
            "Allow uploading runs whose configuration.json is missing required "
            "provenance fields (container_image). Without this flag, such runs "
            "are rejected because they cannot be strictly reproduced from S3."
        ),
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

    # Reject runs missing required provenance fields unless explicitly allowed.
    incomplete = [(d, _check_provenance(d)) for d, _ in queue]
    incomplete = [(d, missing) for d, missing in incomplete if missing]
    if incomplete:
        for d, missing in incomplete:
            print(f"  {d}: missing {missing}", file=sys.stderr)
        if not args.allow_incomplete_provenance:
            sys.exit(
                f"Error: {len(incomplete)} run(s) are missing required provenance "
                f"fields. Set CONTAINER_IMAGE before running, or pass "
                f"--allow-incomplete-provenance to upload anyway."
            )
        print(
            f"Warning: --allow-incomplete-provenance — uploading {len(incomplete)} "
            f"run(s) with missing provenance.",
            file=sys.stderr,
        )

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
            # Raised by upload_run_dir when the destination already exists.
            if args.skip_existing:
                print(f"  Skipped: {exc}")
                skipped += 1
            else:
                print(f"  Error: {exc}")
                errors += 1
        except Exception as exc:
            # ClientError (auth, throttling, network), OSError on a single file,
            # etc. Keep going so the other runs in a sweep still upload, and
            # report a per-run summary at the end. Sweep-level non-zero exit
            # still flags the overall run as failed. Print the full traceback so
            # boto3 ClientError details are recoverable from the log.
            print(f"  Error: {type(exc).__name__}: {exc}")
            traceback.print_exc(file=sys.stderr)
            errors += 1

    print(f"\nDone: {uploaded} uploaded, {skipped} skipped, {errors} failed.")
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
