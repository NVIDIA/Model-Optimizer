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

"""S3 upload utilities for specdec_bench results."""

from pathlib import Path

S3_DEFAULT_ENDPOINT = "https://pdx.s8k.io"
S3_DEFAULT_KEY_ID = ""
S3_DEFAULT_SECRET = ""


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
        endpoint_url=endpoint,
        aws_access_key_id=key_id,
        aws_secret_access_key=secret,
        region_name="us-east-1",
        config=Config(s3={"addressing_style": "path"}),
    )


def s3_prefix_exists(s3, bucket: str, prefix: str) -> bool:
    """Return True if any object exists under prefix."""
    resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix.rstrip("/") + "/", MaxKeys=1)
    return bool(resp.get("Contents"))


def _upload_files(s3, local_dir: Path, bucket: str, s3_prefix: str) -> None:
    """Upload all files under local_dir without any existence check."""
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
