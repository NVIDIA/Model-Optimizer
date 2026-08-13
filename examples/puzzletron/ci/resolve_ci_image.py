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

"""Validate and resolve the immutable image used by Puzzletron GPU CI."""

import json
import os
import re
import sys
from pathlib import Path

__all__ = ["resolve_image_reference", "validate_repository_contract"]

_NVCR_IMAGE = re.compile(
    r"nvcr\.io/(?:[A-Za-z0-9._-]+/)*[A-Za-z0-9._-]+@sha256:(?P<digest>[0-9a-f]{64})"
)
_CUDA_BASE_IMAGE = re.compile(r"nvidia/cuda:[A-Za-z0-9._-]+@sha256:[0-9a-f]{64}")


def resolve_image_reference(image: str) -> tuple[str, str]:
    """Return an immutable nvcr.io image and its digest cache key."""
    match = _NVCR_IMAGE.fullmatch(image)
    if match is None:
        raise ValueError("PUZZLETRON_GPU_CI_IMAGE must be an immutable nvcr.io digest")
    return image, match.group("digest")


def validate_repository_contract(repository_root: Path) -> None:
    """Verify the checked-out image recipe agrees with its recorded environment."""
    ci_root = repository_root / "examples/puzzletron"
    environment = json.loads((ci_root / "ci_environment.json").read_text())
    dockerfile = (ci_root / "ci/Dockerfile").read_text()
    base_image = environment["gpu_image"]["base_image"]

    if _CUDA_BASE_IMAGE.fullmatch(base_image) is None:
        raise ValueError("gpu_image.base_image must use a full lowercase SHA-256 digest")

    required_lines = (
        f"FROM {base_image}",
        "ENV PUZZLETRON_CI_ENVIRONMENT=/opt/puzzletron/ci_environment.json",
        "ENV PUZZLETRON_REQUIREMENTS=/opt/puzzletron/requirements.txt",
        "COPY pyproject.toml LICENSE_HEADER /opt/modelopt-dependencies/",
        "RUN bash /opt/puzzletron/setup_env.sh --deps",
        'python -m pip install "/opt/modelopt-dependencies[hf,puzzletron,dev-test]"',
        "bash /opt/puzzletron/setup_env.sh --verify",
    )
    missing = [line for line in required_lines if line not in dockerfile]
    if missing:
        raise ValueError(f"Dockerfile is missing recorded contract lines: {missing}")


def main() -> int:
    """Write validated values in GitHub output format."""
    try:
        validate_repository_contract(Path.cwd())
        configured_image = os.environ.get("PUZZLETRON_GPU_CI_IMAGE", "")
        if not configured_image:
            print(
                "::warning::PUZZLETRON_GPU_CI_IMAGE is not configured; "
                "skipping Puzzletron GPU tests until an immutable image is published.",
                file=sys.stderr,
            )
            print("configured=false")
            return 0
        image, cache_key = resolve_image_reference(configured_image)
    except (KeyError, OSError, ValueError, json.JSONDecodeError) as error:
        print(f"::error::{error}", file=sys.stderr)
        return 1

    print("configured=true")
    print(f"image={image}")
    print(f"cache_key={cache_key}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
