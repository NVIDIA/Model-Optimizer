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

"""Build and optionally export the Linux amd64 Puzzletron worker image."""

from __future__ import annotations

import argparse
import os
import platform
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

__all__ = [
    "build_parser",
    "main",
    "sqsh_name",
]

_PLATFORM = "linux/amd64"
_REVISION_PATTERN = re.compile(r"[0-9a-f]{40}")


def sqsh_name(revision: str) -> str:
    """Return the SquashFS filename for an image built from ``revision``."""

    if not _REVISION_PATTERN.fullmatch(revision):
        raise ValueError("Puzzletron image revision must be a full lowercase Git commit")
    return f"modelopt-puzzletron-linux-amd64-git-{revision[:12]}.sqsh"


def _run(command: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(command, check=True, **kwargs)


def _source_revision(repository_root: Path) -> str:
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=repository_root,
        text=True,
    )
    if status:
        raise RuntimeError("Build the Puzzletron worker image from a clean Git checkout")
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repository_root, text=True
    ).strip()
    if not _REVISION_PATTERN.fullmatch(revision):
        raise RuntimeError(f"Git returned an invalid Puzzletron image revision: {revision!r}")
    return revision


def _require_tools(*tools: str) -> None:
    missing = [tool for tool in tools if shutil.which(tool) is None]
    if missing:
        raise RuntimeError(f"Missing required command(s): {', '.join(missing)}")


def _require_linux_amd64() -> None:
    if platform.system() != "Linux" or platform.machine() not in {"amd64", "x86_64"}:
        raise RuntimeError("Build the Puzzletron worker image on a Linux amd64 host")


def _output_path(output_dir: Path, name: str) -> Path:
    path = output_dir / name
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing artifact: {path}")
    return path


def _export_sqsh(image: str, output: Path) -> None:
    partial = output.with_name(f".{output.stem}.partial.sqsh")
    if partial.exists():
        raise FileExistsError(f"Refusing to overwrite incomplete artifact: {partial}")

    with tempfile.TemporaryDirectory(prefix="puzzletron-enroot-") as enroot_root:
        environment = {
            **os.environ,
            "ENROOT_CACHE_PATH": f"{enroot_root}/cache",
            "ENROOT_DATA_PATH": f"{enroot_root}/data",
            "ENROOT_RUNTIME_PATH": f"{enroot_root}/runtime",
        }
        try:
            _run(
                ["enroot", "import", "--output", str(partial), f"dockerd://{image}"],
                env=environment,
            )
        except (OSError, subprocess.CalledProcessError):
            partial.unlink(missing_ok=True)
            raise
    partial.replace(output)


def build_parser() -> argparse.ArgumentParser:
    """Build the worker-image command-line parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="directory for optional exported artifacts",
    )
    parser.add_argument(
        "--sqsh",
        action="store_true",
        help="export an Enroot/Pyxis SquashFS image",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Build, verify, and optionally export the Puzzletron worker image."""

    parser = build_parser()
    args = parser.parse_args(argv)
    if args.sqsh and args.output_dir is None:
        parser.error("--output-dir is required with --sqsh")
    if args.output_dir is not None and not args.sqsh:
        parser.error("--output-dir requires --sqsh")

    repository_root = Path(__file__).resolve().parents[2]
    revision = _source_revision(repository_root)
    _require_linux_amd64()
    image = f"modelopt-puzzletron:linux-amd64-git-{revision[:12]}"

    sqsh = None
    if args.output_dir is not None:
        args.output_dir = args.output_dir.expanduser().resolve()
        if args.output_dir.is_relative_to(repository_root):
            parser.error("--output-dir must be outside the repository")
        sqsh = _output_path(args.output_dir, sqsh_name(revision))

    required_tools = ["docker"]
    if args.sqsh:
        required_tools.append("enroot")
    _require_tools(*required_tools)
    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    _run(
        [
            "docker",
            "build",
            "--platform",
            _PLATFORM,
            "--file",
            "examples/puzzletron/Dockerfile",
            "--build-arg",
            f"MODELOPT_REVISION={revision}",
            "--tag",
            image,
            ".",
        ],
        cwd=repository_root,
    )
    recorded_revision = subprocess.check_output(
        [
            "docker",
            "image",
            "inspect",
            "--format",
            '{{ index .Config.Labels "org.opencontainers.image.revision" }}',
            image,
        ],
        text=True,
    ).strip()
    if recorded_revision != revision:
        raise RuntimeError(
            f"Puzzletron image revision mismatch: expected {revision}, found {recorded_revision}"
        )
    if sqsh is not None:
        _export_sqsh(image, sqsh)

    print(f"Docker image: {image}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
