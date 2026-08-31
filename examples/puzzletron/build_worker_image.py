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
import hashlib
import os
import platform
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

__all__ = [
    "artifact_names",
    "artifact_stem",
    "build_parser",
    "main",
    "write_checksum",
]

_PLATFORM = "linux/amd64"
_REVISION_PATTERN = re.compile(r"[0-9a-f]{40}")


def artifact_stem(revision: str) -> str:
    """Return the common filename stem for artifacts built from ``revision``."""

    if not _REVISION_PATTERN.fullmatch(revision):
        raise ValueError("Puzzletron image revision must be a full lowercase Git commit")
    return f"modelopt-puzzletron-linux-amd64-git-{revision[:12]}"


def artifact_names(revision: str) -> dict[str, str]:
    """Return the filenames shared by the Docker and SquashFS export workflow."""

    stem = artifact_stem(revision)
    return {
        "archive": f"{stem}.tar.zst",
        "sqsh": f"{stem}.sqsh",
    }


def write_checksum(path: Path) -> str:
    """Write and return the SHA-256 checksum for an exported artifact."""

    digest = hashlib.sha256()
    with path.open("rb") as artifact:
        while chunk := artifact.read(4 * 1024 * 1024):
            digest.update(chunk)
    checksum = digest.hexdigest()
    path.with_name(f"{path.name}.sha256").write_text(f"{checksum}  {path.name}\n")
    return checksum


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
    existing = [
        candidate for candidate in (path, path.with_name(f"{name}.sha256")) if candidate.exists()
    ]
    if existing:
        raise FileExistsError(f"Refusing to overwrite existing artifact: {existing[0]}")
    return path


def _export_archive(image: str, output: Path) -> None:
    partial = output.with_name(f".{output.name}.partial")
    if partial.exists():
        raise FileExistsError(f"Refusing to overwrite incomplete artifact: {partial}")

    save = subprocess.Popen(["docker", "save", image], stdout=subprocess.PIPE)
    if save.stdout is None:
        raise RuntimeError("Docker archive export did not open its output stream")
    try:
        compressed = subprocess.run(
            ["zstd", "--threads=0", "--quiet", "--output", str(partial)],
            stdin=save.stdout,
            check=False,
        )
    finally:
        save.stdout.close()
    save_returncode = save.wait()
    if compressed.returncode or save_returncode:
        partial.unlink(missing_ok=True)
        raise RuntimeError(
            f"Docker archive export failed: docker={save_returncode}, zstd={compressed.returncode}"
        )
    partial.replace(output)


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
        "--archive",
        action="store_true",
        help="export a compressed Docker archive and checksum",
    )
    parser.add_argument(
        "--sqsh",
        action="store_true",
        help="export an Enroot/Pyxis SquashFS image and checksum",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Build, verify, and optionally export the Puzzletron worker image."""

    parser = build_parser()
    args = parser.parse_args(argv)
    if (args.archive or args.sqsh) and args.output_dir is None:
        parser.error("--output-dir is required with --archive or --sqsh")
    if args.output_dir is not None and not (args.archive or args.sqsh):
        parser.error("--output-dir requires --archive or --sqsh")

    repository_root = Path(__file__).resolve().parents[2]
    revision = _source_revision(repository_root)
    _require_linux_amd64()
    names = artifact_names(revision)
    image = f"modelopt-puzzletron:linux-amd64-git-{revision[:12]}"

    archive = None
    sqsh = None
    if args.output_dir is not None:
        args.output_dir = args.output_dir.expanduser().resolve()
        if args.output_dir.is_relative_to(repository_root):
            parser.error("--output-dir must be outside the repository")
        archive = _output_path(args.output_dir, names["archive"]) if args.archive else None
        sqsh = _output_path(args.output_dir, names["sqsh"]) if args.sqsh else None

    required_tools = ["docker"]
    if args.archive:
        required_tools.append("zstd")
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
    _run(
        [
            "docker",
            "run",
            "--rm",
            image,
            "python",
            "/opt/puzzletron/verify_image_environment.py",
            "--environment",
            "/opt/puzzletron/ci_environment.json",
        ]
    )

    if args.output_dir is not None:
        if archive is not None:
            _export_archive(image, archive)
            write_checksum(archive)
        if sqsh is not None:
            _export_sqsh(image, sqsh)
            write_checksum(sqsh)

    print(f"Docker image: {image}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
