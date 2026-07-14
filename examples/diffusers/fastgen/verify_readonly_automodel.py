# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify and snapshot the exact released AutoModel distribution used by PDD."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
from typing import Any

_MANIFEST_PATH = Path(__file__).with_name("automodel_dependency.json")
_GENERATED_NAMES = {"INSTALLER", "RECORD", "REQUESTED"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_dependency_manifest(path: Path = _MANIFEST_PATH) -> dict[str, Any]:
    """Load the immutable AutoModel dependency declaration."""
    data = json.loads(path.read_text())
    required = {
        "distribution",
        "import_name",
        "package_file_count",
        "package_tree_sha256",
        "release_commit",
        "release_tag",
        "runtime_versions",
        "version",
        "wheel",
        "wheel_sha256",
    }
    if set(data) != required:
        raise RuntimeError(
            f"AutoModel dependency manifest keys mismatch: expected={sorted(required)}, "
            f"actual={sorted(data)}."
        )
    return data


def _distribution_root(
    distribution: importlib.metadata.Distribution, manifest: dict[str, Any]
) -> Path:
    root = Path(distribution.locate_file("")).resolve()
    dist_info = root / f"{manifest['distribution']}-{manifest['version']}.dist-info"
    if not dist_info.is_dir() or not dist_info.name.endswith(".dist-info"):
        raise RuntimeError(f"AutoModel has no regular wheel dist-info directory: {dist_info}.")
    return root


def _package_files(root: Path, manifest: dict[str, Any]) -> list[Path]:
    package_root = root / manifest["import_name"]
    dist_info_root = root / f"{manifest['distribution']}-{manifest['version']}.dist-info"
    if not package_root.is_dir() or not dist_info_root.is_dir():
        raise RuntimeError(
            "AutoModel package or exact-version dist-info directory is missing from the "
            f"installed distribution root {root}."
        )

    files: list[Path] = []
    for base in (package_root, dist_info_root):
        for candidate in base.rglob("*"):
            if candidate.is_symlink():
                raise RuntimeError(f"AutoModel distribution contains a symlink: {candidate}.")
            if not candidate.is_file():
                continue
            if "__pycache__" in candidate.parts or candidate.suffix == ".pyc":
                continue
            if candidate.name in _GENERATED_NAMES:
                continue
            files.append(candidate)
    return sorted(files, key=lambda path: path.relative_to(root).as_posix())


def snapshot_installed_distribution() -> dict[str, Any]:
    """Return a deterministic content snapshot after enforcing the frozen wheel tree."""
    manifest = load_dependency_manifest()
    distribution = importlib.metadata.distribution(manifest["distribution"])
    if distribution.version != manifest["version"]:
        raise RuntimeError(
            f"PDD requires {manifest['distribution']}=={manifest['version']}, "
            f"found {distribution.version}."
        )

    runtime_versions = {
        name: importlib.metadata.version(name) for name in manifest["runtime_versions"]
    }
    if runtime_versions != manifest["runtime_versions"]:
        raise RuntimeError(
            "PDD runtime dependency versions mismatch: "
            f"expected {manifest['runtime_versions']}, found {runtime_versions}."
        )

    root = _distribution_root(distribution, manifest)
    direct_url_text = distribution.read_text("direct_url.json")
    if direct_url_text is not None:
        direct_url = json.loads(direct_url_text)
        if direct_url.get("dir_info", {}).get("editable", False):
            raise RuntimeError("PDD rejects editable AutoModel installations.")

    spec = importlib.util.find_spec(manifest["import_name"])
    if spec is None or spec.origin is None:
        raise RuntimeError(f"Cannot resolve import {manifest['import_name']!r}.")
    import_origin = Path(spec.origin).resolve()
    try:
        import_origin.relative_to(root)
    except ValueError as error:
        raise RuntimeError(
            f"AutoModel import {import_origin} is shadowing distribution root {root}."
        ) from error
    files = _package_files(root, manifest)
    file_records: list[dict[str, Any]] = []
    tree_digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(root).as_posix()
        digest = _sha256(path)
        size = path.stat().st_size
        tree_digest.update(relative.encode())
        tree_digest.update(b"\0")
        tree_digest.update(digest.encode())
        tree_digest.update(b"\0")
        tree_digest.update(str(size).encode())
        tree_digest.update(b"\n")
        file_records.append({"path": relative, "sha256": digest, "size": size})

    actual_tree_digest = tree_digest.hexdigest()
    if len(file_records) != manifest["package_file_count"]:
        raise RuntimeError(
            "AutoModel package file count does not match the frozen wheel: "
            f"expected {manifest['package_file_count']}, found {len(file_records)}."
        )
    if actual_tree_digest != manifest["package_tree_sha256"]:
        raise RuntimeError(
            "AutoModel package tree does not match the frozen official wheel: "
            f"expected {manifest['package_tree_sha256']}, found {actual_tree_digest}."
        )

    return {
        "distribution": manifest["distribution"],
        "files": file_records,
        "import_origin": str(import_origin),
        "package_file_count": len(file_records),
        "package_tree_sha256": actual_tree_digest,
        "release_commit": manifest["release_commit"],
        "release_tag": manifest["release_tag"],
        "root": str(root),
        "runtime_versions": runtime_versions,
        "version": distribution.version,
        "wheel": manifest["wheel"],
        "wheel_sha256": manifest["wheel_sha256"],
    }


def write_snapshot(output: Path) -> None:
    """Atomically write a distribution snapshot outside the installation."""
    snapshot = snapshot_installed_distribution()
    output = output.resolve()
    root = Path(snapshot["root"])
    try:
        output.relative_to(root)
    except ValueError:
        pass
    else:
        raise ValueError("Snapshot output must be outside the AutoModel distribution.")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    snapshot = subparsers.add_parser("snapshot", help="verify and write a content snapshot")
    snapshot.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.command == "snapshot":
        write_snapshot(args.output)


if __name__ == "__main__":
    main()
