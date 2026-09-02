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

"""Verification helpers for the pinned Puzzletron CI environment."""

from __future__ import annotations

import hashlib
import json
import subprocess
from importlib import metadata
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

__all__ = ["verify_installed_vcs_source", "verify_vcs_checkout"]


def _normalized_repository(url: object) -> str:
    return str(url or "").removesuffix(".git").rstrip("/")


def verify_vcs_checkout(checkout: Path, package: str, expected: dict[str, Any]) -> str:
    """Verify the exact revision and optional tracked compatibility patch of a checkout."""
    repository = subprocess.check_output(
        ["git", "-C", str(checkout), "remote", "get-url", "origin"], text=True
    ).strip()
    commit = subprocess.check_output(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
    ).strip()
    expected_source = (_normalized_repository(expected["repository"]), expected["commit"])
    actual_source = (_normalized_repository(repository), commit)
    if actual_source != expected_source:
        raise RuntimeError(
            f"Pinned Puzzletron dependency {package!r} source mismatch: "
            f"actual={actual_source!r}, expected={expected_source!r}"
        )

    status = subprocess.check_output(
        ["git", "-C", str(checkout), "status", "--porcelain", "--untracked-files=all"],
        text=True,
    )
    patch_sha256 = expected.get("compatibility_patch_sha256")
    if patch_sha256 is None:
        if status:
            raise RuntimeError(
                f"Pinned Puzzletron dependency {package!r} is dirty: {status.strip()}"
            )
        return commit

    expected_files = sorted(str(path) for path in expected.get("compatibility_patch_files", ()))
    actual_files = sorted(line[3:] for line in status.splitlines() if len(line) >= 4)
    if not expected_files or actual_files != expected_files:
        raise RuntimeError(
            f"Pinned Puzzletron dependency {package!r} compatibility patch files differ: "
            f"actual={actual_files!r}, expected={expected_files!r}"
        )
    diff = subprocess.check_output(
        ["git", "-C", str(checkout), "diff", "--binary", "HEAD"], text=True
    )
    actual_patch_sha256 = hashlib.sha256(diff.encode()).hexdigest()
    if actual_patch_sha256 != patch_sha256:
        raise RuntimeError(
            f"Pinned Puzzletron dependency {package!r} compatibility patch differs: "
            f"actual={actual_patch_sha256}, expected={patch_sha256}"
        )
    return commit


def _installed_vcs_source(package: str, expected: dict[str, Any]) -> tuple[str | None, str | None]:
    payload = json.loads(metadata.distribution(package).read_text("direct_url.json") or "{}")
    vcs_info = payload.get("vcs_info") or {}
    if vcs_info.get("commit_id"):
        if expected.get("compatibility_patch_sha256"):
            raise RuntimeError(
                f"Pinned Puzzletron dependency {package!r} requires a verified editable "
                "compatibility-patched checkout"
            )
        return payload.get("url"), vcs_info["commit_id"]
    if (payload.get("dir_info") or {}).get("editable") and str(payload.get("url", "")).startswith(
        "file:"
    ):
        root = unquote(urlparse(payload["url"]).path)
        commit = verify_vcs_checkout(Path(root), package, expected)
        return expected["repository"], commit
    return payload.get("url"), vcs_info.get("commit_id")


def verify_installed_vcs_source(package: str, expected: dict[str, Any]) -> None:
    """Require an installed VCS dependency to match its repository and commit."""

    repository, commit = _installed_vcs_source(package, expected)
    expected_source = (_normalized_repository(expected["repository"]), expected["commit"])
    actual_source = (_normalized_repository(repository), commit)
    if actual_source != expected_source:
        raise RuntimeError(
            f"Pinned Puzzletron dependency {package!r} source mismatch: "
            f"actual={actual_source!r}, expected={expected_source!r}"
        )
