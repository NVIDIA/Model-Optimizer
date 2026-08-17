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

import json
import subprocess
from importlib import metadata
from typing import Any
from urllib.parse import unquote, urlparse

__all__ = ["verify_installed_vcs_source"]

_GIT_TIMEOUT_SECONDS = 10


def _normalized_repository(url: object) -> str:
    return str(url or "").removesuffix(".git").rstrip("/")


def _installed_vcs_source(package: str) -> tuple[str | None, str | None]:
    try:
        distribution = metadata.distribution(package)
    except metadata.PackageNotFoundError as error:
        raise RuntimeError(f"Pinned Puzzletron dependency {package!r} is not installed") from error
    try:
        raw_payload = distribution.read_text("direct_url.json")
    except OSError as error:
        raise RuntimeError(
            f"Pinned Puzzletron dependency {package!r} direct URL metadata is unreadable"
        ) from error
    if raw_payload is None:
        raise RuntimeError(f"Pinned Puzzletron dependency {package!r} has no direct URL metadata")
    try:
        payload = json.loads(raw_payload)
    except (json.JSONDecodeError, TypeError) as error:
        raise RuntimeError(
            f"Pinned Puzzletron dependency {package!r} has malformed direct URL metadata"
        ) from error
    if not isinstance(payload, dict):
        raise RuntimeError(
            f"Pinned Puzzletron dependency {package!r} has invalid direct URL metadata"
        )
    repository = payload.get("url")
    vcs_info = payload.get("vcs_info", {})
    dir_info = payload.get("dir_info", {})
    if not isinstance(repository, str) or not repository:
        raise RuntimeError(
            f"Pinned Puzzletron dependency {package!r} has invalid direct URL metadata"
        )
    if not isinstance(vcs_info, dict) or not isinstance(dir_info, dict):
        raise RuntimeError(
            f"Pinned Puzzletron dependency {package!r} has invalid direct URL metadata"
        )
    if "editable" in dir_info and not isinstance(dir_info["editable"], bool):
        raise RuntimeError(
            f"Pinned Puzzletron dependency {package!r} has invalid direct URL metadata"
        )
    commit = vcs_info.get("commit_id")
    if commit is not None:
        if not isinstance(commit, str) or not commit:
            raise RuntimeError(
                f"Pinned Puzzletron dependency {package!r} has invalid direct URL metadata"
            )
        return repository, commit
    if dir_info.get("editable") and repository.startswith("file:"):
        root = unquote(urlparse(repository).path)
        try:
            git_outputs = [
                subprocess.check_output(
                    ["git", "-C", root, "remote", "get-url", "origin"],
                    text=True,
                    timeout=_GIT_TIMEOUT_SECONDS,
                ),
                subprocess.check_output(
                    ["git", "-C", root, "rev-parse", "HEAD"],
                    text=True,
                    timeout=_GIT_TIMEOUT_SECONDS,
                ),
                subprocess.check_output(
                    ["git", "-C", root, "status", "--porcelain", "--untracked-files=all"],
                    text=True,
                    timeout=_GIT_TIMEOUT_SECONDS,
                ),
            ]
        except (OSError, subprocess.SubprocessError) as error:
            raise RuntimeError(
                f"Pinned Puzzletron dependency {package!r} editable source could not be inspected"
            ) from error
        repository, commit, dirty = (output.strip() for output in git_outputs)
        if dirty:
            raise RuntimeError(f"Pinned Puzzletron dependency {package!r} is dirty: {dirty}")
        return repository, commit
    return repository, None


def verify_installed_vcs_source(package: str, expected: dict[str, Any]) -> None:
    """Require an installed VCS dependency to match its repository and commit."""

    repository, commit = _installed_vcs_source(package)
    expected_source = (_normalized_repository(expected["repository"]), expected["commit"])
    actual_source = (_normalized_repository(repository), commit)
    if actual_source != expected_source:
        raise RuntimeError(
            f"Pinned Puzzletron dependency {package!r} source mismatch: "
            f"actual={actual_source!r}, expected={expected_source!r}"
        )
