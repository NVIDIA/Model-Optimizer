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


def _normalized_repository(url: object) -> str:
    return str(url or "").removesuffix(".git").rstrip("/")


def _installed_vcs_source(package: str) -> tuple[str | None, str | None]:
    payload = json.loads(metadata.distribution(package).read_text("direct_url.json") or "{}")
    vcs_info = payload.get("vcs_info") or {}
    if vcs_info.get("commit_id"):
        return payload.get("url"), vcs_info["commit_id"]
    if (payload.get("dir_info") or {}).get("editable") and str(payload.get("url", "")).startswith(
        "file:"
    ):
        root = unquote(urlparse(payload["url"]).path)
        repository = subprocess.check_output(
            ["git", "-C", root, "remote", "get-url", "origin"], text=True
        ).strip()
        commit = subprocess.check_output(
            ["git", "-C", root, "rev-parse", "HEAD"], text=True
        ).strip()
        dirty = subprocess.check_output(
            ["git", "-C", root, "status", "--porcelain", "--untracked-files=all"],
            text=True,
        ).strip()
        if dirty:
            raise RuntimeError(f"Pinned Puzzletron dependency {package!r} is dirty: {dirty}")
        return repository, commit
    return payload.get("url"), vcs_info.get("commit_id")


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
