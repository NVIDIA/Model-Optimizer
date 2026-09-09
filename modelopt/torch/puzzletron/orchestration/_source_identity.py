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

"""Source identity and worker checkout validation for resolved run bundles."""

from __future__ import annotations

import asyncio
import hashlib
import os
import re
import shutil
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping

    from ._recipe_inputs import Site

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
_SOURCE_PATHSPECS = (".", ":(exclude,attr:filter=lfs)")

_STANDALONE_SOURCE_GUARD = r"""
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

repository = Path(sys.argv[1]).expanduser().resolve()
expected = json.loads(sys.argv[2])
pathspecs = (".", ":(exclude,attr:filter=lfs)")


def git_output(*args):
    executable = shutil.which("git")
    if executable is None:
        raise OSError("git executable not found")
    result = subprocess.run(
        [executable, *args],
        cwd=repository,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout


def packaged_revision():
    if repository.parent.name != "src":
        return None
    marker = repository.parent.parent / "modelopt_revision"
    try:
        revision = marker.read_text().strip().lower()
    except OSError:
        return None
    if not re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", revision):
        return None
    return {"revision": revision, "dirty": False}


def working_tree_fingerprint():
    digest = hashlib.sha256()
    digest.update(git_output("diff", "--binary", "HEAD", "--", *pathspecs))
    untracked = git_output("ls-files", "--others", "--exclude-standard", "-z").split(b"\0")
    for encoded in sorted(item for item in untracked if item):
        digest.update(b"\0path\0" + encoded + b"\0")
        candidate = repository / os.fsdecode(encoded)
        if candidate.is_symlink():
            digest.update(b"symlink\0" + os.fsencode(os.readlink(candidate)))
            continue
        with candidate.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def repository_revision():
    try:
        revision = git_output("rev-parse", "HEAD").decode().strip()
        dirty = bool(git_output("status", "--porcelain", "--", *pathspecs).strip())
    except (OSError, subprocess.CalledProcessError):
        return packaged_revision() or {"revision": None, "dirty": None}
    code = {"revision": revision, "dirty": dirty}
    if dirty:
        try:
            code["working_tree_sha256"] = working_tree_fingerprint()
        except (OSError, subprocess.CalledProcessError):
            code["working_tree_sha256"] = None
    return code


def _assert_worker_source():
    actual = repository_revision()
    if actual.get("revision") != expected.get("revision"):
        raise RuntimeError(
            "Worker source revision changed after the run bundle was resolved: "
            f"expected {expected.get('revision')}, got {actual.get('revision')}"
        )
    expected_dirty = expected.get("dirty")
    if expected_dirty is None:
        expected_dirty = False
    if actual.get("dirty") != expected_dirty:
        raise RuntimeError(
            "Worker source state changed after the run bundle was resolved: "
            f"expected dirty={expected_dirty}, got dirty={actual.get('dirty')}"
        )
    if expected_dirty and actual.get("working_tree_sha256") != expected.get(
        "working_tree_sha256"
    ):
        raise RuntimeError("Worker source contents changed after the run bundle was resolved")


_assert_worker_source()
""".strip()


async def _git_output(repository: Path, *args: str) -> bytes:
    """Run one fixed-argument Git query without involving a shell."""

    executable = shutil.which("git")
    if executable is None:
        raise OSError("git executable not found")
    process = await asyncio.create_subprocess_exec(
        executable,
        *args,
        cwd=repository,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate()
    if process.returncode:
        raise RuntimeError(stderr.decode(errors="replace").strip() or "Git query failed")
    return stdout


def _run_git(repository: Path, *args: str) -> bytes:
    """Run a Git query from the synchronous configuration API."""

    return asyncio.run(_git_output(repository, *args))


def standalone_source_guard() -> str:
    """Return the dependency-free worker identity guard sealed into run bundles."""

    return _STANDALONE_SOURCE_GUARD


def _packaged_revision(repository: Path) -> dict[str, Any] | None:
    """Read the revision baked beside the source tree in Puzzletron worker images."""

    if repository.parent.name != "src":
        return None
    marker = repository.parent.parent / "modelopt_revision"
    try:
        revision = marker.read_text().strip().lower()
    except OSError:
        return None
    if not re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", revision):
        return None
    return {"revision": revision, "dirty": False}


def working_tree_fingerprint(repository: Path = REPOSITORY_ROOT) -> str:
    """Hash tracked source changes and untracked files, excluding LFS materialization."""

    digest = hashlib.sha256()
    tracked = _run_git(repository, "diff", "--binary", "HEAD", "--", *_SOURCE_PATHSPECS)
    digest.update(tracked)
    untracked = _run_git(repository, "ls-files", "--others", "--exclude-standard", "-z").split(
        b"\0"
    )
    for encoded in sorted(item for item in untracked if item):
        digest.update(b"\0path\0" + encoded + b"\0")
        candidate = repository / os.fsdecode(encoded)
        if candidate.is_symlink():
            digest.update(b"symlink\0" + os.fsencode(os.readlink(candidate)))
            continue
        with candidate.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def repository_revision(repository: Path) -> dict[str, Any]:
    """Return an immutable revision plus a dirty-tree fingerprint when available."""

    try:
        revision = _run_git(repository, "rev-parse", "HEAD").decode().strip()
        dirty = bool(
            _run_git(repository, "status", "--porcelain", "--", *_SOURCE_PATHSPECS).strip()
        )
    except (OSError, RuntimeError):
        return _packaged_revision(repository) or {"revision": None, "dirty": None}
    code = {"revision": revision, "dirty": dirty}
    if dirty:
        try:
            code["working_tree_sha256"] = working_tree_fingerprint(repository)
        except (OSError, RuntimeError):
            code["working_tree_sha256"] = None
    return code


@lru_cache(maxsize=1)
def code_revision() -> dict[str, Any]:
    """Return the controller checkout identity."""

    return repository_revision(REPOSITORY_ROOT)


def worker_code(
    site: Site,
    controller_code: Mapping[str, Any],
    *,
    detect_revision=None,
) -> dict[str, Any]:
    """Resolve and validate the worker checkout identity selected by a site."""

    environment = dict(site.body["environment"])
    repository = Path(str(environment["repository"])).expanduser().resolve()
    explicit = environment.get("source_revision")
    if explicit and not re.fullmatch(r"(?:[0-9a-fA-F]{40}|[0-9a-fA-F]{64})", str(explicit)):
        raise ValueError("site.environment.source_revision must be a full immutable Git commit")
    if explicit:
        explicit = str(explicit).lower()
    detector = detect_revision or repository_revision
    detected = dict(controller_code) if repository == REPOSITORY_ROOT else detector(repository)
    detected_revision = detected.get("revision")
    if explicit and detected_revision and explicit != detected_revision:
        raise ValueError(
            "site.environment.source_revision does not match the configured worker "
            f"repository HEAD ({detected_revision})"
        )
    if detected_revision:
        if detected.get("dirty") and not detected.get("working_tree_sha256"):
            raise ValueError(
                "The configured worker repository is dirty but could not be fingerprinted"
            )
        return {**detected, "source": "detected from site.environment.repository"}
    if explicit:
        return {
            "revision": explicit,
            "dirty": None,
            "source": "site.environment.source_revision",
        }
    raise ValueError(
        "Cannot determine the worker source revision from site.environment.repository; "
        "set site.environment.source_revision to its full Git commit"
    )


def assert_worker_source(
    repository: str,
    expected: Mapping[str, Any],
    *,
    detect_revision=None,
) -> None:
    """Reject a worker checkout that no longer matches its sealed identity."""

    detector = detect_revision or repository_revision
    actual = detector(Path(repository).expanduser().resolve())
    if actual.get("revision") != expected.get("revision"):
        raise RuntimeError(
            "Worker source revision changed after the run bundle was resolved: "
            f"expected {expected.get('revision')}, got {actual.get('revision')}"
        )
    expected_dirty = expected.get("dirty")
    actual_dirty = actual.get("dirty")
    if expected_dirty is None:
        expected_dirty = False
    if actual_dirty != expected_dirty:
        raise RuntimeError(
            "Worker source state changed after the run bundle was resolved: "
            f"expected dirty={expected_dirty}, got dirty={actual_dirty}"
        )
    if expected_dirty and actual.get("working_tree_sha256") != expected.get("working_tree_sha256"):
        raise RuntimeError("Worker source contents changed after the run bundle was resolved")
