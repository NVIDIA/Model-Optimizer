#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""User workspace management.

Each user gets a workspace root directory. Inside it, the agent creates
model/task-specific subdirectories (e.g., qwen3-0.6b/, llama-3.1-8b-fp8/).
Each subdirectory is a copy of the Model-Optimizer repo (no .git) that the
agent can freely modify.

The agent decides when to reuse an existing workspace vs create a fresh copy.
This module provides the copy utility and cleanup logic; the actual decision
is driven by skill instructions (see skills/common/workspace-management.md).

Layout:
    <workspace_root>/
        qwen3-0.6b/                  ← agent-created, reused across PTQ/deploy/eval
            .claude/skills/...
            .claude/clusters.yaml    ← injected from user config
            examples/...
            output/                  ← PTQ output checkpoint
        llama-3.1-8b-fp8/           ← different model, separate workspace
"""

import asyncio
import logging
import os
import shutil
import time
from pathlib import Path

logger = logging.getLogger(__name__)

# Max age for workspaces before auto-cleanup (seconds). 0 = never.
WORKSPACE_MAX_AGE = int(os.environ.get("WORKSPACE_MAX_AGE", str(30 * 24 * 3600)))  # 30 days

# Max workspaces per user. 0 = unlimited.
MAX_WORKSPACES_PER_USER = int(os.environ.get("MAX_WORKSPACES_PER_USER", "20"))

# Rsync excludes when copying the repo
COPY_EXCLUDES = [
    ".git",
    "__pycache__",
    "*.pyc",
    "*.pyo",
    "node_modules",
    "*.egg-info",
    ".tox",
    ".pytest_cache",
    "dist",
    "build",
    "*.sqsh",
    ".claude/clusters.yaml",  # per-user, never copy from repo
]


class WorkspaceManager:
    """Manages per-user workspace roots and repo copies."""

    def __init__(self, repo_dir: str | Path):
        """Initialize the workspace manager.

        Args:
            repo_dir: Path to the shared Model-Optimizer repo (upstream, read-only).
        """
        self._repo_dir = Path(repo_dir)
        if not self._repo_dir.exists():
            raise FileNotFoundError(f"Repo dir not found: {self._repo_dir}")

    @property
    def repo_dir(self) -> Path:
        """Return the upstream repo directory."""
        return self._repo_dir

    async def create_workspace(
        self,
        workspace_root: Path,
        name: str,
        clusters_yaml: str | None = None,
    ) -> Path:
        """Create a new workspace (fresh repo copy) for a model/task.

        Args:
            workspace_root: User's workspace root directory
            name: Workspace name (e.g., "qwen3-0.6b", "llama-3.1-8b-fp8")
            clusters_yaml: User's cluster config to inject

        Returns:
            Path to the created workspace directory.
        """
        workspace_root.mkdir(parents=True, exist_ok=True)

        # Enforce limit
        if MAX_WORKSPACES_PER_USER > 0:
            await self._enforce_limit(workspace_root)

        dest = workspace_root / name
        if dest.exists():
            logger.info("Workspace %s already exists, skipping copy", dest)
        else:
            await self._copy_repo(dest)

        # Remove any repo-level clusters.yaml (don't leak other users' config)
        claude_dir = dest / ".claude"
        repo_clusters = claude_dir / "clusters.yaml"
        if repo_clusters.exists():
            repo_clusters.unlink()

        # Inject user's own cluster config if they have one
        if clusters_yaml:
            claude_dir.mkdir(exist_ok=True)
            repo_clusters.write_text(clusters_yaml, encoding="utf-8")

        return dest

    async def ensure_default_workspace(
        self,
        workspace_root: Path,
        clusters_yaml: str | None = None,
    ) -> Path:
        """Ensure there's at least one workspace (named 'default') for the user.

        Used when the agent starts — the agent can then create model-specific
        workspaces from within the session using the copy utility.
        """
        return await self.create_workspace(workspace_root, "default", clusters_yaml)

    def list_workspaces(self, workspace_root: Path) -> list[dict]:
        """List all workspaces for a user."""
        if not workspace_root.exists():
            return []
        result = []
        for entry in sorted(workspace_root.iterdir()):
            if not entry.is_dir():
                continue
            result.append(
                {
                    "name": entry.name,
                    "path": str(entry),
                    "modified": time.strftime(
                        "%Y-%m-%d %H:%M", time.localtime(entry.stat().st_mtime)
                    ),
                    "size_mb": self._dir_size_mb(entry),
                }
            )
        return result

    async def cleanup_old(self, workspace_root: Path, max_age_days: int | None = None) -> int:
        """Remove workspaces older than max_age_days. Returns count removed."""
        max_age_secs = (max_age_days * 86400) if max_age_days else WORKSPACE_MAX_AGE
        if max_age_secs <= 0 or not workspace_root.exists():
            return 0
        cutoff = time.time() - max_age_secs
        removed = 0
        for entry in sorted(workspace_root.iterdir()):
            if entry.is_dir() and entry.stat().st_mtime < cutoff:
                logger.info("Cleaning up old workspace: %s", entry)
                await asyncio.to_thread(shutil.rmtree, entry, ignore_errors=True)
                removed += 1
        return removed

    async def copy_repo_to(self, dest: Path) -> None:
        """Public interface for the agent to request a fresh repo copy."""
        await self._copy_repo(dest)

    # ── Internal ─────────────────────────────────────────────────────

    async def _copy_repo(self, dest: Path) -> None:
        dest.mkdir(parents=True, exist_ok=True)
        exclude_args = []
        for excl in COPY_EXCLUDES:
            exclude_args.extend(["--exclude", excl])

        cmd = [
            "rsync",
            "-a",
            "--quiet",
            *exclude_args,
            f"{self._repo_dir}/",
            f"{dest}/",
        ]
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(f"Failed to copy repo: {stderr.decode(errors='replace')}")
        logger.info("Copied repo to %s", dest)

    async def _enforce_limit(self, workspace_root: Path) -> None:
        if not workspace_root.exists():
            return
        dirs = sorted(
            [d for d in workspace_root.iterdir() if d.is_dir()],
            key=lambda d: d.stat().st_mtime,
        )
        while len(dirs) >= MAX_WORKSPACES_PER_USER:
            oldest = dirs.pop(0)
            logger.info("Removing oldest workspace to enforce limit: %s", oldest)
            await asyncio.to_thread(shutil.rmtree, oldest, ignore_errors=True)

    @staticmethod
    def _dir_size_mb(path: Path) -> float:
        total = 0
        try:
            for f in path.rglob("*"):
                if f.is_file():
                    total += f.stat().st_size
        except OSError:
            pass
        return round(total / (1024 * 1024), 1)
