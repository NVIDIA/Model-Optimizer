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

"""Path resolution for a portable, contained FastGen dataset cache."""

from __future__ import annotations

from pathlib import Path

__all__ = ["resolve_cache_root", "resolve_under_root"]


def _existing_directory(path: str | Path, label: str) -> Path:
    resolved = Path(path).resolve(strict=True)
    if not resolved.is_dir():
        raise NotADirectoryError(f"{label} is not a directory: {resolved}")
    return resolved


def resolve_cache_root(configured_root: str | Path) -> Path:
    """Resolve the configured cache root to an existing directory."""
    return _existing_directory(configured_root, "configured cache root")


def resolve_under_root(root: str | Path, candidate: str | Path, label: str) -> Path:
    """Resolve an existing path and require its canonical target to remain beneath ``root``."""
    canonical_root = _existing_directory(root, "cache root")
    candidate_path = Path(candidate)
    unresolved = candidate_path if candidate_path.is_absolute() else canonical_root / candidate_path
    resolved = unresolved.resolve(strict=True)
    try:
        resolved.relative_to(canonical_root)
    except ValueError as exc:
        raise ValueError(
            f"{label} resolves outside cache root {canonical_root}: {candidate!s}"
        ) from exc
    return resolved
