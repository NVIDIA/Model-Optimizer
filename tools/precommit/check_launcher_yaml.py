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

"""Pre-commit hook: validate launcher YAML references to recipes and templates.

Scans every ``tools/launcher/examples/**/*.yaml`` file for path-bearing args
the launcher will pass to ``main.py``, and verifies the referenced files
exist (and load as recipes, when applicable):

* ``--config <path>`` — must resolve to a file; if the file lives under
  ``modelopt_recipes/``, ``load_recipe()`` is invoked to catch schema breakage.
* ``data.chat_template=<path>`` — must resolve to a file.

Path resolution mirrors how the launcher itself runs: paths starting with
``modules/Model-Optimizer/`` (the launcher's submodule symlink) resolve under
the repo root; bare paths resolve under ``tools/launcher/``.

The hook ignores the per-file paths pre-commit passes (``pass_filenames:
false``) and re-scans the full launcher YAML set on every invocation, so
recipe-side changes still trigger validation of all launcher references.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import yaml

# tools/precommit/<this>.py → repo root
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_LAUNCHER_DIR = _REPO_ROOT / "tools" / "launcher"
_LAUNCHER_EXAMPLES = _LAUNCHER_DIR / "examples"

# Launcher submodule symlink prefix (resolves to repo root via
# tools/launcher/modules/Model-Optimizer -> ../..).
_MODELOPT_PREFIX = "modules/Model-Optimizer/"


def _is_interpolated(value: str) -> bool:
    """Return True for values the hook can't validate statically.

    Covers launcher runtime interpolation (``<<global_vars.x>>``) and YAML
    placeholder strings like ``<path to data>``.
    """
    return "<<" in value or value.startswith("<")


def _resolve(path_str: str) -> Path | None:
    """Resolve a launcher-YAML path string to an absolute filesystem path.

    Returns None if the path uses runtime interpolation that we can't validate
    statically.
    """
    if _is_interpolated(path_str):
        return None
    if path_str.startswith(_MODELOPT_PREFIX):
        return _REPO_ROOT / path_str[len(_MODELOPT_PREFIX) :]
    return _LAUNCHER_DIR / path_str


def _extract_paths(args: list) -> list[tuple[str, str]]:
    """Return [(kind, path)] for path-bearing entries in a task's args list.

    ``kind`` is ``--config`` or ``chat_template`` (used in error messages).
    """
    out: list[tuple[str, str]] = []
    for arg in args:
        if not isinstance(arg, str):
            continue
        stripped = arg.strip()
        # ``--config <path>`` (single string, space-separated)
        m = re.match(r"^--config\s+(\S+)\s*$", stripped)
        if m:
            out.append(("--config", m.group(1)))
            continue
        # ``data.chat_template=<path>`` (and any other ``.chat_template=`` override)
        if ".chat_template=" in stripped:
            _, _, value = stripped.partition("=")
            out.append(("chat_template", value.strip()))
    return out


def _try_load_recipe(recipe_path: Path, source: Path) -> list[str]:
    """Invoke ``load_recipe`` for paths under ``modelopt_recipes/``.

    No-op if modelopt isn't installed (matches ``check_modelopt_recipes.py``
    behavior).
    """
    try:
        from modelopt.recipe.loader import load_recipe
    except ImportError:
        return []
    try:
        load_recipe(str(recipe_path))
    except Exception as exc:
        return [f"{source}: --config {recipe_path} failed to load: {exc}"]
    return []


def _scan_launcher_yaml(path: Path) -> list[str]:
    errors: list[str] = []
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return [f"{path}: failed to parse YAML: {exc}"]
    if not isinstance(data, dict):
        return []
    pipeline = data.get("pipeline")
    if not isinstance(pipeline, dict):
        return []

    for task in pipeline.values():
        if not isinstance(task, dict):
            continue
        args = task.get("args")
        if not isinstance(args, list):
            continue
        for kind, path_str in _extract_paths(args):
            resolved = _resolve(path_str)
            if resolved is None:
                continue
            if not resolved.is_file():
                errors.append(
                    f"{path}: {kind} path does not exist: {path_str!r} (resolved to {resolved})"
                )
                continue
            # Recipes under modelopt_recipes/ go through Pydantic validation.
            if kind == "--config" and "modelopt_recipes" in resolved.parts:
                errors.extend(_try_load_recipe(resolved, path))
    return errors


def main() -> int:
    """Validate every launcher YAML under examples/, exit 1 on errors."""
    if not _LAUNCHER_EXAMPLES.is_dir():
        return 0
    errors: list[str] = []
    for yaml_file in sorted(_LAUNCHER_EXAMPLES.rglob("*.yaml")):
        errors.extend(_scan_launcher_yaml(yaml_file))
    if errors:
        for e in errors:
            print(f"ERROR: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
