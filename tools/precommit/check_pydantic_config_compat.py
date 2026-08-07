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

"""Pre-commit hook: detect backward-incompatible pydantic config changes.

``ModeloptBaseConfig`` uses ``extra="forbid"``. That means a checkpoint saved
with an older schema fails to load if a field was removed or renamed without a
``validation_alias`` (or ``alias``) preserving the old name.

For each modified Python file, this hook AST-diffs the version at ``HEAD``
against the new (working tree) version. Any field present in a pydantic config
class in ``HEAD`` that is missing from the new version — and not preserved via
``alias=`` or ``validation_alias=`` on some other field — is flagged.

Recognized base classes (resolved transitively *within a single file*):

    BaseModel, ModeloptBaseConfig, ModeloptBaseRule, ModeloptBaseRuleConfig

See the "Pydantic config compatibility" section in ``CONTRIBUTING.md``.
"""

from __future__ import annotations

import argparse
import ast
import subprocess
import sys
from pathlib import Path

# Known pydantic base class names. A class is considered a pydantic config if
# it transitively (within the same file) inherits from one of these names.
_PYDANTIC_BASES = frozenset(
    {
        "BaseModel",
        "ModeloptBaseConfig",
        "ModeloptBaseRule",
        "ModeloptBaseRuleConfig",
    }
)

# Annotations that do NOT produce a pydantic field and should be ignored.
_NON_FIELD_ANNOTATIONS = frozenset({"ClassVar", "InitVar"})


def _git_show(path: str) -> str | None:
    """Return the contents of *path* at ``HEAD``, or ``None`` if absent."""
    try:
        result = subprocess.run(
            ["git", "show", f"HEAD:{path}"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return result.stdout


def _parse(source: str, filename: str) -> ast.Module | None:
    try:
        return ast.parse(source, filename=filename)
    except SyntaxError:
        return None


def _base_name(node: ast.expr) -> str | None:
    """Extract the leaf name of a base-class expression (``Foo`` or ``mod.Foo``)."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _annotation_name(node: ast.expr) -> str | None:
    """Return the top-level name of an annotation (``ClassVar[int]`` → ``ClassVar``)."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Subscript):
        return _annotation_name(node.value)
    return None


def _collect_pydantic_classes(tree: ast.Module) -> dict[str, ast.ClassDef]:
    """Return ``{class_name: ClassDef}`` for classes that look like pydantic configs."""
    all_classes: dict[str, ast.ClassDef] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            all_classes[node.name] = node

    memo: dict[str, bool] = {}

    def _is_pydantic(name: str, stack: set[str]) -> bool:
        if name in memo:
            return memo[name]
        if name in _PYDANTIC_BASES:
            return True
        cls = all_classes.get(name)
        if cls is None or name in stack:
            return False
        stack.add(name)
        result = False
        for base in cls.bases:
            base_name = _base_name(base)
            if base_name and _is_pydantic(base_name, stack):
                result = True
                break
        stack.remove(name)
        memo[name] = result
        return result

    return {name: cls for name, cls in all_classes.items() if _is_pydantic(name, set())}


def _is_pydantic_field(stmt: ast.AnnAssign) -> bool:
    """Return True if *stmt* declares a pydantic field (not ClassVar/InitVar)."""
    if not isinstance(stmt.target, ast.Name):
        return False
    if stmt.target.id.startswith("_"):
        # Private attributes aren't user-facing config keys.
        return False
    ann_name = _annotation_name(stmt.annotation)
    if ann_name in _NON_FIELD_ANNOTATIONS:
        return False
    return True


def _collect_fields(cls: ast.ClassDef) -> set[str]:
    """Collect the names of pydantic fields declared directly on *cls*."""
    return {
        stmt.target.id  # type: ignore[attr-defined]
        for stmt in cls.body
        if isinstance(stmt, ast.AnnAssign) and _is_pydantic_field(stmt)
    }


def _literal_alias_values(node: ast.expr) -> list[str]:
    """Extract string alias values from an ``alias=`` / ``validation_alias=`` expr.

    Recognizes:
        ``"literal"`` — single string
        ``AliasChoices("a", "b", ...)`` — any of the listed strings
        ``AliasPath("top", ...)`` — the top-level key
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return [node.value]
    if isinstance(node, ast.Call):
        func = _base_name(node.func)
        if func == "AliasChoices":
            return [
                arg.value
                for arg in node.args
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str)
            ]
        if func == "AliasPath" and node.args:
            first = node.args[0]
            if isinstance(first, ast.Constant) and isinstance(first.value, str):
                return [first.value]
    return []


def _collect_aliases(cls: ast.ClassDef) -> set[str]:
    """Collect every alias name referenced by any field on *cls*."""
    aliases: set[str] = set()
    for stmt in cls.body:
        if not (isinstance(stmt, ast.AnnAssign) and _is_pydantic_field(stmt)):
            continue
        if not isinstance(stmt.value, ast.Call):
            continue
        for kw in stmt.value.keywords:
            if kw.arg in {"alias", "validation_alias"}:
                aliases.update(_literal_alias_values(kw.value))
    return aliases


def _check_file(path: str) -> list[str]:
    """Return backward-compat errors for *path*, or an empty list if none."""
    try:
        new_source = Path(path).read_text(encoding="utf-8")
    except OSError:
        return []
    old_source = _git_show(path)
    if old_source is None:
        return []  # new file — nothing to compare against

    old_tree = _parse(old_source, f"HEAD:{path}")
    new_tree = _parse(new_source, path)
    if old_tree is None or new_tree is None:
        return []  # unparseable — other hooks will catch it

    old_classes = _collect_pydantic_classes(old_tree)
    new_classes = _collect_pydantic_classes(new_tree)

    errors: list[str] = []
    for cls_name, old_cls in old_classes.items():
        new_cls = new_classes.get(cls_name)
        if new_cls is None:
            # Class was removed or renamed. Renaming the class itself is also
            # breaking, but we don't catch that here — the mode system keys off
            # the class name in modelopt_state, and that's best-flagged by a
            # human reviewer.
            continue

        old_fields = _collect_fields(old_cls)
        new_fields = _collect_fields(new_cls)
        new_aliases = _collect_aliases(new_cls)

        for field_name in sorted(old_fields - new_fields):
            if field_name in new_aliases:
                continue
            errors.append(
                f"{path}: {cls_name}.{field_name}: field was removed or renamed "
                "without a validation_alias preserving the old name. This breaks "
                'loading checkpoints saved with the previous schema (ModeloptBaseConfig uses extra="forbid"). '
                "See CONTRIBUTING.md, section 'Pydantic config compatibility'."
            )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="*", help="Files to check (passed by pre-commit).")
    args = parser.parse_args()

    errors: list[str] = []
    for f in args.files:
        if not f.endswith(".py"):
            continue
        if not Path(f).is_file():
            continue
        errors.extend(_check_file(f))

    if errors:
        print("Backward-incompatible pydantic config change(s) detected:", file=sys.stderr)
        print("", file=sys.stderr)
        for e in errors:
            print(f"  {e}", file=sys.stderr)
        print("", file=sys.stderr)
        print(
            "To rename a field while preserving backward compatibility, keep the old name reachable "
            "via validation_alias (or pydantic.AliasChoices). See CONTRIBUTING.md for guidance.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
