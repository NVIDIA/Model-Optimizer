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

"""Pre-commit hook: read ``MODELOPT_*`` environment variables only through ``modelopt/env.py``.

``modelopt/env.py`` holds one documented accessor per ``MODELOPT_*`` variable ModelOpt reads. The hook
fails on an ``os.environ`` or ``os.getenv`` access to a ``MODELOPT_*`` name in any other Python file
under ``modelopt/``. Under ``examples/`` it fails only on a name ``modelopt/env.py`` reads: an example
keeps its own variables, but reads the library's through their accessors. A name is recognized as a
string literal, the leading text of an f-string, or a module-level string constant.
"""

import ast
import re
import sys
from collections.abc import Iterator
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_ENV_MODULE = _REPO_ROOT / "modelopt" / "env.py"
_PREFIX = "MODELOPT_"
# os.environ methods and os functions whose first argument is a variable's name.
_ENVIRON_METHODS = {"get", "pop", "setdefault"}
_OS_FUNCTIONS = {"getenv", "putenv", "unsetenv"}


def _is_os(node: ast.expr) -> bool:
    return isinstance(node, ast.Name) and node.id == "os"


def _is_environ(node: ast.expr) -> bool:
    """``os.environ``, or ``environ`` imported from ``os``."""
    if isinstance(node, ast.Attribute):
        return node.attr == "environ" and _is_os(node.value)
    return isinstance(node, ast.Name) and node.id == "environ"


def _is_env_function(func: ast.expr) -> bool:
    """``os.environ.get`` and the like, or ``os.getenv`` and the like, also imported from ``os``."""
    if isinstance(func, ast.Attribute):
        return (func.attr in _ENVIRON_METHODS and _is_environ(func.value)) or (
            func.attr in _OS_FUNCTIONS and _is_os(func.value)
        )
    return isinstance(func, ast.Name) and func.id in _OS_FUNCTIONS


def _accessed_names(tree: ast.AST) -> Iterator[ast.expr]:
    """The name expression of every environment access in ``tree``."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and node.args and _is_env_function(node.func):
            yield node.args[0]
        elif isinstance(node, ast.Subscript) and _is_environ(node.value):
            yield node.slice
        elif (
            isinstance(node, ast.Compare)
            and isinstance(node.ops[0], (ast.In, ast.NotIn))
            and _is_environ(node.comparators[0])
        ):
            yield node.left


def _string_constants(tree: ast.Module) -> dict[str, str]:
    """Module-level ``NAME = "text"`` assignments, to see through ``os.environ.get(NAME)``."""
    constants = {}
    for node in tree.body:
        if isinstance(node, ast.Assign | ast.AnnAssign) and isinstance(node.value, ast.Constant):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name) and isinstance(node.value.value, str):
                    constants[target.id] = node.value.value
    return constants


def _text(name: ast.expr, constants: dict[str, str]) -> str | None:
    """The accessed name, or an f-string's leading text; ``None`` when it is not known statically."""
    if isinstance(name, ast.Constant) and isinstance(name.value, str):
        return name.value
    if isinstance(name, ast.JoinedStr) and name.values and isinstance(name.values[0], ast.Constant):
        return str(name.values[0].value)
    if isinstance(name, ast.Name):
        return constants.get(name.id)
    return None


def _violations(path: Path, library_knobs: set[str]) -> list[str]:
    in_library = path.relative_to(_REPO_ROOT).parts[0] == "modelopt"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    constants = _string_constants(tree)
    found = []
    for name in _accessed_names(tree):
        text = _text(name, constants)
        if text and text.startswith(_PREFIX) and (in_library or text in library_knobs):
            found.append(
                f"{path.relative_to(_REPO_ROOT)}:{name.lineno}: reads {text} from the environment; "
                "read it through its accessor in modelopt/env.py, adding one there if it is new."
            )
    return found


def main() -> int:
    """Check the Python files pre-commit passes in, exit 1 on a violation."""
    library_knobs = set(re.findall(r"[\"'](MODELOPT_[A-Z0-9_]+)[\"']", _ENV_MODULE.read_text()))
    errors: list[str] = []
    for arg in sys.argv[1:]:
        path = Path(arg).resolve()
        if path.suffix == ".py" and path != _ENV_MODULE:
            errors.extend(_violations(path, library_knobs))
    for error in errors:
        print(f"ERROR: {error}", file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
