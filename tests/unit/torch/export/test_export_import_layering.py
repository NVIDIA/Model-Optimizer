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

"""Pins the import layering of the unified HF export package.

The package is a DAG: ``hf_export_prep`` and ``hf_weight_export`` are leaves, the
exporters sit above them, and ``unified_export_hf`` dispatches to the exporters. That
invariant is what lets every module import its helpers directly instead of lazily, and
nothing about running the package catches a regression: ``export/__init__.py`` imports
its submodules in one fixed order, so a reintroduced cycle resolves under that order and
stays invisible. ``import modelopt.torch.export.<submodule>`` does not probe a different
order either — Python runs the package initializer first, so every submodule import
replays that same sequence.

So read the graph instead of running it: parse the module-scope intra-package imports out
of each source file and assert the shape directly. Only module-scope imports can form an
import-time cycle, which is exactly what the parse sees.
"""

import ast
from pathlib import Path

import modelopt.torch.export

EXPORT_DIR = Path(modelopt.torch.export.__file__).parent

# Modules that must not grow a dependency on any exporter — the leaves the split created.
LEAF_MODULES = ["hf_export_prep", "hf_weight_export"]

EXPORTERS = [
    "unified_export_diffusers",
    "unified_export_hf",
    "unified_export_hf_streaming",
]


def _relative_imports(tree: ast.AST, level: int, *, module_scope_only: bool) -> set[str]:
    """Sibling module names pulled in by ``from .<name> import ...`` / ``from . import <name>``.

    ``level`` selects how far up the relative import points: 1 for a file directly in the
    export package, 2 for one in a subpackage such as ``plugins``. With
    ``module_scope_only``, function and class bodies are skipped, leaving only the imports
    that run at import time.
    """
    names: set[str] = set()
    stack = list(ast.iter_child_nodes(tree))
    while stack:
        node = stack.pop()
        if module_scope_only and isinstance(
            node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef
        ):
            continue
        if isinstance(node, ast.ImportFrom) and node.level == level:
            if node.module:
                names.add(node.module.split(".")[0])
            else:
                # `from . import x, y` — the imported names are the sibling modules.
                names.update(alias.name for alias in node.names)
        stack.extend(ast.iter_child_nodes(node))
    return names


def _import_graph() -> dict[str, set[str]]:
    """Map each export module to the sibling export modules it imports at module scope.

    ``plugins`` is one node, carrying the union of what its files import back out of the
    export package, so a module-scope cycle routed through a plugin still shows up.
    """
    plugins_dir = EXPORT_DIR / "plugins"
    nodes = {p.stem for p in EXPORT_DIR.glob("*.py")} - {"__init__"}
    if plugins_dir.is_dir():
        nodes.add("plugins")

    graph: dict[str, set[str]] = {}
    for name in sorted(nodes - {"plugins"}):
        tree = ast.parse((EXPORT_DIR / f"{name}.py").read_text())
        graph[name] = _relative_imports(tree, 1, module_scope_only=True) & nodes

    if "plugins" in nodes:
        edges: set[str] = set()
        for path in sorted(plugins_dir.glob("*.py")):
            tree = ast.parse(path.read_text())
            edges |= _relative_imports(tree, 2, module_scope_only=True)
        graph["plugins"] = edges & nodes
    return graph


def _find_cycle(graph: dict[str, set[str]]) -> list[str] | None:
    """Return one import cycle as a node path, or None if the graph is acyclic."""
    visiting: set[str] = set()
    done: set[str] = set()

    def walk(node: str, path: list[str]) -> list[str] | None:
        visiting.add(node)
        path.append(node)
        for child in sorted(graph.get(node, ())):
            if child in visiting:
                return [*path[path.index(child) :], child]
            if child not in done:
                cycle = walk(child, path)
                if cycle:
                    return cycle
        path.pop()
        visiting.discard(node)
        done.add(node)
        return None

    for node in sorted(graph):
        if node not in done:
            cycle = walk(node, [])
            if cycle:
                return cycle
    return None


def test_export_import_graph_is_acyclic():
    """No export module imports, directly or transitively, a module that imports it back."""
    cycle = _find_cycle(_import_graph())
    assert cycle is None, "import cycle in modelopt.torch.export: " + " -> ".join(cycle or [])


def test_leaf_modules_never_import_an_exporter():
    """The two leaves stay leaves, at module scope and inside functions alike.

    A function-local import of an exporter would not deadlock the interpreter, so the
    acyclicity check above cannot see it — but it is the exact cycle-dodging shape the
    split removed, so pin its absence too.
    """
    for name in LEAF_MODULES:
        tree = ast.parse((EXPORT_DIR / f"{name}.py").read_text())
        imported = _relative_imports(tree, 1, module_scope_only=False)
        offenders = sorted(imported.intersection(EXPORTERS))
        assert not offenders, f"{name}.py must not import an exporter, but imports {offenders}"


def test_import_graph_covers_the_split_modules():
    """Guards the parse itself: a rename that empties the graph must not pass silently."""
    graph = _import_graph()
    for name in [*LEAF_MODULES, *EXPORTERS]:
        assert name in graph, f"{name} missing from the parsed export import graph"
    assert graph["unified_export_hf"].issuperset(
        {"unified_export_diffusers", "unified_export_hf_streaming"}
    ), "unified_export_hf should dispatch to the diffusers and streaming exporters"
    assert graph["unified_export_hf_streaming"].issuperset(LEAF_MODULES), (
        "unified_export_hf_streaming should build on both leaf modules"
    )
