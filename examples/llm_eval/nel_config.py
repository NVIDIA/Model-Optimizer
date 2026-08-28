#!/usr/bin/env python3
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

"""Compile canonical text benchmark tasks into a NeMo Evaluator config."""

from __future__ import annotations

import argparse
import copy
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, TextIO
from urllib.parse import urlsplit

import yaml

__all__ = [
    "DEFAULT_TASKS",
    "SUPPORTED_TASKS",
    "add_compiler_arguments",
    "build_parser",
    "compile_nel_config",
    "load_nel_config",
    "main",
    "write_nel_config",
]


_TASK_CATALOG_PATH = Path(__file__).with_name("task_contracts.yaml")


def _load_task_catalog(path: Path = _TASK_CATALOG_PATH) -> dict[str, Any]:
    with path.open(encoding="utf-8") as catalog_file:
        catalog = yaml.safe_load(catalog_file)
    if not isinstance(catalog, Mapping) or catalog.get("schema_version") != 1:
        raise ValueError(f"Unsupported task contract catalog: {path}")

    tasks = catalog.get("tasks")
    defaults = catalog.get("default_tasks")
    if not isinstance(tasks, Mapping) or not isinstance(defaults, list):
        raise ValueError(f"Invalid task contract catalog: {path}")

    normalized_tasks: dict[str, dict[str, Any]] = {}
    for name, entry in tasks.items():
        if not isinstance(name, str) or not isinstance(entry, Mapping):
            raise ValueError(f"Invalid task contract catalog entry: {name!r}")
        contract = entry.get("contract")
        bindings = entry.get("bindings", [])
        if not isinstance(contract, Mapping) or contract.get("name") != name:
            raise ValueError(f"Task contract name mismatch: {name!r}")
        if not isinstance(bindings, list) or not all(
            isinstance(item, Mapping) for item in bindings
        ):
            raise ValueError(f"Invalid task contract bindings: {name!r}")
        normalized_tasks[name] = copy.deepcopy(dict(entry))

    if not all(isinstance(name, str) and name in normalized_tasks for name in defaults):
        raise ValueError(f"Invalid default task list: {path}")
    return {"default_tasks": defaults.copy(), "tasks": normalized_tasks}


_TASK_CATALOG = _load_task_catalog()
_TASK_ENTRIES: dict[str, dict[str, Any]] = _TASK_CATALOG["tasks"]
DEFAULT_TASKS = tuple(_TASK_CATALOG["default_tasks"])
SUPPORTED_TASKS = tuple(_TASK_ENTRIES)


def _binding_options() -> dict[str, dict[str, Any]]:
    options: dict[str, dict[str, Any]] = {}
    for entry in _TASK_ENTRIES.values():
        for binding in entry.get("bindings", []):
            value_name = binding.get("value")
            option = binding.get("option")
            if not isinstance(value_name, str) or not isinstance(option, str):
                raise ValueError("Invalid task contract binding")
            definition = {
                key: binding[key]
                for key in ("option", "validator", "suffix", "maximum")
                if key in binding
            }
            existing = options.get(value_name)
            if existing is not None and existing != definition:
                raise ValueError(f"Conflicting task contract binding: {value_name!r}")
            options[value_name] = definition
    return options


_BINDING_OPTIONS = _binding_options()


def load_nel_config(path: Path) -> dict[str, Any]:
    """Load a YAML NEL base config and require a mapping at its root."""

    with path.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    if not isinstance(config, Mapping):
        raise ValueError(f"NEL base config must contain a mapping: {path}")
    return copy.deepcopy(dict(config))


def _indexed_tasks(tasks: list[Any]) -> dict[str, Mapping[str, Any]]:
    indexed: dict[str, Mapping[str, Any]] = {}
    for index, task in enumerate(tasks):
        if not isinstance(task, Mapping):
            raise ValueError(f"evaluation.tasks[{index}] must be a mapping")
        name = task.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(f"evaluation.tasks[{index}].name must be a non-empty string")
        if name in indexed:
            raise ValueError(f"evaluation.tasks contains duplicate task {name!r}")
        indexed[name] = task
    return indexed


def _require_option(value: str | None, option: str, task: str) -> str:
    if value is None or not value.strip():
        raise ValueError(f"{task} requires {option}")
    return value


def _require_endpoint(value: str | None, option: str, task: str, path_suffix: str) -> str:
    endpoint = _require_option(value, option, task)
    parsed = urlsplit(endpoint)
    try:
        parsed.port
    except ValueError as error:
        raise ValueError(f"{option} must be an HTTP(S) endpoint ending in {path_suffix}") from error
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or not parsed.path.rstrip("/").endswith(path_suffix)
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(f"{option} must be an HTTP(S) endpoint ending in {path_suffix}")
    return endpoint


def _require_parallelism(
    value: int | None, option: str, task: str, maximum: int | None = None
) -> int:
    if value is None:
        raise ValueError(f"{task} requires {option}")
    if value <= 0:
        raise ValueError(f"{option} must be greater than zero")
    if maximum is not None and value > maximum:
        raise ValueError(f"{option} must not exceed {maximum}")
    return value


def _set_contract_value(contract: dict[str, Any], path: object, value: object) -> None:
    if not isinstance(path, list) or not path or not all(isinstance(key, str) for key in path):
        raise ValueError("Task contract binding path must be a non-empty string list")
    cursor = contract
    for key in path[:-1]:
        nested = cursor.get(key)
        if not isinstance(nested, dict):
            raise ValueError(f"Task contract binding path does not exist: {'.'.join(path)}")
        cursor = nested
    if path[-1] not in cursor:
        raise ValueError(f"Task contract binding path does not exist: {'.'.join(path)}")
    cursor[path[-1]] = value


def _task_contracts(
    task_names: Sequence[str],
    values: Mapping[str, str | int | None],
) -> dict[str, dict[str, Any]]:
    contracts: dict[str, dict[str, Any]] = {}
    for task in task_names:
        entry = _TASK_ENTRIES[task]
        contract = copy.deepcopy(dict(entry["contract"]))
        for binding in entry.get("bindings", []):
            value_name = binding.get("value")
            option = binding.get("option")
            if not isinstance(value_name, str) or not isinstance(option, str):
                raise ValueError(f"Invalid task contract binding: {task!r}")
            raw_value = values.get(value_name)
            validator = binding.get("validator")
            if validator == "endpoint":
                if raw_value is not None and not isinstance(raw_value, str):
                    raise ValueError(f"Invalid endpoint binding: {task!r}")
                suffix = binding.get("suffix")
                if not isinstance(suffix, str):
                    raise ValueError(f"Invalid endpoint binding suffix: {task!r}")
                value: object = _require_endpoint(raw_value, option, task, suffix)
            elif validator == "positive_integer":
                if raw_value is not None and type(raw_value) is not int:
                    raise ValueError(f"Invalid parallelism binding: {task!r}")
                maximum = binding.get("maximum")
                if maximum is not None and not isinstance(maximum, int):
                    raise ValueError(f"Invalid parallelism binding maximum: {task!r}")
                value = _require_parallelism(raw_value, option, task, maximum)
            elif validator is None:
                if raw_value is not None and not isinstance(raw_value, str):
                    raise ValueError(f"Invalid string binding: {task!r}")
                value = _require_option(raw_value, option, task)
            else:
                raise ValueError(f"Unknown task contract binding validator: {validator!r}")
            _set_contract_value(contract, binding.get("path"), value)
        contracts[task] = contract
    return contracts


def compile_nel_config(
    base_config: Mapping[str, Any],
    task_names: Sequence[str] = DEFAULT_TASKS,
    binding_values: Mapping[str, str | int | None] | None = None,
) -> dict[str, Any]:
    """Return a copied NEL config containing the selected canonical task contracts.

    Existing unrelated tasks are preserved. An existing selected task is accepted
    only when it exactly matches the maintained contract, making repeated compilation
    idempotent without silently replacing local overrides.
    """

    if not isinstance(base_config, Mapping):
        raise ValueError("NEL base config must be a mapping")
    if isinstance(task_names, str):
        raise ValueError("task selection must be a sequence of task names")

    selected = tuple(task_names)
    if len(selected) != len(set(selected)):
        raise ValueError("task selection contains duplicates")
    unsupported = sorted(set(selected) - set(SUPPORTED_TASKS))
    if unsupported:
        raise ValueError(f"unsupported canonical task(s): {', '.join(unsupported)}")
    values = dict(binding_values or {})
    unknown_values = sorted(set(values) - set(_BINDING_OPTIONS))
    if unknown_values:
        raise ValueError(f"unknown task binding(s): {', '.join(unknown_values)}")
    selected_values = {
        binding["value"] for task in selected for binding in _TASK_ENTRIES[task].get("bindings", [])
    }
    unused_options = sorted(
        _BINDING_OPTIONS[name]["option"]
        for name, value in values.items()
        if value is not None and name not in selected_values
    )
    if unused_options:
        raise ValueError(f"task option(s) not used by selected tasks: {', '.join(unused_options)}")

    contracts = _task_contracts(selected, values)

    compiled = copy.deepcopy(dict(base_config))
    evaluation = compiled.setdefault("evaluation", {})
    if not isinstance(evaluation, dict):
        raise ValueError("evaluation must be a mapping")
    tasks = evaluation.setdefault("tasks", [])
    if not isinstance(tasks, list):
        raise ValueError("evaluation.tasks must be a list")

    indexed = _indexed_tasks(tasks)
    for name in selected:
        contract = contracts[name]
        existing = indexed.get(name)
        if existing is not None:
            if existing != contract:
                raise ValueError(
                    f"existing task {name!r} conflicts with the maintained canonical contract"
                )
            continue
        tasks.append(copy.deepcopy(contract))
    return compiled


def write_nel_config(
    config: Mapping[str, Any], output: Path | None, stream: TextIO | None = None
) -> None:
    """Write YAML to a new file, or to ``stream`` when no output path is given."""

    if output is None:
        yaml.safe_dump(dict(config), stream if stream is not None else sys.stdout, sort_keys=False)
        return
    with output.open("x", encoding="utf-8") as config_file:
        yaml.safe_dump(dict(config), config_file, sort_keys=False)


def add_compiler_arguments(parser: argparse.ArgumentParser) -> None:
    """Add canonical task-selection and binding arguments to ``parser``."""

    parser.add_argument(
        "--task",
        action="append",
        choices=SUPPORTED_TASKS,
        dest="tasks",
        help="Canonical task to add. Repeat for multiple tasks; defaults to fixed contracts.",
    )
    for value_name, binding in _BINDING_OPTIONS.items():
        parser.add_argument(
            binding["option"],
            dest=value_name,
            type=int if binding.get("validator") == "positive_integer" else str,
            help=f"Value required by task contracts using {binding['option']}.",
        )


def build_parser() -> argparse.ArgumentParser:
    """Build the non-launching NEL config compiler CLI."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", type=Path, required=True, help="Existing NEL YAML config.")
    add_compiler_arguments(parser)
    parser.add_argument(
        "--output",
        type=Path,
        help="New YAML path. Existing files are never overwritten; omit to print to stdout.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Compile and write selected task contracts, reporting input errors through argparse."""

    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        base_config = load_nel_config(args.base_config)
        compiled = compile_nel_config(
            base_config,
            args.tasks or DEFAULT_TASKS,
            {name: getattr(args, name) for name in _BINDING_OPTIONS},
        )
        write_nel_config(compiled, args.output)
    except (OSError, ValueError, yaml.YAMLError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
