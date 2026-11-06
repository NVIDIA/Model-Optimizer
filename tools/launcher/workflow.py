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

"""Versioned workflow compilation for the ModelOpt launcher."""

import argparse
import copy
import dataclasses
import json
import os
import re
from pathlib import Path

import fiddle
import yaml
from modelopt_launcher.core import SandboxPipeline, SandboxTask
from modelopt_launcher.slurm_config import SlurmConfig, slurm_factory

__all__ = ["load_workflow", "workflow_cli"]

_PARAMETER = re.compile(r"\$\{parameters\.([A-Za-z_]\w*)\}")
_RESOURCES = {"nodes", "ntasks_per_node", "gpus_per_node", "time", "mem"}


class _WorkflowLoader(yaml.SafeLoader):
    def construct_mapping(self, node, deep=False):
        keys = [self.construct_object(key, deep=deep) for key, _ in node.value]
        if len(keys) != len(set(keys)):
            raise ValueError("Duplicate YAML mapping keys are not allowed")
        return super().construct_mapping(node, deep=deep)


def _read(path):
    loader = _WorkflowLoader(Path(path).read_text())
    try:
        return loader.get_single_data()
    finally:
        loader.dispose()


def _mapping(value, allowed=None, label="configuration"):
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{label} must be a mapping with string keys")
    if allowed is not None and set(value) - allowed:
        raise ValueError(f"Unsupported {label} fields: {sorted(set(value) - allowed)}")
    return value


def _expand(value, parameters):
    if isinstance(value, dict):
        return {key: _expand(item, parameters) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand(item, parameters) for item in value]
    if not isinstance(value, str):
        return value

    def lookup(match):
        name = match[1]
        if name not in parameters:
            raise ValueError(f"Unknown workflow parameter: {name}")
        return copy.deepcopy(parameters[name])

    match = _PARAMETER.fullmatch(value)
    if match:
        return lookup(match)
    return _PARAMETER.sub(lambda match: str(lookup(match)), value)


def load_workflow(path, profile=None, parameters=None):
    """Compile a modelopt/v2 YAML and optional profile into legacy launcher arguments.

    Parameter overrides are a mapping of declared names to JSON-compatible values.
    Compilation validates all task dependencies and performs no transfers or launches.
    """
    document = _mapping(
        _read(path),
        {"api_version", "name", "parameters", "inputs", "tasks", "outputs"},
        "workflow",
    )
    if document.get("api_version") != "modelopt/v2":
        raise ValueError("Expected api_version: modelopt/v2")
    name = document.get("name")
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z][\w-]*", name):
        raise ValueError("Workflow name must use letters, digits, '_' or '-'")
    defaults = _mapping(document.get("parameters", {}), label="parameters")
    overrides = _mapping(parameters or {}, label="parameter overrides")
    if set(overrides) - set(defaults):
        raise ValueError("Parameter overrides must refer to declared parameters")
    values = {**defaults, **overrides}
    if any(not re.fullmatch(r"[A-Za-z_]\w*", key) for key in values):
        raise ValueError("Parameter names must be identifiers")
    try:
        json.dumps(values, allow_nan=False)
    except (ValueError, TypeError) as exc:
        raise ValueError("Parameters must be JSON-compatible values") from exc
    document = _expand(document, values)
    settings = _mapping(
        _read(profile) if profile else {},
        {
            "executor",
            "user",
            "identity",
            "job_dir",
            "hf_local",
            "slurm",
            "asset_resolver",
            "asset_timeout",
        },
        "profile",
    )
    mode = settings.get("executor", "slurm")
    if mode not in ("slurm", "docker", "local"):
        raise ValueError("Supported executors are slurm, docker and local; Kubernetes is pending")
    if settings.get("asset_resolver") is not None and (
        not isinstance(settings["asset_resolver"], list)
        or not settings["asset_resolver"]
        or not all(isinstance(item, str) and item for item in settings["asset_resolver"])
    ):
        raise ValueError("asset_resolver must be a nonempty argv list")
    if (
        type(settings.get("asset_timeout", 900)) is not int
        or settings.get("asset_timeout", 900) <= 0
    ):
        raise ValueError("asset_timeout must be a positive integer")
    slurm = dataclasses.asdict(fiddle.build(slurm_factory()))
    slurm.update(_mapping(settings.get("slurm", {}), set(slurm), "profile slurm"))
    if mode == "local":
        slurm["host"] = "localhost"
        slurm.update(container=None, container_mounts=[], srun_args=[], gpus_per_node=None)
    tasks = []
    for task_name, declaration in _mapping(document.get("tasks"), label="tasks").items():
        fields = dict(
            _mapping(
                declaration,
                {
                    "script",
                    "inline",
                    "args",
                    "environment",
                    "reqs",
                    "reqs_file",
                    "depends_on",
                    "inputs",
                    "outputs",
                    "resources",
                    "container",
                },
                f"task {task_name}",
            )
        )
        if bool(fields.get("script")) == bool(fields.get("inline")):
            raise ValueError(f"{task_name}: set exactly one of script or inline")
        if fields.get("inline") and fields.get("args"):
            raise ValueError(f"{task_name}: inline does not accept separate args")
        for key in ("script", "inline", "reqs", "reqs_file"):
            if key in fields and not isinstance(fields[key], str):
                raise ValueError(f"{task_name}: {key} must be a string")
        args = fields.get("args", [])
        if not isinstance(args, list) or not all(isinstance(arg, str) for arg in args):
            raise ValueError(f"{task_name}: args must be a list of strings")
        environment = _mapping(fields.get("environment", {}), label="task environment")
        if not all(
            re.fullmatch(r"[A-Za-z_]\w*", key)
            and isinstance(value, (str, int, float, bool, type(None)))
            for key, value in environment.items()
        ):
            raise ValueError("Environment values must be scalar and names must be identifiers")
        resources = _mapping(fields.pop("resources", {}), _RESOURCES, "resources")
        if mode == "local" and (resources or "container" in fields):
            raise ValueError("Local CPU execution cannot enforce container or resource requests")
        for key in ("nodes", "ntasks_per_node", "gpus_per_node"):
            if key in resources and resources[key] is not None:
                if type(resources[key]) is not int or resources[key] < (
                    0 if key == "gpus_per_node" else 1
                ):
                    raise ValueError(f"Invalid resource value: {key}")
        config = {**slurm, **resources}
        if "container" in fields:
            config["container"] = fields.pop("container")
        for key in ("inputs", "outputs"):
            if key in fields:
                _mapping(fields[key], label=f"task {key}")
        tasks.append(SandboxTask(name=task_name, slurm_config=SlurmConfig(**config), **fields))
    if not tasks:
        raise ValueError("A workflow needs at least one task")
    outputs = _mapping(document.get("outputs", {}), label="workflow outputs")
    by_name = {task.name: task for task in tasks}
    for output_name, spec in outputs.items():
        if not re.fullmatch(r"[A-Za-z][\w-]*", output_name):
            raise ValueError("Workflow output names must use letters, digits, '_' or '-'")
        _mapping(spec, {"from"}, "workflow output")
        if not isinstance(spec.get("from"), str):
            raise ValueError("Workflow output must reference task.output")
        source = spec.get("from", "").split(".")
        if (
            len(source) != 2
            or source[0] not in by_name
            or source[1] not in (by_name[source[0]].outputs or {})
        ):
            raise ValueError("Workflow output must reference an existing task.output")
    pipeline = SandboxPipeline(
        tasks=tasks,
        inputs=document.get("inputs"),
        parameters=values,
        workflow_outputs=outputs,
        local_execution=mode == "local",
        asset_resolver=settings.get("asset_resolver"),
        asset_timeout=settings.get("asset_timeout", 900),
    )
    if mode == "local" and (
        pipeline.inputs
        or any("env" not in spec for task in tasks for spec in (task.inputs or {}).values())
    ):
        raise ValueError(
            "Local execution uses env-bound task outputs and cannot mount catalog inputs"
        )
    if mode == "docker" and not settings.get("hf_local"):
        raise ValueError("Docker profiles require hf_local")
    return {
        "job_name": name,
        "pipeline": pipeline,
        "job_dir": settings.get(
            "job_dir", os.environ.get("SLURM_JOB_DIR", str(Path.home() / "experiments"))
        ),
        "user": settings.get("user") or os.environ.get("SLURM_USER"),
        "identity": settings.get("identity"),
        "hf_local": settings.get("hf_local") if mode == "docker" else None,
    }


def workflow_cli(argv, execute):
    """Parse v2 CLI options, export a normalized plan or invoke the shared launcher."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yaml", required=True)
    parser.add_argument("--profile")
    parser.add_argument("--param", action="append", default=[], metavar="NAME=VALUE")
    parser.add_argument("--dryrun", action="store_true")
    parser.add_argument("--to-yaml")
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--yes", "-y", action="store_true")
    args = parser.parse_args(argv)
    try:
        parameters = {}
        for item in args.param:
            name, separator, value = item.partition("=")
            if not separator or name in parameters:
                raise ValueError("Use each --param NAME=VALUE once")
            parameters[name] = yaml.safe_load(value)
        compiled = load_workflow(args.yaml, args.profile, parameters)
        pipeline = compiled["pipeline"]
        if args.detach and (pipeline.local_execution or compiled["hf_local"] is not None):
            raise ValueError("Detached v2 execution requires a Slurm profile")
        plan = _expand(_read(args.yaml), pipeline.parameters)
        plan["parameters"] = pipeline.parameters
        plan["tasks"] = {
            task.name: {**plan["tasks"][task.name], "depends_on": task.depends_on}
            for task in pipeline.tasks
        }
        if args.to_yaml:
            Path(args.to_yaml).write_text(yaml.safe_dump(plan, sort_keys=False))
        if args.dryrun or args.to_yaml:
            print(yaml.safe_dump(plan, sort_keys=False))
            return
        if not args.yes and input(f"Launch {compiled['job_name']}? [y/N] ").lower() not in (
            "y",
            "yes",
        ):
            return
        execute(**compiled, detach=args.detach)
    except (ValueError, OSError, TypeError, yaml.YAMLError) as exc:
        parser.error(str(exc))
