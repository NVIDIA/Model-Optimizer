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

"""Closed recipe and site schemas for the public Puzzletron contract."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from difflib import get_close_matches
from pathlib import Path
from typing import Any

import yaml
from yaml.nodes import MappingNode, Node, ScalarNode, SequenceNode

from ._public_catalog import ROUTES, ROUTES_BY_KEY
from .schema import ExecutionMode

PUBLIC_SCHEMA_VERSION = 1
DATA_FIELDS = {"path", "revision"}

_RECIPE_FIELDS = {
    "schema_version",
    "name",
    "model",
    "workflow",
    "mode",
    "run_root",
    "resource_profile",
    "data",
    "advanced",
}
_ADVANCED_FIELDS = {"experiment", "execution"}
_ADVANCED_EXECUTION_FIELDS = {"defaults", "stages"}
_SITE_FIELDS = {"schema_version", "site", "resources"}
_SITE_BODY_FIELDS = {"kind", "environment", "paths", "slurm", "baremetal"}
_PATH_FIELDS = {"hf_home"}
_ENVIRONMENT_FIELDS = {
    "repository",
    "source_revision",
    "venv",
    "container",
    "container_mounts",
    "setup_env",
    "prerun_commands",
    "postrun_commands",
}
_SLURM_FIELDS = {
    "account",
    "job_name_prefix",
    "partition",
    "partition_interactive",
    "partition_batch",
    "partition_cpu",
    "cpu_cpus_per_task",
    "cpu_memory_mb",
    "interactive_max_nodes",
    "time_limit",
    "qos",
    "log_dir",
}
_BAREMETAL_FIELDS = {"hosts", "rendezvous_host", "rendezvous_port_base"}
_HOST_FIELDS = {"hostname", "gpus"}
_RESOURCE_FIELDS = {"mode", "gpus_per_node", "max_nodes", "partition"}
_PUBLIC_OWNED_EXPERIMENT_PATHS = {
    "data.revision",
    "dataset_path",
    "descriptor",
    "experiment.dir",
    "input_hf_model_path",
    "model.descriptor_override",
    "model.revision",
    "model.source",
    "prepare_dataset.evaluation_hf_home",
    "prepare_dataset.revision",
    "puzzle_dir",
}
_SITE_OWNED_EXECUTION_FIELDS = {"gpus_per_node", "partition"}


class _UniqueKeyLoader(yaml.SafeLoader):
    """Safe YAML loader that refuses duplicate mapping keys."""


def _construct_unique_mapping(
    loader: _UniqueKeyLoader, node: MappingNode, deep: bool = False
) -> dict[Any, Any]:
    loader.flatten_mapping(node)
    result: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if not isinstance(key, (str, int, float, bool, type(None))):
            raise TypeError(
                f"YAML mapping key at line {key_node.start_mark.line + 1} must be scalar"
            )
        if key in result:
            raise ValueError(f"Duplicate YAML key {key!r} at line {key_node.start_mark.line + 1}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _collect_locations(
    node: Node,
    *,
    path: str,
    source: Path,
    output: dict[str, str],
) -> None:
    if isinstance(node, MappingNode):
        for key_node, value_node in node.value:
            if not isinstance(key_node, ScalarNode):
                continue
            key = str(key_node.value)
            child = f"{path}.{key}" if path else key
            output[child] = f"{source}:{key_node.start_mark.line + 1}"
            _collect_locations(value_node, path=child, source=source, output=output)
    elif isinstance(node, SequenceNode):
        for index, value_node in enumerate(node.value):
            child = f"{path}[{index}]"
            output[child] = f"{source}:{value_node.start_mark.line + 1}"
            _collect_locations(value_node, path=child, source=source, output=output)


def _load_yaml(path: str | Path) -> tuple[dict[str, Any], dict[str, str]]:
    source = Path(path).resolve()
    text = source.read_text()
    loader = _UniqueKeyLoader(text)
    try:
        payload = loader.get_single_data()
    finally:
        loader.dispose()
    if not isinstance(payload, Mapping):
        raise TypeError(f"YAML root must be a mapping: {source}")
    locations: dict[str, str] = {}
    root = yaml.compose(text, Loader=yaml.SafeLoader)
    if root is not None:
        _collect_locations(root, path="", source=source, output=locations)
    return dict(payload), locations


def location(locations: Mapping[str, str], path: str, fallback: Path) -> str:
    current = path
    while current:
        if current in locations:
            return locations[current]
        current = current.rpartition(".")[0]
    return str(fallback)


def _reject_unknown(
    payload: Mapping[str, Any],
    allowed: set[str],
    *,
    path: str,
    locations: Mapping[str, str],
    source: Path,
) -> None:
    for key in payload:
        if not isinstance(key, str):
            raise TypeError(f"{path or 'config'} keys must be strings; got {key!r}")
        if key in allowed:
            continue
        dotted = f"{path}.{key}" if path else key
        suggestion = get_close_matches(key, sorted(allowed), n=1)
        suffix = f"; did you mean {suggestion[0]!r}?" if suggestion else ""
        raise ValueError(f"Unknown field {dotted} at {location(locations, dotted, source)}{suffix}")


def required_string(value: Any, *, path: str, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{path} at {location} must be a non-empty string")
    return value.strip()


def _positive_int(value: Any, *, path: str, location: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{path} at {location} must be a positive integer")
    if value < 1:
        raise ValueError(f"{path} at {location} must be at least 1")
    return value


def _optional_string(value: Any, *, path: str, location: str) -> str | None:
    if value is None:
        return None
    return required_string(value, path=path, location=location)


def _string_sequence(value: Any, *, path: str, location: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{path} at {location} must be a sequence of command strings")
    return [
        required_string(item, path=f"{path}[{index}]", location=location)
        for index, item in enumerate(value)
    ]


def _partition_selector(value: Any, *, path: str, location: str) -> str | list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return required_string(value, path=path, location=location)
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{path} at {location} must be a string or sequence of strings")
    selected = [
        required_string(item, path=f"{path}[{index}]", location=location)
        for index, item in enumerate(value)
    ]
    if not selected:
        raise ValueError(f"{path} at {location} must not be empty")
    return selected


def mapping(value: Any, *, path: str, location: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{path} at {location} must be a mapping")
    return dict(value)


@dataclass(frozen=True)
class Recipe:
    source: Path
    locations: Mapping[str, str]
    name: str
    model: str
    workflow: str
    mode: str
    run_root: Path
    run_root_source: str
    resource_profile: str
    data: Mapping[str, str]
    advanced_experiment: Mapping[str, Any]
    advanced_execution: Mapping[str, Any]

    def as_public_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": PUBLIC_SCHEMA_VERSION,
            "name": self.name,
            "model": self.model,
            "workflow": self.workflow,
            "mode": self.mode,
            "run_root": str(self.run_root),
            "resource_profile": self.resource_profile,
        }
        if self.data:
            payload["data"] = dict(self.data)
        advanced: dict[str, Any] = {}
        if self.advanced_experiment:
            advanced["experiment"] = dict(self.advanced_experiment)
        if self.advanced_execution:
            advanced["execution"] = deepcopy(dict(self.advanced_execution))
        if advanced:
            payload["advanced"] = advanced
        return payload


@dataclass(frozen=True)
class Site:
    source: Path
    locations: Mapping[str, str]
    body: Mapping[str, Any]
    resources: Mapping[str, Mapping[str, Any]]


def parse_recipe(path: str | Path, *, run_root: str | Path | None) -> Recipe:
    source = Path(path).resolve()
    payload, locations = _load_yaml(source)
    _reject_unknown(payload, _RECIPE_FIELDS, path="", locations=locations, source=source)
    version = payload.get("schema_version")
    if version != PUBLIC_SCHEMA_VERSION:
        raise ValueError(
            f"schema_version at {location(locations, 'schema_version', source)} must be "
            f"{PUBLIC_SCHEMA_VERSION}; got {version!r}"
        )
    values = {
        key: required_string(payload.get(key), path=key, location=location(locations, key, source))
        for key in ("name", "model", "workflow", "mode", "resource_profile")
    }
    selected_run_root = str(run_root) if run_root is not None else payload.get("run_root")
    run_root_text = required_string(
        selected_run_root,
        path="run_root",
        location=(
            "command line" if run_root is not None else location(locations, "run_root", source)
        ),
    )
    if "$" in run_root_text:
        raise ValueError("run_root does not support environment interpolation; use --run-root")
    data = mapping(
        payload.get("data", {}),
        path="data",
        location=location(locations, "data", source),
    )
    _reject_unknown(data, DATA_FIELDS, path="data", locations=locations, source=source)
    normalized_data = {
        key: required_string(
            value,
            path=f"data.{key}",
            location=location(locations, f"data.{key}", source),
        )
        for key, value in data.items()
    }
    if "$" in normalized_data.get("path", ""):
        raise ValueError("data.path does not support environment interpolation")
    resolved_run_root = Path(run_root_text).expanduser().resolve()
    if "path" in normalized_data:
        data_path = Path(normalized_data["path"]).expanduser()
        if not data_path.is_absolute():
            data_path = resolved_run_root / data_path
        normalized_data["path"] = str(data_path)
    advanced = mapping(
        payload.get("advanced") or {},
        path="advanced",
        location=location(locations, "advanced", source),
    )
    _reject_unknown(advanced, _ADVANCED_FIELDS, path="advanced", locations=locations, source=source)
    advanced_experiment = mapping(
        advanced.get("experiment", {}),
        path="advanced.experiment",
        location=location(locations, "advanced.experiment", source),
    )
    for dotted in advanced_experiment:
        if not isinstance(dotted, str) or not dotted or any(not part for part in dotted.split(".")):
            raise ValueError(
                f"advanced.experiment keys must be non-empty dotted paths; got {dotted!r}"
            )
        if (
            dotted in _PUBLIC_OWNED_EXPERIMENT_PATHS
            or dotted == "model_info"
            or dotted.startswith("model_info.")
            or dotted.endswith(".evaluator_revision")
        ):
            raise ValueError(
                f"advanced.experiment.{dotted} is owned by the selected public route, recipe, "
                "or site and cannot be overridden"
            )
    advanced_execution = mapping(
        advanced.get("execution", {}),
        path="advanced.execution",
        location=location(locations, "advanced.execution", source),
    )
    _reject_unknown(
        advanced_execution,
        _ADVANCED_EXECUTION_FIELDS,
        path="advanced.execution",
        locations=locations,
        source=source,
    )
    for execution_field in _ADVANCED_EXECUTION_FIELDS:
        if execution_field not in advanced_execution:
            continue
        execution_values = mapping(
            advanced_execution[execution_field],
            path=f"advanced.execution.{execution_field}",
            location=location(locations, f"advanced.execution.{execution_field}", source),
        )
        if execution_field == "defaults":
            reserved = sorted(_SITE_OWNED_EXECUTION_FIELDS.intersection(execution_values))
            if reserved:
                raise ValueError(
                    "advanced.execution.defaults cannot override site-owned fields: "
                    + ", ".join(reserved)
                )
            continue
        for stage_id, raw_stage in execution_values.items():
            if not isinstance(raw_stage, Mapping):
                continue
            reserved = sorted(_SITE_OWNED_EXECUTION_FIELDS.intersection(raw_stage))
            if reserved:
                raise ValueError(
                    f"advanced.execution.stages.{stage_id} cannot override site-owned fields: "
                    + ", ".join(reserved)
                )
    return Recipe(
        source=source,
        locations=locations,
        name=values["name"],
        model=values["model"],
        workflow=values["workflow"],
        mode=values["mode"],
        run_root=resolved_run_root,
        run_root_source=(
            "command line --run-root"
            if run_root is not None
            else location(locations, "run_root", source)
        ),
        resource_profile=values["resource_profile"],
        data=normalized_data,
        advanced_experiment=advanced_experiment,
        advanced_execution=advanced_execution,
    )


def parse_site(path: str | Path) -> Site:
    source = Path(path).resolve()
    payload, locations = _load_yaml(source)
    _reject_unknown(payload, _SITE_FIELDS, path="", locations=locations, source=source)
    if payload.get("schema_version") != PUBLIC_SCHEMA_VERSION:
        raise ValueError(
            f"schema_version at {location(locations, 'schema_version', source)} must be "
            f"{PUBLIC_SCHEMA_VERSION}; got {payload.get('schema_version')!r}"
        )
    site = mapping(payload.get("site"), path="site", location=location(locations, "site", source))
    _reject_unknown(site, _SITE_BODY_FIELDS, path="site", locations=locations, source=source)
    kind = required_string(
        site.get("kind"), path="site.kind", location=location(locations, "site.kind", source)
    )
    if kind not in {"slurm", "baremetal"}:
        raise ValueError(f"site.kind must be 'slurm' or 'baremetal'; got {kind!r}")
    environment = mapping(
        site.get("environment"),
        path="site.environment",
        location=location(locations, "site.environment", source),
    )
    _reject_unknown(
        environment,
        _ENVIRONMENT_FIELDS,
        path="site.environment",
        locations=locations,
        source=source,
    )
    normalized_environment = {
        required: required_string(
            environment.get(required),
            path=f"site.environment.{required}",
            location=location(locations, f"site.environment.{required}", source),
        )
        for required in ("repository", "venv")
    }
    for optional in ("source_revision", "container", "container_mounts", "setup_env"):
        if optional in environment:
            normalized_environment[optional] = _optional_string(
                environment[optional],
                path=f"site.environment.{optional}",
                location=location(locations, f"site.environment.{optional}", source),
            )
    for commands in ("prerun_commands", "postrun_commands"):
        if commands in environment:
            normalized_environment[commands] = _string_sequence(
                environment[commands],
                path=f"site.environment.{commands}",
                location=location(locations, f"site.environment.{commands}", source),
            )
    environment = normalized_environment
    if environment.get("container_mounts") and not environment.get("container"):
        raise ValueError(
            "site.environment.container_mounts is unused without site.environment.container"
        )
    if kind == "baremetal" and (
        environment.get("container") or environment.get("container_mounts")
    ):
        raise ValueError(
            "site.environment container settings are unused when site.kind is 'baremetal'"
        )
    paths = mapping(
        site.get("paths", {}),
        path="site.paths",
        location=location(locations, "site.paths", source),
    )
    _reject_unknown(paths, _PATH_FIELDS, path="site.paths", locations=locations, source=source)
    paths = {
        name: required_string(
            value,
            path=f"site.paths.{name}",
            location=location(locations, f"site.paths.{name}", source),
        )
        for name, value in paths.items()
    }
    if "hf_home" not in paths:
        raise ValueError("site.paths.hf_home is required")
    hf_home_assignment = re.compile(r"(?:^|\s)(?:export\s+)?HF_HOME\s*=")
    for index, command in enumerate(environment.get("prerun_commands", ())):
        if hf_home_assignment.search(command):
            raise ValueError(
                f"site.environment.prerun_commands[{index}] assigns HF_HOME, which conflicts "
                "with site.paths.hf_home"
            )
    slurm = mapping(
        site.get("slurm", {}),
        path="site.slurm",
        location=location(locations, "site.slurm", source),
    )
    baremetal = mapping(
        site.get("baremetal", {}),
        path="site.baremetal",
        location=location(locations, "site.baremetal", source),
    )
    _reject_unknown(slurm, _SLURM_FIELDS, path="site.slurm", locations=locations, source=source)
    _reject_unknown(
        baremetal,
        _BAREMETAL_FIELDS,
        path="site.baremetal",
        locations=locations,
        source=source,
    )
    if kind == "slurm" and baremetal:
        raise ValueError("site.baremetal is unused when site.kind is 'slurm'")
    if kind == "baremetal" and slurm:
        raise ValueError("site.slurm is unused when site.kind is 'baremetal'")
    if kind == "baremetal":
        hosts = baremetal.get("hosts")
        if isinstance(hosts, (str, bytes)) or not isinstance(hosts, Sequence) or not hosts:
            raise TypeError("site.baremetal.hosts must be a non-empty sequence")
        for index, raw_host in enumerate(hosts):
            host = mapping(
                raw_host,
                path=f"site.baremetal.hosts[{index}]",
                location=location(locations, f"site.baremetal.hosts[{index}]", source),
            )
            _reject_unknown(
                host,
                _HOST_FIELDS,
                path=f"site.baremetal.hosts[{index}]",
                locations=locations,
                source=source,
            )
            required_string(
                host.get("hostname"),
                path=f"site.baremetal.hosts[{index}].hostname",
                location=location(locations, f"site.baremetal.hosts[{index}].hostname", source),
            )
            _positive_int(
                host.get("gpus", 8),
                path=f"site.baremetal.hosts[{index}].gpus",
                location=location(locations, f"site.baremetal.hosts[{index}].gpus", source),
            )
        if "rendezvous_host" in baremetal:
            baremetal["rendezvous_host"] = _optional_string(
                baremetal["rendezvous_host"],
                path="site.baremetal.rendezvous_host",
                location=location(locations, "site.baremetal.rendezvous_host", source),
            )
        if "rendezvous_port_base" in baremetal:
            baremetal["rendezvous_port_base"] = _positive_int(
                baremetal["rendezvous_port_base"],
                path="site.baremetal.rendezvous_port_base",
                location=location(locations, "site.baremetal.rendezvous_port_base", source),
            )
    normalized_slurm: dict[str, Any] = {}
    if kind == "slurm":
        normalized_slurm["account"] = required_string(
            slurm.get("account"),
            path="site.slurm.account",
            location=location(locations, "site.slurm.account", source),
        )
        for name in ("job_name_prefix", "time_limit"):
            if name in slurm:
                normalized_slurm[name] = required_string(
                    slurm[name],
                    path=f"site.slurm.{name}",
                    location=location(locations, f"site.slurm.{name}", source),
                )
        for name in ("qos", "log_dir"):
            if name in slurm:
                normalized_slurm[name] = _optional_string(
                    slurm[name],
                    path=f"site.slurm.{name}",
                    location=location(locations, f"site.slurm.{name}", source),
                )
        for name in ("partition", "partition_interactive", "partition_batch", "partition_cpu"):
            if name in slurm:
                normalized_slurm[name] = _partition_selector(
                    slurm[name],
                    path=f"site.slurm.{name}",
                    location=location(locations, f"site.slurm.{name}", source),
                )
        for name in ("cpu_cpus_per_task", "cpu_memory_mb", "interactive_max_nodes"):
            if name in slurm:
                normalized_slurm[name] = _positive_int(
                    slurm[name],
                    path=f"site.slurm.{name}",
                    location=location(locations, f"site.slurm.{name}", source),
                )
        slurm = normalized_slurm
    resources = mapping(
        payload.get("resources"),
        path="resources",
        location=location(locations, "resources", source),
    )
    if not resources:
        raise ValueError("resources must define at least one named resource profile")
    normalized_resources: dict[str, dict[str, Any]] = {}
    for name, raw_resource in resources.items():
        if not isinstance(name, str) or not name.strip():
            raise TypeError("resource profile names must be non-empty strings")
        resource = mapping(
            raw_resource,
            path=f"resources.{name}",
            location=location(locations, f"resources.{name}", source),
        )
        _reject_unknown(
            resource,
            _RESOURCE_FIELDS,
            path=f"resources.{name}",
            locations=locations,
            source=source,
        )
        mode = required_string(
            resource.get("mode", "per_attempt"),
            path=f"resources.{name}.mode",
            location=location(locations, f"resources.{name}.mode", source),
        )
        if mode not in {item.value for item in ExecutionMode}:
            raise ValueError(f"resources.{name}.mode has unsupported value {mode!r}")
        gpus_per_node = _positive_int(
            resource.get("gpus_per_node"),
            path=f"resources.{name}.gpus_per_node",
            location=location(locations, f"resources.{name}.gpus_per_node", source),
        )
        max_nodes = _positive_int(
            resource.get("max_nodes"),
            path=f"resources.{name}.max_nodes",
            location=location(locations, f"resources.{name}.max_nodes", source),
        )
        if kind == "baremetal":
            if mode == ExecutionMode.REUSABLE_ALLOCATION.value:
                raise ValueError(
                    f"resources.{name}.mode reusable_allocation is only available for Slurm"
                )
            if "partition" in resource:
                raise ValueError(f"resources.{name}.partition is unused for bare-metal sites")
            hosts = list(baremetal["hosts"])
            if max_nodes > len(hosts):
                raise ValueError(
                    f"resources.{name}.max_nodes={max_nodes} exceeds the {len(hosts)} "
                    "bare-metal hosts"
                )
            undersized = [
                str(host["hostname"])
                for host in hosts[:max_nodes]
                if int(host.get("gpus", 8)) < gpus_per_node
            ]
            if undersized:
                raise ValueError(
                    f"resources.{name}.gpus_per_node={gpus_per_node} exceeds host capacity: "
                    + ", ".join(undersized)
                )
        normalized_resources[name] = {
            "mode": mode,
            "gpus_per_node": gpus_per_node,
            "max_nodes": max_nodes,
            **(
                {
                    "partition": _partition_selector(
                        resource["partition"],
                        path=f"resources.{name}.partition",
                        location=location(locations, f"resources.{name}.partition", source),
                    )
                }
                if "partition" in resource
                else {}
            ),
        }
    return Site(
        source=source,
        locations=locations,
        body={
            **site,
            "kind": kind,
            "environment": environment,
            "paths": paths,
            "slurm": slurm,
            "baremetal": baremetal,
        },
        resources=normalized_resources,
    )


def recipe_template(
    *,
    model: str = "qwen3.5-0.8b",
    workflow: str = "vlm-pruning",
    mode: str = "smoke",
    resource_profile: str = "smoke",
    run_root: str = "puzzle_runs/my-puzzletron-run",
    name: str = "my-puzzletron-run",
) -> dict[str, Any]:
    """Return the exact public recipe contract used by the resolver."""

    route = ROUTES_BY_KEY.get((model, workflow, mode))
    if route is None:
        choices = ", ".join(sorted(item.route_id for item in ROUTES))
        raise ValueError(f"Unknown route {(model, workflow, mode)!r}; choose one of: {choices}")
    recipe = {
        "schema_version": PUBLIC_SCHEMA_VERSION,
        "name": name,
        "model": model,
        "workflow": workflow,
        "mode": mode,
        "run_root": run_root,
        "resource_profile": resource_profile,
    }
    if route.requires_data:
        recipe["data"] = {
            "path": "REPLACE_WITH_PREPARED_DATASET",
            "revision": "REPLACE_WITH_IMMUTABLE_DATASET_REVISION",
        }
    return recipe


def site_template() -> dict[str, Any]:
    """Return one portable site contract with small and multi-node profiles."""

    return {
        "schema_version": PUBLIC_SCHEMA_VERSION,
        "site": {
            "kind": "slurm",
            "environment": {
                "repository": "REPLACE_WITH_WORKER_VISIBLE_MODELOPT_CHECKOUT",
                "venv": "REPLACE_WITH_WORKER_VISIBLE_MODELOPT_VENV",
                "container": None,
                "container_mounts": None,
                "prerun_commands": [],
                "postrun_commands": [],
            },
            "paths": {"hf_home": "REPLACE_WITH_SHARED_HF_HOME"},
            "slurm": {
                "account": "REPLACE_WITH_SLURM_ACCOUNT",
                "partition": "REPLACE_WITH_SLURM_PARTITION",
                "partition_cpu": None,
                "cpu_cpus_per_task": 4,
                "cpu_memory_mb": 32768,
                "time_limit": "4:00:00",
            },
        },
        "resources": {
            "single-gpu": {
                "mode": "per_attempt",
                "gpus_per_node": 1,
                "max_nodes": 1,
            },
            "smoke": {
                "mode": "reusable_allocation",
                "gpus_per_node": 2,
                "max_nodes": 1,
            },
            "campaign": {
                "mode": "reusable_allocation",
                "gpus_per_node": 8,
                "max_nodes": 1,
            },
            "multinode": {
                "mode": "per_attempt",
                "gpus_per_node": 8,
                "max_nodes": 64,
            },
        },
    }
