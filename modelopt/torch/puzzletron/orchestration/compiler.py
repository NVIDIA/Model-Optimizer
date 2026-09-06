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

"""Compile experiment, runner, and execution configs into a CampaignPlan."""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import asdict
from difflib import get_close_matches
from pathlib import Path
from typing import Any, Mapping

import yaml

if (__package__ or "").startswith("puzzletron_orchestrator"):
    from puzzletron_orchestrator.profiles import expand_mip_variants, select_mip_values
else:
    from ..mip.profiles import expand_mip_variants, select_mip_values

from .config import load_experiment_config
from .identity import execution_contract_hash, with_contract_hash
from .mesh import (
    ParallelMesh,
    extract_stage_mesh,
    gpus_per_instance,
    pack_gpu_allocation,
    vllm_topology_to_mesh,
)
from .schema import (
    BareMetalHost,
    BareMetalRunnerConfig,
    CampaignPlan,
    ExecutionContract,
    ExecutionStrategy,
    FailurePolicy,
    HaltPolicy,
    ParallelMeshOverride,
    RunnerEnvironment,
    SlurmRunnerConfig,
    StageExecutionSpec,
    StagePlanNode,
    normalize_slurm_partition,
)
from .stages import (
    configured_parent_stage_ids,
    configured_stage_ids,
    default_stage_resource,
    distributed_stage_ids,
    stage_ids,
)
from .vllm_measurements import normalize_vllm_measurements

__all__ = [
    "compile_campaign_plan",
    "load_execution_config",
    "load_runner_config",
    "mip_resource",
    "plan_to_dict",
    "resolve_stage_execution_specs",
    "validate_runner_ready",
]

_CONTROLLER_REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
_DEFAULT_ARTIFACT_SETTLING_TIMEOUT_SECONDS = 300.0

_RUNNER_FIELDS = {"kind", "execution_contract", "slurm", "inventory"}
_EXECUTION_CONTRACT_FIELDS = {
    "repository",
    "venv",
    "container",
    "container_mounts",
    "mounts",
    "setup_env",
    "prerun_commands",
    "prerun",
    "postrun_commands",
    "postrun",
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
    "max_nodes",
    "time_limit",
    "qos",
    "log_dir",
}
_INVENTORY_FIELDS = {"hosts", "rendezvous_host", "rendezvous_port_base"}
_HOST_FIELDS = {"hostname", "gpus"}
_EXECUTION_FIELDS = {"schema_version", "defaults", "stages"}
_SUPPORTED_EXECUTION_SCHEMA_VERSION = 1
_EXECUTION_DEFAULT_FIELDS = {
    "artifact_settling_timeout_seconds",
    "failure_policy",
    "halt_policy",
    "gpus_per_node",
    "partition",
    "resource",
}
_STAGE_EXECUTION_FIELDS = {
    "strategy",
    "instances",
    "num_jobs",
    "failure_policy",
    "gpus_per_node",
    "partition",
    "resource",
    "parallel",
}
_FINAL_REPORT_FIELDS = {"partition"}
_PARALLEL_FIELDS = {
    "tp",
    "cp",
    "pp",
    "ep",
    "dp",
    "dp_shard",
    "dp_replicate",
}


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


_DEFAULT_STAGE_STRATEGIES: dict[str, ExecutionStrategy] = {
    "vllm_stats": ExecutionStrategy.SHARDED,
    "replacement_scoring": ExecutionStrategy.PERSISTENT_POOL,
    "depth_importance": ExecutionStrategy.PERSISTENT_POOL,
    "zero_shot_evaluation": ExecutionStrategy.SHARDED,
    "aiperf": ExecutionStrategy.SHARDED,
}


def _post_mip_stage_metadata(config: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    """Compile canonical dependency-light dynamic stage facts for the controller."""

    if (__package__ or "").startswith("puzzletron_orchestrator"):
        from puzzletron_orchestrator.post_mip.base import compile_post_mip_flows
    else:
        from ..post_mip.base import compile_post_mip_flows

    return tuple(
        {
            "stage_id": node.stage_id,
            "node_id": node.node_id,
            "node_type": node.node_type,
            "parents": node.dependency_stage_ids,
            "distributed": node.capabilities.distributed,
            "default_strategy": ExecutionStrategy(node.capabilities.default_strategy),
            "default_resource": node.capabilities.default_resource,
            "config": dict(node.config.get("config") or {}),
        }
        for node in compile_post_mip_flows(config)
    )


def _required_mapping(value: Any, *, path: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{path} must be a mapping")
    return dict(value)


def _reject_unknown_fields(
    payload: Mapping[str, Any],
    allowed: set[str],
    *,
    path: str,
) -> None:
    for field in payload:
        if not isinstance(field, str):
            raise TypeError(f"{path} field names must be strings; got {field!r}")
        if field in allowed:
            continue
        suggestion = get_close_matches(field, sorted(allowed), n=1)
        suffix = f"; did you mean {suggestion[0]!r}?" if suggestion else ""
        raise ValueError(f"Unknown config field {path}.{field}{suffix}")


def _positive_int(value: Any, *, path: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{path} must be a positive integer")
    if isinstance(value, int):
        parsed = value
    elif isinstance(value, str) and value.strip().isdigit():
        parsed = int(value)
    else:
        raise TypeError(f"{path} must be a positive integer")
    if parsed < 1:
        raise ValueError(f"{path} must be at least 1")
    return parsed


def _command_sequence(value: Any, *, path: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if not isinstance(value, Sequence) or any(not isinstance(item, str) for item in value):
        raise TypeError(f"{path} must be a string or a sequence of strings")
    return tuple(value)


_ENV_ASSIGNMENT = re.compile(r"\b(?:export\s+)?([A-Z_][A-Z0-9_]*)\s*=\s*([^\s;]+)", re.IGNORECASE)
_SECRET_VARIABLE_SUFFIXES = ("TOKEN", "PASSWORD", "SECRET", "API_KEY", "ACCESS_KEY", "PRIVATE_KEY")
_INHERITED_VARIABLE = re.compile(
    r"\$(?:[A-Z_][A-Z0-9_]*|\{[A-Z_][A-Z0-9_]*(?::?\?[^}]*)?\})",
    re.IGNORECASE,
)
_REQUIRED_VARIABLE = re.compile(r"^\$\{[A-Z_][A-Z0-9_]*:?\?", re.IGNORECASE)


def _reject_inline_secret_assignments(commands: Sequence[str], *, path: str) -> None:
    """Keep literal credentials out of rendered and persisted worker scripts."""

    for command in commands:
        for match in _ENV_ASSIGNMENT.finditer(command):
            variable = match.group(1).upper()
            if not any(
                variable == suffix or variable.endswith(f"_{suffix}")
                for suffix in _SECRET_VARIABLE_SUFFIXES
            ):
                continue
            value = match.group(2).strip("\"'")
            if (
                _INHERITED_VARIABLE.fullmatch(value)
                or _REQUIRED_VARIABLE.match(value)
                or value.startswith("$(")
            ):
                continue
            raise ValueError(
                f"{path} assigns a literal value to {match.group(1)!r}; inherit it from the "
                "environment or source a protected setup_env file instead"
            )


def _validate_execution_payload(execution: Mapping[str, Any]) -> None:
    _reject_unknown_fields(execution, _EXECUTION_FIELDS, path="execution")
    schema_version = execution.get("schema_version")
    if schema_version is not None and (
        type(schema_version) is not int or schema_version != _SUPPORTED_EXECUTION_SCHEMA_VERSION
    ):
        raise ValueError(
            f"Unsupported execution schema {schema_version!r}; expected "
            f"{_SUPPORTED_EXECUTION_SCHEMA_VERSION}. Update the execution config to a supported "
            "schema version."
        )
    defaults = _required_mapping(execution.get("defaults", {}), path="execution.defaults")
    _reject_unknown_fields(defaults, _EXECUTION_DEFAULT_FIELDS, path="execution.defaults")
    if "failure_policy" in defaults:
        FailurePolicy(str(defaults["failure_policy"]))
    if "halt_policy" in defaults:
        HaltPolicy(str(defaults["halt_policy"]))
    if "gpus_per_node" in defaults:
        _positive_int(defaults["gpus_per_node"], path="execution.defaults.gpus_per_node")
    if "partition" in defaults:
        normalize_slurm_partition(defaults["partition"], path="execution.defaults.partition")
    if "resource" in defaults and str(defaults["resource"]) not in {"cpu", "gpu"}:
        raise ValueError("execution.defaults.resource must be 'cpu' or 'gpu'")

    stages = _required_mapping(execution.get("stages", {}), path="execution.stages")
    for stage_id, raw_stage in stages.items():
        if not isinstance(stage_id, str) or not stage_id:
            raise TypeError(f"execution.stages keys must be non-empty strings; got {stage_id!r}")
        stage_path = f"execution.stages.{stage_id}"
        stage = _required_mapping(raw_stage, path=stage_path)
        allowed_fields = (
            _FINAL_REPORT_FIELDS if stage_id == "final_report" else _STAGE_EXECUTION_FIELDS
        )
        _reject_unknown_fields(stage, allowed_fields, path=stage_path)
        if stage_id == "final_report":
            if "partition" in stage:
                normalize_slurm_partition(stage["partition"], path=f"{stage_path}.partition")
            continue
        if "instances" in stage and "num_jobs" in stage:
            raise ValueError(f"{stage_path} cannot set both instances and legacy num_jobs")
        if "strategy" in stage:
            ExecutionStrategy(str(stage["strategy"]))
        if "failure_policy" in stage:
            FailurePolicy(str(stage["failure_policy"]))
        if "instances" in stage:
            _positive_int(stage["instances"], path=f"{stage_path}.instances")
        if "num_jobs" in stage:
            _positive_int(stage["num_jobs"], path=f"{stage_path}.num_jobs")
        if "gpus_per_node" in stage:
            _positive_int(stage["gpus_per_node"], path=f"{stage_path}.gpus_per_node")
        if "partition" in stage:
            normalize_slurm_partition(stage["partition"], path=f"{stage_path}.partition")
        if "resource" in stage and str(stage["resource"]) not in {"cpu", "gpu"}:
            raise ValueError(f"{stage_path}.resource must be 'cpu' or 'gpu'")
        if "parallel" in stage:
            parallel = _required_mapping(stage["parallel"], path=f"{stage_path}.parallel")
            if "sequence_parallel" in parallel:
                raise ValueError(
                    f"{stage_path}.parallel.sequence_parallel belongs in the experiment "
                    "model-parallel profile; it does not affect scheduler allocation"
                )
            _reject_unknown_fields(parallel, _PARALLEL_FIELDS, path=f"{stage_path}.parallel")
            if "dp" in parallel and "dp_replicate" in parallel:
                raise ValueError(f"{stage_path}.parallel cannot set both dp and dp_replicate")
            for field, value in parallel.items():
                _positive_int(value, path=f"{stage_path}.parallel.{field}")


def _validate_execution_stage_ids(
    execution: Mapping[str, Any],
    *,
    dynamic_stage_ids: Sequence[str],
) -> None:
    stages = _required_mapping(execution.get("stages", {}), path="execution.stages")
    allowed = {*stage_ids(), *dynamic_stage_ids, "final_report"}
    _reject_unknown_fields(stages, allowed, path="execution.stages")


def _load_yaml(path: str | Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text())
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return payload


def _worker_experiment_path(path: Path, runner: RunnerEnvironment) -> str:
    """Map a checked-in config to the repository path visible to workers."""

    resolved = path.resolve()
    try:
        relative = resolved.relative_to(_CONTROLLER_REPOSITORY_ROOT)
    except ValueError:
        # Generated campaign bundles normally live on shared storage outside
        # the checkout and retain that explicitly configured absolute path.
        return str(resolved)
    return str(Path(runner.contract.repository) / relative)


def load_runner_config(path: str | Path) -> RunnerEnvironment:
    """Load a runner environment YAML file."""

    payload = _load_yaml(path)
    _reject_unknown_fields(payload, {"runner"}, path="config")
    runner = _required_mapping(payload.get("runner"), path="runner")
    _reject_unknown_fields(runner, _RUNNER_FIELDS, path="runner")
    kind = str(runner.get("kind", "slurm"))
    contract_payload = _required_mapping(
        runner.get("execution_contract", {}), path="runner.execution_contract"
    )
    _reject_unknown_fields(
        contract_payload,
        _EXECUTION_CONTRACT_FIELDS,
        path="runner.execution_contract",
    )
    for canonical, alias in (
        ("container_mounts", "mounts"),
        ("prerun_commands", "prerun"),
        ("postrun_commands", "postrun"),
    ):
        if canonical in contract_payload and alias in contract_payload:
            raise ValueError(
                f"runner.execution_contract cannot set both {canonical} and legacy {alias}"
            )
    prerun = _command_sequence(
        contract_payload.get("prerun_commands", contract_payload.get("prerun")),
        path="runner.execution_contract.prerun_commands",
    )
    postrun = _command_sequence(
        contract_payload.get("postrun_commands", contract_payload.get("postrun")),
        path="runner.execution_contract.postrun_commands",
    )
    _reject_inline_secret_assignments(prerun, path="runner.execution_contract.prerun_commands")
    _reject_inline_secret_assignments(postrun, path="runner.execution_contract.postrun_commands")
    contract = ExecutionContract(
        repository=str(contract_payload.get("repository", ".")),
        venv=str(contract_payload.get("venv", ".venv")),
        container=contract_payload.get("container"),
        container_mounts=contract_payload.get("container_mounts") or contract_payload.get("mounts"),
        setup_env=contract_payload.get("setup_env"),
        prerun_commands=prerun,
        postrun_commands=postrun,
    )
    slurm = None
    baremetal = None
    if kind == "slurm":
        if "inventory" in runner:
            raise ValueError("runner.inventory is only valid when runner.kind is 'baremetal'")
        slurm_payload = _required_mapping(runner.get("slurm", {}), path="runner.slurm")
        _reject_unknown_fields(slurm_payload, _SLURM_FIELDS, path="runner.slurm")
        max_nodes = (
            _positive_int(slurm_payload["max_nodes"], path="runner.slurm.max_nodes")
            if slurm_payload.get("max_nodes") is not None
            else None
        )
        slurm = SlurmRunnerConfig(
            account=str(slurm_payload.get("account", "")),
            job_name_prefix=str(slurm_payload.get("job_name_prefix", "pt")),
            partition=slurm_payload.get("partition"),
            partition_interactive=slurm_payload.get("partition_interactive"),
            partition_batch=slurm_payload.get("partition_batch"),
            partition_cpu=slurm_payload.get("partition_cpu"),
            cpu_cpus_per_task=(
                _positive_int(
                    slurm_payload["cpu_cpus_per_task"],
                    path="runner.slurm.cpu_cpus_per_task",
                )
                if slurm_payload.get("cpu_cpus_per_task") is not None
                else None
            ),
            cpu_memory_mb=(
                _positive_int(
                    slurm_payload["cpu_memory_mb"],
                    path="runner.slurm.cpu_memory_mb",
                )
                if slurm_payload.get("cpu_memory_mb") is not None
                else None
            ),
            interactive_max_nodes=_positive_int(
                slurm_payload.get("interactive_max_nodes", 2),
                path="runner.slurm.interactive_max_nodes",
            ),
            max_nodes=max_nodes,
            time_limit=str(slurm_payload.get("time_limit", "4:00:00")),
            qos=slurm_payload.get("qos"),
            log_dir=slurm_payload.get("log_dir"),
        )
    elif kind == "baremetal":
        if "slurm" in runner:
            raise ValueError("runner.slurm is only valid when runner.kind is 'slurm'")
        inventory = _required_mapping(runner.get("inventory", {}), path="runner.inventory")
        _reject_unknown_fields(inventory, _INVENTORY_FIELDS, path="runner.inventory")
        raw_hosts = inventory.get("hosts", ())
        if isinstance(raw_hosts, (str, bytes)) or not isinstance(raw_hosts, Sequence):
            raise TypeError("runner.inventory.hosts must be a sequence of host mappings")
        hosts_list = []
        for index, raw_host in enumerate(raw_hosts):
            host_path = f"runner.inventory.hosts[{index}]"
            host = _required_mapping(raw_host, path=host_path)
            _reject_unknown_fields(host, _HOST_FIELDS, path=host_path)
            hostname = str(host.get("hostname", "")).strip()
            if not hostname:
                raise ValueError(f"{host_path}.hostname must be non-empty")
            hosts_list.append(
                BareMetalHost(
                    hostname=hostname,
                    gpus=_positive_int(host.get("gpus", 8), path=f"{host_path}.gpus"),
                )
            )
        hosts = tuple(hosts_list)
        if not hosts:
            raise ValueError("runner.inventory.hosts must contain at least one host")
        hostnames = [host.hostname for host in hosts]
        if len(hostnames) != len(set(hostnames)):
            raise ValueError("runner.inventory.hosts contains duplicate hostnames")
        rendezvous_host = inventory.get("rendezvous_host")
        if rendezvous_host is not None and str(rendezvous_host) not in hostnames:
            raise ValueError("runner.inventory.rendezvous_host must name an inventory host")
        baremetal = BareMetalRunnerConfig(
            hosts=hosts,
            rendezvous_host=str(rendezvous_host) if rendezvous_host is not None else None,
            rendezvous_port_base=_positive_int(
                inventory.get("rendezvous_port_base", 29500),
                path="runner.inventory.rendezvous_port_base",
            ),
        )
    else:
        raise ValueError(f"Unsupported runner kind: {kind}")

    environment = RunnerEnvironment(
        kind=kind,
        contract=contract,
        slurm=slurm,
        baremetal=baremetal,
    )
    updated_contract = with_contract_hash(environment)
    return RunnerEnvironment(
        kind=environment.kind,
        contract=updated_contract,
        slurm=environment.slurm,
        baremetal=environment.baremetal,
    )


def _placeholder_paths(value: Any, *, path: str) -> list[str]:
    if isinstance(value, str):
        return [path] if "REPLACE_WITH_" in value else []
    if isinstance(value, Mapping):
        placeholders = []
        for key, item in value.items():
            display_key = (
                {"contract": "execution_contract", "baremetal": "inventory"}.get(key, key)
                if path == "runner"
                else key
            )
            placeholders.extend(_placeholder_paths(item, path=f"{path}.{display_key}"))
        return placeholders
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [
            placeholder
            for index, item in enumerate(value)
            for placeholder in _placeholder_paths(item, path=f"{path}[{index}]")
        ]
    return []


def validate_runner_ready(runner: RunnerEnvironment) -> None:
    """Reject portable runner templates that still contain site placeholders."""

    placeholders = _placeholder_paths(asdict(runner), path="runner")
    if placeholders:
        raise ValueError(
            "runner contains unresolved REPLACE_WITH_ placeholders: " + ", ".join(placeholders)
        )


def load_execution_config(path: str | Path) -> dict[str, Any]:
    """Load execution semantics YAML."""

    payload = _load_yaml(path)
    _reject_unknown_fields(payload, {"execution"}, path="config")
    execution = _required_mapping(payload.get("execution"), path="execution")
    _validate_execution_payload(execution)
    return execution


def _resolve_artifact_settling_timeout_seconds(
    execution_defaults: Mapping[str, Any],
) -> float:
    """Resolve the finite positive timeout for publishing completed-stage artifacts."""

    value = execution_defaults.get(
        "artifact_settling_timeout_seconds",
        _DEFAULT_ARTIFACT_SETTLING_TIMEOUT_SECONDS,
    )
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(
            "execution.defaults.artifact_settling_timeout_seconds must be a positive "
            f"finite number, got {value!r}"
        )
    timeout_seconds = float(value)
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError(
            "execution.defaults.artifact_settling_timeout_seconds must be a positive "
            f"finite number, got {value!r}"
        )
    return timeout_seconds


def _parse_mesh_override(payload: Mapping[str, Any] | None) -> ParallelMeshOverride | None:
    if not payload:
        return None
    return ParallelMeshOverride(
        tp=payload.get("tp"),
        cp=payload.get("cp"),
        pp=payload.get("pp"),
        ep=payload.get("ep"),
        dp_shard=payload.get("dp_shard"),
        dp_replicate=payload.get("dp_replicate", payload.get("dp")),
    )


def _vllm_stage_mesh(config: Mapping[str, Any], override: Mapping[str, Any] | None) -> ParallelMesh:
    """Resolve the primary measurement mesh and reject a conflicting duplicate."""

    measurement_id, measurement = next(iter(normalize_vllm_measurements(config).items()))
    topology_mesh = vllm_topology_to_mesh(measurement.topology)
    if override:
        overridden = topology_mesh.as_dict()
        overridden.update(override)
        if ParallelMesh.from_mapping(overridden) != topology_mesh:
            raise ValueError(
                "vllm_stats execution parallel override conflicts with primary "
                f"vLLM measurement topology {measurement_id!r}"
            )
    return topology_mesh


def resolve_stage_execution_specs(
    execution: Mapping[str, Any],
    enabled_stages: tuple[str, ...],
    *,
    dynamic_defaults: Mapping[str, ExecutionStrategy] | None = None,
    dynamic_resources: Mapping[str, str] | None = None,
) -> dict[str, StageExecutionSpec]:
    """Resolve per-stage execution specs with defaults."""

    defaults = _mapping(execution.get("defaults"))
    _resolve_artifact_settling_timeout_seconds(defaults)
    default_gpus_per_node = _positive_int(
        defaults.get("gpus_per_node", 8), path="execution.defaults.gpus_per_node"
    )
    default_policy = FailurePolicy(str(defaults.get("failure_policy", FailurePolicy.STRICT.value)))
    stage_payload = _mapping(execution.get("stages"))
    dynamic_defaults = dict(dynamic_defaults or {})
    dynamic_resources = dict(dynamic_resources or {})
    resolved: dict[str, StageExecutionSpec] = {}

    for stage_id in enabled_stages:
        payload = _mapping(stage_payload.get(stage_id))
        strategy_name = payload.get("strategy")
        if strategy_name is None:
            strategy = dynamic_defaults.get(
                stage_id,
                _DEFAULT_STAGE_STRATEGIES.get(stage_id, ExecutionStrategy.SINGLE),
            )
        else:
            strategy = ExecutionStrategy(str(strategy_name))
        instances = _positive_int(
            payload.get("instances", payload.get("num_jobs", 1)),
            path=f"execution.stages.{stage_id}.instances",
        )
        if strategy is ExecutionStrategy.SINGLE and instances != 1:
            raise ValueError(
                f"execution.stages.{stage_id}.instances must be 1 for strategy 'single'"
            )
        policy = FailurePolicy(str(payload.get("failure_policy", default_policy.value)))
        gpus_per_node = payload.get("gpus_per_node", defaults.get("gpus_per_node"))
        partition_path = (
            f"execution.stages.{stage_id}.partition"
            if "partition" in payload
            else "execution.defaults.partition"
        )
        partition = normalize_slurm_partition(
            payload.get("partition", defaults.get("partition")), path=partition_path
        )
        default_resource = dynamic_resources.get(stage_id)
        if default_resource is None:
            default_resource = (
                "gpu" if stage_id.startswith("post.") else default_stage_resource(stage_id)
            )
        if "resource" in payload:
            resource_path = f"execution.stages.{stage_id}.resource"
            resource = str(payload["resource"])
        elif "resource" in defaults:
            resource_path = "execution.defaults.resource"
            resource = str(defaults["resource"])
        else:
            resource_path = None
            resource = default_resource
        if resource not in {"cpu", "gpu"}:
            location = resource_path or f"stage {stage_id!r}"
            raise ValueError(f"{location} resource must be 'cpu' or 'gpu', got {resource!r}")
        resolved[stage_id] = StageExecutionSpec(
            stage_id=stage_id,
            strategy=strategy,
            instances=instances,
            failure_policy=policy,
            mesh_override=_parse_mesh_override(payload.get("parallel")),
            gpus_per_node=(
                int(gpus_per_node) if gpus_per_node is not None else default_gpus_per_node
            ),
            partition=partition,
            resource=resource,
        )
    return resolved


def mip_resource(experiment_config: Mapping[str, Any]) -> str:
    """Return the default resource for the effective MIP/realization behavior."""

    mip_runs = _mapping(experiment_config.get("mip")).get("runs")
    if isinstance(mip_runs, Mapping) and mip_runs:
        return "cpu"
    if bool(experiment_config.get("skip_realize_model", False)):
        return "cpu"
    raw_realize_model = experiment_config.get("realize_model")
    if not isinstance(raw_realize_model, Mapping):
        return "cpu"
    realize_model = _mapping(raw_realize_model)
    return "cpu" if bool(realize_model.get("skip_validation", False)) else "gpu"


def _named_mip_validation_error(path: str, message: str) -> ValueError:
    return ValueError(
        f"{path} {message}. This campaign uses the current named width/depth MIP interface; "
        "update the named-MIP configuration, then rerun the orchestrator dry-run"
    )


def _validate_named_mip_search_domains(
    runs: Mapping[str, Any],
    *,
    widths: tuple[int, ...],
    maximum_depth: int,
) -> None:
    for run_id, raw_run in runs.items():
        if raw_run is False:
            continue
        run_path = f"mip.runs.{run_id}"
        if not isinstance(raw_run, Mapping):
            raise _named_mip_validation_error(run_path, "must be a mapping or false")
        variants = raw_run.get("variants") or {}
        if not isinstance(variants, Mapping):
            raise _named_mip_validation_error(f"{run_path}.variants", "must be a mapping")
        for variant_id, variant in variants.items():
            if not isinstance(variant, Mapping):
                raise _named_mip_validation_error(
                    f"{run_path}.variants.{variant_id}",
                    "must be a mapping",
                )
    try:
        variants = expand_mip_variants({"runs": runs})
    except (TypeError, ValueError) as error:
        raise _named_mip_validation_error("mip.runs", str(error)) from error
    for variant in variants:
        search = _mapping(variant.config.get("search_space"))
        try:
            select_mip_values(search.get("embedding"), widths, "embedding")
        except (TypeError, ValueError) as error:
            raise _named_mip_validation_error(
                variant.selector_path("embedding"),
                f"must use configured embedding_pruning.widths={list(widths)}",
            ) from error
        raw_depth = search.get("depth")
        if isinstance(raw_depth, Mapping) and set(raw_depth) != {"range"}:
            continue
        try:
            select_mip_values(raw_depth, tuple(range(maximum_depth + 1)), "depth")
        except (TypeError, ValueError) as error:
            raise _named_mip_validation_error(
                variant.selector_path("depth"),
                f"must select removals between 0 and the configured maximum {maximum_depth}",
            ) from error


def _validate_named_mip_geometry(experiment_config: Mapping[str, Any]) -> None:
    """Reject stale named-MIP bundles before worker launch."""

    mip = _mapping(experiment_config.get("mip"))
    runs = mip.get("runs")
    embedding = _mapping(experiment_config.get("embedding_pruning"))
    if not isinstance(runs, Mapping) or not runs:
        if embedding.get("enabled"):
            raise _named_mip_validation_error(
                "mip.runs",
                "must define at least one active named solve for the enabled scenario driver",
            )
        return
    if not any(run is not False for run in runs.values()):
        raise _named_mip_validation_error("mip.runs", "must define at least one active named solve")

    model_info = _mapping(experiment_config.get("model_info"))
    hidden_size = model_info.get("hidden_size")
    num_layers = model_info.get("num_hidden_layers")
    if not isinstance(hidden_size, int) or isinstance(hidden_size, bool) or hidden_size < 1:
        raise _named_mip_validation_error(
            "model_info.hidden_size",
            "must contain the positive inspected teacher width",
        )
    if not isinstance(num_layers, int) or isinstance(num_layers, bool) or num_layers < 1:
        raise _named_mip_validation_error(
            "model_info.num_hidden_layers",
            "must contain the positive inspected teacher depth",
        )

    raw_widths = embedding.get("widths")
    if (
        not embedding.get("enabled")
        or not isinstance(raw_widths, Sequence)
        or isinstance(raw_widths, (str, bytes))
    ):
        raise _named_mip_validation_error(
            "embedding_pruning",
            "must enable the scenario driver and list the inspected teacher width",
        )
    if any(
        not isinstance(width, int) or isinstance(width, bool) or width < 1 for width in raw_widths
    ):
        raise _named_mip_validation_error(
            "embedding_pruning.widths",
            "must contain only positive integer widths",
        )
    widths = tuple(int(width) for width in raw_widths)
    if not widths or hidden_size not in widths or max(widths) != hidden_size:
        raise _named_mip_validation_error(
            "embedding_pruning.widths",
            f"must include teacher hidden_size={hidden_size} as its largest width",
        )

    depth = _mapping(experiment_config.get("depth_importance") or experiment_config.get("depth"))
    maximum = depth.get("max_subblocks_to_remove", depth.get("max_removals", 0))
    granularity = str(depth.get("granularity", "block")).lower()
    teacher_depth = (
        depth.get("expected_initial_sublayers") if granularity == "subblock" else num_layers
    )
    if not isinstance(teacher_depth, int) or isinstance(teacher_depth, bool) or teacher_depth < 1:
        raise _named_mip_validation_error(
            "depth_importance.expected_initial_sublayers",
            "must contain the positive inspected teacher sublayer depth",
        )
    if (
        not isinstance(maximum, int)
        or isinstance(maximum, bool)
        or not 0 <= maximum < teacher_depth
    ):
        raise _named_mip_validation_error(
            "depth_importance.max_subblocks_to_remove",
            f"must be a non-negative count below the inspected teacher limit {teacher_depth}",
        )
    _validate_named_mip_search_domains(
        runs,
        widths=widths,
        maximum_depth=maximum,
    )


def compile_campaign_plan(
    *,
    experiment_config_path: str | Path,
    runner: RunnerEnvironment,
    execution: Mapping[str, Any],
    overrides: list[str] | None = None,
    stage_filter: str | None = None,
) -> CampaignPlan:
    """Compile one campaign plan from experiment + runner + execution configs."""

    _validate_execution_payload(execution)
    experiment_path = Path(experiment_config_path)
    experiment_config = load_experiment_config(experiment_path, overrides=overrides or [])
    default_mip_resource = mip_resource(experiment_config)
    puzzle_dir = Path(
        experiment_config.get("puzzle_dir")
        or (experiment_config.get("experiment") or {}).get("dir")
        or "."
    )
    post_mip_stages = _post_mip_stage_metadata(experiment_config)
    _validate_execution_stage_ids(
        execution,
        dynamic_stage_ids=tuple(row["stage_id"] for row in post_mip_stages),
    )
    enabled = configured_stage_ids(
        experiment_config,
        dynamic_post_mip_stage_ids=(row["stage_id"] for row in post_mip_stages),
    )
    if stage_filter and stage_filter != "full":
        if stage_filter not in enabled:
            raise ValueError(f"Stage {stage_filter!r} is not enabled in the experiment config")
        enabled = (stage_filter,)
    if "mip" in enabled:
        _validate_named_mip_geometry(experiment_config)
    dynamic_execution_defaults = {
        row["stage_id"]: row["default_strategy"] for row in post_mip_stages
    }
    dynamic_resource_defaults = {
        row["stage_id"]: row["default_resource"] for row in post_mip_stages
    }
    dynamic_resource_defaults["mip"] = default_mip_resource
    execution_specs = resolve_stage_execution_specs(
        execution,
        enabled,
        dynamic_defaults=dynamic_execution_defaults,
        dynamic_resources=dynamic_resource_defaults,
    )
    execution_defaults = _mapping(execution.get("defaults"))
    final_report = _mapping(_mapping(execution.get("stages")).get("final_report"))
    if final_report.get("partition") is not None:
        final_report_partition = normalize_slurm_partition(
            final_report["partition"], path="execution.stages.final_report.partition"
        )
    elif runner.slurm is not None and runner.slurm.partition_cpu is not None:
        final_report_partition = runner.slurm.partition_cpu
    else:
        final_report_partition = normalize_slurm_partition(
            execution_defaults.get("partition"), path="execution.defaults.partition"
        )
    distributed = set(distributed_stage_ids())
    nodes: list[StagePlanNode] = []
    post_mip_by_stage = {row["stage_id"]: row for row in post_mip_stages}

    for stage_id in enabled:
        spec = execution_specs[stage_id]
        override = None
        if spec.mesh_override is not None:
            override = {
                key: value for key, value in asdict(spec.mesh_override).items() if value is not None
            }
        dynamic = post_mip_by_stage.get(stage_id)
        if dynamic is None:
            if stage_id == "vllm_stats":
                mesh = _vllm_stage_mesh(experiment_config, override)
            else:
                mesh = extract_stage_mesh(experiment_config, stage_id, override)
        else:
            node_config = _mapping(dynamic.get("config"))
            parallel = _mapping(node_config.get("parallel"))
            if not parallel:
                parallel = _mapping(_mapping(node_config.get("automodel")).get("parallel"))
            if dynamic["node_type"] == "global_kd" and not parallel:
                global_kd = _mapping(experiment_config.get("global_distillation"))
                parallel = _mapping(_mapping(global_kd.get("automodel")).get("parallel"))
                for key in ("tp", "cp", "pp", "ep", "dp_shard", "dp_replicate", "dp"):
                    if key in node_config:
                        parallel[key] = node_config[key]
                    elif key in global_kd and key not in parallel:
                        parallel[key] = global_kd[key]
            if dynamic["node_type"] in {"aiperf", "downstream_evaluation"}:
                topology = _mapping(node_config.get("topology"))
                topology_mesh = vllm_topology_to_mesh(topology)
                if override:
                    overridden = topology_mesh.as_dict()
                    overridden.update(override)
                    if ParallelMesh.from_mapping(overridden) != topology_mesh:
                        raise ValueError(
                            f"{stage_id} execution parallel override conflicts with "
                            "its vLLM topology"
                        )
                mesh = topology_mesh
            else:
                mesh_values = ParallelMesh.from_mapping(parallel).as_dict()
                mesh_values.update(override or {})
                mesh = ParallelMesh.from_mapping(mesh_values)
        instance_count = spec.instances if spec.strategy is not ExecutionStrategy.SINGLE else 1
        if spec.resource == "cpu":
            if instance_count != 1:
                raise ValueError(
                    f"CPU stage {stage_id!r} supports one task; got instances={instance_count}"
                )
            per_instance = 0
            allocation_nodes = 1
            allocation_gpus_per_node = 0
            allocation_total_gpus = 0
            allocation_exclusive = False
        else:
            per_instance = gpus_per_instance(mesh)
            allocation = pack_gpu_allocation(
                mesh=mesh,
                instances=instance_count,
                gpus_per_node=spec.gpus_per_node
                or int(execution.get("defaults", {}).get("gpus_per_node", 8)),
            )
            allocation_nodes = allocation.nodes
            allocation_gpus_per_node = allocation.gpus_per_node
            allocation_total_gpus = allocation.total_gpus
            allocation_exclusive = allocation.exclusive
        partition = spec.partition
        stage_execution = _mapping(_mapping(execution.get("stages")).get(stage_id))
        if (
            spec.resource == "cpu"
            and runner.slurm is not None
            and runner.slurm.partition_cpu is not None
            and stage_execution.get("partition") is None
        ):
            partition = runner.slurm.partition_cpu
        nodes.append(
            StagePlanNode(
                stage_id=stage_id,
                strategy=spec.strategy,
                instances=instance_count,
                failure_policy=spec.failure_policy,
                mesh=mesh.as_dict(),
                gpus_per_instance=per_instance,
                gpus_per_node=allocation_gpus_per_node,
                nodes=allocation_nodes,
                total_gpus=allocation_total_gpus,
                exclusive=allocation_exclusive,
                parents=(
                    dynamic["parents"]
                    if dynamic is not None
                    else configured_parent_stage_ids(stage_id, experiment_config)
                ),
                distributed=(
                    False
                    if spec.resource == "cpu"
                    else (
                        bool(dynamic["distributed"])
                        if dynamic is not None
                        else stage_id in distributed
                    )
                ),
                partition=partition,
                resource=spec.resource,
            )
        )

    contract_hash = execution_contract_hash(runner)
    return CampaignPlan(
        experiment_config_path=_worker_experiment_path(experiment_path, runner),
        puzzle_dir=puzzle_dir,
        experiment_config=experiment_config,
        runner=runner,
        execution_defaults=execution_defaults,
        stages=tuple(nodes),
        contract_hash=contract_hash,
        overrides=tuple(overrides or ()),
        final_report_partition=final_report_partition,
    )


def plan_to_dict(plan: CampaignPlan) -> dict[str, Any]:
    """Serialize a campaign plan for durable storage."""

    return {
        "experiment_config_path": plan.experiment_config_path,
        "puzzle_dir": str(plan.puzzle_dir),
        "contract_hash": plan.contract_hash,
        "overrides": list(plan.overrides),
        "runner_kind": plan.runner.kind,
        "execution_defaults": dict(plan.execution_defaults),
        "final_report": {
            "resource": "cpu",
            "partition": plan.final_report_partition,
        },
        "stages": [
            {
                "stage_id": node.stage_id,
                "strategy": node.strategy.value,
                "instances": node.instances,
                "failure_policy": node.failure_policy.value,
                "mesh": dict(node.mesh),
                "gpus_per_instance": node.gpus_per_instance,
                "gpus_per_node": node.gpus_per_node,
                "nodes": node.nodes,
                "total_gpus": node.total_gpus,
                "exclusive": node.exclusive,
                "parents": list(node.parents),
                "distributed": node.distributed,
                "partition": node.partition,
                "resource": node.resource,
            }
            for node in plan.stages
        ],
    }
