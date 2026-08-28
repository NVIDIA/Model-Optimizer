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
from collections.abc import Sequence
from dataclasses import asdict
from difflib import get_close_matches
from pathlib import Path
from typing import Any, Mapping

import yaml

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
    distributed_stage_ids,
    stage_ids,
    topological_mapping_items,
)
from .vllm_measurements import normalize_vllm_measurements

__all__ = [
    "compile_campaign_plan",
    "load_execution_config",
    "load_runner_config",
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
    "partition",
    "partition_interactive",
    "partition_batch",
    "partition_cpu",
    "interactive_max_nodes",
    "max_nodes",
    "time_limit",
    "qos",
    "log_dir",
}
_INVENTORY_FIELDS = {"hosts", "rendezvous_host", "rendezvous_port_base"}
_HOST_FIELDS = {"hostname", "gpus"}
_EXECUTION_FIELDS = {"defaults", "stages"}
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

_POST_MIP_NODE_METADATA: dict[str, dict[str, Any]] = {
    "filter": {"kind": "selector", "accepts": {"config", "checkpoint"}},
    "manual_filter": {"kind": "selector", "accepts": {"config", "checkpoint"}},
    "materialize": {
        "kind": "transformer",
        "accepts": {"config", "checkpoint"},
        "output": "checkpoint",
    },
    "evaluation": {"kind": "evaluator", "accepts": {"config", "checkpoint"}},
    "aiperf": {"kind": "evaluator", "accepts": {"checkpoint"}},
    "global_kd": {
        "kind": "transformer",
        "accepts": {"checkpoint"},
        "output": "checkpoint",
    },
    "ptq": {
        "kind": "transformer",
        "accepts": {"checkpoint"},
        "output": "checkpoint",
        "implemented": False,
    },
    "downstream_evaluation": {
        "kind": "evaluator",
        "accepts": {"checkpoint"},
    },
}


def _post_mip_stage_metadata(config: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    """Compile dependency-light dynamic stage facts for the controller.

    Full node/config/artifact validation runs again in the worker through the
    canonical post-MIP registry. Keeping this pass dependency-light lets plan
    compilation continue to run on login nodes without importing PyTorch.
    """

    post_mip = config.get("post_mip") or {}
    if not isinstance(post_mip, Mapping):
        raise TypeError("post_mip must be a mapping")
    if set(post_mip) - {"flows"}:
        raise ValueError(f"unknown post_mip fields: {sorted(set(post_mip) - {'flows'})}")
    flows = post_mip.get("flows") or {}
    if not isinstance(flows, Mapping):
        raise TypeError("post_mip.flows must be a mapping")
    compiled: list[dict[str, Any]] = []
    global_node_ids: set[str] = set()
    for flow_id, flow_value in flows.items():
        if not str(flow_id) or any(
            character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
            for character in str(flow_id)
        ):
            raise ValueError(f"invalid post-MIP flow ID {flow_id!r}")
        if not isinstance(flow_value, Mapping):
            raise TypeError(f"post-MIP flow {flow_id!r} must be a mapping")
        if set(flow_value) - {"source", "nodes"}:
            raise ValueError(f"unknown fields in post-MIP flow {flow_id!r}")
        source = flow_value.get("source") or {}
        if not isinstance(source, Mapping) or not source.get("run"):
            raise ValueError(f"post-MIP flow {flow_id!r} must select one source.run")
        mip_runs = _mapping(_mapping(config.get("mip")).get("runs"))
        if source["run"] not in mip_runs or mip_runs[source["run"]] is False:
            raise ValueError(
                f"post-MIP flow {flow_id!r} selects unknown or disabled MIP run {source['run']!r}"
            )
        if set(source) - {"run", "variants", "objectives"}:
            raise ValueError(f"unknown source fields in post-MIP flow {flow_id!r}")
        nodes = flow_value.get("nodes") or {}
        if not isinstance(nodes, Mapping) or not nodes:
            raise ValueError(f"post-MIP flow {flow_id!r} must contain nodes")
        prepared_nodes: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
        for node_id, node_value in nodes.items():
            if not str(node_id) or any(
                character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
                for character in str(node_id)
            ):
                raise ValueError(f"invalid post-MIP node ID {node_id!r}")
            if not isinstance(node_value, Mapping):
                raise TypeError(f"post-MIP node {flow_id}.{node_id} must be a mapping")
            if str(node_id) in global_node_ids:
                raise ValueError(
                    f"post-MIP node IDs must be campaign-unique; duplicate {node_id!r}"
                )
            global_node_ids.add(str(node_id))
            node_type = str(node_value.get("type") or "")
            node_metadata = _POST_MIP_NODE_METADATA.get(node_type)
            if node_metadata is None:
                raise ValueError(f"unknown post-MIP node type {node_type!r}")
            if not node_metadata.get("implemented", True):
                raise NotImplementedError(
                    f"post-MIP node type {node_type!r} is declared but not implemented"
                )
            prepared_nodes[str(node_id)] = (dict(node_value), dict(node_metadata))

        def dependency_ids(
            _node_id: str,
            prepared: tuple[dict[str, Any], dict[str, Any]],
        ) -> tuple[str, ...]:
            node_value, _metadata = prepared
            dependencies = []
            input_id = str(node_value.get("input", "source"))
            if input_id != "source":
                dependencies.append(input_id)
            model_source = str(node_value.get("model_source", "latest"))
            if model_source not in {"latest", "origin"}:
                dependencies.append(model_source)
            if str(node_value.get("type")) == "filter":
                references = []
                if node_value.get("metric"):
                    references.append(str(node_value["metric"]))
                for entry in node_value.get("metrics") or ():
                    if isinstance(entry, Mapping) and entry.get("metric"):
                        references.append(str(entry["metric"]))
                for reference in references:
                    owner, separator, _metric = reference.partition(".")
                    if separator and owner != "mip":
                        dependencies.append(owner)
            return tuple(dependencies)

        stage_by_node = {"source": "mip"}
        artifact_by_node = {"source": {"config", "checkpoint"}}
        kind_by_node = {"source": "transformer"}
        for node_id, prepared in topological_mapping_items(prepared_nodes, dependency_ids):
            node_value, metadata = prepared
            node_type = str(node_value["type"])
            input_id = str(node_value.get("input", "source"))
            model_source = str(node_value.get("model_source", "latest"))
            if (
                model_source not in {"latest", "origin"}
                and kind_by_node[model_source] != "transformer"
            ):
                raise ValueError(f"model_source {model_source!r} is not a transformer node")
            source_artifacts = (
                artifact_by_node[input_id]
                if model_source == "latest"
                else artifact_by_node["source"]
                if model_source == "origin"
                else artifact_by_node[model_source]
            )
            dependency_stages = [stage_by_node[input_id]]
            if model_source not in {"latest", "origin"}:
                dependency_stages.append(stage_by_node[model_source])
            if metadata["kind"] != "selector" and not source_artifacts <= metadata["accepts"]:
                raise ValueError(
                    f"post-MIP node {flow_id}.{node_id} cannot consume "
                    f"{sorted(source_artifacts)}; add an explicit materialize node"
                )
            if node_type == "filter":
                references = []
                if node_value.get("metric"):
                    references.append(str(node_value["metric"]))
                for entry in node_value.get("metrics") or ():
                    if isinstance(entry, Mapping) and entry.get("metric"):
                        references.append(str(entry["metric"]))
                for reference in references:
                    owner, separator, _metric = reference.partition(".")
                    if not separator or (owner != "mip" and owner not in stage_by_node):
                        raise ValueError(
                            f"post-MIP node {flow_id}.{node_id} has invalid or forward "
                            f"metric reference {reference!r}"
                        )
                    if owner != "mip":
                        dependency_stages.append(stage_by_node[owner])
            output_artifacts = (
                {metadata["output"]} if metadata.get("output") else set(artifact_by_node[input_id])
            )
            stage_id = f"post.{flow_id}.{node_id}"
            compiled.append(
                {
                    "stage_id": stage_id,
                    "node_id": str(node_id),
                    "node_type": node_type,
                    "parents": tuple(dict.fromkeys(dependency_stages)),
                    "distributed": metadata["kind"] != "selector",
                    "default_strategy": (
                        ExecutionStrategy.SINGLE
                        if metadata["kind"] == "selector"
                        else ExecutionStrategy.SHARDED
                    ),
                    "config": dict(node_value.get("config") or {}),
                }
            )
            stage_by_node[str(node_id)] = stage_id
            artifact_by_node[str(node_id)] = output_artifacts
            kind_by_node[str(node_id)] = str(metadata["kind"])
    return tuple(compiled)


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


def _validate_execution_payload(execution: Mapping[str, Any]) -> None:
    _reject_unknown_fields(execution, _EXECUTION_FIELDS, path="execution")
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
            partition=slurm_payload.get("partition"),
            partition_interactive=slurm_payload.get("partition_interactive"),
            partition_batch=slurm_payload.get("partition_batch"),
            partition_cpu=slurm_payload.get("partition_cpu"),
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
        resource = str(payload.get("resource", defaults.get("resource", "gpu")))
        if resource not in {"cpu", "gpu"}:
            raise ValueError(
                f"stage {stage_id!r} resource must be 'cpu' or 'gpu', got {resource!r}"
            )
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
    dynamic_execution_defaults = {
        row["stage_id"]: row["default_strategy"] for row in post_mip_stages
    }
    execution_specs = resolve_stage_execution_specs(
        execution,
        enabled,
        dynamic_defaults=dynamic_execution_defaults,
    )
    execution_defaults = _mapping(execution.get("defaults"))
    final_report = _mapping(_mapping(execution.get("stages")).get("final_report"))
    final_report_partition_path = (
        "execution.stages.final_report.partition"
        if "partition" in final_report
        else "execution.defaults.partition"
    )
    final_report_partition = normalize_slurm_partition(
        final_report.get("partition", execution_defaults.get("partition")),
        path=final_report_partition_path,
    )
    if final_report_partition is None and runner.slurm is not None:
        final_report_partition = runner.slurm.partition_cpu
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
        if partition is None and spec.resource == "cpu" and runner.slurm is not None:
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
