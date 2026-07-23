# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compile experiment, runner, and execution configs into a CampaignPlan."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import yaml

from .config import load_experiment_config
from .identity import execution_contract_hash, with_contract_hash
from .mesh import ParallelMesh, extract_stage_mesh, gpus_per_instance, pack_gpu_allocation
from .schema import (
    BareMetalHost,
    BareMetalRunnerConfig,
    CampaignPlan,
    ExecutionContract,
    ExecutionStrategy,
    FailurePolicy,
    ParallelMeshOverride,
    RunnerEnvironment,
    SlurmRunnerConfig,
    StageExecutionSpec,
    StagePlanNode,
)
from .stages import (
    configured_stage_ids,
    distributed_stage_ids,
    selected_parent_stage_ids,
    topological_mapping_items,
)

__all__ = [
    "compile_campaign_plan",
    "load_execution_config",
    "load_runner_config",
    "plan_to_dict",
    "resolve_stage_execution_specs",
]

_DEFAULT_STAGE_STRATEGIES: dict[str, ExecutionStrategy] = {
    "vllm_stats": ExecutionStrategy.SHARDED,
    "replacement_scoring": ExecutionStrategy.PERSISTENT_POOL,
    "depth_importance": ExecutionStrategy.PERSISTENT_POOL,
    "zero_shot_evaluation": ExecutionStrategy.SHARDED,
    "aiperf": ExecutionStrategy.SHARDED,
}

def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


_POST_MIP_NODE_METADATA = {
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
        "implemented": False,
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
            character
            not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
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
                f"post-MIP flow {flow_id!r} selects unknown or disabled MIP run "
                f"{source['run']!r}"
            )
        if set(source) - {"run", "variants", "objectives"}:
            raise ValueError(f"unknown source fields in post-MIP flow {flow_id!r}")
        nodes = flow_value.get("nodes") or {}
        if not isinstance(nodes, Mapping) or not nodes:
            raise ValueError(f"post-MIP flow {flow_id!r} must contain nodes")
        prepared_nodes: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
        for node_id, node_value in nodes.items():
            if not str(node_id) or any(
                character
                not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
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
            metadata = _POST_MIP_NODE_METADATA.get(node_type)
            if metadata is None:
                raise ValueError(f"unknown post-MIP node type {node_type!r}")
            if not metadata.get("implemented", True):
                raise NotImplementedError(
                    f"post-MIP node type {node_type!r} is declared but not implemented"
                )
            prepared_nodes[str(node_id)] = (dict(node_value), metadata)

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
                {metadata["output"]}
                if metadata.get("output")
                else set(artifact_by_node[input_id])
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


def _load_yaml(path: str | Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text())
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return payload


def load_runner_config(path: str | Path) -> RunnerEnvironment:
    """Load a runner environment YAML file."""

    payload = _load_yaml(path)
    runner = _mapping(payload.get("runner"))
    kind = str(runner.get("kind", "slurm"))
    contract_payload = _mapping(runner.get("execution_contract"))
    prerun = contract_payload.get("prerun_commands") or contract_payload.get("prerun") or ()
    postrun = contract_payload.get("postrun_commands") or contract_payload.get("postrun") or ()
    if isinstance(prerun, str):
        prerun = (prerun,)
    if isinstance(postrun, str):
        postrun = (postrun,)
    contract = ExecutionContract(
        repository=str(contract_payload.get("repository", ".")),
        venv=str(contract_payload.get("venv", ".venv")),
        container=contract_payload.get("container"),
        container_mounts=contract_payload.get("container_mounts") or contract_payload.get("mounts"),
        setup_env=contract_payload.get("setup_env"),
        prerun_commands=tuple(str(item) for item in prerun),
        postrun_commands=tuple(str(item) for item in postrun),
    )
    slurm = None
    baremetal = None
    if kind == "slurm":
        slurm_payload = _mapping(runner.get("slurm"))
        slurm = SlurmRunnerConfig(
            account=str(slurm_payload.get("account", "")),
            partition=str(
                slurm_payload.get(
                    "partition",
                    slurm_payload.get("partition_batch", "batch"),
                )
            ),
            partition_interactive=slurm_payload.get("partition_interactive"),
            partition_batch=slurm_payload.get("partition_batch"),
            partition_cpu=slurm_payload.get("partition_cpu"),
            interactive_max_nodes=int(slurm_payload.get("interactive_max_nodes", 2)),
            max_nodes=(
                int(slurm_payload["max_nodes"])
                if slurm_payload.get("max_nodes") is not None
                else None
            ),
            time_limit=str(slurm_payload.get("time_limit", "4:00:00")),
            qos=slurm_payload.get("qos"),
            log_dir=slurm_payload.get("log_dir"),
        )
    elif kind == "baremetal":
        inventory = _mapping(runner.get("inventory"))
        hosts = tuple(
            BareMetalHost(hostname=str(item["hostname"]), gpus=int(item.get("gpus", 8)))
            for item in inventory.get("hosts", [])
        )
        baremetal = BareMetalRunnerConfig(
            hosts=hosts,
            rendezvous_host=inventory.get("rendezvous_host"),
            rendezvous_port_base=int(inventory.get("rendezvous_port_base", 29500)),
        )
    else:
        raise ValueError(f"Unsupported runner kind: {kind}")

    environment = RunnerEnvironment(
        kind=kind,
        contract=contract,
        slurm=slurm,
        baremetal=baremetal,
        defaults=_mapping(runner.get("defaults")),
    )
    updated_contract = with_contract_hash(environment)
    return RunnerEnvironment(
        kind=environment.kind,
        contract=updated_contract,
        slurm=environment.slurm,
        baremetal=environment.baremetal,
        defaults=environment.defaults,
    )


def load_execution_config(path: str | Path) -> dict[str, Any]:
    """Load execution semantics YAML."""

    payload = _load_yaml(path)
    return _mapping(payload.get("execution"))


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


def resolve_stage_execution_specs(
    execution: Mapping[str, Any],
    enabled_stages: tuple[str, ...],
    *,
    dynamic_defaults: Mapping[str, ExecutionStrategy] | None = None,
) -> dict[str, StageExecutionSpec]:
    """Resolve per-stage execution specs with defaults."""

    defaults = _mapping(execution.get("defaults"))
    default_gpus_per_node = int(defaults.get("gpus_per_node", 8))
    default_policy = FailurePolicy(str(defaults.get("failure_policy", FailurePolicy.STRICT.value)))
    stage_payload = _mapping(execution.get("stages"))
    dynamic_defaults = dict(dynamic_defaults or {})
    resolved: dict[str, StageExecutionSpec] = {}

    for stage_id in enabled_stages:
        payload = _mapping(stage_payload.get(stage_id))
        strategy_name = payload.get("strategy")
        if strategy_name is None:
            if stage_id in dynamic_defaults:
                strategy = dynamic_defaults[stage_id]
            else:
                strategy = _DEFAULT_STAGE_STRATEGIES.get(
                    stage_id, ExecutionStrategy.SINGLE
                )
        else:
            strategy = ExecutionStrategy(str(strategy_name))
        instances = int(payload.get("instances", 1))
        if strategy is ExecutionStrategy.SHARDED and instances == 1:
            instances = int(payload.get("instances", payload.get("num_jobs", 1)))
        policy = FailurePolicy(str(payload.get("failure_policy", default_policy.value)))
        gpus_per_node = payload.get("gpus_per_node", defaults.get("gpus_per_node"))
        partition = payload.get("partition", defaults.get("partition"))
        resource = str(payload.get("resource", defaults.get("resource", "gpu")))
        if resource not in {"cpu", "gpu"}:
            raise ValueError(
                f"stage {stage_id!r} resource must be 'cpu' or 'gpu', got {resource!r}"
            )
        resolved[stage_id] = StageExecutionSpec(
            stage_id=stage_id,
            strategy=strategy,
            instances=max(1, instances),
            failure_policy=policy,
            mesh_override=_parse_mesh_override(payload.get("parallel")),
            gpus_per_node=(
                int(gpus_per_node) if gpus_per_node is not None else default_gpus_per_node
            ),
            partition=str(partition) if partition is not None else None,
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

    experiment_path = Path(experiment_config_path)
    experiment_config = load_experiment_config(experiment_path, overrides=overrides or [])
    puzzle_dir = Path(
        experiment_config.get("puzzle_dir")
        or (experiment_config.get("experiment") or {}).get("dir")
        or "."
    )
    post_mip_stages = _post_mip_stage_metadata(experiment_config)
    enabled = configured_stage_ids(
        experiment_config,
        dynamic_post_mip_stage_ids=(row["stage_id"] for row in post_mip_stages),
    )
    if stage_filter and stage_filter != "full":
        if stage_filter not in enabled:
            raise ValueError(f"Stage {stage_filter!r} is not enabled in the experiment config")
        enabled = (stage_filter,)
    execution_specs = resolve_stage_execution_specs(
        execution,
        enabled,
        dynamic_defaults={
            row["stage_id"]: row["default_strategy"] for row in post_mip_stages
        },
    )
    distributed = set(distributed_stage_ids())
    default_gpus_per_node = int(_mapping(execution.get("defaults")).get("gpus_per_node", 8))
    default_policy = FailurePolicy(
        str(_mapping(execution.get("defaults")).get("failure_policy", FailurePolicy.STRICT.value))
    )
    nodes: list[StagePlanNode] = []
    post_mip_by_stage = {row["stage_id"]: row for row in post_mip_stages}

    for stage_id in enabled:
        spec = execution_specs.get(stage_id)
        if spec is None:
            strategy = (
                ExecutionStrategy.SINGLE
                if stage_id not in _DEFAULT_STAGE_STRATEGIES
                else _DEFAULT_STAGE_STRATEGIES[stage_id]
            )
            spec = StageExecutionSpec(
                stage_id=stage_id,
                strategy=strategy,
                instances=1,
                failure_policy=default_policy,
                gpus_per_node=default_gpus_per_node,
            )
        override = None
        if spec.mesh_override is not None:
            override = {
                key: value
                for key, value in asdict(spec.mesh_override).items()
                if value is not None
            }
        dynamic = post_mip_by_stage.get(stage_id)
        if dynamic is None:
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
            if dynamic["node_type"] == "aiperf":
                topology = _mapping(node_config.get("topology"))
                mesh_values = ParallelMesh(
                    tp=int(
                        topology.get(
                            "gpu_group_size",
                            topology.get("tensor_parallel_size", 1),
                        )
                    ),
                    pp=int(topology.get("pipeline_parallel_size", 1)),
                    cp=int(topology.get("prefill_context_parallel_size", 1)),
                ).as_dict()
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
                    else selected_parent_stage_ids(stage_id, experiment_config)
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
        experiment_config_path=str(experiment_path.resolve()),
        puzzle_dir=puzzle_dir,
        experiment_config=experiment_config,
        runner=runner,
        execution_defaults=_mapping(execution.get("defaults")),
        stages=tuple(nodes),
        contract_hash=contract_hash,
    )


def plan_to_dict(plan: CampaignPlan) -> dict[str, Any]:
    """Serialize a campaign plan for durable storage."""

    return {
        "experiment_config_path": plan.experiment_config_path,
        "puzzle_dir": str(plan.puzzle_dir),
        "contract_hash": plan.contract_hash,
        "runner_kind": plan.runner.kind,
        "execution_defaults": dict(plan.execution_defaults),
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
