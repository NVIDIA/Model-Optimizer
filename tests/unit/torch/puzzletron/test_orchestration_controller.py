# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for orchestration controller and adapters."""

from pathlib import Path

import pytest
import yaml

from puzzletron_orchestrator.adapters.registry import adapter_for_stage
from puzzletron_orchestrator.compiler import (
    compile_campaign_plan,
    load_execution_config,
    load_runner_config,
)
from puzzletron_orchestrator.controller import dry_run_plan
from puzzletron_orchestrator.executors.baremetal import GpuLeaseManager
from puzzletron_orchestrator.schema import (
    AttemptSpec,
    BareMetalHost,
    CommandSpec,
    ExecutionStrategy,
    TaskTopology,
)
from puzzletron_orchestrator.task_topology import resolve_task_topology


def _write_configs(tmp_path: Path):
    experiment = tmp_path / "experiment.yaml"
    experiment.write_text(
        yaml.safe_dump(
            {
                "experiment": {"dir": str(tmp_path / "run")},
                "convert": {"enabled": True},
                "vllm_stats": {
                    "enabled": True,
                    "runtime_stats": {
                        "topology": {
                            "tensor_parallel_size": 1,
                            "pipeline_parallel_size": 1,
                            "prefill_context_parallel_size": 1,
                            "gpu_group_size": 1,
                        }
                    },
                },
            }
        )
    )
    runner = tmp_path / "runner.yaml"
    runner.write_text(
        yaml.safe_dump(
            {
                "runner": {
                    "kind": "slurm",
                    "slurm": {"account": "test"},
                    "execution_contract": {
                        "repository": str(tmp_path),
                        "venv": str(tmp_path / ".venv"),
                    },
                }
            }
        )
    )
    execution = tmp_path / "execution.yaml"
    execution.write_text(
        yaml.safe_dump(
            {
                "execution": {
                    "defaults": {"gpus_per_node": 8},
                    "stages": {
                        "convert": {"strategy": "single", "instances": 1},
                        "vllm_stats": {"strategy": "sharded", "instances": 4},
                    },
                }
            }
        )
    )
    return experiment, runner, execution


def test_dry_run_plan_packs_vllm_shards_into_one_allocation(tmp_path: Path):
    experiment, runner_path, execution_path = _write_configs(tmp_path)
    plan = compile_campaign_plan(
        experiment_config_path=experiment,
        runner=load_runner_config(runner_path),
        execution=load_execution_config(execution_path),
    )
    submissions = dry_run_plan(plan)
    vllm = [item for item in submissions if item.stage_id == "vllm_stats"]
    assert len(vllm) == 1
    assert vllm[0].nodes == 1
    assert vllm[0].gpus == 4
    assert vllm[0].task_count == 4
    assert vllm[0].gpus_per_task == 1


def test_dry_run_plan_uses_heterogeneous_vllm_measurement_topologies(tmp_path: Path):
    experiment, runner_path, execution_path = _write_configs(tmp_path)
    experiment_config = yaml.safe_load(experiment.read_text())
    experiment_config["vllm_stats"]["runtime_stats"]["topology"].update(
        {
            "tensor_parallel_size": 2,
            "gpu_group_size": 2,
        }
    )
    experiment_config["vllm_stats"]["measurements"] = {
        "primary": {},
        "secondary": {
            "runtime_stats": {
                "topology": {
                    "tensor_parallel_size": 4,
                    "gpu_group_size": 4,
                }
            }
        },
    }
    experiment.write_text(yaml.safe_dump(experiment_config))
    plan = compile_campaign_plan(
        experiment_config_path=experiment,
        runner=load_runner_config(runner_path),
        execution=load_execution_config(execution_path),
    )
    submissions = dry_run_plan(plan)
    vllm = sorted(
        (
            item.work_id,
            item.nodes,
            item.gpus,
            item.task_count,
            item.gpus_per_task,
        )
        for item in submissions
        if item.stage_id == "vllm_stats"
    )

    assert vllm == [
        ("vllm_stats:primary:gang", 1, 8, 4, 2),
        ("vllm_stats:secondary:gang", 2, 16, 4, 4),
    ]


def test_gpu_lease_manager_allocates_disjoint_gpus(tmp_path: Path):
    manager = GpuLeaseManager(
        (BareMetalHost("node-a", 4), BareMetalHost("node-b", 4)),
        tmp_path / "leases.json",
    )
    topology = resolve_task_topology(
        AttemptSpec(
            attempt_id="topology",
            work_id="stage:0",
            stage_id="stage",
            command=CommandSpec(argv=("python", "worker.py")),
            allocation_nodes=1,
            allocation_gpus=2,
            metadata={"gpus_per_node": 4},
            task_topology=TaskTopology(task_count=1, gpus_per_task=2),
        )
    )

    first = manager.acquire_topology("a1", topology)[0]
    second = manager.acquire_topology("a2", topology)[0]

    assert first.hostname == "node-a"
    assert set(first.gpu_ids).isdisjoint(second.gpu_ids)


def test_adapter_registry_selects_sharded_adapter(tmp_path: Path):
    experiment, runner_path, execution_path = _write_configs(tmp_path)
    plan = compile_campaign_plan(
        experiment_config_path=experiment,
        runner=load_runner_config(runner_path),
        execution=load_execution_config(execution_path),
    )
    node = next(item for item in plan.stages if item.stage_id == "vllm_stats")
    adapter = adapter_for_stage(node)
    work_plan = adapter.plan(plan, node)
    assert work_plan.strategy is ExecutionStrategy.SHARDED
    assert [item.work_id for item in work_plan.items] == ["vllm_stats:default:gang"]


def test_vllm_stats_rejects_conflicting_execution_mesh(tmp_path: Path):
    experiment, runner_path, execution_path = _write_configs(tmp_path)
    experiment_config = yaml.safe_load(experiment.read_text())
    experiment_config["vllm_stats"]["measurements"] = {"latency": {}}
    experiment.write_text(yaml.safe_dump(experiment_config))
    execution = load_execution_config(execution_path)
    execution["stages"]["vllm_stats"]["parallel"] = {
        "ep": 2,
        "dp_shard": 2,
    }

    with pytest.raises(
        ValueError,
        match="vllm_stats execution parallel override conflicts",
    ):
        compile_campaign_plan(
            experiment_config_path=experiment,
            runner=load_runner_config(runner_path),
            execution=execution,
        )
