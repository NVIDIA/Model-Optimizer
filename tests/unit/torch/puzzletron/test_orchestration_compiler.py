# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for orchestration compiler and plan generation."""

from pathlib import Path

import pytest
import yaml

from modelopt.torch.puzzletron.orchestration.compiler import (
    _post_mip_stage_metadata,
    compile_campaign_plan,
    load_execution_config,
    load_runner_config,
    resolve_stage_execution_specs,
)
from modelopt.torch.puzzletron.orchestration.controller import CampaignController
from modelopt.torch.puzzletron.orchestration.schema import ExecutionStrategy, HaltPolicy


@pytest.fixture
def tmp_configs(tmp_path: Path):
    experiment = tmp_path / "experiment.yaml"
    experiment.write_text(
        yaml.safe_dump(
            {
                "experiment": {"dir": str(tmp_path / "run")},
                "pruning": {
                    "automodel": {
                        "parallel": {
                            "pp": 2,
                            "ep": 2,
                            "dp_shard": 4,
                            "tp": 1,
                            "cp": 1,
                            "dp_replicate": 1,
                        }
                    }
                },
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
                "width_importance": {"enabled": True},
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
                        "width_importance": {"strategy": "single", "instances": 1},
                        "vllm_stats": {"strategy": "sharded", "instances": 16},
                    },
                }
            }
        )
    )
    return experiment, runner, execution


def test_resolve_stage_execution_specs_assigns_default_strategies(tmp_configs):
    _, _, execution_path = tmp_configs
    execution = load_execution_config(execution_path)
    specs = resolve_stage_execution_specs(
        {},
        (
            "width_importance",
            "vllm_stats",
            "depth_importance",
            "replacement_scoring",
            "zero_shot_evaluation",
            "aiperf",
        ),
    )
    assert specs["width_importance"].strategy is ExecutionStrategy.SINGLE
    assert specs["vllm_stats"].strategy is ExecutionStrategy.SHARDED
    assert specs["depth_importance"].strategy is ExecutionStrategy.PERSISTENT_POOL
    assert specs["replacement_scoring"].strategy is ExecutionStrategy.PERSISTENT_POOL
    assert specs["zero_shot_evaluation"].strategy is ExecutionStrategy.SHARDED
    assert specs["aiperf"].strategy is ExecutionStrategy.SHARDED

    configured = resolve_stage_execution_specs(execution, ("vllm_stats",))
    assert configured["vllm_stats"].instances == 16


def test_compile_campaign_plan_packs_vllm_stats_instances(tmp_configs):
    experiment_path, runner_path, execution_path = tmp_configs
    runner = load_runner_config(runner_path)
    execution = load_execution_config(execution_path)
    plan = compile_campaign_plan(
        experiment_config_path=experiment_path,
        runner=runner,
        execution=execution,
    )
    vllm_node = next(node for node in plan.stages if node.stage_id == "vllm_stats")
    assert vllm_node.instances == 16
    assert vllm_node.gpus_per_instance == 1
    assert vllm_node.nodes == 2
    assert vllm_node.total_gpus == 16


def test_compile_campaign_plan_preserves_drain_halt_policy(tmp_configs):
    experiment_path, runner_path, execution_path = tmp_configs
    runner = load_runner_config(runner_path)
    execution = load_execution_config(execution_path)
    execution["defaults"]["halt_policy"] = "drain"

    plan = compile_campaign_plan(
        experiment_config_path=experiment_path,
        runner=runner,
        execution=execution,
    )
    controller = CampaignController(plan)

    assert plan.execution_defaults["halt_policy"] == "drain"
    assert controller._halt_policy is HaltPolicy.DRAIN


def test_compile_campaign_plan_uses_cpu_partition_without_gpus(tmp_configs):
    experiment_path, runner_path, execution_path = tmp_configs
    runner_payload = yaml.safe_load(runner_path.read_text())
    runner_payload["runner"]["slurm"]["partition_cpu"] = "cpu"
    runner_path.write_text(yaml.safe_dump(runner_payload))
    execution_payload = yaml.safe_load(execution_path.read_text())
    execution_payload["execution"]["stages"]["convert"] = {
        "strategy": "single",
        "resource": "cpu",
    }
    execution_path.write_text(yaml.safe_dump(execution_payload))

    plan = compile_campaign_plan(
        experiment_config_path=experiment_path,
        runner=load_runner_config(runner_path),
        execution=load_execution_config(execution_path),
    )
    convert = next(node for node in plan.stages if node.stage_id == "convert")

    assert convert.resource == "cpu"
    assert convert.partition == "cpu"
    assert convert.gpus_per_instance == 0
    assert convert.total_gpus == 0
    assert convert.nodes == 1


def test_post_mip_compiler_topologically_orders_serialized_nodes() -> None:
    config = {
        "mip": {"runs": {"memory": {}}},
        "post_mip": {
            "flows": {
                "memory": {
                    "source": {"run": "memory"},
                    # JSON manifests sort these keys alphabetically.
                    "nodes": {
                        "best": {
                            "type": "filter",
                            "input": "final_eval",
                            "metric": "final_eval.kl_div",
                        },
                        "final_eval": {"type": "evaluation", "input": "initial"},
                        "initial": {"type": "filter", "metric": "mip.score"},
                    },
                }
            }
        },
    }

    stages = _post_mip_stage_metadata(config)

    assert [stage["node_id"] for stage in stages] == ["initial", "final_eval", "best"]
