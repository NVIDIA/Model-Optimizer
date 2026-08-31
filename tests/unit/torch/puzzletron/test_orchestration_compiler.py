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

"""Tests for orchestration compiler and plan generation."""

from pathlib import Path

import pytest
import yaml

from modelopt.torch.puzzletron.orchestration.compiler import (
    _post_mip_stage_metadata,
    compile_campaign_plan,
    load_execution_config,
    load_runner_config,
    plan_to_dict,
    resolve_stage_execution_specs,
)
from modelopt.torch.puzzletron.orchestration.controller import CampaignController
from modelopt.torch.puzzletron.orchestration.identity import execution_contract_hash, hash_payload
from modelopt.torch.puzzletron.orchestration.schema import (
    ExecutionContract,
    ExecutionStrategy,
    HaltPolicy,
    RunnerEnvironment,
    SlurmRunnerConfig,
)


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


def test_runner_config_rejects_unknown_field_with_suggestion(tmp_configs) -> None:
    _, runner_path, _ = tmp_configs
    payload = yaml.safe_load(runner_path.read_text())
    payload["runner"]["slurm"]["partition_name"] = "gpu"
    runner_path.write_text(yaml.safe_dump(payload))

    with pytest.raises(
        ValueError,
        match="runner.slurm.partition_name; did you mean 'partition'",
    ):
        load_runner_config(runner_path)


def test_runner_config_rejects_unused_defaults_mapping(tmp_configs) -> None:
    _, runner_path, _ = tmp_configs
    payload = yaml.safe_load(runner_path.read_text())
    payload["runner"]["defaults"] = {"partition": "ignored"}
    runner_path.write_text(yaml.safe_dump(payload))

    with pytest.raises(ValueError, match=r"Unknown config field runner\.defaults"):
        load_runner_config(runner_path)


def test_runner_config_preserves_legacy_partition_routing(tmp_configs) -> None:
    _, runner_path, _ = tmp_configs
    payload = yaml.safe_load(runner_path.read_text())
    payload["runner"]["slurm"].update(
        {
            "partition_interactive": "interactive",
            "partition_batch": "batch",
            "partition_cpu": "cpu",
            "interactive_max_nodes": 2,
        }
    )
    runner_path.write_text(yaml.safe_dump(payload))

    runner = load_runner_config(runner_path)

    assert runner.slurm is not None
    assert runner.slurm.partition_for_nodes(1) == "interactive"
    assert runner.slurm.partition_for_nodes(3) == "batch"
    assert runner.slurm.partition_cpu == "cpu"


def test_runner_config_normalizes_multiple_eligible_partitions(tmp_configs) -> None:
    _, runner_path, _ = tmp_configs
    payload = yaml.safe_load(runner_path.read_text())
    payload["runner"]["slurm"]["partition"] = ["gpu-a", "gpu-b"]
    runner_path.write_text(yaml.safe_dump(payload))

    runner = load_runner_config(runner_path)

    assert runner.slurm is not None
    assert runner.slurm.partition == "gpu-a,gpu-b"


def test_partition_set_changes_slurm_execution_contract_identity() -> None:
    first = RunnerEnvironment(
        kind="slurm",
        contract=ExecutionContract(repository="/repo", venv="/venv"),
        slurm=SlurmRunnerConfig(account="acct", partition=["gpu-a", "gpu-b"]),
    )
    second = RunnerEnvironment(
        kind="slurm",
        contract=ExecutionContract(repository="/repo", venv="/venv"),
        slurm=SlurmRunnerConfig(account="acct", partition=["gpu-a", "gpu-c"]),
    )

    assert execution_contract_hash(first) != execution_contract_hash(second)


def test_partition_schema_migration_changes_slurm_execution_contract_identity() -> None:
    runner = RunnerEnvironment(
        kind="slurm",
        contract=ExecutionContract(repository="/repo", venv="/venv"),
        slurm=SlurmRunnerConfig(account="acct", partition="batch"),
    )
    legacy_identity = hash_payload(
        {
            "repository": "/repo",
            "venv": "/venv",
            "container": None,
            "container_mounts": None,
            "setup_env": None,
            "prerun_commands": [],
            "postrun_commands": [],
            "runner_kind": "slurm",
            "task_topology_contract": 1,
            "slurm": {
                "account": "acct",
                "partition_interactive": None,
                "partition_batch": "batch",
                "partition_cpu": None,
                "interactive_max_nodes": 2,
                "max_nodes": None,
                "time_limit": "4:00:00",
                "qos": None,
            },
        }
    )

    assert execution_contract_hash(runner) != legacy_identity


def test_runner_config_rejects_duplicate_partitions(tmp_configs) -> None:
    _, runner_path, _ = tmp_configs
    payload = yaml.safe_load(runner_path.read_text())
    payload["runner"]["slurm"]["partition"] = ["gpu", "gpu"]
    runner_path.write_text(yaml.safe_dump(payload))

    with pytest.raises(
        ValueError,
        match=r"runner\.slurm\.partition contains duplicate partition names",
    ):
        load_runner_config(runner_path)


def test_runner_config_rejects_partition_directive_injection(tmp_configs) -> None:
    _, runner_path, _ = tmp_configs
    payload = yaml.safe_load(runner_path.read_text())
    payload["runner"]["slurm"]["partition"] = "gpu\n#SBATCH --qos=unexpected"
    runner_path.write_text(yaml.safe_dump(payload))

    with pytest.raises(
        ValueError,
        match=r"runner\.slurm\.partition contains an invalid partition name",
    ):
        load_runner_config(runner_path)


def test_runner_config_rejects_job_name_prefix_directive_injection(tmp_configs) -> None:
    _experiment_path, runner_path, _execution_path = tmp_configs
    payload = yaml.safe_load(runner_path.read_text())
    payload["runner"]["slurm"]["job_name_prefix"] = "trusted\n#SBATCH --qos=unexpected"
    runner_path.write_text(yaml.safe_dump(payload))

    with pytest.raises(ValueError, match=r"runner\.slurm\.job_name_prefix"):
        load_runner_config(runner_path)


@pytest.mark.parametrize("scope", ["defaults", "stage"])
def test_execution_config_rejects_partition_directive_injection(tmp_configs, scope: str) -> None:
    _, _, execution_path = tmp_configs
    payload = yaml.safe_load(execution_path.read_text())
    if scope == "defaults":
        payload["execution"]["defaults"]["partition"] = "gpu\n#SBATCH --qos=unexpected"
        error_path = r"execution\.defaults\.partition"
    else:
        payload["execution"]["stages"]["vllm_stats"]["partition"] = "gpu\n#SBATCH --qos=unexpected"
        error_path = r"execution\.stages\.vllm_stats\.partition"
    execution_path.write_text(yaml.safe_dump(payload))

    with pytest.raises(ValueError, match=rf"{error_path} contains an invalid partition name"):
        load_execution_config(execution_path)


def test_runner_config_rejects_invalid_command_sequence(tmp_configs) -> None:
    _, runner_path, _ = tmp_configs
    payload = yaml.safe_load(runner_path.read_text())
    payload["runner"]["execution_contract"]["prerun_commands"] = ["module load cuda", 7]
    runner_path.write_text(yaml.safe_dump(payload))

    with pytest.raises(TypeError, match="prerun_commands must be a string or a sequence"):
        load_runner_config(runner_path)


@pytest.mark.parametrize(
    ("canonical", "legacy", "value"),
    [
        ("container_mounts", "mounts", ["/host:/container"]),
        ("prerun_commands", "prerun", ["module load cuda"]),
        ("postrun_commands", "postrun", ["echo done"]),
    ],
)
def test_runner_config_rejects_canonical_and_legacy_contract_fields(
    tmp_configs, canonical: str, legacy: str, value: list[str]
) -> None:
    _, runner_path, _ = tmp_configs
    payload = yaml.safe_load(runner_path.read_text())
    contract = payload["runner"]["execution_contract"]
    contract[canonical] = value
    contract[legacy] = value
    runner_path.write_text(yaml.safe_dump(payload))

    with pytest.raises(
        ValueError,
        match=rf"cannot set both {canonical} and legacy {legacy}",
    ):
        load_runner_config(runner_path)


def test_execution_config_rejects_unknown_nested_field(tmp_configs) -> None:
    _, _, execution_path = tmp_configs
    payload = yaml.safe_load(execution_path.read_text())
    payload["execution"]["stages"]["width_importance"]["instance_count"] = 2
    execution_path.write_text(yaml.safe_dump(payload))

    with pytest.raises(
        ValueError,
        match="width_importance.instance_count; did you mean 'instances'",
    ):
        load_execution_config(execution_path)


def test_execution_config_rejects_non_partition_final_report_fields(tmp_configs) -> None:
    _, _, execution_path = tmp_configs
    payload = yaml.safe_load(execution_path.read_text())
    payload["execution"]["stages"]["final_report"] = {"resource": "cpu"}
    execution_path.write_text(yaml.safe_dump(payload))

    with pytest.raises(
        ValueError,
        match=r"Unknown config field execution\.stages\.final_report\.resource",
    ):
        load_execution_config(execution_path)


def test_execution_config_rejects_fractional_instance_count(tmp_configs) -> None:
    _, _, execution_path = tmp_configs
    payload = yaml.safe_load(execution_path.read_text())
    payload["execution"]["stages"]["width_importance"]["instances"] = 1.5
    execution_path.write_text(yaml.safe_dump(payload))

    with pytest.raises(TypeError, match="width_importance.instances must be a positive integer"):
        load_execution_config(execution_path)


def test_execution_config_rejects_model_runtime_field_in_allocation_mesh(
    tmp_configs,
) -> None:
    _, _, execution_path = tmp_configs
    payload = yaml.safe_load(execution_path.read_text())
    payload["execution"]["stages"]["width_importance"]["parallel"] = {
        "tp": 1,
        "sequence_parallel": True,
    }
    execution_path.write_text(yaml.safe_dump(payload))

    with pytest.raises(
        ValueError,
        match="sequence_parallel belongs in the experiment model-parallel profile",
    ):
        load_execution_config(execution_path)


def test_compile_campaign_plan_rejects_unknown_execution_stage(tmp_configs) -> None:
    experiment_path, runner_path, execution_path = tmp_configs
    execution = load_execution_config(execution_path)
    execution["stages"]["width_importnace"] = {"strategy": "single", "instances": 1}

    with pytest.raises(ValueError, match="width_importnace; did you mean 'width_importance'"):
        compile_campaign_plan(
            experiment_config_path=experiment_path,
            runner=load_runner_config(runner_path),
            execution=execution,
        )


def test_execution_config_rejects_single_strategy_with_multiple_instances(
    tmp_configs,
) -> None:
    experiment_path, runner_path, execution_path = tmp_configs
    execution = load_execution_config(execution_path)
    execution["stages"]["width_importance"]["instances"] = 2

    with pytest.raises(ValueError, match="must be 1 for strategy 'single'"):
        compile_campaign_plan(
            experiment_config_path=experiment_path,
            runner=load_runner_config(runner_path),
            execution=execution,
        )


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


def test_external_campaign_config_keeps_shared_absolute_path(tmp_configs) -> None:
    experiment_path, runner_path, execution_path = tmp_configs
    runner_payload = yaml.safe_load(runner_path.read_text())
    runner_payload["runner"]["execution_contract"]["repository"] = "/worker/modelopt"
    runner_path.write_text(yaml.safe_dump(runner_payload))

    plan = compile_campaign_plan(
        experiment_config_path=experiment_path,
        runner=load_runner_config(runner_path),
        execution=load_execution_config(execution_path),
    )

    assert plan.experiment_config_path == str(experiment_path.resolve())


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
    assert controller.artifact_settling_timeout_seconds == 300.0


def test_compile_campaign_plan_configures_artifact_settling_timeout(tmp_configs):
    experiment_path, runner_path, execution_path = tmp_configs
    execution = load_execution_config(execution_path)
    execution["defaults"]["artifact_settling_timeout_seconds"] = 45

    plan = compile_campaign_plan(
        experiment_config_path=experiment_path,
        runner=load_runner_config(runner_path),
        execution=execution,
    )

    assert plan.execution_defaults["artifact_settling_timeout_seconds"] == 45
    assert CampaignController(plan).artifact_settling_timeout_seconds == 45.0


@pytest.mark.parametrize(
    ("value", "error_type"),
    [
        (True, TypeError),
        ("300", TypeError),
        (0, ValueError),
        (-1, ValueError),
        (float("nan"), ValueError),
        (float("inf"), ValueError),
    ],
)
def test_compile_campaign_plan_rejects_invalid_artifact_settling_timeout(
    tmp_configs, value, error_type
):
    experiment_path, runner_path, execution_path = tmp_configs
    execution = load_execution_config(execution_path)
    execution["defaults"]["artifact_settling_timeout_seconds"] = value

    with pytest.raises(error_type, match="artifact_settling_timeout_seconds"):
        compile_campaign_plan(
            experiment_config_path=experiment_path,
            runner=load_runner_config(runner_path),
            execution=execution,
        )


def test_compile_campaign_plan_uses_stage_partition_list_without_gpus(tmp_configs):
    experiment_path, runner_path, execution_path = tmp_configs
    execution_payload = yaml.safe_load(execution_path.read_text())
    execution_payload["execution"]["stages"]["convert"] = {
        "strategy": "single",
        "resource": "cpu",
        "partition": ["cpu-a", "cpu-b"],
    }
    execution_path.write_text(yaml.safe_dump(execution_payload))

    plan = compile_campaign_plan(
        experiment_config_path=experiment_path,
        runner=load_runner_config(runner_path),
        execution=load_execution_config(execution_path),
    )
    convert = next(node for node in plan.stages if node.stage_id == "convert")

    assert convert.resource == "cpu"
    assert convert.partition == "cpu-a,cpu-b"
    assert convert.gpus_per_instance == 0
    assert convert.total_gpus == 0
    assert convert.nodes == 1


def test_compile_campaign_plan_migrates_legacy_cpu_partition(tmp_configs):
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

    assert convert.partition == "cpu"
    assert plan.final_report_partition == "cpu"


def test_compile_campaign_plan_routes_final_report_to_eligible_cpu_partitions(tmp_configs):
    experiment_path, runner_path, execution_path = tmp_configs
    execution_payload = yaml.safe_load(execution_path.read_text())
    execution_payload["execution"]["stages"]["final_report"] = {"partition": ["cpu-a", "cpu-b"]}
    execution_path.write_text(yaml.safe_dump(execution_payload))

    plan = compile_campaign_plan(
        experiment_config_path=experiment_path,
        runner=load_runner_config(runner_path),
        execution=load_execution_config(execution_path),
    )

    assert plan.final_report_partition == "cpu-a,cpu-b"
    assert plan_to_dict(plan)["final_report"] == {
        "resource": "cpu",
        "partition": "cpu-a,cpu-b",
    }


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


def test_compile_campaign_plan_allocates_downstream_evaluation_from_vllm_topology(
    tmp_configs,
) -> None:
    experiment_path, runner_path, execution_path = tmp_configs
    experiment = yaml.safe_load(experiment_path.read_text())
    experiment.update(
        {
            "mip": {"runs": {"runtime": {}}},
            "post_mip": {
                "flows": {
                    "runtime": {
                        "source": {"run": "runtime"},
                        "nodes": {
                            "materialized": {"type": "materialize"},
                            "lmms_eval": {
                                "type": "downstream_evaluation",
                                "input": "materialized",
                                "config": {
                                    "tasks": ["ifeval"],
                                    "topology": {
                                        "tensor_parallel_size": 4,
                                        "pipeline_parallel_size": 2,
                                        "data_parallel_size": 1,
                                        "prefill_context_parallel_size": 1,
                                        "decode_context_parallel_size": 1,
                                        "enable_expert_parallel": False,
                                        "gpu_group_size": 8,
                                    },
                                },
                            },
                        },
                    }
                }
            },
        }
    )
    experiment_path.write_text(yaml.safe_dump(experiment))
    execution = yaml.safe_load(execution_path.read_text())
    execution["execution"]["stages"]["post.runtime.lmms_eval"] = {
        "strategy": "sharded",
        "instances": 2,
    }
    execution_path.write_text(yaml.safe_dump(execution))

    plan = compile_campaign_plan(
        experiment_config_path=experiment_path,
        runner=load_runner_config(runner_path),
        execution=load_execution_config(execution_path),
        stage_filter="post.runtime.lmms_eval",
    )
    node = plan.stages[0]

    assert node.stage_id == "post.runtime.lmms_eval"
    assert node.parents == ("post.runtime.materialized",)
    assert node.gpus_per_instance == 8
    assert node.instances == 2
    assert node.nodes == 2
