# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for orchestration executors."""

from pathlib import Path

import yaml

from puzzletron_orchestrator.adapters.registry import adapter_for_stage
from puzzletron_orchestrator.compiler import (
    compile_campaign_plan,
    load_execution_config,
    load_runner_config,
)
from puzzletron_orchestrator.executors.local import LocalExecutor
from puzzletron_orchestrator.executors.slurm import SlurmExecutor, render_sbatch_script
from puzzletron_orchestrator.schema import (
    AttemptSpec,
    CampaignPlan,
    CommandSpec,
    ExecutionContract,
    ExecutionStrategy,
    FailurePolicy,
    RunnerEnvironment,
    SlurmRunnerConfig,
    StagePlanNode,
    TaskTopology,
)


def test_local_executor_runs_successful_command(tmp_path: Path):
    executor = LocalExecutor()
    log_path = tmp_path / "ok.log"
    attempt = AttemptSpec(
        attempt_id="a1",
        work_id="convert:0",
        stage_id="convert",
        command=CommandSpec(
            argv=("python", "-c", "print('ok')"),
            log_path=str(log_path),
        ),
    )
    handle = executor.submit(attempt)
    statuses = executor.poll([handle])
    assert statuses[0].state.value == "completed"
    assert log_path.read_text().strip() == "ok"


def test_render_sbatch_script_requests_gpus_per_node():
    runner = RunnerEnvironment(
        kind="slurm",
        contract=ExecutionContract(
            repository="/repo",
            venv="/repo/.venv",
            prerun_commands=("source /site/setup-envs.sh",),
        ),
        slurm=SlurmRunnerConfig(
            account="acct",
            partition="batch",
            partition_interactive="interactive",
            partition_batch="batch",
        ),
    )
    attempt = AttemptSpec(
        attempt_id="a1",
        work_id="vllm_stats:3",
        stage_id="vllm_stats",
        command=CommandSpec(argv=("python", "-c", "print(1)"), log_path="/tmp/out.log"),
        allocation_nodes=2,
        allocation_gpus=16,
        metadata={"gpus_per_node": 8},
    )
    script = render_sbatch_script(
        attempt=attempt,
        runner=runner,
        partition=runner.slurm.partition_for_nodes(attempt.allocation_nodes),
        account="acct",
        time_limit="4:00:00",
        qos=None,
        job_name="pt-vllm",
    )
    assert "#SBATCH --gpus-per-node=8" in script
    assert "#SBATCH --nodes=2" in script
    assert "#SBATCH --partition=interactive" in script
    assert "source /site/setup-envs.sh" in script
    assert script.startswith("#!/bin/bash\n")


def test_render_sbatch_script_omits_gpu_requests_for_cpu_stage():
    runner = RunnerEnvironment(
        kind="slurm",
        contract=ExecutionContract(repository="/repo", venv="/repo/.venv"),
        slurm=SlurmRunnerConfig(account="acct", partition_cpu="cpu"),
    )
    attempt = AttemptSpec(
        attempt_id="a1",
        work_id="mip:0",
        stage_id="mip",
        command=CommandSpec(argv=("python", "worker.py"), log_path="/tmp/out.log"),
        allocation_nodes=1,
        allocation_gpus=0,
        metadata={"gpus_per_node": 0, "partition": "cpu"},
        task_topology=TaskTopology(task_count=1, gpus_per_task=0),
    )

    script = render_sbatch_script(
        attempt=attempt,
        runner=runner,
        partition="cpu",
        account="acct",
        time_limit="4:00:00",
        qos=None,
        job_name="pt-mip",
    )

    assert "#SBATCH --partition=cpu" in script
    assert "#SBATCH --gpus" not in script
    srun = next(line for line in script.splitlines() if line.startswith("srun "))
    assert "--gpus-per-task" not in srun
    assert "--gpu-bind" not in srun


def test_render_sbatch_script_exclusive_and_qos_keep_shebang_first():
    """Regression: exclusive/qos must not break shebang via textwrap.dedent."""

    runner = RunnerEnvironment(
        kind="slurm",
        contract=ExecutionContract(
            repository="/repo",
            venv="/repo/.venv",
        ),
        slurm=SlurmRunnerConfig(
            account="acct",
            partition="batch",
            partition_interactive="interactive",
            partition_batch="batch",
        ),
    )
    attempt = AttemptSpec(
        attempt_id="a1",
        work_id="vllm_stats:pack0",
        stage_id="vllm_stats",
        command=CommandSpec(argv=("python", "-c", "print(1)"), log_path="/tmp/out.log"),
        allocation_nodes=1,
        allocation_gpus=8,
        exclusive=True,
        metadata={"gpus_per_node": 8},
    )
    script = render_sbatch_script(
        attempt=attempt,
        runner=runner,
        partition="batch",
        account="acct",
        time_limit="4:00:00",
        qos="normal",
        job_name="pt-vllm",
    )
    assert script.startswith("#!/bin/bash\n")
    assert "#SBATCH --exclusive" in script
    assert "#SBATCH --qos=normal" in script
    assert "#SBATCH --gpus-per-node=8" in script
    assert not script.startswith(" ")
    assert "\n#SBATCH --exclusive\n" in script
    assert "\n#SBATCH --nodes=1\n" in script
    assert "tee -a" not in script


def test_partition_for_nodes_prefers_batch_above_interactive_cap():
    slurm = SlurmRunnerConfig(
        account="acct",
        partition_interactive="interactive",
        partition_batch="batch",
        interactive_max_nodes=2,
    )
    assert slurm.partition_for_nodes(2) == "interactive"
    assert slurm.partition_for_nodes(3) == "batch"


def test_slurm_submit_retries_transient_controller_timeout(tmp_path: Path, monkeypatch):
    calls: list[list[str]] = []
    submit_count = 0

    def _fake_run(argv):
        nonlocal submit_count
        calls.append(list(argv))

        class _Result:
            returncode = 0
            stdout = ""
            stderr = ""

        result = _Result()
        if argv[0] == "sbatch":
            submit_count += 1
            if submit_count == 1:
                result.returncode = 1
                result.stderr = (
                    "sbatch: error: Batch job submission failed: "
                    "Socket timed out on send/recv operation"
                )
            else:
                result.stdout = "12345\n"
        return result

    monkeypatch.setattr("puzzletron_orchestrator.executors.slurm._run_command", _fake_run)
    monkeypatch.setattr("puzzletron_orchestrator.executors.slurm.time.sleep", lambda _: None)
    monkeypatch.setenv("USER", "tester")
    runner = RunnerEnvironment(
        kind="slurm",
        contract=ExecutionContract(repository=str(tmp_path), venv=str(tmp_path / ".venv")),
        slurm=SlurmRunnerConfig(account="acct", partition="interactive"),
    )
    attempt = AttemptSpec(
        attempt_id="abcdef12-3456",
        work_id="width_importance:0",
        stage_id="width_importance",
        command=CommandSpec(argv=("python", "-c", "pass")),
        allocation_gpus=8,
        metadata={"gpus_per_node": 8},
    )

    handle = SlurmExecutor(runner, scripts_dir=tmp_path / "sbatch").submit(attempt)

    assert handle.handle_id == "slurm-12345"
    assert submit_count == 2
    assert any(
        call
        == [
            "squeue",
            "-h",
            "--user",
            "tester",
            "--name",
            "pt-width_importance-abcdef12",
            "-o",
            "%A",
        ]
        for call in calls
    )


def test_slurm_submit_recovers_job_after_ambiguous_timeout(tmp_path: Path, monkeypatch):
    submit_count = 0

    def _fake_run(argv):
        nonlocal submit_count

        class _Result:
            returncode = 0
            stdout = ""
            stderr = ""

        result = _Result()
        if argv[0] == "sbatch":
            submit_count += 1
            result.returncode = 1
            result.stderr = "Socket timed out on send/recv operation"
        elif argv[0] == "squeue":
            result.stdout = "54321\n"
        return result

    monkeypatch.setattr("puzzletron_orchestrator.executors.slurm._run_command", _fake_run)
    monkeypatch.setattr("puzzletron_orchestrator.executors.slurm.time.sleep", lambda _: None)
    runner = RunnerEnvironment(
        kind="slurm",
        contract=ExecutionContract(repository=str(tmp_path), venv=str(tmp_path / ".venv")),
        slurm=SlurmRunnerConfig(account="acct", partition="interactive"),
    )
    attempt = AttemptSpec(
        attempt_id="abcdef12-3456",
        work_id="width_importance:0",
        stage_id="width_importance",
        command=CommandSpec(argv=("python", "-c", "pass")),
    )

    handle = SlurmExecutor(runner, scripts_dir=tmp_path / "sbatch").submit(attempt)

    assert handle.handle_id == "slurm-54321"
    assert submit_count == 1


def test_depth_pool_uses_one_four_node_gang_allocation(tmp_path: Path):
    runner = RunnerEnvironment(
        kind="slurm",
        contract=ExecutionContract(repository=str(tmp_path), venv=str(tmp_path / ".venv")),
        slurm=SlurmRunnerConfig(account="acct", partition_batch="batch"),
    )
    node = StagePlanNode(
        stage_id="depth_importance",
        strategy=ExecutionStrategy.PERSISTENT_POOL,
        instances=4,
        failure_policy=FailurePolicy.STRICT,
        mesh={"tp": 1, "cp": 1, "pp": 2, "ep": 4, "dp_shard": 4, "dp_replicate": 1},
        gpus_per_instance=8,
        gpus_per_node=8,
        nodes=4,
        total_gpus=32,
        exclusive=True,
        parents=("tokenize_data",),
        distributed=True,
        partition="batch",
    )
    plan = CampaignPlan(
        experiment_config_path=str(tmp_path / "experiment.yaml"),
        puzzle_dir=tmp_path / "run",
        experiment_config={"depth_importance": {"output_dir": str(tmp_path / "depth")}},
        runner=runner,
        execution_defaults={"gpus_per_node": 8},
        stages=(node,),
        contract_hash="contract",
    )
    adapter = adapter_for_stage(node)
    work_plan = adapter.plan(plan, node)

    assert [item.work_id for item in work_plan.items] == ["depth_importance:gang"]
    attempt = adapter.command(
        plan=plan,
        node=node,
        item=work_plan.items[0],
        attempt_id="a1",
        runner=runner,
    )
    assert attempt.allocation_nodes == 4
    assert attempt.allocation_gpus == 32
    assert attempt.metadata["kill_on_bad_exit"] is True
    assert attempt.command.argv[-1].endswith("run_depth_pool.sh")

    script = render_sbatch_script(
        attempt=attempt,
        runner=runner,
        partition="batch",
        account="acct",
        time_limit="4:00:00",
        qos=None,
        job_name="pt-depth",
    )
    assert "#SBATCH --nodes=4" in script
    assert "#SBATCH --ntasks=4" in script
    assert "#SBATCH --gpus-per-node=8" in script
    assert "--kill-on-bad-exit=1" in script
    assert "run_depth_pool.sh" in script


def test_depth_pool_packs_four_two_gpu_workers_per_node(tmp_path: Path):
    runner = RunnerEnvironment(
        kind="slurm",
        contract=ExecutionContract(repository=str(tmp_path), venv=str(tmp_path / ".venv")),
        slurm=SlurmRunnerConfig(account="acct", partition_batch="batch"),
    )
    node = StagePlanNode(
        stage_id="depth_importance",
        strategy=ExecutionStrategy.PERSISTENT_POOL,
        instances=8,
        failure_policy=FailurePolicy.STRICT,
        mesh={"tp": 1, "cp": 1, "pp": 1, "ep": 2, "dp_shard": 2, "dp_replicate": 1},
        gpus_per_instance=2,
        gpus_per_node=8,
        nodes=2,
        total_gpus=16,
        exclusive=True,
        parents=("tokenize_data",),
        distributed=True,
        partition="batch",
    )
    plan = CampaignPlan(
        experiment_config_path=str(tmp_path / "experiment.yaml"),
        puzzle_dir=tmp_path / "run",
        experiment_config={"depth_importance": {"output_dir": str(tmp_path / "depth")}},
        runner=runner,
        execution_defaults={"gpus_per_node": 8},
        stages=(node,),
        contract_hash="contract",
    )
    adapter = adapter_for_stage(node)
    work_plan = adapter.plan(plan, node)

    assert [item.work_id for item in work_plan.items] == ["depth_importance:gang"]
    attempt = adapter.command(
        plan=plan,
        node=node,
        item=work_plan.items[0],
        attempt_id="a1",
        runner=runner,
    )
    assert attempt.allocation_nodes == 2
    assert attempt.allocation_gpus == 16
    assert attempt.task_topology.task_count == 8
    assert attempt.task_topology.gpus_per_task == 2

    script = render_sbatch_script(
        attempt=attempt,
        runner=runner,
        partition="batch",
        account="acct",
        time_limit="4:00:00",
        qos=None,
        job_name="pt-depth",
    )
    assert "#SBATCH --nodes=2" in script
    assert "#SBATCH --ntasks=8" in script
    assert "#SBATCH --ntasks-per-node=4" in script
    assert "--gpus-per-task=2" in script


def test_post_mip_workers_share_one_packed_allocation(tmp_path: Path):
    runner = RunnerEnvironment(
        kind="slurm",
        contract=ExecutionContract(repository=str(tmp_path), venv=str(tmp_path / ".venv")),
        slurm=SlurmRunnerConfig(account="acct", partition_batch="batch"),
    )
    node = StagePlanNode(
        stage_id="post.profile.online_eval",
        strategy=ExecutionStrategy.SHARDED,
        instances=8,
        failure_policy=FailurePolicy.STRICT,
        mesh={"tp": 1, "cp": 1, "pp": 1, "ep": 2, "dp_shard": 2, "dp_replicate": 1},
        gpus_per_instance=2,
        gpus_per_node=8,
        nodes=2,
        total_gpus=16,
        exclusive=True,
        parents=("mip",),
        distributed=True,
        partition="batch",
    )
    plan = CampaignPlan(
        experiment_config_path=str(tmp_path / "experiment.yaml"),
        puzzle_dir=tmp_path / "run",
        experiment_config={
            "post_mip": {"flows": {"profile": {"nodes": {"online_eval": {"type": "evaluation"}}}}}
        },
        runner=runner,
        execution_defaults={"gpus_per_node": 8},
        stages=(node,),
        contract_hash="contract",
    )
    adapter = adapter_for_stage(node)
    work_plan = adapter.plan(plan, node)

    assert [item.work_id for item in work_plan.items] == ["post.profile.online_eval:gang"]
    attempt = adapter.command(
        plan=plan,
        node=node,
        item=work_plan.items[0],
        attempt_id="a1",
        runner=runner,
    )
    assert attempt.allocation_nodes == 2
    assert attempt.allocation_gpus == 16
    assert attempt.task_topology.task_count == 8
    assert attempt.task_topology.gpus_per_task == 2
    assert attempt.task_topology.tasks_per_group == 1
    assert attempt.task_topology.launcher.value == "torchrun"
    assert "--shard-index" not in attempt.command.argv
    assert attempt.command.argv[-2:] == ("--shard-count", "8")


def test_replacement_pool_uses_one_four_node_gang_allocation(tmp_path: Path):
    runner = RunnerEnvironment(
        kind="slurm",
        contract=ExecutionContract(repository=str(tmp_path), venv=str(tmp_path / ".venv")),
        slurm=SlurmRunnerConfig(account="acct", partition_batch="batch"),
    )
    node = StagePlanNode(
        stage_id="replacement_scoring",
        strategy=ExecutionStrategy.PERSISTENT_POOL,
        instances=4,
        failure_policy=FailurePolicy.STRICT,
        mesh={"tp": 1, "cp": 1, "pp": 2, "ep": 4, "dp_shard": 4, "dp_replicate": 1},
        gpus_per_instance=8,
        gpus_per_node=8,
        nodes=4,
        total_gpus=32,
        exclusive=True,
        parents=("build_library",),
        distributed=True,
        partition="batch",
    )
    plan = CampaignPlan(
        experiment_config_path=str(tmp_path / "experiment.yaml"),
        puzzle_dir=tmp_path / "run",
        experiment_config={"replacement_scoring": {}},
        runner=runner,
        execution_defaults={"gpus_per_node": 8},
        stages=(node,),
        contract_hash="contract",
    )
    adapter = adapter_for_stage(node)
    work_plan = adapter.plan(plan, node)

    assert [item.work_id for item in work_plan.items] == ["replacement_scoring:gang"]
    attempt = adapter.command(
        plan=plan,
        node=node,
        item=work_plan.items[0],
        attempt_id="a1",
        runner=runner,
    )
    assert attempt.allocation_nodes == 4
    assert attempt.allocation_gpus == 32
    assert attempt.metadata["kill_on_bad_exit"] is True
    assert attempt.metadata["partition"] == "batch"
    assert attempt.command.argv[-1].endswith("run_replacement_pool.sh")


def test_replacement_pool_splits_workers_across_embedding_widths(tmp_path: Path):
    runner = RunnerEnvironment(
        kind="slurm",
        contract=ExecutionContract(repository=str(tmp_path), venv=str(tmp_path / ".venv")),
        slurm=SlurmRunnerConfig(account="acct", partition_batch="batch"),
    )
    node = StagePlanNode(
        stage_id="replacement_scoring",
        strategy=ExecutionStrategy.PERSISTENT_POOL,
        instances=8,
        failure_policy=FailurePolicy.STRICT,
        mesh={"tp": 1, "cp": 1, "pp": 2, "ep": 1, "dp_shard": 2, "dp_replicate": 1},
        gpus_per_instance=4,
        gpus_per_node=8,
        nodes=4,
        total_gpus=32,
        exclusive=True,
        parents=("build_library",),
        distributed=True,
        partition="batch",
    )
    plan = CampaignPlan(
        experiment_config_path=str(tmp_path / "experiment.yaml"),
        puzzle_dir=tmp_path / "run",
        experiment_config={
            "embedding_pruning": {"enabled": True, "widths": [2048, 1792]},
            "replacement_scoring": {"granularity": "subblock"},
        },
        runner=runner,
        execution_defaults={"gpus_per_node": 8},
        stages=(node,),
        contract_hash="contract",
    )
    adapter = adapter_for_stage(node)
    work_plan = adapter.plan(plan, node)

    assert [item.work_id for item in work_plan.items] == [
        "replacement_scoring:width-2048",
        "replacement_scoring:width-1792",
    ]
    assert [item.metadata["worker_count"] for item in work_plan.items] == [4, 4]

    attempts = [
        adapter.command(
            plan=plan,
            node=node,
            item=item,
            attempt_id=f"a{index}",
            runner=runner,
        )
        for index, item in enumerate(work_plan.items)
    ]
    assert [attempt.allocation_nodes for attempt in attempts] == [2, 2]
    assert [attempt.allocation_gpus for attempt in attempts] == [16, 16]
    assert [attempt.task_topology.task_count for attempt in attempts] == [4, 4]
    assert [attempt.task_topology.gpus_per_task for attempt in attempts] == [4, 4]
    assert [attempt.command.env["WORKER_COUNT"] for attempt in attempts] == ["4", "4"]
    assert [
        attempt.command.env["FINALIZE_EXPECTED_COMPLETIONS"] for attempt in attempts
    ] == ["2", "2"]
    assert [
        attempt.command.env["FINALIZE_COMPLETION_MARKER"] for attempt in attempts
    ] == ["width-2048", "width-1792"]
    assert (
        attempts[0].command.env["FINALIZE_COMPLETION_DIR"]
        == attempts[1].command.env["FINALIZE_COMPLETION_DIR"]
    )
    assert attempts[0].command.env["PUZZLE_DIR"].endswith(
        "scenarios/width-2048/depth-00"
    )
    assert attempts[1].command.env["PUZZLE_DIR"].endswith(
        "scenarios/width-1792/depth-00"
    )


def test_stage_partition_override_forces_batch(tmp_path: Path):
    experiment = tmp_path / "experiment.yaml"
    experiment.write_text(
        yaml.safe_dump(
            {
                "experiment": {"dir": str(tmp_path / "run")},
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
                    "slurm": {
                        "account": "test",
                        "partition_interactive": "interactive",
                        "partition_batch": "batch",
                        "interactive_max_nodes": 2,
                    },
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
                        "vllm_stats": {
                            "strategy": "sharded",
                            "instances": 4,
                            "partition": "batch",
                        },
                    },
                }
            }
        )
    )
    plan = compile_campaign_plan(
        experiment_config_path=experiment,
        runner=load_runner_config(runner),
        execution=load_execution_config(execution),
    )
    node = next(item for item in plan.stages if item.stage_id == "vllm_stats")
    assert node.partition == "batch"
    assert plan.runner.slurm.partition_for_nodes(1) == "interactive"
    adapter = adapter_for_stage(node)
    attempt = adapter.command(
        plan=plan,
        node=node,
        item=adapter.plan(plan, node).items[0],
        attempt_id="a1",
        runner=plan.runner,
    )
    assert attempt.metadata["partition"] == "batch"
