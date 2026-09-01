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

"""Tests for orchestration executors."""

import subprocess
import time
from pathlib import Path

import pytest
import yaml

import puzzletron_orchestrator.adapters.sharded as sharded_module
from modelopt.torch.puzzletron.distributed_eval.config import load_runtime_config
from puzzletron_orchestrator.adapters.registry import adapter_for_stage
from puzzletron_orchestrator.compiler import (
    compile_campaign_plan,
    load_execution_config,
    load_runner_config,
)
from puzzletron_orchestrator.executors.baremetal import BareMetalSSHExecutor
from puzzletron_orchestrator.executors.local import LocalExecutor
from puzzletron_orchestrator.executors.slurm import SlurmExecutor, render_sbatch_script
from puzzletron_orchestrator.schema import (
    AttemptSpec,
    BareMetalHost,
    BareMetalRunnerConfig,
    CampaignPlan,
    CommandSpec,
    ExecutionContract,
    ExecutionStrategy,
    FailurePolicy,
    JobHandle,
    JobState,
    JobStatus,
    RunnerEnvironment,
    SlurmRunnerConfig,
    StagePlanNode,
    TaskTopology,
    WorkItem,
    WorkPlan,
)

# Local and bare-metal execution


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
    deadline = time.monotonic() + 5
    while (status := executor.poll([handle])[0]).state is JobState.RUNNING:
        assert time.monotonic() < deadline
        time.sleep(0.01)
    assert status.state is JobState.COMPLETED
    assert log_path.read_text().splitlines()[-1] == "ok"


def test_baremetal_preflight_checks_repository_and_venv_on_every_host(
    tmp_path: Path, monkeypatch
) -> None:
    calls = []

    def _fake_run(argv):
        calls.append(list(argv))
        return subprocess.CompletedProcess(argv, 0, stdout="worker\nGPU 0\n", stderr="")

    monkeypatch.setattr("puzzletron_orchestrator.executors.baremetal._run_command", _fake_run)
    executor = BareMetalSSHExecutor(_baremetal_runner(), state_dir=tmp_path)

    executor.preflight()

    assert calls == [
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            hostname,
            "hostname && test -d /shared/modelopt && cd /shared/modelopt && "
            "test -f /shared/puzzletron-venv/bin/activate && nvidia-smi -L",
        ]
        for hostname in ("node-a", "node-b")
    ]


def test_baremetal_submit_uses_one_execution_contract_on_every_host(
    tmp_path: Path, monkeypatch
) -> None:
    calls = []

    def _fake_run(argv):
        calls.append(list(argv))
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    monkeypatch.setattr("puzzletron_orchestrator.executors.baremetal._run_command", _fake_run)
    executor = BareMetalSSHExecutor(_baremetal_runner(), state_dir=tmp_path)
    attempt = AttemptSpec(
        attempt_id="a1",
        work_id="vllm_stats:gang",
        stage_id="vllm_stats",
        command=CommandSpec(argv=("python", "-c", "print('ok')")),
        allocation_nodes=2,
        allocation_gpus=2,
        metadata={"gpus_per_node": 1},
        task_topology=TaskTopology(task_count=2, gpus_per_task=1),
    )

    handle = executor.submit(attempt)

    assert [call[3] for call in calls] == ["node-a", "node-b"]
    for call in calls:
        remote_command = call[-1]
        assert "cd /shared/modelopt; source /shared/puzzletron-venv/bin/activate;" in remote_command
        assert "export PYTHONPATH=/shared/modelopt:${PYTHONPATH:-};" in remote_command
    assert [task["hostname"] for task in handle.metadata["tasks"]] == ["node-a", "node-b"]


# Slurm script rendering and AIPerf aggregation


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
            partition=["gpu-a", "gpu-b"],
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
        task_topology=TaskTopology(task_count=2, gpus_per_task=8),
    )
    script = render_sbatch_script(
        attempt=attempt,
        runner=runner,
        partition=runner.slurm.partition,
        account="acct",
        time_limit="4:00:00",
        qos=None,
        job_name="pt-vllm",
    )
    assert "#SBATCH --gpus-per-node=8" in script
    assert "#SBATCH --nodes=2" in script
    assert "#SBATCH --partition=gpu-a,gpu-b" in script
    assert "#SBATCH --no-requeue" in script
    assert "source /site/setup-envs.sh" in script
    assert script.startswith("#!/bin/bash\n")
    srun = next(line for line in script.splitlines() if line.startswith("srun "))
    assert "--account=acct" in srun
    assert "--job-name=pt-vllm" in srun


def test_render_sbatch_script_omits_gpu_requests_for_cpu_stage():
    runner = RunnerEnvironment(
        kind="slurm",
        contract=ExecutionContract(repository="/repo", venv="/repo/.venv"),
        slurm=SlurmRunnerConfig(account="acct", partition="cpu"),
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


@pytest.mark.parametrize("configured_log_dir", [False, True], ids=("fallback", "configured"))
def test_work_adapters_use_effective_campaign_log_dir(tmp_path: Path, configured_log_dir: bool):
    """Every adapter must use the plan's effective shared log directory."""

    custom_log_dir = tmp_path / "shared-logs" if configured_log_dir else None
    runner = RunnerEnvironment(
        kind="slurm",
        contract=ExecutionContract(repository=str(tmp_path), venv=str(tmp_path / ".venv")),
        slurm=SlurmRunnerConfig(
            account="acct",
            partition="batch",
            log_dir=str(custom_log_dir) if custom_log_dir else None,
        ),
    )

    def stage(stage_id: str, strategy: ExecutionStrategy) -> StagePlanNode:
        return StagePlanNode(
            stage_id=stage_id,
            strategy=strategy,
            instances=1,
            failure_policy=FailurePolicy.STRICT,
            mesh={},
            gpus_per_instance=1,
            gpus_per_node=8,
            nodes=1,
            total_gpus=1,
            exclusive=False,
            parents=(),
            distributed=False,
            partition="batch",
        )

    nodes = (
        stage("convert", ExecutionStrategy.SINGLE),
        stage("depth_importance", ExecutionStrategy.PERSISTENT_POOL),
        stage("vllm_stats", ExecutionStrategy.SHARDED),
        stage("post.profile.online_eval", ExecutionStrategy.SHARDED),
    )
    plan = CampaignPlan(
        experiment_config_path=str(tmp_path / "experiment.yaml"),
        puzzle_dir=tmp_path / "run",
        experiment_config={
            "puzzle_dir": str(tmp_path / "run"),
            "depth_importance": {"output_dir": str(tmp_path / "depth")},
            "post_mip": {"flows": {"profile": {"nodes": {"online_eval": {"type": "evaluation"}}}}},
        },
        runner=runner,
        execution_defaults={"gpus_per_node": 8},
        stages=nodes,
        contract_hash="contract",
    )
    items = (
        WorkItem("convert:0", "convert", 0, 1, 1),
        WorkItem(
            "depth_importance:gang",
            "depth_importance",
            0,
            1,
            1,
            metadata={"role": "gang", "worker_count": 1},
        ),
        WorkItem("vllm_stats:0", "vllm_stats", 0, 1, 1),
        WorkItem(
            "post.profile.online_eval:0",
            "post.profile.online_eval",
            0,
            1,
            1,
            metadata={"logical_shard_count": 1},
        ),
    )
    expected_log_dir = custom_log_dir or plan.puzzle_dir / "logs"

    for node, item in zip(nodes, items, strict=True):
        attempt = adapter_for_stage(node).command(
            plan=plan,
            node=node,
            item=item,
            attempt_id="a1",
            runner=runner,
        )
        assert attempt.command.log_path is not None
        assert Path(attempt.command.log_path).parent == expected_log_dir


def test_campaign_plan_anchors_relative_log_dir_to_puzzle_dir(tmp_path: Path) -> None:
    runner = RunnerEnvironment(
        kind="slurm",
        contract=ExecutionContract(repository=str(tmp_path), venv=str(tmp_path / ".venv")),
        slurm=SlurmRunnerConfig(account="acct", log_dir="shared-logs"),
    )
    plan = CampaignPlan(
        experiment_config_path=str(tmp_path / "experiment.yaml"),
        puzzle_dir=tmp_path / "run",
        experiment_config={},
        runner=runner,
        execution_defaults={},
        stages=(),
        contract_hash="contract",
    )

    assert plan.log_dir == tmp_path / "run" / "shared-logs"


@pytest.mark.parametrize(
    ("experiment_config", "trust_remote_code", "online_tokenizer"),
    [
        ({}, False, False),
        ({"model": {"trust_remote_code": True}}, True, False),
        (
            {
                "model": {"trust_remote_code": True},
                "aiperf": {
                    "trust_remote_code": False,
                    "allow_aiperf_v011_online_tokenizer_resolution": True,
                },
            },
            False,
            True,
        ),
    ],
)
def test_sharded_aiperf_command_applies_explicit_security_policy(
    tmp_path, experiment_config, trust_remote_code, online_tokenizer
):
    argv = _aiperf_attempt(tmp_path, experiment_config).command.argv

    assert ("--trust-remote-code" in argv) is trust_remote_code
    assert ("--allow-aiperf-v011-online-tokenizer-resolution" in argv) is online_tokenizer


@pytest.mark.parametrize(
    "experiment_config",
    [
        {"model": {"trust_remote_code": "true"}},
        {"aiperf": {"trust_remote_code": 1}},
        {"aiperf": {"allow_aiperf_v011_online_tokenizer_resolution": "false"}},
    ],
)
def test_sharded_aiperf_command_rejects_non_boolean_security_policy(tmp_path, experiment_config):
    with pytest.raises(ValueError, match="must be a boolean"):
        _aiperf_attempt(tmp_path, experiment_config)


@pytest.mark.parametrize(
    ("slurm_kwargs", "node_partition", "expected_partition", "configured_log_dir"),
    [
        ({"partition": "runner-a"}, "reserved-a,reserved-b", "reserved-a,reserved-b", False),
        ({"partition": "runner-a"}, "reserved-a,reserved-b", "reserved-a,reserved-b", True),
        (
            {
                "partition_interactive": "interactive",
                "partition_batch": "batch",
                "partition_cpu": "cpu",
            },
            None,
            "cpu",
            False,
        ),
        (
            {
                "partition_interactive": "interactive",
                "partition_batch": "batch",
                "partition_cpu": "cpu",
            },
            None,
            "cpu",
            True,
        ),
    ],
    ids=(
        "stage-override-fallback-log",
        "stage-override-configured-log",
        "legacy-cpu-fallback-log",
        "legacy-cpu-configured-log",
    ),
)
def test_vllm_aggregation_uses_slurm_execution_contract(
    tmp_path: Path,
    monkeypatch,
    slurm_kwargs,
    node_partition,
    expected_partition,
    configured_log_dir,
):
    """Controller-side merges must run in the same container/venv as workers."""

    slurm_kwargs = dict(slurm_kwargs)
    if configured_log_dir:
        slurm_kwargs["log_dir"] = str(tmp_path / "shared-logs")
    runner = RunnerEnvironment(
        kind="slurm",
        contract=ExecutionContract(
            repository=str(tmp_path),
            venv=".venv-worker",
            container="/images/pytorch.sqsh",
        ),
        slurm=SlurmRunnerConfig(account="acct", **slurm_kwargs),
    )
    node = StagePlanNode(
        stage_id="vllm_stats",
        strategy=ExecutionStrategy.SHARDED,
        instances=1,
        failure_policy=FailurePolicy.STRICT,
        mesh={},
        gpus_per_instance=1,
        gpus_per_node=8,
        nodes=1,
        total_gpus=1,
        exclusive=False,
        parents=("convert",),
        distributed=False,
        partition=node_partition,
    )
    plan = CampaignPlan(
        experiment_config_path=str(tmp_path / "experiment.yaml"),
        puzzle_dir=tmp_path / "run",
        experiment_config={"puzzle_dir": str(tmp_path / "run")},
        runner=runner,
        execution_defaults={"gpus_per_node": 8},
        stages=(node,),
        contract_hash="contract",
    )
    submitted = []

    class FakeSlurmExecutor:
        def __init__(self, configured_runner, *, scripts_dir=None):
            assert configured_runner is runner
            assert scripts_dir == plan.puzzle_dir / "orchestration" / "sbatch"

        def submit(self, attempt):
            submitted.append(attempt)
            return JobHandle(
                backend="slurm",
                handle_id="slurm-123",
                attempt_id=attempt.attempt_id,
                metadata={"job_id": "123", "log_paths": (attempt.command.log_path,)},
            )

        def poll(self, handles):
            return [
                JobStatus(
                    handle=handles[0],
                    state=JobState.COMPLETED,
                    exit_code=0,
                    log_paths=tuple(handles[0].metadata["log_paths"]),
                )
            ]

    monkeypatch.setattr(sharded_module, "SlurmExecutor", FakeSlurmExecutor, raising=False)

    class FailLocalExecutor:
        def __init__(self, *_args, **_kwargs):
            del _args, _kwargs
            raise AssertionError("aggregation escaped the Slurm execution contract")

    monkeypatch.setattr(sharded_module, "LocalExecutor", FailLocalExecutor)
    adapter = adapter_for_stage(node)
    result = adapter.aggregate(
        plan=plan,
        node=node,
        work_plan=WorkPlan(
            stage_id="vllm_stats",
            strategy=ExecutionStrategy.SHARDED,
            items=(),
            aggregate_required=True,
        ),
    )

    assert result is not None
    assert len(submitted) == 1
    attempt = submitted[0]
    assert attempt.allocation_gpus == 0
    assert attempt.metadata["gpus_per_node"] == 0
    assert attempt.metadata["partition"] == expected_partition
    expected_log_dir = tmp_path / ("shared-logs" if configured_log_dir else "run/logs")
    assert attempt.command.log_path is not None
    assert Path(attempt.command.log_path).parent == expected_log_dir
    assert attempt.command.argv[-1] == "--merge"
    assert result.summary["merge_handle"] == "slurm-123"


def test_aiperf_aggregation_uses_reviewed_local_executor(tmp_path: Path, monkeypatch):
    runner = RunnerEnvironment(
        kind="local",
        contract=ExecutionContract(repository=str(tmp_path), venv=str(tmp_path / ".venv")),
        slurm=SlurmRunnerConfig(account="test", log_dir="shared-logs"),
    )
    node = StagePlanNode(
        stage_id="aiperf",
        strategy=ExecutionStrategy.SHARDED,
        instances=1,
        failure_policy=FailurePolicy.STRICT,
        mesh={},
        gpus_per_instance=8,
        gpus_per_node=8,
        nodes=1,
        total_gpus=8,
        exclusive=False,
        parents=("mip",),
        distributed=False,
    )
    plan = CampaignPlan(
        experiment_config_path=str(tmp_path / "experiment.yaml"),
        puzzle_dir=tmp_path / "run",
        experiment_config={"aiperf": {"profile_id": "runtime-075"}},
        runner=runner,
        execution_defaults={"gpus_per_node": 8},
        stages=(node,),
        contract_hash="contract",
    )
    submitted = []

    class FakeLocalExecutor:
        def __init__(self, configured_runner):
            assert configured_runner is runner

        def submit(self, attempt):
            submitted.append(attempt)
            return JobHandle(
                backend="local",
                handle_id="local-1",
                attempt_id=attempt.attempt_id,
                metadata={"log_paths": (attempt.command.log_path,)},
            )

        def poll(self, handles):
            return [
                JobStatus(
                    handle=handles[0],
                    state=JobState.COMPLETED,
                    exit_code=0,
                    log_paths=tuple(handles[0].metadata["log_paths"]),
                )
            ]

    monkeypatch.setattr(sharded_module, "LocalExecutor", FakeLocalExecutor)

    result = adapter_for_stage(node).aggregate(
        plan=plan,
        node=node,
        work_plan=WorkPlan(
            stage_id="aiperf",
            strategy=ExecutionStrategy.SHARDED,
            items=(),
            aggregate_required=True,
        ),
    )

    assert result is not None
    assert len(submitted) == 1
    attempt = submitted[0]
    assert attempt.allocation_gpus == 0
    assert attempt.command.cwd == str(tmp_path)
    assert attempt.command.log_path is not None
    assert Path(attempt.command.log_path).parent == tmp_path / "run" / "shared-logs"
    assert attempt.command.argv[-1] == "--merge"


# Slurm submission and recovery


def test_render_sbatch_script_never_requests_exclusive():
    """Exclusive attempt metadata must not request exclusive Slurm nodes."""

    runner = RunnerEnvironment(
        kind="slurm",
        contract=ExecutionContract(
            repository="/repo",
            venv="/repo/.venv",
        ),
        slurm=SlurmRunnerConfig(
            account="acct",
            partition="gpu",
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
    assert "#SBATCH --exclusive" not in script
    assert "#SBATCH --qos=normal" in script
    assert "#SBATCH --gpus-per-node=8" in script
    assert not script.startswith(" ")
    assert "\n#SBATCH --nodes=1\n" in script
    assert "tee -a" not in script


@pytest.mark.parametrize(
    ("runner_partition", "metadata", "expected_directive"),
    [
        (["runner-a", "runner-b"], {}, "#SBATCH --partition=runner-a,runner-b"),
        (["runner-a"], {"partition": "stage-a,stage-b"}, "#SBATCH --partition=stage-a,stage-b"),
        (None, {}, None),
    ],
)
def test_slurm_submit_resolves_partition_precedence(
    tmp_path: Path,
    monkeypatch,
    runner_partition,
    metadata,
    expected_directive,
):
    monkeypatch.setattr(
        "puzzletron_orchestrator.executors.slurm._run_command",
        lambda argv: subprocess.CompletedProcess(argv, 0, stdout="12345\n", stderr=""),
    )
    runner = RunnerEnvironment(
        kind="slurm",
        contract=ExecutionContract(repository=str(tmp_path), venv=str(tmp_path / ".venv")),
        slurm=SlurmRunnerConfig(account="acct", partition=runner_partition),
    )
    attempt = AttemptSpec(
        attempt_id="a1",
        work_id="mip:0",
        stage_id="mip",
        command=CommandSpec(argv=("python", "worker.py")),
        metadata=metadata,
    )

    SlurmExecutor(runner, scripts_dir=tmp_path / "sbatch").submit(attempt)

    script = (tmp_path / "sbatch" / "mip_a1.sh").read_text()
    if expected_directive is None:
        assert "#SBATCH --partition=" not in script
    else:
        assert expected_directive in script


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
        slurm=SlurmRunnerConfig(account="acct", partition="gpu", job_name_prefix="acct-puzzletron"),
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
            "acct-puzzletron-width_importance-abcdef12",
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
        slurm=SlurmRunnerConfig(account="acct", partition="gpu"),
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


def test_slurm_recover_keeps_running_when_squeue_misses_and_sacct_says_running(
    tmp_path: Path, monkeypatch
):
    """Regression: empty squeue + sacct RUNNING must not become FAILED."""

    def _fake_run(argv):
        class _Result:
            returncode = 0
            stdout = ""
            stderr = ""

        result = _Result()
        if argv[0] == "squeue":
            result.stdout = ""
        elif argv[0] == "sacct":
            result.stdout = "RUNNING|0:0\n"
        return result

    monkeypatch.setattr("puzzletron_orchestrator.executors.slurm._run_command", _fake_run)
    handle = JobHandle(
        backend="slurm",
        handle_id="slurm-15214984",
        attempt_id="a1",
        metadata={"job_id": "15214984"},
    )

    status = _slurm_executor(tmp_path).recover(handle)

    assert status.state is JobState.RUNNING
    assert status.exit_code is None


def test_slurm_recover_maps_sacct_pending_and_failed_states(tmp_path: Path, monkeypatch):
    responses = {
        "PENDING|0:0\n": JobState.PENDING,
        "REQUEUED|0:0\n": JobState.PENDING,
        "FAILED|1:0\n": JobState.FAILED,
        "TIMEOUT|1:0\n": JobState.FAILED,
        "CANCELLED by 1234|0:0\n": JobState.CANCELLED,
        "COMPLETED|0:0\n": JobState.COMPLETED,
    }

    for sacct_stdout, expected in responses.items():

        def _fake_run(argv, *, _stdout=sacct_stdout):
            class _Result:
                returncode = 0
                stdout = ""
                stderr = ""

            result = _Result()
            if argv[0] == "squeue":
                result.stdout = ""
            elif argv[0] == "sacct":
                result.stdout = _stdout
            return result

        monkeypatch.setattr("puzzletron_orchestrator.executors.slurm._run_command", _fake_run)
        handle = JobHandle(
            backend="slurm",
            handle_id="slurm-1",
            attempt_id="a1",
            metadata={"job_id": "1"},
        )
        status = _slurm_executor(tmp_path).recover(handle)
        assert status.state is expected, sacct_stdout


# Persistent-pool allocation and packing


def test_depth_pool_uses_one_four_node_gang_allocation(tmp_path: Path):
    runner = RunnerEnvironment(
        kind="slurm",
        contract=ExecutionContract(repository=str(tmp_path), venv=str(tmp_path / ".venv")),
        slurm=SlurmRunnerConfig(account="acct", partition="batch"),
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
        slurm=SlurmRunnerConfig(account="acct", partition="batch"),
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
        slurm=SlurmRunnerConfig(account="acct", partition="batch"),
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
        slurm=SlurmRunnerConfig(account="acct", partition="batch"),
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
    plan, work_plan, attempts = _replacement_width_attempts(tmp_path, [2048, 1792])
    runner = plan.runner
    node = plan.stages[0]
    adapter = adapter_for_stage(node)

    assert [item.work_id for item in work_plan.items] == [
        "replacement_scoring:width-2048",
        "replacement_scoring:width-1792",
    ]
    assert [item.metadata["worker_count"] for item in work_plan.items] == [4, 4]

    assert [attempt.allocation_nodes for attempt in attempts] == [2, 2]
    assert [attempt.allocation_gpus for attempt in attempts] == [16, 16]
    assert [attempt.task_topology.task_count for attempt in attempts] == [4, 4]
    assert [attempt.task_topology.gpus_per_task for attempt in attempts] == [4, 4]
    assert [attempt.command.env["WORKER_COUNT"] for attempt in attempts] == ["4", "4"]
    assert [attempt.command.env["FINALIZE_OVERRIDES"] for attempt in attempts] == [
        "++replacement_scoring.automodel.lm_head_backend=streaming",
        "++replacement_scoring.automodel.lm_head_backend=streaming",
    ]
    assert all(
        "puzzle_dir=" not in attempt.command.env["FINALIZE_OVERRIDES"] for attempt in attempts
    )
    assert all(
        "puzzle_dir=" in attempt.command.env["DISTRIBUTED_EVAL_OVERRIDES"] for attempt in attempts
    )
    assert [attempt.command.env["FINALIZE_EXPECTED_COMPLETIONS"] for attempt in attempts] == [
        "2",
        "2",
    ]
    assert [attempt.command.env["FINALIZE_COMPLETION_MARKER"] for attempt in attempts] == [
        "width-2048",
        "width-1792",
    ]
    assert (
        attempts[0].command.env["FINALIZE_COMPLETION_DIR"]
        == attempts[1].command.env["FINALIZE_COMPLETION_DIR"]
    )
    changed_plan = CampaignPlan(
        experiment_config_path=plan.experiment_config_path,
        puzzle_dir=plan.puzzle_dir,
        experiment_config={
            **plan.experiment_config,
            "replacement_scoring": {
                "granularity": "subblock",
                "default_metric": "mse_loss_hidden_states",
            },
        },
        runner=runner,
        execution_defaults=plan.execution_defaults,
        stages=(node,),
        contract_hash=plan.contract_hash,
    )
    changed_work_plan = adapter.plan(changed_plan, node)
    changed_attempt = adapter.command(
        plan=changed_plan,
        node=node,
        item=changed_work_plan.items[0],
        attempt_id="changed",
        runner=runner,
        overrides=["++replacement_scoring.automodel.lm_head_backend=streaming"],
    )
    assert (
        changed_attempt.command.env["FINALIZE_COMPLETION_DIR"]
        != attempts[0].command.env["FINALIZE_COMPLETION_DIR"]
    )
    assert attempts[0].command.env["PUZZLE_DIR"].endswith("scenarios/width-2048/depth-00")
    assert attempts[1].command.env["PUZZLE_DIR"].endswith("scenarios/width-1792/depth-00")


def test_replacement_width_overrides_reach_runtime_config(tmp_path: Path):
    _, _, attempts = _replacement_width_attempts(tmp_path, [2048])
    attempt = attempts[0]

    config = load_runtime_config(
        Path(__file__).parents[4] / "examples/puzzletron/configs/base.yaml",
        overrides=[
            f"++input_hf_model_path={tmp_path / 'model'}",
            *attempt.command.env["DISTRIBUTED_EVAL_OVERRIDES"].splitlines(),
        ],
    )

    teacher = str(Path(attempt.command.env["PUZZLE_DIR"]) / "ckpts" / "sorted_teacher")
    assert config.build_replacement_library.source_checkpoint_dir == teacher
    assert config.scoring.source_checkpoint_dir == teacher
    assert config.scoring.target_teacher_dir == teacher


def test_replacement_pool_completion_identity_changes_with_embedding_widths(tmp_path: Path):
    _, _, baseline_attempts = _replacement_width_attempts(tmp_path, [2048, 1792])
    _, _, changed_attempts = _replacement_width_attempts(tmp_path, [2048, 1792, 1536, 1280])

    baseline_completion_dirs = {
        attempt.command.env["FINALIZE_COMPLETION_DIR"] for attempt in baseline_attempts
    }
    changed_completion_dirs = {
        attempt.command.env["FINALIZE_COMPLETION_DIR"] for attempt in changed_attempts
    }
    assert len(baseline_completion_dirs) == 1
    assert len(changed_completion_dirs) == 1
    assert changed_completion_dirs != baseline_completion_dirs
    assert [attempt.command.env["FINALIZE_COMPLETION_MARKER"] for attempt in changed_attempts] == [
        "width-2048",
        "width-1792",
        "width-1536",
        "width-1280",
    ]
    assert [
        attempt.command.env["FINALIZE_EXPECTED_COMPLETIONS"] for attempt in changed_attempts
    ] == ["4", "4", "4", "4"]


def test_stage_partition_override_uses_eligible_partition_list(tmp_path: Path):
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
                        "partition": ["gpu-a", "gpu-b"],
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
                            "partition": ["reserved-a", "reserved-b"],
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
    assert node.partition == "reserved-a,reserved-b"
    assert plan.runner.slurm.partition == "gpu-a,gpu-b"
    adapter = adapter_for_stage(node)
    attempt = adapter.command(
        plan=plan,
        node=node,
        item=adapter.plan(plan, node).items[0],
        attempt_id="a1",
        runner=plan.runner,
    )
    assert attempt.metadata["partition"] == "reserved-a,reserved-b"


def _baremetal_runner() -> RunnerEnvironment:
    return RunnerEnvironment(
        kind="baremetal",
        contract=ExecutionContract(
            repository="/shared/modelopt",
            venv="/shared/puzzletron-venv",
        ),
        baremetal=BareMetalRunnerConfig(
            hosts=(
                BareMetalHost(hostname="node-a", gpus=1),
                BareMetalHost(hostname="node-b", gpus=1),
            ),
            rendezvous_host="node-a",
        ),
    )


def _aiperf_attempt(tmp_path: Path, experiment_config: dict) -> AttemptSpec:
    runner = RunnerEnvironment(
        kind="slurm",
        contract=ExecutionContract(repository=str(tmp_path), venv=str(tmp_path / ".venv")),
        slurm=SlurmRunnerConfig(account="acct", partition_batch="batch"),
    )
    node = StagePlanNode(
        stage_id="aiperf",
        strategy=ExecutionStrategy.SHARDED,
        instances=2,
        failure_policy=FailurePolicy.STRICT,
        mesh={},
        gpus_per_instance=8,
        gpus_per_node=8,
        nodes=2,
        total_gpus=16,
        exclusive=False,
        parents=("mip",),
        distributed=False,
    )
    plan = CampaignPlan(
        experiment_config_path=str(tmp_path / "experiment.yaml"),
        puzzle_dir=tmp_path / "run",
        experiment_config=experiment_config,
        runner=runner,
        execution_defaults={"gpus_per_node": 8},
        stages=(node,),
        contract_hash="contract",
    )
    adapter = adapter_for_stage(node)
    work_plan = adapter.plan(plan, node)
    return adapter.command(
        plan=plan,
        node=node,
        item=work_plan.items[0],
        attempt_id="a1",
        runner=runner,
    )


def _slurm_executor(tmp_path: Path) -> SlurmExecutor:
    return SlurmExecutor(
        RunnerEnvironment(
            kind="slurm",
            contract=ExecutionContract(repository=str(tmp_path), venv=str(tmp_path / ".venv")),
            slurm=SlurmRunnerConfig(account="acct", partition="interactive"),
        ),
        scripts_dir=tmp_path / "sbatch",
    )


def _replacement_width_attempts(
    tmp_path: Path,
    widths: list[int],
) -> tuple[CampaignPlan, WorkPlan, list[AttemptSpec]]:
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
            "embedding_pruning": {"enabled": True, "widths": widths},
            "replacement_scoring": {"granularity": "subblock"},
        },
        runner=runner,
        execution_defaults={"gpus_per_node": 8},
        stages=(node,),
        contract_hash="contract",
    )
    adapter = adapter_for_stage(node)
    work_plan = adapter.plan(plan, node)
    attempts = [
        adapter.command(
            plan=plan,
            node=node,
            item=item,
            attempt_id=f"a{index}",
            runner=runner,
            overrides=["++replacement_scoring.automodel.lm_head_backend=streaming"],
        )
        for index, item in enumerate(work_plan.items)
    ]
    return plan, work_plan, attempts
