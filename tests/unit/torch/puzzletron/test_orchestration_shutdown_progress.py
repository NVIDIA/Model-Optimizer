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

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for orchestrator shutdown and progress reporting."""

from __future__ import annotations

import json
import signal
from collections.abc import Sequence
from pathlib import Path

import yaml

from puzzletron_orchestrator.adapters.registry import adapter_for_stage
from puzzletron_orchestrator.compiler import (
    compile_campaign_plan,
    load_execution_config,
    load_runner_config,
)
from puzzletron_orchestrator.controller import CampaignController
from puzzletron_orchestrator.executors.base import Executor
from puzzletron_orchestrator.progress import summarize_active_progress, summarize_stage_artifacts
from puzzletron_orchestrator.schema import (
    AttemptSpec,
    CommandSpec,
    JobHandle,
    JobState,
    JobStatus,
)
from puzzletron_orchestrator.terminal import ShutdownAction


class _FakeExecutor(Executor):
    backend = "fake"

    def __init__(self) -> None:
        self.cancelled: list[JobHandle] = []
        self._handles: dict[str, JobHandle] = {}

    def submit(self, attempt: AttemptSpec) -> JobHandle:
        handle = JobHandle(
            backend=self.backend,
            handle_id=f"fake-{attempt.attempt_id}",
            attempt_id=attempt.attempt_id,
            metadata={"log_paths": (attempt.command.log_path,) if attempt.command.log_path else ()},
        )
        self._handles[handle.handle_id] = handle
        return handle

    def poll(self, handles: Sequence[JobHandle]) -> list[JobStatus]:
        return [
            JobStatus(handle=handle, state=JobState.RUNNING, log_paths=self.fetch_logs(handle))
            for handle in handles
        ]

    def cancel(self, handles: Sequence[JobHandle]) -> None:
        self.cancelled.extend(handles)

    def recover(self, handle: JobHandle) -> JobStatus:
        return JobStatus(handle=handle, state=JobState.RUNNING, log_paths=self.fetch_logs(handle))


def _write_configs(tmp_path: Path):
    experiment = tmp_path / "experiment.yaml"
    experiment.write_text(
        yaml.safe_dump(
            {
                "experiment": {"dir": str(tmp_path / "run")},
                "convert": {"enabled": True},
                "depth_importance": {
                    "enabled": True,
                    "max_removals": 5,
                    "expected_initial_sublayers": 80,
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
                        "vllm_stats": {"strategy": "sharded", "instances": 2},
                    },
                }
            }
        )
    )
    return experiment, runner, execution


def _write_sanity_drain_configs(tmp_path: Path):
    run_dir = tmp_path / "run"
    experiment = tmp_path / "experiment.yaml"
    experiment.write_text(
        yaml.safe_dump(
            {
                "experiment": {"dir": str(run_dir)},
                "sort_sanity": {"enabled": True},
                "width_sanity": {"enabled": True},
                "bypass_sanity": {"enabled": True},
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
                    "defaults": {"gpus_per_node": 8, "halt_policy": "drain"},
                    "stages": {
                        "sort_sanity": {"strategy": "single", "instances": 1},
                        "width_sanity": {"strategy": "single", "instances": 1},
                        "bypass_sanity": {"strategy": "single", "instances": 1},
                    },
                }
            }
        )
    )
    return experiment, runner, execution, run_dir


def _seed_sort_complete(run_dir: Path) -> None:
    manifests = run_dir / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    (manifests / "sort.json").write_text(json.dumps({"status": "success"}) + "\n")
    sorted_dir = run_dir / "ckpts" / "sorted_teacher"
    sorted_dir.mkdir(parents=True, exist_ok=True)
    (sorted_dir / "config.json").write_text("{}\n")
    (sorted_dir / "parallel_sort_manifest.json").write_text(
        json.dumps({"status": "complete"}) + "\n"
    )
    (sorted_dir / "sorted_permutations.json").write_text("{}\n")
    (sorted_dir / "model.safetensors").write_text("weights\n")


def _seed_sort_sanity_complete(run_dir: Path) -> None:
    summary = run_dir / "artifacts" / "sort_sanity" / "summary.json"
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(json.dumps({"passed": True}) + "\n")


class _TrackingFakeExecutor(_FakeExecutor):
    def __init__(self) -> None:
        super().__init__()
        self.submitted_stage_ids: list[str] = []

    def submit(self, attempt: AttemptSpec) -> JobHandle:
        self.submitted_stage_ids.append(attempt.work_id.split(":", 1)[0])
        handle = JobHandle(
            backend=self.backend,
            handle_id=f"fake-{attempt.attempt_id}",
            attempt_id=attempt.attempt_id,
            metadata={
                "work_id": attempt.work_id,
                "log_paths": (attempt.command.log_path,) if attempt.command.log_path else (),
            },
        )
        self._handles[handle.handle_id] = handle
        return handle

    @staticmethod
    def _stage_id(handle: JobHandle) -> str:
        return str(handle.metadata.get("work_id", "")).split(":", 1)[0]


def _blocked_descendants(plan, failed_stages: set[str]) -> set[str]:
    blocked = set(failed_stages)
    changed = True
    while changed:
        changed = False
        for node in plan.stages:
            if node.stage_id not in blocked and any(parent in blocked for parent in node.parents):
                blocked.add(node.stage_id)
                changed = True
    return blocked


def test_depth_progress_reports_removal_and_candidate_counts(tmp_path: Path):
    iteration = tmp_path / "depth" / "iterative" / "iteration_00"
    iteration.mkdir(parents=True)
    for index in range(3):
        (iteration / f"candidate_layer_{index:03d}_moe.json").write_text("{}")
    summary = summarize_stage_artifacts(
        tmp_path,
        "depth_importance",
        config={"depth_importance": {"max_removals": 5, "expected_initial_sublayers": 80}},
    )
    assert summary == "removing layer 1 out of 5, current progress 3/80"


def test_width_progress_reports_minibatch_from_log(tmp_path: Path):
    log = tmp_path / "width.log"
    log.write_text(
        "[activation/automodel] entering calibration loop: target 128 iteration(s)\n"
        "[activation/automodel] iter 12/128 (1.2s/iter, peak 10.0 GiB, elapsed 1.0 min)\n"
    )
    summary = summarize_stage_artifacts(
        tmp_path,
        "width_importance",
        log_paths=(str(log),),
    )
    assert summary == "minibatch 12/128"


def test_width_progress_reports_minibatch_from_resume_marker(tmp_path: Path):
    progress = (
        tmp_path
        / "pruning"
        / "pruning_scores"
        / "automodel"
        / "all_axes"
        / ".native_resume"
        / "progress.json"
    )
    progress.parent.mkdir(parents=True)
    progress.write_text('{"version": 1, "next_step": 40, "total": 128}\n')
    summary = summarize_stage_artifacts(tmp_path, "width_importance")
    assert summary == "minibatch 40/128"


def test_vllm_progress_reports_validated_over_total(tmp_path: Path):
    cache = tmp_path / "runtime_cache"
    cache.mkdir()
    cache_index = 0
    for spec_index in range(4):
        for output_len in (1, 1024):
            (cache / f"{cache_index}.json").write_text(
                json.dumps(
                    {
                        "cache_identity": {
                            "schema_version": 4,
                            "model_config": {"spec": spec_index},
                            "benchmark_args": {
                                "input_len": 8192,
                                "output_len": output_len,
                                "max_model_len": 8192 + output_len,
                                "effective_command": ["vllm", str(output_len)],
                            },
                        }
                    }
                )
            )
            cache_index += 1
    # A combined phase without its prefill pair is not a validated spec.
    (cache / "incomplete.json").write_text(
        json.dumps(
            {
                "cache_identity": {
                    "schema_version": 4,
                    "model_config": {"spec": "incomplete"},
                    "benchmark_args": {"input_len": 8192, "output_len": 1024},
                }
            }
        )
    )
    log = tmp_path / "vllm.log"
    log.write_text(
        "Computing runtime for 36 subblocks (72 unique benchmarks) across 8 GPU(s)\n"
        "Benchmarking runtime shard 1/16 (5/72 specs) on 1 GPU group(s):   "
        "5%|▌         | 2/5 [00:20<06:00, 9.00s/it]\n"
    )
    summary = summarize_stage_artifacts(
        tmp_path,
        "vllm_stats",
        config={
            "vllm_stats": {
                "generation_seq_len": 1024,
                "runtime_stats": {"granularity": "subblock"},
            }
        },
        log_paths=(str(log),),
    )
    assert summary == "validated 4/72 specs (36 subblocks)"


def test_vllm_progress_estimates_unique_subblocks_from_library(tmp_path: Path):
    library = []
    for width in (2048, 1920):
        for experts in (256, 128):
            library.append(
                {
                    "mamba_config": {
                        "kind": "mamba",
                        "name": "m",
                        "no_op": False,
                        "hidden_size": width,
                    },
                    "moe_config": {
                        "kind": "moe",
                        "name": "e",
                        "no_op": False,
                        "num_experts": experts,
                        "hidden_size": width,
                    },
                }
            )
    # Duplicate rows must not inflate the unique subblock count.
    library.extend(library)
    (tmp_path / "subblock_library.json").write_text(json.dumps(library))
    summary = summarize_stage_artifacts(
        tmp_path,
        "vllm_stats",
        config={"vllm_stats": {"runtime_stats": {"granularity": "subblock"}}},
    )
    # 2 unique mamba + 4 unique moe = 6 subblocks → 12 specs.
    assert summary == "validated 0/12 specs (6 subblocks)"


def test_summarize_active_progress_prefers_stage_lines(tmp_path: Path):
    iteration = tmp_path / "depth" / "iterative" / "iteration_00"
    iteration.mkdir(parents=True)
    (iteration / "candidate_layer_001_moe.json").write_text("{}")
    handle = JobHandle(backend="fake", handle_id="h1", attempt_id="a1")
    lines = summarize_active_progress(
        puzzle_dir=tmp_path,
        active={"h1": (handle, "depth_importance:worker:0", "a1")},
        log_paths_by_work_id={},
        config={"depth_importance": {"max_removals": 5, "expected_initial_sublayers": 80}},
    )
    assert lines == ["depth_importance: removing layer 1 out of 5, current progress 1/80"]


def test_controller_shutdown_cancels_active_jobs(tmp_path: Path):
    experiment, runner_path, execution_path = _write_configs(tmp_path)
    plan = compile_campaign_plan(
        experiment_config_path=experiment,
        runner=load_runner_config(runner_path),
        execution=load_execution_config(execution_path),
        stage_filter="convert",
    )
    executor = _FakeExecutor()
    controller = CampaignController(plan, executor=executor, poll_interval_seconds=0.01)
    result = controller.run(once=True)
    assert result["halted"] is False
    assert executor.cancelled == []
    assert controller._active

    cancelled = controller.shutdown(reason="test")
    assert cancelled == 1
    assert len(executor.cancelled) == 1
    assert controller._active == {}
    attempts = controller.store.list_attempts("convert")
    assert attempts
    assert attempts[-1]["status"] == JobState.CANCELLED.value


def test_controller_waits_for_parent_job_after_artifact_appears(tmp_path: Path, monkeypatch):
    experiment, runner_path, execution_path = _write_configs(tmp_path)
    plan = compile_campaign_plan(
        experiment_config_path=experiment,
        runner=load_runner_config(runner_path),
        execution=load_execution_config(execution_path),
        stage_filter="vllm_stats",
    )
    controller = CampaignController(plan, executor=_FakeExecutor())
    child = next(node for node in plan.stages if node.stage_id == "vllm_stats")
    handle = JobHandle(backend="fake", handle_id="parent", attempt_id="a1")
    controller._active[handle.handle_id] = (handle, "convert:0", "a1")
    monkeypatch.setattr(
        "puzzletron_orchestrator.controller.stage_is_complete",
        lambda _config, _stage_id: True,
    )

    assert not controller._parents_ready(child)
    controller._active.clear()
    assert controller._parents_ready(child)


def test_controller_completed_retry_satisfies_work_plan(tmp_path: Path, monkeypatch):
    experiment, runner_path, execution_path = _write_configs(tmp_path)
    plan = compile_campaign_plan(
        experiment_config_path=experiment,
        runner=load_runner_config(runner_path),
        execution=load_execution_config(execution_path),
        stage_filter="convert",
    )
    controller = CampaignController(plan, executor=_FakeExecutor())
    attempts = [
        {"work_id": "convert:0", "status": JobState.CANCELLED.value},
        {"work_id": "convert:0", "status": JobState.COMPLETED.value},
    ]

    assert controller._required_work_is_completed(plan.stages[0], attempts)
    monkeypatch.setattr(controller.store, "list_attempts", lambda _stage_id=None: attempts)
    assert not controller._stage_has_active_or_completed_work(plan.stages[0])


def test_controller_aggregates_completed_work_before_resubmitting(tmp_path: Path, monkeypatch):
    experiment, runner_path, execution_path = _write_configs(tmp_path)
    plan = compile_campaign_plan(
        experiment_config_path=experiment,
        runner=load_runner_config(runner_path),
        execution=load_execution_config(execution_path),
        stage_filter="convert",
    )
    node = plan.stages[0]
    delegate = adapter_for_stage(node)
    item = delegate.plan(plan, node).items[0]
    attempt = delegate.command(
        plan=plan,
        node=node,
        item=item,
        attempt_id="completed-attempt",
        runner=plan.runner,
    )
    executor = _FakeExecutor()
    controller = CampaignController(plan, executor=executor, poll_interval_seconds=0.01)
    controller.store.save_attempt(attempt, None, JobState.COMPLETED.value)

    class _AggregateAdapter:
        def __getattr__(self, name):
            return getattr(delegate, name)

        def aggregate(self, *, plan, node, work_plan):
            teacher = plan.puzzle_dir / "ckpts" / "teacher"
            teacher.mkdir(parents=True)
            (teacher / "config.json").write_text("{}")
            (plan.puzzle_dir / "subblock_library.json").write_text("[]")

    aggregate_adapter = _AggregateAdapter()
    monkeypatch.setattr(
        "puzzletron_orchestrator.controller.adapter_for_stage",
        lambda _node: aggregate_adapter,
    )

    result = controller.run(once=True)

    assert result["halted"] is False
    assert not executor._handles
    assert controller.store.stage_is_complete("convert")


def test_controller_shutdown_cancels_store_tracked_jobs(tmp_path: Path):
    experiment, runner_path, execution_path = _write_configs(tmp_path)
    plan = compile_campaign_plan(
        experiment_config_path=experiment,
        runner=load_runner_config(runner_path),
        execution=load_execution_config(execution_path),
        stage_filter="convert",
    )
    executor = _FakeExecutor()
    controller = CampaignController(plan, executor=executor, poll_interval_seconds=0.01)
    controller.run(once=True)
    # Simulate a controller crash that lost in-memory handles but left durable
    # running attempts behind.
    controller._active.clear()
    cancelled = controller.shutdown(reason="test-store")
    assert cancelled == 1
    assert len(executor.cancelled) == 1
    assert controller.store.list_attempts("convert")[-1]["status"] == JobState.CANCELLED.value


def test_controller_preserves_live_job_when_cancel_fails(tmp_path: Path):
    experiment, runner_path, execution_path = _write_configs(tmp_path)
    plan = compile_campaign_plan(
        experiment_config_path=experiment,
        runner=load_runner_config(runner_path),
        execution=load_execution_config(execution_path),
        stage_filter="convert",
    )

    class _FailingCancelExecutor(_FakeExecutor):
        def cancel(self, handles: Sequence[JobHandle]) -> None:
            raise RuntimeError("still queued")

    executor = _FailingCancelExecutor()
    controller = CampaignController(plan, executor=executor, poll_interval_seconds=0.01)
    controller.run(once=True)

    assert controller.shutdown(reason="test-failure") == 0
    assert controller.store.list_attempts("convert")[-1]["status"] == JobState.RUNNING.value
    assert len(controller.store.list_live_handles()) == 1


def test_slurm_job_id_falls_back_to_handle_id():
    from puzzletron_orchestrator.executors.slurm import _slurm_job_id

    assert (
        _slurm_job_id(
            JobHandle(backend="slurm", handle_id="slurm-14208687", attempt_id="a1", metadata={})
        )
        == "14208687"
    )
    assert (
        _slurm_job_id(
            JobHandle(
                backend="slurm",
                handle_id="slurm-1",
                attempt_id="a1",
                metadata={"job_id": "999"},
            )
        )
        == "999"
    )


def test_slurm_cancel_batches_job_ids(tmp_path: Path, monkeypatch):
    from puzzletron_orchestrator.executors.slurm import SlurmExecutor
    from puzzletron_orchestrator.schema import (
        ExecutionContract,
        RunnerEnvironment,
        SlurmRunnerConfig,
    )

    calls: list[list[str]] = []

    def _fake_run(argv):
        calls.append(list(argv))

        class _Result:
            returncode = 0
            stdout = ""
            stderr = ""

        return _Result()

    monkeypatch.setattr("puzzletron_orchestrator.executors.slurm._run_command", _fake_run)
    executor = SlurmExecutor(
        RunnerEnvironment(
            kind="slurm",
            slurm=SlurmRunnerConfig(account="test"),
            contract=ExecutionContract(repository=str(tmp_path), venv=str(tmp_path / ".venv")),
        ),
        scripts_dir=tmp_path / "sbatch",
    )
    executor.cancel(
        [
            JobHandle(backend="slurm", handle_id="slurm-11", attempt_id="a", metadata={}),
            JobHandle(
                backend="slurm",
                handle_id="slurm-12",
                attempt_id="b",
                metadata={"job_id": "12"},
            ),
        ]
    )
    assert calls == [
        ["scancel", "11", "12"],
        ["scancel", "--full", "-s", "SIGKILL", "11", "12"],
        ["squeue", "-h", "-o", "%A"],
    ]


def test_slurm_cancel_detects_jobs_that_remain_queued(tmp_path: Path, monkeypatch):
    from puzzletron_orchestrator.executors.slurm import SlurmExecutor
    from puzzletron_orchestrator.schema import (
        ExecutionContract,
        RunnerEnvironment,
        SlurmRunnerConfig,
    )

    def _fake_run(argv):
        class _Result:
            returncode = 0
            stdout = "11\n" if argv[0] == "squeue" else ""
            stderr = ""

        return _Result()

    monkeypatch.setattr("puzzletron_orchestrator.executors.slurm._run_command", _fake_run)
    monkeypatch.setattr("puzzletron_orchestrator.executors.slurm.time.sleep", lambda _: None)
    executor = SlurmExecutor(
        RunnerEnvironment(
            kind="slurm",
            slurm=SlurmRunnerConfig(account="test"),
            contract=ExecutionContract(repository=str(tmp_path), venv=str(tmp_path / ".venv")),
        ),
        scripts_dir=tmp_path / "sbatch",
    )
    handle = JobHandle(backend="slurm", handle_id="slurm-11", attempt_id="a", metadata={})
    try:
        executor.cancel([handle])
    except RuntimeError as exc:
        assert "remained queued" in str(exc)
    else:
        raise AssertionError("cancel must fail while the Slurm job remains queued")


def test_slurm_cancel_waits_for_slow_allocation_cleanup(tmp_path: Path, monkeypatch):
    from puzzletron_orchestrator.executors.slurm import SlurmExecutor
    from puzzletron_orchestrator.schema import (
        ExecutionContract,
        RunnerEnvironment,
        SlurmRunnerConfig,
    )

    queue_polls = 0

    def _fake_run(argv):
        nonlocal queue_polls

        class _Result:
            returncode = 0
            stdout = ""
            stderr = ""

        result = _Result()
        if argv[0] == "squeue":
            queue_polls += 1
            result.stdout = "11\n" if queue_polls <= 10 else ""
        return result

    monkeypatch.setattr("puzzletron_orchestrator.executors.slurm._run_command", _fake_run)
    monkeypatch.setattr("puzzletron_orchestrator.executors.slurm.time.sleep", lambda _: None)
    executor = SlurmExecutor(
        RunnerEnvironment(
            kind="slurm",
            slurm=SlurmRunnerConfig(account="test"),
            contract=ExecutionContract(repository=str(tmp_path), venv=str(tmp_path / ".venv")),
        ),
        scripts_dir=tmp_path / "sbatch",
    )

    executor.cancel([JobHandle(backend="slurm", handle_id="slurm-11", attempt_id="a", metadata={})])

    assert queue_polls == 11


def test_convert_progress_is_suppressed(tmp_path: Path):
    handle = JobHandle(backend="fake", handle_id="h1", attempt_id="a1")
    log = tmp_path / "convert.log"
    log.write_text("Puzzletron stage 'convert' finished with status success\n")
    lines = summarize_active_progress(
        puzzle_dir=tmp_path,
        active={"h1": (handle, "convert:0", "a1")},
        log_paths_by_work_id={"convert:0": (str(log),)},
        config={},
    )
    assert lines == []


def test_tokenize_progress_reports_samples_tokenized(tmp_path: Path):
    output = tmp_path / "dataset_cache" / "train.tokens"
    progress = output.parent / f".{output.name}.progress"
    progress.mkdir(parents=True)
    (progress / "worker_0000.json").write_text(
        json.dumps({"worker": 0, "rows_complete": 40, "rows_total": 100})
    )
    (progress / "worker_0001.json").write_text(
        json.dumps({"worker": 1, "rows_complete": 35, "rows_total": 100})
    )
    summary = summarize_stage_artifacts(
        tmp_path,
        "tokenize_data",
        config={
            "tokenize_data": {
                "caches": [
                    {"output": str(output), "num_samples": 200},
                    {
                        "output": str(tmp_path / "dataset_cache" / "val.tokens"),
                        "num_samples": 50,
                    },
                ]
            }
        },
    )
    assert summary == "75/250 samples tokenized"


def test_sort_progress_counts_unique_checkpoint_shards(tmp_path: Path):
    log = tmp_path / "sort.log"
    log.write_text(
        "*****************************************\n"
        "[sorted_teacher] shard complete rank=0 "
        "shard=model-00001-of-00026.safetensors (1/4)\n"
        "[sorted_teacher] shard complete rank=0 "
        "shard=model-00001-of-00026.safetensors (1/4)\n"
        "[sorted_teacher] shard complete rank=1 "
        "shard=model-00002-of-00026.safetensors (1/4)\n"
        "new kernel: registered at torch_bindings.cpp\n"
    )

    summary = summarize_stage_artifacts(tmp_path, "sort", log_paths=(str(log),))

    assert summary == "sorted checkpoint shards 2/26"


def test_width_sanity_progress_reports_parent_case(tmp_path: Path):
    log = tmp_path / "width-sanity.log"
    log.write_text(
        "[solution/automodel] parent sweep load | role=activation checkpoint=/tmp/model "
        "solutions=10 pending=10\n"
        "[solution/automodel] parent sweep candidate | "
        "role=activation solution=3 target={'layer': 1}\n"
    )

    summary = summarize_stage_artifacts(tmp_path, "width_sanity", log_paths=(str(log),))

    assert summary == "activation parent: scoring case 4/10"


def test_bypass_sanity_progress_reports_probe_step_and_loss(tmp_path: Path):
    log = tmp_path / "bypass-sanity.log"
    log.write_text(
        "[bypass/automodel] running fixed-batch overfit acceptance probe "
        "mode=fixed_smallest (1/2) for 128 steps\n"
        "[bypass/automodel] step=5/128 loss=0.125 layers=40\n"
        "GpuFreq=control_disabled\n"
    )

    summary = summarize_stage_artifacts(tmp_path, "bypass_sanity", log_paths=(str(log),))

    assert summary == "fixed_smallest probe 1/2: step 5/128, loss 0.125"


def test_unknown_stage_progress_does_not_echo_arbitrary_log_tail(tmp_path: Path):
    handle = JobHandle(backend="fake", handle_id="h1", attempt_id="a1")
    log = tmp_path / "future.log"
    log.write_text("new kernel: registered at torch_bindings.cpp\n")

    lines = summarize_active_progress(
        puzzle_dir=tmp_path,
        active={"h1": (handle, "future_stage:0", "a1")},
        log_paths_by_work_id={"future_stage:0": (str(log),)},
        config={},
    )

    assert lines == []


def test_controller_keyboard_interrupt_cancels_jobs(tmp_path: Path, monkeypatch):
    experiment, runner_path, execution_path = _write_configs(tmp_path)
    plan = compile_campaign_plan(
        experiment_config_path=experiment,
        runner=load_runner_config(runner_path),
        execution=load_execution_config(execution_path),
        stage_filter="convert",
    )
    executor = _FakeExecutor()
    controller = CampaignController(plan, executor=executor, poll_interval_seconds=0.01)

    def _raise_interrupt(_seconds: float) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(
        "puzzletron_orchestrator.controller.time.sleep",
        _raise_interrupt,
    )
    result = controller.run()
    assert result["cancelled"] is True
    assert result["halted"] is True
    assert len(executor.cancelled) == 1
    assert controller.store.list_attempts("convert")[-1]["status"] == JobState.CANCELLED.value


def test_controller_can_resume_quit_menu_then_detach_live_jobs(tmp_path: Path):
    experiment, runner_path, execution_path = _write_configs(tmp_path)
    plan = compile_campaign_plan(
        experiment_config_path=experiment,
        runner=load_runner_config(runner_path),
        execution=load_execution_config(execution_path),
        stage_filter="convert",
    )
    executor = _FakeExecutor()

    class _Controls:
        enabled = True

        def __init__(self) -> None:
            self.actions = [ShutdownAction.CONTINUE, ShutdownAction.DETACH]

        def start(self) -> None:
            return None

        def stop(self) -> None:
            return None

        def poll_quit(self) -> bool:
            return True

        def choose_shutdown(self) -> ShutdownAction:
            return self.actions.pop(0)

    controller = CampaignController(
        plan,
        executor=executor,
        poll_interval_seconds=0.01,
        terminal_controls=_Controls(),
    )

    result = controller.run()

    assert result["detached"] is True
    assert result["cancelled"] is False
    assert executor.cancelled == []
    assert controller.store.list_attempts("convert")[-1]["status"] == JobState.RUNNING.value
    assert controller.store.list_live_handles()


def test_controller_fatal_failure_drains_without_cancelling_siblings(tmp_path: Path):
    experiment, runner_path, execution_path = _write_configs(tmp_path)
    run_dir = tmp_path / "run"
    teacher = run_dir / "ckpts" / "teacher"
    teacher.mkdir(parents=True)
    (teacher / "config.json").write_text("{}")
    (run_dir / "subblock_library.json").write_text("[]")
    plan = compile_campaign_plan(
        experiment_config_path=experiment,
        runner=load_runner_config(runner_path),
        execution=load_execution_config(execution_path),
    )

    class _DrainingExecutor(_FakeExecutor):
        def __init__(self) -> None:
            super().__init__()
            self.poll_count = 0
            self.submitted_work_ids = []

        def submit(self, attempt: AttemptSpec) -> JobHandle:
            self.submitted_work_ids.append(attempt.work_id)
            return super().submit(attempt)

        def poll(self, handles: Sequence[JobHandle]) -> list[JobStatus]:
            self.poll_count += 1
            if self.poll_count == 1:
                return [
                    JobStatus(
                        handle=handle,
                        state=JobState.FAILED if index == 0 else JobState.RUNNING,
                        reason="test failure" if index == 0 else None,
                    )
                    for index, handle in enumerate(handles)
                ]
            return [JobStatus(handle=handle, state=JobState.COMPLETED) for handle in handles]

    executor = _DrainingExecutor()
    controller = CampaignController(plan, executor=executor, poll_interval_seconds=0.01)
    result = controller.run()

    assert result["halted"] is True
    assert result["cancelled"] is False
    failed_stage = executor.submitted_work_ids[0].split(":", 1)[0]
    assert result["failed_stages"] == [failed_stage]
    assert executor.cancelled == []
    assert any(
        attempt["status"] == JobState.COMPLETED.value
        and attempt["work_id"].split(":", 1)[0] != failed_stage
        for attempt in controller.store.list_attempts()
    )
    blocked = {failed_stage}
    changed = True
    while changed:
        changed = False
        for node in plan.stages:
            if node.stage_id not in blocked and any(parent in blocked for parent in node.parents):
                blocked.add(node.stage_id)
                changed = True
    attempted_stages = {
        attempt["work_id"].split(":", 1)[0] for attempt in controller.store.list_attempts()
    }
    assert not ((blocked - {failed_stage}) & attempted_stages)


def test_controller_collects_failed_attempt_log_paths(tmp_path: Path):
    experiment, runner_path, execution_path = _write_configs(tmp_path)
    plan = compile_campaign_plan(
        experiment_config_path=experiment,
        runner=load_runner_config(runner_path),
        execution=load_execution_config(execution_path),
    )
    controller = CampaignController(plan, executor=_FakeExecutor())
    attempt = AttemptSpec(
        attempt_id="failed-attempt",
        work_id="vllm_stats:0",
        stage_id="vllm_stats",
        command=CommandSpec(argv=("python", "worker.py"), log_path="/logs/fallback.log"),
    )
    handle = JobHandle("fake", "fake-failed-attempt", attempt.attempt_id)
    controller.store.save_attempt(attempt, handle, JobState.RUNNING.value)
    controller.store.update_attempt_status(
        attempt.work_id,
        attempt.attempt_id,
        JobStatus(
            handle=handle,
            state=JobState.FAILED,
            log_paths=("/logs/worker.log",),
        ),
    )

    assert controller._failed_log_paths({"vllm_stats"}) == {
        "vllm_stats": ["/logs/worker.log"]
    }


def test_controller_fatal_failure_cancels_other_jobs_in_fail_fast_mode(tmp_path: Path):
    experiment, runner_path, execution_path = _write_configs(tmp_path)
    run_dir = tmp_path / "run"
    teacher = run_dir / "ckpts" / "teacher"
    teacher.mkdir(parents=True)
    (teacher / "config.json").write_text("{}")
    (run_dir / "subblock_library.json").write_text("[]")
    plan = compile_campaign_plan(
        experiment_config_path=experiment,
        runner=load_runner_config(runner_path),
        execution=load_execution_config(execution_path),
    )

    class _FailingExecutor(_FakeExecutor):
        def poll(self, handles: Sequence[JobHandle]) -> list[JobStatus]:
            return [
                JobStatus(
                    handle=handle,
                    state=JobState.FAILED if index == 0 else JobState.RUNNING,
                    reason="test failure" if index == 0 else None,
                )
                for index, handle in enumerate(handles)
            ]

    executor = _FailingExecutor()
    controller = CampaignController(plan, executor=executor, poll_interval_seconds=0.01)
    controller._halt_policy = "fail_fast"
    result = controller.run()

    assert result["halted"] is True
    assert result["cancelled"] is False
    assert executor.cancelled
    assert all(
        attempt["status"] in {JobState.FAILED.value, JobState.CANCELLED.value}
        for attempt in controller.store.list_attempts()
    )


def test_controller_width_sanity_failure_drains_independent_bypass_sanity(
    tmp_path: Path, monkeypatch
):
    experiment, runner_path, execution_path, run_dir = _write_sanity_drain_configs(tmp_path)
    _seed_sort_complete(run_dir)
    _seed_sort_sanity_complete(run_dir)
    upstream = {"convert", "tokenize_data", "width_importance", "sort", "sort_sanity"}
    monkeypatch.setattr(
        "puzzletron_orchestrator.controller.stage_is_complete",
        lambda _config, stage_id: stage_id in upstream,
    )
    plan = compile_campaign_plan(
        experiment_config_path=experiment,
        runner=load_runner_config(runner_path),
        execution=load_execution_config(execution_path),
    )

    class _SanityDrainExecutor(_TrackingFakeExecutor):
        def __init__(self) -> None:
            super().__init__()
            self.poll_count = 0

        def poll(self, handles: Sequence[JobHandle]) -> list[JobStatus]:
            self.poll_count += 1
            statuses = []
            for handle in handles:
                stage_id = self._stage_id(handle)
                if stage_id == "width_sanity":
                    state = JobState.FAILED
                    reason = "width sanity worker failure"
                elif stage_id == "bypass_sanity":
                    state = JobState.COMPLETED if self.poll_count > 1 else JobState.RUNNING
                    reason = None
                else:
                    state = JobState.COMPLETED
                    reason = None
                statuses.append(JobStatus(handle=handle, state=state, reason=reason))
            return statuses

    executor = _SanityDrainExecutor()
    controller = CampaignController(plan, executor=executor, poll_interval_seconds=0.01)
    result = controller.run()

    assert result["halted"] is True
    assert result["cancelled"] is False
    assert result["failed_stages"] == ["width_sanity"]
    assert executor.cancelled == []
    assert "bypass_sanity" in executor.submitted_stage_ids
    assert "slicing_sanity" not in executor.submitted_stage_ids
    assert any(
        attempt["work_id"].startswith("bypass_sanity:")
        and attempt["status"] == JobState.COMPLETED.value
        for attempt in controller.store.list_attempts()
    )


def test_controller_failed_ancestor_blocks_descendant_submit(tmp_path: Path, monkeypatch):
    experiment, runner_path, execution_path, run_dir = _write_sanity_drain_configs(tmp_path)
    _seed_sort_complete(run_dir)
    upstream = {"convert", "tokenize_data", "width_importance", "sort"}
    monkeypatch.setattr(
        "puzzletron_orchestrator.controller.stage_is_complete",
        lambda _config, stage_id: stage_id in upstream,
    )
    plan = compile_campaign_plan(
        experiment_config_path=experiment,
        runner=load_runner_config(runner_path),
        execution=load_execution_config(execution_path),
    )

    class _SanityDrainExecutor(_TrackingFakeExecutor):
        def poll(self, handles: Sequence[JobHandle]) -> list[JobStatus]:
            return [
                JobStatus(
                    handle=handle,
                    state=(
                        JobState.FAILED
                        if self._stage_id(handle) == "sort_sanity"
                        else JobState.COMPLETED
                    ),
                    reason=(
                        "sort sanity worker failure"
                        if self._stage_id(handle) == "sort_sanity"
                        else None
                    ),
                )
                for handle in handles
            ]

    executor = _SanityDrainExecutor()
    controller = CampaignController(plan, executor=executor, poll_interval_seconds=0.01)
    result = controller.run()

    assert result["halted"] is True
    assert result["failed_stages"] == ["sort_sanity"]
    assert "bypass_sanity" in executor.submitted_stage_ids
    assert "width_sanity" not in executor.submitted_stage_ids
    blocked = _blocked_descendants(plan, {"sort_sanity"})
    assert "width_sanity" in blocked
    assert "bypass_sanity" not in blocked


def test_controller_multiple_failures_drain_both_sanity_branches(tmp_path: Path, monkeypatch):
    experiment, runner_path, execution_path, run_dir = _write_sanity_drain_configs(tmp_path)
    _seed_sort_complete(run_dir)
    upstream = {"convert", "tokenize_data", "width_importance", "sort"}
    monkeypatch.setattr(
        "puzzletron_orchestrator.controller.stage_is_complete",
        lambda _config, stage_id: stage_id in upstream,
    )
    plan = compile_campaign_plan(
        experiment_config_path=experiment,
        runner=load_runner_config(runner_path),
        execution=load_execution_config(execution_path),
    )

    class _SanityDrainExecutor(_TrackingFakeExecutor):
        def poll(self, handles: Sequence[JobHandle]) -> list[JobStatus]:
            return [
                JobStatus(
                    handle=handle,
                    state=(
                        JobState.FAILED
                        if self._stage_id(handle) in {"sort_sanity", "bypass_sanity"}
                        else JobState.COMPLETED
                    ),
                    reason=(
                        "sanity worker failure"
                        if self._stage_id(handle) in {"sort_sanity", "bypass_sanity"}
                        else None
                    ),
                )
                for handle in handles
            ]

    executor = _SanityDrainExecutor()
    controller = CampaignController(plan, executor=executor, poll_interval_seconds=0.01)
    result = controller.run()

    assert result["halted"] is True
    assert result["cancelled"] is False
    assert set(result["failed_stages"]) == {"sort_sanity", "bypass_sanity"}
    assert executor.cancelled == []


def test_controller_sigint_during_recovery_cancels_jobs(tmp_path: Path):
    experiment, runner_path, execution_path = _write_configs(tmp_path)
    plan = compile_campaign_plan(
        experiment_config_path=experiment,
        runner=load_runner_config(runner_path),
        execution=load_execution_config(execution_path),
        stage_filter="convert",
    )
    first_executor = _FakeExecutor()
    CampaignController(plan, executor=first_executor, poll_interval_seconds=0.01).run(once=True)

    class _InterruptingExecutor(_FakeExecutor):
        def recover(self, handle: JobHandle) -> JobStatus:
            signal.raise_signal(signal.SIGINT)
            raise AssertionError("SIGINT handler should interrupt recovery")

    executor = _InterruptingExecutor()
    controller = CampaignController(plan, executor=executor, poll_interval_seconds=0.01)
    result = controller.run()

    assert result["cancelled"] is True
    assert result["halted"] is True
    assert len(executor.cancelled) == 1
    assert controller.store.list_attempts("convert")[-1]["status"] == JobState.CANCELLED.value


def test_controller_shutdown_flag_cancels_without_keyboard_interrupt(tmp_path: Path, monkeypatch):
    experiment, runner_path, execution_path = _write_configs(tmp_path)
    plan = compile_campaign_plan(
        experiment_config_path=experiment,
        runner=load_runner_config(runner_path),
        execution=load_execution_config(execution_path),
        stage_filter="convert",
    )
    executor = _FakeExecutor()
    controller = CampaignController(plan, executor=executor, poll_interval_seconds=0.05)

    def _trip_shutdown(_seconds: float) -> None:
        controller._shutdown_requested = True

    monkeypatch.setattr(
        "puzzletron_orchestrator.controller.time.sleep",
        _trip_shutdown,
    )
    result = controller.run()
    assert result["cancelled"] is True
    assert len(executor.cancelled) == 1
    assert controller.store.list_attempts("convert")[-1]["status"] == JobState.CANCELLED.value
