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

"""Tests for post-MIP orchestration adapter launch policy."""

import json
import subprocess
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import puzzletron_orchestrator.adapters.post_mip as post_mip_adapter
from puzzletron_orchestrator.adapters.post_mip import PostMIPAdapter
from puzzletron_orchestrator.schema import (
    CampaignPlan,
    ExecutionContract,
    ExecutionStrategy,
    FailurePolicy,
    RunnerEnvironment,
    StagePlanNode,
    TaskLauncher,
    WorkItem,
    WorkPlan,
)


def _plan(tmp_path: Path, *, stage_id: str, node_type: str) -> tuple[CampaignPlan, StagePlanNode]:
    puzzle_dir = tmp_path / "run"
    puzzle_dir.mkdir()
    flow_id, node_id = stage_id.removeprefix("post.").split(".", 1)
    node = StagePlanNode(
        stage_id=stage_id,
        strategy=ExecutionStrategy.SHARDED,
        instances=2,
        failure_policy=FailurePolicy.STRICT,
        mesh={},
        gpus_per_instance=1,
        gpus_per_node=8,
        nodes=1,
        total_gpus=2,
        exclusive=False,
        parents=("mip",),
        distributed=True,
    )
    plan = CampaignPlan(
        experiment_config_path=str(tmp_path / "experiment.yaml"),
        puzzle_dir=puzzle_dir,
        experiment_config={
            "puzzle_dir": str(puzzle_dir),
            "post_mip": {"flows": {flow_id: {"nodes": {node_id: {"type": node_type}}}}},
        },
        runner=RunnerEnvironment(
            kind="slurm",
            contract=ExecutionContract(repository=str(tmp_path), venv=str(tmp_path / ".venv")),
        ),
        execution_defaults={"gpus_per_node": 8},
        stages=(node,),
        contract_hash="contract",
    )
    return plan, node


class _PostMIPExecutionContractUnavailableError(RuntimeError):
    pass


def _candidate_count_api(candidate_count: int | Exception):
    def expected_post_mip_candidate_count(_config, _stage_id):
        if isinstance(candidate_count, Exception):
            raise candidate_count
        return candidate_count

    return SimpleNamespace(
        PostMIPExecutionContractUnavailable=_PostMIPExecutionContractUnavailableError,
        expected_post_mip_candidate_count=expected_post_mip_candidate_count,
    )


def test_post_mip_evaluation_preserves_pre_ledger_dry_run_fallback(tmp_path: Path, monkeypatch):
    plan, node = _plan(tmp_path, stage_id="post.params.online_eval", node_type="evaluation")
    identity_api = _candidate_count_api(
        _PostMIPExecutionContractUnavailableError("post-MIP candidate registry is unavailable")
    )
    monkeypatch.setattr(post_mip_adapter, "_post_mip_identity_api", lambda: identity_api)

    work_plan = PostMIPAdapter().plan(plan, node)

    assert work_plan.items[0].metadata["logical_shard_count"] == node.instances


def test_post_mip_evaluation_with_existing_registry_fails_closed(tmp_path: Path, monkeypatch):
    plan, node = _plan(tmp_path, stage_id="post.params.online_eval", node_type="evaluation")
    registry = plan.puzzle_dir / "artifacts" / "post_mip" / "candidate_registry.json"
    registry.parent.mkdir(parents=True)
    registry.write_text("{}\n")
    message = "post-MIP candidate registry does not reflect the active MIP execution"
    identity_api = _candidate_count_api(_PostMIPExecutionContractUnavailableError(message))
    monkeypatch.setattr(post_mip_adapter, "_post_mip_identity_api", lambda: identity_api)

    with pytest.raises(_PostMIPExecutionContractUnavailableError, match=message):
        PostMIPAdapter().plan(plan, node)


def test_post_mip_evaluation_clamps_workers_to_available_candidates(tmp_path: Path, monkeypatch):
    plan, node = _plan(tmp_path, stage_id="post.params.online_eval", node_type="evaluation")
    identity_api = _candidate_count_api(1)
    monkeypatch.setattr(post_mip_adapter, "_post_mip_identity_api", lambda: identity_api)

    work_plan = PostMIPAdapter().plan(plan, node)

    assert [item.work_id for item in work_plan.items] == [f"{node.stage_id}:0"]


def test_post_mip_evaluation_uses_torchrun_for_single_gpu_workers(tmp_path: Path):
    plan, node = _plan(tmp_path, stage_id="post.params.online_eval", node_type="evaluation")
    attempt = PostMIPAdapter().command(
        plan=plan,
        node=node,
        item=WorkItem(
            work_id=f"{node.stage_id}:gang",
            stage_id=node.stage_id,
            shard_index=0,
            shard_count=1,
            gpus_per_instance=1,
            metadata={"logical_shard_count": 2},
        ),
        attempt_id="a1",
        runner=plan.runner,
    )

    assert attempt.task_topology.launcher is TaskLauncher.TORCHRUN
    assert attempt.task_topology.gpus_per_task == 1
    assert attempt.task_topology.tasks_per_group == 1


def test_post_mip_global_kd_uses_torchrun_for_single_gpu_workers(tmp_path: Path):
    plan, node = _plan(tmp_path, stage_id="post.params.short_kd", node_type="global_kd")
    attempt = PostMIPAdapter().command(
        plan=plan,
        node=node,
        item=WorkItem(
            work_id=f"{node.stage_id}:0",
            stage_id=node.stage_id,
            shard_index=0,
            shard_count=1,
            gpus_per_instance=1,
        ),
        attempt_id="a1",
        runner=plan.runner,
    )

    assert attempt.task_topology.launcher is TaskLauncher.TORCHRUN


def test_post_mip_filter_keeps_direct_launcher(tmp_path: Path):
    plan, node = _plan(tmp_path, stage_id="post.params.best_lm", node_type="filter")
    node = StagePlanNode(
        stage_id=node.stage_id,
        strategy=node.strategy,
        instances=1,
        failure_policy=node.failure_policy,
        mesh=node.mesh,
        gpus_per_instance=0,
        gpus_per_node=0,
        nodes=1,
        total_gpus=0,
        exclusive=False,
        parents=node.parents,
        distributed=False,
        resource="cpu",
    )
    attempt = PostMIPAdapter().command(
        plan=plan,
        node=node,
        item=WorkItem(
            work_id=f"{node.stage_id}:0",
            stage_id=node.stage_id,
            shard_index=0,
            shard_count=1,
            gpus_per_instance=0,
        ),
        attempt_id="a1",
        runner=plan.runner,
    )

    assert attempt.task_topology.launcher is TaskLauncher.DIRECT


def test_post_mip_aggregation_forwards_campaign_overrides(tmp_path: Path, monkeypatch):
    plan, node = _plan(tmp_path, stage_id="post.params.online_eval", node_type="evaluation")
    plan = replace(
        plan,
        overrides=(
            "post_mip.flows.params.nodes.online_eval.config.eval_samples=2",
            "+post_mip.flows.params.nodes.short_kd.config.checkpoint_every_steps=2",
        ),
    )
    commands = []

    def run(command, **_kwargs):
        commands.append(tuple(command))
        return subprocess.CompletedProcess(command, 0, stdout=json.dumps({"status": "success"}))

    monkeypatch.setattr("puzzletron_orchestrator.adapters.post_mip.subprocess.run", run)

    publication = PostMIPAdapter().aggregate(
        plan=plan,
        node=node,
        work_plan=WorkPlan(stage_id=node.stage_id, strategy=node.strategy, items=()),
    )

    assert commands == [
        (
            "python",
            str(tmp_path / "examples" / "puzzletron" / "run_post_mip_node.py"),
            "--config",
            plan.experiment_config_path,
            "--stage-id",
            node.stage_id,
            "--aggregate",
            "--override",
            plan.overrides[0],
            "--override",
            plan.overrides[1],
        )
    ]
    assert publication is not None
    assert publication.summary == {"status": "success"}
