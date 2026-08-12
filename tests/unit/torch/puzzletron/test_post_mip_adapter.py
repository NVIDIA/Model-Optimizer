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

from pathlib import Path

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
