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

"""Adapter for campaign-configured post-MIP nodes."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Sequence

from ..schema import (
    AttemptSpec,
    CampaignPlan,
    CommandSpec,
    ExecutionStrategy,
    PublishedOutput,
    StagePlanNode,
    TaskLauncher,
    ValidatedResult,
    WorkItem,
    WorkPlan,
)
from .base import WorkAdapter
from .packing import packed_allocation
from .stage_compat import _hf_checkpoint_is_complete, post_mip_summary_is_current

__all__ = ["ManualInputRequired", "PostMIPAdapter"]

_DEFAULT_AGGREGATION_TIMEOUT_SECONDS = 300.0


async def _communicate_with_timeout(
    argv: Sequence[str], *, cwd: Path, timeout_seconds: float
) -> tuple[int, str, str]:
    process = await asyncio.create_subprocess_exec(
        *argv,
        cwd=cwd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout_seconds)
    except TimeoutError:
        process.kill()
        await process.communicate()
        raise
    return_code = process.returncode
    if return_code is None:
        raise RuntimeError("aggregation subprocess exited without a return code")
    return return_code, stdout.decode(), stderr.decode()


def _run_aggregation_command(
    argv: Sequence[str], *, cwd: Path, timeout_seconds: float
) -> tuple[int, str, str]:
    """Run one argv-only aggregation process within a finite deadline."""

    return asyncio.run(_communicate_with_timeout(argv, cwd=cwd, timeout_seconds=timeout_seconds))


def _post_mip_identity_api() -> Any:
    """Load the producer identity contract after orchestration initialization."""

    if (__package__ or "").startswith("puzzletron_orchestrator."):
        from puzzletron_orchestrator.post_mip import identity as identity_api
    else:
        from ...post_mip import identity as identity_api
    return identity_api


class ManualInputRequired(RuntimeError):
    """A durable manual-filter review exists and needs a user decision."""

    def __init__(
        self,
        node_id: str,
        revision_ids: tuple[str, ...],
        prompt: str,
        execution_identity: str,
    ):
        super().__init__(f"post-MIP node {node_id!r} is waiting for manual input")
        self.node_id = node_id
        self.revision_ids = revision_ids
        self.prompt = prompt
        self.execution_identity = execution_identity


def _node_id(stage_id: str) -> str:
    parts = stage_id.split(".", 2)
    if len(parts) != 3 or parts[0] != "post":
        raise ValueError(f"invalid post-MIP stage ID {stage_id!r}")
    return parts[2]


def _node_config(plan: CampaignPlan, stage_id: str) -> dict:
    _prefix, flow_id, node_id = stage_id.split(".", 2)
    return dict(plan.experiment_config["post_mip"]["flows"][flow_id]["nodes"][node_id])


def _node_root(plan: CampaignPlan, stage_id: str) -> Path:
    return plan.puzzle_dir / "artifacts" / "post_mip" / "nodes" / _node_id(stage_id)


def _identity_config(plan: CampaignPlan) -> dict[str, Any]:
    return {**plan.experiment_config, "puzzle_dir": str(plan.puzzle_dir)}


def _available_evaluation_candidates(plan: CampaignPlan, stage_id: str) -> int | None:
    identity_api = _post_mip_identity_api()
    try:
        return identity_api.expected_post_mip_candidate_count(_identity_config(plan), stage_id)
    except identity_api.PostMIPExecutionContractUnavailable:
        registry = plan.puzzle_dir / "artifacts" / "post_mip" / "candidate_registry.json"
        if registry.exists():
            raise
        return None


def _full_node_instance_count(node: StagePlanNode, count: int) -> int:
    if not 0 < node.gpus_per_instance < node.gpus_per_node:
        return count
    if node.gpus_per_node % node.gpus_per_instance:
        return count
    instances_per_node = node.gpus_per_node // node.gpus_per_instance
    if count <= instances_per_node:
        return count
    return count - count % instances_per_node


class PostMIPAdapter(WorkAdapter):
    """Launch independent candidate workers and publish one candidate set."""

    strategy = ExecutionStrategy.SHARDED

    def prepare_execution_identity_projection(
        self,
        *,
        plan: CampaignPlan,
        node: StagePlanNode,
    ) -> None:
        """Prepare the candidate registry only on the attempt-submission path."""

        del node
        _post_mip_identity_api().prepare_post_mip_candidate_ledger(_identity_config(plan))

    def execution_identity_projection(
        self,
        *,
        plan: CampaignPlan,
        node: StagePlanNode,
        work_plan: WorkPlan,
    ) -> dict[str, Any]:
        """Bind scheduler attempts to the canonical producer execution contract."""

        del work_plan
        return _post_mip_identity_api().expected_post_mip_execution_contract(
            _identity_config(plan), node.stage_id
        )

    def plan(self, plan: CampaignPlan, node: StagePlanNode) -> WorkPlan:
        config = _node_config(plan, node.stage_id)
        node_type = str(config.get("type"))
        count = 1 if node_type in {"filter", "manual_filter"} else node.instances
        if node_type in {"evaluation", "downstream_evaluation"}:
            available = _available_evaluation_candidates(plan, node.stage_id)
            if available is not None:
                if available < 1:
                    raise RuntimeError(
                        f"{node.stage_id} has no candidate architectures to evaluate"
                    )
                count = min(count, available)
            count = _full_node_instance_count(node, count)
        if count == 1:
            items = (
                WorkItem(
                    work_id=f"{node.stage_id}:0",
                    stage_id=node.stage_id,
                    shard_index=0,
                    shard_count=1,
                    gpus_per_instance=node.gpus_per_instance,
                ),
            )
        else:
            items = (
                WorkItem(
                    work_id=f"{node.stage_id}:gang",
                    stage_id=node.stage_id,
                    shard_index=0,
                    shard_count=1,
                    gpus_per_instance=node.gpus_per_instance,
                    metadata={"logical_shard_count": count},
                ),
            )
        return WorkPlan(
            stage_id=node.stage_id,
            strategy=node.strategy,
            items=items,
            aggregate_required=True,
        )

    def command(
        self,
        *,
        plan: CampaignPlan,
        node: StagePlanNode,
        item: WorkItem,
        attempt_id: str,
        runner,
        overrides: list[str] | None = None,
    ) -> AttemptSpec:
        repo = Path(runner.contract.repository)
        script = repo / "examples" / "puzzletron" / "run_post_mip_node.py"
        node_type = str(_node_config(plan, node.stage_id).get("type"))
        argv = [
            "python",
            str(script),
            "--config",
            plan.experiment_config_path,
            "--stage-id",
            node.stage_id,
        ]
        if node_type in {"filter", "manual_filter"}:
            argv.append("--aggregate")
        else:
            argv.extend(
                [
                    "--shard-count",
                    str(int(item.metadata.get("logical_shard_count", node.instances))),
                ]
            )
        for override in overrides or ():
            argv.extend(["--override", override])
        log_path = plan.log_dir / f"{node.stage_id}_{item.shard_index}_{attempt_id}.log"
        # evaluation/global_kd always call torch.distributed, so even 1-GPU
        # workers need torchrun to export RANK/WORLD_SIZE.
        distributed_worker = node_type in {"evaluation", "global_kd"}
        logical_count = int(item.metadata.get("logical_shard_count", 1))
        allocation_nodes, allocation_gpus, topology = packed_allocation(
            node,
            instances=logical_count,
            launcher=TaskLauncher.TORCHRUN if distributed_worker else TaskLauncher.DIRECT,
        )
        allocated_gpus_per_node = node.gpus_per_node
        if allocation_nodes == 1 and allocation_gpus < allocated_gpus_per_node:
            allocated_gpus_per_node = allocation_gpus
        return AttemptSpec(
            attempt_id=attempt_id,
            work_id=item.work_id,
            stage_id=node.stage_id,
            command=CommandSpec(argv=tuple(argv), cwd=str(repo), env={}, log_path=str(log_path)),
            allocation_nodes=allocation_nodes,
            allocation_gpus=allocation_gpus,
            exclusive=allocation_gpus == allocation_nodes * allocated_gpus_per_node,
            contract_hash=plan.contract_hash,
            metadata={
                "shard_index": item.shard_index,
                "shard_count": logical_count,
                "gpus_per_node": allocated_gpus_per_node,
                **({"partition": node.partition} if node.partition else {}),
            },
            task_topology=topology,
        )

    def aggregate(
        self,
        *,
        plan: CampaignPlan,
        node: StagePlanNode,
        work_plan: WorkPlan,
    ) -> PublishedOutput | None:
        repo = Path(plan.runner.contract.repository)
        script = repo / "examples" / "puzzletron" / "run_post_mip_node.py"
        argv = [
            "python",
            str(script),
            "--config",
            plan.experiment_config_path,
            "--stage-id",
            node.stage_id,
            "--aggregate",
        ]
        for override in plan.overrides:
            argv.extend(["--override", override])
        timeout_seconds = float(
            plan.execution_defaults.get(
                "artifact_settling_timeout_seconds",
                _DEFAULT_AGGREGATION_TIMEOUT_SECONDS,
            )
        )
        try:
            return_code, stdout, stderr = _run_aggregation_command(
                argv,
                cwd=repo,
                timeout_seconds=timeout_seconds,
            )
        except TimeoutError as error:
            raise RuntimeError(
                f"{node.stage_id} aggregation timed out after {timeout_seconds:g}s"
            ) from error
        if return_code:
            raise RuntimeError(
                f"{node.stage_id} aggregation failed: {stderr.strip() or stdout.strip()}"
            )
        output_lines = [line for line in stdout.splitlines() if line.strip()]
        if not output_lines:
            raise RuntimeError(f"{node.stage_id} aggregation produced no summary")
        payload = json.loads(output_lines[-1])
        if payload.get("status") == "waiting_for_input":
            config = _node_config(plan, node.stage_id)
            raise ManualInputRequired(
                _node_id(node.stage_id),
                tuple(str(value) for value in payload.get("revision_ids") or ()),
                str(config.get("prompt") or "Select the candidates to continue"),
                str(payload["execution_identity"]),
            )
        return PublishedOutput(
            stage_id=node.stage_id,
            artifacts=(str(_node_root(plan, node.stage_id) / "summary.json"),),
            summary=payload,
        )

    def validate(self, *, plan: CampaignPlan, node: StagePlanNode) -> ValidatedResult:
        summary_path = _node_root(plan, node.stage_id) / "summary.json"
        try:
            payload = json.loads(summary_path.read_text())
        except (OSError, ValueError):
            return ValidatedResult(valid=False, reason="post-MIP summary is missing")
        if payload.get("status") != "success":
            return ValidatedResult(valid=False, reason="post-MIP summary is not successful")
        if not post_mip_summary_is_current(
            plan.experiment_config, plan.puzzle_dir, node.stage_id, payload
        ):
            return ValidatedResult(
                valid=False, reason="post-MIP summary belongs to a stale execution"
            )
        incomplete = [
            str(path)
            for path in payload.get("checkpoints") or ()
            if not _hf_checkpoint_is_complete(Path(str(path)))
        ]
        if incomplete:
            return ValidatedResult(
                valid=False,
                reason=f"post-MIP checkpoints are incomplete: {incomplete}",
            )
        return ValidatedResult(
            valid=True,
            reason="post-MIP candidate set published",
            artifacts=(str(summary_path),),
        )
