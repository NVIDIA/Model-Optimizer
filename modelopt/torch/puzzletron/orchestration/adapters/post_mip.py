# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Adapter for campaign-configured post-MIP nodes."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from puzzletron_orchestrator.post_mip.records import CandidateLedger
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


def _available_evaluation_candidates(
    plan: CampaignPlan, stage_id: str, config: dict
) -> int | None:
    input_id = str(config.get("input", "source"))
    ledger = CandidateLedger(plan.puzzle_dir / "artifacts" / "post_mip")
    if input_id == "source":
        active_mip = plan.puzzle_dir / "mip" / "active_profiles.json"
        if not active_mip.is_file():
            return None
        ledger.ingest_mip(plan.puzzle_dir)
        _prefix, flow_id, _node_id_value = stage_id.split(".", 2)
        flow = plan.experiment_config["post_mip"]["flows"][flow_id]
        candidate_set = ledger.root_set(flow_id, flow["source"])
    else:
        current = (
            plan.puzzle_dir
            / "artifacts"
            / "post_mip"
            / "nodes"
            / input_id
            / "current.json"
        )
        if not current.is_file():
            return None
        candidate_set = ledger.load_candidate_set(input_id)
    return len(candidate_set.revision_ids)


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

    def plan(self, plan: CampaignPlan, node: StagePlanNode) -> WorkPlan:
        config = _node_config(plan, node.stage_id)
        node_type = str(config.get("type"))
        count = 1 if node_type in {"filter", "manual_filter"} else node.instances
        if node_type == "evaluation":
            available = _available_evaluation_candidates(plan, node.stage_id, config)
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
        log_path = plan.puzzle_dir / "logs" / f"{node.stage_id}_{item.shard_index}_{attempt_id}.log"
        distributed_worker = node_type in {"evaluation", "global_kd"} and node.gpus_per_instance > 1
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
        result = subprocess.run(
            (
                "python",
                str(script),
                "--config",
                plan.experiment_config_path,
                "--stage-id",
                node.stage_id,
                "--aggregate",
            ),
            cwd=repo,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode:
            raise RuntimeError(
                f"{node.stage_id} aggregation failed: "
                f"{result.stderr.strip() or result.stdout.strip()}"
            )
        output_lines = [line for line in result.stdout.splitlines() if line.strip()]
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
