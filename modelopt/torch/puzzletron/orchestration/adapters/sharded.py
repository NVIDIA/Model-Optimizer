# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sharded stage adapter for independent worker instances."""

from __future__ import annotations

import subprocess
from dataclasses import replace
from pathlib import Path

from ..schema import (
    AttemptSpec,
    CampaignPlan,
    CommandSpec,
    ExecutionStrategy,
    PublishedOutput,
    StagePlanNode,
    ValidatedResult,
    WorkItem,
    WorkPlan,
)
from ..vllm_measurements import normalize_vllm_measurements
from .base import WorkAdapter
from .packing import packed_allocation
from .stage_compat import stage_is_complete, stage_output_patterns

__all__ = ["ShardedStageAdapter"]

_SHARDED_ENTRYPOINTS = {
    "vllm_stats": ("examples/puzzletron/run_runtime_stats_shard.py", []),
    "zero_shot_evaluation": (
        "examples/puzzletron/run_profile_online_evaluation.py",
        ["--run-shard"],
    ),
    "aiperf": ("examples/puzzletron/run_profile_aiperf_worker.py", []),
}


def _gang_item(
    node: StagePlanNode,
    *,
    instances: int | None = None,
    suffix: str = "gang",
    metadata: dict | None = None,
) -> WorkItem:
    logical_count = int(node.instances if instances is None else instances)
    return WorkItem(
        work_id=f"{node.stage_id}:{suffix}",
        stage_id=node.stage_id,
        shard_index=0,
        shard_count=1,
        gpus_per_instance=node.gpus_per_instance,
        metadata={"logical_shard_count": logical_count, **(metadata or {})},
    )


class ShardedStageAdapter(WorkAdapter):
    """Launch deterministic one-shot shards for fan-out stages."""

    strategy = ExecutionStrategy.SHARDED

    def plan(self, plan: CampaignPlan, node: StagePlanNode) -> WorkPlan:
        if node.stage_id == "vllm_stats":
            items = tuple(
                _gang_item(
                    node,
                    suffix=f"{measurement_id}:gang",
                    metadata={
                        "measurement_id": measurement_id,
                        "measurement_identity": measurement.identity,
                        "runtime_topology": dict(measurement.topology),
                    },
                )
                for measurement_id, measurement in normalize_vllm_measurements(
                    plan.experiment_config
                ).items()
            )
            items = tuple(
                replace(item, gpus_per_instance=measurement.gpu_group_size)
                for item, measurement in zip(
                    items,
                    normalize_vllm_measurements(plan.experiment_config).values(),
                )
            )
            return WorkPlan(
                stage_id=node.stage_id,
                strategy=self.strategy,
                items=items,
                aggregate_required=True,
            )
        if node.stage_id == "zero_shot_evaluation":
            widths = tuple(
                int(width)
                for width in (plan.experiment_config.get("embedding_pruning") or {}).get(
                    "widths", []
                )
            ) or (None,)
            if node.instances % len(widths):
                raise ValueError(
                    "zero_shot_evaluation instances must be divisible by the number "
                    f"of embedding widths; got instances={node.instances}, widths={widths}"
                )
            shards_per_width = node.instances // len(widths)
            items = tuple(
                _gang_item(
                    node,
                    instances=shards_per_width,
                    suffix=f"w{width}:gang",
                    metadata={"width": width},
                )
                for width in widths
            )
            return WorkPlan(
                stage_id=node.stage_id,
                strategy=self.strategy,
                items=items,
                aggregate_required=True,
            )
        return WorkPlan(
            stage_id=node.stage_id,
            strategy=self.strategy,
            items=(_gang_item(node),),
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
        log_dir = plan.puzzle_dir / "logs"
        logical_count = int(item.metadata.get("logical_shard_count", node.instances))
        script, extra_args = _SHARDED_ENTRYPOINTS.get(
            node.stage_id,
            ("examples/puzzletron/main.py", ["--worker-stage", node.stage_id]),
        )
        script_path = repo / script
        measurement_id = item.metadata.get("measurement_id")
        name_suffix = f"_{measurement_id}" if measurement_id else ""
        log_path = str(
            log_dir
            / f"{node.stage_id}{name_suffix}_shard{item.shard_index}_{attempt_id}.log"
        )
        if node.stage_id == "aiperf":
            aiperf = plan.experiment_config.get("aiperf") or {}
            argv = [
                "python",
                str(script_path),
                "--puzzle-dir",
                str(plan.puzzle_dir),
                "--profile-id",
                str(aiperf.get("profile_id", "runtime-075")),
                "--worker-count",
                str(logical_count),
                "--input-tokens",
                str(aiperf.get("input_tokens", 8192)),
                "--output-tokens",
                str(aiperf.get("output_tokens", 1024)),
            ]
        else:
            argv = ["python", str(script_path), "--config", plan.experiment_config_path]
            argv.extend(extra_args)
        if node.stage_id == "zero_shot_evaluation":
            argv.extend(
                [
                    "--puzzle-dir",
                    str(plan.puzzle_dir),
                    "--shard-count",
                    str(logical_count),
                ]
            )
            width = item.metadata.get("width")
            if width is not None:
                argv.extend(["--width", str(width)])
        elif node.stage_id == "vllm_stats" and measurement_id is not None:
            argv.extend(["--measurement-id", str(measurement_id)])
        env = {}
        if node.stage_id == "vllm_stats":
            env["PUZZLETRON_RUNTIME_SHARD_COUNT"] = str(logical_count)
        elif node.stage_id == "replacement_scoring":
            env["MODELOPT_SKIP_VLLM_PLUGIN"] = "1"
        if node.stage_id != "aiperf":
            for override in overrides or []:
                argv.extend(["--override", override])
        allocation_node = node
        if node.stage_id == "vllm_stats":
            allocation_node = replace(
                node,
                gpus_per_instance=item.gpus_per_instance,
                exclusive=False,
            )
        allocation_nodes, allocation_gpus, topology = packed_allocation(
            allocation_node, instances=logical_count
        )
        return AttemptSpec(
            attempt_id=attempt_id,
            work_id=item.work_id,
            stage_id=node.stage_id,
            command=CommandSpec(argv=tuple(argv), cwd=str(repo), env=env, log_path=log_path),
            allocation_nodes=allocation_nodes,
            allocation_gpus=allocation_gpus,
            exclusive=False,
            contract_hash=plan.contract_hash,
            metadata={
                "shard_index": item.shard_index,
                "shard_count": logical_count,
                "gpus_per_node": node.gpus_per_node,
                **({"partition": node.partition} if node.partition else {}),
            },
            task_topology=topology,
        )

    def validate(self, *, plan: CampaignPlan, node: StagePlanNode) -> ValidatedResult:
        if stage_is_complete(plan.experiment_config, node.stage_id):
            return ValidatedResult(
                valid=True,
                reason="aggregate stage outputs present",
                artifacts=stage_output_patterns(plan.experiment_config, node.stage_id),
            )
        return ValidatedResult(valid=False, reason="aggregate stage outputs missing")

    def aggregate(
        self,
        *,
        plan: CampaignPlan,
        node: StagePlanNode,
        work_plan: WorkPlan,
    ) -> PublishedOutput | None:
        command: tuple[str, ...] | None = None
        if node.stage_id == "zero_shot_evaluation":
            repo = Path(plan.runner.contract.repository)
            merge_script = repo / "examples" / "puzzletron" / "run_profile_online_evaluation.py"
            evaluation = plan.experiment_config.get("zero_shot_evaluation") or {}
            command = (
                "python",
                str(merge_script),
                "--puzzle-dir",
                str(plan.puzzle_dir),
                "--merge",
                "--eval-samples",
                str(evaluation.get("eval_samples", 128)),
                "--block-size",
                str(evaluation.get("block_size", 8192)),
            )
        elif node.stage_id == "aiperf":
            repo = Path(plan.runner.contract.repository)
            merge_script = repo / "examples" / "puzzletron" / "run_profile_aiperf_worker.py"
            aiperf = plan.experiment_config.get("aiperf") or {}
            command = (
                "python",
                str(merge_script),
                "--puzzle-dir",
                str(plan.puzzle_dir),
                "--profile-id",
                str(aiperf.get("profile_id", "runtime-075")),
                "--merge",
            )
        elif node.stage_id == "vllm_stats":
            repo = Path(plan.runner.contract.repository)
            merge_script = repo / "examples" / "puzzletron" / "run_runtime_stats_shard.py"
            command = (
                "python",
                str(merge_script),
                "--config",
                plan.experiment_config_path,
                "--merge",
            )
        if command is not None:
            result = subprocess.run(
                command,
                cwd=plan.runner.contract.repository,
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode:
                raise RuntimeError(
                    f"{node.stage_id} aggregation failed: "
                    f"{result.stderr.strip() or result.stdout.strip()}"
                )
        return PublishedOutput(
            stage_id=node.stage_id,
            artifacts=stage_output_patterns(plan.experiment_config, node.stage_id),
            summary={"merge_command": command} if command is not None else {},
        )
