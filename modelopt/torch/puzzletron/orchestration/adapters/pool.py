# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Persistent pool adapter for coordinator/worker stages."""

from __future__ import annotations

from pathlib import Path

from ..schema import (
    AttemptSpec,
    CampaignPlan,
    CommandSpec,
    ExecutionStrategy,
    PublishedOutput,
    StagePlanNode,
    TaskTopology,
    ValidatedResult,
    WorkItem,
    WorkPlan,
)
from .base import WorkAdapter
from .packing import packed_allocation
from .stage_compat import stage_is_complete, stage_output_patterns

__all__ = ["PersistentPoolAdapter"]

_POOL_STAGES = {
    "replacement_scoring": {
        "coordinator": "examples/puzzletron/distributed_eval/run_coordinator.sh",
        "worker": "examples/puzzletron/distributed_eval/run_worker.sh",
    },
    "depth_importance": {
        "coordinator": "examples/puzzletron/distributed_eval/run_depth_coordinator.sh",
        "worker": "examples/puzzletron/distributed_eval/run_worker.sh",
    },
}


def _replacement_widths(plan: CampaignPlan) -> tuple[int | None, ...]:
    embedding = plan.experiment_config.get("embedding_pruning") or {}
    if not bool(embedding.get("enabled", False)):
        return (None,)
    widths = tuple(int(width) for width in embedding.get("widths", ()))
    if not widths:
        raise ValueError("embedding replacement scoring requires at least one width")
    return widths


def _replacement_puzzle_dir(plan: CampaignPlan, width: int | None) -> Path:
    if width is None:
        return plan.puzzle_dir
    return plan.puzzle_dir / "scenarios" / f"width-{int(width):04d}" / "depth-00"


def _replacement_environment(plan: CampaignPlan, puzzle_dir: Path) -> dict[str, str]:
    scoring = plan.experiment_config.get("replacement_scoring") or {}
    granularity = str(scoring.get("granularity", "block"))
    is_embedding_scenario = puzzle_dir != plan.puzzle_dir
    stem = (
        "single_subblock_replacement_solutions"
        if granularity == "subblock"
        else "single_sequence_replacement_solutions"
    )
    solutions_path = puzzle_dir / f"{stem}.json"
    output_dir = puzzle_dir / "artifacts" / "replacement_scoring"
    compatibility_output_dir = puzzle_dir / f"{stem}--validation"
    if not is_embedding_scenario:
        solutions_path = Path(
            scoring.get("subblock_solutions_path")
            or scoring.get("solutions_path")
            or solutions_path
        )
        output_dir = Path(scoring.get("report_output_dir") or output_dir)
        compatibility_output_dir = Path(
            scoring.get("subblock_output_dir")
            or scoring.get("output_dir")
            or compatibility_output_dir
        )
    teacher_dir = puzzle_dir / "ckpts" / "sorted_teacher"
    return {
        "MODELOPT_SKIP_VLLM_PLUGIN": "1",
        "SOLUTIONS_PATH": str(solutions_path),
        "OUTPUT_DIR": str(output_dir),
        "COMPATIBILITY_OUTPUT_DIR": str(compatibility_output_dir),
        "REPLACEMENT_LIBRARY_PATH": str(puzzle_dir / "replacement_library.json"),
        "TEACHER_DIR": str(teacher_dir),
        "SUBBLOCK_MANIFEST_PATH": str(puzzle_dir / "subblock_replacement_manifest.json"),
        "PREPARE_SUBBLOCK_SOLUTIONS": "1" if granularity == "subblock" else "0",
        "TRUST_REMOTE_CODE": (
            "1"
            if bool((plan.experiment_config.get("model") or {}).get("trust_remote_code", False))
            else "0"
        ),
        "FINALIZE_REPLACEMENT_SCORING": "1",
        "FINALIZE_CONFIG_PATH": str(plan.experiment_config_path),
        "FINALIZE_PUZZLE_DIR": str(plan.puzzle_dir),
    }


def _replacement_overrides(plan: CampaignPlan, puzzle_dir: Path) -> tuple[str, ...]:
    if puzzle_dir == plan.puzzle_dir:
        return ()
    scoring = plan.experiment_config.get("replacement_scoring") or {}
    granularity = str(scoring.get("granularity", "block"))
    stem = (
        "single_subblock_replacement_solutions"
        if granularity == "subblock"
        else "single_sequence_replacement_solutions"
    )
    teacher = puzzle_dir / "ckpts" / "sorted_teacher"
    overrides = [
        f"puzzle_dir={puzzle_dir}",
        f"experiment.dir={puzzle_dir}",
        f"teacher_dir={teacher}",
        f"convert.teacher_dir={teacher}",
        "bypass.enabled=false",
        f"replacement_library_path={puzzle_dir / 'replacement_library.json'}",
        f"build_replacement_library.source_checkpoint_dir={teacher}",
        f"replacement_scoring.teacher_dir={teacher}",
        f"replacement_scoring.source_checkpoint_dir={teacher}",
        f"replacement_scoring.target_teacher_dir={teacher}",
        f"replacement_scoring.solutions_path={puzzle_dir / f'{stem}.json'}",
        f"replacement_scoring.output_dir={puzzle_dir / f'{stem}--validation'}",
    ]
    if scoring.get("bypass_checkpoint_dir") is not None:
        overrides.append(
            f"replacement_scoring.bypass_checkpoint_dir={puzzle_dir / 'ckpts' / 'bypass_overlay'}"
        )
    return tuple(overrides)


class PersistentPoolAdapter(WorkAdapter):
    """Launch one coordinator plus resident worker pool."""

    strategy = ExecutionStrategy.PERSISTENT_POOL

    def plan(self, plan: CampaignPlan, node: StagePlanNode) -> WorkPlan:
        if node.stage_id == "replacement_scoring":
            widths = _replacement_widths(plan)
            if node.instances < len(widths):
                raise ValueError(
                    "replacement-scoring persistent pool needs at least one worker "
                    f"per width; instances={node.instances}, widths={widths}"
                )
            workers_per_width, remainder = divmod(node.instances, len(widths))
            items = tuple(
                WorkItem(
                    work_id=(
                        f"{node.stage_id}:gang"
                        if len(widths) == 1
                        else f"{node.stage_id}:width-{int(width):04d}"
                    ),
                    stage_id=node.stage_id,
                    shard_index=index,
                    shard_count=len(widths),
                    gpus_per_instance=node.gpus_per_instance,
                    metadata={
                        "role": "gang",
                        "worker_count": workers_per_width + (index < remainder),
                        **({"width": int(width)} if width is not None else {}),
                    },
                )
                for index, width in enumerate(widths)
            )
            return WorkPlan(
                stage_id=node.stage_id,
                strategy=self.strategy,
                items=items,
                aggregate_required=True,
            )

        if node.stage_id == "depth_importance":
            return WorkPlan(
                stage_id=node.stage_id,
                strategy=self.strategy,
                items=(
                    WorkItem(
                        work_id=f"{node.stage_id}:gang",
                        stage_id=node.stage_id,
                        shard_index=0,
                        shard_count=1,
                        gpus_per_instance=node.gpus_per_instance,
                        metadata={"role": "gang", "worker_count": node.instances},
                    ),
                ),
                aggregate_required=True,
            )

        coordinator = WorkItem(
            work_id=f"{node.stage_id}:coordinator",
            stage_id=node.stage_id,
            shard_index=0,
            shard_count=node.instances + 1,
            gpus_per_instance=0,
            metadata={"role": "coordinator"},
        )
        workers = tuple(
            WorkItem(
                work_id=f"{node.stage_id}:worker:{index}",
                stage_id=node.stage_id,
                shard_index=index + 1,
                shard_count=node.instances + 1,
                gpus_per_instance=node.gpus_per_instance,
                local_gpu_ids=tuple(range(node.gpus_per_instance)),
                metadata={"role": "worker", "worker_id": index},
            )
            for index in range(node.instances)
        )
        return WorkPlan(
            stage_id=node.stage_id,
            strategy=self.strategy,
            items=(coordinator, *workers),
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
        role = item.metadata.get("role", "worker")
        log_dir = plan.puzzle_dir / "logs"
        replacement_puzzle_dir = (
            _replacement_puzzle_dir(plan, item.metadata.get("width"))
            if node.stage_id == "replacement_scoring"
            else plan.puzzle_dir
        )
        campaign_dir = replacement_puzzle_dir / "distributed_eval" / node.stage_id
        effective_overrides = list(overrides or [])
        if node.stage_id == "replacement_scoring":
            effective_overrides.extend(_replacement_overrides(plan, replacement_puzzle_dir))
        if role == "gang":
            worker_count = int(item.metadata.get("worker_count", node.instances))
            env = {
                "CAMPAIGN_DIR": str(campaign_dir),
                "CONFIG_PATH": plan.experiment_config_path,
                "PUZZLE_DIR": str(replacement_puzzle_dir),
                "WORLD_SIZE": str(node.gpus_per_instance),
                "NPROC_PER_NODE": str(node.gpus_per_instance),
                "WORKER_COUNT": str(worker_count),
            }
            if node.stage_id == "depth_importance":
                depth = plan.experiment_config.get("depth_importance") or {}
                env["OUTPUT_DIR"] = str(
                    depth.get("output_dir") or plan.puzzle_dir / "depth" / "iterative"
                )
                script = repo / "examples/puzzletron/distributed_eval/run_depth_pool.sh"
            else:
                env.update(_replacement_environment(plan, replacement_puzzle_dir))
                replacement_widths = _replacement_widths(plan)
                if len(replacement_widths) > 1:
                    width = int(item.metadata["width"])
                    env.update(
                        {
                            "FINALIZE_COMPLETION_DIR": str(
                                plan.puzzle_dir
                                / "artifacts"
                                / "replacement_scoring"
                                / ".pool_completion"
                                / plan.contract_hash
                            ),
                            "FINALIZE_COMPLETION_MARKER": f"width-{width}",
                            "FINALIZE_EXPECTED_COMPLETIONS": str(
                                len(replacement_widths)
                            ),
                        }
                    )
                script = repo / "examples/puzzletron/distributed_eval/run_replacement_pool.sh"
            for override in effective_overrides:
                existing = env.get("DISTRIBUTED_EVAL_OVERRIDES", "")
                env["DISTRIBUTED_EVAL_OVERRIDES"] = f"{existing}\n{override}".strip()
            log_path = str(log_dir / f"{node.stage_id}_gang_{attempt_id}.log")
            allocation_nodes, allocation_gpus, topology = packed_allocation(
                node, instances=worker_count
            )
            return AttemptSpec(
                attempt_id=attempt_id,
                work_id=item.work_id,
                stage_id=node.stage_id,
                command=CommandSpec(
                    argv=("bash", str(script)),
                    cwd=str(repo),
                    env=env,
                    log_path=log_path,
                    shell=False,
                ),
                allocation_nodes=allocation_nodes,
                allocation_gpus=allocation_gpus,
                exclusive=allocation_gpus == allocation_nodes * node.gpus_per_node,
                contract_hash=plan.contract_hash,
                metadata={
                    "role": role,
                    "worker_count": worker_count,
                    **(
                        {"width": int(item.metadata["width"])}
                        if item.metadata.get("width") is not None
                        else {}
                    ),
                    "gpus_per_node": node.gpus_per_node,
                    "kill_on_bad_exit": True,
                    **({"partition": node.partition} if node.partition else {}),
                },
                task_topology=topology,
            )

        scripts = _POOL_STAGES.get(node.stage_id, _POOL_STAGES["replacement_scoring"])
        script = repo / (scripts["coordinator"] if role == "coordinator" else scripts["worker"])
        log_path = str(log_dir / f"{node.stage_id}_{role}_{attempt_id}.log")
        env = {
            "CAMPAIGN_DIR": str(campaign_dir),
            "CONFIG_PATH": plan.experiment_config_path,
            "PUZZLE_DIR": str(replacement_puzzle_dir),
            "WORLD_SIZE": str(node.gpus_per_instance),
            "WORKER_ID": str(item.metadata.get("worker_id", 0)),
            "WORKER_COUNT": str(node.instances),
        }
        for override in effective_overrides:
            existing = env.get("DISTRIBUTED_EVAL_OVERRIDES", "")
            env["DISTRIBUTED_EVAL_OVERRIDES"] = f"{existing}\n{override}".strip()
        if node.stage_id == "replacement_scoring":
            env.update(_replacement_environment(plan, replacement_puzzle_dir))
        elif node.stage_id == "depth_importance":
            depth = plan.experiment_config.get("depth_importance") or {}
            env["OUTPUT_DIR"] = str(
                depth.get("output_dir") or plan.puzzle_dir / "depth" / "iterative"
            )
        if role == "worker":
            env["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu) for gpu in item.local_gpu_ids)
            env["NPROC_PER_NODE"] = str(node.gpus_per_instance)
            env["NNODES"] = "1"
            env["NODE_RANK"] = "0"
            worker_id = int(item.metadata.get("worker_id", 0))
            env["WORKER_GROUP_INDEX"] = str(worker_id)
            env["WORKER_PORT"] = str(5010 + worker_id)
            env["RDZV_ENDPOINT"] = f"127.0.0.1:{29500 + worker_id}"
            env["RDZV_ID"] = f"{node.stage_id}-{attempt_id}"
        argv = ("bash", str(script))
        # GPU partitions reject zero-GPU jobs; coordinators still need one GPU slot.
        allocation_gpus = 1 if role == "coordinator" else node.gpus_per_instance
        return AttemptSpec(
            attempt_id=attempt_id,
            work_id=item.work_id,
            stage_id=node.stage_id,
            command=CommandSpec(argv=argv, cwd=str(repo), env=env, log_path=log_path, shell=False),
            allocation_nodes=1,
            allocation_gpus=allocation_gpus,
            exclusive=False,
            contract_hash=plan.contract_hash,
            metadata={
                "role": role,
                "gpus_per_node": node.gpus_per_node,
                **({"partition": node.partition} if node.partition else {}),
            },
            task_topology=TaskTopology(gpus_per_task=allocation_gpus),
        )

    def validate(self, *, plan: CampaignPlan, node: StagePlanNode) -> ValidatedResult:
        if stage_is_complete(plan.experiment_config, node.stage_id):
            return ValidatedResult(
                valid=True,
                reason="pool stage outputs present",
                artifacts=stage_output_patterns(plan.experiment_config, node.stage_id),
            )
        return ValidatedResult(valid=False, reason="pool stage outputs missing")

    def aggregate(
        self,
        *,
        plan: CampaignPlan,
        node: StagePlanNode,
        work_plan: WorkPlan,
    ) -> PublishedOutput | None:
        return PublishedOutput(
            stage_id=node.stage_id,
            artifacts=stage_output_patterns(plan.experiment_config, node.stage_id),
            summary={"workers": node.instances},
        )
