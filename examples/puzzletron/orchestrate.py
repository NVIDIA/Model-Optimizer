#!/usr/bin/env python3
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

"""CLI entrypoint for the Puzzletron v2 campaign orchestrator."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from examples.puzzletron.expectations import (  # noqa: E402
    ExpectationResult,
    verify_expected_results,
)
from puzzletron_orchestrator.compiler import (  # noqa: E402
    compile_campaign_plan,
    load_execution_config,
    load_runner_config,
    validate_runner_ready,
)
from puzzletron_orchestrator.controller import CampaignController, dry_run_plan  # noqa: E402
from puzzletron_orchestrator.executors.slurm import render_slurm_attempt_script  # noqa: E402
from puzzletron_orchestrator.logging import OrchestratorLogger  # noqa: E402
from puzzletron_orchestrator.reusable_allocation import (  # noqa: E402
    REUSABLE_PLAN_IDENTITY_ENV,
    build_reusable_allocation_attempt,
    reusable_plan_identity,
    run_reusable_allocation,
)
from puzzletron_orchestrator.schema import CampaignPlan, ExecutionMode  # noqa: E402
from puzzletron_orchestrator.state import CampaignStateStore  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run or resume a Puzzletron v2 campaign from its three configuration files."
    )
    parser.add_argument(
        "--experiment",
        required=True,
        help="Experiment YAML: model, data, enabled stages, and output directory.",
    )
    parser.add_argument(
        "--runner",
        required=True,
        help=(
            "Runner YAML: where and how worker jobs run, including repository, "
            "environment, container, and mounts."
        ),
    )
    parser.add_argument(
        "--execution",
        required=True,
        help="Execution YAML: how each stage runs, including resources and failure policy.",
    )
    parser.add_argument(
        "--stage",
        default="full",
        help=(
            "'full' runs every enabled stage in dependency order; a stage id runs only "
            "that stage and requires its parent artifacts."
        ),
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Repeatable config override; KEY=VALUE and ++KEY=VALUE are supported.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compile and print packed submissions and scheduler scripts without submitting jobs.",
    )
    parser.add_argument("--local", action="store_true", help="Use the local subprocess executor.")
    parser.add_argument(
        "--once",
        action="store_true",
        help=(
            "Recover, poll, and submit ready work once, then exit; jobs keep running and "
            "the same command continues them."
        ),
    )
    parser.add_argument("--max-iterations", type=int, default=None)
    parser.add_argument(
        "--color",
        choices=("auto", "always", "never"),
        default="auto",
        help="Colorize progress logs on stderr (default: auto).",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=5.0,
        help="Seconds between scheduler polls (default: 5).",
    )
    parser.add_argument(
        "--expect",
        type=Path,
        help=(
            "After full completion, require the final report and verify campaign artifacts "
            "against this versioned expectation contract."
        ),
    )
    return parser


def _reusable_worker_command(
    args: argparse.Namespace, plan: CampaignPlan, plan_identity: str
) -> tuple[str, ...]:
    inputs = plan.puzzle_dir / "orchestration" / "reusable_inputs" / plan_identity
    command = [
        "python",
        "examples/puzzletron/orchestrate.py",
        "--experiment",
        plan.experiment_config_path,
        "--runner",
        str(inputs / "runner.yaml"),
        "--execution",
        str(inputs / "execution.yaml"),
        "--stage",
        args.stage,
        "--local",
        "--color",
        "never",
        "--poll-interval",
        str(args.poll_interval),
    ]
    for override in args.override:
        command.extend(("--override", override))
    if args.max_iterations is not None:
        command.extend(("--max-iterations", str(args.max_iterations)))
    return tuple(command)


def _write_reusable_inputs(
    args: argparse.Namespace, plan: CampaignPlan, plan_identity: str
) -> None:
    store = CampaignStateStore(plan.puzzle_dir)
    store.write_allocation_input(
        plan_identity=plan_identity,
        name="runner.yaml",
        contents=Path(args.runner).read_text(),
    )
    store.write_allocation_input(
        plan_identity=plan_identity,
        name="execution.yaml",
        contents=Path(args.execution).read_text(),
    )


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logger = OrchestratorLogger(color=args.color)
    try:
        runner = load_runner_config(args.runner)
        if not args.dry_run:
            validate_runner_ready(runner)
        execution = load_execution_config(args.execution)
        plan = compile_campaign_plan(
            experiment_config_path=args.experiment,
            runner=runner,
            execution=execution,
            overrides=args.override,
            stage_filter=args.stage,
        )
        submissions = dry_run_plan(plan, overrides=args.override) if args.dry_run else None
    except (KeyError, OSError, TypeError, ValueError, yaml.YAMLError) as error:
        logger.error(f"cannot build campaign plan: {error}")
        return 2
    if args.dry_run:
        assert submissions is not None
        allocation = None
        if plan.execution_mode is ExecutionMode.REUSABLE_ALLOCATION and not args.local:
            identity = reusable_plan_identity(plan)
            command = _reusable_worker_command(args, plan, identity)
            attempt = build_reusable_allocation_attempt(
                plan,
                command,
                attempt_id="dry-run",
            )
            allocation = {
                "mode": plan.execution_mode.value,
                "nodes": attempt.allocation_nodes,
                "gpus": attempt.allocation_gpus,
                "scheduler_script": render_slurm_attempt_script(attempt, plan.runner),
            }
        logger.banner("dry-run only; no jobs will be submitted")
        if allocation is None:
            summary = f"{len(submissions)} submission(s)"
        else:
            summary = f"one scheduler allocation, {len(submissions)} logical attempt(s)"
        logger.plan(f"{len(plan.stages)} stage(s), {summary}, root={plan.puzzle_dir}")
        for node in plan.stages:
            count = sum(item.stage_id == node.stage_id for item in submissions)
            unit = "logical attempt" if allocation is not None else "submission"
            logger.stage(
                f"{node.stage_id}: {count} {unit}(s), strategy={node.strategy.value}, "
                f"{node.gpus_per_instance} GPU(s)/instance"
            )
        payload = [
            {
                "stage_id": item.stage_id,
                "work_id": item.work_id,
                "attempt_id": item.attempt_id,
                "resource": item.resource,
                "nodes": item.nodes,
                "gpus": item.gpus,
                "gpus_per_node": item.gpus_per_node,
                "task_count": item.task_count,
                "gpus_per_task": item.gpus_per_task,
                "tasks_per_group": item.tasks_per_group,
                "group_count": item.group_count,
                "task_capacity": item.task_capacity,
                "unused_gpus": item.unused_gpus,
                "launcher": item.launcher,
                "exclusive": item.exclusive,
                "argv": list(item.argv),
                "scheduler_script": (None if allocation is not None else item.scheduler_script),
            }
            for item in submissions
        ]
        output = {"plan": str(plan.puzzle_dir), "submissions": payload}
        if allocation is not None:
            output["allocation"] = allocation
        print(json.dumps(output, indent=2))
        return 0
    allocation_identity = os.environ.get(REUSABLE_PLAN_IDENTITY_ENV)
    if (
        plan.execution_mode is ExecutionMode.REUSABLE_ALLOCATION
        and not allocation_identity
        and not args.local
    ):
        identity = reusable_plan_identity(plan)
        _write_reusable_inputs(args, plan, identity)
        result = run_reusable_allocation(
            plan,
            _reusable_worker_command(args, plan, identity),
            logger=logger,
            poll_interval_seconds=args.poll_interval,
            once=args.once,
        )
    else:
        if allocation_identity:
            identity = reusable_plan_identity(plan)
            if not os.environ.get("SLURM_JOB_ID"):
                logger.error("reusable allocation worker must run inside a Slurm job")
                return 2
            if allocation_identity != identity:
                logger.error("reusable allocation plan identity does not match the compiled plan")
                return 2
        controller = CampaignController(
            plan,
            local=args.local,
            poll_interval_seconds=args.poll_interval,
            logger=logger,
            environment_prepared=allocation_identity is not None,
        )
        result = controller.run(
            overrides=args.override,
            once=args.once,
            max_iterations=args.max_iterations,
        )
    verify_expectation_here = allocation_identity is None and not result.get("detached")
    if args.expect is not None and verify_expectation_here:
        if result.get("report_status") == "completed" or result.get("result_finalized"):
            expectation = verify_expected_results(args.expect, puzzle_dir=plan.puzzle_dir)
        else:
            expectation = ExpectationResult(
                status="skipped",
                exit_code=2,
                comparison_path=None,
                reason="campaign did not reach clean completion",
            )
        result.update(expectation.as_dict())
        if expectation.exit_code:
            result["halted"] = True
            logger.error(
                "campaign expectation verification "
                f"{expectation.status}: {expectation.reason or 'comparison failed'}"
            )
    if allocation_identity is not None:
        CampaignStateStore(plan.puzzle_dir).write_allocation_result(
            plan_identity=allocation_identity,
            result=result,
        )
    failed_stages = list(result.get("failed_stages") or ())
    if failed_stages:
        logger.error(f"failed stage(s): {', '.join(failed_stages)}")
        for stage_id, paths in (result.get("failed_log_paths") or {}).items():
            for path in paths:
                logger.error(f"{stage_id} log: {path}")
    print(json.dumps(result, indent=2))
    expectation_exit_code = result.get("expectation_exit_code")
    if expectation_exit_code is not None:
        return int(expectation_exit_code)
    optional_view_failed_without_result = result.get(
        "report_status"
    ) == "failed" and not result.get("result_finalized")
    return 0 if not result.get("halted") and not optional_view_failed_without_result else 1


if __name__ == "__main__":
    raise SystemExit(main())
