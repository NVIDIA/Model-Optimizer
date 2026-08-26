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
import sys
from pathlib import Path

import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from puzzletron_orchestrator.compiler import (  # noqa: E402
    compile_campaign_plan,
    load_execution_config,
    load_runner_config,
)
from puzzletron_orchestrator.controller import CampaignController, dry_run_plan  # noqa: E402
from puzzletron_orchestrator.logging import OrchestratorLogger  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Puzzletron v2 campaign orchestrator.")
    parser.add_argument("--experiment", required=True, help="Path to the experiment YAML.")
    parser.add_argument("--runner", required=True, help="Path to the runner environment YAML.")
    parser.add_argument("--execution", required=True, help="Path to the execution semantics YAML.")
    parser.add_argument(
        "--stage", default="full", help="Stage id or 'full' for all enabled stages."
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Repeatable config override; KEY=VALUE and ++KEY=VALUE are supported.",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print packed submissions without submitting."
    )
    parser.add_argument("--local", action="store_true", help="Use the local subprocess executor.")
    parser.add_argument("--once", action="store_true", help="Run one controller iteration.")
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
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logger = OrchestratorLogger(color=args.color)
    try:
        runner = load_runner_config(args.runner)
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
        logger.banner("dry-run only; no jobs will be submitted")
        logger.plan(
            f"{len(plan.stages)} stage(s), {len(submissions)} submission(s), root={plan.puzzle_dir}"
        )
        for node in plan.stages:
            count = sum(item.stage_id == node.stage_id for item in submissions)
            logger.stage(
                f"{node.stage_id}: {count} submission(s), strategy={node.strategy.value}, "
                f"{node.gpus_per_instance} GPU(s)/instance"
            )
        payload = [
            {
                "stage_id": item.stage_id,
                "work_id": item.work_id,
                "attempt_id": item.attempt_id,
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
            }
            for item in submissions
        ]
        print(json.dumps({"plan": str(plan.puzzle_dir), "submissions": payload}, indent=2))
        return 0
    controller = CampaignController(
        plan,
        local=args.local,
        poll_interval_seconds=args.poll_interval,
        logger=logger,
    )
    result = controller.run(
        overrides=args.override,
        once=args.once,
        max_iterations=args.max_iterations,
    )
    failed_stages = list(result.get("failed_stages") or ())
    if failed_stages:
        logger.error(f"failed stage(s): {', '.join(failed_stages)}")
        for stage_id, paths in (result.get("failed_log_paths") or {}).items():
            for path in paths:
                logger.error(f"{stage_id} log: {path}")
    print(json.dumps(result, indent=2))
    return 0 if not result.get("halted") else 1


if __name__ == "__main__":
    raise SystemExit(main())
