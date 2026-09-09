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

"""One public command for Puzzletron configuration and campaign lifecycle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from examples.puzzletron import orchestrate  # noqa: E402
from puzzletron_orchestrator.recipe_config import (  # noqa: E402
    ROUTES,
    bundle_for_run_root,
    explain_resolved_run,
    materialize_resolved_bundle,
    resolve_recipe_run,
)


def _add_recipe_inputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("recipe", type=Path, help="Concise Puzzletron recipe YAML.")
    parser.add_argument("--site", type=Path, required=True, help="Reusable site YAML.")
    parser.add_argument(
        "--run-root",
        type=Path,
        help="Override recipe.run_root, normally to create a distinct immutable run.",
    )


def _add_run_options(parser: argparse.ArgumentParser, *, dry_run: bool = False) -> None:
    if not dry_run:
        parser.add_argument(
            "--once",
            action="store_true",
            help="Recover, poll, and submit ready work once, then exit.",
        )
        parser.add_argument(
            "--max-iterations",
            type=int,
            help="Stop the controller after this many polling iterations.",
        )
        parser.add_argument(
            "--expect",
            type=Path,
            help="Verify completed artifacts against this expectation contract.",
        )
    parser.add_argument(
        "--color",
        choices=("auto", "always", "never"),
        default="auto",
        help="Colorize progress logs (default: auto).",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=5.0,
        help="Seconds between scheduler polls (default: 5).",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate, explain, run, resume, and inspect Puzzletron from recipe + site."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    routes = commands.add_parser(
        "routes", help="List maintained model, workflow, mode, and intent choices."
    )
    routes.add_argument("--json", action="store_true", help="Print one JSON document.")

    commands.add_parser(
        "setup",
        help="Set up a custom model when no maintained recipe fits.",
    )

    validate = commands.add_parser("validate", help="Validate without writing run artifacts.")
    _add_recipe_inputs(validate)

    explain = commands.add_parser(
        "explain", help="Show winning sources and the resolved stage mesh."
    )
    _add_recipe_inputs(explain)
    explain.add_argument("--json", action="store_true", help="Print machine-readable output.")

    dry_run = commands.add_parser("dry-run", help="Seal a preview bundle and render submissions.")
    _add_recipe_inputs(dry_run)
    _add_run_options(dry_run, dry_run=True)

    launch = commands.add_parser("launch", help="Seal and launch one immutable run bundle.")
    _add_recipe_inputs(launch)
    _add_run_options(launch)

    resume = commands.add_parser("resume", help="Resume the bundle already bound to a run root.")
    resume.add_argument("run_root", type=Path, help="Run root containing the active sealed bundle.")
    _add_run_options(resume)

    inspect = commands.add_parser("inspect", help="Inspect a run's sealed inputs and plan.")
    inspect.add_argument(
        "run_root", type=Path, help="Run root containing the active sealed bundle."
    )
    inspect.add_argument("--json", action="store_true", help="Print one JSON document.")

    return parser


def _orchestrator_args(bundle: Path, args: argparse.Namespace, *, dry_run: bool) -> list[str]:
    command = [
        "--experiment",
        str(bundle / "experiment.runtime.yaml"),
        "--runner",
        str(bundle / "runner.yaml"),
        "--execution",
        str(bundle / "execution.yaml"),
        "--stage",
        "full",
        "--color",
        args.color,
        "--poll-interval",
        str(args.poll_interval),
    ]
    if dry_run:
        command.append("--dry-run")
    if not dry_run:
        if args.once:
            command.append("--once")
        if args.max_iterations is not None:
            command.extend(("--max-iterations", str(args.max_iterations)))
        if args.expect is not None:
            command.extend(("--expect", str(args.expect)))
    return command


def _inspect(bundle: Path, *, as_json: bool) -> None:
    names = ("manifest.json", "plan.json", "provenance.json")
    payload = {
        name.removesuffix(".json"): json.loads((bundle / name).read_text()) for name in names
    }
    if as_json:
        print(json.dumps({"bundle": str(bundle), **payload}, indent=2))
        return
    manifest = payload["manifest"]
    plan = payload["plan"]
    print(f"bundle: {manifest['bundle_id']}")
    print(f"path: {bundle}")
    print("runtime input: experiment.runtime.yaml (sealed; do not edit)")
    print("audit view: experiment.resolved.yaml (resolved output; do not edit)")
    print(f"controller revision: {manifest['code']['controller']['revision']}")
    print(f"worker revision: {manifest['code']['worker']['revision']}")
    print(f"execution contract: {plan['contract_hash']}")
    print(f"stages: {len(plan['stages'])}")
    for stage in plan["stages"]:
        print(
            f"  {stage['stage_id']}: {stage['nodes']} node(s), "
            f"{stage['total_gpus']} GPU(s), mesh={stage['mesh']}"
        )


def main(argv: list[str] | None = None) -> int:
    command_line = list(sys.argv[1:] if argv is None else argv)
    if command_line[:1] == ["setup"]:
        from puzzletron_setup.v2.cli import main as setup_main

        return setup_main(command_line[1:])
    args = _build_parser().parse_args(command_line)
    try:
        if args.command == "routes":
            rows = [
                {
                    "model": route.model,
                    "workflow": route.workflow,
                    "mode": route.mode,
                    "intent": {
                        "search": route.search,
                        "evaluation": route.evaluation,
                        "distillation": route.distillation,
                    },
                }
                for route in ROUTES
            ]
            if args.json:
                print(json.dumps({"routes": rows}, indent=2))
            else:
                for row in rows:
                    intent = ", ".join(f"{key}={value}" for key, value in row["intent"].items())
                    print(f"{row['model']}  {row['workflow']}  {row['mode']}  {intent}")
            return 0
        if args.command == "resume":
            bundle = bundle_for_run_root(args.run_root)
            return orchestrate.main(_orchestrator_args(bundle, args, dry_run=False))
        if args.command == "inspect":
            _inspect(bundle_for_run_root(args.run_root), as_json=args.json)
            return 0

        resolved = resolve_recipe_run(args.recipe, args.site, run_root=args.run_root)
        if args.command == "validate":
            print(
                f"valid: {resolved.recipe.name} -> {resolved.route.route_id} "
                f"({len(resolved.plan['stages'])} stages, {resolved.bundle_id})"
            )
            return 0
        if args.command == "explain":
            if args.json:
                print(
                    json.dumps(
                        {
                            "bundle_id": resolved.bundle_id,
                            "plan": resolved.plan,
                            "provenance": resolved.provenance,
                        },
                        indent=2,
                        default=str,
                    )
                )
            else:
                print(explain_resolved_run(resolved))
            return 0
        if args.command == "dry-run":
            bundle = materialize_resolved_bundle(resolved, activate=False)
            return orchestrate.main(_orchestrator_args(bundle, args, dry_run=True))
        if args.command == "launch":
            bundle = materialize_resolved_bundle(resolved, activate=True)
            return orchestrate.main(_orchestrator_args(bundle, args, dry_run=False))
        raise AssertionError(f"Unhandled command: {args.command}")
    except (
        KeyError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        yaml.YAMLError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
