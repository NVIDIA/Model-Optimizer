# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Sweep Controller — generates and validates recipe × model × cluster combinations.

Core orchestration logic for Big Pareto sweeps. Generates SweepJob objects
that can be executed via quant_flow or exported for inspection.
"""

from __future__ import annotations

import itertools
import json
import typing
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

# Bridge is now within the same package
from modelopt.torch.recipes.bridge import recipe_to_hf_ptq_args, summarize_recipe
from modelopt.torch.recipes.utils import make_serializable

if typing.TYPE_CHECKING:
    from .config import SweepConfig


@dataclass
class SweepJob:
    """A single job in a sweep — one (model, recipe, launcher) combination."""

    job_id: int
    model: str
    recipe_path: str
    launcher: str
    ptq_config: dict[str, Any]
    eval_config: dict[str, Any]
    recipe_summary: dict[str, Any]
    resolved_config: dict[str, Any]


class SweepController:
    """Orchestrates Big Pareto sweeps.

    Validates recipes, generates Cartesian product of jobs,
    and produces formatted dry-run output.
    """

    def __init__(self, config: SweepConfig):
        """Initialize controller with a sweep configuration."""
        self.config = config
        self._validated_recipes: dict[str, dict[str, Any]] = {}
        self._raw_yamls: dict[str, dict] = {}
        self._validation_errors: dict[str, Exception] = {}

    def validate_recipes(self) -> list[tuple[str, dict | Exception]]:
        """Phase 1: Validate all recipes via load_recipe().

        Returns list of (recipe_path, resolved_dict_or_exception).
        """
        return [self._validate_single_recipe(p) for p in self.config.recipes]

    def _validate_single_recipe(self, recipe_path: str) -> tuple[str, dict | Exception]:
        """Validate a single recipe, returning (path, result_or_error)."""
        from modelopt.torch.recipes import load_recipe

        try:
            resolved = load_recipe(recipe_path)
            self._validated_recipes[recipe_path] = resolved
            with open(recipe_path) as f:
                self._raw_yamls[recipe_path] = yaml.safe_load(f)
            return (recipe_path, resolved)
        except Exception as e:
            self._validation_errors[recipe_path] = e
            return (recipe_path, e)

    def generate_jobs(self) -> list[SweepJob]:
        """Phase 3: Generate Cartesian product of all sweep dimensions.

        Must call validate_recipes() first (called automatically if not done).
        """
        if not self._validated_recipes and not self._validation_errors:
            self.validate_recipes()

        jobs: list[SweepJob] = []
        job_id = 0
        for model, recipe_path, launcher in itertools.product(
            self.config.models, self.config.recipes, self.config.launchers
        ):
            job_id += 1
            if recipe_path in self._validation_errors:
                continue  # Skip invalid recipes

            resolved = self._validated_recipes[recipe_path]
            raw_yaml = self._raw_yamls[recipe_path]
            hf_ptq = recipe_to_hf_ptq_args(resolved)
            hf_ptq["pyt_ckpt_path"] = model

            jobs.append(
                SweepJob(
                    job_id=job_id,
                    model=model,
                    recipe_path=recipe_path,
                    launcher=launcher,
                    ptq_config={
                        "execution": {"time_limit": self.config.execution.time_limit},
                        "hf_ptq": hf_ptq,
                    },
                    eval_config=self.config.eval.resolve_for_job(model, launcher),
                    recipe_summary=summarize_recipe(recipe_path, resolved, raw_yaml),
                    resolved_config=resolved,
                )
            )
        return jobs

    def dry_run(self, verbose: bool = False) -> str:
        """Run all 4 phases, return formatted output string."""
        lines: list[str] = []

        def section(title: str):
            lines.append("")
            lines.append(title)
            lines.append("─" * 60)

        # Header
        lines.append("=" * 60)
        lines.append("  Big Pareto Sweep — Dry Run")
        lines.append("=" * 60)
        lines.append("")
        lines.append(
            f"Models: {len(self.config.models)}  |  "
            f"Recipes: {len(self.config.recipes)}  |  "
            f"Clusters: {len(self.config.launchers)}  |  "
            f"Eval tasks: {len(self.config.eval.tasks)}"
        )
        lines.append(f"Total jobs: {self.config.total_jobs}")
        if self.config.eval.benchmark_set:
            lines.append(f"Benchmark set: {self.config.eval.benchmark_set}")
        if self.config.eval.model_overrides:
            lines.append(
                f"Model overrides: {len(self.config.eval.model_overrides)} "
                f"({', '.join(m.split('/')[-1] for m in self.config.eval.model_overrides)})"
            )
        if self.config.eval.launcher_overrides:
            lines.append(
                f"Launcher overrides: {len(self.config.eval.launcher_overrides)} "
                f"({', '.join(self.config.eval.launcher_overrides)})"
            )

        # Phase 1: Recipe Validation
        section("Phase 1: Recipe Validation (load_recipe → resolved config)")

        validation_results = self.validate_recipes()
        valid_count = 0
        for i, (path, result) in enumerate(validation_results, 1):
            name = Path(path).name
            if isinstance(result, Exception):
                lines.append(f"[{i}/{len(validation_results)}] {name:40s} FAILED")
                lines.append(f"      Error: {result}")
            else:
                valid_count += 1
                raw = self._raw_yamls[path]
                summary = summarize_recipe(path, result, raw)
                parts = [
                    f"Preset: {summary['preset']}",
                    f"Algorithm: {summary['algorithm']}",
                    f"KV: {summary['kv_cache']}",
                    f"Overrides: {summary['overrides_count']}",
                ]
                lines.append(f"[{i}/{len(validation_results)}] {name:40s} VALID")
                lines.append(f"      {' | '.join(parts)}")

        lines.append("")
        if valid_count == len(validation_results):
            lines.append(f"All {valid_count} recipes validated successfully.")
        else:
            failed = len(validation_results) - valid_count
            lines.append(f"{valid_count} valid, {failed} failed.")

        # Phase 2: Recipe → quant_flow Config Translation
        section("Phase 2: Recipe → quant_flow Config Translation")

        for i, (path, result) in enumerate(validation_results, 1):
            if isinstance(result, Exception):
                continue
            name = Path(path).name
            raw = self._raw_yamls[path]
            hf_ptq = recipe_to_hf_ptq_args(result)

            lines.append(f"[{i}/{len(validation_results)}] {name}")
            lines.append("  Human-friendly (recipe YAML):")
            # Show relevant YAML section compactly
            quant = raw.get("quantization", raw.get("auto_quantize", {}))
            for k, v in quant.items():
                if k == "overrides":
                    lines.append(f"    {k}: [{len(v)} entries]")
                elif isinstance(v, dict):
                    lines.append(f"    {k}:")
                    for kk, vv in v.items():
                        lines.append(f"      {kk}: {vv}")
                else:
                    lines.append(f"    {k}: {v}")

            lines.append("")
            lines.append("  Machine-friendly (PTQConfig.hf_ptq):")
            for k, v in sorted(hf_ptq.items()):
                if k == "_resolved_quantize_config":
                    if verbose:
                        lines.append(f"    {k}:")
                        _format_dict(v, lines, indent=6)
                    else:
                        lines.append(f"    {k}: <{len(v)} keys — use --verbose>")
                else:
                    lines.append(f"    {k}: {v}")
            lines.append("")

        # Phase 3: Pipeline Generation
        section("Phase 3: Pipeline Generation (PTQConfig + EvalConfig per job)")

        jobs = self.generate_jobs()
        for job in jobs:
            recipe_name = Path(job.recipe_path).stem
            model_short = job.model.split("/")[-1]
            lines.append(
                f"Job {job.job_id:2d}/{self.config.total_jobs}: "
                f"{model_short} x {recipe_name} x {job.launcher}"
            )
            # Show PTQ config summary (exclude resolved config)
            ptq_summary = {
                k: v
                for k, v in job.ptq_config["hf_ptq"].items()
                if k != "_resolved_quantize_config"
            }
            lines.append(f"  PTQ:  {ptq_summary}")
            eval_tasks = [t["name"] for t in job.eval_config["evaluation"]["tasks"]]
            eval_line = (
                f"  Eval: engine={job.eval_config['deployment']['engine']}, "
                f"tp={job.eval_config['deployment']['tensor_parallel_size']}, "
                f"tasks={eval_tasks}"
            )
            # Indicate when overrides are active
            overrides_active = []
            if job.model in self.config.eval.model_overrides:
                overrides_active.append("model")
            if job.launcher in self.config.eval.launcher_overrides:
                overrides_active.append("launcher")
            if overrides_active:
                eval_line += f"  [{'+'.join(overrides_active)} override]"
            lines.append(eval_line)
            lines.append(
                f"  CLI:  quant_flow --ptq_config <generated> "
                f"--eval_config <generated> --launcher {job.launcher} --dryrun"
            )

        # Phase 4: Summary
        section("Phase 4: Summary")

        # Build table
        launchers = self.config.launchers
        header = f"{'Recipe':<30s} {'Model':<22s} " + " ".join(f"{lnch:<14s}" for lnch in launchers)
        sep = "─" * len(header)
        lines.append(header)
        lines.append(sep)

        job_map: dict[tuple[str, str, str], int] = {
            (j.recipe_path, j.model, j.launcher): j.job_id for j in jobs
        }
        for recipe in self.config.recipes:
            recipe_name = Path(recipe).stem
            for model in self.config.models:
                model_short = model.split("/")[-1]
                cells = []
                for launcher in launchers:
                    jid = job_map.get((recipe, model, launcher))
                    cells.append(f"Job {jid}" if jid else "—")
                lines.append(
                    f"{recipe_name:<30s} {model_short:<22s} " + " ".join(f"{c:<14s}" for c in cells)
                )

        lines.append("")
        lines.append(f"{len(jobs)} jobs validated. Dry run complete.")
        lines.append("To submit: remove --dry-run flag (requires GITLAB_API_TOKEN).")

        return "\n".join(lines)

    def export_jobs(self, output_path: str) -> None:
        """Write generated jobs to JSON file."""
        jobs = self.generate_jobs()
        serializable = []
        for job in jobs:
            d = {
                "job_id": job.job_id,
                "model": job.model,
                "recipe": job.recipe_path,
                "launcher": job.launcher,
                "ptq_config": make_serializable(job.ptq_config),
                "eval_config": job.eval_config,
                "recipe_summary": job.recipe_summary,
            }
            serializable.append(d)

        with open(output_path, "w") as f:
            json.dump(serializable, f, indent=2, default=str)


def _format_dict(d: dict, lines: list[str], indent: int = 4) -> None:
    """Format a dict for display, truncating long values."""
    prefix = " " * indent
    for k, v in sorted(d.items(), key=lambda x: str(x[0])):
        if isinstance(v, dict) and len(str(v)) > 80:
            lines.append(f"{prefix}{k}:")
            _format_dict(v, lines, indent + 2)
        else:
            s = str(v)
            if len(s) > 100:
                s = s[:97] + "..."
            lines.append(f"{prefix}{k}: {s}")
