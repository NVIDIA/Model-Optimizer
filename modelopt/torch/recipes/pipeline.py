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

"""Pipeline orchestrator for multi-technique recipes.

Given a RecipeConfig, the pipeline planner:
1. Determines execution order (pruning → sparsity → quantization → distillation)
2. Resolves each technique's config to internal API format
3. Produces a PipelinePlan — an ordered list of steps with their configs

This module does NOT execute anything. It produces a plan that can be:
- Inspected (dry-run): see exactly what each ModelOpt API call would receive
- Executed: by an external runner that calls the actual APIs
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .schema.models import RecipeConfig
from .utils import make_serializable

# Technique execution order — earlier techniques run first.
# This matches ModelOpt's recommended pipeline:
#   pruning (restructure) → sparsity (zero patterns) → quantization (compress) → distillation (recover)
TECHNIQUE_ORDER = ["pruning", "sparsity", "quantization", "distillation"]

# Maps technique names to ModelOpt API calls
TECHNIQUE_API = {
    "pruning": "modelopt.torch.prune.prune(model, mode, constraints, dummy_input, config)",
    "sparsity": "modelopt.torch.sparsity.sparsify(model, config)",
    "quantization_ptq": "modelopt.torch.quantization.quantize(model, config, forward_loop)",
    "quantization_qat": "modelopt.torch.quantization.quantize(model, config, forward_loop) → train()",
    "auto_quantize": "modelopt.torch.quantization.auto_quantize(model, **kwargs, data_loader=...)",
    "distillation": "modelopt.torch.distill.convert(model, mode='kd_loss', config=KDLossConfig(...))",
}


@dataclass
class PipelineStep:
    """A single step in the execution pipeline."""

    technique: str  # sparsity, quantization, distillation, auto_quantize
    api_call: str  # ModelOpt API signature
    config: dict[str, Any]  # resolved config dict for this API
    calibration: dict[str, Any] | None = None
    training: dict[str, Any] | None = None


@dataclass
class PipelinePlan:
    """Ordered execution plan for a recipe."""

    recipe_path: str
    steps: list[PipelineStep] = field(default_factory=list)
    model: dict[str, Any] | None = None
    export: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None

    def dry_run(self, verbose: bool = False) -> str:
        """Format the plan as a human-readable dry-run report."""
        lines: list[str] = []
        lines.append("=" * 70)
        lines.append(f"Pipeline Plan: {self.recipe_path}")
        lines.append("=" * 70)

        if self.model:
            lines.append(f"  Model: {self.model.get('path', '?')}")
            if self.model.get("trust_remote_code"):
                lines.append("  trust_remote_code: True")
            if self.model.get("attn_implementation"):
                lines.append(f"  attn_implementation: {self.model['attn_implementation']}")

        if self.metadata:
            name = self.metadata.get("name", "")
            desc = self.metadata.get("description", "")
            if name:
                lines.append(f"  Name: {name}")
            if desc:
                lines.append(f"  Description: {desc}")
            lines.append("")

        lines.append(f"  Steps: {len(self.steps)}")
        lines.append(f"  Execution order: {' → '.join(s.technique for s in self.steps)}")
        lines.append("")

        for i, step in enumerate(self.steps, 1):
            lines.append(f"--- Step {i}: {step.technique} ---")
            lines.append(f"  API: {step.api_call}")
            lines.append("")

            if step.calibration:
                ds = step.calibration.get("dataset", "?")
                ns = step.calibration.get("num_samples", "?")
                lines.append(f"  Calibration: dataset={ds}, num_samples={ns}")

            if step.training:
                lr = step.training.get("learning_rate", "?")
                ep = step.training.get("num_epochs", "?")
                ms = step.training.get("max_steps")
                lines.append(
                    f"  Training: lr={lr}, epochs={ep}" + (f", max_steps={ms}" if ms else "")
                )

            if verbose:
                lines.append("")
                lines.append("  Resolved config:")
                config_yaml = yaml.dump(
                    make_serializable(step.config), default_flow_style=False, sort_keys=False
                )
                lines.extend(f"    {line}" for line in config_yaml.strip().split("\n"))

            lines.append("")

        if self.export:
            lines.append("--- Export ---")
            lines.append(f"  Format: {self.export.get('format', '?')}")
            tp = self.export.get("tensor_parallel", 1)
            if tp > 1:
                lines.append(f"  Tensor parallel: {tp}")
            lines.append(f"  Output: {self.export.get('output_dir', './output')}")
            lines.append("")

        lines.append("=" * 70)
        return "\n".join(lines)


def plan_pipeline(recipe: RecipeConfig, recipe_path: str = "<inline>") -> PipelinePlan:
    """Create an execution plan from a RecipeConfig.

    Determines technique ordering, resolves each technique's config to the
    internal format that ModelOpt APIs accept, and returns a PipelinePlan.
    """
    from .schema.resolver import resolve_recipe

    plan = PipelinePlan(recipe_path=recipe_path)

    if recipe.model:
        plan.model = recipe.model.model_dump(exclude_none=True)

    if recipe.metadata:
        plan.metadata = recipe.metadata.model_dump(exclude_none=True)

    # Resolve the full recipe (quantization + auto_quantize)
    resolved = resolve_recipe(recipe)

    # Build steps in execution order: pruning → sparsity → quantization → distillation
    if recipe.pruning:
        plan.steps.append(_plan_pruning(recipe.pruning))

    if recipe.sparsity:
        plan.steps.append(_plan_sparsity(recipe.sparsity))

    if recipe.quantization:
        plan.steps.append(_plan_quantization(recipe.quantization, resolved))
    elif recipe.auto_quantize:
        plan.steps.append(_plan_auto_quantize(recipe.auto_quantize, resolved))

    if recipe.distillation:
        plan.steps.append(_plan_distillation(recipe.distillation))

    if recipe.export:
        plan.export = recipe.export.model_dump(exclude_none=True)

    return plan


def _section_to_config(
    section,
) -> tuple[dict[str, Any], dict[str, Any] | None, dict[str, Any] | None]:
    """Extract config, calibration, and training from a section via model_dump.

    Preserves all fields including extra technique-specific options.
    """
    raw = section.model_dump(exclude_none=True)
    calib = raw.pop("calibration", None)
    training = raw.pop("training", None)
    return raw, calib, training


def _plan_pruning(section) -> PipelineStep:
    """Resolve pruning section to a pipeline step."""
    config, calib, training = _section_to_config(section)

    return PipelineStep(
        technique="pruning",
        api_call=TECHNIQUE_API["pruning"],
        config=config,
        calibration=calib,
        training=training,
    )


def _plan_sparsity(section) -> PipelineStep:
    """Resolve sparsity section to a pipeline step."""
    config, calib, _ = _section_to_config(section)

    return PipelineStep(
        technique="sparsity",
        api_call=TECHNIQUE_API["sparsity"],
        config=config,
        calibration=calib,
    )


def _plan_quantization(section, resolved: dict) -> PipelineStep:
    """Resolve quantization section to a pipeline step."""
    mode = section.mode  # "ptq" or "qat"
    api_key = f"quantization_{mode}"

    config = resolved.get("quantize_config", {})

    calib = None
    if section.calibration:
        calib = section.calibration.model_dump()

    training = None
    if section.training:
        training = section.training.model_dump()

    return PipelineStep(
        technique=f"quantization ({mode})",
        api_call=TECHNIQUE_API[api_key],
        config=config,
        calibration=calib,
        training=training,
    )


def _plan_auto_quantize(section, resolved: dict) -> PipelineStep:
    """Resolve auto_quantize section to a pipeline step."""
    config = resolved.get("auto_quantize_kwargs", {})

    calib = None
    if section.calibration:
        calib = section.calibration.model_dump()

    return PipelineStep(
        technique="auto_quantize",
        api_call=TECHNIQUE_API["auto_quantize"],
        config=config,
        calibration=calib,
    )


def _plan_distillation(section) -> PipelineStep:
    """Resolve distillation section to a pipeline step.

    Reshapes section fields into the distill.convert() API format while
    preserving any extra technique-specific options.
    """
    raw, calib, training = _section_to_config(section)

    # Reshape to distill API format
    config: dict[str, Any] = {
        "teacher_model": raw.pop("teacher"),
        "criterion": raw.pop("criterion", "kl_div"),
        "loss_balancer": {
            "type": "StaticLossBalancer",
            "kd_loss_weight": raw.pop("kd_loss_weight", 0.5),
        },
    }

    layer_pairs = raw.pop("layer_pairs", None)
    if layer_pairs:
        config["layer_pairs"] = layer_pairs

    # Preserve any extra fields from the section
    config.update(raw)

    return PipelineStep(
        technique="distillation",
        api_call=TECHNIQUE_API["distillation"],
        config=config,
        calibration=calib,
        training=training,
    )


def load_and_plan(path: str | Path) -> PipelinePlan:
    """Load a recipe YAML and produce a pipeline plan.

    This is the main entry point for end-to-end dry-run:
        YAML file → load → validate → resolve → plan
    """
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    recipe = RecipeConfig.model_validate(raw)
    return plan_pipeline(recipe, recipe_path=str(path))
