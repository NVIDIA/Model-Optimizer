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

"""End-to-end tests: YAML → schema → resolve → pipeline → sweep.

These tests exercise the full stack from YAML files through to dry-run output,
ensuring all components integrate correctly.
"""

import json
import tempfile

import pytest

from modelopt.torch.recipes import load_recipe
from modelopt.torch.recipes.experiment import SweepConfig, SweepController
from modelopt.torch.recipes.pipeline import load_and_plan

# ── Recipe → resolve → pipeline E2E ──


def test_e2e_all_recipe_yamls_resolve_and_plan(examples_dir):
    """Every recipe YAML loads, resolves, and produces a valid pipeline plan."""
    recipe_files = sorted(p for p in examples_dir.rglob("*.yaml") if "experiments" not in p.parts)
    assert len(recipe_files) >= 8, f"Expected ≥8 recipe YAMLs, found {len(recipe_files)}"

    for path in recipe_files:
        # load_recipe: YAML → parse → resolve
        result = load_recipe(path)
        assert isinstance(result, dict), f"{path.name}: load_recipe returned non-dict"

        # load_and_plan: YAML → parse → resolve → plan
        plan = load_and_plan(str(path))
        assert len(plan.steps) >= 1, f"{path.name}: empty pipeline plan"

        # dry_run produces output
        output = plan.dry_run()
        assert "Pipeline Plan" in output, f"{path.name}: dry_run missing header"
        assert "Step 1" in output, f"{path.name}: dry_run missing step"


def test_e2e_fp8_full_flow(examples_dir):
    """FP8 recipe: load → resolve → plan → dry_run with quant_cfg validation."""
    path = examples_dir / "ptq" / "ptq_fp8.yaml"
    result = load_recipe(path)

    # Validate resolved config structure
    qcfg = result["quantize_config"]["quant_cfg"]
    assert "*weight_quantizer" in qcfg
    assert "*input_quantizer" in qcfg

    # Pipeline plan
    plan = load_and_plan(str(path))
    assert len(plan.steps) == 1
    assert plan.steps[0].technique == "quantization (ptq)"

    # Verbose dry_run includes resolved config
    verbose_output = plan.dry_run(verbose=True)
    assert "Resolved config:" in verbose_output
    assert "weight_quantizer" in verbose_output


def test_e2e_nvfp4_awq_full_flow(examples_dir):
    """NVFP4+AWQ recipe: load → resolve → plan with algorithm and export."""
    path = examples_dir / "ptq" / "ptq_nvfp4_awq.yaml"
    result = load_recipe(path)

    assert "quantize_config" in result
    assert "export" in result

    plan = load_and_plan(str(path))
    assert plan.steps[0].technique == "quantization (ptq)"
    assert plan.export is not None


def test_e2e_sparse_quantize_multi_step(examples_dir):
    """Sparse+quantize recipe produces a 2-step pipeline in correct order."""
    path = examples_dir / "multi_technique" / "sparse_quantize.yaml"
    plan = load_and_plan(str(path))

    assert len(plan.steps) == 2
    assert plan.steps[0].technique == "sparsity"
    assert plan.steps[1].technique == "quantization (ptq)"


def test_e2e_distill_fp8_multi_step(examples_dir):
    """Distill+FP8 recipe produces quantization + distillation steps."""
    path = examples_dir / "multi_technique" / "distill_fp8.yaml"
    plan = load_and_plan(str(path))

    assert len(plan.steps) == 2
    assert plan.steps[0].technique == "quantization (ptq)"
    assert plan.steps[1].technique == "distillation"
    assert "teacher_model" in plan.steps[1].config


# ── Sweep / Experiment E2E ──


def test_e2e_sweep_demo_full_flow(sweep_examples_dir):
    """sweep_demo.yaml: load → validate → generate jobs → export → dry_run."""
    config = SweepConfig.from_yaml(sweep_examples_dir / "sweep_demo.yaml")

    # Validation
    errors = config.validate()
    assert errors == [], f"Validation errors: {errors}"

    # Job count: 2 models x 4 recipes x 2 launchers = 16
    assert config.total_jobs == 16

    # Controller: generate and export
    controller = SweepController(config)
    jobs = controller.generate_jobs()
    assert len(jobs) == 16

    # Each job has required fields
    for job in jobs:
        assert job.model is not None
        assert job.recipe_path is not None
        assert job.launcher is not None

    # Export to JSON
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        output_path = f.name
    controller.export_jobs(output_path)
    with open(output_path) as f:
        data = json.load(f)
    assert len(data) == 16

    # Dry-run output
    output = controller.dry_run()
    assert "Big Pareto Sweep" in output
    assert "16 jobs validated" in output


def test_e2e_experiment_full_flow(sweep_examples_dir):
    """experiment.yaml: load → validate → generate jobs → dry_run with eval."""
    config = SweepConfig.from_yaml(sweep_examples_dir / "experiment.yaml")

    # Validation
    errors = config.validate()
    assert errors == [], f"Validation errors: {errors}"

    # Job count: 2 models x 3 recipes x 2 launchers = 12
    assert config.total_jobs == 12

    # Eval config present
    assert config.eval is not None
    assert config.eval.benchmark_set == "lite"
    assert len(config.eval.tasks) == 4

    # Model overrides
    assert "Qwen/Qwen3-8B" in config.eval.model_overrides

    # Launcher overrides
    assert "lyris_gb200" in config.eval.launcher_overrides

    # Controller
    controller = SweepController(config)
    jobs = controller.generate_jobs()
    assert len(jobs) == 12

    # Dry-run
    output = controller.dry_run()
    assert "12 jobs validated" in output


def test_e2e_experiment_eval_resolves_per_job(sweep_examples_dir):
    """Experiment eval config resolves correctly per model/launcher combo."""
    config = SweepConfig.from_yaml(sweep_examples_dir / "experiment.yaml")

    # Default: 8-way TP, 4 tasks
    result = config.eval.resolve_for_job("meta-llama/Llama-3.1-8B-Instruct", "eos")
    assert result["deployment"]["tensor_parallel_size"] == 8
    assert len(result["evaluation"]["tasks"]) == 4

    # Qwen model override: different tasks
    result = config.eval.resolve_for_job("Qwen/Qwen3-8B", "eos")
    task_names = [t["name"] for t in result["evaluation"]["tasks"]]
    assert "simple_evals.humaneval" in task_names
    assert "simple_evals.livecodebench_v5" in task_names

    # GB200 launcher override: TP=4
    result = config.eval.resolve_for_job("meta-llama/Llama-3.1-8B-Instruct", "lyris_gb200")
    assert result["deployment"]["tensor_parallel_size"] == 4


def test_e2e_sweep_verbose_includes_resolved_config(sweep_examples_dir):
    """Verbose dry-run includes resolved quantize config for each job."""
    config = SweepConfig.from_yaml(sweep_examples_dir / "sweep_demo.yaml")
    controller = SweepController(config)
    output = controller.dry_run(verbose=True)
    assert "_resolved_quantize_config:" in output


def test_invalid_recipe_path():
    with pytest.raises(FileNotFoundError):
        load_recipe("/nonexistent/path.yaml")
