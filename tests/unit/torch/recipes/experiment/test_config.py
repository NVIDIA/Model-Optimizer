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

"""Tests for sweep configuration (experiment/config.py)."""

from modelopt.torch.recipes.experiment import SweepConfig
from modelopt.torch.recipes.experiment.config import SweepEvalConfig


def test_load_sweep_config(sweep_examples_dir):
    config = SweepConfig.from_yaml(sweep_examples_dir / "sweep_demo.yaml")
    assert len(config.models) == 2
    assert len(config.recipes) == 4
    assert len(config.launchers) == 2
    assert config.total_jobs == 16
    assert len(config.eval.tasks) == 2


def test_validate_valid_config(sweep_examples_dir):
    config = SweepConfig.from_yaml(sweep_examples_dir / "sweep_demo.yaml")
    errors = config.validate()
    assert errors == []


def test_validate_empty_config():
    config = SweepConfig()
    errors = config.validate()
    assert len(errors) == 4  # models, recipes, launchers, tasks all missing
    assert "No models specified" in errors
    assert "No recipes specified" in errors
    assert "No launchers specified" in errors
    assert "No eval tasks specified" in errors


def test_total_jobs():
    config = SweepConfig(
        models=["a", "b", "c"],
        recipes=["r1", "r2"],
        launchers=["l1"],
    )
    assert config.total_jobs == 6


def test_load_experiment_config(sweep_examples_dir):
    """Test new experiment.yaml format with evaluation section."""
    config = SweepConfig.from_yaml(sweep_examples_dir / "experiment.yaml")
    assert len(config.models) == 2
    assert len(config.recipes) == 3
    assert len(config.launchers) == 2
    assert config.total_jobs == 12
    assert len(config.eval.tasks) == 4
    assert config.eval.benchmark_set == "lite"
    assert "Qwen/Qwen3-8B" in config.eval.model_overrides
    assert "lyris_gb200" in config.eval.launcher_overrides


def test_eval_overrides_resolve():
    """Test that per-model and per-launcher overrides are applied correctly."""
    eval_cfg = SweepEvalConfig(
        engine="vllm",
        tensor_parallel_size=8,
        tasks=["mmlu", "gpqa"],
        model_overrides={
            "code-model": {"tasks": ["humaneval", "livecodebench"]},
        },
        launcher_overrides={
            "gb200": {"tensor_parallel_size": 4},
        },
    )

    # No overrides
    result = eval_cfg.resolve_for_job("llama-8b", "eos")
    assert result["deployment"]["tensor_parallel_size"] == 8
    assert len(result["evaluation"]["tasks"]) == 2
    assert result["evaluation"]["tasks"][0]["name"] == "mmlu"

    # Model override: different tasks
    result = eval_cfg.resolve_for_job("code-model", "eos")
    assert len(result["evaluation"]["tasks"]) == 2
    assert result["evaluation"]["tasks"][0]["name"] == "humaneval"

    # Launcher override: different TP
    result = eval_cfg.resolve_for_job("llama-8b", "gb200")
    assert result["deployment"]["tensor_parallel_size"] == 4

    # Both overrides
    result = eval_cfg.resolve_for_job("code-model", "gb200")
    assert result["deployment"]["tensor_parallel_size"] == 4
    assert result["evaluation"]["tasks"][0]["name"] == "humaneval"


def test_eval_resolve_with_engine_overrides():
    """Model and launcher overrides can change engine and tensor_parallel_size."""
    eval_cfg = SweepEvalConfig(
        engine="vllm",
        tensor_parallel_size=8,
        tasks=["mmlu", "gsm8k"],
        model_overrides={
            "code-model": {"tasks": ["humaneval"], "engine": "trtllm", "tensor_parallel_size": 4}
        },
        launcher_overrides={"gb200": {"tensor_parallel_size": 2, "engine": "sglang"}},
    )

    # Model override changes engine and TP
    result = eval_cfg.resolve_for_job("code-model", "h100")
    assert result["deployment"]["engine"] == "trtllm"
    assert result["deployment"]["tensor_parallel_size"] == 4
    assert result["evaluation"]["tasks"] == [{"name": "humaneval"}]

    # Launcher override takes precedence for TP and engine
    result = eval_cfg.resolve_for_job("generic-model", "gb200")
    assert result["deployment"]["tensor_parallel_size"] == 2
    assert result["deployment"]["engine"] == "sglang"


def test_eval_benchmark_set():
    """Benchmark set appears in resolved config."""
    eval_cfg = SweepEvalConfig(tasks=["mmlu"], benchmark_set="standard_v2")
    result = eval_cfg.resolve_for_job("model", "launcher")
    assert result["evaluation"]["benchmark_set"] == "standard_v2"
