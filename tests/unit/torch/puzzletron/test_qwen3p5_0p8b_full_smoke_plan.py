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

"""CPU plan contracts for the complete Qwen 3.5 0.8B smoke journey."""

from itertools import pairwise
from pathlib import Path

import yaml

from puzzletron_orchestrator.compiler import (
    compile_campaign_plan,
    load_execution_config,
    load_runner_config,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
FAMILY_ROOT = REPOSITORY_ROOT / "examples/puzzletron/configs/families/qwen3_5"
FAMILY_PRESETS_PATH = FAMILY_ROOT / "setup_v2_defaults.yaml"
EXPLICIT_SETUP_DEFAULTS_PATH = (
    REPOSITORY_ROOT / "tests/_test_utils/torch/puzzletron/tiny_qwen_setup_defaults.yaml"
)
RUN_PATH = FAMILY_ROOT / "qwen3p5_0p8b/runs/full_smoke.yaml"
ORCHESTRATION_ROOT = REPOSITORY_ROOT / "examples/puzzletron/configs/orchestration/qwen3p5_0p8b"
RUNNER_PATH = ORCHESTRATION_ROOT / "runner.slurm.yaml"
EXECUTION_PATH = ORCHESTRATION_ROOT / "execution.full_smoke.yaml"
QUALITY_RUN_PATH = FAMILY_ROOT / "qwen3p5_0p8b/runs/quality_regression.yaml"
QUALITY_EXECUTION_PATH = ORCHESTRATION_ROOT / "execution.quality_regression.yaml"


def _compile_plan(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("PUZZLETRON_RUN_ROOT", str(tmp_path / "results"))
    monkeypatch.setenv("PUZZLETRON_DATASET_PATH", str(tmp_path / "dataset"))
    return compile_campaign_plan(
        experiment_config_path=RUN_PATH,
        runner=load_runner_config(RUNNER_PATH),
        execution=load_execution_config(EXECUTION_PATH),
        stage_filter="full",
    )


def _compile_quality_plan(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("PUZZLETRON_RUN_ROOT", str(tmp_path / "quality-results"))
    monkeypatch.setenv("PUZZLETRON_DATASET_PATH", str(tmp_path / "dataset"))
    return compile_campaign_plan(
        experiment_config_path=QUALITY_RUN_PATH,
        runner=load_runner_config(RUNNER_PATH),
        execution=load_execution_config(QUALITY_EXECUTION_PATH),
        stage_filter="full",
    )


def test_qwen3p5_0p8b_setup_contracts_remain_distinct() -> None:
    family_presets = yaml.safe_load(FAMILY_PRESETS_PATH.read_text())
    explicit_defaults = yaml.safe_load(EXPLICIT_SETUP_DEFAULTS_PATH.read_text())

    assert family_presets["schema_version"] == 2
    assert explicit_defaults["schema_version"] == 1


def test_qwen3p5_0p8b_full_smoke_declares_its_hydra_inheritance(
    monkeypatch,
    tmp_path: Path,
) -> None:
    run_config = yaml.safe_load(RUN_PATH.read_text())
    config = _compile_plan(monkeypatch, tmp_path).experiment_config

    assert run_config["defaults"] == ["mip_smoke", "_self_"]
    assert config["model"]["revision"] == "2fc06364715b967f1860aea9cf38778875588b17"
    assert config["mip"]["runs"]["params-90"]["search_space"] == {
        "depth": [0],
        "embedding": [1024],
        "axes_default": "teacher",
        "axes": {"ffn.intermediate_size": "all"},
    }
    assert config["sort"]["deferred_axes"] == [
        "kv_groups",
        "q_heads_per_group",
        "gdn_key_groups",
        "gdn_value_heads_per_group",
        "gdn_key_head_dim",
        "gdn_value_head_dim",
    ]
    assert config["post_mip"]["flows"]["params-90"]["source"] == {
        "run": "params-90",
        "variants": "all",
        "objectives": "all",
    }


def test_qwen3p5_0p8b_full_smoke_compiles_the_complete_one_gpu_route(
    monkeypatch,
    tmp_path: Path,
) -> None:
    plan = _compile_plan(monkeypatch, tmp_path)
    stage_ids = tuple(stage.stage_id for stage in plan.stages)
    post_stage_ids = tuple(stage_id for stage_id in stage_ids if stage_id.startswith("post."))

    assert post_stage_ids == (
        "post.params-90.online_eval",
        "post.params-90.best_lm",
        "post.params-90.materialized",
        "post.params-90.checkpoint_eval",
        "post.params-90.serving",
        "post.params-90.fastest",
        "post.params-90.short_kd",
        "post.params-90.post_kd_checkpoint_eval",
        "post.params-90.final_eval",
        "post.params-90.best",
    )
    post_nodes = tuple(stage for stage in plan.stages if stage.stage_id.startswith("post."))
    assert post_nodes[0].parents == ("mip",)
    for parent, node in pairwise(post_nodes):
        assert node.parents == (parent.stage_id,)
    assert stage_ids[-1] == "post.params-90.best"
    assert all(stage.total_gpus == 1 for stage in plan.stages)


def test_qwen3p5_0p8b_full_smoke_keeps_runtime_budgets_bounded(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config = _compile_plan(monkeypatch, tmp_path).experiment_config
    nodes = config["post_mip"]["flows"]["params-90"]["nodes"]

    assert nodes["online_eval"]["config"] == {"eval_samples": 2, "block_size": 512}
    assert config["mip"]["runs"]["params-90"]["solver"]["num_solutions"] == 1
    assert config["mip"]["runs"]["params-90"]["homogeneous"]["keep"] == 5
    assert nodes["best_lm"]["top_k"] == 2
    assert nodes["checkpoint_eval"]["type"] == "downstream_evaluation"
    assert nodes["checkpoint_eval"]["failure_policy"] == "strict"
    assert nodes["checkpoint_eval"]["config"]["tasks"] == ["ifeval"]
    assert nodes["checkpoint_eval"]["config"]["limit"] == 2
    assert nodes["checkpoint_eval"]["config"]["batch_size"] == 1
    assert nodes["checkpoint_eval"]["config"]["timeout_seconds"] == 600
    assert nodes["serving"]["config"]["readiness_timeout"] == 300
    assert nodes["serving"]["config"]["benchmark_timeout"] == 300
    assert nodes["serving"]["config"]["request_count"] == 4
    assert nodes["serving"]["config"]["concurrency"] == [1]
    assert nodes["fastest"]["top_k"] == 1
    assert nodes["short_kd"]["config"] == {
        "max_steps": 2,
        "global_batch_size": 1,
        "local_batch_size": 1,
        "checkpoint_every_steps": 2,
    }
    assert nodes["post_kd_checkpoint_eval"]["type"] == "downstream_evaluation"
    assert nodes["post_kd_checkpoint_eval"]["failure_policy"] == "strict"
    assert nodes["post_kd_checkpoint_eval"]["config"] == nodes["checkpoint_eval"]["config"]
    assert nodes["final_eval"]["config"] == {"eval_samples": 2, "block_size": 512}
    assert nodes["best"]["top_k"] == 1


def test_qwen3p5_0p8b_quality_regression_is_opt_in_and_compares_teacher(
    monkeypatch,
    tmp_path: Path,
) -> None:
    plan = _compile_quality_plan(monkeypatch, tmp_path)
    config = plan.experiment_config
    nodes = config["post_mip"]["flows"]["params-90"]["nodes"]
    benchmark = nodes["full_benchmarks"]
    stage_ids = tuple(stage.stage_id for stage in plan.stages)

    assert "post.params-90.full_benchmarks" in stage_ids
    assert stage_ids[-2:] == (
        "post.params-90.ifeval_quality_gate",
        "post.params-90.gsm8k_quality_gate",
    )
    assert benchmark["input"] == "short_kd"
    assert benchmark["failure_policy"] == "strict"
    assert benchmark["config"]["reference_checkpoint"] == config["teacher_dir"]
    assert benchmark["config"]["tasks"] == ["ifeval", "gsm8k"]
    assert benchmark["config"]["compatibility_tasks"] == ["gsm8k"]
    assert benchmark["config"]["batch_size"] == 8
    assert "limit" not in benchmark["config"]
    assert benchmark["config"]["gen_kwargs"] == {
        "do_sample": False,
        "temperature": 0,
    }
    assert nodes["ifeval_quality_gate"]["require_match"] is True
    assert nodes["ifeval_quality_gate"]["min"] == -0.26
    assert nodes["gsm8k_quality_gate"]["metric"].endswith(
        "modelopt_gsm8k.exact_match_flexible-extract"
    )
    assert nodes["gsm8k_quality_gate"]["min"] == -0.50
    assert nodes["gsm8k_quality_gate"]["require_match"] is True
    assert all(stage.total_gpus == 1 for stage in plan.stages)
