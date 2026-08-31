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

"""CPU plan contracts for the complete Qwen 3.5 0.8B VLM smoke."""

from itertools import pairwise
from pathlib import Path

import yaml

from puzzletron_orchestrator.compiler import (
    compile_campaign_plan,
    load_execution_config,
    load_runner_config,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
RUN_PATH = (
    REPOSITORY_ROOT
    / "examples/puzzletron/configs/families/qwen3_5/qwen3p5_0p8b/runs/full_vlm_smoke.yaml"
)
ORCHESTRATION_ROOT = REPOSITORY_ROOT / "examples/puzzletron/configs/orchestration/qwen3p5_0p8b"
RUNNER_PATH = ORCHESTRATION_ROOT / "runner.slurm.yaml"
EXECUTION_PATH = (
    REPOSITORY_ROOT / "examples/puzzletron/configs/orchestration/execution.single_gpu.yaml"
)
PRODUCTION_RUN_PATH = (
    REPOSITORY_ROOT
    / "examples/puzzletron/configs/families/qwen3_5/qwen3p5_0p8b/runs/vlm_campaign.yaml"
)
COMPARISON_RUN_PATH = (
    REPOSITORY_ROOT
    / "examples/puzzletron/configs/families/qwen3_5/qwen3p5_0p8b/runs/e2e_vlm_quality_comparison.yaml"
)
FAMILY_PRESETS_PATH = (
    REPOSITORY_ROOT / "examples/puzzletron/configs/families/qwen3_5/setup_v2_defaults.yaml"
)
VLM_EVALUATION_PATH = (
    REPOSITORY_ROOT
    / "examples/puzzletron/configs/families/qwen3_5/qwen3p5_0p8b/vlm_quality_evaluation.yaml"
)


def _compile_plan(monkeypatch, tmp_path: Path, *, dataset_revision="fixture-revision"):
    monkeypatch.setenv("PUZZLETRON_RUN_ROOT", str(tmp_path / "results"))
    monkeypatch.setenv("PUZZLETRON_DATASET_PATH", str(tmp_path / "dataset"))
    if dataset_revision is None:
        monkeypatch.delenv("PUZZLETRON_DATASET_REVISION", raising=False)
    else:
        monkeypatch.setenv("PUZZLETRON_DATASET_REVISION", dataset_revision)
    return compile_campaign_plan(
        experiment_config_path=RUN_PATH,
        runner=load_runner_config(RUNNER_PATH),
        execution=load_execution_config(EXECUTION_PATH),
        stage_filter="full",
    )


def _compile_campaign(monkeypatch, tmp_path: Path, *, run_path: Path, execution_path: Path):
    monkeypatch.setenv("PUZZLETRON_RUN_ROOT", str(tmp_path / run_path.stem))
    monkeypatch.setenv("PUZZLETRON_DATASET_PATH", str(tmp_path / "dataset"))
    monkeypatch.setenv("PUZZLETRON_DATASET_REVISION", "fixture-revision")
    return compile_campaign_plan(
        experiment_config_path=run_path,
        runner=load_runner_config(RUNNER_PATH),
        execution=load_execution_config(execution_path),
        stage_filter="full",
    )


def test_qwen3p5_0p8b_full_vlm_smoke_defaults_to_the_tested_dataset_snapshot(
    monkeypatch,
    tmp_path: Path,
) -> None:
    run_config = yaml.safe_load(RUN_PATH.read_text())
    config = _compile_plan(monkeypatch, tmp_path, dataset_revision=None).experiment_config

    assert run_config["defaults"] == ["mip_vlm_smoke", "_self_"]
    assert config["data"]["revision"] == "51f4f4d219315c3283950994d4eb3d7fc30aa87b"
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


def test_qwen3p5_0p8b_full_vlm_smoke_compiles_the_one_gpu_lifecycle(
    monkeypatch,
    tmp_path: Path,
) -> None:
    plan = _compile_plan(monkeypatch, tmp_path)
    stage_ids = tuple(stage.stage_id for stage in plan.stages)
    post_stage_ids = tuple(stage_id for stage_id in stage_ids if stage_id.startswith("post."))

    assert "tokenize_data" not in stage_ids
    assert post_stage_ids == (
        "post.params-90.image_eval",
        "post.params-90.best_vlm_loss",
        "post.params-90.materialized",
        "post.params-90.checkpoint_eval",
        "post.params-90.vlm_serving",
        "post.params-90.fastest_vlm",
        "post.params-90.short_vlm_kd",
        "post.params-90.post_kd_checkpoint_eval",
        "post.params-90.final_image_eval",
        "post.params-90.best",
    )
    post_nodes = tuple(stage for stage in plan.stages if stage.stage_id.startswith("post."))
    assert post_nodes[0].parents == ("mip",)
    for parent, node in pairwise(post_nodes):
        assert node.parents == (parent.stage_id,)
    assert stage_ids[-1] == "post.params-90.best"
    assert all(stage.total_gpus == 1 for stage in plan.stages)


def test_qwen3p5_0p8b_full_vlm_smoke_bounds_work_and_declares_vlm_kd(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config = _compile_plan(monkeypatch, tmp_path).experiment_config
    nodes = config["post_mip"]["flows"]["params-90"]["nodes"]

    assert nodes["image_eval"]["config"] == {"eval_samples": 2, "block_size": 512}
    assert config["mip"]["runs"]["params-90"]["solver"]["num_solutions"] == 1
    assert config["mip"]["runs"]["params-90"]["homogeneous"]["keep"] == 5
    assert nodes["best_vlm_loss"]["top_k"] == 2
    assert nodes["checkpoint_eval"]["type"] == "downstream_evaluation"
    assert nodes["checkpoint_eval"]["failure_policy"] == "strict"
    assert nodes["checkpoint_eval"]["config"]["profile"] == "qwen35_vlm_realworldqa"
    assert nodes["checkpoint_eval"]["config"]["batch_size"] == 1
    assert nodes["checkpoint_eval"]["config"]["timeout_seconds"] == 600
    assert nodes["checkpoint_eval"]["config"]["limit_mm_per_prompt"] == {"image": 1}
    assert nodes["vlm_serving"]["config"]["readiness_timeout"] == 300
    assert nodes["vlm_serving"]["config"]["benchmark_timeout"] == 300
    assert nodes["fastest_vlm"] == {
        "type": "filter",
        "input": "vlm_serving",
        "mode": "top_k",
        "metric": "vlm_serving.images_12.concurrency_1.image_throughput",
        "direction": "maximize",
        "top_k": 1,
    }
    assert nodes["short_vlm_kd"]["input"] == "fastest_vlm"
    assert nodes["short_vlm_kd"]["config"] == {
        "max_steps": 2,
        "global_batch_size": 1,
        "local_batch_size": 1,
        "checkpoint_every_steps": 2,
    }
    assert nodes["post_kd_checkpoint_eval"]["type"] == "downstream_evaluation"
    assert nodes["post_kd_checkpoint_eval"]["failure_policy"] == "strict"
    assert nodes["post_kd_checkpoint_eval"]["config"] == nodes["checkpoint_eval"]["config"]
    assert nodes["final_image_eval"]["config"] == {
        "eval_samples": 2,
        "block_size": 512,
    }
    assert nodes["best"]["top_k"] == 1
    assert nodes["vlm_serving"]["config"]["endpoint_type"] == "chat"
    assert nodes["vlm_serving"]["config"]["image_batch_sizes"] == [1, 6, 12]
    assert nodes["vlm_serving"]["config"]["image_width_mean"] == 1280
    assert nodes["vlm_serving"]["config"]["image_height_mean"] == 720
    assert nodes["vlm_serving"]["config"]["concurrency"] == [1]
    assert nodes["vlm_serving"]["config"]["request_count"] == 1
    assert nodes["vlm_serving"]["config"]["topology"]["server_context_overhead_tokens"] == 16384
    assert config["global_distillation"]["domain"] == "vlm"
    assert config["global_distillation"]["freeze_policy"] == "train_all"


def test_qwen3p5_0p8b_vlm_routes_are_independent_and_share_the_evaluation_contract(
    monkeypatch,
    tmp_path: Path,
) -> None:
    production = _compile_campaign(
        monkeypatch,
        tmp_path,
        run_path=PRODUCTION_RUN_PATH,
        execution_path=EXECUTION_PATH,
    )
    comparison = _compile_campaign(
        monkeypatch,
        tmp_path,
        run_path=COMPARISON_RUN_PATH,
        execution_path=EXECUTION_PATH,
    )
    production_config = production.experiment_config
    config = comparison.experiment_config
    production_nodes = production_config["post_mip"]["flows"]["params-90"]["nodes"]
    nodes = config["post_mip"]["flows"]["params-90"]["nodes"]
    benchmark = nodes["quality_benchmarks"]
    stages = {stage.stage_id: stage for stage in comparison.stages}
    family_presets = yaml.safe_load(FAMILY_PRESETS_PATH.read_text())
    evaluator = yaml.safe_load(VLM_EVALUATION_PATH.read_text())["vlm_quality_evaluation"]

    production_defaults = yaml.safe_load(PRODUCTION_RUN_PATH.read_text())["defaults"]
    comparison_defaults = yaml.safe_load(COMPARISON_RUN_PATH.read_text())["defaults"]
    shared_evaluator = "/families/qwen3_5/qwen3p5_0p8b/vlm_quality_evaluation@_global_"
    assert production_defaults == ["full_vlm_smoke", shared_evaluator, "_self_"]
    assert comparison_defaults == ["full_vlm_smoke", shared_evaluator, "_self_"]
    assert production_config["mip"]["runs"]["params-90"]["search_space"] == {
        "depth": [0],
        "embedding": [1024],
        "axes_default": "teacher",
        "axes": {"ffn.intermediate_size": "all"},
    }

    production_evaluation = dict(production_nodes["quality_benchmarks"]["config"])
    comparison_evaluation = dict(benchmark["config"])
    assert production_evaluation.pop("reference_checkpoint") == production_config["teacher_dir"]
    assert comparison_evaluation.pop("reference_checkpoint") == config["teacher_dir"]
    recorded_observation = comparison_evaluation.pop("recorded_observation")
    assert comparison_evaluation == production_evaluation
    assert stages["post.params-90.quality_benchmarks"].parents == ("post.params-90.short_vlm_kd",)
    assert benchmark["input"] == "short_vlm_kd"
    assert benchmark["failure_policy"] == "strict"
    assert benchmark["config"]["profile"] == "qwen35_vlm_e2e_full_eval"
    assert benchmark["config"]["reference_checkpoint"] == config["teacher_dir"]
    assert benchmark["config"]["batch_size"] == 1
    assert benchmark["config"]["timeout_seconds"] == 14400
    assert benchmark["config"]["max_model_len"] == 16384
    assert benchmark["config"]["limit_mm_per_prompt"] == {"image": 12}
    assert recorded_observation == {
        "repeat_count": 2,
        "candidate_architecture": {
            "parameter_pruning_percent": 10.042,
            "parameter_retention_percent": 89.958,
            "ffn_layers": 24,
            "teacher_intermediate_size": 3584,
            "student_intermediate_size": 2048,
            "ffn_width_pruning_percent": 42.857,
        },
        "metrics": {
            "candidate.modelopt_vlm_benchmark_realworldqa.exact_match_flexible-extract": 0.24,
            "reference.modelopt_vlm_benchmark_realworldqa.exact_match_flexible-extract": 0.60,
            "candidate.modelopt_vlm_benchmark_mmmu_val.mmmu_acc_none": 0.25,
            "reference.modelopt_vlm_benchmark_mmmu_val.mmmu_acc_none": 0.35,
        },
    }
    assert all("quality_gate" not in node_id for node_id in nodes)
    wizard_quality = family_presets["model_overrides"]["qwen3p5_0p8b"]["defaults"]["post_mip"][
        "quality_comparison"
    ]["by_modality"]["multimodal"]
    assert wizard_quality.pop("enabled") is True
    assert "recorded_observation" not in evaluator
    assert wizard_quality == evaluator
    assert production_config["pruning"]["eval_samples"] == 128
    assert production_config["mip"]["runs"]["params-90"]["solver"]["num_solutions"] == 16
    assert production_nodes["image_eval"]["config"]["eval_samples"] == 16
    assert production_nodes["best_vlm_loss"]["top_k"] == 4
    assert config["pruning"]["eval_samples"] == 8
    assert production_nodes["vlm_serving"]["config"]["request_count"] == 32
    assert nodes["vlm_serving"]["config"]["request_count"] == 1
    assert production_nodes["short_vlm_kd"]["config"]["max_steps"] == 256
    assert nodes["short_vlm_kd"]["config"]["max_steps"] == 2
    assert all(stage.total_gpus == 1 for stage in (*production.stages, *comparison.stages))
