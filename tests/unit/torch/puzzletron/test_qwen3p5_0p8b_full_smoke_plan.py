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

from modelopt.torch.puzzletron.pipeline_config import pipeline_config_from_path
from puzzletron_orchestrator.compiler import (
    compile_campaign_plan,
    load_execution_config,
    load_runner_config,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
FAMILY_ROOT = REPOSITORY_ROOT / "examples/puzzletron/configs/families/qwen3_5"
FAMILY_PRESETS_PATH = FAMILY_ROOT / "setup_v2_defaults.yaml"
QUALITY_EVALUATION_PATH = FAMILY_ROOT / "qwen3p5_0p8b/quality_evaluation.yaml"
EXPLICIT_SETUP_DEFAULTS_PATH = (
    REPOSITORY_ROOT / "tests/_test_utils/torch/puzzletron/tiny_qwen_setup_defaults.yaml"
)
RUN_PATH = FAMILY_ROOT / "qwen3p5_0p8b/runs/full_smoke.yaml"
ORCHESTRATION_ROOT = REPOSITORY_ROOT / "examples/puzzletron/configs/orchestration/qwen3p5_0p8b"
RUNNER_PATH = ORCHESTRATION_ROOT / "runner.slurm.yaml"
EXECUTION_PATH = (
    REPOSITORY_ROOT / "examples/puzzletron/configs/orchestration/execution.single_gpu.yaml"
)
TWO_GPU_KD_EXECUTION_PATH = (
    REPOSITORY_ROOT / "examples/puzzletron/configs/orchestration/execution.two_gpu_kd.yaml"
)
QUALITY_COMPARISON_RUN_PATH = FAMILY_ROOT / "qwen3p5_0p8b/runs/e2e_quality_comparison.yaml"
EXTENDED_SMOKE_PATH = FAMILY_ROOT / "qwen3p5_0p8b/runs/full_smoke_extended.yaml"
EXTENDED_QUALITY_COMPARISON_PATH = (
    FAMILY_ROOT / "qwen3p5_0p8b/runs/e2e_quality_comparison_extended.yaml"
)
CAMPAIGN_PATH = FAMILY_ROOT / "qwen3p5_0p8b/runs/campaign.yaml"
EXTENDED_CAMPAIGN_PATH = FAMILY_ROOT / "qwen3p5_0p8b/runs/campaign_extended.yaml"


def _compile_plan(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("PUZZLETRON_RUN_ROOT", str(tmp_path / "results"))
    monkeypatch.setenv("PUZZLETRON_DATASET_PATH", str(tmp_path / "dataset"))
    return compile_campaign_plan(
        experiment_config_path=RUN_PATH,
        runner=load_runner_config(RUNNER_PATH),
        execution=load_execution_config(EXECUTION_PATH),
        stage_filter="full",
    )


def _compile_quality_comparison_plan(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("PUZZLETRON_RUN_ROOT", str(tmp_path / "quality-comparison-results"))
    monkeypatch.setenv("PUZZLETRON_DATASET_PATH", str(tmp_path / "dataset"))
    return compile_campaign_plan(
        experiment_config_path=QUALITY_COMPARISON_RUN_PATH,
        runner=load_runner_config(RUNNER_PATH),
        execution=load_execution_config(EXECUTION_PATH),
        stage_filter="full",
    )


def _compile_extended_plan(monkeypatch, tmp_path: Path, run_path: Path):
    monkeypatch.setenv("PUZZLETRON_RUN_ROOT", str(tmp_path / run_path.stem))
    monkeypatch.setenv("PUZZLETRON_DATASET_PATH", str(tmp_path / "dataset"))
    return compile_campaign_plan(
        experiment_config_path=run_path,
        runner=load_runner_config(RUNNER_PATH),
        execution=load_execution_config(EXECUTION_PATH),
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


def test_qwen3p5_0p8b_e2e_quality_comparison_is_opt_in_and_compares_teacher(
    monkeypatch,
    tmp_path: Path,
) -> None:
    plan = _compile_quality_comparison_plan(monkeypatch, tmp_path)
    config = plan.experiment_config
    nodes = config["post_mip"]["flows"]["params-90"]["nodes"]
    benchmark = nodes["quality_benchmarks"]
    stage_ids = tuple(stage.stage_id for stage in plan.stages)
    stages = {stage.stage_id: stage for stage in plan.stages}

    assert "post.params-90.quality_benchmarks" in stage_ids
    assert stages["post.params-90.quality_benchmarks"].parents == ("post.params-90.short_kd",)
    assert benchmark["input"] == "short_kd"
    assert benchmark["failure_policy"] == "strict"
    assert benchmark["config"]["reference_checkpoint"] == config["teacher_dir"]
    assert benchmark["config"]["tasks"] == [
        "ifeval",
        "gsm8k",
        "mmlu_pro_computer_science",
        "mmlu_pro_history",
    ]
    assert benchmark["config"]["compatibility_tasks"] == benchmark["config"]["tasks"]
    assert benchmark["config"]["task_dataset_revisions"] == {
        "ifeval": "5a5661c2a35488308556cf4453dc074d1eba91a0",
        "gsm8k": "740312add88f781978c0658806c59bc2815b9866",
        "mmlu_pro_computer_science": "b189ec765aa7ed75c8acfea42df31fdae71f97be",
        "mmlu_pro_history": "b189ec765aa7ed75c8acfea42df31fdae71f97be",
    }
    assert benchmark["config"]["batch_size"] == 8
    assert benchmark["config"]["limit"] == 256
    assert benchmark["config"]["gen_kwargs"] == {
        "do_sample": False,
        "temperature": 0,
    }
    observation = benchmark["config"]["recorded_observation"]
    assert observation["repeat_count"] > 0
    assert set(observation["metrics"]) == {
        "candidate.modelopt_ifeval.prompt_level_strict_acc_none",
        "reference.modelopt_ifeval.prompt_level_strict_acc_none",
        "candidate.modelopt_gsm8k.exact_match_flexible-extract",
        "reference.modelopt_gsm8k.exact_match_flexible-extract",
    }
    assert 0 < observation["candidate_architecture"]["parameter_pruning_percent"] < 100
    assert stages["post.params-90.quality_benchmarks"].total_gpus == 1


def test_qwen3p5_0p8b_extended_grid_is_shared_by_smoke_and_regression(
    monkeypatch,
    tmp_path: Path,
) -> None:
    smoke = pipeline_config_from_path(EXTENDED_SMOKE_PATH)
    regression = pipeline_config_from_path(EXTENDED_QUALITY_COMPARISON_PATH)
    campaign = pipeline_config_from_path(EXTENDED_CAMPAIGN_PATH)
    expected_axes = {
        "hidden_width": {"enabled": True, "teacher_value": 1024, "values": [768]},
        "kv_groups": {"enabled": True, "teacher_value": 2, "values": [1]},
        "q_heads_per_group": {"enabled": True, "teacher_value": 4, "values": [2]},
        "ffn_intermediate": {
            "enabled": True,
            "teacher_value": 3584,
            "values": [3072, 2560, 2048, 1792, 1536],
        },
        "gdn_key_groups": {"enabled": True, "teacher_value": 16, "values": [12, 8]},
        "gdn_value_heads_per_group": {
            "enabled": False,
            "teacher_value": 1,
            "values": [],
        },
        "gdn_key_head_dim": {
            "enabled": False,
            "teacher_value": 128,
            "values": [],
        },
        "gdn_value_head_dim": {
            "enabled": True,
            "teacher_value": 128,
            "values": [96],
        },
    }
    expected_mip_grid = {
        "depth": [0, 1, 2],
        "embedding": [1024, 768],
        "axes_default": "all",
        "axes": {"ffn.intermediate_size": "all"},
    }

    assert smoke["search_space"]["axes"] == expected_axes
    assert regression["search_space"]["axes"] == expected_axes
    assert campaign["search_space"]["axes"] == expected_axes
    assert smoke["embedding_pruning"]["widths"] == [1024, 768]
    assert smoke["pruning"]["attention_scored_axes"] == ["kv_groups", "q_heads_per_group"]
    assert smoke["pruning"]["gdn_scored_axes"] == ["gdn_key_groups", "gdn_value_head_dim"]
    assert smoke["sort"]["deferred_axes"] == []
    assert smoke["width_sanity"]["hidden_width_diagnostic"] is True
    assert smoke["width_sanity"]["axes"] == [
        "hidden_width",
        "ffn_intermediate",
        "kv_groups",
        "q_heads_per_group",
        "gdn_key_groups",
        "gdn_value_head_dim",
    ]
    assert smoke["depth_importance"]["enabled"] is True
    assert smoke["mip"]["runs"]["params-90"]["search_space"] == expected_mip_grid
    assert regression["mip"]["runs"]["params-90"]["search_space"] == expected_mip_grid
    assert campaign["mip"]["runs"]["params-90"]["search_space"] == expected_mip_grid
    assert "recorded_observation" not in regression["quality_evaluation"]

    smoke_plan = _compile_extended_plan(monkeypatch, tmp_path, EXTENDED_SMOKE_PATH)
    regression_plan = _compile_extended_plan(
        monkeypatch,
        tmp_path,
        EXTENDED_QUALITY_COMPARISON_PATH,
    )
    smoke_stages = {stage.stage_id: stage for stage in smoke_plan.stages}
    regression_stages = {stage.stage_id: stage for stage in regression_plan.stages}
    assert "depth_importance" in smoke_stages
    assert "depth_importance" in regression_stages
    assert "post.params-90.quality_benchmarks" not in smoke_stages
    assert regression_stages["post.params-90.quality_benchmarks"].parents == (
        "post.params-90.short_kd",
    )


def test_qwen3p5_0p8b_campaign_reuses_the_bounded_quality_settings(
    monkeypatch,
    tmp_path: Path,
) -> None:
    campaign = pipeline_config_from_path(CAMPAIGN_PATH)
    comparison = pipeline_config_from_path(QUALITY_COMPARISON_RUN_PATH)
    family_presets = yaml.safe_load(FAMILY_PRESETS_PATH.read_text())
    monkeypatch.setenv("PUZZLETRON_RUN_ROOT", str(tmp_path / "campaign"))
    monkeypatch.setenv("PUZZLETRON_DATASET_PATH", str(tmp_path / "dataset"))
    plan = compile_campaign_plan(
        experiment_config_path=CAMPAIGN_PATH,
        runner=load_runner_config(RUNNER_PATH),
        execution=load_execution_config(TWO_GPU_KD_EXECUTION_PATH),
        stage_filter="full",
    )

    assert campaign["model"]["revision"] == campaign["model_info"]["hf_revision"]
    train_cache = next(
        cache
        for cache in campaign["tokenize_data"]["caches"]
        if cache["output"] == campaign["train_token_cache_path"]
    )
    assert train_cache["num_samples"] == 4096
    assert campaign["data"]["calibration"]["num_samples"] == 4096
    assert train_cache["num_samples"] >= 256 * 16
    assert campaign["search_space"]["axes"] == {
        "ffn_intermediate": {
            "enabled": True,
            "teacher_value": 3584,
            "values": [3328, 3072],
        }
    }
    assert campaign["mip"]["runs"]["params-90"] is False
    conservative_run = campaign["mip"]["runs"]["conservative-ffn"]
    assert conservative_run["search_space"] == {
        "depth": [0],
        "embedding": [1024],
        "axes_default": "teacher",
        "axes": {"ffn.intermediate_size": "all"},
    }
    assert conservative_run["variants"] == {
        "width-3328": {
            "constraints": {"params": {"max": "99%"}},
            "search_space": {"axes": {"ffn.intermediate_size": [3328]}},
        },
        "width-3072": {
            "constraints": {"params": {"max": "97%"}},
            "search_space": {"axes": {"ffn.intermediate_size": [3072]}},
        },
    }
    assert campaign["bypass"]["enabled"] is False
    assert campaign["depth_importance"]["enabled"] is False
    nodes = campaign["post_mip"]["flows"]["candidate-evaluation"]["nodes"]
    screen_quality = nodes["quality_screen"]["config"]
    full_quality = nodes["quality_benchmarks"]["config"]
    comparison_quality = comparison["post_mip"]["flows"]["params-90"]["nodes"][
        "quality_benchmarks"
    ]["config"]
    wizard_quality = family_presets["model_overrides"]["qwen3p5_0p8b"]["defaults"]["post_mip"][
        "quality_comparison"
    ]["by_modality"]["text"]
    assert wizard_quality["enabled"] is True
    assert {
        key: value
        for key, value in wizard_quality.items()
        if key not in {"enabled", "reference_checkpoint"}
    } == {
        key: value
        for key, value in campaign["quality_evaluation"].items()
        if key != "reference_checkpoint"
    }
    assert screen_quality == campaign["quality_evaluation"]
    assert full_quality == campaign["quality_evaluation_full"]
    assert screen_quality["tasks"] == [
        "ifeval",
        "gsm8k",
        "mmlu_pro_computer_science",
        "mmlu_pro_history",
    ]
    assert screen_quality["compatibility_tasks"] == screen_quality["tasks"]
    assert screen_quality["task_dataset_revisions"] == {
        "ifeval": "5a5661c2a35488308556cf4453dc074d1eba91a0",
        "gsm8k": "740312add88f781978c0658806c59bc2815b9866",
        "mmlu_pro_computer_science": "b189ec765aa7ed75c8acfea42df31fdae71f97be",
        "mmlu_pro_history": "b189ec765aa7ed75c8acfea42df31fdae71f97be",
    }
    assert screen_quality["limit"] == 256
    assert full_quality["limit"] is None
    comparable_screen = dict(screen_quality)
    comparable_full = dict(full_quality)
    comparable_screen.pop("limit")
    comparable_full.pop("limit")
    assert comparable_screen == comparable_full
    assert screen_quality["seed"] == 42
    assert screen_quality["num_fewshot"] == 0
    assert screen_quality["log_samples"] is True
    assert screen_quality["gen_kwargs"] == {"do_sample": False, "temperature": 0}
    assert "evaluation_policy" not in campaign
    assert "quality_evaluation_finalist" not in campaign
    assert "quality_evaluation_confirmation" not in campaign
    assert "paired_analysis" not in screen_quality
    parallel = {
        "automodel": {
            "parallel": {
                "tp": 1,
                "cp": 1,
                "pp": 1,
                "ep": 1,
                "dp_shard": 1,
                "dp_replicate": 2,
                "sequence_parallel": False,
            }
        }
    }
    assert nodes["screening_kd"] == {
        "type": "global_kd",
        "input": "serving",
        "config": {
            "seed": 1111,
            "validation_seed": 445,
            "shuffle_training_data": True,
            "max_steps": 128,
            "global_batch_size": 16,
            "local_batch_size": 4,
            "checkpoint_every_steps": 32,
            **parallel,
        },
    }
    assert nodes["screening_eval"] == {
        "type": "evaluation",
        "input": "screening_kd",
        "config": {"eval_samples": 32, "block_size": 2048},
    }
    assert nodes["quality_screen"]["input"] == "screening_eval"
    assert nodes["winner"] == {
        "type": "filter",
        "input": "quality_screen",
        "mode": "aggregate_rank",
        "metrics": [
            {
                "metric": "screening_eval.lm_loss",
                "direction": "minimize",
            },
            {
                "metric": "quality_screen.modelopt_ifeval.prompt_level_strict_acc_none",
                "direction": "maximize",
            },
            {
                "metric": "quality_screen.modelopt_gsm8k.exact_match_flexible-extract",
                "direction": "maximize",
            },
            {
                "metric": (
                    "quality_screen.modelopt_mmlu_pro_computer_science.exact_match_flexible-extract"
                ),
                "direction": "maximize",
            },
            {
                "metric": ("quality_screen.modelopt_mmlu_pro_history.exact_match_flexible-extract"),
                "direction": "maximize",
            },
        ],
        "top_k": 1,
    }
    assert nodes["global_kd"] == {
        "type": "global_kd",
        "input": "winner",
        "model_source": "materialized",
        "config": {
            "seed": 1111,
            "validation_seed": 445,
            "shuffle_training_data": True,
            "max_steps": 256,
            "global_batch_size": 16,
            "local_batch_size": 4,
            "checkpoint_every_steps": 32,
            **parallel,
        },
    }
    assert nodes["final_eval"]["input"] == "global_kd"
    assert nodes["quality_benchmarks"]["input"] == "final_eval"
    assert comparison_quality == comparison["quality_evaluation"]
    assert "recorded_observation" not in full_quality
    assert "recorded_observation" not in wizard_quality
    assert {
        key: value
        for key, value in comparison_quality.items()
        if key not in {"recorded_observation", "reference_checkpoint"}
    } == {
        key: value
        for key, value in campaign["quality_evaluation"].items()
        if key != "reference_checkpoint"
    }
    assert full_quality["reference_checkpoint"] == campaign["teacher_dir"]
    assert comparison_quality["reference_checkpoint"] == comparison["teacher_dir"]
    stages = {stage.stage_id: stage for stage in plan.stages}
    two_candidate_stages = {
        "post.candidate-evaluation.online_eval",
        "post.candidate-evaluation.materialized",
        "post.candidate-evaluation.serving",
        "post.candidate-evaluation.screening_kd",
        "post.candidate-evaluation.screening_eval",
        "post.candidate-evaluation.quality_screen",
    }
    winner_stages = {
        "post.candidate-evaluation.global_kd",
        "post.candidate-evaluation.final_eval",
        "post.candidate-evaluation.quality_benchmarks",
    }
    assert all(stages[name].instances == 2 for name in two_candidate_stages)
    assert all(stages[name].instances == 1 for name in winner_stages)
    assert stages["post.candidate-evaluation.screening_kd"].total_gpus == 4
    assert stages["post.candidate-evaluation.global_kd"].total_gpus == 2


def test_qwen3p5_0p8b_extended_campaign_exposes_additional_axes(
    monkeypatch,
    tmp_path: Path,
) -> None:
    campaign = pipeline_config_from_path(EXTENDED_CAMPAIGN_PATH)
    assert campaign["mip"]["runs"]["conservative-ffn"] is False
    assert campaign["post_mip"]["flows"]["candidate-evaluation"]["source"]["run"] == ("params-90")

    monkeypatch.setenv("PUZZLETRON_RUN_ROOT", str(tmp_path / "extended-campaign"))
    monkeypatch.setenv("PUZZLETRON_DATASET_PATH", str(tmp_path / "dataset"))
    plan = compile_campaign_plan(
        experiment_config_path=EXTENDED_CAMPAIGN_PATH,
        runner=load_runner_config(RUNNER_PATH),
        execution=load_execution_config(TWO_GPU_KD_EXECUTION_PATH),
        stage_filter="full",
    )

    stage_ids = {stage.stage_id for stage in plan.stages}
    assert "depth_importance" in stage_ids
    assert "post.candidate-evaluation.quality_benchmarks" in stage_ids
