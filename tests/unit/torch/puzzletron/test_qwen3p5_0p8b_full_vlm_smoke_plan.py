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

"""CPU plan contracts for the end-to-end Qwen 3.5 0.8B VLM lifecycle smoke."""

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
CAMPAIGN_EXECUTION_PATH = ORCHESTRATION_ROOT / "execution.campaign.yaml"
CAMPAIGN_PATH = (
    REPOSITORY_ROOT
    / "examples/puzzletron/configs/families/qwen3_5/qwen3p5_0p8b/runs/vlm_campaign.yaml"
)
COMPARISON_RUN_PATH = (
    REPOSITORY_ROOT
    / "examples/puzzletron/configs/families/qwen3_5/qwen3p5_0p8b/runs/e2e_vlm_quality_comparison.yaml"
)
EXTENDED_RUN_PATH = (
    REPOSITORY_ROOT
    / "examples/puzzletron/configs/families/qwen3_5/qwen3p5_0p8b/runs/full_vlm_smoke_extended.yaml"
)
EXTENDED_COMPARISON_RUN_PATH = (
    REPOSITORY_ROOT
    / "examples/puzzletron/configs/families/qwen3_5/qwen3p5_0p8b/runs/e2e_vlm_quality_comparison_extended.yaml"
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


def _compile_campaign(monkeypatch, tmp_path: Path, *, run_path: Path):
    monkeypatch.setenv("PUZZLETRON_RUN_ROOT", str(tmp_path / run_path.stem))
    monkeypatch.setenv("PUZZLETRON_DATASET_PATH", str(tmp_path / "dataset"))
    monkeypatch.setenv("PUZZLETRON_DATASET_REVISION", "fixture-revision")
    monkeypatch.setenv("PUZZLETRON_VLM_SHORT_V1_MANIFEST", str(tmp_path / "short-v1.json"))
    monkeypatch.setenv("PUZZLETRON_VLM_SHORT_V1_SHA256", "a" * 64)
    return compile_campaign_plan(
        experiment_config_path=run_path,
        runner=load_runner_config(RUNNER_PATH),
        execution=load_execution_config(EXECUTION_PATH),
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


def test_qwen3p5_0p8b_vlm_comparison_uses_the_shared_evaluator(
    monkeypatch,
    tmp_path: Path,
) -> None:
    comparison = _compile_campaign(monkeypatch, tmp_path, run_path=COMPARISON_RUN_PATH)
    comparison_config = comparison.experiment_config
    comparison_nodes = comparison_config["post_mip"]["flows"]["params-90"]["nodes"]
    benchmark = comparison_nodes["quality_benchmarks"]
    stages = {stage.stage_id: stage for stage in comparison.stages}
    family_presets = yaml.safe_load(FAMILY_PRESETS_PATH.read_text())
    evaluator = yaml.safe_load(VLM_EVALUATION_PATH.read_text())["vlm_quality_evaluation"]

    shared_evaluator = "/families/qwen3_5/qwen3p5_0p8b/vlm_quality_evaluation@_global_"
    assert yaml.safe_load(COMPARISON_RUN_PATH.read_text())["defaults"] == [
        "full_vlm_smoke",
        shared_evaluator,
        "_self_",
    ]
    comparison_evaluation = dict(benchmark["config"])
    assert comparison_evaluation.pop("reference_checkpoint") == comparison_config["teacher_dir"]
    evaluator_without_reference = dict(evaluator)
    evaluator_without_reference.pop("reference_checkpoint")
    assert comparison_evaluation == evaluator_without_reference
    assert "recorded_observation" not in benchmark["config"]
    assert stages["post.params-90.quality_benchmarks"].parents == ("post.params-90.short_vlm_kd",)
    assert benchmark["input"] == "short_vlm_kd"
    assert benchmark["failure_policy"] == "strict"
    assert benchmark["config"]["profile"] == "qwen35_vlm_e2e_full_eval"

    wizard_quality = family_presets["model_overrides"]["qwen3p5_0p8b"]["defaults"]["post_mip"][
        "quality_comparison"
    ]["by_modality"]["multimodal"]
    assert wizard_quality.pop("enabled") is True
    assert wizard_quality == evaluator
    assert all(stage.total_gpus == 1 for stage in comparison.stages)


def test_qwen3p5_0p8b_extended_vlm_smoke_realizes_one_in_band_mixed_candidate(
    monkeypatch,
    tmp_path: Path,
) -> None:
    smoke = _compile_campaign(monkeypatch, tmp_path, run_path=EXTENDED_RUN_PATH)
    regression = _compile_campaign(
        monkeypatch,
        tmp_path,
        run_path=EXTENDED_COMPARISON_RUN_PATH,
    )
    config = smoke.experiment_config
    profile = config["mip"]["runs"]["params-90"]
    stages = {stage.stage_id: stage for stage in smoke.stages}
    regression_stages = {stage.stage_id: stage for stage in regression.stages}

    assert yaml.safe_load(EXTENDED_RUN_PATH.read_text())["defaults"] == [
        "full_vlm_smoke",
        "/families/qwen3_5/qwen3p5_0p8b/advanced@_global_",
        "_self_",
    ]
    assert config["sort"]["deferred_axes"] == []
    assert config["width_sanity"]["axes"] == [
        "hidden_width",
        "ffn_intermediate",
    ]
    assert config["depth_importance"]["max_removals"] == 1
    assert profile["constraints"] == {"params": {"min": "85%", "max": "95%"}}
    assert profile["search_space"] == {
        "depth": [1],
        "embedding": [960],
        "axes_default": "teacher",
        "axes": {
            "ffn.intermediate_size": [2816],
        },
    }
    assert profile["solver"]["num_solutions"] == 1
    assert profile["homogeneous"]["enabled"] is False
    assert "depth_importance" in stages
    assert stages["post.params-90.materialized"].parents == ("post.params-90.best_vlm_loss",)
    assert stages["post.params-90.post_kd_checkpoint_eval"].parents == (
        "post.params-90.short_vlm_kd",
    )
    assert regression_stages["post.params-90.quality_benchmarks"].parents == (
        "post.params-90.short_vlm_kd",
    )
    assert all(stage.total_gpus == 1 for stage in (*smoke.stages, *regression.stages))


def test_qwen3p5_0p8b_vlm_campaign_uses_default_candidate_selection(
    monkeypatch,
    tmp_path: Path,
) -> None:
    campaign = _compile_campaign(
        monkeypatch,
        tmp_path,
        run_path=CAMPAIGN_PATH,
    )
    comparison = _compile_campaign(
        monkeypatch,
        tmp_path,
        run_path=COMPARISON_RUN_PATH,
    )
    config = campaign.experiment_config
    comparison_config = comparison.experiment_config
    nodes = config["post_mip"]["flows"]["candidate-evaluation"]["nodes"]
    comparison_nodes = comparison_config["post_mip"]["flows"]["params-90"]["nodes"]

    assert yaml.safe_load(CAMPAIGN_PATH.read_text())["defaults"] == [
        "mip_vlm_smoke",
        "/families/qwen3_5/qwen3p5_0p8b/advanced@_global_",
        "_self_",
    ]
    candidates = config["mip"]["runs"]["params-90"]
    assert candidates["constraints"] == {"params": {"max": "90%"}}
    assert candidates["solver"]["num_solutions"] == 1
    assert candidates["search_space"] == {
        "depth": [0, 1, 2],
        "embedding": [1024, 960, 896],
        "axes_default": "all",
        "axes": {"ffn.intermediate_size": "all"},
    }
    assert config["global_distillation"]["domain"] == "vlm"
    assert config["global_distillation"]["freeze_policy"] == "train_all"
    controls = config["mip"]["runs"]["calibration-controls"]
    assert controls["variants"] == {
        "width-3328": {
            "constraints": {"params": {"min": "95%", "max": "100%"}},
            "search_space": {"axes": {"ffn.intermediate_size": [3328]}},
        },
        "width-3072": {
            "constraints": {"params": {"min": "95%", "max": "100%"}},
            "search_space": {"axes": {"ffn.intermediate_size": [3072]}},
        },
    }
    assert config["post_mip"]["flows"]["candidate-evaluation"]["source"]["run"] == ("params-90")
    assert "input" not in nodes["online_eval"]
    assert nodes["best_lm"] == {
        "type": "filter",
        "input": "online_eval",
        "mode": "top_k",
        "metric": "online_eval.lm_loss",
        "direction": "minimize",
        "top_k": 2,
    }
    assert nodes["pre_kd_short_v1"]["input"] == "serving"
    assert nodes["kd_64"]["input"] == "pre_kd_short_v1"
    trajectory_nodes = [nodes[f"kd_{steps}"] for steps in (64, 128, 256, 512, 1024)]
    assert [node["config"]["max_steps"] for node in trajectory_nodes] == [
        64,
        128,
        256,
        512,
        1024,
    ]
    assert {node["trajectory"] for node in trajectory_nodes} == {
        "retained-candidate-learning-curve"
    }
    assert {node["model_source"] for node in trajectory_nodes} == {"materialized"}
    assert all(node["config"]["resume"] is True for node in trajectory_nodes)
    assert all(node["failure_policy"] == "strict" for node in trajectory_nodes)
    assert all(node["config"]["global_batch_size"] == 4 for node in trajectory_nodes)
    assert [node["exposure"]["cumulative_examples"] for node in trajectory_nodes] == [
        256,
        512,
        1024,
        2048,
        4096,
    ]
    assert [node["exposure"]["estimated_cumulative_gpu_hours"] for node in trajectory_nodes] == [
        0.25,
        0.5,
        1.0,
        2.0,
        4.0,
    ]
    milestone_evaluations = [nodes[f"short_v1_{steps}"]["config"] for steps in (64, 128, 256)]
    assert all(settings["profile"] == "qwen35_vlm_short_v1" for settings in milestone_evaluations)
    assert {settings["row_manifest"] for settings in milestone_evaluations} == {
        str(tmp_path / "short-v1.json")
    }
    assert {settings["row_manifest_sha256"] for settings in milestone_evaluations} == {"a" * 64}
    assert {settings["reference_checkpoint"] for settings in milestone_evaluations} == {
        config["teacher_dir"]
    }
    assert all(settings["reference_once"] is True for settings in milestone_evaluations)
    assert all(
        settings["limit_mm_per_prompt"] == {"image": 32}
        for settings in (
            nodes["pre_kd_short_v1"]["config"],
            *milestone_evaluations,
            config["post_mip"]["flows"]["control-learning-curve"]["nodes"][
                "control_pre_kd_short_v1"
            ]["config"],
        )
    )
    assert all(
        settings["max_model_len"] == 65536
        for settings in (
            nodes["pre_kd_short_v1"]["config"],
            *milestone_evaluations,
            config["post_mip"]["flows"]["control-learning-curve"]["nodes"][
                "control_pre_kd_short_v1"
            ]["config"],
        )
    )
    assert {settings["reference_cache_id"] for settings in milestone_evaluations} == {
        "qwen35-0p8b-short-v1-teacher"
    }
    assert nodes["selected"]["type"] == "manual_filter"
    assert nodes["selected"]["input"] == "short_v1_256"
    assert nodes["approve_512"]["type"] == "manual_filter"
    assert nodes["approve_512"]["input"] == "bounded_result"
    assert nodes["approve_1024"]["type"] == "manual_filter"
    assert nodes["approve_1024"]["input"] == "short_v1_512"
    assert nodes["bounded_result"]["config"] == {
        "pre_kd_source": "materialized",
        "pre_kd_evaluation": "pre_kd_short_v1",
        "profile": "qwen35_vlm_short_v1",
        "row_manifest": str(tmp_path / "short-v1.json"),
        "row_manifest_sha256": "a" * 64,
        "reference_checkpoint": config["teacher_dir"],
        "reference_cache_id": "qwen35-0p8b-short-v1-teacher",
        "milestones": [
            {"steps": 64, "kd": "kd_64", "evaluation": "short_v1_64"},
            {"steps": 128, "kd": "kd_128", "evaluation": "short_v1_128"},
            {"steps": 256, "kd": "kd_256", "evaluation": "short_v1_256"},
        ],
    }
    control_nodes = config["post_mip"]["flows"]["control-learning-curve"]["nodes"]
    assert config["post_mip"]["flows"]["control-learning-curve"]["source"]["run"] == (
        "calibration-controls"
    )
    assert [
        control_nodes[f"control_kd_{steps}"]["config"]["max_steps"] for steps in (64, 128, 256)
    ] == [
        64,
        128,
        256,
    ]
    assert {control_nodes[f"control_kd_{steps}"]["trajectory"] for steps in (64, 128, 256)} == {
        "conservative-control-learning-curve"
    }
    assert all(
        control_nodes[f"control_kd_{steps}"]["failure_policy"] == "strict"
        for steps in (64, 128, 256)
    )
    assert control_nodes["control_result"]["config"]["milestones"] == [
        {"steps": 64, "kd": "control_kd_64", "evaluation": "control_short_v1_64"},
        {"steps": 128, "kd": "control_kd_128", "evaluation": "control_short_v1_128"},
        {"steps": 256, "kd": "control_kd_256", "evaluation": "control_short_v1_256"},
    ]
    assert control_nodes["control_selected"]["input"] == "control_result"
    assert control_nodes["control_approve_512"]["input"] == "control_selected"
    assert control_nodes["control_kd_512"]["input"] == "control_approve_512"
    assert control_nodes["control_kd_512"]["trajectory"] == ("conservative-control-learning-curve")
    assert control_nodes["control_kd_512"]["config"]["max_steps"] == 512
    assert control_nodes["control_kd_512"]["exposure"]["cumulative_examples"] == 2048
    assert control_nodes["control_extended_result"]["config"]["milestones"][-1] == {
        "steps": 512,
        "kd": "control_kd_512",
        "evaluation": "control_short_v1_512",
    }
    assert control_nodes["control_approve_1024"]["input"] == "control_extended_result"
    assert control_nodes["control_kd_1024"]["input"] == "control_approve_1024"
    assert control_nodes["control_kd_1024"]["config"]["max_steps"] == 1024
    assert control_nodes["control_kd_1024"]["exposure"]["cumulative_examples"] == 4096
    assert control_nodes["control_pre_kd_short_v1"]["input"] == "control_materialized"
    assert control_nodes["control_kd_64"]["input"] == "control_pre_kd_short_v1"
    comparison_benchmark = comparison_nodes["quality_benchmarks"]
    assert comparison_benchmark["input"] == "short_vlm_kd"
    assert comparison_benchmark["failure_policy"] == "strict"
    assert (
        comparison_benchmark["config"]["reference_checkpoint"] == comparison_config["teacher_dir"]
    )
    campaign_post_stages = tuple(
        stage.stage_id for stage in campaign.stages if stage.stage_id.startswith("post.")
    )
    assert campaign_post_stages[:8] == (
        "post.candidate-evaluation.online_eval",
        "post.candidate-evaluation.best_lm",
        "post.candidate-evaluation.materialized",
        "post.candidate-evaluation.serving",
        "post.candidate-evaluation.pre_kd_short_v1",
        "post.candidate-evaluation.kd_64",
        "post.candidate-evaluation.short_v1_64",
        "post.candidate-evaluation.kd_128",
    )
    assert campaign_post_stages[8:12] == (
        "post.candidate-evaluation.short_v1_128",
        "post.candidate-evaluation.kd_256",
        "post.candidate-evaluation.short_v1_256",
        "post.candidate-evaluation.selected",
    )
    assert all(stage.total_gpus in {0, 1} for stage in (*campaign.stages, *comparison.stages))


def test_qwen3p5_0p8b_vlm_campaign_execution_names_every_learning_curve_stage(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("PUZZLETRON_RUN_ROOT", str(tmp_path / "campaign"))
    monkeypatch.setenv("PUZZLETRON_DATASET_PATH", str(tmp_path / "dataset"))
    monkeypatch.setenv("PUZZLETRON_DATASET_REVISION", "fixture-revision")
    monkeypatch.setenv("PUZZLETRON_VLM_SHORT_V1_MANIFEST", str(tmp_path / "short-v1.json"))
    monkeypatch.setenv("PUZZLETRON_VLM_SHORT_V1_SHA256", "a" * 64)
    execution = load_execution_config(CAMPAIGN_EXECUTION_PATH)
    campaign = compile_campaign_plan(
        experiment_config_path=CAMPAIGN_PATH,
        runner=load_runner_config(RUNNER_PATH),
        execution=execution,
        stage_filter="full",
    )
    stages = {stage.stage_id: stage for stage in campaign.stages}
    configured = set(execution["stages"])
    compiled = set(stages)

    assert configured <= compiled
    assert not any(
        stale in stage_id
        for stage_id in configured
        for stale in ("screening_kd", "screening_eval", "quality_screen")
    )
    for prefix in ("post.candidate-evaluation", "post.control-learning-curve"):
        for steps in (64, 128, 256):
            kd = stages[f"{prefix}.{'control_' if 'control' in prefix else ''}kd_{steps}"]
            evaluation = stages[
                f"{prefix}.{'control_' if 'control' in prefix else ''}short_v1_{steps}"
            ]
            assert kd.total_gpus == 2
            assert evaluation.total_gpus == 2
            assert kd.gpus_per_instance == evaluation.gpus_per_instance == 1
    for steps in (512, 1024):
        kd = stages[f"post.control-learning-curve.control_kd_{steps}"]
        evaluation = stages[f"post.control-learning-curve.control_short_v1_{steps}"]
        assert kd.total_gpus == evaluation.total_gpus == 1
        assert kd.gpus_per_instance == evaluation.gpus_per_instance == 1
