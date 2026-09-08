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

"""CPU contracts for the maintained Qwen 3.5 4B VLM example."""

from itertools import pairwise
from pathlib import Path

import yaml

from puzzletron_orchestrator.compiler import (
    compile_campaign_plan,
    load_execution_config,
    load_runner_config,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
FAMILY_ROOT = REPOSITORY_ROOT / "examples/puzzletron/configs/families/qwen3_5/qwen3p5_4b"
MODEL_PATH = FAMILY_ROOT / "model.yaml"
MIP_RUN_PATH = FAMILY_ROOT / "runs/mip_vlm_smoke.yaml"
LIFECYCLE_RUN_PATH = FAMILY_ROOT / "runs/vlm_lifecycle_smoke.yaml"
LEGACY_FULL_RUN_PATH = FAMILY_ROOT / "runs/full_vlm_smoke.yaml"
CAMPAIGN_RUN_PATH = FAMILY_ROOT / "runs/ffn_width_10to20pct_kd_search.yaml"
LEGACY_CAMPAIGN_RUN_PATH = FAMILY_ROOT / "runs/vlm_campaign.yaml"
RUNNER_PATH = (
    REPOSITORY_ROOT / "examples/puzzletron/configs/orchestration/qwen3p5_4b/runner.slurm.yaml"
)
EXECUTION_PATH = (
    REPOSITORY_ROOT / "examples/puzzletron/configs/orchestration/execution.single_gpu.yaml"
)
LIFECYCLE_EXECUTION_PATH = (
    REPOSITORY_ROOT
    / "examples/puzzletron/configs/orchestration/qwen3p5_4b/execution.vlm_lifecycle_smoke.yaml"
)
LEGACY_FULL_EXECUTION_PATH = (
    REPOSITORY_ROOT
    / "examples/puzzletron/configs/orchestration/qwen3p5_4b/execution.full_vlm_smoke.yaml"
)
CAMPAIGN_EXECUTION_PATH = REPOSITORY_ROOT / (
    "examples/puzzletron/configs/orchestration/qwen3p5_4b/"
    "execution.ffn_width_10to20pct_kd_search.yaml"
)
LEGACY_CAMPAIGN_EXECUTION_PATH = (
    REPOSITORY_ROOT / "examples/puzzletron/configs/orchestration/qwen3p5_4b/execution.campaign.yaml"
)


def _compile_plan(
    monkeypatch,
    tmp_path: Path,
    run_path: Path,
    execution_path=EXECUTION_PATH,
    run_root_name: str | None = None,
):
    monkeypatch.setenv("PUZZLETRON_RUN_ROOT", str(tmp_path / (run_root_name or run_path.stem)))
    monkeypatch.setenv("PUZZLETRON_DATASET_PATH", str(tmp_path / "dataset"))
    monkeypatch.setenv("PUZZLETRON_DATASET_REVISION", "fixture-revision")
    return compile_campaign_plan(
        experiment_config_path=run_path,
        runner=load_runner_config(RUNNER_PATH),
        execution=load_execution_config(execution_path),
        stage_filter="full",
    )


def test_qwen3p5_4b_model_pins_the_bounded_ffn_grid() -> None:
    model = yaml.safe_load(MODEL_PATH.read_text())

    assert model["input_hf_model_path"] == "Qwen/Qwen3.5-4B"
    assert model["model_info"]["hf_revision"] == ("851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a")
    assert model["model_info"]["model_type"] == "qwen3_5"
    assert model["model_info"]["layer_counts"] == {
        "linear_attention": 24,
        "full_attention": 8,
    }
    widths = [8704, 8192, 7168, 6144, 5632, 5120, 4608]
    assert model["pruning"] == {"intermediate_size_list": widths}


def test_qwen3p5_4b_default_compiles_the_complete_ffn_grid_and_stops_at_mip(
    monkeypatch,
    tmp_path: Path,
) -> None:
    plan = _compile_plan(monkeypatch, tmp_path, MIP_RUN_PATH)
    config = plan.experiment_config
    stage_ids = tuple(stage.stage_id for stage in plan.stages)

    assert stage_ids == (
        "convert",
        "width_importance",
        "sort",
        "sort_sanity",
        "width_sanity",
        "slicing_sanity",
        "build_library",
        "replacement_scoring",
        "mip",
    )
    assert config["vllm_stats"]["enabled"] is False
    runs = config["mip"]["runs"]
    assert set(runs) == {"params-80", "memory-85"}
    assert all(run["solver"]["num_solutions"] == 3 for run in runs.values())
    assert all(run["homogeneous"]["keep"] == 2 for run in runs.values())
    assert config["post_mip"]["flows"] == {}


def test_qwen3p5_4b_opt_in_lifecycle_materializes_reloads_and_bounds_kd_and_evaluation(
    monkeypatch,
    tmp_path: Path,
) -> None:
    plan = _compile_plan(
        monkeypatch,
        tmp_path,
        LIFECYCLE_RUN_PATH,
        execution_path=LIFECYCLE_EXECUTION_PATH,
    )
    config = plan.experiment_config
    post_stages = tuple(stage for stage in plan.stages if stage.stage_id.startswith("post."))
    nodes = config["post_mip"]["flows"]["params-80"]["nodes"]

    assert tuple(stage.stage_id for stage in post_stages) == (
        "post.params-80.image_eval",
        "post.params-80.best_vlm_loss",
        "post.params-80.materialized",
        "post.params-80.checkpoint_eval",
        "post.params-80.short_vlm_kd",
        "post.params-80.post_kd_checkpoint_eval",
        "post.params-80.final_image_eval",
        "post.params-80.best",
    )
    assert post_stages[0].parents == ("mip",)
    for parent, stage in pairwise(post_stages):
        assert stage.parents == (parent.stage_id,)
    assert nodes["materialized"]["input"] == "best_vlm_loss"
    assert nodes["checkpoint_eval"]["input"] == "materialized"
    assert nodes["checkpoint_eval"]["failure_policy"] == "strict"
    kd = nodes["short_vlm_kd"]
    assert kd["input"] == "checkpoint_eval"
    assert kd["config"]["automodel"]["parallel"]["tp"] == 2
    assert (
        next(stage for stage in plan.stages if stage.stage_id.endswith("short_vlm_kd")).total_gpus
        == 2
    )


def test_qwen3p5_4b_legacy_full_smoke_alias_resolves_to_lifecycle(
    monkeypatch,
    tmp_path: Path,
) -> None:
    lifecycle_plan = _compile_plan(
        monkeypatch,
        tmp_path,
        LIFECYCLE_RUN_PATH,
        execution_path=LIFECYCLE_EXECUTION_PATH,
        run_root_name="lifecycle-alias-contract",
    )
    legacy_plan = _compile_plan(
        monkeypatch,
        tmp_path,
        LEGACY_FULL_RUN_PATH,
        execution_path=LIFECYCLE_EXECUTION_PATH,
        run_root_name="lifecycle-alias-contract",
    )

    assert legacy_plan.experiment_config["post_mip"] == lifecycle_plan.experiment_config["post_mip"]
    assert (
        legacy_plan.experiment_config["global_distillation"]
        == (lifecycle_plan.experiment_config["global_distillation"])
    )
    assert load_execution_config(LEGACY_FULL_EXECUTION_PATH) == load_execution_config(
        LIFECYCLE_EXECUTION_PATH
    )


def test_qwen3p5_4b_campaign_compares_pruning_bands_and_teacher(monkeypatch, tmp_path) -> None:
    plan = _compile_plan(
        monkeypatch,
        tmp_path,
        CAMPAIGN_RUN_PATH,
        execution_path=CAMPAIGN_EXECUTION_PATH,
    )
    config = plan.experiment_config
    candidates = config["mip"]["runs"]["ffn-candidates"]
    nodes = config["post_mip"]["flows"]["candidate-evaluation"]["nodes"]

    assert tuple(config["mip"]["runs"]) == ("params-80", "memory-85", "ffn-candidates")
    assert config["mip"]["runs"]["params-80"] is False
    assert config["mip"]["runs"]["memory-85"] is False
    assert set(candidates["variants"]) == {"width-7168", "width-6144", "width-5120"}
    assert nodes["quality_benchmarks"]["config"]["reference_checkpoint"] == config["teacher_dir"]
    assert nodes["global_kd"]["model_source"] == "materialized"
    assert tuple(stage.stage_id for stage in plan.stages)[-11:-1] == (
        "post.candidate-evaluation.online_eval",
        "post.candidate-evaluation.materialized",
        "post.candidate-evaluation.serving",
        "post.candidate-evaluation.screening_kd",
        "post.candidate-evaluation.screening_eval",
        "post.candidate-evaluation.quality_screen",
        "post.candidate-evaluation.selected",
        "post.candidate-evaluation.global_kd",
        "post.candidate-evaluation.final_eval",
        "post.candidate-evaluation.quality_benchmarks",
    )
    assert tuple(stage.stage_id for stage in plan.stages)[-1] == "post.candidate-evaluation.best"
    stages = {stage.stage_id: stage for stage in plan.stages}
    candidate_stages = {
        "post.candidate-evaluation.online_eval",
        "post.candidate-evaluation.materialized",
        "post.candidate-evaluation.serving",
        "post.candidate-evaluation.screening_kd",
        "post.candidate-evaluation.screening_eval",
        "post.candidate-evaluation.quality_screen",
    }
    assert all(stages[stage_id].instances == 4 for stage_id in candidate_stages)
    assert stages["post.candidate-evaluation.screening_kd"].total_gpus == 8
    assert all(stages[stage_id].gpus_per_node == 8 for stage_id in candidate_stages)
    assert stages["post.candidate-evaluation.global_kd"].total_gpus == 2


def test_qwen3p5_4b_legacy_campaign_alias_resolves_to_named_search(
    monkeypatch,
    tmp_path: Path,
) -> None:
    campaign_plan = _compile_plan(
        monkeypatch,
        tmp_path,
        CAMPAIGN_RUN_PATH,
        execution_path=CAMPAIGN_EXECUTION_PATH,
        run_root_name="campaign-alias-contract",
    )
    legacy_plan = _compile_plan(
        monkeypatch,
        tmp_path,
        LEGACY_CAMPAIGN_RUN_PATH,
        execution_path=CAMPAIGN_EXECUTION_PATH,
        run_root_name="campaign-alias-contract",
    )

    assert legacy_plan.experiment_config["mip"] == campaign_plan.experiment_config["mip"]
    assert legacy_plan.experiment_config["post_mip"] == campaign_plan.experiment_config["post_mip"]
    assert load_execution_config(LEGACY_CAMPAIGN_EXECUTION_PATH) == load_execution_config(
        CAMPAIGN_EXECUTION_PATH
    )
