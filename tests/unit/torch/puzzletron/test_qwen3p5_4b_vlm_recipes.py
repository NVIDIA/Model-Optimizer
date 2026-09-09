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

"""CPU contracts for the maintained Qwen 3.5 4B VLM recipes."""

from itertools import pairwise
from pathlib import Path

import yaml

from modelopt.torch.puzzletron.mip.profiles import normalize_mip_profiles
from puzzletron_orchestrator.compiler import (
    compile_campaign_plan,
    load_execution_config,
    load_runner_config,
)
from puzzletron_orchestrator.recipe_config import (
    materialize_resolved_bundle,
    resolve_recipe_run,
    site_template,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
FAMILY_ROOT = REPOSITORY_ROOT / "examples/puzzletron/configs/families/qwen3_5/qwen3p5_4b"
MODEL_PATH = FAMILY_ROOT / "model.yaml"
SMOKE_RECIPE_PATH = (
    REPOSITORY_ROOT / "examples/puzzletron/configs/recipes/qwen3p5_4b_vlm_smoke.yaml"
)
CAMPAIGN_RECIPE_PATH = (
    REPOSITORY_ROOT / "examples/puzzletron/configs/recipes/qwen3p5_4b_vlm_campaign.yaml"
)

ALL_AXIS_DOMAINS = {
    "hidden_width": {"enabled": True, "teacher_value": 2560, "values": [2400]},
    "kv_groups": {"enabled": True, "teacher_value": 4, "values": [2]},
    "q_heads_per_group": {"enabled": True, "teacher_value": 4, "values": [3]},
    "ffn_intermediate": {"enabled": True, "teacher_value": 9216, "values": [8704]},
    "gdn_key_groups": {"enabled": True, "teacher_value": 16, "values": [14]},
    "gdn_value_heads_per_group": {
        "enabled": True,
        "teacher_value": 2,
        "values": [1],
    },
    "gdn_key_head_dim": {"enabled": True, "teacher_value": 128, "values": [112]},
    "gdn_value_head_dim": {"enabled": True, "teacher_value": 128, "values": [112]},
}


def _compile_plan(tmp_path: Path, recipe_source: Path):
    run_root = tmp_path / recipe_source.stem
    dataset = tmp_path / "dataset"
    recipe = yaml.safe_load(recipe_source.read_text())
    recipe["run_root"] = str(run_root)
    recipe["resource_profile"] = "selected"
    recipe["data"] = {"path": str(dataset), "revision": "fixture-revision"}
    recipe_path = tmp_path / f"{run_root.name}.recipe.yaml"
    recipe_path.write_text(yaml.safe_dump(recipe, sort_keys=False))
    site = site_template()
    site["site"]["environment"].update({"repository": str(REPOSITORY_ROOT), "venv": ".venv"})
    site["site"]["paths"]["hf_home"] = str(tmp_path / "hf")
    site["site"]["slurm"].update({"account": "test", "partition": "test"})
    site["resources"]["selected"] = {
        "mode": "per_attempt",
        "gpus_per_node": 8,
        "max_nodes": 1,
    }
    site_path = tmp_path / "site.yaml"
    site_path.write_text(yaml.safe_dump(site, sort_keys=False))
    bundle = materialize_resolved_bundle(resolve_recipe_run(recipe_path, site_path), activate=False)
    return compile_campaign_plan(
        experiment_config_path=bundle / "experiment.runtime.yaml",
        runner=load_runner_config(bundle / "runner.yaml"),
        execution=load_execution_config(bundle / "execution.yaml"),
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


def test_qwen3p5_4b_smoke_covers_all_axes_and_emits_comparable_results(tmp_path: Path) -> None:
    plan = _compile_plan(tmp_path, SMOKE_RECIPE_PATH)
    config = plan.experiment_config
    nodes = config["post_mip"]["flows"]["params-80"]["nodes"]
    post_stages = tuple(stage for stage in plan.stages if stage.stage_id.startswith("post."))

    assert config["search_space"]["axes"] == ALL_AXIS_DOMAINS
    assert config["width_sanity"]["axes"] == list(ALL_AXIS_DOMAINS)
    assert config["depth_importance"]["max_removals"] == 1
    profiles = normalize_mip_profiles(
        config["mip"], available_depths=[0, 1], available_embeddings=[2560, 2400]
    )
    assert len(profiles) == 1
    assert (
        sum(
            len(profile.embedding_widths)
            * len(profile.depth_selections)
            * profile.solver.num_solutions
            for profile in profiles
        )
        == 4
    )
    assert tuple(stage.stage_id for stage in post_stages) == (
        "post.params-80.image_eval",
        "post.params-80.best_vlm_loss",
        "post.params-80.materialized",
        "post.params-80.checkpoint_eval",
        "post.params-80.serving_smoke",
        "post.params-80.short_vlm_kd",
        "post.params-80.post_kd_checkpoint_eval",
        "post.params-80.result",
        "post.params-80.final_serving_smoke",
        "post.params-80.final_image_eval",
        "post.params-80.best",
    )
    for parent, stage in pairwise(post_stages[:7]):
        assert stage.parents == (parent.stage_id,)
    result_stage = next(stage for stage in post_stages if stage.stage_id.endswith(".result"))
    assert set(result_stage.parents) == {
        "post.params-80.materialized",
        "post.params-80.checkpoint_eval",
        "post.params-80.short_vlm_kd",
        "post.params-80.post_kd_checkpoint_eval",
    }
    assert nodes["checkpoint_eval"]["config"]["profile"] == "qwen35_vlm_core3_24row_smoke_v2"
    assert nodes["checkpoint_eval"]["config"] == nodes["post_kd_checkpoint_eval"]["config"]
    assert nodes["serving_smoke"]["config"] == nodes["final_serving_smoke"]["config"]
    assert nodes["short_vlm_kd"]["config"]["max_steps"] == 2
    assert nodes["short_vlm_kd"]["config"]["automodel"]["parallel"]["tp"] == 2
    assert nodes["result"]["config"]["milestones"] == [
        {"steps": 2, "kd": "short_vlm_kd", "evaluation": "post_kd_checkpoint_eval"}
    ]
    stages = {stage.stage_id: stage for stage in plan.stages}
    assert stages["post.params-80.image_eval"].instances == 2
    assert stages["post.params-80.short_vlm_kd"].total_gpus == 2
    assert plan.final_report_partition == plan.runner.slurm.partition_cpu


def test_qwen3p5_4b_campaign_runs_exactly_four_candidates_through_matched_kd128(
    tmp_path: Path,
) -> None:
    plan = _compile_plan(tmp_path, CAMPAIGN_RECIPE_PATH)
    config = plan.experiment_config
    nodes = config["post_mip"]["flows"]["candidate-evaluation"]["nodes"]

    assert config["embedding_pruning"]["widths"] == [2560, 2400, 2240]
    assert set(config["search_space"]["axes"]) == set(ALL_AXIS_DOMAINS)
    assert config["depth_importance"]["max_removals"] == 2
    profiles = normalize_mip_profiles(
        config["mip"], available_depths=[0, 1, 2], available_embeddings=[2560, 2400, 2240]
    )
    search_profiles = [profile for profile in profiles if profile.run_id == "search-candidates"]
    assert {profile.variant_id for profile in search_profiles} == {"params-82", "memory-85"}
    assert (
        sum(
            len(profile.embedding_widths)
            * len(profile.depth_selections)
            * profile.solver.num_solutions
            for profile in search_profiles
        )
        == 18
    )
    assert nodes["selected"] == {
        "type": "filter",
        "input": "online_eval",
        "mode": "top_k",
        "metric": "online_eval.lm_loss",
        "direction": "minimize",
        "top_k": 4,
        "require_exact_count": True,
    }
    kd = nodes["kd_128"]
    assert kd["input"] == "serving_smoke"
    assert kd["model_source"] == "materialized"
    assert kd["config"]["resume"] is True
    assert kd["config"]["max_steps"] == 128
    assert kd["config"]["global_batch_size"] == 4
    assert kd["config"]["checkpoint_every_steps"] == 128
    assert kd["exposure"]["cumulative_examples"] == 512
    assert nodes["pre_kd_quality"]["config"] == nodes["quality_benchmarks"]["config"]
    assert nodes["pre_kd_quality"]["config"]["profile"] == (
        "qwen35_vlm_realworldqa64_mmmu120_mvbench160_frozen_rows_v3"
    )
    assert nodes["comparison_ready"]["config"]["milestones"] == [
        {"steps": 128, "kd": "kd_128", "evaluation": "quality_benchmarks"}
    ]
    assert [entry["weight"] for entry in nodes["best"]["metrics"]] == [100, 100, 1]
    stages = {stage.stage_id: stage for stage in plan.stages}
    assert stages["replacement_scoring"].instances == 8
    assert stages["post.candidate-evaluation.online_eval"].instances == 8
    four_candidate_stages = {
        "post.candidate-evaluation.materialized",
        "post.candidate-evaluation.pre_kd_quality",
        "post.candidate-evaluation.serving_smoke",
        "post.candidate-evaluation.kd_128",
        "post.candidate-evaluation.final_eval",
        "post.candidate-evaluation.quality_benchmarks",
        "post.candidate-evaluation.student_performance",
    }
    assert all(stages[stage_id].instances == 4 for stage_id in four_candidate_stages)
    assert stages["post.candidate-evaluation.kd_128"].total_gpus == 8
    assert all(
        stages[stage_id].total_gpus == 4
        for stage_id in four_candidate_stages - {"post.candidate-evaluation.kd_128"}
    )
    assert plan.runner.slurm.max_nodes == 1
    assert all(stages[stage_id].nodes <= 1 for stage_id in four_candidate_stages)
    performance = nodes["student_performance"]["config"]
    assert performance["repetitions"] == 3
    assert performance["image_batch_sizes"] == [1, 6, 12]
    assert performance["concurrency"] == [1, 4]
    teacher_nodes = config["post_mip"]["flows"]["teacher-performance"]["nodes"]
    assert teacher_nodes["teacher_performance"]["config"] == performance
    assert plan.final_report_partition == plan.runner.slurm.partition_cpu
