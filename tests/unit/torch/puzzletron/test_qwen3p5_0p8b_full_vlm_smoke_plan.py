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

"""CPU contracts for the maintained Qwen 3.5 0.8B VLM examples."""

from pathlib import Path

import yaml

from examples.puzzletron.evaluation.vlm import contracts, suites
from puzzletron_orchestrator.compiler import (
    compile_campaign_plan,
    load_execution_config,
    load_runner_config,
)
from puzzletron_orchestrator.public_config import (
    materialize_resolved_bundle,
    resolve_public_run,
    site_template,
)
from puzzletron_orchestrator.schema import ExecutionMode, ExecutionStrategy

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
SMOKE_RECIPE_PATH = (
    REPOSITORY_ROOT / "examples/puzzletron/configs/recipes/qwen3p5_0p8b_vlm_smoke.yaml"
)
CAMPAIGN_RECIPE_PATH = (
    REPOSITORY_ROOT / "examples/puzzletron/configs/recipes/qwen3p5_0p8b_vlm_campaign.yaml"
)


def _compile(tmp_path: Path, recipe_source: Path):
    recipe = yaml.safe_load(recipe_source.read_text())
    recipe["run_root"] = str(tmp_path / recipe_source.stem)
    recipe_path = tmp_path / recipe_source.name
    recipe_path.write_text(yaml.safe_dump(recipe, sort_keys=False))
    site = site_template()
    site["site"]["environment"].update({"repository": str(REPOSITORY_ROOT), "venv": ".venv"})
    site["site"]["paths"]["hf_home"] = str(tmp_path / "hf-home")
    site["site"]["slurm"].update({"account": "test", "partition": "test"})
    site_path = tmp_path / "site.yaml"
    site_path.write_text(yaml.safe_dump(site, sort_keys=False))
    bundle = materialize_resolved_bundle(resolve_public_run(recipe_path, site_path), activate=False)
    return compile_campaign_plan(
        experiment_config_path=bundle / "experiment.runtime.yaml",
        runner=load_runner_config(bundle / "runner.yaml"),
        execution=load_execution_config(bundle / "execution.yaml"),
        stage_filter="full",
    )


def test_full_vlm_smoke_compiles_one_complete_bounded_lifecycle(tmp_path: Path) -> None:
    plan = _compile(tmp_path, SMOKE_RECIPE_PATH)
    stages = {stage.stage_id: stage for stage in plan.stages}
    config = plan.experiment_config
    nodes = config["post_mip"]["flows"]["params-90"]["nodes"]

    assert plan.execution_mode is ExecutionMode.REUSABLE_ALLOCATION
    assert plan.execution_defaults["gpus_per_node"] == 2
    assert "tokenize_data" not in stages
    assert tuple(node for node in stages if node.startswith("post.")) == (
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
    assert nodes["post_kd_checkpoint_eval"]["config"] == nodes["checkpoint_eval"]["config"]
    assert nodes["fastest_vlm"]["metric"] == (
        "vlm_serving.images_12.concurrency_1.image_throughput"
    )
    cpu_stages = [stage for stage in stages.values() if stage.resource == "cpu"]
    assert cpu_stages
    assert all(stage.total_gpus == 0 for stage in cpu_stages)
    assert all(stage.total_gpus == 1 for stage in stages.values() if stage.resource != "cpu")


def test_vlm_campaign_compiles_the_multi_axis_flow(tmp_path: Path) -> None:
    plan = _compile(tmp_path, CAMPAIGN_RECIPE_PATH)
    stages = {stage.stage_id: stage for stage in plan.stages}
    config = plan.experiment_config
    candidates = config["post_mip"]["flows"]["candidates"]["nodes"]
    assert plan.execution_mode is ExecutionMode.REUSABLE_ALLOCATION
    assert plan.execution_defaults["gpus_per_node"] == 8
    assert stages["replacement_scoring"].strategy is ExecutionStrategy.PERSISTENT_POOL
    assert stages["replacement_scoring"].instances == 8
    assert stages["replacement_scoring"].total_gpus == 8
    assert candidates["pre_kd_eval"]["config"] == config["vlm_quality_evaluation"]
    assert candidates["pre_kd_eval"]["config"] == candidates["post_kd_eval"]["config"]
    assert stages["post.candidates.serving"].parents == ("post.candidates.result",)
    concurrent_candidate_stages = {
        "post.candidates.image_eval",
        "post.candidates.materialized",
        "post.candidates.pre_kd_eval",
        "post.candidates.kd",
        "post.candidates.post_kd_eval",
    }
    assert {
        stage_id for stage_id, stage in stages.items() if stage.total_gpus == 5
    } == concurrent_candidate_stages
    assert stages["post.candidates.serving"].total_gpus == 1

    profile_rows = contracts.load_profile("core-3_344-examples_r1-vllm").exact_rows
    assert profile_rows is not None
    assert candidates["pre_kd_eval"]["config"]["row_manifest_sha256"] == (
        suites.manifest_sha256(profile_rows)
    )
