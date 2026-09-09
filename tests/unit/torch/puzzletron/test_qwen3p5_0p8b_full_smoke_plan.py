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

"""CPU contracts for the maintained Qwen 3.5 0.8B text examples."""

from pathlib import Path

import yaml

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
RECIPE_PATH = REPOSITORY_ROOT / "examples/puzzletron/configs/recipes/qwen3p5_0p8b_text_smoke.yaml"


def _compile(tmp_path: Path):
    recipe = yaml.safe_load(RECIPE_PATH.read_text())
    recipe["run_root"] = str(tmp_path / "run")
    recipe["data"] = {
        "path": str(tmp_path / "prepared-data"),
        "revision": "fixture-revision",
    }
    recipe_path = tmp_path / "recipe.yaml"
    recipe_path.write_text(yaml.safe_dump(recipe, sort_keys=False))
    site = site_template()
    site["site"]["environment"].update({"repository": str(REPOSITORY_ROOT), "venv": ".venv"})
    site["site"]["paths"]["hf_home"] = str(tmp_path / "hf")
    site["site"]["slurm"].update({"account": "test", "partition": "test"})
    site_path = tmp_path / "site.yaml"
    site_path.write_text(yaml.safe_dump(site, sort_keys=False))
    bundle = materialize_resolved_bundle(resolve_recipe_run(recipe_path, site_path), activate=False)
    return compile_campaign_plan(
        experiment_config_path=bundle / "experiment.runtime.yaml",
        runner=load_runner_config(bundle / "runner.yaml"),
        execution=load_execution_config(bundle / "execution.yaml"),
        stage_filter="full",
    )


def test_full_smoke_compiles_one_complete_bounded_lifecycle(tmp_path: Path) -> None:
    plan = _compile(tmp_path)
    stages = {stage.stage_id: stage for stage in plan.stages}
    nodes = plan.experiment_config["post_mip"]["flows"]["params-90"]["nodes"]

    assert plan.experiment_config["dataset_path"] == str(tmp_path / "prepared-data")
    assert plan.experiment_config["data"]["revision"] == "fixture-revision"
    assert tuple(node for node in stages if node.startswith("post.")) == (
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
    assert nodes["post_kd_checkpoint_eval"]["config"] == nodes["checkpoint_eval"]["config"]
    cpu_stages = [stage for stage in stages.values() if stage.resource == "cpu"]
    assert cpu_stages
    assert all(stage.total_gpus == 0 for stage in cpu_stages)
    assert all(stage.total_gpus == 1 for stage in stages.values() if stage.resource != "cpu")
