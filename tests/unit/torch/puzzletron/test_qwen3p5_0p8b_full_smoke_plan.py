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

from puzzletron_orchestrator.compiler import (
    compile_campaign_plan,
    load_execution_config,
    load_runner_config,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
FAMILY_ROOT = REPOSITORY_ROOT / "examples/puzzletron/configs/families/qwen3_5/qwen3p5_0p8b"
RUN_PATH = FAMILY_ROOT / "runs/full_smoke.yaml"
RUNNER_PATH = (
    REPOSITORY_ROOT / "examples/puzzletron/configs/orchestration/qwen3p5_0p8b/runner.slurm.yaml"
)
SINGLE_GPU_EXECUTION_PATH = (
    REPOSITORY_ROOT / "examples/puzzletron/configs/orchestration/execution.single_gpu.yaml"
)


def _compile(monkeypatch, tmp_path: Path, experiment: Path, execution: Path):
    monkeypatch.setenv("PUZZLETRON_RUN_ROOT", str(tmp_path / experiment.stem))
    monkeypatch.setenv("PUZZLETRON_DATASET_PATH", str(tmp_path / "dataset"))
    return compile_campaign_plan(
        experiment_config_path=experiment,
        runner=load_runner_config(RUNNER_PATH),
        execution=load_execution_config(execution),
        stage_filter="full",
    )


def test_full_smoke_compiles_one_complete_bounded_lifecycle(monkeypatch, tmp_path: Path) -> None:
    plan = _compile(monkeypatch, tmp_path, RUN_PATH, SINGLE_GPU_EXECUTION_PATH)
    stages = {stage.stage_id: stage for stage in plan.stages}
    nodes = plan.experiment_config["post_mip"]["flows"]["params-90"]["nodes"]

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
    assert nodes["online_eval"]["config"]["eval_samples"] == 2
    assert nodes["checkpoint_eval"]["config"]["limit"] == 2
    assert nodes["serving"]["config"]["request_count"] == 4
    assert nodes["short_kd"]["config"]["max_steps"] == 2
    assert nodes["post_kd_checkpoint_eval"]["config"] == nodes["checkpoint_eval"]["config"]
    assert all(stage.total_gpus == 0 for stage in stages.values() if stage.resource == "cpu")
    assert all(stage.total_gpus == 1 for stage in stages.values() if stage.resource != "cpu")
