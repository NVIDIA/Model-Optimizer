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
CAMPAIGN_PATH = FAMILY_ROOT / "runs/campaign.yaml"
RUNNER_PATH = (
    REPOSITORY_ROOT / "examples/puzzletron/configs/orchestration/qwen3p5_0p8b/runner.slurm.yaml"
)
SINGLE_GPU_EXECUTION_PATH = (
    REPOSITORY_ROOT / "examples/puzzletron/configs/orchestration/execution.single_gpu.yaml"
)
CAMPAIGN_EXECUTION_PATH = (
    REPOSITORY_ROOT
    / "examples/puzzletron/configs/orchestration/qwen3p5_0p8b/execution.campaign.yaml"
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


def test_campaign_keeps_a_matched_ffn_only_comparison(monkeypatch, tmp_path: Path) -> None:
    plan = _compile(monkeypatch, tmp_path, CAMPAIGN_PATH, CAMPAIGN_EXECUTION_PATH)
    config = plan.experiment_config
    nodes = config["post_mip"]["flows"]["candidate-evaluation"]["nodes"]

    assert config["search_space"]["axes"] == {
        "ffn_intermediate": {
            "enabled": True,
            "teacher_value": 3584,
            "values": [3328, 3072],
        }
    }
    assert set(config["mip"]["runs"]) == {"params-90", "ffn-candidates"}
    assert config["mip"]["runs"]["params-90"] is False
    assert set(config["mip"]["runs"]["ffn-candidates"]["variants"]) == {
        "width-3328",
        "width-3072",
    }
    assert nodes["screening_kd"]["config"]["max_steps"] == 128
    assert nodes["global_kd"]["config"]["max_steps"] == 256
    assert "reference_checkpoint" not in nodes["quality_screen"]["config"]
    assert nodes["quality_benchmarks"]["config"]["reference_checkpoint"] == config["teacher_dir"]
    assert nodes["selected"]["input"] == "quality_screen"
    assert nodes["selected"]["top_k"] == 1
    assert "post.candidate-evaluation.quality_benchmarks" in {
        stage.stage_id for stage in plan.stages
    }
