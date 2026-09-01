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
FULL_RUN_PATH = FAMILY_ROOT / "runs/full_vlm_smoke.yaml"
CAMPAIGN_RUN_PATH = FAMILY_ROOT / "runs/vlm_campaign.yaml"
RUNNER_PATH = (
    REPOSITORY_ROOT / "examples/puzzletron/configs/orchestration/qwen3p5_4b/runner.slurm.yaml"
)
EXECUTION_PATH = (
    REPOSITORY_ROOT / "examples/puzzletron/configs/orchestration/execution.single_gpu.yaml"
)
CAMPAIGN_EXECUTION_PATH = (
    REPOSITORY_ROOT / "examples/puzzletron/configs/orchestration/qwen3p5_4b/execution.campaign.yaml"
)


def _compile_plan(monkeypatch, tmp_path: Path, run_path: Path, execution_path=EXECUTION_PATH):
    monkeypatch.setenv("PUZZLETRON_RUN_ROOT", str(tmp_path / run_path.stem))
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
    assert model["model_info"] == {
        "hf_repo": model["input_hf_model_path"],
        "hf_revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        "model_type": "qwen3_5",
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "num_hidden_layers": 32,
        "hidden_size": 2560,
        "intermediate_size": 9216,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "head_dim": 256,
        "vocab_size": 248320,
        "tie_word_embeddings": True,
        "max_position_embeddings": 262144,
        "mtp_num_hidden_layers": 1,
        "layer_counts": {"linear_attention": 24, "full_attention": 8},
        "mamba": {
            "linear_key_head_dim": 128,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 32,
            "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4,
        },
    }
    widths = [8704, 8192, 7168, 6144, 5632, 5120, 4608]
    assert model["pruning"] == {"intermediate_size_list": widths}
    assert model["search_space"]["axes"] == {
        "ffn_intermediate": {
            "enabled": True,
            "teacher_value": 9216,
            "values": widths,
        }
    }


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
    assert all(stage.total_gpus == 1 for stage in plan.stages)
    assert config["model"]["source"] == "Qwen/Qwen3.5-4B"
    assert config["model"]["revision"] == "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
    assert config["data"]["processor_identity"] == (
        "Qwen/Qwen3.5-4B@851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
    )
    assert config["search_space"]["axes"]["ffn_intermediate"] == {
        "enabled": True,
        "teacher_value": 9216,
        "values": [8704, 8192, 7168, 6144, 5632, 5120, 4608],
    }
    assert config["vllm_stats"]["enabled"] is False
    assert config["vllm_stats"]["runtime_stats"]["enabled"] is False
    assert config["mip"]["runs"] == {
        "params-80": {
            "constraints": {"params": {"max": "82%"}},
            "objectives": [
                {
                    "metric": "metrics.cosine_embedding_loss_hidden_states",
                    "direction": "minimize",
                }
            ],
            "search_space": {
                "depth": [0],
                "embedding": [2560],
                "axes_default": "teacher",
                "axes": {"ffn.intermediate_size": "all"},
            },
            "solver": {
                "backend": "auto",
                "num_solutions": 3,
                "min_hamming_distance": 2,
                "max_seconds_per_solution": 30,
            },
            "homogeneous": {"enabled": True, "keep": 2, "rank_by": "objective"},
        },
        "memory-85": {
            "constraints": {"memory": {"at": {"serving-default": {"max": "85%"}}}},
            "objectives": [
                {
                    "metric": "metrics.cosine_embedding_loss_hidden_states",
                    "direction": "minimize",
                }
            ],
            "search_space": {
                "depth": [0],
                "embedding": [2560],
                "axes_default": "teacher",
                "axes": {"ffn.intermediate_size": "all"},
            },
            "solver": {
                "backend": "auto",
                "num_solutions": 3,
                "min_hamming_distance": 2,
                "max_seconds_per_solution": 30,
            },
            "homogeneous": {"enabled": True, "keep": 2, "rank_by": "objective"},
        },
    }
    assert config["post_mip"]["flows"] == {}


def test_qwen3p5_4b_opt_in_lifecycle_materializes_reloads_and_bounds_kd_and_evaluation(
    monkeypatch,
    tmp_path: Path,
) -> None:
    plan = _compile_plan(monkeypatch, tmp_path, FULL_RUN_PATH)
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
    assert nodes["materialized"] == {"type": "materialize", "input": "best_vlm_loss"}
    assert nodes["checkpoint_eval"]["input"] == "materialized"
    assert nodes["checkpoint_eval"]["failure_policy"] == "strict"
    assert nodes["checkpoint_eval"]["config"]["profile"] == "qwen35_vlm_realworldqa"
    assert nodes["checkpoint_eval"]["config"]["batch_size"] == 1
    assert nodes["checkpoint_eval"]["config"]["timeout_seconds"] == 900
    assert nodes["checkpoint_eval"]["config"]["limit_mm_per_prompt"] == {"image": 1}
    assert nodes["short_vlm_kd"] == {
        "type": "global_kd",
        "input": "checkpoint_eval",
        "config": {
            "max_steps": 2,
            "global_batch_size": 1,
            "local_batch_size": 1,
            "checkpoint_every_steps": 2,
        },
    }
    assert nodes["post_kd_checkpoint_eval"]["config"] == nodes["checkpoint_eval"]["config"]
    assert nodes["final_image_eval"]["config"] == {"eval_samples": 2, "block_size": 512}
    assert all(stage.total_gpus == 1 for stage in plan.stages)


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

    assert candidates["variants"] == {
        "width-7168": {
            "constraints": {"params": {"max": "92%"}},
            "search_space": {"axes": {"ffn.intermediate_size": [7168]}},
        },
        "width-6144": {
            "constraints": {"params": {"max": "87%"}},
            "search_space": {"axes": {"ffn.intermediate_size": [6144]}},
        },
        "width-5120": {
            "constraints": {"params": {"max": "82%"}},
            "search_space": {"axes": {"ffn.intermediate_size": [5120]}},
        },
    }
    assert nodes["screening_kd"]["config"]["max_steps"] == 64
    assert nodes["global_kd"]["config"]["max_steps"] == 256
    assert nodes["quality_screen"]["config"]["profile"] == "qwen35_vlm_e2e_full_eval"
    assert "reference_checkpoint" not in nodes["quality_screen"]["config"]
    assert nodes["quality_benchmarks"]["config"]["reference_checkpoint"] == config["teacher_dir"]
    assert nodes["selected"]["top_k"] == 1
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
    assert all(stages[stage_id].total_gpus == 4 for stage_id in candidate_stages)
    assert stages["post.candidate-evaluation.global_kd"].total_gpus == 1
