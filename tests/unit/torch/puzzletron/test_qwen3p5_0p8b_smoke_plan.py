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

"""CPU plan contracts for the Qwen 3.5 0.8B MIP smoke wiring."""

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
    / "examples/puzzletron/configs/families/qwen3_5/qwen3p5_0p8b/runs/mip_smoke.yaml"
)
ORCHESTRATION_ROOT = REPOSITORY_ROOT / "examples/puzzletron/configs/orchestration/qwen3p5_0p8b"
RUNNER_PATH = ORCHESTRATION_ROOT / "runner.slurm.yaml"
EXECUTION_PATH = ORCHESTRATION_ROOT / "execution.smoke.yaml"


def _compile_plan(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("PUZZLETRON_RUN_ROOT", str(tmp_path / "results"))
    monkeypatch.setenv("PUZZLETRON_DATASET_PATH", str(tmp_path / "dataset"))
    return compile_campaign_plan(
        experiment_config_path=RUN_PATH,
        runner=load_runner_config(RUNNER_PATH),
        execution=load_execution_config(EXECUTION_PATH),
        stage_filter="full",
    )


def test_qwen3p5_0p8b_smoke_selects_the_default_model_choice() -> None:
    config = yaml.safe_load(RUN_PATH.read_text(encoding="utf-8"))

    assert config["defaults"] == [
        "/base@_global_",
        "/families/qwen3_5/family@_global_",
        "/families/qwen3_5/qwen3p5_0p8b/model@_global_",
        "_self_",
    ]


def test_qwen3p5_0p8b_run_uses_the_accepted_quick_smoke_budgets(
    monkeypatch, tmp_path: Path
) -> None:
    plan = _compile_plan(monkeypatch, tmp_path)
    config = plan.experiment_config
    caches = {cache["split"]: cache for cache in config["tokenize_data"]["caches"]}

    assert config["data"]["max_sample_length"] == 512
    assert caches["train"]["num_samples"] == config["pruning"]["eval_samples"] == 8
    assert caches["validation"]["num_samples"] == 2
    assert caches["train"]["seq_length"] == caches["validation"]["seq_length"] == 512
    assert config["replacement_scoring"]["eval_samples"] == 2
    assert config["mip"]["runs"]["params-90"]["solver"]["num_solutions"] == 1
    assert config["mip"]["canonical_solutions_path"].endswith(
        "/single_subblock_replacement_solutions.json"
    )
    assert config["mip"]["single_block_replacement_validation_dir"].endswith(
        "/single_subblock_replacement_solutions--validation"
    )
    assert config["pruning"]["intermediate_size_list"] == [3072, 2048]
    assert config["search_space"]["axes"] == {
        "ffn_intermediate": {
            "enabled": True,
            "teacher_value": 3584,
            "values": [3072, 2048],
        }
    }


def test_qwen3p5_0p8b_run_keeps_non_ffn_mip_dimensions_at_teacher(
    monkeypatch, tmp_path: Path
) -> None:
    config = _compile_plan(monkeypatch, tmp_path).experiment_config

    assert config["embedding_pruning"]["enabled"] is True
    assert config["embedding_pruning"]["widths"] == [1024]
    assert config["width_sanity"]["hidden_width_diagnostic"] is False
    assert config["width_sanity"]["axes"] == ["ffn_intermediate"]
    assert config["mip"]["runs"]["params-90"]["search_space"] == {
        "depth": [0],
        "embedding": [1024],
        "axes_default": "teacher",
        "axes": {"ffn.intermediate_size": "all"},
    }


def test_qwen3p5_0p8b_full_plan_has_mip_as_its_exact_terminal_stage(
    monkeypatch,
    tmp_path: Path,
) -> None:
    plan = _compile_plan(monkeypatch, tmp_path)
    config = plan.experiment_config
    stage_ids = [stage.stage_id for stage in plan.stages]

    assert stage_ids[-1] == "mip"
    assert {
        "convert",
        "tokenize_data",
        "width_importance",
        "sort",
        "sort_sanity",
        "width_sanity",
        "slicing_sanity",
        "build_library",
        "replacement_scoring",
        "mip",
    } == set(stage_ids)
    assert not {
        "vllm_stats",
        "bypass_sanity",
        "bypass",
        "zero_shot_evaluation",
        "aiperf",
        "global_distillation_sanity",
        "global_distillation",
        "post_distillation_evaluation",
    } & set(stage_ids)
    assert all(stage.total_gpus == 1 for stage in plan.stages)
    assert config["model"]["revision"] == "2fc06364715b967f1860aea9cf38778875588b17"
    assert config["build_library"]["include_bypass"] is False
    assert config["depth_importance"]["enabled"] is False
    assert config["depth_importance"]["max_subblocks_to_remove"] == 0
    assert config["width_sanity"]["axes"] == ["ffn_intermediate"]
    assert config["width_sanity"]["physical_realization"] is True


def test_qwen3p5_0p8b_runner_requires_an_explicit_site_contract() -> None:
    runner = load_runner_config(RUNNER_PATH)

    assert runner.slurm is not None
    assert runner.slurm.max_nodes == 1
    assert runner.slurm.time_limit == "1:00:00"
    assert runner.slurm.account.startswith("REPLACE_WITH_")
    assert runner.slurm.partition.startswith("REPLACE_WITH_")
    assert runner.contract.repository.startswith("REPLACE_WITH_")
    assert runner.contract.venv.startswith("REPLACE_WITH_")
    assert runner.contract.container is not None
    assert runner.contract.container.startswith("REPLACE_WITH_")
    assert runner.contract.container_mounts is not None
    assert runner.contract.container_mounts.startswith("REPLACE_WITH_")


def test_qwen3p5_0p8b_plan_uses_worker_visible_experiment_path(
    monkeypatch,
    tmp_path: Path,
) -> None:
    plan = _compile_plan(monkeypatch, tmp_path)
    expected = Path(plan.runner.contract.repository) / RUN_PATH.resolve().relative_to(
        REPOSITORY_ROOT.resolve()
    )

    assert plan.experiment_config_path == str(expected)
