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

import subprocess
import sys
from pathlib import Path

import pytest

from examples.puzzletron.embedding_pipeline import scenario_worker_commands
from examples.puzzletron.main import (
    PIPELINE_STAGE_ORDER,
    _is_externally_launched,
    _parse_args,
    _resume_kwargs,
    build_worker_command,
    run_pipeline,
    stage_sequence,
)
from modelopt.torch.puzzletron.pipeline_config import pipeline_config_from_path
from modelopt.torch.puzzletron.stage_runner import STAGES as RUNNER_STAGES
from modelopt.torch.puzzletron.stages.future import _select_best_evaluated_checkpoint
from modelopt.torch.puzzletron.stages.graph import STAGE_REGISTRY, topological_stage_ids

CONFIG = Path("examples/puzzletron/configs/clean/my_paths.example.yaml")
EXPERIMENT_CONFIG = Path(
    "examples/puzzletron/configs/clean/families/qwen3_5/qwen3_5_9b/full_pipeline.yaml"
)
SANITY_CHECK_CONFIG = Path(
    "examples/puzzletron/configs/clean/families/qwen3_5/qwen3_5_9b/sanity_check.yaml"
)


def test_qwen35_9b_config_enables_the_complete_one_node_pipeline():
    cfg = pipeline_config_from_path(CONFIG)

    assert topological_stage_ids() == PIPELINE_STAGE_ORDER
    assert tuple(STAGE_REGISTRY) == RUNNER_STAGES
    assert cfg["model"]["source"] == "Qwen/Qwen3.5-9B"
    assert cfg["model"]["force_hf"] is False
    assert cfg["parallel"] == {"tp": 1, "cp": 4, "pp": 2, "ep": 1, "dp": 1}
    assert stage_sequence(None, cfg) == (
        "convert",
        "prepare_data",
        "activation",
        "sort",
        "sort_equivalence",
        "activation_diagnostic",
        "depth",
        "build_library",
        "vllm_stats_diagnostic",
        "scoring",
        "scoring_diagnostic",
        "mip",
        "evaluation",
        "aiperf",
        "distillation_overfit",
        "distillation",
    )
    assert PIPELINE_STAGE_ORDER[:3] == ("convert", "prepare_data", "activation")
    assert cfg["prepare_data"]["enabled"] is True
    assert [cache["split"] for cache in cfg["prepare_data"]["caches"]] == [
        "train",
        "validation",
    ]

    assert cfg["embedding_pruning"]["widths"] == [4096, 3840, 3584]
    assert cfg["activation_diagnostic"]["single_load_parent_sweep"] is True
    assert cfg["activation_diagnostic"]["methods"] == ["activation", "random", "reverse"]
    assert cfg["activation_diagnostic"].get("hidden_width_diagnostic", True) is True
    assert cfg["bypass"]["overfit"]["modes"] == ["smallest_fixed", "diverse_resampled"]
    assert cfg["depth"]["max_removals"] == 6
    assert cfg["depth"]["max_subblocks_to_remove"] == 6
    assert cfg["pruning"]["eval_samples"] == 64 * 1024
    assert cfg["pruning"]["block_size"] == 16 * 1024
    assert cfg["bypass"]["training"]["training_tokens"] == 1024**3
    assert (
        cfg["bypass"]["training"]["micro_batch_size"]
        * cfg["bypass"]["training"]["grad_accumulation_steps"]
        * cfg["parallel"]["dp"]
        * cfg["bypass"]["data"]["block_size"]
        == 1024**2
    )
    for stage in (
        "activation_diagnostic",
        "sort_equivalence",
        "depth",
        "vllm_stats_diagnostic",
        "scoring_diagnostic",
        "evaluation",
        "aiperf",
        "distillation_overfit",
        "distillation",
    ):
        assert cfg[stage]["enabled"] is True
    for stage in ("bypass_overfit", "bypass", "bypass_diagnostic"):
        assert cfg[stage]["enabled"] is False
    assert cfg["bypass"]["use_nested_bypassed_checkpoint_for_scoring"] is False


def test_qwen35_9b_sanity_check_config_selects_imported_artifacts_and_diagnostics():
    cfg = pipeline_config_from_path(SANITY_CHECK_CONFIG)

    assert cfg["puzzle_dir"] == (
        "/shared/fs1/portfolios/coreai/projects/coreai_dlalgo_llm/users/ssameni/engineering/"
        "puzzle_runs/qwen3_5/qwen3_5_9b/sanity_check"
    )
    assert cfg["display_name"] == "Qwen3.5-9B - Sanity Check"
    assert cfg["artifact_import"] == {
        "source_root": (
            "/shared/fs1/portfolios/coreai/projects/coreai_dlalgo_llm/users/ssameni/engineering/"
            "puzzle_runs/qwen3_5/qwen3_5_9b/full_pipeline"
        ),
        "bundles": ["activation", "depth", "vllm_stats", "scoring", "bypass_evidence"],
        "bypass_evidence_report_only": True,
    }
    assert stage_sequence(None, cfg) == (
        "convert",
        "prepare_data",
        "activation",
        "sort",
        "sort_equivalence",
        "width_slice_equivalence",
        "activation_diagnostic",
        "bypass_overfit",
        "depth",
        "vllm_stats",
        "build_library",
        "vllm_stats_diagnostic",
        "scoring",
        "scoring_diagnostic",
        "mip",
        "evaluation",
        "aiperf",
        "distillation_overfit",
    )
    for stage in (
        "sort_equivalence",
        "width_slice_equivalence",
        "activation_diagnostic",
        "bypass_overfit",
        "depth",
        "vllm_stats",
        "build_library",
        "vllm_stats_diagnostic",
        "scoring",
        "scoring_diagnostic",
        "mip",
        "evaluation",
        "aiperf",
        "distillation_overfit",
    ):
        assert cfg[stage]["enabled"] is True
    for stage in ("bypass", "distillation", "post_kd_evaluation"):
        assert cfg[stage]["enabled"] is False
    assert cfg["bypass"]["overfit"]["modes"] == ["smallest_fixed", "diverse_resampled"]
    assert cfg["aiperf"]["num_best_to_eval"] == 2
    assert cfg["distillation"]["num_best_to_distill"] == 1


def test_paths_overlay_contains_every_machine_specific_checkout():
    cfg = pipeline_config_from_path(CONFIG)

    assert set(cfg["paths"]) == {
        "modelopt_root",
        "automodel_root",
        "vllm_root",
        "aiperf_root",
        "dataset_path",
        "puzzle_root",
    }
    assert cfg["dataset_path"] == cfg["paths"]["dataset_path"]
    assert cfg["puzzle_root"] == cfg["paths"]["puzzle_root"]
    assert cfg["aiperf"]["executable"].endswith("/aiperf")
    assert cfg["aiperf"]["checkpoint_source"] == "scenario_grid"
    assert cfg["distillation"]["selection"] == "best_evaluation"
    assert "oc.env" not in EXPERIMENT_CONFIG.read_text()


def test_worker_command_self_launches_torchrun_only_for_distributed_stages():
    distributed = build_worker_command(
        config_path="my_paths.yaml",
        stage="activation",
        overrides=("pruning.eval_samples=8",),
        gpus_per_node=8,
    )
    assert distributed[:4] == (
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
    )
    assert "--nproc_per_node=8" in distributed
    assert distributed[-2:] == ("--override", "pruning.eval_samples=8")

    single = build_worker_command(
        config_path="my_paths.yaml",
        stage="mip",
        overrides=(),
        gpus_per_node=8,
    )
    assert "torch.distributed.run" not in single
    assert single[-2:] == ("--worker-stage", "mip")


def test_detects_an_externally_launched_distributed_world(monkeypatch):
    monkeypatch.setenv("RANK", "4")
    monkeypatch.setenv("WORLD_SIZE", "16")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("TORCHELASTIC_RUN_ID", "puzzletron-test")

    assert _is_externally_launched()


def test_srun_rank_environment_is_not_mistaken_for_torchrun(monkeypatch):
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.delenv("TORCHELASTIC_RUN_ID", raising=False)

    assert not _is_externally_launched()


def test_main_rejects_obsolete_node_metadata_flags(monkeypatch):
    monkeypatch.setattr(sys, "argv", ("main.py", "--config", "my_paths.yaml", "--nodes", "2"))

    with pytest.raises(SystemExit) as error:
        _parse_args()

    assert error.value.code == 2


def test_documented_main_command_starts_from_the_example_directory():
    result = subprocess.run(
        (sys.executable, "main.py", "--help"),
        cwd=Path("examples/puzzletron"),
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--config" in result.stdout


def test_pipeline_skips_valid_completions_and_stops_on_first_failure():
    launched = []
    report_states = []

    def run(command):
        launched.append(command)
        return subprocess.CompletedProcess(command, 17 if "sort" in command else 0)

    with pytest.raises(subprocess.CalledProcessError) as error:
        run_pipeline(
            config_path="my_paths.yaml",
            config={"convert": {}, "activation": {}, "sort": {}},
            stages=("convert", "activation", "sort", "scoring"),
            overrides=(),
            gpus_per_node=8,
            force=False,
            is_complete=lambda stage: stage == "convert",
            mark_complete=lambda stage: None,
            refresh_report=lambda running_stage: report_states.append(running_stage),
            command_runner=run,
        )

    assert error.value.returncode == 17
    assert [command[-1] for command in launched] == ["activation", "sort"]
    assert report_states == [None, "activation", None, "sort", None]


def test_resume_tracks_required_stage_outputs(tmp_path: Path):
    config = {
        "puzzle_dir": str(tmp_path),
        "convert": {},
        "prepare_data": {"enabled": True},
        "activation": {},
        "sort": {},
        "build_library": {},
        "scoring": {},
        "mip": {},
    }

    kwargs = _resume_kwargs(config, tmp_path / "config.yaml", "prepare_data")

    assert kwargs["required_patterns"] == (
        "manifests/prepare_data.json",
        "dataset_cache/*.tokens",
        "dataset_cache/*.tokens.json",
    )
    assert tuple(kwargs["upstream_markers"]) == ("convert",)


def test_sort_resume_depends_on_activation_not_a_later_diagnostic(tmp_path: Path):
    config = {
        "puzzle_dir": str(tmp_path),
        "convert": {},
        "prepare_data": {"enabled": True},
        "activation": {},
        "activation_diagnostic": {"enabled": True},
        "sort": {},
    }

    kwargs = _resume_kwargs(config, tmp_path / "config.yaml", "sort")

    assert tuple(kwargs["upstream_markers"]) == ("activation",)


def test_build_library_resume_uses_selected_sorted_teacher_parent(tmp_path: Path):
    config = {
        "puzzle_dir": str(tmp_path),
        "convert": {},
        "prepare_data": {"enabled": True},
        "activation": {},
        "sort": {},
        "depth": {"enabled": True},
        "build_library": {},
        "scoring": {},
        "mip": {},
    }

    kwargs = _resume_kwargs(config, tmp_path / "config.yaml", "build_library")

    assert tuple(kwargs["upstream_markers"]) == ("sort",)


def test_embedding_scoring_fans_out_once_per_width(tmp_path: Path):
    config = {
        "puzzle_dir": str(tmp_path),
        "embedding_pruning": {"enabled": True, "widths": [4096, 3840, 3584]},
    }

    commands = scenario_worker_commands(
        config_path="configs/clean/my_paths.yaml",
        config=config,
        stage="scoring",
        gpus_per_node=8,
    )

    assert len(commands) == 3
    assert all("torch.distributed.run" in command for command in commands)
    assert all("--scenario-child" in command for command in commands)
    assert "scenarios/width-4096/depth-00" in " ".join(commands[0])
    assert "scenarios/width-3840/depth-00" in " ".join(commands[1])
    assert "scenarios/width-3584/depth-00" in " ".join(commands[2])


def test_global_kd_selects_the_lowest_finite_evaluation_loss(tmp_path: Path):
    summary = tmp_path / "evaluation_summary.json"
    summary.write_text(
        '[{"checkpoint":"/candidate-a","metrics":{"lm_loss":2.0}},'
        '{"checkpoint":"/candidate-b","metrics":{"lm_loss":1.5}},'
        '{"checkpoint":"/broken","metrics":{"lm_loss":"nan"}}]'
    )

    assert _select_best_evaluated_checkpoint(summary) == Path("/candidate-b")


def test_readme_exposes_one_command_and_section_by_section_verification():
    readme = Path("examples/puzzletron/README.md").read_text()

    assert "python main.py --config configs/clean/my_paths.yaml" in readme
    assert "campaign_report.html" in readme
    assert 'python -m pip install -e "$AIPERF_ROOT"' in readme
    assert "source .venv/bin/activate" in readme
    for heading in (
        "Convert",
        "Activation scoring",
        "Sort",
        "Bypass",
        "Replacement library",
        "Replacement scoring",
        "MIP",
        "Evaluation",
        "AIPerf",
        "Global distillation",
    ):
        assert heading in readme
