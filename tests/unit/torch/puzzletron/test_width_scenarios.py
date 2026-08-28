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

"""Tests for width-scenario identities, preparation, and projection."""

import json
import sys
from pathlib import Path

import pytest

import examples.puzzletron.finalize_replacement_scoring as replacement_finalizer
from examples.puzzletron.embedding_pipeline import (
    _project_vllm_stats_to_scenarios,
    _scenario_overrides,
    _visible_gpu_count,
    run_embedding_stage,
    scenario_preparation_commands,
    scenario_worker_commands,
)
from examples.puzzletron.prepare_width_scenarios import _prepare_scenario_destination
from modelopt.torch.puzzletron.block_config import BlockConfig, FFNConfig
from modelopt.torch.puzzletron.candidates import build_candidate_library, load_stats_identity_cache
from modelopt.torch.puzzletron.orchestration.config import _apply_override
from modelopt.torch.puzzletron.scenarios import ScenarioKey
from puzzletron_orchestrator.adapters.stage_compat import stage_is_complete


def _finalize_replacement_scoring(tmp_path, monkeypatch):
    config_path = tmp_path / "experiment.yaml"
    config_path.touch()
    root_override = "++replacement_scoring.automodel.lm_head_backend=streaming"
    loaded_overrides = []
    config = {
        "puzzle_dir": str(tmp_path),
        "model": {"path": "tiny-qwen"},
        "embedding_pruning": {"enabled": True, "widths": [256]},
        "replacement_scoring": {
            "granularity": "subblock",
            "automodel": {"lm_head_backend": "streaming"},
        },
    }

    def load_config(path, *, overrides=None):
        assert path == config_path
        loaded_overrides.extend(overrides or ())
        return config

    monkeypatch.setattr(replacement_finalizer, "pipeline_config_from_path", load_config)
    report = {"scenario_count": 1, "widths": [256]}

    def publish_report(config):
        summary = tmp_path / "artifacts" / "replacement_scoring" / "summary.json"
        summary.parent.mkdir(parents=True)
        summary.write_text(json.dumps(report))
        return report

    monkeypatch.setattr(
        replacement_finalizer,
        "finalize_replacement_scoring_diagnostics",
        publish_report,
    )

    published_report = replacement_finalizer.finalize_replacement_scoring(
        config_path,
        tmp_path,
        overrides=[root_override],
    )
    return config, loaded_overrides, published_report


def test_replacement_scoring_finalizer_publishes_current_terminal_manifest(tmp_path, monkeypatch):
    config, loaded_overrides, published_report = _finalize_replacement_scoring(
        tmp_path, monkeypatch
    )

    manifest_path = tmp_path / "manifests" / "replacement_scoring.json"
    manifest = json.loads(manifest_path.read_text())
    assert loaded_overrides == ["++replacement_scoring.automodel.lm_head_backend=streaming"]
    assert published_report == {"scenario_count": 1, "widths": [256]}
    assert manifest["stage"] == "replacement_scoring"
    assert manifest["status"] == "success"
    assert manifest["semantic_config"]["replacement_scoring"]["automodel"] == {
        "lm_head_backend": "streaming"
    }
    assert manifest["outputs"]["report"] == published_report
    assert stage_is_complete(config, "replacement_scoring")


@pytest.mark.parametrize(
    "stale_input",
    [
        "missing-summary",
        "changed-identity",
        "malformed-manifest",
        "malformed-outputs",
        "missing-report",
    ],
)
def test_replacement_scoring_finalization_marker_rejects_stale_inputs(
    tmp_path, monkeypatch, stale_input
):
    _finalize_replacement_scoring(tmp_path, monkeypatch)
    manifest_path = tmp_path / "manifests" / "replacement_scoring.json"
    summary = tmp_path / "artifacts" / "replacement_scoring" / "summary.json"

    marker_a = tmp_path / "completion-a" / "finalized"
    marker_a.parent.mkdir()
    replacement_finalizer.write_finalization_marker(marker_a, manifest_path)
    assert replacement_finalizer.finalization_marker_is_current(marker_a, manifest_path, summary)

    if stale_input == "missing-summary":
        summary.unlink()
    elif stale_input == "changed-identity":
        manifest = json.loads(manifest_path.read_text())
        manifest["semantic_identity"] = "replacement_scoring_semantic_b"
        manifest_path.write_text(json.dumps(manifest))
    elif stale_input == "malformed-manifest":
        manifest_path.write_text("[]\n")
    elif stale_input == "malformed-outputs":
        manifest = json.loads(manifest_path.read_text())
        manifest["outputs"] = ["invalid"]
        manifest_path.write_text(json.dumps(manifest))
    else:
        manifest = json.loads(manifest_path.read_text())
        manifest["outputs"] = {}
        manifest_path.write_text(json.dumps(manifest))
        summary.write_text("null\n")

    assert not replacement_finalizer.finalization_marker_is_current(
        marker_a, manifest_path, summary
    )


def test_replacement_scoring_finalizer_main_reads_root_overrides(monkeypatch):
    captured = {}
    overrides = [
        "++replacement_scoring.automodel.lm_head_backend=streaming",
        "embedding_pruning.enabled=false",
    ]

    def finalize(config_path, puzzle_dir, *, overrides=None):
        captured.update(
            config_path=config_path,
            puzzle_dir=puzzle_dir,
            overrides=overrides,
        )

    monkeypatch.setenv("FINALIZE_OVERRIDES", "\n".join(overrides))
    monkeypatch.setattr(replacement_finalizer, "finalize_replacement_scoring", finalize)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "finalize_replacement_scoring.py",
            "--config",
            "experiment.yaml",
            "--puzzle-dir",
            "run",
        ],
    )

    replacement_finalizer.main()

    assert captured == {
        "config_path": "experiment.yaml",
        "puzzle_dir": "run",
        "overrides": overrides,
    }


def _write_scenario_manifest(
    puzzle_dir: Path,
    width: int,
    *,
    bypass_checkpoint: Path | None = None,
) -> Path:
    scenario = puzzle_dir / "scenarios" / f"width-{width:04d}" / "depth-00"
    scenario.mkdir(parents=True, exist_ok=True)
    (scenario / "scenario_manifest.json").write_text(
        json.dumps(
            {
                "parent_checkpoint": str(scenario / "ckpts" / "sorted_teacher"),
                "bypass_checkpoint": (
                    str(bypass_checkpoint) if bypass_checkpoint is not None else None
                ),
            }
        )
    )
    return scenario


def test_candidate_identity_and_metadata_are_width_specific():
    block = BlockConfig(subblock_configs=(FFNConfig(intermediate_size=16),))
    wide = build_candidate_library(
        [block],
        parent_checkpoint_identity="teacher",
        hidden_width=1024,
    )
    narrow = build_candidate_library(
        [block],
        parent_checkpoint_identity="teacher",
        hidden_width=768,
    )

    assert wide[0].hidden_width == 1024
    assert narrow[0].hidden_width == 768
    assert wide[0].identity.value != narrow[0].identity.value


def test_scenario_artifact_path_encodes_width_and_depth():
    assert ScenarioKey(hidden_width=768, removed_sublayers=1).relative_path.as_posix() == (
        "scenarios/width-0768/depth-01"
    )


def test_runtime_stats_cache_filters_exact_hidden_width(tmp_path):
    stats_path = tmp_path / "subblock_stats.json"
    stats_path.write_text(
        json.dumps(
            [
                {
                    "args": {"n_embd": width, "batch_size": 1},
                    "subblocks": [
                        {
                            "parent_layer_index": 0,
                            "subblock_config": {
                                "kind": "ffn",
                                "name": "ffn",
                                "intermediate_size": 8,
                            },
                            "runtime_ms": width / 1000,
                        }
                    ],
                }
                for width in (1024, 768)
            ]
        )
    )

    wide = load_stats_identity_cache(stats_path, hidden_width=1024)
    narrow = load_stats_identity_cache(stats_path, hidden_width=768)

    assert len(wide) == 2
    assert len(narrow) == 2
    assert set(wide.values()) != set(narrow.values())


def test_embedding_pipeline_rejects_vllm_stats_missing_a_configured_width(tmp_path):
    (tmp_path / "subblock_stats.json").write_text(
        json.dumps([{"args": {"n_embd": 1024, "runtime_stats": True}}])
    )

    with pytest.raises(ValueError, match="hidden width 768"):
        _project_vllm_stats_to_scenarios(
            {
                "puzzle_dir": str(tmp_path),
                "embedding_pruning": {"widths": [1024, 768]},
            }
        )


def test_embedding_pipeline_uses_public_subblock_replacement_scoring_contract(tmp_path):
    """Build replacement-scoring commands with the current scenario-local config keys."""

    _write_scenario_manifest(tmp_path, 768)
    packed_token_cache = tmp_path / "dataset_cache" / "validation.tokens"
    (command,) = scenario_worker_commands(
        config_path="experiment.yaml",
        config={
            "puzzle_dir": str(tmp_path),
            "embedding_pruning": {"widths": [768]},
            "replacement_scoring": {
                "granularity": "subblock",
                "packed_token_cache_path": str(packed_token_cache),
            },
        },
        stage="replacement_scoring",
        gpus_per_node=8,
    )

    assert command[1:4] == ("-m", "torch.distributed.run", "--standalone")
    assert tuple(
        command[command.index("--worker-stage") : command.index("--worker-stage") + 2]
    ) == ("--worker-stage", "replacement_scoring")
    overrides = [command[index + 1] for index, value in enumerate(command) if value == "--override"]
    overrides_by_key = dict(override.split("=", 1) for override in overrides)
    assert Path(overrides_by_key["replacement_scoring.solutions_path"]).name == (
        "single_subblock_replacement_solutions.json"
    )
    assert Path(overrides_by_key["replacement_scoring.output_dir"]).name == (
        "single_subblock_replacement_solutions--validation"
    )
    assert f"++replacement_scoring.packed_token_cache_path={packed_token_cache}" in overrides
    assert {
        "++replacement_scoring.source_checkpoint_dir",
        "++replacement_scoring.target_teacher_dir",
        "++scoring_diagnostic.scores_dir",
        "++vllm_stats_diagnostic.stats_path",
    } <= overrides_by_key.keys()


def test_embedding_pipeline_launches_block_library_with_torchrun(tmp_path):
    """Launch block-library work with one process and current runtime-stat keys."""

    _write_scenario_manifest(tmp_path, 768)
    (command,) = scenario_worker_commands(
        config_path="experiment.yaml",
        config={
            "puzzle_dir": str(tmp_path),
            "embedding_pruning": {"widths": [768]},
        },
        stage="build_library",
        gpus_per_node=1,
    )

    assert command[1:4] == ("-m", "torch.distributed.run", "--standalone")
    assert "--nproc_per_node=1" in command
    overrides = [command[index + 1] for index, value in enumerate(command) if value == "--override"]
    assert "embedding_pruning.enabled=false" in overrides
    scenario_teacher = tmp_path / "scenarios/width-0768/depth-00/ckpts/sorted_teacher"
    assert f"build_library.source_checkpoint_dir={scenario_teacher}" in overrides
    assert "++vllm_stats.runtime_stats.execution=inline" in overrides
    assert "vllm_stats.runtime_stats.execution=inline" not in overrides
    assert "calc_subblock_stats.runtime_stats.execution=inline" not in overrides


def test_embedding_pipeline_scenario_overrides_compose_with_current_config(tmp_path):
    """Apply every generated scenario override to a representative current config."""

    scenario = tmp_path / "scenarios" / "width-0768" / "depth-00"
    scenario.mkdir(parents=True)
    (scenario / "scenario_manifest.json").write_text(
        json.dumps({"width": 768, "bypass_checkpoint": str(tmp_path / "accepted-bypass")})
    )
    packed_token_cache = tmp_path / "dataset_cache" / "validation.tokens"
    overrides = _scenario_overrides(
        {
            "replacement_scoring": {
                "granularity": "subblock",
                "packed_token_cache_path": str(packed_token_cache),
            }
        },
        scenario,
    )

    config = {
        "puzzle_dir": "/initial/puzzle",
        "experiment": {"dir": "/initial/puzzle"},
        "teacher_dir": "/initial/teacher",
        "convert": {"teacher_dir": "/initial/teacher"},
        "bypass": {"enabled": True},
        "embedding_pruning": {"enabled": True},
        "replacement_library_path": "/initial/replacement_library.json",
        "build_library": {"source_checkpoint_dir": "/initial/teacher"},
        "vllm_stats": {"runtime_stats": {"enabled": True}},
        "replacement_scoring": {
            "teacher_dir": "/initial/teacher",
            "solutions_path": "/initial/solutions.json",
            "output_dir": "/initial/scores",
        },
    }
    for override in overrides:
        _apply_override(config, override)

    teacher = scenario / "ckpts" / "sorted_teacher"

    # Scenario workers must use only scenario-local inputs and outputs.
    assert config["puzzle_dir"] == str(scenario)
    assert config["experiment"]["dir"] == str(scenario)
    assert config["teacher_dir"] == str(teacher)
    assert config["convert"]["teacher_dir"] == str(teacher)
    assert config["replacement_library_path"] == str(scenario / "replacement_library.json")
    assert config["build_library"]["source_checkpoint_dir"] == str(teacher)

    # Composite workers disable stages already completed by the parent campaign.
    assert config["bypass"]["enabled"] is False
    assert config["embedding_pruning"]["enabled"] is False

    # Runtime-stat and scoring artifacts remain isolated within the scenario.
    assert config["vllm_stats"]["runtime_stats"]["execution"] == "inline"
    assert config["replacement_scoring"]["teacher_dir"] == str(teacher)
    assert config["replacement_scoring"]["source_checkpoint_dir"] == str(teacher)
    assert config["replacement_scoring"]["target_teacher_dir"] == str(teacher)
    assert config["replacement_scoring"]["solutions_path"] == str(
        scenario / "single_subblock_replacement_solutions.json"
    )
    assert config["replacement_scoring"]["output_dir"] == str(
        scenario / "single_subblock_replacement_solutions--validation"
    )
    assert config["replacement_scoring"]["packed_token_cache_path"] == str(packed_token_cache)
    assert config["replacement_scoring"]["bypass_checkpoint_dir"] == str(
        scenario / "ckpts" / "bypass_overlay"
    )
    assert config["vllm_stats_diagnostic"]["stats_path"] == str(scenario / "subblock_stats.json")
    assert config["vllm_stats_diagnostic"]["output_dir"] == str(
        scenario / "artifacts/vllm_stats_diagnostic"
    )
    assert config["scoring_diagnostic"]["scores_dir"] == str(
        scenario / "single_subblock_replacement_solutions--validation"
    )
    assert config["scoring_diagnostic"]["output_dir"] == str(
        scenario / "artifacts/scoring_diagnostic"
    )


def test_embedding_pipeline_skips_composite_work_on_nonzero_rank(tmp_path, monkeypatch):
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setattr(
        "examples.puzzletron.embedding_pipeline.subprocess.run",
        lambda *args, **kwargs: pytest.fail("nonzero rank launched composite work"),
    )

    outputs = run_embedding_stage(
        config_path="experiment.yaml",
        config={
            "puzzle_dir": str(tmp_path),
            "embedding_pruning": {"widths": [768]},
        },
        stage="build_library",
        gpus_per_node=1,
    )

    assert outputs["skipped_nonzero_rank"] is True


def test_embedding_pipeline_routes_width_local_bypass_overlay(tmp_path):
    _write_scenario_manifest(
        tmp_path,
        768,
        bypass_checkpoint=tmp_path / "accepted-bypass",
    )
    (command,) = scenario_worker_commands(
        config_path="experiment.yaml",
        config={
            "puzzle_dir": str(tmp_path),
            "embedding_pruning": {"widths": [768]},
            "replacement_scoring": {
                "granularity": "subblock",
                "bypass_checkpoint_dir": str(tmp_path / "accepted-bypass"),
            },
        },
        stage="replacement_scoring",
        gpus_per_node=2,
    )

    overrides = [command[index + 1] for index, value in enumerate(command) if value == "--override"]
    assert (
        "++replacement_scoring.bypass_checkpoint_dir="
        f"{tmp_path}/scenarios/width-0768/depth-00/ckpts/bypass_overlay"
    ) in overrides


def test_embedding_pipeline_prepares_subblock_solutions_for_every_width(tmp_path):
    commands = scenario_preparation_commands(
        config={
            "puzzle_dir": str(tmp_path),
            "embedding_pruning": {"widths": [1024, 768]},
            "replacement_scoring": {"granularity": "subblock"},
        },
        stage="replacement_scoring",
    )

    assert len(commands) == 2
    assert all(
        command[1].endswith("prepare_subblock_replacement_scoring.py") for command in commands
    )
    assert {Path(command[command.index("--puzzle-dir") + 1]).name for command in commands} == {
        "depth-00"
    }
    assert {
        Path(command[command.index("--puzzle-dir") + 1]).parent.name for command in commands
    } == {"width-1024", "width-0768"}


def test_width_scenario_destination_rejects_or_replaces_stale_parent(tmp_path):
    scenario_dir = tmp_path / "scenarios" / "width-0768" / "depth-00"
    scenario_dir.mkdir(parents=True)
    (scenario_dir / "stale.txt").write_text("stale")

    with pytest.raises(FileExistsError, match="different parent identity"):
        _prepare_scenario_destination(
            scenario_dir,
            source_checkpoint_fingerprint="new-parent",
            overwrite_stale=False,
        )

    assert _prepare_scenario_destination(
        scenario_dir,
        source_checkpoint_fingerprint="new-parent",
        overwrite_stale=True,
    )
    assert not scenario_dir.exists()


def test_width_scenario_destination_reuses_complete_matching_parent(tmp_path):
    scenario_dir = tmp_path / "scenarios" / "width-0768" / "depth-00"
    scenario_dir.mkdir(parents=True)
    (scenario_dir / "scenario_manifest.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "source_checkpoint_fingerprint": "same-parent",
            }
        )
    )
    (scenario_dir / "replacement_library.json").write_text("{}")
    (scenario_dir / "single_sequence_replacement_solutions.json").write_text("[]")
    checkpoint = scenario_dir / "ckpts" / "sorted_teacher"
    checkpoint.mkdir(parents=True)
    (checkpoint / "config.json").write_text("{}")

    assert not _prepare_scenario_destination(
        scenario_dir,
        source_checkpoint_fingerprint="same-parent",
        overwrite_stale=False,
    )
    assert scenario_dir.exists()


def test_embedding_pipeline_uses_task_visible_gpu_count(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,5")

    assert _visible_gpu_count(8) == 2
