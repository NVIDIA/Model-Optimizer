# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from examples.puzzletron.embedding_pipeline import (
    _project_vllm_stats_to_scenarios,
    _visible_gpu_count,
    finalize_replacement_scoring_diagnostics,
    run_embedding_stage,
    scenario_preparation_commands,
    scenario_worker_commands,
)
from examples.puzzletron.prepare_width_scenarios import (
    _prepare_scenario_destination,
    _resolve_source_checkpoint,
)
from modelopt.torch.puzzletron.block_config import AttentionConfig, BlockConfig, FFNConfig
from modelopt.torch.puzzletron.candidates import build_candidate_library, load_stats_identity_cache
from modelopt.torch.puzzletron.depth.schema import DepthScenario
from modelopt.torch.puzzletron.replacement_library.build_replacement_library import (
    build_replacement_library_from_sorted_teacher,
)
from modelopt.torch.puzzletron.replacement_library.library import ReplacementLibrary
from modelopt.torch.puzzletron.scenarios import ScenarioKey


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


def test_depth_scenario_identity_includes_hidden_width():
    common = {
        "parent_checkpoint_identity": "teacher",
        "data_identity": "data",
        "evaluator_revision": "rev",
    }
    wide = DepthScenario(hidden_width=1024, **common)
    narrow = DepthScenario(hidden_width=768, **common)

    assert wide.scenario_id != narrow.scenario_id


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


def test_replacement_library_and_solutions_record_hidden_width(tmp_path, monkeypatch):
    block = BlockConfig(subblock_configs=(FFNConfig(intermediate_size=16),))
    config = SimpleNamespace(
        block_configs=[block],
        hidden_size=1024,
        intermediate_size=16,
        num_attention_heads=8,
        num_key_value_heads=2,
        num_hidden_layers=1,
    )

    class Descriptor:
        @staticmethod
        def requires_trust_remote_code():
            return False

        @staticmethod
        def get_language_model_config(model_config):
            return model_config

    sorted_teacher = tmp_path / "sorted_teacher"
    sorted_teacher.mkdir()
    monkeypatch.setattr(
        "modelopt.torch.puzzletron.replacement_library.build_replacement_library.load_model_config",
        lambda *args, **kwargs: config,
    )

    build_replacement_library_from_sorted_teacher(
        master_puzzle_dir=tmp_path / "width-0768",
        sorted_teacher_dir=sorted_teacher,
        descriptor=Descriptor,
        search_space={"axes": {"ffn_intermediate": {"sizes": [8]}}},
        include_noops=False,
        hidden_width=768,
    )

    library = json.loads((tmp_path / "width-0768" / "replacement_library.json").read_text())
    solutions = json.loads(
        (tmp_path / "width-0768" / "single_sequence_replacement_solutions.json").read_text()
    )
    assert library["hidden_width"] == 768
    assert library["scenario"] == "width-0768"
    assert solutions
    assert {solution["hidden_width"] for solution in solutions} == {768}
    loaded = ReplacementLibrary(
        tmp_path / "width-0768" / "replacement_library.json",
        Descriptor,
    )
    assert loaded.hidden_width == 768


def test_replacement_library_uses_canonical_single_noops_without_joint_noop(
    tmp_path, monkeypatch
):
    blocks = [
        BlockConfig(
            subblock_configs=(
                AttentionConfig(num_query_heads=8, num_kv_heads=2),
                FFNConfig(intermediate_size=16),
            )
        )
    ]
    config = SimpleNamespace(
        block_configs=blocks,
        hidden_size=1024,
        intermediate_size=16,
        num_attention_heads=8,
        num_key_value_heads=2,
        num_hidden_layers=1,
    )

    class Descriptor:
        @staticmethod
        def requires_trust_remote_code():
            return False

        @staticmethod
        def get_language_model_config(model_config):
            return model_config

    sorted_teacher = tmp_path / "sorted_teacher"
    sorted_teacher.mkdir()
    monkeypatch.setattr(
        "modelopt.torch.puzzletron.replacement_library.build_replacement_library.load_model_config",
        lambda *args, **kwargs: config,
    )

    build_replacement_library_from_sorted_teacher(
        master_puzzle_dir=tmp_path / "library",
        sorted_teacher_dir=sorted_teacher,
        descriptor=Descriptor,
        search_space={
            "axes": {},
            "no_op": {
                "subblocks": ["attention", "ffn"],
                "whole_block": False,
                "cartesian": True,
            },
        },
        include_noops=True,
        hidden_width=1024,
    )

    library = json.loads((tmp_path / "library" / "replacement_library.json").read_text())
    patterns = {
        tuple(
            sorted(
                subblock["kind"]
                for subblock in entry["child_block_configs"][0]["subblock_configs"]
                if subblock.get("no_op")
            )
        )
        for entry in library["entries"]
    }
    assert ("attention",) in patterns
    assert ("ffn",) in patterns
    assert ("attention", "ffn") not in patterns


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
                            "subblock_config": {"kind": "ffn", "name": "ffn", "intermediate_size": 8},
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


def test_embedding_pipeline_projects_root_vllm_stats_by_hidden_width(tmp_path):
    root_stats = [
        {
            "args": {
                "n_embd": width,
                "runtime_stats": runtime_stats,
                "weights_dtype": dtype,
            },
            "subblocks": [{"runtime_ms": width / 1000}],
        }
        for width in (1024, 768)
        for runtime_stats, dtype in ((False, "nvfp4"), (True, "torch.bfloat16"))
    ]
    (tmp_path / "subblock_stats.json").write_text(json.dumps(root_stats))

    outputs = _project_vllm_stats_to_scenarios(
        {
            "puzzle_dir": str(tmp_path),
            "embedding_pruning": {"widths": [1024, 768]},
        }
    )

    assert set(outputs) == {1024, 768}
    for width, output in outputs.items():
        scenario_stats = json.loads(output.read_text())
        assert len(scenario_stats) == 2
        assert {entry["args"]["n_embd"] for entry in scenario_stats} == {width}


def test_embedding_pipeline_uses_public_subblock_replacement_scoring_contract(tmp_path):
    _write_scenario_manifest(tmp_path, 768)
    (command,) = scenario_worker_commands(
        config_path="experiment.yaml",
        config={
            "puzzle_dir": str(tmp_path),
            "embedding_pruning": {"widths": [768]},
            "replacement_scoring": {"granularity": "subblock"},
        },
        stage="replacement_scoring",
        gpus_per_node=8,
    )

    assert command[1:4] == ("-m", "torch.distributed.run", "--standalone")
    assert tuple(
        command[command.index("--worker-stage") : command.index("--worker-stage") + 2]
    ) == ("--worker-stage", "replacement_scoring")
    overrides = [
        command[index + 1] for index, value in enumerate(command) if value == "--override"
    ]
    assert any(
        override.endswith("/single_subblock_replacement_solutions.json")
        and override.startswith("replacement_scoring.solutions_path=")
        for override in overrides
    )
    assert any(
        override.endswith("/single_subblock_replacement_solutions--validation")
        and override.startswith("replacement_scoring.output_dir=")
        for override in overrides
    )


def test_embedding_pipeline_launches_block_library_with_torchrun(tmp_path):
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

    overrides = [
        command[index + 1] for index, value in enumerate(command) if value == "--override"
    ]
    assert (
        "replacement_scoring.bypass_checkpoint_dir="
        f"{tmp_path}/scenarios/width-0768/depth-00/ckpts/bypass_overlay"
    ) in overrides


def test_width_scenarios_expose_bypass_overlay_resolver():
    width_scenarios = sys.modules["examples.puzzletron.prepare_width_scenarios"]
    resolver = getattr(width_scenarios, "_resolve_bypass_checkpoint", None)

    assert callable(resolver)


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
    assert all(command[1].endswith("prepare_subblock_replacement_scoring.py") for command in commands)
    assert {
        Path(command[command.index("--puzzle-dir") + 1]).name
        for command in commands
    } == {"depth-00"}
    assert {
        Path(command[command.index("--puzzle-dir") + 1]).parent.name
        for command in commands
    } == {"width-1024", "width-0768"}


def test_embedding_pipeline_publishes_root_replacement_scoring_summary(
    tmp_path, monkeypatch
):
    calls = []

    def fake_report(puzzle_dir, **kwargs):
        width = int(Path(puzzle_dir).parent.name.removeprefix("width-"))
        calls.append((Path(puzzle_dir), kwargs))
        return {
            "kind": "replacement_scoring",
            "granularity": "subblock",
            "record_count": width,
            "warning_count": width // 256,
            "axes": [f"axis-{width}"],
            "metrics": ["raw_replacement_loss"],
            "outputs": [str(kwargs["output_dir"] / "summary.json")],
        }

    monkeypatch.setattr(
        "examples.puzzletron.embedding_pipeline.generate_replace_block_report",
        fake_report,
    )
    config = {
        "puzzle_dir": str(tmp_path),
        "embedding_pruning": {"widths": [1024, 768]},
        "replacement_scoring": {
            "granularity": "subblock",
            "default_metric": "raw_replacement_loss",
        },
    }

    summary = finalize_replacement_scoring_diagnostics(config)

    assert len(calls) == 2
    assert summary["widths"] == [1024, 768]
    assert summary["record_count"] == 1792
    assert summary["scenario_count"] == 2
    assert summary["axes"] == ["axis-1024", "axis-768"]
    assert json.loads(
        (tmp_path / "artifacts/replacement_scoring/summary.json").read_text()
    ) == summary


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


def test_width_scenario_destination_rebuilds_changed_bypass_overlay(tmp_path):
    scenario_dir = tmp_path / "scenarios" / "width-0768" / "depth-00"
    scenario_dir.mkdir(parents=True)
    (scenario_dir / "scenario_manifest.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "source_checkpoint_fingerprint": "same-parent",
                "bypass_source_fingerprint": "old-bypass",
            }
        )
    )
    (scenario_dir / "replacement_library.json").write_text("{}")
    (scenario_dir / "single_sequence_replacement_solutions.json").write_text("[]")
    for name in ("sorted_teacher", "bypass_overlay"):
        checkpoint = scenario_dir / "ckpts" / name
        checkpoint.mkdir(parents=True)
        (checkpoint / "config.json").write_text("{}")

    assert _prepare_scenario_destination(
        scenario_dir,
        source_checkpoint_fingerprint="same-parent",
        bypass_source_fingerprint="new-bypass",
        overwrite_stale=True,
    )
    assert not scenario_dir.exists()


def test_width_scenarios_resolve_the_fingerprinted_scoring_parent(tmp_path):
    sorted_teacher = tmp_path / "ckpts" / "sorted_teacher"
    sorted_teacher.mkdir(parents=True)
    (sorted_teacher / "config.json").write_text("{}\n")
    config = {
        "experiment": {"dir": str(tmp_path)},
        "bypass": {
            "enabled": False,
            "use_nested_bypassed_checkpoint_for_scoring": False,
        },
    }

    source = _resolve_source_checkpoint(config, explicit=None)

    assert source == sorted_teacher.resolve()
    assert (tmp_path / "artifacts" / "scoring_parent.json").is_file()


def test_embedding_pipeline_uses_task_visible_gpu_count(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,5")

    assert _visible_gpu_count(8) == 2
