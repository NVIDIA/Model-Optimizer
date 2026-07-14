# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from types import SimpleNamespace

import pytest

from examples.puzzletron.prepare_width_scenarios import (
    _prepare_scenario_destination,
    _resolve_source_checkpoint,
)

from modelopt.torch.puzzletron.block_config import AttentionConfig, BlockConfig, FFNConfig
from modelopt.torch.puzzletron.candidates import (
    build_candidate_library,
    load_stats_identity_cache,
)
from modelopt.torch.puzzletron.depth.schema import DepthScenario
from modelopt.torch.puzzletron.replacement_library.build_replacement_library import (
    build_replacement_library_from_sorted_teacher,
)
from modelopt.torch.puzzletron.replacement_library.library import ReplacementLibrary
from modelopt.torch.puzzletron.scenarios import ScenarioKey


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
