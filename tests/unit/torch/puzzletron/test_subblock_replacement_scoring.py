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

import pytest
from omegaconf import OmegaConf

from modelopt.torch.puzzletron.block_config import AttentionConfig, BlockConfig, FFNConfig
from modelopt.torch.puzzletron.replacement_library.subblock_scoring import (
    build_subblock_replacement_solutions,
)
from modelopt.torch.puzzletron.scoring import resolve_scoring_paths


def _replacement(layer_idx, block):
    return {
        "weight_paths": [],
        "parent_layer_indices": [layer_idx],
        "child_block_configs": [block.to_dict()],
    }


def _by_identity(block):
    return {(subblock.kind, subblock.name): subblock for subblock in block.subblock_configs}


def test_builds_each_unique_one_subblock_replacement_for_arbitrary_named_subblocks():
    teacher = BlockConfig(
        subblock_configs=(
            AttentionConfig(name="mixer", num_query_heads=8, num_kv_heads=2),
            FFNConfig(name="up", intermediate_size=16),
            FFNConfig(name="down", intermediate_size=12),
        )
    )
    mixer_small = AttentionConfig(name="mixer", num_query_heads=4, num_kv_heads=2)
    up_small = FFNConfig(name="up", intermediate_size=8)
    down_small = FFNConfig(name="down", intermediate_size=6)
    full_candidates = [
        teacher,
        BlockConfig(subblock_configs=(mixer_small, up_small, down_small)),
        # Same semantic candidate in a different serialized order must not duplicate output.
        BlockConfig(subblock_configs=(down_small, mixer_small, up_small)),
    ]

    solutions = build_subblock_replacement_solutions(
        [_replacement(0, block) for block in full_candidates],
        [teacher],
    )

    assert len(solutions) == 3
    teacher_by_id = _by_identity(teacher)
    changed_identities = set()
    for solution in solutions:
        replacement = solution["single_sequence_replacement"]
        block = BlockConfig(**replacement["child_block_configs"][0])
        changed = {
            identity
            for identity, subblock in _by_identity(block).items()
            if subblock != teacher_by_id[identity]
        }
        assert len(changed) == 1
        assert changed == {
            (
                solution["subblock_replacement"]["kind"],
                solution["subblock_replacement"]["name"],
            )
        }
        assert len(solution["chosen_replacements"]) == 1
        changed_identities.update(changed)

    assert changed_identities == {
        ("attention", "mixer"),
        ("ffn", "up"),
        ("ffn", "down"),
    }


def test_builds_complete_teacher_context_for_each_layer():
    teachers = [
        BlockConfig(
            subblock_configs=(
                AttentionConfig(num_query_heads=8, num_kv_heads=2),
                FFNConfig(intermediate_size=16),
            )
        ),
        BlockConfig(
            subblock_configs=(
                AttentionConfig(num_query_heads=4, num_kv_heads=1),
                FFNConfig(intermediate_size=12),
            )
        ),
    ]
    candidates = [
        _replacement(0, teachers[0]),
        _replacement(
            0,
            BlockConfig(
                subblock_configs=(
                    AttentionConfig(num_query_heads=4, num_kv_heads=2),
                    FFNConfig(intermediate_size=8),
                )
            ),
        ),
        _replacement(1, teachers[1]),
    ]

    solutions = build_subblock_replacement_solutions(candidates, teachers)

    assert len(solutions) == 2
    for solution in solutions:
        assert [
            replacement["parent_layer_indices"][0]
            for replacement in solution["chosen_replacements"]
        ] == [0, 1]
        assert len(solution["block_configs"]) == 2


def test_rejects_candidate_with_incompatible_subblock_identity_set():
    teacher = BlockConfig(
        subblock_configs=(
            AttentionConfig(name="mixer", num_query_heads=8, num_kv_heads=2),
            FFNConfig(name="ffn", intermediate_size=16),
        )
    )
    incompatible = BlockConfig(
        subblock_configs=(
            AttentionConfig(name="different_mixer", num_query_heads=4, num_kv_heads=2),
            FFNConfig(name="ffn", intermediate_size=16),
        )
    )

    with pytest.raises(ValueError, match="subblock identities"):
        build_subblock_replacement_solutions(
            [_replacement(0, teacher), _replacement(0, incompatible)],
            [teacher],
        )


def test_prepares_annotated_payload_without_mutating_canonical_library():
    from examples.puzzletron.prepare_subblock_replacement_scoring import (
        prepare_subblock_replacement_payload,
    )

    teacher = BlockConfig(
        subblock_configs=(
            AttentionConfig(num_query_heads=8, num_kv_heads=2),
            FFNConfig(intermediate_size=16),
        )
    )
    candidate = BlockConfig(
        subblock_configs=(
            AttentionConfig(num_query_heads=4, num_kv_heads=2),
            FFNConfig(intermediate_size=8),
        )
    )
    library = {
        "version": 2,
        "hidden_width": 1024,
        "teacher_hidden_width": 1024,
        "scenario": "width-1024",
        "entries": [_replacement(0, teacher), _replacement(0, candidate)],
    }

    manifest, solutions = prepare_subblock_replacement_payload(library, [teacher])

    assert len(library["entries"]) == 2
    assert len(solutions) == 2
    assert manifest["canonical_entry_count"] == 2
    assert manifest["subblock_solution_count"] == 2
    assert manifest["full_search_space_preserved"] is True
    assert all(solution["hidden_width"] == 1024 for solution in solutions)
    assert all(solution["scenario"] == "width-1024" for solution in solutions)


def test_scoring_paths_follow_explicit_granularity_without_breaking_legacy_paths():
    cfg = OmegaConf.create(
        {
            "scoring": {
                "granularity": "subblock",
                "solutions_path": "/legacy/solutions.json",
                "output_dir": "/legacy/output",
                "subblock_solutions_path": "/atomic/solutions.json",
                "subblock_output_dir": "/atomic/output",
            }
        }
    )

    solutions_path, output_dir = resolve_scoring_paths(cfg)

    assert str(solutions_path) == "/atomic/solutions.json"
    assert str(output_dir) == "/atomic/output"

    cfg.scoring.granularity = "block"
    solutions_path, output_dir = resolve_scoring_paths(cfg)
    assert str(solutions_path) == "/legacy/solutions.json"
    assert str(output_dir) == "/legacy/output"
