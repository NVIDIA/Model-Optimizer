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

import json

import pytest
from omegaconf import OmegaConf

from modelopt.torch.puzzletron.block_config import (
    AttentionConfig,
    BlockConfig,
    MambaConfig,
    MoEConfig,
)
from modelopt.torch.puzzletron.mip.mip_with_multi_layer_replacements import run_mip
from modelopt.torch.puzzletron.mip.run_puzzle import PuzzleConstraints, run_single_puzzle_config
from modelopt.torch.puzzletron.mip.search_space import (
    filter_replacements_by_axes,
    rank_homogeneous_solutions,
)
from modelopt.torch.puzzletron.replacement_library.replacement_utils import (
    extract_block_configs_and_locations,
)


def _block(*subblocks):
    return BlockConfig(subblock_configs=tuple(subblocks))


def _replacement(layer, block, loss, params, *, teacher=False):
    return {
        "parent_layer_indices": [layer],
        "block_idx": layer,
        "block_config": block,
        "child_block_configs": [block],
        "is_teacher": teacher,
        "metrics": {"loss": loss},
        "stats": {"num_params": params},
    }


def _metrics():
    teacher = _block(
        AttentionConfig(num_kv_heads=2, num_query_heads=32),
        MoEConfig(num_experts=128, expert_intermediate_size=1856, top_k=6),
    )
    small = _block(
        AttentionConfig(num_kv_heads=1, num_query_heads=8),
        MoEConfig(num_experts=96, expert_intermediate_size=1344, top_k=2),
    )
    return {
        "l0-teacher": _replacement(0, teacher, 0.1, 100, teacher=True),
        "l0-small": _replacement(0, small, 0.4, 60),
        "l1-teacher": _replacement(1, teacher, 0.2, 100, teacher=True),
        "l1-small": _replacement(1, small, 0.3, 60),
    }


def test_axis_filter_supports_teacher_default_all_and_derived_q_per_group():
    filtered = filter_replacements_by_axes(
        _metrics(),
        axes_default="teacher",
        axis_options={
            "n_routed_experts": "all",
            "moe_intermediate_size": "all",
            "num_experts_per_tok": "all",
            "num_key_value_heads": "all",
            "q_per_group": [8],
        },
    )

    assert set(filtered) == {"l0-small", "l1-small"}


def test_unrestricted_axis_filter_preserves_legacy_metrics_without_teacher():
    metrics = {"candidate": _replacement(0, _block(MoEConfig(top_k=2)), 0.1, 5)}

    assert filter_replacements_by_axes(metrics) == metrics


def test_axis_filter_binds_a_nonteacher_value_to_one_layer():
    metrics = _metrics()
    teacher = metrics["l1-teacher"]["block_config"]
    metrics["l1-kv-small"] = _replacement(
        1,
        _block(
            AttentionConfig(num_kv_heads=1, num_query_heads=16),
            teacher.require_subblock("moe"),
        ),
        0.3,
        80,
    )
    filtered = filter_replacements_by_axes(
        metrics,
        axes_default="teacher",
        axis_options={
            "attention.num_kv_heads": {
                "default": "teacher",
                "layers": {1: [1]},
            }
        },
    )

    assert set(filtered) == {"l0-teacher", "l1-kv-small"}


def test_axis_filter_supports_mamba_num_groups():
    metrics = {
        "teacher": _replacement(
            0,
            _block(MambaConfig(num_heads=4, num_groups=2, head_dim=8, state_dim=16)),
            0.1,
            100,
            teacher=True,
        ),
        "small": _replacement(
            0,
            _block(MambaConfig(num_heads=4, num_groups=1, head_dim=8, state_dim=16)),
            0.2,
            90,
        ),
    }

    filtered = filter_replacements_by_axes(
        metrics,
        axis_options={"mamba.num_groups": [1]},
    )

    assert set(filtered) == {"small"}


def test_axis_filter_supports_range_selector():
    filtered = filter_replacements_by_axes(
        _metrics(),
        axis_options={"moe.top_k": {"range": [1, 2]}},
    )

    assert set(filtered) == {"l0-small", "l1-small"}


def test_axis_filter_rejects_layer_scope_where_axis_is_absent():
    with pytest.raises(ValueError, match=r"not present in teacher blocks at layers \[3\]"):
        filter_replacements_by_axes(
            _metrics(),
            axes_default="teacher",
            axis_options={
                "attention.num_kv_heads": {
                    "default": "teacher",
                    "layers": {3: [1]},
                }
            },
        )


@pytest.mark.parametrize("layer_key", [True, 1.0, "1"])
def test_axis_filter_rejects_non_integer_layer_keys(layer_key):
    with pytest.raises(ValueError, match="layer keys must be integers"):
        filter_replacements_by_axes(
            _metrics(),
            axes_default="teacher",
            axis_options={
                "attention.num_kv_heads": {
                    "default": "teacher",
                    "layers": {layer_key: [1]},
                }
            },
        )


def test_homogeneous_topk_is_ranked_separately_and_honors_constraints():
    solutions = rank_homogeneous_solutions(
        _metrics(),
        objective="metrics.loss",
        constraints={"stats.num_params": 160},
        bigger_is_better=False,
        num_solutions=1,
    )

    assert len(solutions) == 1
    assert solutions[0]["homogeneous_assignment"] == {
        "moe.expert_intermediate_size": 1344,
        "moe.num_experts": 96,
        "moe.top_k": 2,
        "attention.num_kv_heads": 1,
        "attention.q_per_group": 8,
    }
    assert solutions[0]["total_value"] == 0.7
    assert solutions[0]["total_costs"] == {"stats.num_params": 120}


def test_homogeneous_minus_one_retains_every_feasible_assignment():
    solutions = rank_homogeneous_solutions(
        _metrics(),
        objective="metrics.loss",
        constraints={"stats.num_params": 250},
        bigger_is_better=False,
        num_solutions=-1,
    )

    assert len(solutions) == 2


def test_mip_solution_pool_enforces_layer_hamming_distance():
    solutions = run_mip(
        _metrics(),
        objective="metrics.loss",
        constraints={},
        bigger_is_better=False,
        num_solutions=2,
        min_hamming_distance=2,
    )

    assert [solution["solution_rank"] for solution in solutions] == [0, 1]
    assert [solution["total_value"] for solution in solutions] == pytest.approx([0.3, 0.7])
    assert all(len(solution["chosen_replacements"]) == 2 for solution in solutions)


def test_extract_block_configs_accepts_canonical_candidate_without_legacy_alias():
    block = _block(MoEConfig(num_experts=4, expert_intermediate_size=8, top_k=2))
    candidate = {
        "parent_layer_indices": [0],
        "block_config": block,
        "layer_replacement": {
            "parent_layer_indices": [0],
            "child_block_configs": [block],
            "weight_paths": [],
        },
    }

    block_configs, locations = extract_block_configs_and_locations([candidate])

    assert block_configs == [block]
    assert locations == [(candidate, 0)]


def test_single_puzzle_writes_separate_homogeneous_topk(tmp_path):
    teacher = _block(MoEConfig(num_experts=8, expert_intermediate_size=16, top_k=2))
    small = _block(MoEConfig(num_experts=4, expert_intermediate_size=8, top_k=2))
    replacements = {
        "l0-teacher": _replacement(0, teacher, 0.1, 10, teacher=True),
        "l0-small": _replacement(0, small, 0.4, 5),
        "l1-teacher": _replacement(1, teacher, 0.2, 10, teacher=True),
        "l1-small": _replacement(1, small, 0.3, 5),
    }
    stats = [
        {
            "args": {"batch_size": 1, "generation_seq_len": 1},
            "non_block": {"num_params": 0},
            "subblocks": [
                {
                    "subblock_config_class": "MoEConfig",
                    "subblock_config": block.require_subblock("moe").to_dict(),
                    "parent_layer_index": layer,
                    "num_params": params,
                }
                for layer in (0, 1)
                for block, params in ((teacher, 10), (small, 5))
            ],
        }
    ]
    args = OmegaConf.create(
        {
            "objective": "metrics.loss",
            "bigger_is_better": False,
            "materialization_tp": 1,
            "report_additional_costs": ["stats.num_params"],
            "num_homogeneous_solutions": 1,
            "axes_default": "all",
            "axis_options": {},
        }
    )

    solution_path = run_single_puzzle_config(
        args,
        replacements,
        stats,
        {"batch_size": 1, "generation_seq_len": 1},
        PuzzleConstraints(
            type=PuzzleConstraints.Type.MIP,
            constraints={"stats.num_params": 10},
        ),
        tmp_path,
    )

    assert len(json.loads(solution_path.read_text())) == 1
    homogeneous = json.loads(solution_path.with_name("homogeneous_solutions.json").read_text())
    assert len(homogeneous) == 1
    assert homogeneous[0]["homogeneous_assignment"] == {
        "moe.expert_intermediate_size": 8,
        "moe.num_experts": 4,
        "moe.top_k": 2,
    }
