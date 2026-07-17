# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

from omegaconf import OmegaConf

from modelopt.torch.puzzletron.block_config import (
    AttentionConfig,
    BlockConfig,
    MoEConfig,
)
from modelopt.torch.puzzletron.mip.run_puzzle import (
    PuzzleConstraints,
    run_single_puzzle_config,
)
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
    homogeneous = json.loads(
        solution_path.with_name("homogeneous_solutions.json").read_text()
    )
    assert len(homogeneous) == 1
    assert homogeneous[0]["homogeneous_assignment"] == {
        "moe.expert_intermediate_size": 8,
        "moe.num_experts": 4,
        "moe.top_k": 2,
    }
