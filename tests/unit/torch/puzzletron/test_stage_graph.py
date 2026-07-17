# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the scheduler-neutral Puzzletron stage registry."""

import pytest

from modelopt.torch.puzzletron.stages.graph import (
    STAGE_REGISTRY,
    STAGE_SPECS,
    StageSpec,
    selected_parent_stage_ids,
    stage_display_name,
    topological_stage_ids,
)


def test_registry_contains_every_public_stage_in_deterministic_topological_order():
    assert tuple(STAGE_REGISTRY) == (
        "convert",
        "tokenize_data",
        "vllm_stats",
        "depth_importance",
        "width_importance",
        "sort",
        "sort_sanity",
        "width_sanity",
        "slicing_sanity",
        "bypass_sanity",
        "bypass",
        "build_library",
        "replacement_scoring",
        "mip",
        "zero_shot_evaluation",
        "aiperf",
        "global_distillation_sanity",
        "global_distillation",
        "post_distillation_evaluation",
    )
    assert topological_stage_ids() == tuple(STAGE_REGISTRY)
    assert all(spec.report_order == index for index, spec in enumerate(STAGE_SPECS))


def test_registry_uses_the_approved_fixed_dependencies():
    assert selected_parent_stage_ids("tokenize_data", {}) == ("convert",)
    assert selected_parent_stage_ids("vllm_stats", {}) == ("convert",)
    assert selected_parent_stage_ids("depth_importance", {}) == ("tokenize_data",)
    assert selected_parent_stage_ids("width_importance", {}) == ("tokenize_data",)
    assert selected_parent_stage_ids("sort", {}) == ("width_importance",)
    for stage in ("sort_sanity", "width_sanity", "slicing_sanity", "bypass_sanity"):
        assert selected_parent_stage_ids(stage, {}) == ("sort",)
    assert selected_parent_stage_ids("bypass", {}) == ("bypass_sanity",)
    assert selected_parent_stage_ids("build_library", {}) == ("bypass",)
    assert selected_parent_stage_ids("replacement_scoring", {}) == ("build_library",)
    assert selected_parent_stage_ids("mip", {}) == (
        "vllm_stats",
        "depth_importance",
        "replacement_scoring",
    )
    assert selected_parent_stage_ids("zero_shot_evaluation", {}) == ("mip",)
    assert selected_parent_stage_ids("aiperf", {}) == ("mip",)
    assert selected_parent_stage_ids("global_distillation_sanity", {}) == ("mip",)
    assert selected_parent_stage_ids("global_distillation", {}) == (
        "global_distillation_sanity",
    )
    assert selected_parent_stage_ids("post_distillation_evaluation", {}) == (
        "global_distillation",
    )


def test_model_executing_sanity_stages_are_distributed():
    assert STAGE_REGISTRY["sort_sanity"].distributed
    assert STAGE_REGISTRY["width_sanity"].distributed
    assert STAGE_REGISTRY["bypass_sanity"].distributed
    assert not STAGE_REGISTRY["slicing_sanity"].distributed


def test_block_library_stage_is_distributed():
    assert STAGE_REGISTRY["build_library"].distributed


def test_stage_labels_use_independent_granularity():
    assert stage_display_name("vllm_stats", granularity="block") == "Block vLLM Stats"
    assert stage_display_name("vllm_stats", granularity="subblock") == "Subblock vLLM Stats"
    assert stage_display_name("bypass", granularity="subblock") == "Subblock Bypass"
    assert (
        stage_display_name("replacement_scoring", granularity="subblock")
        == "Replace-one-subblock Scoring"
    )
    assert stage_display_name("build_library", granularity="subblock") == "Build Block Library"


def test_topological_order_rejects_unknown_parents():
    specs = (StageSpec("child", "Child", parents=("missing",)),)

    with pytest.raises(ValueError, match="unknown parent 'missing' for stage 'child'"):
        topological_stage_ids(specs)


def test_topological_order_rejects_cycles():
    specs = (
        StageSpec("first", "First", parents=("second",)),
        StageSpec("second", "Second", parents=("first",)),
    )

    with pytest.raises(ValueError, match="Stage graph contains a cycle: first, second"):
        topological_stage_ids(specs)
