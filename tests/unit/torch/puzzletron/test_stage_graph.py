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

"""Tests for the scheduler-neutral Puzzletron stage registry."""

import pytest

from modelopt.torch.puzzletron.stages.graph import (
    SHARED_SEMANTIC_CONFIG_SECTIONS,
    STAGE_REGISTRY,
    STAGE_SPECS,
    StageSpec,
    selected_parent_stage_ids,
    semantic_stage_config,
    stage_display_name,
    topological_stage_ids,
)

# Frozen compatibility oracle for the reviewed semantic projection contract.
_EXPECTED_SEMANTIC_CONFIG_SECTIONS = {
    "convert": ("convert",),
    "tokenize_data": (
        "tokenize_data",
        "convert",
        "data",
        "dataset_path",
        "train_token_cache_path",
        "validation_token_cache_path",
        "pruning",
        "replacement_scoring",
        "depth_importance",
        "sort_sanity",
        "width_sanity",
    ),
    "width_importance": ("width_importance", "pruning"),
    "sort": ("sort", "pruning"),
    "sort_sanity": ("sort_sanity", "sanity", "sort", "pruning", "replacement_scoring"),
    "slicing_sanity": (
        "slicing_sanity",
        "sanity",
        "sort",
        "pruning",
        "replacement_scoring",
    ),
    "width_sanity": ("width_sanity", "sanity", "pruning", "replacement_scoring"),
    "bypass_sanity": ("bypass_sanity", "sanity", "bypass", "pruning"),
    "bypass": ("bypass", "pruning"),
    "depth_importance": ("depth_importance", "pruning", "replacement_scoring"),
    "vllm_stats": ("vllm_stats", "build_library", "library"),
    "build_library": ("build_library", "vllm_stats", "library", "bypass"),
    "replacement_scoring": (
        "replacement_scoring",
        "build_library",
        "library",
        "pruning",
    ),
    "mip": ("mip", "realize_model", "replacement_scoring", "vllm_stats", "library", "bypass"),
    "zero_shot_evaluation": ("zero_shot_evaluation", "convert", "replacement_scoring"),
    "aiperf": ("aiperf", "zero_shot_evaluation"),
    "global_distillation_sanity": (
        "global_distillation_sanity",
        "sanity",
        "global_distillation",
        "replacement_scoring",
        "calibration",
    ),
    "global_distillation": (
        "global_distillation",
        "zero_shot_evaluation",
        "replacement_scoring",
    ),
    "post_distillation_evaluation": (
        "post_distillation_evaluation",
        "global_distillation",
        "zero_shot_evaluation",
        "replacement_scoring",
    ),
}


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


def test_registry_owns_explicit_semantic_sections_and_preserves_projection() -> None:
    sections = set(SHARED_SEMANTIC_CONFIG_SECTIONS)
    sections.update(
        section
        for stage_sections in _EXPECTED_SEMANTIC_CONFIG_SECTIONS.values()
        for section in stage_sections
    )
    config = {section: {"value": section} for section in sections}
    config["report"] = {"excluded": True}

    assert set(_EXPECTED_SEMANTIC_CONFIG_SECTIONS) == set(STAGE_REGISTRY)
    for stage_id, expected_sections in _EXPECTED_SEMANTIC_CONFIG_SECTIONS.items():
        spec = STAGE_REGISTRY[stage_id]
        assert spec.semantic_config_sections == expected_sections
        expected_projection = {
            section: config[section]
            for section in dict.fromkeys((*SHARED_SEMANTIC_CONFIG_SECTIONS, *expected_sections))
        }
        assert semantic_stage_config(config, stage_id) == expected_projection


def test_dynamic_stage_semantic_projection_keeps_stage_id_fallback() -> None:
    config = {"model": {"name": "teacher"}, "post.custom": {"threshold": 0.5}}

    assert semantic_stage_config(config, "post.custom") == config


def test_semantic_projection_normalizes_empty_optional_sections() -> None:
    baseline = semantic_stage_config({"build_library": {"enabled": True}}, "build_library")

    assert (
        semantic_stage_config(
            {"build_library": {"enabled": True}, "library": None}, "build_library"
        )
        == baseline
    )
    assert (
        semantic_stage_config({"build_library": {"enabled": True}, "library": {}}, "build_library")
        == baseline
    )
    assert (
        semantic_stage_config(
            {
                "build_library": {"enabled": True},
                "library": {"vllm": {"enabled": True}},
            },
            "build_library",
        )
        != baseline
    )


def test_semantic_projection_uses_authored_config_unless_effective_view_is_requested() -> None:
    config = {
        "model": {"source": "normalized-model"},
        "convert": {"teacher_dir": "normalized-teacher"},
        "_runtime": {
            "authored_config": {
                "model": {"source": "authored-model"},
                "convert": {"teacher_dir": "authored-teacher"},
            }
        },
    }

    assert semantic_stage_config(config, "convert") == {
        "model": {"source": "authored-model"},
        "convert": {"teacher_dir": "authored-teacher"},
    }
    assert semantic_stage_config(config, "convert", use_authored=False) == {
        "model": {"source": "normalized-model"},
        "convert": {"teacher_dir": "normalized-teacher"},
    }


def test_registry_uses_the_approved_fixed_dependencies():
    assert selected_parent_stage_ids("tokenize_data", {}) == ("convert",)
    assert selected_parent_stage_ids("vllm_stats", {}) == ("convert",)
    assert selected_parent_stage_ids("depth_importance", {}) == ("tokenize_data",)
    assert selected_parent_stage_ids("width_importance", {}) == ("tokenize_data",)
    assert selected_parent_stage_ids("sort", {}) == ("width_importance",)
    assert selected_parent_stage_ids("sort_sanity", {}) == ("sort",)
    assert selected_parent_stage_ids("width_sanity", {}) == ("sort_sanity",)
    assert selected_parent_stage_ids("slicing_sanity", {}) == ("width_sanity",)
    assert selected_parent_stage_ids("bypass_sanity", {}) == ("sort",)
    assert selected_parent_stage_ids("bypass", {}) == ("bypass_sanity",)
    assert selected_parent_stage_ids("build_library", {}) == ("bypass",)
    assert selected_parent_stage_ids("build_library", {"vllm_stats": {"enabled": True}}) == (
        "bypass",
        "vllm_stats",
    )
    assert selected_parent_stage_ids("replacement_scoring", {}) == ("build_library",)
    assert selected_parent_stage_ids("mip", {}) == (
        "vllm_stats",
        "depth_importance",
        "replacement_scoring",
    )
    assert selected_parent_stage_ids("zero_shot_evaluation", {}) == ("mip",)
    assert selected_parent_stage_ids("aiperf", {}) == ("mip",)
    assert selected_parent_stage_ids("global_distillation_sanity", {}) == ("mip",)
    assert selected_parent_stage_ids("global_distillation", {}) == ("global_distillation_sanity",)
    assert selected_parent_stage_ids("post_distillation_evaluation", {}) == ("global_distillation",)


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
    specs = (
        StageSpec(
            "child",
            "Child",
            semantic_config_sections=("child",),
            parents=("missing",),
        ),
    )

    with pytest.raises(ValueError, match="unknown parent 'missing' for stage 'child'"):
        topological_stage_ids(specs)


def test_topological_order_rejects_cycles():
    specs = (
        StageSpec(
            "first",
            "First",
            semantic_config_sections=("first",),
            parents=("second",),
        ),
        StageSpec(
            "second",
            "Second",
            semantic_config_sections=("second",),
            parents=("first",),
        ),
    )

    with pytest.raises(ValueError, match="Stage graph contains a cycle: first, second"):
        topological_stage_ids(specs)
