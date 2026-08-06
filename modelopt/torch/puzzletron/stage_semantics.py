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

"""Dependency-free semantic configuration shared by stage consumers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = ["semantic_stage_config"]

_SHARED_SEMANTIC_CONFIG_SECTIONS = (
    "model",
    "data",
    "dataset",
    "parallel",
    "search_space",
    "embedding_pruning",
    "granularity",
    "capability_validation",
)

_STAGE_SEMANTIC_CONFIG_SECTIONS = {
    "convert": ("convert",),
    "tokenize_data": (
        "tokenize_data",
        "convert",
        "dataset_path",
        "pruning",
        "replacement_scoring",
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


def semantic_stage_config(config: Mapping[str, Any], stage_id: str) -> dict[str, Any]:
    """Return configuration that can change the semantic result of one stage."""

    sections = dict.fromkeys(
        (
            *_SHARED_SEMANTIC_CONFIG_SECTIONS,
            *_STAGE_SEMANTIC_CONFIG_SECTIONS.get(stage_id, (stage_id,)),
        )
    )
    return {key: config[key] for key in sections if key in config}
