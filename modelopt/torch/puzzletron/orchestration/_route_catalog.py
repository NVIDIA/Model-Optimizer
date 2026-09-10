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

"""Maintained model and workflow routes for concise Puzzletron recipes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = ["MODEL_IDS", "ROUTES", "ROUTES_BY_KEY", "RouteProfile"]


@dataclass(frozen=True)
class RouteProfile:
    """One maintained logical workflow; no site facts live here."""

    model: str
    workflow: str
    mode: str
    search: str
    evaluation: str
    distillation: str
    experiment_template: str
    execution_stages: Mapping[str, Mapping[str, Any]]
    requires_data: bool = False

    @property
    def route_id(self) -> str:
        return f"{self.model}/{self.workflow}/{self.mode}"


_TEXT_SMOKE_STAGES: dict[str, dict[str, Any]] = {
    "replacement_scoring": {"strategy": "single"},
}

_VLM_SMOKE_STAGES: dict[str, dict[str, Any]] = {
    "replacement_scoring": {"strategy": "single"},
}

_VLM_CAMPAIGN_STAGES: dict[str, dict[str, Any]] = {
    "depth_importance": {"strategy": "single"},
    "replacement_scoring": {"instances": 8},
    "post.candidates.image_eval": {"instances": 5},
    "post.candidates.materialized": {"instances": 5},
    "post.candidates.pre_kd_eval": {"instances": 5},
    "post.candidates.kd": {"instances": 5},
    "post.candidates.post_kd_eval": {"instances": 5},
}

_FOUR_B_CAMPAIGN_STAGES: dict[str, dict[str, Any]] = {
    "depth_importance": {"strategy": "single"},
    "replacement_scoring": {"strategy": "persistent_pool", "instances": 8},
    "post.candidate-evaluation.online_eval": {"instances": 8},
    "post.candidate-evaluation.materialized": {"instances": 4},
    "post.candidate-evaluation.pre_kd_quality": {"instances": 4},
    "post.candidate-evaluation.serving_smoke": {"instances": 4},
    "post.candidate-evaluation.kd_128": {"instances": 4},
    "post.candidate-evaluation.final_eval": {"instances": 4},
    "post.candidate-evaluation.quality_benchmarks": {"instances": 4},
    "post.candidate-evaluation.student_performance": {"instances": 4},
}

ROUTES = (
    RouteProfile(
        model="qwen3.5-0.8b",
        workflow="text-pruning",
        mode="smoke",
        search="bounded-ffn",
        evaluation="smoke",
        distillation="smoke",
        experiment_template="families/qwen3_5/qwen3p5_0p8b/runs/full_smoke.yaml",
        execution_stages=_TEXT_SMOKE_STAGES,
        requires_data=True,
    ),
    RouteProfile(
        model="qwen3.5-0.8b",
        workflow="vlm-pruning",
        mode="smoke",
        search="bounded-ffn",
        evaluation="smoke",
        distillation="smoke",
        experiment_template="families/qwen3_5/qwen3p5_0p8b/runs/vlm_smoke.yaml",
        execution_stages=_VLM_SMOKE_STAGES,
    ),
    RouteProfile(
        model="qwen3.5-0.8b",
        workflow="vlm-pruning",
        mode="campaign",
        search="multi-axis",
        evaluation="quality",
        distillation="short",
        experiment_template="families/qwen3_5/qwen3p5_0p8b/runs/vlm_campaign.yaml",
        execution_stages=_VLM_CAMPAIGN_STAGES,
    ),
    RouteProfile(
        model="qwen3.5-4b",
        workflow="vlm-pruning",
        mode="smoke",
        search="bounded-all-axis",
        evaluation="smoke",
        distillation="smoke",
        experiment_template="families/qwen3_5/qwen3p5_4b/runs/vlm_smoke.yaml",
        execution_stages={
            "depth_importance": {"strategy": "single"},
            "replacement_scoring": {"strategy": "single"},
            "post.params-80.image_eval": {"instances": 2},
        },
        requires_data=True,
    ),
    RouteProfile(
        model="qwen3.5-4b",
        workflow="vlm-pruning",
        mode="campaign",
        search="multi-axis-exact-four",
        evaluation="quality",
        distillation="matched-kd128",
        experiment_template="families/qwen3_5/qwen3p5_4b/runs/all_axis_kd_search.yaml",
        execution_stages=_FOUR_B_CAMPAIGN_STAGES,
        requires_data=True,
    ),
)

ROUTES_BY_KEY = {(route.model, route.workflow, route.mode): route for route in ROUTES}
MODEL_IDS = tuple(sorted({route.model for route in ROUTES}))
