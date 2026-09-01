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

"""Built-in post-MIP node declarations."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .base import NodeCapabilities, NodeKind, PostMIPNode, post_mip_node
from .filters import filter_metric_references, validate_filter_config
from .records import ArtifactKind
from .reporting import (
    render_aiperf_report,
    render_downstream_evaluation_report,
    render_evaluation_report,
    render_global_kd_report,
)


@post_mip_node
class FilterNode(PostMIPNode):
    type_name = "filter"
    capabilities = NodeCapabilities(NodeKind.SELECTOR, frozenset(ArtifactKind))

    @classmethod
    def validate_config(cls, config: Mapping[str, Any]) -> None:
        validate_filter_config(config)

    @classmethod
    def metric_references(cls, config: Mapping[str, Any]) -> tuple[str, ...]:
        return filter_metric_references(config)


@post_mip_node
class ManualFilterNode(PostMIPNode):
    type_name = "manual_filter"
    capabilities = NodeCapabilities(NodeKind.SELECTOR, frozenset(ArtifactKind))
    common_fields = PostMIPNode.common_fields | {"prompt"}


@post_mip_node
class ResultManifestNode(PostMIPNode):
    type_name = "result_manifest"
    capabilities = NodeCapabilities(NodeKind.SELECTOR, frozenset(ArtifactKind))

    @classmethod
    def validate_config(cls, config: Mapping[str, Any]) -> None:
        super().validate_config(config)
        settings = dict(config.get("config") or {})
        milestones = settings.get("milestones")
        if not isinstance(milestones, list) or not milestones:
            raise ValueError("result_manifest.config.milestones must be a non-empty list")
        steps = []
        for milestone in milestones:
            if not isinstance(milestone, Mapping) or set(milestone) != {
                "steps",
                "kd",
                "evaluation",
            }:
                raise ValueError(
                    "result_manifest milestones require exactly steps, kd, and evaluation"
                )
            steps.append(int(milestone["steps"]))
        if steps != sorted(set(steps)):
            raise ValueError("result_manifest milestone steps must increase strictly")
        if not settings.get("pre_kd_source"):
            raise ValueError("result_manifest.config.pre_kd_source is required")

    @classmethod
    def metric_references(cls, config: Mapping[str, Any]) -> tuple[str, ...]:
        settings = dict(config.get("config") or {})
        owners = [str(settings["pre_kd_source"])]
        for milestone in settings.get("milestones") or ():
            owners.extend((str(milestone["kd"]), str(milestone["evaluation"])))
        return tuple(f"{owner}.manifest_dependency" for owner in dict.fromkeys(owners))


@post_mip_node
class MaterializeNode(PostMIPNode):
    type_name = "materialize"
    capabilities = NodeCapabilities(
        NodeKind.TRANSFORMER,
        frozenset(ArtifactKind),
        output=ArtifactKind.CHECKPOINT,
        distributed=True,
        default_strategy="sharded",
    )


@post_mip_node
class EvaluationNode(PostMIPNode):
    type_name = "evaluation"
    capabilities = NodeCapabilities(
        NodeKind.EVALUATOR,
        frozenset(ArtifactKind),
        distributed=True,
        default_strategy="sharded",
    )

    @classmethod
    def render_report(cls, node, payload):
        return render_evaluation_report(str(payload["section_id"]), payload)


@post_mip_node
class AIPerfNode(PostMIPNode):
    type_name = "aiperf"
    capabilities = NodeCapabilities(
        NodeKind.EVALUATOR,
        frozenset({ArtifactKind.CHECKPOINT}),
        distributed=True,
        default_strategy="sharded",
    )

    @classmethod
    def render_report(cls, node, payload):
        return render_aiperf_report(str(payload["section_id"]), payload)


@post_mip_node
class GlobalKDNode(PostMIPNode):
    type_name = "global_kd"
    common_fields = PostMIPNode.common_fields | {"trajectory", "exposure"}
    capabilities = NodeCapabilities(
        NodeKind.TRANSFORMER,
        frozenset({ArtifactKind.CHECKPOINT}),
        output=ArtifactKind.CHECKPOINT,
        distributed=True,
        default_strategy="sharded",
    )

    @classmethod
    def render_report(cls, node, payload):
        return render_global_kd_report(str(payload["section_id"]), payload)

    @classmethod
    def validate_config(cls, config: Mapping[str, Any]) -> None:
        super().validate_config(config)
        trajectory = config.get("trajectory")
        if trajectory is not None:
            text = str(trajectory)
            allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
            if not text or any(character not in allowed for character in text):
                raise ValueError(
                    "global_kd.trajectory must contain only letters, digits, '_' and '-'"
                )
            settings = dict(config.get("config") or {})
            if settings.get("resume") is not True:
                raise ValueError("trajectory global_kd nodes require config.resume=true")
        exposure = config.get("exposure")
        if exposure is not None:
            if not isinstance(exposure, Mapping):
                raise TypeError("global_kd.exposure must be a mapping")
            settings = dict(config.get("config") or {})
            steps = int(settings.get("max_steps", 0))
            gbs = int(settings.get("global_batch_size", 0))
            expected_examples = steps * gbs
            if int(exposure.get("cumulative_steps", -1)) != steps:
                raise ValueError("global_kd exposure cumulative_steps must equal config.max_steps")
            if int(exposure.get("global_batch_size", -1)) != gbs:
                raise ValueError(
                    "global_kd exposure global_batch_size must equal config.global_batch_size"
                )
            if int(exposure.get("cumulative_examples", -1)) != expected_examples:
                raise ValueError(
                    "global_kd exposure cumulative_examples must equal max_steps * "
                    "global_batch_size"
                )
            if int(exposure.get("max_sample_length", 0)) < 1:
                raise ValueError("global_kd exposure max_sample_length must be positive")
            if float(exposure.get("estimated_cumulative_gpu_hours", 0.0)) <= 0:
                raise ValueError(
                    "global_kd exposure estimated_cumulative_gpu_hours must be positive"
                )


@post_mip_node
class PTQNode(PostMIPNode):
    type_name = "ptq"
    capabilities = NodeCapabilities(
        NodeKind.TRANSFORMER,
        frozenset({ArtifactKind.CHECKPOINT}),
        output=ArtifactKind.CHECKPOINT,
        distributed=True,
        implemented=False,
        default_strategy="sharded",
    )


@post_mip_node
class DownstreamEvaluationNode(PostMIPNode):
    type_name = "downstream_evaluation"
    capabilities = NodeCapabilities(
        NodeKind.EVALUATOR,
        frozenset({ArtifactKind.CHECKPOINT}),
        distributed=True,
        default_strategy="sharded",
    )

    @classmethod
    def render_report(cls, node, payload):
        return render_downstream_evaluation_report(str(payload["section_id"]), payload)
