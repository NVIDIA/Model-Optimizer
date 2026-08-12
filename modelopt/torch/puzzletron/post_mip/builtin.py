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

from typing import TYPE_CHECKING, Any

from .base import NodeCapabilities, NodeKind, PostMIPNode, post_mip_node
from .filters import filter_metric_references, validate_filter_config
from .records import ArtifactKind
from .reporting import (
    render_aiperf_report,
    render_downstream_evaluation_report,
    render_evaluation_report,
    render_global_kd_report,
)

if TYPE_CHECKING:
    from collections.abc import Mapping


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
