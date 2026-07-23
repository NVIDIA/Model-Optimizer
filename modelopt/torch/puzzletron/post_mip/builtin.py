# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Built-in post-MIP node declarations."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .base import NodeCapabilities, NodeKind, PostMIPNode, post_mip_node
from .filters import filter_metric_references, validate_filter_config
from .records import ArtifactKind


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


@post_mip_node
class AIPerfNode(PostMIPNode):
    type_name = "aiperf"
    capabilities = NodeCapabilities(
        NodeKind.EVALUATOR,
        frozenset({ArtifactKind.CHECKPOINT}),
        distributed=True,
        default_strategy="sharded",
    )


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
        implemented=False,
        default_strategy="sharded",
    )
