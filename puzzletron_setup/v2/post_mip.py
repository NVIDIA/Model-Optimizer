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

"""Editable, compiler-validated post-MIP flow DAGs."""

# This module predates repository-wide public-API docstring enforcement.
# ruff: noqa: D101, D102, D107

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .resources import StageResources

__all__ = [
    "IMPLEMENTED_NODE_TYPES",
    "RESERVED_NODE_TYPES",
    "FlowDraft",
    "FlowReview",
    "NodeDraft",
    "PostMIPFlowEditor",
    "recommended_flow",
]

IMPLEMENTED_NODE_TYPES = (
    "filter",
    "manual_filter",
    "materialize",
    "evaluation",
    "aiperf",
    "downstream_evaluation",
    "global_kd",
)
RESERVED_NODE_TYPES = ("ptq",)


@dataclass(frozen=True)
class NodeDraft:
    """One post-MIP node plus its separately rendered resource card."""

    node_id: str
    node_type: str
    input_id: str = "source"
    model_source: str = "latest"
    failure_policy: str = "record_and_continue"
    config: Mapping[str, Any] = field(default_factory=dict)
    selector: Mapping[str, Any] = field(default_factory=dict)
    prompt: str | None = None
    resources: StageResources | None = None

    def __post_init__(self) -> None:
        if self.node_type in RESERVED_NODE_TYPES:
            raise ValueError(
                f"post-MIP node type {self.node_type!r} is reserved and not implemented"
            )
        if self.node_type not in IMPLEMENTED_NODE_TYPES:
            raise ValueError(f"unknown post-MIP node type {self.node_type!r}")

    def to_config(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "type": self.node_type,
            **deepcopy(dict(self.selector)),
        }
        if self.input_id != "source":
            payload["input"] = self.input_id
        if self.model_source != "latest":
            payload["model_source"] = self.model_source
        if self.failure_policy != "record_and_continue":
            payload["failure_policy"] = self.failure_policy
        if self.config:
            payload["config"] = deepcopy(dict(self.config))
        if self.prompt is not None:
            payload["prompt"] = self.prompt
        return payload


@dataclass(frozen=True)
class FlowDraft:
    flow_id: str
    run_id: str
    variants: Any = "all"
    objectives: Any = "all"
    nodes: OrderedDict[str, NodeDraft] = field(default_factory=OrderedDict)

    def to_config(self) -> dict[str, Any]:
        return {
            "source": {
                "run": self.run_id,
                "variants": deepcopy(self.variants),
                "objectives": deepcopy(self.objectives),
            },
            "nodes": OrderedDict(
                (node_id, node.to_config()) for node_id, node in self.nodes.items()
            ),
        }


@dataclass(frozen=True)
class FlowReview:
    flow_id: str
    node_order: tuple[str, ...]
    parents: Mapping[str, tuple[str, ...]]
    artifacts: Mapping[str, str]
    resources: Mapping[str, StageResources]


class PostMIPFlowEditor:
    """CRUD, branching, and canonical DAG validation for post-MIP flows."""

    def __init__(self, mip_runs: Mapping[str, Any] | None = None) -> None:
        self._flows: OrderedDict[str, FlowDraft] = OrderedDict()
        self._mip_runs = deepcopy(dict(mip_runs or {}))

    def flows(self) -> Mapping[str, FlowDraft]:
        return OrderedDict(self._flows)

    def flow(self, flow_id: str) -> FlowDraft:
        return self._flows[flow_id]

    def add_flow(self, flow: FlowDraft) -> None:
        if flow.flow_id in self._flows:
            raise ValueError(f"duplicate post-MIP flow {flow.flow_id!r}")
        duplicate_nodes = set(flow.nodes) & {
            node_id for existing in self._flows.values() for node_id in existing.nodes
        }
        if duplicate_nodes:
            raise ValueError(
                f"post-MIP node IDs must be campaign-unique: {sorted(duplicate_nodes)}"
            )
        self._flows[flow.flow_id] = flow

    def clone_flow(self, source_id: str, target_id: str, *, node_prefix: str = "") -> FlowDraft:
        source = deepcopy(self._flows[source_id])
        nodes = OrderedDict(
            (
                f"{node_prefix}{node_id}",
                replace(
                    node,
                    node_id=f"{node_prefix}{node_id}",
                    input_id=(
                        f"{node_prefix}{node.input_id}"
                        if node.input_id in source.nodes
                        else node.input_id
                    ),
                    model_source=(
                        f"{node_prefix}{node.model_source}"
                        if node.model_source in source.nodes
                        else node.model_source
                    ),
                ),
            )
            for node_id, node in source.nodes.items()
        )
        clone = replace(source, flow_id=target_id, nodes=nodes)
        self.add_flow(clone)
        return clone

    def delete_flow(self, flow_id: str) -> None:
        del self._flows[flow_id]

    def add_node(self, flow_id: str, node: NodeDraft, *, position: int | None = None) -> None:
        if any(node.node_id in flow.nodes for flow in self._flows.values()):
            raise ValueError(f"duplicate post-MIP node {node.node_id!r}")
        flow = self._flows[flow_id]
        items = list(flow.nodes.items())
        insertion = len(items) if position is None else int(position)
        items.insert(insertion, (node.node_id, node))
        self._flows[flow_id] = replace(flow, nodes=OrderedDict(items))

    def edit_node(self, flow_id: str, node_id: str, **changes: Any) -> NodeDraft:
        flow = self._flows[flow_id]
        updated = replace(flow.nodes[node_id], **changes)
        nodes = OrderedDict(flow.nodes)
        nodes[node_id] = updated
        self._flows[flow_id] = replace(flow, nodes=nodes)
        return updated

    def clone_node(self, flow_id: str, node_id: str, target_id: str) -> NodeDraft:
        clone = replace(deepcopy(self._flows[flow_id].nodes[node_id]), node_id=target_id)
        self.add_node(flow_id, clone)
        return clone

    def move_node(self, flow_id: str, node_id: str, position: int) -> None:
        flow = self._flows[flow_id]
        node = flow.nodes[node_id]
        items = [(name, value) for name, value in flow.nodes.items() if name != node_id]
        items.insert(int(position), (node_id, node))
        self._flows[flow_id] = replace(flow, nodes=OrderedDict(items))

    def dependents(self, flow_id: str, node_id: str) -> tuple[str, ...]:
        dependents = []
        for name, node in self._flows[flow_id].nodes.items():
            references = {node.input_id, node.model_source}
            for value in node.selector.values():
                candidates = value if isinstance(value, list) else (value,)
                for candidate in candidates:
                    if isinstance(candidate, Mapping):
                        metric = candidate.get("metric")
                    else:
                        metric = candidate if isinstance(candidate, str) else None
                    if metric:
                        references.add(str(metric).partition(".")[0])
            if node_id in references:
                dependents.append(name)
        return tuple(dependents)

    def delete_node(self, flow_id: str, node_id: str) -> tuple[str, ...]:
        dependents = self.dependents(flow_id, node_id)
        if dependents:
            return dependents
        flow = self._flows[flow_id]
        nodes = OrderedDict((name, value) for name, value in flow.nodes.items() if name != node_id)
        self._flows[flow_id] = replace(flow, nodes=nodes)
        return ()

    def redirect_node(self, flow_id: str, node_id: str, new_input: str) -> None:
        self.edit_node(flow_id, node_id, input_id=new_input)

    def to_config(self) -> OrderedDict[str, Any]:
        return OrderedDict((flow_id, flow.to_config()) for flow_id, flow in self._flows.items())

    def compile(self, mip_runs: Mapping[str, Any] | None = None):
        # Keep the setup package usable without PyTorch until canonical validation.
        from modelopt.torch.puzzletron.post_mip.base import compile_post_mip_flows

        runs = deepcopy(dict(self._mip_runs if mip_runs is None else mip_runs))
        return compile_post_mip_flows(
            {"mip": {"runs": runs}, "post_mip": {"flows": self.to_config()}}
        )

    def review(
        self,
        flow_id: str,
        mip_runs: Mapping[str, Any] | None = None,
    ) -> FlowReview:
        compiled = [node for node in self.compile(mip_runs) if node.flow_id == flow_id]
        flow = self._flows[flow_id]
        return FlowReview(
            flow_id=flow_id,
            node_order=tuple(node.node_id for node in compiled),
            parents={
                node.node_id: tuple(
                    dict.fromkeys(
                        ("mip" if parent == "mip" else parent.rsplit(".", 1)[-1])
                        for parent in node.dependency_stage_ids
                    )
                )
                for node in compiled
            },
            artifacts={node.node_id: node.output_artifact for node in compiled},
            resources={
                node_id: node.resources
                for node_id, node in flow.nodes.items()
                if node.resources is not None
            },
        )


def recommended_flow(
    run_id: str,
    objective_metrics: Sequence[str],
    data: Mapping[str, Any],
    serving: Mapping[str, Any],
    *,
    quality_comparison: Mapping[str, Any] | None = None,
    node_prefix: str = "",
) -> FlowDraft:
    """Build the recommended flow, optionally including a downstream comparison."""
    sequence_length = int(data.get("sequence_length", 4096))
    raw_concurrency = serving.get("concurrency", [1])
    if isinstance(raw_concurrency, (int, str)):
        concurrency = [int(raw_concurrency)]
    else:
        concurrency = [int(item) for item in raw_concurrency]
    online_eval = f"{node_prefix}online_eval"
    best_lm = f"{node_prefix}best_lm"
    materialized = f"{node_prefix}materialized"
    serving_id = f"{node_prefix}serving"
    fastest = f"{node_prefix}fastest"
    short_kd = f"{node_prefix}short_kd"
    final_eval = f"{node_prefix}final_eval"
    best = f"{node_prefix}best"
    quality_benchmarks = f"{node_prefix}quality_benchmarks"
    nodes = OrderedDict(
        (
            (
                online_eval,
                NodeDraft(
                    online_eval,
                    "evaluation",
                    config={"eval_samples": 128, "block_size": sequence_length},
                ),
            ),
            (
                best_lm,
                NodeDraft(
                    best_lm,
                    "filter",
                    input_id=online_eval,
                    selector={
                        "mode": "top_k",
                        "metric": f"{online_eval}.lm_loss",
                        "direction": "minimize",
                        "top_k": 32,
                    },
                ),
            ),
            (
                materialized,
                NodeDraft(materialized, "materialize", input_id=best_lm),
            ),
            (
                serving_id,
                NodeDraft(
                    serving_id,
                    "aiperf",
                    input_id=materialized,
                    config={
                        "input_tokens": int(serving.get("input_tokens", 4096)),
                        "output_tokens": int(serving.get("output_tokens", 1024)),
                        "concurrency": concurrency,
                        "request_count": int(serving.get("request_count", 32)),
                        "use_server_token_count": True,
                        "benchmark_timeout": 900,
                        **(
                            {"topology": deepcopy(serving["topology"])}
                            if serving.get("topology")
                            else {}
                        ),
                    },
                ),
            ),
            (
                fastest,
                NodeDraft(
                    fastest,
                    "filter",
                    input_id=serving_id,
                    selector={
                        "mode": "top_k",
                        "metric": f"{serving_id}.output_token_throughput",
                        "direction": "maximize",
                        "top_k": 4,
                        "best_selection_mode": str(
                            serving.get("best_selection_mode", "individual_best")
                        ),
                    },
                ),
            ),
            (
                short_kd,
                NodeDraft(
                    short_kd,
                    "global_kd",
                    input_id=fastest,
                    config={
                        "max_steps": 128,
                        "global_batch_size": 128,
                    },
                ),
            ),
            (
                final_eval,
                NodeDraft(
                    final_eval,
                    "evaluation",
                    input_id=short_kd,
                    config={"eval_samples": 128, "block_size": sequence_length},
                ),
            ),
            (
                best,
                NodeDraft(
                    best,
                    "filter",
                    input_id=final_eval,
                    selector={
                        "mode": "top_k",
                        "metric": f"{final_eval}.lm_loss",
                        "direction": "minimize",
                        "top_k": 1,
                    },
                ),
            ),
        )
    )
    if quality_comparison is not None:
        nodes[quality_benchmarks] = NodeDraft(
            quality_benchmarks,
            "downstream_evaluation",
            input_id=best,
            failure_policy="strict",
            config=deepcopy(dict(quality_comparison)),
        )
    return FlowDraft(
        flow_id=run_id,
        run_id=run_id,
        variants="all",
        objectives=list(objective_metrics) or "all",
        nodes=nodes,
    )
