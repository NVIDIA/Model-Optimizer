# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Registered post-MIP node contracts and campaign flow compilation."""

from __future__ import annotations

import html
from abc import ABC
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar

from ..stages import topological_mapping_items
from .records import ArtifactKind

__all__ = [
    "CompiledPostMIPNode",
    "NodeCapabilities",
    "NodeKind",
    "PostMIPNode",
    "compile_post_mip_flows",
    "post_mip_node",
    "render_post_mip_node_report",
]


class NodeKind(str, Enum):
    SELECTOR = "selector"
    EVALUATOR = "evaluator"
    TRANSFORMER = "transformer"


@dataclass(frozen=True)
class NodeCapabilities:
    kind: NodeKind
    accepts: frozenset[ArtifactKind]
    output: ArtifactKind | None = None
    distributed: bool = False
    implemented: bool = True
    default_strategy: str = "single"


@dataclass(frozen=True)
class CompiledPostMIPNode:
    flow_id: str
    node_id: str
    stage_id: str
    node_type: str
    input_id: str
    parent_stage_id: str
    dependency_stage_ids: tuple[str, ...]
    model_source: str
    config: dict[str, Any]
    capabilities: NodeCapabilities
    input_artifact: str
    output_artifact: str
    metric_references: tuple[str, ...]


_REGISTRY: dict[str, type[PostMIPNode]] = {}


def post_mip_node(cls: type[PostMIPNode]) -> type[PostMIPNode]:
    name = cls.type_name
    if not name or name in _REGISTRY:
        raise ValueError(f"duplicate or empty post-MIP node type {name!r}")
    _REGISTRY[name] = cls
    return cls


class PostMIPNode(ABC):
    type_name: ClassVar[str]
    capabilities: ClassVar[NodeCapabilities]
    common_fields: ClassVar[frozenset[str]] = frozenset(
        {"type", "input", "model_source", "failure_policy", "config"}
    )

    @classmethod
    def validate_config(cls, config: Mapping[str, Any]) -> None:
        unknown = set(config) - cls.common_fields
        if unknown:
            raise ValueError(f"unknown {cls.type_name} node fields: {sorted(unknown)}")
        policy = str(config.get("failure_policy", "record_and_continue"))
        if policy not in {"record_and_continue", "strict"}:
            raise ValueError("failure_policy must be record_and_continue or strict")
        if config.get("config") is not None and not isinstance(config["config"], Mapping):
            raise TypeError(f"{cls.type_name}.config must be a mapping")

    @classmethod
    def metric_references(cls, config: Mapping[str, Any]) -> tuple[str, ...]:
        return ()

    @classmethod
    def render_report(cls, node: CompiledPostMIPNode, payload: Mapping[str, Any]) -> str:
        status = html.escape(str(payload.get("status", "pending")))
        input_count = int(payload.get("input_count", 0))
        output_count = int(payload.get("output_count", 0))
        metrics = ", ".join(str(value) for value in payload.get("metric_names") or ())
        metric_text = f"<p>metrics={html.escape(metrics)}</p>" if metrics else ""
        return (
            f"<h3>{html.escape(node.node_id)}</h3>"
            f"<p>type={html.escape(node.node_type)} · status={status} · "
            f"candidates={input_count}→{output_count} · "
            f"model source={html.escape(node.model_source)}</p>"
            f"{metric_text}"
        )


def _artifact_label(values: set[ArtifactKind]) -> str:
    return next(iter(values)).value if len(values) == 1 else "config_or_checkpoint"


def _validate_id(value: Any, label: str) -> str:
    text = str(value)
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
    if not text or any(character not in allowed for character in text):
        raise ValueError(f"{label} must contain only letters, digits, '_' and '-': {text!r}")
    return text


def compile_post_mip_flows(config: Mapping[str, Any]) -> tuple[CompiledPostMIPNode, ...]:
    """Compile configured flows into validated single-parent dynamic stages."""

    from . import builtin as _builtin  # noqa: F401

    post_mip = config.get("post_mip") or {}
    if not isinstance(post_mip, Mapping):
        raise TypeError("post_mip must be a mapping")
    unknown_root = set(post_mip) - {"flows"}
    if unknown_root:
        raise ValueError(f"unknown post_mip fields: {sorted(unknown_root)}")
    flows = post_mip.get("flows") or {}
    if not isinstance(flows, Mapping):
        raise TypeError("post_mip.flows must be a mapping")
    compiled = []
    global_stage_ids = set()
    global_node_ids = set()
    for flow_id, flow_value in flows.items():
        flow_id = _validate_id(flow_id, "post-MIP flow ID")
        if not isinstance(flow_value, Mapping):
            raise TypeError(f"post-MIP flow {flow_id!r} must be a mapping")
        flow = dict(flow_value)
        unknown_flow = set(flow) - {"source", "nodes"}
        if unknown_flow:
            raise ValueError(f"unknown fields in post-MIP flow {flow_id!r}: {sorted(unknown_flow)}")
        source = flow.get("source") or {}
        if not isinstance(source, Mapping) or not source.get("run"):
            raise ValueError(f"post-MIP flow {flow_id!r} must select one source.run")
        mip_runs = (config.get("mip") or {}).get("runs") or {}
        if source["run"] not in mip_runs or mip_runs[source["run"]] is False:
            raise ValueError(
                f"post-MIP flow {flow_id!r} selects unknown or disabled MIP run "
                f"{source['run']!r}"
            )
        unknown_source = set(source) - {"run", "variants", "objectives"}
        if unknown_source:
            raise ValueError(
                f"unknown source fields in post-MIP flow {flow_id!r}: "
                f"{sorted(unknown_source)}"
            )
        nodes = flow.get("nodes") or {}
        if not isinstance(nodes, Mapping) or not nodes:
            raise ValueError(f"post-MIP flow {flow_id!r} must contain nodes")
        prepared_nodes: dict[str, tuple[dict[str, Any], type[PostMIPNode]]] = {}
        for node_id, node_value in nodes.items():
            node_id = _validate_id(node_id, "post-MIP node ID")
            if not isinstance(node_value, Mapping):
                raise TypeError(f"post-MIP node {flow_id}.{node_id} must be a mapping")
            node_config = dict(node_value)
            if str(node_id) in global_node_ids:
                raise ValueError(
                    f"post-MIP node IDs must be campaign-unique; duplicate {node_id!r}"
                )
            global_node_ids.add(str(node_id))
            node_type = str(node_config.get("type") or "")
            if node_type not in _REGISTRY:
                raise ValueError(f"unknown post-MIP node type {node_type!r}")
            node_class = _REGISTRY[node_type]
            node_class.validate_config(node_config)
            prepared_nodes[node_id] = (node_config, node_class)

        def dependency_ids(
            _node_id: str,
            prepared: tuple[dict[str, Any], type[PostMIPNode]],
        ) -> tuple[str, ...]:
            node_config, node_class = prepared
            dependencies = []
            input_id = str(node_config.get("input", "source"))
            if input_id != "source":
                dependencies.append(input_id)
            model_source = str(node_config.get("model_source", "latest"))
            if model_source not in {"latest", "origin"}:
                dependencies.append(model_source)
            for reference in node_class.metric_references(node_config):
                owner, separator, _metric = reference.partition(".")
                if separator and owner != "mip":
                    dependencies.append(owner)
            return tuple(dependencies)

        artifact_by_node: dict[str, set[ArtifactKind]] = {
            "source": {ArtifactKind.CONFIG, ArtifactKind.CHECKPOINT}
        }
        parent_by_node = {"source": "mip"}
        kind_by_node = {"source": NodeKind.TRANSFORMER}
        for node_id, prepared in topological_mapping_items(prepared_nodes, dependency_ids):
            node_config, node_class = prepared
            node_type = str(node_config["type"])
            input_id = str(node_config.get("input", "source"))
            model_source = str(node_config.get("model_source", "latest"))
            if (
                model_source not in {"latest", "origin"}
                and kind_by_node[model_source] is not NodeKind.TRANSFORMER
            ):
                raise ValueError(f"model_source {model_source!r} is not a transformer node")
            source_artifacts = (
                artifact_by_node[input_id]
                if model_source == "latest"
                else artifact_by_node["source"]
                if model_source == "origin"
                else artifact_by_node[model_source]
            )
            capabilities = node_class.capabilities
            if (
                capabilities.kind is not NodeKind.SELECTOR
                and not source_artifacts <= capabilities.accepts
            ):
                raise ValueError(
                    f"post-MIP node {flow_id}.{node_id} accepts "
                    f"{sorted(item.value for item in capabilities.accepts)}, but model_source "
                    f"{model_source!r} can provide "
                    f"{sorted(item.value for item in source_artifacts)}; "
                    "add an explicit materialize node"
                )
            output_artifacts = (
                {capabilities.output}
                if capabilities.output is not None
                else set(artifact_by_node[input_id])
            )
            stage_id = f"post.{flow_id}.{node_id}"
            if stage_id in global_stage_ids:
                raise ValueError(f"duplicate post-MIP stage ID {stage_id!r}")
            global_stage_ids.add(stage_id)
            metric_references = node_class.metric_references(node_config)
            dependency_stage_ids = [parent_by_node[input_id]]
            for reference in metric_references:
                owner, separator, _metric = reference.partition(".")
                if not separator or (owner != "mip" and owner not in artifact_by_node):
                    raise ValueError(
                        f"post-MIP node {flow_id}.{node_id} has invalid or forward metric "
                        f"reference {reference!r}"
                    )
                if owner != "mip":
                    dependency_stage_ids.append(parent_by_node[owner])
            if model_source not in {"latest", "origin"}:
                dependency_stage_ids.append(parent_by_node[model_source])
            compiled_node = CompiledPostMIPNode(
                flow_id=str(flow_id),
                node_id=str(node_id),
                stage_id=stage_id,
                node_type=node_type,
                input_id=input_id,
                parent_stage_id=parent_by_node[input_id],
                dependency_stage_ids=tuple(dict.fromkeys(dependency_stage_ids)),
                model_source=model_source,
                config=node_config,
                capabilities=capabilities,
                input_artifact=_artifact_label(source_artifacts),
                output_artifact=_artifact_label(output_artifacts),
                metric_references=metric_references,
            )
            compiled.append(compiled_node)
            artifact_by_node[str(node_id)] = output_artifacts
            parent_by_node[str(node_id)] = stage_id
            kind_by_node[str(node_id)] = capabilities.kind
    return tuple(compiled)


def render_post_mip_node_report(
    node: CompiledPostMIPNode, payload: Mapping[str, Any]
) -> str:
    """Render one node using the report hook owned by its registered class."""

    return _REGISTRY[node.node_type].render_report(node, payload)
