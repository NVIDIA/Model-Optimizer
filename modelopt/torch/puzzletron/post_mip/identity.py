# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical execution identities for post-MIP nodes."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..identity import stable_hash
from .base import CompiledPostMIPNode, compile_post_mip_flows
from .records import CandidateLedger, CandidateSet

if __package__.startswith("puzzletron_orchestrator."):
    from puzzletron_orchestrator.adapters.base import ExecutionIdentityProjectionUnavailable
else:
    from ..orchestration.adapters.base import ExecutionIdentityProjectionUnavailable

__all__ = [
    "PostMIPExecutionContractUnavailable",
    "expected_post_mip_candidate_count",
    "expected_post_mip_execution_contract",
    "expected_post_mip_execution_identity",
    "post_mip_execution_contract",
    "post_mip_execution_contract_identity",
    "post_mip_execution_identity",
    "prepare_post_mip_candidate_ledger",
]


class PostMIPExecutionContractUnavailable(ExecutionIdentityProjectionUnavailable):
    """The current upstream artifacts do not yet define a post-MIP execution."""


def _puzzle_dir(config: Mapping[str, Any]) -> Path:
    return Path(config.get("puzzle_dir") or (config.get("experiment") or {})["dir"])


def _compiled_node(config: Mapping[str, Any], stage_id: str) -> CompiledPostMIPNode:
    matches = [node for node in compile_post_mip_flows(config) if node.stage_id == stage_id]
    if len(matches) != 1:
        raise ValueError(f"expected one compiled post-MIP node {stage_id!r}, found {len(matches)}")
    node = matches[0]
    if not node.capabilities.implemented:
        raise NotImplementedError(f"post-MIP node type {node.node_type!r} is not implemented")
    return node


def _input_candidate_set(
    ledger: CandidateLedger,
    config: Mapping[str, Any],
    node: CompiledPostMIPNode,
) -> CandidateSet:
    if node.input_id == "source":
        flow = (config.get("post_mip") or {})["flows"][node.flow_id]
        candidate_set = ledger.root_set(node.flow_id, flow["source"])
    else:
        candidate_set = ledger.load_candidate_set(node.input_id)
    canonical = CandidateSet.create(
        candidate_set.flow_id,
        candidate_set.node_id,
        candidate_set.revision_ids,
        producer_execution_identity=candidate_set.producer_execution_identity,
    )
    if canonical != candidate_set:
        raise RuntimeError(f"post-MIP input node {node.input_id!r} has an invalid candidate set")
    return candidate_set


def _active_mip_contract(config: Mapping[str, Any]) -> tuple[str, set[str]]:
    active_path = _puzzle_dir(config) / "mip" / "active_profiles.json"
    if not active_path.is_file():
        raise PostMIPExecutionContractUnavailable("active MIP profile manifest is unavailable")
    active = json.loads(active_path.read_text())
    if not isinstance(active, Mapping):
        raise TypeError(f"active MIP profile manifest must contain a mapping: {active_path}")
    status = active.get("status")
    if not isinstance(status, str) or not status:
        raise ValueError(f"active MIP profile manifest has an invalid status: {active_path}")
    if status != "success":
        raise PostMIPExecutionContractUnavailable(
            f"active MIP profile manifest is not complete: {active_path}"
        )
    execution_identity = active.get("execution_identity")
    if not isinstance(execution_identity, str) or not execution_identity:
        raise ValueError(f"active MIP profile manifest has no execution identity: {active_path}")
    profile_values = active.get("profile_ids")
    if (
        not isinstance(profile_values, list)
        or not profile_values
        or any(not isinstance(value, str) or not value for value in profile_values)
    ):
        raise ValueError(f"active MIP profile manifest has invalid profile IDs: {active_path}")
    return execution_identity, set(profile_values)


def _expected_post_mip_inputs(
    config: Mapping[str, Any], stage_id: str
) -> tuple[CompiledPostMIPNode, CandidateSet, CandidateLedger]:
    active_execution, active_profiles = _active_mip_contract(config)
    node = _compiled_node(config, stage_id)
    ledger = CandidateLedger(_puzzle_dir(config) / "artifacts" / "post_mip")
    if not ledger.registry_path.is_file():
        raise PostMIPExecutionContractUnavailable("post-MIP candidate registry is unavailable")
    if (
        ledger.active_mip_execution_identity != active_execution
        or ledger.active_profile_ids != active_profiles
    ):
        raise PostMIPExecutionContractUnavailable(
            "post-MIP candidate registry does not reflect the active MIP execution"
        )
    try:
        candidate_set = _input_candidate_set(ledger, config, node)
    except FileNotFoundError as error:
        raise PostMIPExecutionContractUnavailable(
            f"post-MIP inputs for {stage_id!r} are unavailable"
        ) from error
    return node, candidate_set, ledger


def post_mip_execution_contract(
    config: Mapping[str, Any],
    node: CompiledPostMIPNode,
    candidate_set: CandidateSet,
    ledger: CandidateLedger,
) -> dict[str, Any]:
    """Return the exact node, input, dependency, and source-revision contract."""

    dependency_owners = {
        reference.partition(".")[0]
        for reference in node.metric_references
        if not reference.startswith("mip.")
    }
    if node.model_source not in {"latest", "origin"}:
        dependency_owners.add(node.model_source)
    dependency_executions = {}
    for owner in sorted(dependency_owners):
        current_path = (
            _puzzle_dir(config) / "artifacts" / "post_mip" / "nodes" / owner / "current.json"
        )
        dependency_executions[owner] = json.loads(current_path.read_text())["execution_identity"]
    source_revisions = {
        revision_id: ledger.source_revision(revision_id, node.model_source).revision_id
        for revision_id in candidate_set.revision_ids
    }
    return {
        "candidate_set": candidate_set.identity,
        "node": node.config,
        "dependency_executions": dependency_executions,
        "source_revisions": source_revisions,
    }


def post_mip_execution_contract_identity(contract: Mapping[str, Any]) -> str:
    """Hash one already-resolved canonical post-MIP execution contract."""

    return stable_hash(contract, prefix="post_mip_execution")


def post_mip_execution_identity(
    config: Mapping[str, Any],
    node: CompiledPostMIPNode,
    candidate_set: CandidateSet,
    ledger: CandidateLedger,
) -> str:
    """Return the producer identity for one resolved node execution."""

    return post_mip_execution_contract_identity(
        post_mip_execution_contract(config, node, candidate_set, ledger)
    )


def expected_post_mip_execution_contract(
    config: Mapping[str, Any], stage_id: str
) -> dict[str, Any]:
    """Resolve the currently runnable contract for a compiled post-MIP stage."""

    node, candidate_set, ledger = _expected_post_mip_inputs(config, stage_id)
    try:
        return post_mip_execution_contract(config, node, candidate_set, ledger)
    except FileNotFoundError as error:
        raise PostMIPExecutionContractUnavailable(
            f"post-MIP inputs for {stage_id!r} are unavailable"
        ) from error


def expected_post_mip_candidate_count(config: Mapping[str, Any], stage_id: str) -> int:
    """Return the candidate count for the currently runnable node contract."""

    _node, candidate_set, _ledger = _expected_post_mip_inputs(config, stage_id)
    return len(candidate_set.revision_ids)


def prepare_post_mip_candidate_ledger(config: Mapping[str, Any]) -> None:
    """Publish a candidate ledger for the active successful MIP execution if needed."""

    active_execution, active_profiles = _active_mip_contract(config)
    puzzle_dir = _puzzle_dir(config)
    ledger = CandidateLedger(puzzle_dir / "artifacts" / "post_mip")
    if (
        ledger.registry_path.is_file()
        and ledger.active_mip_execution_identity == active_execution
        and ledger.active_profile_ids == active_profiles
    ):
        return
    ledger.ingest_mip(puzzle_dir)
    if (
        ledger.active_mip_execution_identity != active_execution
        or ledger.active_profile_ids != active_profiles
    ):
        raise RuntimeError("post-MIP candidate registry preparation produced stale state")


def expected_post_mip_execution_identity(config: Mapping[str, Any], stage_id: str) -> str:
    """Return the producer identity expected for the current post-MIP inputs."""

    return post_mip_execution_contract_identity(
        expected_post_mip_execution_contract(config, stage_id)
    )
