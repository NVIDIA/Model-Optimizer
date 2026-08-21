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

"""Tests for post-MIP controller execution identities."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from puzzletron_orchestrator.adapters.registry import adapter_for_stage
from puzzletron_orchestrator.adapters.stage_compat import stage_is_complete
from puzzletron_orchestrator.controller import CampaignController
from puzzletron_orchestrator.post_mip.base import compile_post_mip_flows
from puzzletron_orchestrator.post_mip.identity import (
    PostMIPExecutionContractUnavailable,
    expected_post_mip_execution_contract,
)
from puzzletron_orchestrator.post_mip.records import (
    ArchitectureCandidate,
    ArtifactKind,
    CandidateLedger,
    CandidateSet,
    NodeObservation,
)
from puzzletron_orchestrator.schema import (
    CampaignPlan,
    ExecutionContract,
    ExecutionStrategy,
    FailurePolicy,
    JobHandle,
    JobState,
    RunnerEnvironment,
    StagePlanNode,
)
from puzzletron_orchestrator.state import PersistedAttempt, StageRunRecord


class _TrackingExecutor:
    backend = "fake"

    def __init__(self) -> None:
        self.attempts = []

    def submit(self, attempt):
        self.attempts.append(attempt)
        return JobHandle(
            backend=self.backend,
            handle_id=f"fake-{attempt.attempt_id}",
            attempt_id=attempt.attempt_id,
        )


def _identity_fixture(tmp_path: Path) -> tuple[dict, CandidateLedger, tuple[str, ...]]:
    puzzle_dir = tmp_path / "run"
    puzzle_dir.mkdir()
    config = {
        "puzzle_dir": str(puzzle_dir),
        "mip": {"runs": {"tiny": {}}},
        "post_mip": {
            "flows": {
                "params": {
                    "source": {"run": "tiny"},
                    "nodes": {
                        "score": {"type": "evaluation", "config": {"eval_samples": 2}},
                        "select": {
                            "type": "filter",
                            "input": "score",
                            "mode": "top_k",
                            "metric": "score.loss",
                            "top_k": 2,
                            "config": {"label": "baseline"},
                        },
                        "materialize": {"type": "materialize", "input": "select"},
                        "final": {
                            "type": "evaluation",
                            "input": "select",
                            "model_source": "materialize",
                        },
                    },
                }
            }
        },
    }
    (puzzle_dir / "mip").mkdir()
    (puzzle_dir / "mip" / "active_profiles.json").write_text(
        '{"status":"success","execution_identity":"mip-a","profile_ids":["p0"]}\n'
    )
    ledger = CandidateLedger(puzzle_dir / "artifacts" / "post_mip")
    ledger.active_mip_execution_identity = "mip-a"
    ledger.active_profile_ids = {"p0"}
    root_ids = []
    for index in range(3):
        architecture_id = f"architecture-{index}"
        revision = ledger.add_revision(
            architecture_id=architecture_id,
            artifact_kind=ArtifactKind.CONFIG,
            artifact={"kind": "heterogeneous", "mip_metrics": {"loss": float(index)}},
            parent_revision_id=None,
            producer_node="mip",
        )
        ledger.architectures[architecture_id] = ArchitectureCandidate(
            architecture_id=architecture_id,
            block_configs=[],
            mip_metrics={"loss": float(index)},
            origins=[
                {
                    "profile_id": "p0",
                    "mip_execution_identity": "mip-a",
                    "run_id": "tiny",
                    "variant_id": "base",
                    "objective": {"metric": "params"},
                    "kind": "heterogeneous",
                    "rank": index,
                    "revision_id": revision.revision_id,
                }
            ],
            origin_revision_id=revision.revision_id,
        )
        root_ids.append(revision.revision_id)
    roots = tuple(root_ids)
    ledger.publish()
    score_set = CandidateSet.create(
        "params", "score", roots[:2], producer_execution_identity="score-a"
    )
    ledger.publish_node(
        "score",
        [
            NodeObservation(
                node_id="score",
                input_revision_id=revision_id,
                source_revision_id=revision_id,
                output_revision_id=revision_id,
                status="success",
                metrics={"loss": float(index)},
            )
            for index, revision_id in enumerate(roots[:2])
        ],
        score_set,
        "score-a",
    )
    select_set = CandidateSet.create(
        "params", "select", roots[:2], producer_execution_identity="select-a"
    )
    ledger.publish_node(
        "select",
        [
            NodeObservation(
                node_id="select",
                input_revision_id=revision_id,
                source_revision_id=revision_id,
                output_revision_id=revision_id,
                status="selected",
            )
            for revision_id in roots[:2]
        ],
        select_set,
        "select-a",
    )
    materialized_ids = []
    materialized_observations = []
    for index, revision_id in enumerate(roots[:2]):
        revision = ledger.add_revision(
            architecture_id=f"architecture-{index}",
            artifact_kind=ArtifactKind.CHECKPOINT,
            artifact={"checkpoint": f"/checkpoint/{index}"},
            parent_revision_id=revision_id,
            producer_node="materialize",
        )
        materialized_ids.append(revision.revision_id)
        materialized_observations.append(
            NodeObservation(
                node_id="materialize",
                input_revision_id=revision_id,
                source_revision_id=revision_id,
                output_revision_id=revision.revision_id,
                status="success",
            )
        )
    ledger.publish_node(
        "materialize",
        materialized_observations,
        CandidateSet.create(
            "params",
            "materialize",
            materialized_ids,
            producer_execution_identity="materialize-a",
        ),
        "materialize-a",
    )
    return config, ledger, roots


def _plan(config: dict, stage_id: str) -> tuple[CampaignPlan, StagePlanNode]:
    compiled = next(node for node in compile_post_mip_flows(config) if node.stage_id == stage_id)
    node = StagePlanNode(
        stage_id=stage_id,
        strategy=ExecutionStrategy.SHARDED,
        instances=2,
        failure_policy=FailurePolicy.STRICT,
        mesh={},
        gpus_per_instance=1,
        gpus_per_node=8,
        nodes=1,
        total_gpus=2,
        exclusive=False,
        parents=compiled.dependency_stage_ids,
        distributed=True,
    )
    puzzle_dir = Path(config["puzzle_dir"])
    return (
        CampaignPlan(
            experiment_config_path=str(puzzle_dir.parent / "experiment.yaml"),
            puzzle_dir=puzzle_dir,
            experiment_config=config,
            runner=RunnerEnvironment(
                kind="slurm",
                contract=ExecutionContract(
                    repository=str(puzzle_dir.parent),
                    venv=str(puzzle_dir.parent / ".venv"),
                ),
            ),
            execution_defaults={"gpus_per_node": 8},
            stages=(node,),
            contract_hash="contract",
        ),
        node,
    )


def _controller_identity(config: dict, stage_id: str) -> str:
    plan, node = _plan(config, stage_id)
    return CampaignController(plan, executor=object())._stage_execution_identity(node)


def test_post_mip_currentness_does_not_initialize_the_candidate_registry(tmp_path: Path):
    config, _ledger, _roots = _identity_fixture(tmp_path)
    config["post_mip"]["flows"]["params"]["nodes"]["root_select"] = {
        "type": "filter",
        "mode": "top_k",
        "metric": "mip.loss",
        "top_k": 1,
    }
    stage_id = "post.params.root_select"
    execution_identity = _controller_identity(config, stage_id)
    summary_path = Path(config["puzzle_dir"]) / "artifacts/post_mip/nodes/root_select/summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps({"status": "success", "execution_identity": execution_identity}) + "\n"
    )
    registry = Path(config["puzzle_dir"]) / "artifacts/post_mip/candidate_registry.json"
    registry.unlink()

    assert not stage_is_complete(config, stage_id)
    assert not registry.exists()


def test_post_mip_submission_prepares_a_stale_candidate_registry(tmp_path: Path):
    config, ledger, _roots = _identity_fixture(tmp_path)
    config["post_mip"]["flows"]["params"]["nodes"]["root_select"] = {
        "type": "filter",
        "mode": "top_k",
        "metric": "mip.loss",
        "top_k": 1,
    }
    active_path = Path(config["puzzle_dir"]) / "mip/active_profiles.json"
    active_path.write_text(
        '{"status":"success","execution_identity":"mip-b","profile_ids":["p1"]}\n'
    )
    assert ledger.active_mip_execution_identity == "mip-a"

    plan, node = _plan(config, "post.params.root_select")
    executor = _TrackingExecutor()
    controller = CampaignController(plan, executor=executor)

    assert controller._submit_stage(node)
    refreshed = CandidateLedger(Path(config["puzzle_dir"]) / "artifacts/post_mip")

    assert len(executor.attempts) == 1
    assert (
        executor.attempts[0]
        .metadata["stage_execution_identity"]
        .startswith("post.params.root_select_execution_")
    )
    assert refreshed.active_mip_execution_identity == "mip-b"
    assert refreshed.active_profile_ids == {"p1"}


def test_non_success_active_mip_defers_identity_without_mutation(tmp_path: Path):
    config, ledger, _roots = _identity_fixture(tmp_path)
    active_path = Path(config["puzzle_dir"]) / "mip/active_profiles.json"
    registry_before = ledger.registry_path.read_bytes()
    active_path.write_text(json.dumps({"status": "running"}) + "\n")

    with pytest.raises(PostMIPExecutionContractUnavailable):
        expected_post_mip_execution_contract(config, "post.params.select")

    assert ledger.registry_path.read_bytes() == registry_before


def test_malformed_active_mip_fails_closed(tmp_path: Path):
    config, _ledger, _roots = _identity_fixture(tmp_path)
    active_path = Path(config["puzzle_dir"]) / "mip/active_profiles.json"
    active_path.write_text('{"status":"success","execution_identity":"mip-a","profile_ids":"p0"}\n')

    with pytest.raises(ValueError, match="invalid profile IDs"):
        expected_post_mip_execution_contract(config, "post.params.select")


def test_unpublished_dependency_current_defers_execution_contract(
    tmp_path: Path,
):
    config, _ledger, _roots = _identity_fixture(tmp_path)
    current_path = Path(config["puzzle_dir"]) / "artifacts/post_mip/nodes/materialize/current.json"
    current_path.write_text("{}")

    with pytest.raises(
        PostMIPExecutionContractUnavailable,
        match="dependency 'materialize' has no published execution identity",
    ):
        expected_post_mip_execution_contract(config, "post.params.final")


def test_invalid_dependency_execution_identity_fails_closed(tmp_path: Path):
    config, _ledger, _roots = _identity_fixture(tmp_path)
    current_path = Path(config["puzzle_dir"]) / "artifacts/post_mip/nodes/materialize/current.json"
    current_path.write_text('{"execution_identity": null}\n')

    with pytest.raises(ValueError, match="invalid execution identity"):
        expected_post_mip_execution_contract(config, "post.params.final")


def test_incomplete_source_mapping_defers_execution_contract(tmp_path: Path):
    config, _ledger, _roots = _identity_fixture(tmp_path)
    observations_path = (
        Path(config["puzzle_dir"])
        / "artifacts/post_mip/nodes/materialize/executions/materialize-a/observations.json"
    )
    observations = json.loads(observations_path.read_text())
    observations_path.write_text(json.dumps(observations[1:]) + "\n")

    with pytest.raises(
        PostMIPExecutionContractUnavailable,
        match="post-MIP source revision .* is unavailable",
    ):
        expected_post_mip_execution_contract(config, "post.params.final")


def _assert_changed_identity_resubmits(config: dict, changed_config: dict, mutate=None) -> None:
    plan_a, node_a = _plan(config, "post.params.select")
    controller_a = CampaignController(plan_a, executor=object())
    adapter = adapter_for_stage(node_a)
    work_plan = adapter.plan(plan_a, node_a)
    attempt = controller_a._bind_attempt_to_stage_execution(
        node_a,
        work_plan,
        adapter.command(
            plan=plan_a,
            node=node_a,
            item=work_plan.items[0],
            attempt_id="attempt-a",
            runner=plan_a.runner,
        ),
    )
    controller_a.store.save_attempt(
        attempt,
        JobHandle(backend="fake", handle_id="fake-attempt-a", attempt_id=attempt.attempt_id),
        JobState.COMPLETED.value,
    )
    if mutate is not None:
        mutate()

    plan_b, node_b = _plan(changed_config, "post.params.select")
    executor = _TrackingExecutor()
    controller_b = CampaignController(plan_b, executor=executor)
    prior = controller_b.store.list_attempts(node_b.stage_id)
    identity_b = controller_b._stage_execution_identity(node_b)

    assert not controller_b._required_work_is_completed(node_b, prior)
    assert controller_b._submit_stage(node_b)
    assert executor.attempts[0].metadata["stage_execution_identity"] == identity_b


@pytest.mark.parametrize("change", ["config", "candidate_set"])
def test_changed_post_mip_contract_resubmits_completed_work(tmp_path: Path, change: str):
    config, ledger, roots = _identity_fixture(tmp_path)
    changed_config = config
    mutate = None

    if change == "config":
        changed_config = copy.deepcopy(config)
        changed_config["post_mip"]["flows"]["params"]["nodes"]["select"]["config"]["label"] = (
            "changed"
        )

    else:

        def replace_candidate_set() -> None:
            replacement_set = CandidateSet.create(
                "params", "score", roots[1:], producer_execution_identity="score-b"
            )
            ledger.publish_node(
                "score",
                [
                    NodeObservation(
                        node_id="score",
                        input_revision_id=revision_id,
                        source_revision_id=revision_id,
                        output_revision_id=revision_id,
                        status="success",
                        metrics={"loss": float(index)},
                    )
                    for index, revision_id in enumerate(roots[1:])
                ],
                replacement_set,
                "score-b",
            )

        mutate = replace_candidate_set

    _assert_changed_identity_resubmits(config, changed_config, mutate)


def test_unresolved_future_post_mip_node_defers_failed_record_recovery(
    tmp_path: Path,
):
    config, _ledger, _roots = _identity_fixture(tmp_path)
    plan, node = _plan(config, "post.params.final")
    controller = CampaignController(plan, executor=object())
    current = plan.puzzle_dir / "artifacts/post_mip/nodes/materialize/current.json"
    current.unlink()
    controller.store.write_stage_record(
        StageRunRecord(
            stage_id=node.stage_id,
            status=JobState.FAILED.value,
            attempts=[
                PersistedAttempt(
                    attempt_id="old",
                    work_id=f"{node.stage_id}:gang",
                    stage_id=node.stage_id,
                    status=JobState.COMPLETED.value,
                    contract_hash=plan.contract_hash,
                    metadata={"stage_execution_identity": "old"},
                )
            ],
        )
    )

    controller._recover_failed_stages()

    assert controller._failed_stages == set()
