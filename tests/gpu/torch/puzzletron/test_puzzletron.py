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

"""Hermetic GPU coverage for the public Puzzletron campaign route."""

from __future__ import annotations

import json
import math
from hashlib import sha256
from itertools import pairwise
from pathlib import Path
from typing import Any

import pytest
import torch
from _test_utils.torch.puzzletron.tiny_qwen_campaign import (
    TinyQwenCampaign,
    build_tiny_qwen_campaign,
)
from transformers import AutoModelForCausalLM

from modelopt.torch.puzzletron.orchestration.adapters.stage_compat import stage_is_complete
from modelopt.torch.puzzletron.orchestration.state import CampaignStateStore
from modelopt.torch.puzzletron.post_mip.records import ArtifactKind, CandidateLedger, CandidateSet
from puzzletron_orchestrator.adapters.registry import adapter_for_stage


def _json(path: Path) -> Any:
    return json.loads(path.read_text())


def _artifact_digests(paths: list[Path]) -> dict[str, str]:
    return {str(path): sha256(path.read_bytes()).hexdigest() for path in sorted(set(paths))}


def _nested_values(value: Any, key: str) -> list[Any]:
    values = []
    if isinstance(value, dict):
        values.extend(item for name, item in value.items() if name == key)
        for item in value.values():
            values.extend(_nested_values(item, key))
    elif isinstance(value, list):
        for item in value:
            values.extend(_nested_values(item, key))
    return values


def _tensor_values(value: Any):
    if torch.is_tensor(value):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _tensor_values(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _tensor_values(item)


def _assert_compiled_route(campaign: TinyQwenCampaign) -> None:
    for name in ("experiment.yaml", "runner.yaml", "execution.yaml"):
        assert (campaign.smoke_bundle / name).is_file()
    assert campaign.config["model"]["descriptor_override"] == "qwen3_5_text"
    assert campaign.config["embedding_pruning"]["widths"] == [256]

    expected_nodes = (
        "online_eval",
        "best_lm",
        "materialized",
        "serving",
        "fastest",
        "short_kd",
        "final_eval",
        "best",
    )
    prefix = f"post.{campaign.flow_id}."
    post_nodes = tuple(
        node for node in campaign.compiled_plan.stages if node.stage_id.startswith(prefix)
    )
    assert tuple(node.stage_id.removeprefix(prefix) for node in post_nodes) == expected_nodes
    assert post_nodes[0].parents == ("mip",)
    for parent, node in pairwise(post_nodes):
        assert node.parents == (parent.stage_id,)

    flow = campaign.config["post_mip"]["flows"][campaign.flow_id]["nodes"]
    assert tuple(flow) == expected_nodes
    assert flow["online_eval"]["config"]["eval_samples"] == 2
    assert flow["best_lm"]["top_k"] == 3
    assert flow["serving"]["config"]["input_tokens"] == 32
    assert flow["serving"]["config"]["output_tokens"] == 8
    assert flow["serving"]["config"]["request_count"] == 4
    assert flow["fastest"]["top_k"] == 2
    assert flow["short_kd"]["config"]["max_steps"] == 2
    assert flow["short_kd"]["config"]["global_batch_size"] == 1
    assert flow["short_kd"]["config"]["local_batch_size"] == 1
    assert flow["short_kd"]["config"]["checkpoint_every_steps"] == 2
    assert flow["final_eval"]["config"]["eval_samples"] == 2
    assert flow["best"]["top_k"] == 1

    replacement_node = next(
        node for node in campaign.compiled_plan.stages if node.stage_id == "replacement_scoring"
    )
    replacement_work = adapter_for_stage(replacement_node).plan(
        campaign.compiled_plan, replacement_node
    )
    assert replacement_node.instances == 1
    assert replacement_node.gpus_per_instance == 1
    assert replacement_node.total_gpus == 1
    assert replacement_node.nodes == 1
    assert len(replacement_work.items) == 1
    assert replacement_work.items[0].metadata["worker_count"] == 1


def _assert_pruning_and_mip_artifacts(campaign: TinyQwenCampaign) -> list[Path]:
    root = campaign.smoke_root
    pass_manifests = list(
        root.glob("pruning/pruning_scores/automodel/*/activation_passes_manifest.json")
    )
    assert len(pass_manifests) == 1
    score_files = list(pass_manifests[0].parent.glob("*/rank_*.pth"))
    assert score_files
    score_tensors = [
        tensor
        for score_file in score_files
        for tensor in _tensor_values(torch.load(score_file, map_location="cpu", weights_only=False))
    ]
    assert score_tensors
    assert all(tensor.numel() and torch.isfinite(tensor).all() for tensor in score_tensors)

    replacement_summary = _json(root / "artifacts/replacement_scoring/summary.json")
    assert replacement_summary["widths"] == [256]
    assert replacement_summary["scenario_count"] == 1
    replacement_results = [
        _json(path)
        for path in root.glob(
            "scenarios/width-*/depth-*/distributed_eval/replacement_scoring/results/**/*.json"
        )
    ]
    assert replacement_results
    assert all(
        result["provenance"]["score_device_type"] == "cuda"
        and result["provenance"]["visible_cuda_device_count"] == 1
        for result in replacement_results
    )

    candidate_library = _json(root / "candidate_library.json")
    assert 256 in _nested_values(candidate_library, "intermediate_size")
    assert 512 in _nested_values(candidate_library, "intermediate_size")
    active_profiles = _json(root / "mip/active_profiles.json")
    assert active_profiles["status"] == "success"
    grids = [
        _json(root / "mip" / "profiles" / profile_id / "mip_grid.json")
        for profile_id in active_profiles["profile_ids"]
    ]
    assert all(grid["status"] == "success" for grid in grids)
    solution_paths = [
        Path(scenario["solution_path"])
        for grid in grids
        for scenario in grid["scenarios"]
        if scenario["status"] == "feasible"
    ]
    solutions = [solution for path in solution_paths for solution in _json(path)]
    assert len(solutions) >= 3
    solution_ffn_widths = {
        int(width)
        for solution in solutions
        for width in _nested_values(solution["chosen_block_configs"], "intermediate_size")
    }
    assert solution_ffn_widths.intersection({256, 512})
    return [*pass_manifests, *score_files, *solution_paths]


def _assert_node_publication(
    campaign: TinyQwenCampaign,
    ledger: CandidateLedger,
    node_id: str,
    predecessor_identity: str | None,
) -> tuple[dict[str, Any], str]:
    node_root = campaign.smoke_root / "artifacts/post_mip/nodes" / node_id
    summary = _json(node_root / "summary.json")
    current = _json(node_root / "current.json")
    index = _json(node_root / "index.json")
    execution_identity = summary["execution_identity"]
    execution_root = node_root / "executions" / execution_identity
    candidate_path = Path(summary["candidate_set_path"])
    observations_path = Path(summary["observations_path"])
    candidate_set = ledger.load_candidate_set(node_id)
    candidate_payload = _json(candidate_path)

    assert summary["status"] == "success"
    assert summary["stage_id"] == f"post.{campaign.flow_id}.{node_id}"
    assert summary["node_id"] == node_id
    assert current["execution_identity"] == execution_identity
    assert index["current"] == execution_identity
    assert execution_identity in index["executions"]
    assert candidate_path == execution_root / "candidate_set.json"
    assert observations_path == execution_root / "observations.json"
    assert _json(execution_root / "summary.json") == summary
    assert candidate_path.is_file() and observations_path.is_file()
    assert candidate_set.flow_id == campaign.flow_id
    assert candidate_set.node_id == node_id
    assert candidate_set.producer_execution_identity == execution_identity
    assert candidate_set == CandidateSet.create(
        campaign.flow_id,
        node_id,
        candidate_set.revision_ids,
        producer_execution_identity=execution_identity,
    )
    assert candidate_payload == {
        "flow_id": candidate_set.flow_id,
        "identity": candidate_set.identity,
        "node_id": candidate_set.node_id,
        "producer_execution_identity": candidate_set.producer_execution_identity,
        "revision_ids": list(candidate_set.revision_ids),
    }
    assert summary["output_count"] == len(candidate_set.revision_ids)
    assert summary["input_count"] == len(_json(observations_path))
    if predecessor_identity is not None:
        assert summary["execution_contract"]["candidate_set"] == predecessor_identity
    return summary, candidate_set.identity


def _finite_metric(ledger: CandidateLedger, revision_id: str, reference: str) -> float:
    value = ledger.resolve_metric(revision_id, reference)
    assert value is not None and math.isfinite(value)
    return value


def _assert_post_mip_and_final_checkpoint(
    campaign: TinyQwenCampaign,
) -> tuple[Path, list[Path]]:
    root = campaign.smoke_root
    ledger = CandidateLedger(root / "artifacts/post_mip")
    node_ids = (
        "online_eval",
        "best_lm",
        "materialized",
        "serving",
        "fastest",
        "short_kd",
        "final_eval",
        "best",
    )
    predecessor_identity = None
    summaries = {}
    for node_id in node_ids:
        summaries[node_id], predecessor_identity = _assert_node_publication(
            campaign, ledger, node_id, predecessor_identity
        )

    online = ledger.load_candidate_set("online_eval")
    assert len(online.revision_ids) >= 3
    online_observations = ledger.observations["online_eval"]
    assert all(row.status == "success" for row in online_observations.values())
    assert all(Path(row.artifacts["result_path"]).is_file() for row in online_observations.values())
    online_losses = {
        revision_id: _finite_metric(ledger, revision_id, "online_eval.lm_loss")
        for revision_id in online.revision_ids
    }

    best_lm = ledger.load_candidate_set("best_lm")
    assert len(best_lm.revision_ids) == 3
    assert (
        best_lm.revision_ids
        == tuple(
            revision_id
            for _loss, revision_id in sorted((loss, rid) for rid, loss in online_losses.items())
        )[:3]
    )
    assert {
        revision_id
        for revision_id, row in ledger.observations["best_lm"].items()
        if row.status == "selected"
    } == set(best_lm.revision_ids)

    materialized = ledger.load_candidate_set("materialized")
    assert len(materialized.revision_ids) == 3
    materialized_summary = summaries["materialized"]
    materialized_root = (
        root
        / "artifacts/post_mip/nodes/materialized/executions"
        / materialized_summary["execution_identity"]
    )
    for revision_id in materialized.revision_ids:
        revision = ledger.revisions[revision_id]
        assert revision.producer_node == "materialized"
        assert revision.parent_revision_id in set(best_lm.revision_ids)
        assert revision.artifact_kind is ArtifactKind.CHECKPOINT
        checkpoint = Path(revision.artifact["checkpoint"])
        assert checkpoint == materialized_root / "checkpoints" / revision.architecture_id
        assert (checkpoint / "config.json").is_file()
        assert list(checkpoint.glob("*.safetensors"))
    assert {
        ledger.revisions[revision_id].parent_revision_id
        for revision_id in materialized.revision_ids
    } == set(best_lm.revision_ids)

    serving = ledger.load_candidate_set("serving")
    assert len(serving.revision_ids) >= 2
    assert set(serving.revision_ids) <= set(materialized.revision_ids)
    serving_throughputs = {
        revision_id: _finite_metric(
            ledger, revision_id, "serving.concurrency_1.output_token_throughput"
        )
        for revision_id in serving.revision_ids
    }
    assert all(throughput > 0 for throughput in serving_throughputs.values())
    serving_observations = ledger.observations["serving"]
    assert set(serving_observations) == set(materialized.revision_ids)
    assert {
        revision_id for revision_id, row in serving_observations.items() if row.status == "success"
    } == set(serving.revision_ids)
    assert all(
        row.status in {"failed", "timed_out"} and row.error and row.output_revision_id is None
        for row in serving_observations.values()
        if row.status != "success"
    )
    aiperf_paths = [
        Path(path)
        for row in serving_observations.values()
        if row.status == "success"
        for artifacts in row.artifacts["result_paths"]
        for path in artifacts.values()
    ]
    assert aiperf_paths and all(path.is_file() for path in aiperf_paths)

    fastest = ledger.load_candidate_set("fastest")
    assert len(fastest.revision_ids) == 2
    assert fastest.revision_ids == tuple(
        revision_id
        for _throughput, revision_id in sorted(
            (throughput, revision_id) for revision_id, throughput in serving_throughputs.items()
        )[-2:][::-1]
    )
    short_kd = ledger.load_candidate_set("short_kd")
    assert len(short_kd.revision_ids) == 2
    short_kd_summary = summaries["short_kd"]
    short_kd_root = (
        root
        / "artifacts/post_mip/nodes/short_kd/executions"
        / short_kd_summary["execution_identity"]
    )
    kd_paths = []
    for revision_id in short_kd.revision_ids:
        revision = ledger.revisions[revision_id]
        assert revision.producer_node == "short_kd"
        assert revision.parent_revision_id in set(materialized.revision_ids)
        assert revision.artifact_kind is ArtifactKind.CHECKPOINT
        checkpoint = Path(revision.artifact["checkpoint"])
        summary_path = Path(revision.artifact["summary_path"])
        architecture_root = short_kd_root / "checkpoints" / revision.architecture_id
        assert summary_path == architecture_root / "global_distillation_summary.json"
        assert checkpoint.is_relative_to(architecture_root)
        kd_summary = _json(summary_path)
        records = kd_summary["records"]
        assert kd_summary["max_steps"] == 2
        assert len({int(record.get("step", record.get("global_step"))) for record in records}) >= 2
        assert all(
            math.isfinite(float(record.get("loss", record.get("train_loss")))) for record in records
        )
        assert Path(kd_summary["post_kd_checkpoint"]) == checkpoint
        assert (checkpoint / "config.json").is_file()
        assert list(checkpoint.glob("*.safetensors"))
        assert (checkpoint.parents[1] / "saving_completed").is_file()
        kd_paths.extend(
            [
                summary_path,
                checkpoint.parents[2] / "training.jsonl",
                checkpoint / "config.json",
                *checkpoint.glob("*.safetensors"),
                checkpoint.parents[1] / "saving_completed",
            ]
        )
    assert {
        ledger.revisions[revision_id].parent_revision_id for revision_id in short_kd.revision_ids
    } == set(fastest.revision_ids)

    final_eval = ledger.load_candidate_set("final_eval")
    assert set(final_eval.revision_ids) == set(short_kd.revision_ids)
    final_losses = {
        revision_id: _finite_metric(ledger, revision_id, "final_eval.lm_loss")
        for revision_id in final_eval.revision_ids
    }
    assert all(row.status == "success" for row in ledger.observations["final_eval"].values())
    best = ledger.load_candidate_set("best")
    assert len(best.revision_ids) == 1
    selected_id = best.revision_ids[0]
    assert final_losses[selected_id] == min(final_losses.values())
    selected = ledger.revisions[selected_id]
    final_checkpoint = Path(selected.artifact["checkpoint"])
    assert selected.producer_node == "short_kd"
    assert selected.artifact_kind is ArtifactKind.CHECKPOINT
    assert summaries["best"]["checkpoints"] == [str(final_checkpoint)]

    post_paths = [root / "artifacts/post_mip/candidate_registry.json"]
    post_paths.extend(root.glob("artifacts/post_mip/nodes/*/current.json"))
    post_paths.extend(root.glob("artifacts/post_mip/nodes/*/index.json"))
    post_paths.extend(root.glob("artifacts/post_mip/nodes/*/summary.json"))
    post_paths.extend(root.glob("artifacts/post_mip/nodes/*/executions/*/candidate_set.json"))
    post_paths.extend(root.glob("artifacts/post_mip/nodes/*/executions/*/observations.json"))
    post_paths.extend(root.glob("artifacts/post_mip/nodes/*/executions/*/summary.json"))
    return final_checkpoint, [*post_paths, *aiperf_paths, *kd_paths]


def _assert_final_report(campaign: TinyQwenCampaign, result: dict[str, Any]) -> None:
    report_root = campaign.smoke_root / "artifacts/campaign_report"
    html_path = report_root / "campaign_report.html"
    manifest_path = report_root / "report_manifest.json"
    assert result["report_status"] == "completed"
    assert Path(result["report_path"]) == html_path
    assert Path(result["report_manifest_path"]) == manifest_path
    manifest = _json(manifest_path)
    assert manifest["schema_version"] == 1
    assert manifest["verification"] == "passed"
    assert manifest["campaign_identity"]
    html = html_path.read_text()
    for node_id in ("online_eval", "serving", "short_kd", "final_eval"):
        section_id = "post-" + "-".join(
            part.replace("_", "-") for part in (campaign.flow_id, node_id)
        )
        assert f'id="{section_id}"' in html


@pytest.mark.timeout(2400)
def test_tiny_qwen_campaign_uses_current_public_route(
    project_root_path: Path,
    tmp_path: Path,
) -> None:
    """Run the full route in the pinned CUDA 12.9 image with one visible GPU."""

    assert torch.cuda.is_available(), "Puzzletron GPU CI requires CUDA"
    assert torch.cuda.device_count() == 1, "Puzzletron GPU CI requires one visible GPU"
    assert torch.equal(torch.arange(4, device="cuda").cpu(), torch.arange(4))
    campaign = build_tiny_qwen_campaign(project_root_path, tmp_path)
    _assert_compiled_route(campaign)

    completed = campaign.run()
    result = campaign.require_success(completed)
    stage_ids = tuple(node.stage_id for node in campaign.compiled_plan.stages)
    assert tuple(result["completed"]) == stage_ids
    assert result["failed_stages"] == []
    assert not result["halted"]
    assert not result["cancelled"]
    assert not result["detached"]
    assert all(stage_is_complete(campaign.config, stage_id) for stage_id in stage_ids)

    manifest_paths = sorted(campaign.smoke_root.glob("manifests/*.json"))
    assert manifest_paths
    assert all(_json(path)["status"] == "success" for path in manifest_paths)
    pruning_paths = _assert_pruning_and_mip_artifacts(campaign)
    final_checkpoint, post_paths = _assert_post_mip_and_final_checkpoint(campaign)
    model = AutoModelForCausalLM.from_pretrained(
        final_checkpoint,
        dtype=torch.bfloat16,
        local_files_only=True,
    ).cuda()
    with torch.no_grad():
        logits = model(
            torch.tensor([[1, 2, 3, 4]], device="cuda"),
            use_cache=False,
        ).logits
    assert torch.isfinite(logits).all()
    _assert_final_report(campaign, result)

    state = CampaignStateStore(campaign.smoke_root)
    attempts_before = {
        (attempt["work_id"], attempt["attempt_id"]): attempt for attempt in state.list_attempts()
    }
    durable_paths = [*manifest_paths, *pruning_paths, *post_paths]
    before_resume = _artifact_digests(durable_paths)
    resumed = campaign.run()
    resumed_result = campaign.require_success(resumed)
    assert tuple(resumed_result["completed"]) == stage_ids
    assert resumed_result["failed_stages"] == []
    assert not resumed_result["halted"]
    assert not resumed_result["cancelled"]
    assert not resumed_result["detached"]
    assert resumed_result["report_status"] == "completed"
    attempts_after = {
        (attempt["work_id"], attempt["attempt_id"]): attempt for attempt in state.list_attempts()
    }
    assert attempts_after == attempts_before
    assert _artifact_digests(durable_paths) == before_resume
    resumed_checkpoint, _ = _assert_post_mip_and_final_checkpoint(campaign)
    assert resumed_checkpoint == final_checkpoint
