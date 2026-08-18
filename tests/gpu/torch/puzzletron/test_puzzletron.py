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

from modelopt.torch.puzzletron.anymodel.registry import resolve_descriptor_from_pretrained
from modelopt.torch.puzzletron.orchestration.adapters.stage_compat import stage_is_complete
from modelopt.torch.puzzletron.orchestration.state import CampaignStateStore
from modelopt.torch.puzzletron.plugins.automodel.load import load_anymodel_for_scoring
from modelopt.torch.puzzletron.post_mip.records import CandidateLedger
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

    replacement_summary_path = root / "artifacts/replacement_scoring/summary.json"
    replacement_summary = _json(replacement_summary_path)
    assert replacement_summary["widths"] == [256]
    assert replacement_summary["scenario_count"] == 1

    active_profiles_path = root / "mip/active_profiles.json"
    active_profiles = _json(active_profiles_path)
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
    return [*pass_manifests, replacement_summary_path, active_profiles_path, *solution_paths]


def _assert_node_publication(
    campaign: TinyQwenCampaign,
    ledger: CandidateLedger,
    node_id: str,
    predecessor_identity: str | None,
) -> tuple[dict[str, Any], str]:
    node_root = campaign.smoke_root / "artifacts/post_mip/nodes" / node_id
    summary = _json(node_root / "summary.json")
    execution_identity = summary["execution_identity"]
    candidate_set = ledger.load_candidate_set(node_id)

    assert summary["status"] == "success"
    assert summary["stage_id"] == f"post.{campaign.flow_id}.{node_id}"
    assert candidate_set.flow_id == campaign.flow_id
    assert candidate_set.node_id == node_id
    assert candidate_set.producer_execution_identity == execution_identity
    if predecessor_identity is not None:
        assert summary["execution_contract"]["candidate_set"] == predecessor_identity
    return summary, candidate_set.identity


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
    best_lm = ledger.load_candidate_set("best_lm")
    materialized = ledger.load_candidate_set("materialized")
    serving = ledger.load_candidate_set("serving")
    fastest = ledger.load_candidate_set("fastest")
    short_kd = ledger.load_candidate_set("short_kd")
    final_eval = ledger.load_candidate_set("final_eval")
    best = ledger.load_candidate_set("best")

    assert len(online.revision_ids) >= 3
    assert len(best_lm.revision_ids) == len(materialized.revision_ids) == 3
    assert len(serving.revision_ids) >= 2
    assert len(fastest.revision_ids) == len(short_kd.revision_ids) == 2
    assert len(best.revision_ids) == 1
    assert set(best_lm.revision_ids) <= set(online.revision_ids)
    assert set(serving.revision_ids) <= set(materialized.revision_ids)
    assert set(fastest.revision_ids) <= set(serving.revision_ids)
    assert set(final_eval.revision_ids) == set(short_kd.revision_ids)

    serving_throughputs = {
        revision_id: ledger.resolve_metric(
            revision_id, "serving.concurrency_1.output_token_throughput"
        )
        for revision_id in serving.revision_ids
    }
    assert all(
        throughput is not None and throughput > 0 for throughput in serving_throughputs.values()
    )
    serving_observations = ledger.observations["serving"]
    aiperf_paths = [
        Path(path)
        for row in serving_observations.values()
        if row.status == "success"
        for artifacts in row.artifacts["result_paths"]
        for path in artifacts.values()
    ]
    assert aiperf_paths and all(path.is_file() for path in aiperf_paths)

    kd_paths = []
    for revision_id in short_kd.revision_ids:
        revision = ledger.revisions[revision_id]
        checkpoint = Path(revision.artifact["checkpoint"])
        summary_path = Path(revision.artifact["summary_path"])
        kd_summary = _json(summary_path)
        records = kd_summary["records"]
        assert kd_summary["max_steps"] == 2
        assert len({int(record.get("step", record.get("global_step"))) for record in records}) >= 2
        assert Path(kd_summary["post_kd_checkpoint"]) == checkpoint
        assert (checkpoint / "config.json").is_file()
        assert list(checkpoint.glob("*.safetensors"))
        kd_paths.extend(
            [summary_path, checkpoint / "config.json", *checkpoint.glob("*.safetensors")]
        )

    final_losses = {
        revision_id: ledger.resolve_metric(revision_id, "final_eval.lm_loss")
        for revision_id in final_eval.revision_ids
    }
    assert all(loss is not None for loss in final_losses.values())
    selected_id = best.revision_ids[0]
    assert final_losses[selected_id] == min(final_losses.values())
    selected = ledger.revisions[selected_id]
    final_checkpoint = Path(selected.artifact["checkpoint"])
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
    assert manifest["verification"] == "passed"


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
    assert all(stage_is_complete(campaign.config, stage_id) for stage_id in stage_ids)

    manifest_paths = sorted(campaign.smoke_root.glob("manifests/*.json"))
    assert manifest_paths
    assert all(_json(path)["status"] == "success" for path in manifest_paths)
    pruning_paths = _assert_pruning_and_mip_artifacts(campaign)
    final_checkpoint, post_paths = _assert_post_mip_and_final_checkpoint(campaign)
    resolution = resolve_descriptor_from_pretrained(str(final_checkpoint))
    selected_config = _json(final_checkpoint / "config.json")
    assert selected_config["architectures"] == ["AnyModel"]
    assert selected_config["base_architecture"] == "Qwen3_5ForCausalLM"
    selected_block_configs = selected_config["block_configs"]
    selected_ffn_widths = []
    for block_config in selected_block_configs:
        widths = _nested_values(block_config, "intermediate_size")
        assert len(widths) == 1
        selected_ffn_widths.append(int(widths[0]))
    per_layer_config = (selected_config.get("text_config") or selected_config)["per_layer_config"]
    assert [
        int(
            per_layer_config.get(str(index), {}).get(
                "intermediate_size", selected_config["intermediate_size"]
            )
        )
        for index in range(len(selected_block_configs))
    ] == selected_ffn_widths
    model = load_anymodel_for_scoring(
        str(final_checkpoint),
        anymodel_descriptor=resolution.name,
        force_hf=True,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    ).cuda()
    assert [layer.mlp.down_proj.in_features for layer in model.model.layers] == selected_ffn_widths
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
    resumed = campaign.run(timeout=300)
    resumed_result = campaign.require_success(resumed)
    assert tuple(resumed_result["completed"]) == stage_ids
    attempts_after = {
        (attempt["work_id"], attempt["attempt_id"]): attempt for attempt in state.list_attempts()
    }
    assert attempts_after == attempts_before
    assert _artifact_digests(durable_paths) == before_resume
    resumed_checkpoint, _ = _assert_post_mip_and_final_checkpoint(campaign)
    assert resumed_checkpoint == final_checkpoint
