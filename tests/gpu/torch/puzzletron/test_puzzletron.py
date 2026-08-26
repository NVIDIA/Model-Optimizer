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
from _test_utils.torch.puzzletron.tiny_qwen_fixture import (
    TinyQwenCampaign,
    build_tiny_qwen_campaign,
)

from modelopt.torch.puzzletron.anymodel.registry import resolve_descriptor_from_pretrained
from modelopt.torch.puzzletron.orchestration.adapters.stage_compat import stage_is_complete
from modelopt.torch.puzzletron.orchestration.state import CampaignStateStore
from modelopt.torch.puzzletron.plugins.automodel.load import load_anymodel_for_scoring
from modelopt.torch.puzzletron.post_mip.records import CandidateLedger
from puzzletron_orchestrator.adapters.registry import adapter_for_stage


@pytest.mark.timeout(2400)
def test_tiny_qwen_campaign_uses_current_public_route(
    project_root_path: Path,
    tmp_path: Path,
) -> None:
    """Run the full route in the pinned CUDA 12.9 image with one visible GPU."""

    assert torch.cuda.is_available(), "Puzzletron GPU CI requires CUDA"
    assert torch.cuda.device_count() == 1, "Puzzletron GPU CI requires one visible GPU"

    campaign = build_tiny_qwen_campaign(project_root_path, tmp_path)
    _assert_compiled_route(campaign)
    completed = campaign.run()
    result = campaign.require_success(completed)

    stage_ids = tuple(node.stage_id for node in campaign.compiled_plan.stages)
    _assert_completed_campaign(campaign, result, stage_ids)
    manifest_paths = sorted(campaign.smoke_root.glob("manifests/*.json"))
    assert manifest_paths
    pruning_paths = _assert_pruning_and_mip_artifacts(campaign)
    final_checkpoint, post_paths = _assert_post_mip_and_final_checkpoint(campaign)
    _assert_selected_checkpoint_runs(final_checkpoint)
    _assert_final_report(campaign, result)

    state = CampaignStateStore(campaign.smoke_root)
    attempts_before = {
        (attempt["work_id"], attempt["attempt_id"]): attempt for attempt in state.list_attempts()
    }
    durable_paths = [*manifest_paths, *pruning_paths, *post_paths]
    before_resume = _artifact_digests(durable_paths)

    resumed = campaign.run()
    resumed_result = campaign.require_success(resumed)

    _assert_completed_campaign(campaign, resumed_result, stage_ids)
    attempts_after = {
        (attempt["work_id"], attempt["attempt_id"]): attempt for attempt in state.list_attempts()
    }
    assert attempts_after == attempts_before
    assert _artifact_digests(durable_paths) == before_resume


def _assert_compiled_route(campaign: TinyQwenCampaign) -> None:
    """Verify the saved YAML bundle compiles to the intended one-GPU DAG."""

    for name in ("experiment.yaml", "runner.yaml", "execution.yaml"):
        assert (campaign.smoke_bundle / name).is_file()
    serving = campaign.config["post_mip"]["flows"][campaign.flow_id]["nodes"]["serving"]
    assert serving["config"]["allow_aiperf_v011_online_tokenizer_resolution"] is True
    assert serving["config"]["topology"]["extra_vllm_args"] == [
        "-cc.cudagraph_mode=NONE",
        "--no-enable-flashinfer-autotune",
        "--max-num-batched-tokens",
        "128",
        "--max-num-seqs",
        "4",
        "--gpu-memory-utilization",
        "0.5",
    ]

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
    assert (replacement_node.instances, replacement_node.gpus_per_instance) == (1, 1)
    assert len(replacement_work.items) == 1
    assert replacement_work.items[0].metadata["worker_count"] == 1


def _assert_completed_campaign(
    campaign: TinyQwenCampaign,
    result: dict[str, Any],
    stage_ids: tuple[str, ...],
) -> None:
    """Verify the controller and durable stage state agree on completion."""

    assert tuple(result["completed"]) == stage_ids
    assert all(stage_is_complete(campaign.config, stage_id) for stage_id in stage_ids)


def _assert_pruning_and_mip_artifacts(campaign: TinyQwenCampaign) -> list[Path]:
    """Verify pruning and MIP produced finite, feasible campaign artifacts."""

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
    provenances = [result["provenance"] for result in replacement_results]
    assert {provenance["score_device_type"] for provenance in provenances} == {"cuda"}
    assert {provenance["visible_cuda_device_count"] for provenance in provenances} == {1}

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


def _assert_post_mip_and_final_checkpoint(
    campaign: TinyQwenCampaign,
) -> tuple[Path, list[Path]]:
    """Verify the post-MIP flow selects and trains one usable final checkpoint."""

    root = campaign.smoke_root
    ledger = CandidateLedger(root / "artifacts/post_mip")
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
    expected_best_lm = tuple(
        revision_id
        for _loss, revision_id in sorted(
            (loss, revision_id) for revision_id, loss in online_losses.items()
        )[:3]
    )
    assert best_lm.revision_ids == expected_best_lm
    assert {
        revision_id
        for revision_id, row in ledger.observations["best_lm"].items()
        if row.status == "selected"
    } == set(best_lm.revision_ids)

    materialized = ledger.load_candidate_set("materialized")
    assert len(materialized.revision_ids) == 3
    for revision_id in materialized.revision_ids:
        revision = ledger.revisions[revision_id]
        assert revision.parent_revision_id in set(best_lm.revision_ids)
        checkpoint = Path(revision.artifact["checkpoint"])
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
    expected_fastest = tuple(
        revision_id
        for _throughput, revision_id in sorted(
            (throughput, revision_id) for revision_id, throughput in serving_throughputs.items()
        )[-2:][::-1]
    )
    assert fastest.revision_ids == expected_fastest
    short_kd = ledger.load_candidate_set("short_kd")
    assert len(short_kd.revision_ids) == 2
    kd_paths = []
    for revision_id in short_kd.revision_ids:
        revision = ledger.revisions[revision_id]
        assert revision.parent_revision_id in set(materialized.revision_ids)
        checkpoint = Path(revision.artifact["checkpoint"])
        summary_path = Path(revision.artifact["summary_path"])
        kd_summary = _json(summary_path)
        records = kd_summary["records"]
        steps = {int(record.get("step", record.get("global_step"))) for record in records}
        losses = [float(record.get("loss", record.get("train_loss"))) for record in records]
        assert kd_summary["max_steps"] == 2
        assert len(steps) >= 2
        assert all(math.isfinite(loss) for loss in losses)
        assert (checkpoint / "config.json").is_file()
        assert list(checkpoint.glob("*.safetensors"))
        parent = ledger.revisions[revision.parent_revision_id]
        block_configs = _json(checkpoint / "config.json")["block_configs"]
        parent_block_configs = _json(Path(parent.artifact["checkpoint"]) / "config.json")[
            "block_configs"
        ]
        assert block_configs
        assert block_configs == parent_block_configs
        kd_paths.extend(
            [
                summary_path,
                checkpoint.parents[2] / "training.jsonl",
                checkpoint / "config.json",
                *checkpoint.glob("*.safetensors"),
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
    best = ledger.load_candidate_set("best")
    assert len(best.revision_ids) == 1
    selected_id = best.revision_ids[0]
    assert final_losses[selected_id] == min(final_losses.values())
    selected = ledger.revisions[selected_id]
    final_checkpoint = Path(selected.artifact["checkpoint"])

    post_paths = [root / "artifacts/post_mip/candidate_registry.json"]
    post_paths.extend(root.glob("artifacts/post_mip/nodes/*/current.json"))
    post_paths.extend(root.glob("artifacts/post_mip/nodes/*/index.json"))
    post_paths.extend(root.glob("artifacts/post_mip/nodes/*/summary.json"))
    post_paths.extend(root.glob("artifacts/post_mip/nodes/*/executions/*/candidate_set.json"))
    post_paths.extend(root.glob("artifacts/post_mip/nodes/*/executions/*/observations.json"))
    post_paths.extend(root.glob("artifacts/post_mip/nodes/*/executions/*/summary.json"))
    return final_checkpoint, [*post_paths, *aiperf_paths, *kd_paths]


def _assert_selected_checkpoint_runs(final_checkpoint: Path) -> None:
    """Reload the selected physical checkpoint and run a finite CUDA forward pass."""

    resolution = resolve_descriptor_from_pretrained(str(final_checkpoint))
    selected_config = _json(final_checkpoint / "config.json")
    widths_by_block = [
        _nested_values(block_config, "intermediate_size")
        for block_config in selected_config["block_configs"]
    ]
    assert all(len(widths) == 1 for widths in widths_by_block)
    selected_widths = [int(widths[0]) for widths in widths_by_block]
    model = load_anymodel_for_scoring(
        str(final_checkpoint),
        anymodel_descriptor=resolution.name,
        force_hf=True,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    ).cuda()

    assert [layer.mlp.down_proj.in_features for layer in model.model.layers] == selected_widths
    with torch.no_grad():
        logits = model(torch.tensor([[1, 2, 3, 4]], device="cuda"), use_cache=False).logits
    assert torch.isfinite(logits).all()


def _assert_final_report(campaign: TinyQwenCampaign, result: dict[str, Any]) -> None:
    """Verify the completed report links the key post-MIP stages."""

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


def _finite_metric(ledger: CandidateLedger, revision_id: str, reference: str) -> float:
    """Resolve one required finite candidate metric."""

    value = ledger.resolve_metric(revision_id, reference)
    assert value is not None and math.isfinite(value)
    return value
