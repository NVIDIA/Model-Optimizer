# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for iterative depth scoring through the distributed evaluator."""

import asyncio
import json
from types import SimpleNamespace

from modelopt.torch.puzzletron.block_config import (
    AttentionConfig,
    BlockConfig,
    MambaConfig,
    MoEConfig,
)
from modelopt.torch.puzzletron.depth.schema import SublayerRemoval


def _campaign():
    manifest = SimpleNamespace(
        model={"fingerprint": "model"},
        data={"fingerprint": "data"},
        metrics={"lm_loss": {}},
        precision={"torch_dtype": "bfloat16"},
        evaluator_revision="depth-rpc-v1",
    )
    return SimpleNamespace(campaign_id="campaign", manifest=manifest)


def test_build_depth_request_merges_removals_per_layer_and_canonicalizes_order():
    from modelopt.torch.puzzletron.distributed_eval.depth import build_depth_request

    teacher_blocks = [
        BlockConfig(
            subblock_configs=(
                AttentionConfig(
                    num_query_heads=4,
                    num_kv_heads=2,
                    qk_head_dim=2,
                    v_head_dim=2,
                ),
                MoEConfig(num_experts=4, expert_intermediate_size=8, top_k=2),
            )
        ),
        BlockConfig(
            subblock_configs=(
                MambaConfig(state_dim=4, num_heads=2, head_dim=2, num_groups=1),
            )
        ),
    ]
    removals = (
        SublayerRemoval(layer_idx=1, kind="mamba"),
        SublayerRemoval(layer_idx=0, kind="moe"),
        SublayerRemoval(layer_idx=0, kind="attention"),
    )

    request = build_depth_request(
        _campaign(),
        teacher_blocks=teacher_blocks,
        removals=removals,
        hidden_width=8,
    )
    reordered = build_depth_request(
        _campaign(),
        teacher_blocks=teacher_blocks,
        removals=tuple(reversed(removals)),
        hidden_width=8,
    )

    assert request.handler == "depth_candidate"
    assert request.request_id == reordered.request_id
    assert [
        replacement["parent_layer_indices"]
        for replacement in request.payload["layer_replacements"]
    ] == [[0], [1]]
    first_child = request.payload["layer_replacements"][0]["child_block_configs"][0]
    first_subblocks = {item["kind"]: item for item in first_child["subblock_configs"]}
    assert first_subblocks["attention"]["no_op"] is True
    assert first_subblocks["moe"]["no_op"] is True


def test_iterative_depth_rpc_ranks_each_round_and_writes_resumable_trajectory(tmp_path):
    from modelopt.torch.puzzletron.distributed_eval.depth import run_iterative_depth_rpc
    from modelopt.torch.puzzletron.distributed_eval.schema import EvaluationResult

    teacher_blocks = [
        BlockConfig(
            subblock_configs=(
                MambaConfig(state_dim=4, num_heads=2, head_dim=2, num_groups=1),
            )
        ),
        BlockConfig(
            subblock_configs=(
                MoEConfig(num_experts=4, expert_intermediate_size=8, top_k=2),
            )
        ),
    ]
    available = (
        SublayerRemoval(layer_idx=0, kind="mamba"),
        SublayerRemoval(layer_idx=1, kind="moe"),
    )
    losses = {
        (): 1.0,
        ((0, "mamba"),): 1.2,
        ((1, "moe"),): 1.1,
        ((0, "mamba"), (1, "moe")): 1.5,
    }

    class FakeClient:
        def __init__(self):
            self.requests = {}

        async def submit_many(self, requests):
            handles = []
            for request in requests:
                self.requests[request.request_id] = request
                handles.append(SimpleNamespace(request_id=request.request_id))
            return handles

        async def as_completed(self, handles):
            for handle in handles:
                request = self.requests[handle.request_id]
                key = tuple(
                    (item["layer_idx"], item["kind"])
                    for item in request.payload["removals"]
                )
                score = losses[key]
                yield EvaluationResult(
                    request_id=request.request_id,
                    campaign_id=request.campaign_id,
                    metrics={"lm_loss": {"avg": score, "per_sample": [score]}},
                )

    client = FakeClient()
    result = asyncio.run(
        run_iterative_depth_rpc(
            _campaign(),
            client=client,
            teacher_blocks=teacher_blocks,
            available=available,
            hidden_width=8,
            output_dir=tmp_path,
            max_removals=2,
            source_checkpoint_dir="/checkpoint",
            parent_checkpoint_identity="parent",
            data_identity="data",
        )
    )

    trajectory = json.loads((tmp_path / "trajectory.json").read_text())
    assert result["selected"] == [
        {"layer_idx": 1, "kind": "moe"},
        {"layer_idx": 0, "kind": "mamba"},
    ]
    assert trajectory["status"] == "complete"
    assert trajectory["selected"] == result["selected"]
    assert len(trajectory["scenarios"]) == 3
    assert json.loads((tmp_path / "iteration_00/ranking.json").read_text())["rows"][0][
        "candidate"
    ] == {"layer_idx": 1, "kind": "moe"}
    assert json.loads((tmp_path / "iteration_01/ranking.json").read_text())["baseline"] == 1.1
    assert len(client.requests) == 4

    class NoSubmitClient:
        async def submit_many(self, requests):
            raise AssertionError(f"completed trajectory resubmitted {list(requests)}")

    resumed = asyncio.run(
        run_iterative_depth_rpc(
            _campaign(),
            client=NoSubmitClient(),
            teacher_blocks=teacher_blocks,
            available=available,
            hidden_width=8,
            output_dir=tmp_path,
            max_removals=2,
            source_checkpoint_dir="/checkpoint",
            parent_checkpoint_identity="parent",
            data_identity="data",
        )
    )
    assert resumed == result


def test_distributed_eval_cli_exposes_depth_campaign_and_coordinator():
    from modelopt.torch.puzzletron.distributed_eval.cli import build_parser

    parser = build_parser()
    init = parser.parse_args(
        [
            "init",
            "--campaign-dir",
            "/campaign",
            "--config",
            "/config.yaml",
            "--world-size",
            "8",
            "--stage",
            "depth",
        ]
    )
    coordinator = parser.parse_args(
        [
            "depth-coordinator",
            "--campaign-dir",
            "/campaign",
            "--config",
            "/config.yaml",
        ]
    )

    assert init.stage == "depth"
    assert coordinator.command == "depth-coordinator"
