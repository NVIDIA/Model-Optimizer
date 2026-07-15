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

import json

from omegaconf import OmegaConf

from modelopt.torch.puzzletron.block_config import (
    AttentionConfig,
    BlockConfig,
    FFNConfig,
    MambaConfig,
    MoEConfig,
)
from modelopt.torch.puzzletron.depth import mip_scenarios
from modelopt.torch.puzzletron.depth.iterative import _available_removals, _child_blocks
from modelopt.torch.puzzletron.depth.schema import DepthScenario, SublayerRemoval


def _blocks():
    return [
        BlockConfig(
            subblock_configs=(
                AttentionConfig(num_query_heads=8, num_kv_heads=2),
                FFNConfig(intermediate_size=16),
            )
        ),
        BlockConfig(subblock_configs=(FFNConfig(intermediate_size=12),)),
    ]


def test_subblock_depth_exposes_each_removable_subblock():
    removals = _available_removals(_blocks(), granularity="subblock")

    assert [(item.layer_idx, item.kind) for item in removals] == [
        (0, "attention"),
        (0, "ffn"),
        (1, "ffn"),
    ]


def test_subblock_depth_excludes_architectural_no_op_placeholders():
    blocks = [
        BlockConfig(
            subblock_configs=(
                MambaConfig(state_dim=4, num_heads=2, head_dim=2, num_groups=1),
                FFNConfig(no_op=True),
            )
        ),
        BlockConfig(
            subblock_configs=(
                AttentionConfig(no_op=True),
                MoEConfig(num_experts=4, expert_intermediate_size=8, top_k=2),
            )
        ),
    ]

    removals = _available_removals(blocks, granularity="subblock")

    assert [(item.layer_idx, item.kind) for item in removals] == [
        (0, "mamba"),
        (1, "moe"),
    ]


def test_block_depth_exposes_one_atomic_removal_per_layer():
    removals = _available_removals(_blocks(), granularity="block")

    assert [(item.layer_idx, item.kind) for item in removals] == [(0, "block"), (1, "block")]


def test_block_removal_disables_every_subblock_in_layer():
    child = _child_blocks(_blocks(), (SublayerRemoval(layer_idx=0, kind="block"),))[0]

    assert all(subblock.no_op for subblock in child.subblock_configs)


def test_legacy_depth_scenario_defaults_to_subblock():
    scenario = DepthScenario(
        parent_checkpoint_identity="teacher",
        hidden_width=16,
        removals=(),
        data_identity="data",
        evaluator_revision="revision",
    )

    assert scenario.granularity == "subblock"


def test_depth_mip_tournament_continues_after_infeasible_scenario(
    monkeypatch, tmp_path
):
    trajectory = tmp_path / "trajectory.json"
    trajectory.write_text(
        json.dumps({"scenarios": [{"removals": []}, {"removals": ["drop"]}]})
    )
    cfg = OmegaConf.create(
        {
            "mip": {
                "depth_trajectory_path": str(trajectory),
                "depth_scenario_count": 2,
                "output_path": str(tmp_path / "mip"),
            },
            "realize_model": {},
            "skip_realize_model": False,
        }
    )

    def run_puzzle(args):
        assert "stats.num_params" in args.report_additional_costs
        assert "stats.runtime_ms" in args.report_additional_costs
        index = int(str(args.output_path).rsplit("_", maxsplit=1)[-1])
        path = tmp_path / f"solution_{index}.json"
        path.write_text("[]\n" if index == 0 else '[{"score": 1.0}]\n')
        return [path]

    validation = {}
    monkeypatch.setattr(mip_scenarios, "run_puzzle", run_puzzle)
    monkeypatch.setattr(mip_scenarios.dist, "is_master", lambda: True)
    monkeypatch.setattr(mip_scenarios.dist, "barrier", lambda: None)
    monkeypatch.setattr(mip_scenarios, "_broadcast", lambda value: value)
    monkeypatch.setattr(
        mip_scenarios,
        "validate_puzzle_solutions",
        lambda args, hydra_cfg: validation.update(
            {"indices": list(args.solutions_to_validate)}
        ),
    )

    paths = mip_scenarios.run_depth_mip_scenarios(cfg)

    summary = json.loads((tmp_path / "mip/depth_tournament/scenario_results.json").read_text())
    combined = json.loads((tmp_path / "mip/depth_tournament/solutions.json").read_text())
    assert [row["status"] for row in summary] == ["infeasible", "feasible"]
    assert len(combined) == 1
    assert combined[0]["depth_scenario"]["index"] == 1
    assert validation["indices"] == [0]
    assert len(paths) == 4
