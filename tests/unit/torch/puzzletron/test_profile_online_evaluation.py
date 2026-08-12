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
from pathlib import Path

from examples.puzzletron.run_profile_online_evaluation import (
    build_online_evaluation_plan,
    merge_online_evaluation,
    online_execution_contract,
    shard_solution_indices,
    write_online_evaluation_plan,
)


def _write(path: Path, payload) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    return path


def _solution(width: int, experts: int) -> dict:
    return {
        "hidden_width": width,
        "chosen_replacements": [
            {
                "layer_replacement": {
                    "weight_paths": [],
                    "parent_layer_indices": [0],
                    "child_block_configs": [
                        {
                            "subblock_configs": [
                                {
                                    "kind": "moe",
                                    "name": "moe",
                                    "no_op": False,
                                    "num_experts": experts,
                                    "expert_intermediate_size": 8,
                                    "shared_expert_intermediate_size": 16,
                                    "top_k": 2,
                                    "latent_dim": None,
                                }
                            ]
                        }
                    ],
                }
            }
        ],
    }


def test_online_plan_deduplicates_architectures_across_profile_aliases(tmp_path):
    puzzle_dir = tmp_path / "run"
    first = _solution(8, 4)
    second = _solution(8, 2)
    first_path = _write(tmp_path / "first.json", [first])
    second_path = _write(tmp_path / "second.json", [second])
    homogeneous_path = _write(tmp_path / "homogeneous.json", [first])
    for profile_id, solution_path in (("params", first_path), ("runtime", second_path)):
        grid = {
            "profile": {"id": profile_id},
            "teacher": {"hidden_width": 8},
            "scenarios": [
                {
                    "status": "feasible",
                    "hidden_width": 8,
                    "removed_sublayers": 0,
                    "depth_selection": {"total": 0},
                    "solution_path": str(solution_path),
                    "solutions": [{"rank": 0}],
                    "homogeneous_solution_path": str(homogeneous_path),
                    "homogeneous_solutions": [{"rank": 0}],
                }
            ],
        }
        _write(
            puzzle_dir / "mip" / "profiles" / profile_id / "mip_grid.json",
            grid,
        )

    plan = build_online_evaluation_plan(puzzle_dir, ("params", "runtime"))

    assert plan["logical_solution_count"] == 4
    assert plan["unique_architecture_count"] == 2
    assert len(plan["architectures_by_width"][8]) == 2
    aliases = {
        architecture["architecture_id"]: architecture["aliases"]
        for architecture in plan["architectures_by_width"][8]
    }
    assert sorted(len(rows) for rows in aliases.values()) == [1, 3]


def test_online_merge_fans_unique_metrics_back_to_every_profile_alias(tmp_path):
    puzzle_dir = tmp_path / "run"
    first = _solution(8, 4)
    second = _solution(8, 2)
    for profile_id, solution in (("params", first), ("runtime", second)):
        solution_path = _write(tmp_path / f"{profile_id}.json", [solution])
        homogeneous_path = _write(tmp_path / f"{profile_id}-homogeneous.json", [first])
        _write(
            puzzle_dir / "mip" / "profiles" / profile_id / "mip_grid.json",
            {
                "profile": {"id": profile_id},
                "teacher": {
                    "hidden_width": 8,
                    "parameter_count": 100,
                    "total_costs": {
                        "stats.num_params": 100,
                        "stats.runtime_ms@serving-8k": 10.0,
                    },
                },
                "scenarios": [
                    {
                        "status": "feasible",
                        "hidden_width": 8,
                        "removed_sublayers": 0,
                        "solution_path": str(solution_path),
                        "solutions": [{"rank": 0}],
                        "homogeneous_solution_path": str(homogeneous_path),
                        "homogeneous_solutions": [{"rank": 0}],
                    }
                ],
            },
        )
    scenario = puzzle_dir / "scenarios" / "width-0008" / "depth-00"
    _write(
        scenario / "scenario_manifest.json",
        {
            "parent_checkpoint": str(scenario / "ckpts" / "sorted_teacher"),
            "bypass_checkpoint": str(scenario / "ckpts" / "bypass_overlay"),
        },
    )
    plan = build_online_evaluation_plan(puzzle_dir, ("params", "runtime"))
    write_online_evaluation_plan(puzzle_dir, plan)
    written_plan = json.loads(
        (
            puzzle_dir
            / "artifacts"
            / "zero_shot_evaluation"
            / "online_plan"
            / "index.json"
        ).read_text()
    )
    assert written_plan["execution"] == {
        "mode": "resident_sorted_teacher_online",
        "materialized_solution_checkpoints": False,
        "model_loads_per_worker": 1,
    }
    assert written_plan["widths"]["8"]["execution"] == online_execution_contract(
        puzzle_dir, 8
    )
    raw = (
        puzzle_dir
        / "artifacts"
        / "zero_shot_evaluation"
        / "online_plan"
        / "raw"
        / "width-0008"
        / "shard-00"
    )
    for index in range(2):
        _write(
            raw / f"solution_{index}.json",
            {
                "lm_loss": {"avg": 1.0 + index, "per_sample": [1.0 + index]},
                "kl_div": {"avg": float(index), "per_sample": [float(index)]},
            },
        )
    _write(
        raw / "sliced_teacher.json",
        {
            "lm_loss": {"avg": 0.5, "per_sample": [0.5]},
            "kl_div": {"avg": 0.0, "per_sample": [0.0]},
        },
    )

    output = merge_online_evaluation(
        puzzle_dir,
        eval_samples=128,
        block_size=8192,
    )

    assert output.is_file()
    runtime = json.loads(
        (
            puzzle_dir
            / "artifacts"
            / "zero_shot_evaluation"
            / "profiles"
            / "runtime"
            / "text-s128-l8192"
            / "evaluation_summary.json"
        ).read_text()
    )
    assert len(runtime["solutions"]) == 2
    assert [row["rank"] for row in runtime["solutions"]] == [1, 2]
    assert runtime["teacher"]["metrics"]["lm_loss"] == 0.5
    assert runtime["teacher"]["parameter_count"] == 100
    assert runtime["teacher"]["parameter_ratio"] == 1.0
    assert runtime["teacher"]["total_costs"]["stats.runtime_ms@serving-8k"] == 10.0


def test_online_solution_shards_are_disjoint_and_cover_the_plan():
    shards = [shard_solution_indices(11, index, 3) for index in range(3)]

    assert shards == [[0, 3, 6, 9], [1, 4, 7, 10], [2, 5, 8]]
    assert sorted(value for shard in shards for value in shard) == list(range(11))


def test_online_execution_contract_uses_resident_sorted_teacher_and_nested_bypass(tmp_path):
    puzzle_dir = tmp_path / "run"
    scenario = puzzle_dir / "scenarios" / "width-2688" / "depth-00"
    _write(
        scenario / "scenario_manifest.json",
        {
            "parent_checkpoint": str(scenario / "ckpts" / "sorted_teacher"),
            "bypass_checkpoint": str(scenario / "ckpts" / "bypass_overlay"),
        },
    )

    contract = online_execution_contract(puzzle_dir, 2688)

    assert contract == {
        "mode": "resident_sorted_teacher_online",
        "materialized_solution_checkpoints": False,
        "model_loads_per_worker": 1,
        "checkpoint_roles": {
            "source": str(scenario / "ckpts" / "sorted_teacher"),
            "target": str(scenario / "ckpts" / "sorted_teacher"),
            "bypass_overlay": str(scenario / "ckpts" / "bypass_overlay"),
        },
    }
