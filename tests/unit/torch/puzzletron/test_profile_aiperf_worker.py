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

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for sharded AIPerf worker selection, merging, and policy forwarding."""

import json
import sys

import pytest

from examples.puzzletron import run_profile_aiperf_worker as worker_module
from examples.puzzletron.run_profile_aiperf_worker import (
    TOPOLOGIES,
    build_work_items,
    expected_result_count,
    merge_results,
    run_worker,
    select_registry_solutions,
    shard_work,
)
from modelopt.torch.puzzletron import benchmarks


def test_profile_aiperf_filters_registry_to_explicit_solutions():
    registry = {
        "profile_id": "latency-095",
        "solutions": [
            {"solution_id": name, "checkpoint": f"/{name}"}
            for name in ("teacher", "h4096-d3", "h4096-d4")
        ],
    }

    selected = select_registry_solutions(registry, ("teacher", "h4096-d4"))

    assert [row["solution_id"] for row in selected["solutions"]] == [
        "teacher",
        "h4096-d4",
    ]


def test_profile_aiperf_work_matrix_uses_six_all_eight_gpu_topologies():
    registry = {
        "profile_id": "params-080",
        "solutions": [
            {"solution_id": name, "checkpoint": f"/{name}"}
            for name in ("teacher", "h1024-d0", "h1024-d1", "h0512-d0", "h0512-d1")
        ],
    }
    items = build_work_items(registry)

    assert len(items) == 30
    assert len({(row["solution_id"], row["topology_id"]) for row in items}) == 30
    assert all(row["gpu_count"] == 8 for row in items)
    assert any(row["topology"]["enable_expert_parallel"] for row in items)
    assert all("expert_parallel_size" not in row["topology"] for row in items)


def test_profile_aiperf_work_shards_cover_every_item_once():
    items = [{"id": value} for value in range(15)]
    shards = [shard_work(items, worker_index=index, worker_count=8) for index in range(8)]

    assert sorted(row["id"] for shard in shards for row in shard) == list(range(15))
    assert max(map(len, shards)) == 2


def test_profile_aiperf_expected_results_follow_registry_size():
    registry = {
        "solutions": [
            {"solution_id": "best-loss"},
            {"solution_id": "largest"},
        ]
    }

    assert expected_result_count(registry) == 48


def test_profile_aiperf_merge_honors_explicit_concurrency_subset(tmp_path):
    profile_id = "latency-095"
    solutions = ("teacher", "h4096-d4")
    registry_path = tmp_path / "mip/profiles" / profile_id / "selected_solutions.json"
    registry_path.parent.mkdir(parents=True)
    registry_path.write_text(
        json.dumps(
            {
                "profile_id": profile_id,
                "solutions": [
                    {"solution_id": solution_id, "checkpoint": f"/{solution_id}"}
                    for solution_id in solutions
                ],
            }
        )
    )
    root = tmp_path / "artifacts/aiperf/profiles" / profile_id / "isl-16384-osl-4096"
    for solution_id in solutions:
        for topology in TOPOLOGIES:
            result_path = (
                root
                / solution_id
                / topology["topology_id"]
                / "concurrency_1/puzzletron_aiperf_result.json"
            )
            result_path.parent.mkdir(parents=True)
            result_path.write_text(
                json.dumps(
                    {
                        "solution_id": solution_id,
                        "profile_id": profile_id,
                        "topology_id": topology["topology_id"],
                        "concurrency": 1,
                        "failures": 0,
                        "metrics": {
                            "input_sequence_length": 16384,
                            "output_sequence_length": 4096,
                        },
                    }
                )
            )

    output = merge_results(
        tmp_path,
        profile_id=profile_id,
        input_tokens=16384,
        output_tokens=4096,
        solution_ids=solutions,
        concurrencies=(1,),
    )

    payload = json.loads(output.read_text())
    assert payload["concurrencies"] == [1]
    assert len(payload["results"]) == 12


def test_profile_aiperf_cli_forwards_explicit_security_flags(tmp_path, monkeypatch):
    captured = {}

    def run_worker(puzzle_dir, **kwargs):
        captured["puzzle_dir"] = puzzle_dir
        captured.update(kwargs)
        return tmp_path / "worker.json"

    monkeypatch.setattr(worker_module, "run_worker", run_worker)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_profile_aiperf_worker.py",
            "--puzzle-dir",
            str(tmp_path),
            "--trust-remote-code",
            "--allow-aiperf-v011-online-tokenizer-resolution",
        ],
    )

    worker_module.main()

    assert captured["puzzle_dir"] == tmp_path
    assert captured["trust_remote_code"] is True
    assert captured["allow_aiperf_v011_online_tokenizer_resolution"] is True


@pytest.mark.parametrize(
    ("trust_remote_code", "online_tokenizer"),
    [(False, False), (True, False), (False, True), (True, True)],
)
def test_profile_aiperf_worker_forwards_security_policy_to_real_sweep(
    tmp_path, monkeypatch, trust_remote_code, online_tokenizer
):
    profile_id = "runtime-075"
    registry_path = tmp_path / "mip/profiles" / profile_id / "selected_solutions.json"
    registry_path.parent.mkdir(parents=True)
    registry_path.write_text(
        json.dumps(
            {
                "profile_id": profile_id,
                "solutions": [{"solution_id": "teacher", "checkpoint": "/teacher"}],
            }
        )
    )
    observed = []

    def run_aiperf_sweep(*args, **kwargs):
        observed.append((args, kwargs))
        return []

    monkeypatch.setattr(benchmarks, "run_aiperf_sweep", run_aiperf_sweep)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")

    run_worker(
        tmp_path,
        profile_id=profile_id,
        worker_index=0,
        worker_count=6,
        input_tokens=32,
        output_tokens=8,
        trust_remote_code=trust_remote_code,
        allow_aiperf_v011_online_tokenizer_resolution=online_tokenizer,
    )

    assert len(observed) == 1
    _, kwargs = observed[0]
    assert kwargs["trust_remote_code"] is trust_remote_code
    assert kwargs["allow_aiperf_v011_online_tokenizer_resolution"] is online_tokenizer
