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


def test_profile_aiperf_filters_registry_to_explicit_solutions():
    from examples.puzzletron.run_profile_aiperf_worker import select_registry_solutions

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
    from examples.puzzletron.run_profile_aiperf_worker import build_work_items

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
    from examples.puzzletron.run_profile_aiperf_worker import shard_work

    items = [{"id": value} for value in range(15)]
    shards = [shard_work(items, worker_index=index, worker_count=8) for index in range(8)]

    assert sorted(row["id"] for shard in shards for row in shard) == list(range(15))
    assert max(map(len, shards)) == 2


def test_profile_aiperf_expected_results_follow_registry_size():
    from examples.puzzletron.run_profile_aiperf_worker import expected_result_count

    registry = {
        "solutions": [
            {"solution_id": "best-loss"},
            {"solution_id": "largest"},
        ]
    }

    assert expected_result_count(registry) == 48


def test_profile_aiperf_merge_honors_explicit_concurrency_subset(tmp_path):
    import json

    from examples.puzzletron.run_profile_aiperf_worker import TOPOLOGIES, merge_results

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
