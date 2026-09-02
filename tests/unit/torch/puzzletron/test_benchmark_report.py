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

"""Tests for self-contained serving comparison reports."""

import csv
import json
from datetime import datetime, timezone

from modelopt.torch.puzzletron.benchmarks.report import write_aiperf_report
from modelopt.torch.puzzletron.benchmarks.schema import BenchmarkResult


def test_aiperf_report_plots_repetition_medians_by_workload(tmp_path):
    results = []
    for solution_id, profile_id, topology_id, parameters, throughputs in (
        ("teacher", "profile", "topology", 100, (10.0, 14.0)),
        ("student", "profile", "topology-a", 60, (8.0, 12.0)),
        ("student", "alternate-profile", "topology-b", 60, (18.0, 22.0)),
    ):
        for repetition, throughput in enumerate(throughputs):
            results.append(
                BenchmarkResult(
                    architecture_id=f"architecture-{solution_id}",
                    solution_id=solution_id,
                    profile_id=profile_id,
                    topology_id=topology_id,
                    workload_id="images-1",
                    checkpoint_dir=str(tmp_path / "checkpoints" / solution_id),
                    concurrency=1,
                    repetition=repetition,
                    checkpoint_identity={
                        "serialized_size_bytes": parameters * 2,
                        "parameter_count": parameters,
                    },
                    workload={"image_batch_size": 1, "input_tokens": 100, "output_tokens": 80},
                    metrics={"output_token_throughput": throughput},
                    result_fingerprint=f"fingerprint-{solution_id}-{repetition}",
                    failures=0,
                    topology={"gpu_count": 1},
                    raw_artifacts={"profile": f"profile-{solution_id}-{repetition}.json"},
                    command=("aiperf",),
                    started_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
                )
            )

    paths = write_aiperf_report(results, tmp_path)

    report = (tmp_path / "aiperf_report.html").read_text()
    encoded_points = report.split("<script id='plot-data' type='application/json'>", maxsplit=1)[
        1
    ].split("</script>", maxsplit=1)[0]
    points = json.loads(encoded_points)
    assert paths["html"] == str(tmp_path / "aiperf_report.html")
    points_by_identity = {
        (point["solution_id"], point["profile_id"], point["topology_id"]): point for point in points
    }
    assert {
        identity: (point["repetitions"], point["metrics"])
        for identity, point in points_by_identity.items()
    } == {
        ("student", "alternate-profile", "topology-b"): (
            2,
            {"output_token_throughput": 20.0},
        ),
        ("student", "profile", "topology-a"): (2, {"output_token_throughput": 10.0}),
        ("teacher", "profile", "topology"): (2, {"output_token_throughput": 12.0}),
    }
    assert "id='workload'" in report

    json_rows = json.loads((tmp_path / "aiperf_results.json").read_text())
    assert json_rows[0]["repetition"] == 0
    assert json_rows[0]["checkpoint_identity"] == {
        "parameter_count": 100,
        "serialized_size_bytes": 200,
    }
    with (tmp_path / "aiperf_results.csv").open(newline="", encoding="utf-8") as stream:
        csv_rows = list(csv.DictReader(stream))
    assert len(csv_rows) == len(results)
    assert csv_rows[0]["profile_id"] == "profile"
    assert csv_rows[0]["repetition"] == "0"
    assert csv_rows[0]["serialized_size_bytes"] == "200"
    assert csv_rows[0]["result_fingerprint"] == "fingerprint-teacher-0"
    assert csv_rows[0]["output_token_throughput"] == "10.0"
