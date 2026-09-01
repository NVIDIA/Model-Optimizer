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

import pytest

from modelopt.torch.puzzletron.diagnostics.campaign_progress_report import _live_kd_metadata
from modelopt.torch.puzzletron.diagnostics.campaign_report import generate_campaign_report


def _write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _scenario(root: Path, width: int, depth: int) -> dict:
    name = f"width-{width:04d}/depth-{depth:02d}"
    checkpoint = (
        root
        / "artifacts/global_kd/scenarios"
        / name
        / "checkpoints/epoch_7_step_7/model/consolidated"
    )
    training = checkpoint.parents[2] / "training.jsonl"
    training.parent.mkdir(parents=True, exist_ok=True)
    training.write_text(
        "\n".join(
            json.dumps(
                {
                    "step": step,
                    "loss": loss,
                    "gradient_norm_vision": 1.0,
                    "gradient_norm_projector": 1.0,
                    "gradient_norm_language": 1.0,
                    "gradient_norm_mtp": 1.0,
                }
            )
            for step, loss in enumerate((3.0, 2.0, 1.0))
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "checkpoint": str(checkpoint),
        "hidden_width": width,
        "metrics": {"lm_loss": float(depth + width / 1000), "kl_div": 0.5},
        "observability": {"vision_forward_count": 4},
        "result_path": str(root / "result.json"),
    }


def test_live_kd_metadata_reports_configured_optimizer_steps(tmp_path: Path):
    _write(tmp_path / "global_kd_recipe.yaml", {"step_scheduler": {"max_steps": 128}})

    assert _live_kd_metadata(tmp_path)["max_steps"] == 128


def test_report_requires_and_includes_requested_1024_candidate(tmp_path: Path):
    scenarios = [_scenario(tmp_path, 512, depth) for depth in range(4)]
    scenarios.append(_scenario(tmp_path, 1024, 0))
    _write(tmp_path / "artifacts/post_kd_evaluation/evaluation_summary.json", scenarios)
    _write(tmp_path / "artifacts/exact_evaluation/evaluation_summary.json", scenarios)
    _write(tmp_path / "artifacts/sort_equivalence/sort_equivalence_summary.json", {"passed": True})
    _write(
        tmp_path / "artifacts/activation_diagnostic/activation_diagnostic_summary.json",
        {"status": "complete"},
    )
    _write(
        tmp_path / "artifacts/bypass/local_kd_loss_history.json",
        {"records": [{"loss": 2.0}, {"loss": 1.0}]},
    )
    _write(
        tmp_path / "artifacts/bypass/nested_axis_coverage.json",
        {"observed_options": {"width": [512, 1024]}},
    )
    _write(tmp_path / "candidate_library.json", [{"id": 1}])
    _write(tmp_path / "replacement_library.json", [{"id": 1}])
    _write(tmp_path / "subblock_stats.json", {"block_runtimes": {"x": 1.0}})
    _write(tmp_path / "scenarios/mip_grid.json", {"scenarios": []})
    for stage in ("convert", "activation", "sort", "bypass", "build_library"):
        _write(tmp_path / f"manifests/{stage}.json", {"status": "complete"})
    checkpoint_names = [
        "teacher",
        "width-0512__depth-00",
        "width-0512__depth-01",
        "width-0512__depth-02",
        "width-0512__depth-03",
        "width-1024__depth-00",
    ]
    aiperf = [
        {
            "checkpoint_name": name,
            "concurrency": concurrency,
            "metrics": {"request_latency_mean_ms": 10.0, "request_throughput": 2.0},
        }
        for name in checkpoint_names
        for concurrency in (1, 2)
    ]
    _write(tmp_path / "artifacts/aiperf/aiperf_results.json", aiperf)

    result = generate_campaign_report(tmp_path, model_name="Qwen3.5-0.8B", expected_kd_scenarios=5)

    assert result["status"] == "complete"
    assert [row["scenario"] for row in result["post_kd"]][-1] == "width-1024/depth-00"
    assert result["post_kd"][-1]["kd_loss"]["improved"] is True
    assert Path(result["reports"]["json"]).is_file()
    assert Path(result["reports"]["html"]).is_file()


def test_report_rejects_missing_1024_reference(tmp_path: Path):
    scenarios = [_scenario(tmp_path, 512, depth) for depth in range(4)]
    _write(tmp_path / "artifacts/post_kd_evaluation/evaluation_summary.json", scenarios)

    with pytest.raises(RuntimeError, match="expected 5 post-KD scenarios"):
        generate_campaign_report(tmp_path, expected_kd_scenarios=5)
