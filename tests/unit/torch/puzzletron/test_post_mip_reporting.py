# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from modelopt.torch.puzzletron.post_mip.reporting import (
    render_aiperf_report,
    render_global_kd_report,
)


def test_aiperf_report_highlights_downstream_selection():
    payload = {
        "status": "success",
        "observations": [
            {
                "architecture_id": "architecture_fast",
                "label": "fast",
                "color": "#123456",
                "input_revision_id": "revision_fast",
                "status": "success",
                "selected_by": ["fastest"],
                "metrics": {
                    "request_throughput": 10.0,
                    "output_token_throughput": 1280.0,
                    "ttft_mean_ms": 25.0,
                    "tpot_mean_ms": 2.0,
                },
            },
            {
                "architecture_id": "architecture_slow",
                "label": "slow",
                "color": "#654321",
                "input_revision_id": "revision_slow",
                "status": "timed_out",
                "selected_by": [],
                "metrics": {},
                "error": "benchmark timed out",
            },
        ],
    }

    report = render_aiperf_report("post-memory-serving", payload)

    assert "AIPerf candidate trade-offs" in report
    assert "Selected by Fastest" in report
    assert "benchmark timed out" in report
    assert "post-memory-serving-throughput" in report


def test_global_kd_report_compares_candidates_on_shared_plots():
    payload = {
        "status": "success",
        "runs": [
            {
                "architecture_id": "architecture_a",
                "label": "candidate A",
                "color": "#123456",
                "status": "success",
                "records": [
                    {"step": 0, "loss": 4.0, "kd_loss": 3.0},
                    {"step": 1, "loss": 2.0, "kd_loss": 1.5},
                ],
            },
            {
                "architecture_id": "architecture_b",
                "label": "candidate B",
                "color": "#654321",
                "status": "success",
                "records": [
                    {"step": 0, "loss": 5.0, "kd_loss": 3.5},
                    {"step": 1, "loss": 2.5, "kd_loss": 1.75},
                ],
            },
        ],
    }

    report = render_global_kd_report("post-memory-short-kd", payload)

    assert "Short KD comparison" in report
    assert "post-memory-short-kd-loss" in report
    assert "post-memory-short-kd-kd-loss" in report
    assert "candidate A" in report
    assert "candidate B" in report
