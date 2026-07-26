# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from puzzletron_setup.v2.post_mip import recommended_flow


def test_recommended_flow_propagates_aiperf_sweep_selection_mode():
    flow = recommended_flow(
        "params",
        ["metrics.lm_loss"],
        {"sequence_length": 1024},
        {
            "input_tokens": 128,
            "output_tokens": 32,
            "concurrency": [1, 4, 8],
            "request_count": 64,
            "best_selection_mode": "best_per_concurrency",
        },
    )

    serving = flow.nodes["serving"]
    fastest = flow.nodes["fastest"]
    assert serving.config["concurrency"] == [1, 4, 8]
    assert fastest.selector == {
        "mode": "top_k",
        "metric": "serving.output_token_throughput",
        "direction": "maximize",
        "top_k": 4,
        "best_selection_mode": "best_per_concurrency",
    }


def test_recommended_flow_accepts_a_single_concurrency_value():
    flow = recommended_flow(
        "params",
        ["metrics.lm_loss"],
        {"sequence_length": 1024},
        {
            "input_tokens": 128,
            "output_tokens": 32,
            "concurrency": 2,
            "request_count": 32,
        },
    )

    assert flow.nodes["serving"].config["concurrency"] == [2]
    assert flow.nodes["fastest"].selector["best_selection_mode"] == "individual_best"
