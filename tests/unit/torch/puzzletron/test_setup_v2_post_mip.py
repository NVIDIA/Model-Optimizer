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

from collections import OrderedDict

from puzzletron_setup.v2.post_mip import FlowDraft, NodeDraft, PostMIPFlowEditor, recommended_flow


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


def test_downstream_evaluation_node_is_configurable_after_materialization():
    flow = FlowDraft(
        "runtime",
        "runtime",
        nodes=OrderedDict(
            (
                ("materialized", NodeDraft("materialized", "materialize")),
                (
                    "lmms_eval",
                    NodeDraft(
                        "lmms_eval",
                        "downstream_evaluation",
                        input_id="materialized",
                        config={"tasks": ["ifeval"]},
                    ),
                ),
            )
        ),
    )
    editor = PostMIPFlowEditor({"runtime": {}})
    editor.add_flow(flow)

    review = editor.review("runtime")

    assert review.node_order == ("materialized", "lmms_eval")
    assert review.parents["lmms_eval"] == ("materialized",)
    assert review.artifacts["lmms_eval"] == "checkpoint"
