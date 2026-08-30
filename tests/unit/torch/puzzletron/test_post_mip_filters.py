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

import math

import pytest

from modelopt.torch.puzzletron.post_mip.filters import apply_filter, validate_filter_config
from modelopt.torch.puzzletron.post_mip.records import (
    ArchitectureCandidate,
    ArtifactKind,
    CandidateLedger,
    CandidateRevision,
    NodeObservation,
)
from modelopt.torch.puzzletron.post_mip.reporting import render_aiperf_report


def _ledger(tmp_path, metrics_by_revision):
    ledger = CandidateLedger(tmp_path)
    for revision_id, metrics in metrics_by_revision.items():
        architecture_id = revision_id.replace("revision", "architecture")
        ledger.architectures[architecture_id] = ArchitectureCandidate(
            architecture_id=architecture_id,
            block_configs=[],
        )
        ledger.revisions[revision_id] = CandidateRevision(
            revision_id=revision_id,
            architecture_id=architecture_id,
            artifact_kind=ArtifactKind.CHECKPOINT,
            artifact={"checkpoint": f"/checkpoints/{architecture_id}"},
        )
        ledger.observations.setdefault("serving", {})[revision_id] = NodeObservation(
            node_id="serving",
            input_revision_id=revision_id,
            source_revision_id=revision_id,
            output_revision_id=None,
            status="completed",
            metrics=dict(metrics),
        )
    return ledger


def _sweep(*values):
    return {
        f"concurrency_{concurrency}.output_token_throughput": value for concurrency, value in values
    }


def _vlm_sweep(*values):
    return {
        f"images_12.concurrency_{concurrency}.image_throughput": value
        for concurrency, value in values
    }


def test_individual_best_ranks_each_model_by_its_best_concurrency(tmp_path):
    ledger = _ledger(
        tmp_path,
        {
            "revision-a": _sweep((1, 20.0), (8, 10.0)),
            "revision-b": _sweep((1, 15.0), (8, 30.0)),
            "revision-c": _sweep((1, 12.0), (8, 15.0)),
        },
    )

    selected, excluded, scores = apply_filter(
        ledger,
        ("revision-a", "revision-b", "revision-c"),
        {
            "mode": "top_k",
            "metric": "serving.output_token_throughput",
            "direction": "maximize",
            "top_k": 2,
            "best_selection_mode": "individual_best",
        },
    )

    assert selected == ("revision-b", "revision-a")
    assert excluded == {"revision-c": "outside top_k"}
    assert scores == {
        "revision-a": 20.0,
        "revision-b": 30.0,
        "revision-c": 15.0,
    }


def test_multimodal_sweep_selection_resolves_one_image_workload(tmp_path):
    ledger = _ledger(
        tmp_path,
        {
            "revision-a": _vlm_sweep((1, 20.0), (2, 25.0)),
            "revision-b": _vlm_sweep((1, 30.0), (2, 15.0)),
        },
    )

    selected, excluded, scores = apply_filter(
        ledger,
        ("revision-a", "revision-b"),
        {
            "mode": "top_k",
            "metric": "serving.images_12.image_throughput",
            "direction": "maximize",
            "top_k": 1,
            "best_selection_mode": "individual_best",
        },
    )

    assert selected == ("revision-b",)
    assert excluded == {"revision-a": "outside top_k"}
    assert scores == {"revision-a": 25.0, "revision-b": 30.0}


def test_aiperf_report_renders_multimodal_namespaced_metrics():
    html = render_aiperf_report(
        "vlm-serving",
        {
            "status": "completed",
            "observations": [
                {
                    "label": "candidate-a",
                    "status": "success",
                    "selected_by": ["fastest_vlm"],
                    "metrics": {
                        "images_12.concurrency_1.request_throughput": 2.0,
                        "images_12.concurrency_1.image_throughput": 24.0,
                        "images_12.concurrency_1.output_token_throughput": 128.0,
                        "images_12.concurrency_1.ttft_mean_ms": 10.0,
                        "images_12.concurrency_1.tpot_mean_ms": 3.0,
                    },
                }
            ],
        },
    )

    assert "images 12 / concurrency 1" in html
    assert "<td>24</td>" in html
    assert "<td>128</td>" in html
    assert "Selected by Fastest" in html


def test_best_per_concurrency_unions_top_k_from_each_point(tmp_path):
    ledger = _ledger(
        tmp_path,
        {
            "revision-a": _sweep((1, 30.0), (8, 5.0)),
            "revision-b": _sweep((1, 20.0), (8, 25.0)),
            "revision-c": _sweep((1, 10.0), (8, 40.0)),
            "revision-d": _sweep((1, 5.0), (8, 10.0)),
        },
    )

    selected, excluded, scores = apply_filter(
        ledger,
        ("revision-a", "revision-b", "revision-c", "revision-d"),
        {
            "mode": "top_k",
            "metric": "serving.output_token_throughput",
            "direction": "maximize",
            "top_k": 2,
            "best_selection_mode": "best_per_concurrency",
        },
    )

    assert selected == ("revision-a", "revision-c", "revision-b")
    assert excluded == {"revision-d": "outside top_k at every concurrency"}
    assert scores == {
        "revision-a": 1.0,
        "revision-b": 2.0,
        "revision-c": 1.0,
        "revision-d": 3.0,
    }


def test_individual_best_respects_minimize_direction(tmp_path):
    ledger = _ledger(
        tmp_path,
        {
            "revision-a": _sweep((1, 3.0), (8, 1.0)),
            "revision-b": _sweep((1, 2.0), (8, 4.0)),
        },
    )

    selected, _excluded, scores = apply_filter(
        ledger,
        ("revision-a", "revision-b"),
        {
            "mode": "top_k",
            "metric": "serving.output_token_throughput",
            "direction": "minimize",
            "top_k": 1,
            "best_selection_mode": "individual_best",
        },
    )

    assert selected == ("revision-a",)
    assert scores == {"revision-a": 1.0, "revision-b": 2.0}


def test_sweep_selection_excludes_partial_or_non_finite_models(tmp_path):
    ledger = _ledger(
        tmp_path,
        {
            "revision-a": _sweep((1, 20.0), (8, 30.0)),
            "revision-b": _sweep((1, 40.0)),
            "revision-c": _sweep((1, 50.0), (8, math.nan)),
        },
    )

    selected, excluded, scores = apply_filter(
        ledger,
        ("revision-a", "revision-b", "revision-c"),
        {
            "mode": "top_k",
            "metric": "serving.output_token_throughput",
            "direction": "maximize",
            "top_k": 2,
            "best_selection_mode": "individual_best",
        },
    )

    assert selected == ("revision-a",)
    assert excluded == {
        "revision-b": "incomplete concurrency sweep; missing [8]",
        "revision-c": "incomplete concurrency sweep; missing [8]",
    }
    assert scores == {"revision-a": 30.0}


@pytest.mark.parametrize("best_selection_mode", ["unknown", "", 3])
def test_top_k_rejects_unknown_best_selection_mode(best_selection_mode):
    with pytest.raises(ValueError, match="best_selection_mode"):
        validate_filter_config(
            {
                "mode": "top_k",
                "metric": "serving.output_token_throughput",
                "direction": "maximize",
                "top_k": 2,
                "best_selection_mode": best_selection_mode,
            }
        )


def test_best_selection_mode_is_rejected_on_non_top_k_filter():
    with pytest.raises(ValueError, match="best_selection_mode"):
        validate_filter_config(
            {
                "mode": "threshold",
                "metric": "serving.output_token_throughput",
                "min": 1,
                "best_selection_mode": "individual_best",
            }
        )


def test_require_match_must_be_boolean():
    with pytest.raises(TypeError, match="require_match"):
        validate_filter_config(
            {
                "mode": "threshold",
                "metric": "serving.output_token_throughput",
                "min": 1,
                "require_match": "yes",
            }
        )


@pytest.mark.parametrize("metric", ["output_token_throughput", "mip.score"])
def test_sweep_selection_requires_a_node_qualified_metric(metric):
    with pytest.raises(ValueError, match="node-qualified"):
        validate_filter_config(
            {
                "mode": "top_k",
                "metric": metric,
                "direction": "maximize",
                "top_k": 2,
                "best_selection_mode": "individual_best",
            }
        )
