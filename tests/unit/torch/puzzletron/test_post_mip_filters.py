# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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
