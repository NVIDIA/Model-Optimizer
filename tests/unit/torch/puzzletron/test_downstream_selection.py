# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from modelopt.torch.puzzletron.pipeline_config import normalize_pipeline_config
from modelopt.torch.puzzletron.stages.future import (
    _select_evaluated_candidates,
    _with_teacher_checkpoint,
)


def test_downstream_selection_defaults_and_validates_candidate_counts():
    normalized = normalize_pipeline_config({"aiperf": {}, "distillation": {}})

    assert normalized["aiperf"]["num_best_to_eval"] == 1
    assert normalized["distillation"]["num_best_to_distill"] == 1

    with pytest.raises(ValueError, match=r"aiperf.num_best_to_eval must be a positive integer"):
        normalize_pipeline_config({"aiperf": {"num_best_to_eval": 0}})

    with pytest.raises(
        ValueError, match=r"distillation.num_best_to_distill must be a positive integer"
    ):
        normalize_pipeline_config({"distillation": {"num_best_to_distill": True}})


def test_evaluated_candidate_selection_is_deterministic_and_merges_reasons():
    rows = [
        {
            "checkpoint": "/candidates/b",
            "solution_id": "b",
            "metrics": {"lm_loss": 0.5},
            "selection_reasons": ["largest_parameter_count"],
        },
        {
            "checkpoint": "/candidates/a",
            "solution_id": "a",
            "metrics": {"lm_loss": 0.5},
            "selection_reasons": ["lowest_lm_loss"],
        },
        {
            "checkpoint": "/candidates/a",
            "solution_id": "a-duplicate",
            "metrics": {"lm_loss": 0.8},
            "selection_reasons": ["largest_parameter_count"],
        },
    ]

    assert _select_evaluated_candidates(rows, num_best=2) == [
        {
            "checkpoint": "/candidates/a",
            "solution_id": "a",
            "metrics": {"lm_loss": 0.5},
            "selection_reasons": ["lowest_lm_loss", "largest_parameter_count"],
        },
        {
            "checkpoint": "/candidates/b",
            "solution_id": "b",
            "metrics": {"lm_loss": 0.5},
            "selection_reasons": ["largest_parameter_count"],
        },
    ]


def test_teacher_checkpoint_is_prepended_once_to_downstream_candidates():
    assert _with_teacher_checkpoint(
        "/teacher",
        [("candidate", "/candidate"), ("teacher", "/teacher")],
    ) == [("teacher", "/teacher"), ("candidate", "/candidate")]
