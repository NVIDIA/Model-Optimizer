# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from examples.puzzletron.prepare_online_profile_finalists import (
    build_finalist_registry,
    select_online_finalists,
)


def _candidate(name, loss, *, width=2688):
    return {
        "solution_id": name,
        "architecture_id": f"arch-{name}",
        "architecture_solution_index": 0,
        "architecture_solutions_path": f"/{name}/solutions.json",
        "hidden_width": width,
        "removed_sublayers": 1,
        "parameter_count": 75,
        "metrics": {"lm_loss": loss},
    }


def test_online_finalists_rank_finite_candidates_and_exclude_teacher():
    summary = {
        "teacher": {"solution_id": "teacher", "metrics": {"lm_loss": 0.1}},
        "solutions": [
            _candidate("fourth", 0.5),
            _candidate("best", 0.2),
            _candidate("invalid", float("nan")),
            _candidate("third", 0.4),
            _candidate("second", 0.3),
        ],
    }

    assert [row["solution_id"] for row in select_online_finalists(summary, 3)] == [
        "best",
        "second",
        "third",
    ]


def test_finalist_registry_contains_teacher_three_candidates_and_absolute_best():
    finalists = [_candidate("best", 0.2), _candidate("second", 0.3), _candidate("third", 0.4)]
    summary = {
        "teacher": {
            "hidden_width": 2688,
            "checkpoint": "/teacher",
            "metrics": {"lm_loss": 0.1},
        }
    }
    grid = {
        "profile": {"id": "runtime-075"},
        "teacher": {"parameter_count": 100},
    }
    checkpoints = {row["solution_id"]: Path(f"/{row['solution_id']}") for row in finalists}

    registry = build_finalist_registry(
        profile_id="runtime-075",
        summary=summary,
        mip_grid=grid,
        finalists=finalists,
        checkpoints=checkpoints,
    )

    assert registry["absolute_best_solution_id"] == "best"
    assert [row["solution_id"] for row in registry["solutions"]] == [
        "teacher",
        "best",
        "second",
        "third",
    ]
