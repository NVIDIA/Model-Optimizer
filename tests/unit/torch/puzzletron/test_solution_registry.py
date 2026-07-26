# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest


def _grid(tmp_path: Path) -> dict:
    teacher = tmp_path / "teacher"
    teacher.mkdir()
    rows = []
    for width, depth in ((1024, 0), (1024, 1), (512, 0), (512, 1), (512, 2)):
        solution = tmp_path / f"w{width}-d{depth}.json"
        solution.write_text("[]")
        checkpoint = tmp_path / f"w{width}-d{depth}"
        checkpoint.mkdir()
        rows.append(
            {
                "profile_id": "params-080",
                "hidden_width": width,
                "removed_sublayers": depth,
                "status": "feasible",
                "solution_path": str(solution),
                "checkpoint": str(checkpoint),
                "parameter_count": width * 1000 - depth,
                "parameter_ratio": width / 1280,
                "score": float(depth),
                "total_costs": {"stats.memory_mib": float(width)},
            }
        )
    return {
        "profile": {"id": "params-080", "parameter_ratio": 0.8},
        "teacher": {
            "label": "Teacher",
            "hidden_width": 1024,
            "removed_sublayers": 0,
            "checkpoint": str(teacher),
            "parameter_ratio": 1.0,
            "total_costs": {"stats.num_params": 1_280_000},
        },
        "scenarios": rows,
    }


def test_profile_solution_registry_selects_exact_pairs_and_stable_styles(tmp_path: Path):
    from modelopt.torch.puzzletron.solution_registry import build_profile_solution_registry

    registry = build_profile_solution_registry(
        _grid(tmp_path),
        selections=((1024, 0), (1024, 1), (512, 0), (512, 1)),
    )

    assert registry["profile_id"] == "params-080"
    assert [row["solution_id"] for row in registry["solutions"]] == [
        "teacher",
        "h1024-d0",
        "h1024-d1",
        "h0512-d0",
        "h0512-d1",
    ]
    teacher = registry["solutions"][0]
    assert teacher["always_enabled"] is True
    assert teacher["marker"] == "star"
    assert teacher["color"] == "#f5c451"
    assert all(row["marker"] == "circle" for row in registry["solutions"][1:])
    assert registry["solutions"][-1]["total_costs"]["stats.memory_mib"] == 512.0


def test_profile_solution_registry_rejects_missing_or_unrealized_candidate(tmp_path: Path):
    from modelopt.torch.puzzletron.solution_registry import build_profile_solution_registry

    grid = _grid(tmp_path)
    with pytest.raises(ValueError, match="no feasible MIP scenario"):
        build_profile_solution_registry(grid, selections=((768, 0),))

    grid["scenarios"][0].pop("checkpoint")
    with pytest.raises(ValueError, match="has no realized checkpoint"):
        build_profile_solution_registry(grid, selections=((1024, 0),))


def test_profile_solution_registry_writes_atomic_json(tmp_path: Path):
    from modelopt.torch.puzzletron.solution_registry import (
        build_profile_solution_registry,
        write_solution_registry,
    )

    registry = build_profile_solution_registry(_grid(tmp_path), selections=((512, 0),))
    output = write_solution_registry(tmp_path / "profiles" / "selected.json", registry)

    assert output == tmp_path / "profiles" / "selected.json"
    assert json.loads(output.read_text()) == registry
    assert not output.with_suffix(".json.tmp").exists()
