# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Profile-scoped solution identities shared by evaluation and serving reports."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Iterable

__all__ = [
    "build_profile_solution_registry",
    "canonical_solution_id",
    "write_solution_registry",
]


_TEACHER_STYLE = {"color": "#f5c451", "marker": "star", "always_enabled": True}
_CANDIDATE_COLORS = ("#4f8cff", "#22d3ee", "#ff6577", "#ff9f43")


def canonical_solution_id(hidden_width: int, removed_sublayers: int) -> str:
    """Return a stable width/depth identity independent of artifact paths."""

    return f"h{int(hidden_width):04d}-d{int(removed_sublayers)}"


def _checkpoint(row: dict[str, Any], *, label: str) -> str:
    value = row.get("checkpoint")
    if not value:
        raise ValueError(f"{label} has no realized checkpoint")
    path = Path(str(value))
    if not path.is_dir():
        raise ValueError(f"{label} realized checkpoint does not exist: {path}")
    return str(path)


def _costs(row: dict[str, Any]) -> dict[str, float]:
    return {
        str(key): float(value)
        for key, value in (row.get("total_costs") or {}).items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }


def build_profile_solution_registry(
    mip_grid: dict[str, Any],
    *,
    selections: Iterable[tuple[int, int]],
) -> dict[str, Any]:
    """Select realized width/depth scenarios and assign stable report styles."""

    profile = dict(mip_grid.get("profile") or {})
    profile_id = str(profile.get("id") or "")
    if not profile_id:
        raise ValueError("MIP grid profile is missing an id")
    selected = [(int(width), int(depth)) for width, depth in selections]
    if not selected or len(selected) != len(set(selected)):
        raise ValueError("solution selections must be non-empty and unique")
    if len(selected) > len(_CANDIDATE_COLORS):
        raise ValueError(
            f"this comparison registry supports at most {len(_CANDIDATE_COLORS)} candidates"
        )

    teacher = dict(mip_grid.get("teacher") or {})
    teacher_checkpoint = _checkpoint(teacher, label="teacher")
    solutions: list[dict[str, Any]] = [
        {
            "solution_id": "teacher",
            "label": str(teacher.get("label") or "Teacher"),
            "hidden_width": int(teacher["hidden_width"]),
            "removed_sublayers": int(teacher.get("removed_sublayers", 0)),
            "checkpoint": teacher_checkpoint,
            "color": _TEACHER_STYLE["color"],
            "marker": _TEACHER_STYLE["marker"],
            "always_enabled": True,
            "parameter_ratio": float(teacher.get("parameter_ratio", 1.0)),
            "score": teacher.get("score"),
            "total_costs": _costs(teacher),
            "source_identity": {"profile_id": profile_id, "kind": "teacher"},
        }
    ]

    scenarios = list(mip_grid.get("scenarios") or [])
    for color, (width, depth) in zip(_CANDIDATE_COLORS, selected):
        matches = [
            row
            for row in scenarios
            if row.get("status") == "feasible"
            and int(row.get("hidden_width", -1)) == width
            and int(row.get("removed_sublayers", -1)) == depth
        ]
        if len(matches) != 1:
            raise ValueError(
                "no feasible MIP scenario with a unique match for "
                f"width={width}, depth={depth}; found {len(matches)}"
            )
        scenario = copy.deepcopy(matches[0])
        checkpoint = _checkpoint(
            scenario, label=f"scenario width={width}, depth={depth}"
        )
        solutions.append(
            {
                "solution_id": canonical_solution_id(width, depth),
                "label": f"H={width}, Drop={depth}",
                "hidden_width": width,
                "removed_sublayers": depth,
                "checkpoint": checkpoint,
                "color": color,
                "marker": "circle",
                "always_enabled": False,
                "parameter_count": int(scenario["parameter_count"]),
                "parameter_ratio": float(scenario["parameter_ratio"]),
                "score": scenario.get("score"),
                "total_costs": _costs(scenario),
                "solution_path": str(scenario["solution_path"]),
                "forced_removals": list(scenario.get("forced_removals") or []),
                "source_identity": {
                    "profile_id": profile_id,
                    "hidden_width": width,
                    "removed_sublayers": depth,
                    "solution_path": str(scenario["solution_path"]),
                },
            }
        )

    return {
        "version": 1,
        "profile_id": profile_id,
        "profile": profile,
        "solutions": solutions,
    }


def write_solution_registry(path: str | Path, registry: dict[str, Any]) -> Path:
    """Atomically persist a registry so partial writes cannot be resumed."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(registry, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)
    return path
