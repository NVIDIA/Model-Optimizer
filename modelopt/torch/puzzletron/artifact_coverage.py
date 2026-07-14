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

"""Strict identity coverage checks for expensive Puzzletron campaign artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Hashable, Iterable, Mapping

__all__ = [
    "CoverageReport",
    "CoverageRow",
    "verify_campaign_artifacts",
    "verify_real_campaign_artifacts",
]


def _ordered(values: Iterable[Hashable]) -> tuple[Hashable, ...]:
    return tuple(sorted(set(values), key=repr))


@dataclass(frozen=True)
class CoverageRow:
    expected: tuple[Hashable, ...]
    present: tuple[Hashable, ...]
    missing: tuple[Hashable, ...]
    extra: tuple[Hashable, ...]

    @classmethod
    def compare(cls, expected: Iterable[Hashable], present: Iterable[Hashable]) -> "CoverageRow":
        expected_set = set(expected)
        present_set = set(present)
        return cls(
            expected=_ordered(expected_set),
            present=_ordered(present_set),
            missing=_ordered(expected_set - present_set),
            extra=_ordered(present_set - expected_set),
        )


@dataclass(frozen=True)
class CoverageReport:
    rows: Mapping[str, CoverageRow]
    identity_mismatches: Mapping[str, tuple[Any, Any]]
    bypass_enabled: bool

    @property
    def complete(self) -> bool:
        return (
            not self.bypass_enabled
            and not self.identity_mismatches
            and all(not row.missing for row in self.rows.values())
        )

    def require_complete(self) -> None:
        problems = [
            f"{name} missing={list(row.missing)!r}"
            for name, row in self.rows.items()
            if row.missing
        ]
        problems.extend(
            f"{name} identity expected={expected!r} observed={observed!r}"
            for name, (expected, observed) in self.identity_mismatches.items()
        )
        if self.bypass_enabled:
            problems.append("bypass must be disabled for this campaign")
        if problems:
            raise RuntimeError("incomplete Puzzletron artifacts: " + "; ".join(problems))


def verify_campaign_artifacts(
    *,
    expected_scores: Iterable[Hashable],
    present_scores: Iterable[Hashable],
    expected_runtimes: Iterable[Hashable],
    present_runtimes: Iterable[Hashable],
    expected_depths: Iterable[Hashable],
    present_depths: Iterable[Hashable],
    expected_identity: Mapping[str, Any],
    observed_identity: Mapping[str, Any],
    bypass_enabled: bool,
) -> CoverageReport:
    """Compare required campaign identities without mutating any artifact."""

    mismatches = {
        name: (expected, observed_identity.get(name))
        for name, expected in expected_identity.items()
        if observed_identity.get(name) != expected
    }
    return CoverageReport(
        rows={
            "scores": CoverageRow.compare(expected_scores, present_scores),
            "runtimes": CoverageRow.compare(expected_runtimes, present_runtimes),
            "depths": CoverageRow.compare(expected_depths, present_depths),
        },
        identity_mismatches=mismatches,
        bypass_enabled=bool(bypass_enabled),
    )


def _numeric_suffix(path: Path) -> int:
    try:
        return int(path.stem.rsplit("_", 1)[-1])
    except ValueError as error:
        raise ValueError(f"artifact has no numeric identity suffix: {path}") from error


def verify_real_campaign_artifacts(
    puzzle_dir: str | Path,
    *,
    expected_depth_scenarios: int,
    bypass_enabled: bool,
    expected_checkpoint_dir: str | Path | None = None,
    expected_data_identity: Mapping[str, Any] | None = None,
) -> CoverageReport:
    """Audit durable campaign identities immediately before no-bypass MIP."""

    root = Path(puzzle_dir)
    score_manifest_path = root / "subblock_replacement_manifest.json"
    if not score_manifest_path.is_file():
        raise FileNotFoundError(f"missing subblock score manifest: {score_manifest_path}")
    score_manifest = json.loads(score_manifest_path.read_text())
    expected_score_count = int(score_manifest["subblock_solution_count"])
    score_dir = root / "single_subblock_replacement_solutions--validation"
    present_scores = {_numeric_suffix(path) for path in score_dir.glob("solution_*.json")}

    shard_files = list((root / "runtime_cache" / "shards").glob("*/shard_*.json"))
    shard_identities = {_numeric_suffix(path) for path in shard_files}
    expected_runtime_count = max(shard_identities, default=-1) + 1
    present_runtimes = {
        _numeric_suffix(path) for path in (root / "runtime_cache" / "shards").glob("*/shard_*.done")
    }

    trajectory_path = root / "depth" / "iterative" / "trajectory.json"
    if not trajectory_path.is_file():
        raise FileNotFoundError(f"missing depth trajectory: {trajectory_path}")
    trajectory = json.loads(trajectory_path.read_text())
    scenarios = list(trajectory.get("scenarios") or [])

    campaign_manifest_path = root / "subblock_distributed_eval" / "campaign" / "manifest.json"
    campaign_manifest = (
        json.loads(campaign_manifest_path.read_text()) if campaign_manifest_path.is_file() else {}
    )
    expected_identity: dict[str, Any] = {}
    observed_identity: dict[str, Any] = {}
    if expected_checkpoint_dir is not None:
        expected_identity["checkpoint_dir"] = str(Path(expected_checkpoint_dir).resolve())
        observed_checkpoint = (campaign_manifest.get("model") or {}).get("checkpoint_dir")
        observed_identity["checkpoint_dir"] = (
            str(Path(observed_checkpoint).resolve()) if observed_checkpoint else None
        )
    if expected_data_identity is not None:
        expected_identity["data_identity"] = dict(expected_data_identity)
        observed_scoring = (campaign_manifest.get("data") or {}).get("scoring") or {}
        observed_identity["data_identity"] = {
            key: observed_scoring.get(key) for key in expected_data_identity
        }
    if "full_search_space_preserved" in score_manifest:
        expected_identity["full_search_space_preserved"] = True
        observed_identity["full_search_space_preserved"] = bool(
            score_manifest.get("full_search_space_preserved")
        )

    return verify_campaign_artifacts(
        expected_scores=range(expected_score_count),
        present_scores=present_scores,
        expected_runtimes=range(expected_runtime_count),
        present_runtimes=present_runtimes,
        expected_depths=range(int(expected_depth_scenarios)),
        present_depths=range(len(scenarios)),
        expected_identity=expected_identity,
        observed_identity=observed_identity,
        bypass_enabled=bypass_enabled,
    )
