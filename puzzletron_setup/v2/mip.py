# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Guided authoring model for complete Puzzletron MIP runs."""

from __future__ import annotations

import math
from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Optional, Sequence

if TYPE_CHECKING:
    from modelopt.torch.puzzletron.mip.profiles import MIPProfile

__all__ = [
    "ConcreteExpansion",
    "ConstraintDraft",
    "MIPDomains",
    "MIPRunDraft",
    "MIPRunEditor",
    "VariantDraft",
]


@dataclass(frozen=True)
class MIPDomains:
    """Measured search domains available to the MIP compiler."""

    depths: tuple[int, ...]
    embeddings: tuple[int, ...]
    depth_counts: Mapping[str, int] = field(default_factory=dict)
    depth_granularity: str = "subblock"
    axes: Mapping[str, tuple[Any, ...]] = field(default_factory=dict)


@dataclass(frozen=True)
class ConstraintDraft:
    """One friendly or raw-stat constraint with an optional workload."""

    metric: str
    mode: str
    value: Any
    workload: Optional[str] = None

    def to_config(self) -> Any:
        if self.mode in {"directional", "auto", "scalar"}:
            bound: Any = deepcopy(self.value)
        elif self.mode in {"min", "max", "eq", "range"}:
            bound = {self.mode: deepcopy(self.value)}
        else:
            raise ValueError(f"unsupported MIP constraint mode {self.mode!r}")
        return {"at": {self.workload: bound}} if self.workload else bound


@dataclass(frozen=True)
class VariantDraft:
    """Overrides and explicit matrix rows inherited from one run."""

    variant_id: str
    constraints: tuple[ConstraintDraft, ...] = ()
    objectives: Any = None
    search_space: Optional[Mapping[str, Any]] = None
    solver: Optional[Mapping[str, Any]] = None
    homogeneous: Optional[Mapping[str, Any]] = None
    matrix: Mapping[str, Sequence[Any]] = field(default_factory=dict)

    def to_config(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if self.constraints:
            payload["constraints"] = {
                item.metric: item.to_config() for item in self.constraints
            }
        for key in ("objectives", "search_space", "solver", "homogeneous"):
            value = getattr(self, key)
            if value is not None:
                payload[key] = deepcopy(value)
        if self.matrix:
            payload["matrix"] = {
                str(path): list(values) for path, values in self.matrix.items()
            }
        return payload


@dataclass(frozen=True)
class MIPRunDraft:
    """One independent MIP goal with inherited variants."""

    run_id: str
    goal: ConstraintDraft
    objectives: Any
    constraints: tuple[ConstraintDraft, ...] = ()
    search_space: Mapping[str, Any] = field(
        default_factory=lambda: {"depth": "all", "embedding": "all"}
    )
    solver: Mapping[str, Any] = field(
        default_factory=lambda: {
            "backend": "auto",
            "num_solutions": 1000,
            "min_hamming_distance": 2,
            "max_seconds_per_solution": 60,
        }
    )
    homogeneous: Mapping[str, Any] = field(
        default_factory=lambda: {"enabled": True, "keep": "all", "rank_by": "objective"}
    )
    variants: tuple[VariantDraft, ...] = ()

    def to_config(self) -> dict[str, Any]:
        constraints = {self.goal.metric: self.goal.to_config()}
        constraints.update(
            {item.metric: item.to_config() for item in self.constraints}
        )
        payload: dict[str, Any] = {
            "objectives": deepcopy(self.objectives),
            "constraints": constraints,
            "search_space": deepcopy(dict(self.search_space)),
            "solver": deepcopy(dict(self.solver)),
            "homogeneous": deepcopy(dict(self.homogeneous)),
        }
        if self.variants:
            payload["variants"] = OrderedDict(
                (variant.variant_id, variant.to_config()) for variant in self.variants
            )
        return payload


@dataclass(frozen=True)
class ConcreteExpansion:
    variants: int
    matrix_rows: int
    objectives: int
    concrete_solves: int


def _objective_count(value: Any) -> int:
    if isinstance(value, list):
        return max(1, len(value))
    return 1


def _matrix_rows(matrix: Mapping[str, Sequence[Any]]) -> int:
    return math.prod(max(1, len(values)) for values in matrix.values())


class MIPRunEditor:
    """Ordered CRUD editor backed by the canonical MIP compiler."""

    def __init__(
        self,
        domains: MIPDomains,
        *,
        workloads: Optional[Mapping[str, Mapping[str, Any]]] = None,
        defaults: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.domains = domains
        self.workloads = OrderedDict(
            (str(name), deepcopy(dict(value))) for name, value in (workloads or {}).items()
        )
        self.defaults = deepcopy(dict(defaults or {}))
        self._runs: OrderedDict[str, MIPRunDraft] = OrderedDict()

    def runs(self) -> Mapping[str, MIPRunDraft]:
        return OrderedDict(self._runs)

    def add_run(self, run: MIPRunDraft) -> None:
        if run.run_id in self._runs:
            raise ValueError(f"duplicate MIP run {run.run_id!r}")
        self._runs[run.run_id] = run

    def edit_run(self, run_id: str, **changes: Any) -> MIPRunDraft:
        updated = replace(self._runs[run_id], **changes)
        if updated.run_id != run_id and updated.run_id in self._runs:
            raise ValueError(f"duplicate MIP run {updated.run_id!r}")
        if updated.run_id == run_id:
            self._runs[run_id] = updated
        else:
            items = [
                (updated.run_id if name == run_id else name, updated if name == run_id else value)
                for name, value in self._runs.items()
            ]
            self._runs = OrderedDict(items)
        return updated

    def clone_run(self, source_id: str, target_id: str) -> MIPRunDraft:
        clone = replace(deepcopy(self._runs[source_id]), run_id=target_id)
        self.add_run(clone)
        return clone

    def delete_run(self, run_id: str, referenced_by: Sequence[str] = ()) -> tuple[str, ...]:
        if referenced_by:
            return tuple(sorted(referenced_by))
        del self._runs[run_id]
        return ()

    def add_variant(self, run_id: str, variant: VariantDraft) -> MIPRunDraft:
        run = self._runs[run_id]
        if any(item.variant_id == variant.variant_id for item in run.variants):
            raise ValueError(f"duplicate MIP variant {variant.variant_id!r}")
        return self.edit_run(run_id, variants=(*run.variants, variant))

    def clone_variant(
        self, run_id: str, source_id: str, target_id: str
    ) -> VariantDraft:
        run = self._runs[run_id]
        source = next(item for item in run.variants if item.variant_id == source_id)
        clone = replace(deepcopy(source), variant_id=target_id)
        self.add_variant(run_id, clone)
        return clone

    def review_variant(self, run_id: str, variant_id: str) -> Mapping[str, Any]:
        run = self._runs[run_id]
        variant = next(item for item in run.variants if item.variant_id == variant_id)
        return {
            "inherited": run.to_config(),
            "overrides": variant.to_config(),
        }

    def expansion(self, run_id: str) -> ConcreteExpansion:
        run = self._runs[run_id]
        variants = run.variants or (VariantDraft("default"),)
        objective_count = _objective_count(run.objectives)
        rows = 0
        solves = 0
        for variant in variants:
            variant_rows = _matrix_rows(variant.matrix)
            variant_objectives = _objective_count(
                run.objectives if variant.objectives is None else variant.objectives
            )
            rows += variant_rows
            solves += variant_rows * variant_objectives
        return ConcreteExpansion(
            variants=len(variants),
            matrix_rows=rows,
            objectives=objective_count,
            concrete_solves=solves,
        )

    def to_config(self) -> dict[str, Any]:
        return {
            "defaults": deepcopy(self.defaults),
            "workloads": deepcopy(self.workloads),
            "runs": OrderedDict(
                (run_id, run.to_config()) for run_id, run in self._runs.items()
            ),
        }

    def validate(
        self,
        available_depths: Optional[Iterable[int]] = None,
        available_embeddings: Optional[Iterable[int]] = None,
        available_depth_counts: Optional[Mapping[str, int]] = None,
        depth_granularity: Optional[str] = None,
    ) -> tuple["MIPProfile", ...]:
        # Keep the setup package usable without PyTorch until canonical validation.
        from modelopt.torch.puzzletron.mip.profiles import normalize_mip_profiles

        return normalize_mip_profiles(
            self.to_config(),
            available_depths=(
                self.domains.depths if available_depths is None else available_depths
            ),
            available_embeddings=(
                self.domains.embeddings
                if available_embeddings is None
                else available_embeddings
            ),
            available_depth_counts=(
                self.domains.depth_counts
                if available_depth_counts is None
                else available_depth_counts
            ),
            depth_granularity=depth_granularity or self.domains.depth_granularity,
        )
