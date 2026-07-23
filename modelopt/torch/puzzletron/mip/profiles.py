# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Normalize named MIP profiles and compile their aggregate constraints."""

from __future__ import annotations

import re
from copy import deepcopy
from dataclasses import dataclass
from itertools import product
from typing import Any, Iterable, Mapping

__all__ = [
    "BoundValue",
    "DepthSelection",
    "HomogeneousPolicy",
    "MIPProfile",
    "ObjectiveSpec",
    "ProfileConstraint",
    "SolverOptions",
    "compile_profile_constraints",
    "normalize_mip_profiles",
]


@dataclass(frozen=True)
class BoundValue:
    """One absolute value or teacher-relative ratio."""

    value: float
    relative: bool = False


@dataclass(frozen=True)
class ProfileConstraint:
    """One normalized aggregate constraint, optionally bound to a workload."""

    metric: str
    stat_name: str
    workload: str | None
    minimum: BoundValue | None
    maximum: BoundValue | None


@dataclass(frozen=True)
class DepthSelection:
    """One total-prefix or typed-prefix depth scenario."""

    counts: tuple[tuple[str, int], ...]

    @classmethod
    def total_prefix(cls, count: int) -> DepthSelection:
        return cls((("total", int(count)),))

    @property
    def total(self) -> int:
        values = self.as_dict()
        return values["total"] if "total" in values else sum(values.values())

    @property
    def slug(self) -> str:
        values = self.as_dict()
        if "total" in values:
            return f"depth-{values['total']:02d}"
        suffix = "_".join(f"{kind}-{count:02d}" for kind, count in self.counts)
        return f"depth-{suffix}"

    def as_dict(self) -> dict[str, int]:
        return dict(self.counts)


@dataclass(frozen=True)
class ObjectiveSpec:
    """One independent additive objective solve."""

    metric: str
    direction: str

    @property
    def bigger_is_better(self) -> bool:
        return self.direction == "maximize"


@dataclass(frozen=True)
class SolverOptions:
    """Backend-neutral MIP solution-pool controls."""

    backend: str = "pulp"
    num_solutions: int = 1
    min_hamming_distance: int = 1
    max_seconds_per_solution: float | None = 60.0


@dataclass(frozen=True)
class HomogeneousPolicy:
    """Retention and ranking policy for homogeneous Cartesian candidates."""

    enabled: bool = False
    keep: int = -1
    rank_by: str = "objective"
    constraint_weights: tuple[tuple[str, float], ...] = ()

    @property
    def num_solutions(self) -> int:
        return self.keep if self.enabled else 0


@dataclass(frozen=True)
class MIPProfile:
    """One concrete run/variant/matrix/objective solve."""

    profile_id: str
    run_id: str
    variant_id: str
    objective: ObjectiveSpec
    solver: SolverOptions
    homogeneous: HomogeneousPolicy
    constraints: tuple[ProfileConstraint, ...]
    workloads: dict[str, dict[str, Any]]
    depths: tuple[int, ...]
    depth_selections: tuple[DepthSelection, ...]
    embedding_widths: tuple[int, ...]
    axes_default: str
    axis_options: dict[str, Any]

    @property
    def base_profile_id(self) -> str:
        """Compatibility label used by existing MIP artifact reports."""

        return self.run_id

    @property
    def num_homogeneous_solutions(self) -> int:
        return self.homogeneous.num_solutions

    @property
    def required_workloads(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                constraint.workload
                for constraint in self.constraints
                if constraint.workload is not None
            )
        )


@dataclass(frozen=True)
class _MetricSpec:
    stat_name: str
    direction: str
    unit_kind: str
    workload_dependent: bool = False


_METRICS = {
    "params": _MetricSpec("num_params", "max", "count"),
    "active_params": _MetricSpec("active_params", "max", "count"),
    "memory": _MetricSpec("memory_mib", "max", "memory", True),
    "runtime": _MetricSpec("runtime_ms", "max", "time", True),
    "prefill_runtime": _MetricSpec("prefill_runtime_ms", "max", "time", True),
    "throughput": _MetricSpec("runtime_ms", "min", "throughput", True),
    "kv_heads": _MetricSpec("num_kv_heads", "max", "count"),
    "experts": _MetricSpec("num_experts", "max", "count"),
}

_COUNT_SUFFIXES = {"K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}
_MEMORY_SUFFIXES = {"MIB": 1.0, "GIB": 1024.0, "TIB": 1024.0**2}
_TIME_SUFFIXES = {"MS": 1.0, "S": 1000.0}


def _normalize_workloads(raw: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    workloads = {}
    aliases = {
        "isl": "prefill_seq_len",
        "osl": "generation_seq_len",
        "concurrency": "max_num_seqs",
    }
    for name, value in dict(raw or {}).items():
        if not isinstance(value, Mapping):
            raise TypeError(f"workload {name!r} must be a mapping")
        normalized = {}
        for key, item in value.items():
            normalized[aliases.get(str(key), str(key))] = item
        workloads[str(name)] = normalized
    return workloads


def _slug(value: Any) -> str:
    text = str(value).strip().lower().replace("%", "pct")
    text = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    return text or "value"


def _absolute_value(metric: str, value: Any, unit_kind: str) -> BoundValue:
    if isinstance(value, bool):
        raise TypeError(f"{metric} constraint cannot be boolean")
    if isinstance(value, (int, float)):
        return BoundValue(float(value))
    if not isinstance(value, str):
        raise TypeError(f"{metric} constraint must be a number or string, got {type(value)!r}")
    text = value.strip()
    if text.endswith("%"):
        ratio = float(text[:-1]) / 100.0
        if ratio <= 0:
            raise ValueError(f"{metric} percentage must be positive, got {value!r}")
        return BoundValue(ratio, relative=True)
    match = re.fullmatch(r"([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*([A-Za-z/]+)?", text)
    if match is None:
        raise ValueError(f"invalid {metric} constraint value {value!r}")
    number = float(match.group(1))
    suffix = (match.group(2) or "").upper()
    if not suffix:
        return BoundValue(number)
    suffixes = {
        "count": _COUNT_SUFFIXES,
        "memory": _MEMORY_SUFFIXES,
        "time": _TIME_SUFFIXES,
        "throughput": {"TOKENS/S": 1.0, "TOKEN/S": 1.0},
    }[unit_kind]
    if suffix not in suffixes:
        raise ValueError(f"unsupported {metric} unit {match.group(2)!r}")
    return BoundValue(number * suffixes[suffix])


def _bound_values(
    metric: str, raw: Any, spec: _MetricSpec
) -> tuple[BoundValue | None, BoundValue | None]:
    if not isinstance(raw, Mapping):
        value = _absolute_value(metric, raw, spec.unit_kind)
        return (value, None) if spec.direction == "min" else (None, value)
    unknown = set(raw) - {"min", "max", "eq", "range"}
    if unknown:
        raise ValueError(f"unknown {metric} constraint fields: {sorted(unknown)}")
    if "eq" in raw:
        if len(raw) != 1:
            raise ValueError(f"{metric} eq cannot be combined with other bounds")
        value = _absolute_value(metric, raw["eq"], spec.unit_kind)
        return value, value
    if "range" in raw:
        if len(raw) != 1 or not isinstance(raw["range"], (list, tuple)) or len(raw["range"]) != 2:
            raise ValueError(f"{metric} range must contain exactly [min, max]")
        lower, upper = raw["range"]
    else:
        lower, upper = raw.get("min"), raw.get("max")
    if lower is None and upper is None:
        raise ValueError(f"{metric} constraint must define min, max, eq, or range")
    return (
        _absolute_value(metric, lower, spec.unit_kind) if lower is not None else None,
        _absolute_value(metric, upper, spec.unit_kind) if upper is not None else None,
    )


def _metric_spec(metric: str) -> _MetricSpec:
    if metric in _METRICS:
        return _METRICS[metric]
    if metric.startswith("stats."):
        return _MetricSpec(metric.removeprefix("stats."), "max", "count")
    raise ValueError(f"unknown MIP constraint {metric!r}")


def _normalize_constraints(
    raw: Mapping[str, Any], workloads: Mapping[str, dict[str, Any]]
) -> tuple[ProfileConstraint, ...]:
    constraints = []
    for metric, value in dict(raw or {}).items():
        metric = str(metric)
        spec = _metric_spec(metric)
        if isinstance(value, Mapping) and "at" in value:
            if set(value) != {"at"}:
                raise ValueError(f"{metric}.at cannot be combined with profile-level bounds")
            if not spec.workload_dependent:
                raise ValueError(f"{metric} does not accept workload-specific constraints")
            entries = value["at"]
            if not isinstance(entries, Mapping) or not entries:
                raise ValueError(f"{metric}.at must be a non-empty workload mapping")
            for workload, bound in entries.items():
                workload = str(workload)
                if workload not in workloads:
                    raise ValueError(f"{metric} references unknown workload {workload!r}")
                minimum, maximum = _bound_values(metric, bound, spec)
                constraints.append(
                    ProfileConstraint(metric, spec.stat_name, workload, minimum, maximum)
                )
            continue
        if spec.workload_dependent:
            raise ValueError(f"{metric} must select a named workload with at")
        minimum, maximum = _bound_values(metric, value, spec)
        constraints.append(ProfileConstraint(metric, spec.stat_name, None, minimum, maximum))
    return tuple(constraints)


def _selected_values(raw: Any, available: tuple[int, ...], label: str) -> tuple[int, ...]:
    if raw is None or raw == "all":
        return available
    if isinstance(raw, Mapping):
        if set(raw) != {"range"} or len(raw["range"]) != 2:
            raise ValueError(f"{label} selector range must contain [min, max]")
        lower, upper = (int(value) for value in raw["range"])
        selected = tuple(value for value in available if lower <= value <= upper)
    elif isinstance(raw, str) and ".." in raw:
        lower, upper = (int(value.strip()) for value in raw.split("..", 1))
        selected = tuple(value for value in available if lower <= value <= upper)
    elif isinstance(raw, Iterable) and not isinstance(raw, (str, bytes, Mapping)):
        selected = tuple(int(value) for value in raw)
    else:
        selected = (int(raw),)
    missing = sorted(set(selected) - set(available))
    if missing:
        raise ValueError(f"{label} options are unavailable: {missing}")
    if not selected:
        raise ValueError(f"{label} selector matched no available values")
    return selected


def _homogeneous_count(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < -1:
        raise ValueError("num_homogeneous_solutions must be -1 or a non-negative integer")
    return value


def _objective_specs(raw: Any) -> tuple[ObjectiveSpec, ...]:
    if isinstance(raw, (str, Mapping)):
        raw = [raw]
    if not isinstance(raw, (list, tuple)) or not raw:
        raise ValueError("each MIP run must define one or more objectives")
    objectives = []
    for entry in raw:
        if isinstance(entry, str):
            metric, direction = entry, "minimize"
        elif isinstance(entry, Mapping):
            unknown = set(entry) - {"metric", "direction"}
            if unknown or not entry.get("metric"):
                raise ValueError(f"invalid MIP objective fields: {sorted(unknown)}")
            metric = str(entry["metric"])
            direction = str(entry.get("direction", "minimize")).lower()
        else:
            raise TypeError("MIP objectives must be strings or mappings")
        if direction not in {"minimize", "maximize"}:
            raise ValueError("objective direction must be minimize or maximize")
        objectives.append(ObjectiveSpec(str(metric), direction))
    if len({(item.metric, item.direction) for item in objectives}) != len(objectives):
        raise ValueError("duplicate MIP objectives are not allowed")
    return tuple(objectives)


def _solver_options(raw: Mapping[str, Any] | None) -> SolverOptions:
    raw = dict(raw or {})
    unknown = set(raw) - {
        "backend",
        "num_solutions",
        "min_hamming_distance",
        "max_seconds_per_solution",
    }
    if unknown:
        raise ValueError(f"unknown common MIP solver options: {sorted(unknown)}")
    backend = str(raw.get("backend", "pulp")).lower()
    if backend not in {"pulp", "cbc", "cuopt", "auto"}:
        raise ValueError(f"unsupported MIP solver backend {backend!r}")
    num_solutions = int(raw.get("num_solutions", 1))
    min_hamming = int(raw.get("min_hamming_distance", 1))
    timeout = raw.get("max_seconds_per_solution", 60.0)
    if num_solutions < 1:
        raise ValueError("solver.num_solutions must be positive")
    if min_hamming < 1:
        raise ValueError("solver.min_hamming_distance must be positive")
    if timeout is not None and float(timeout) <= 0:
        raise ValueError("solver.max_seconds_per_solution must be positive or null")
    return SolverOptions(
        backend=backend,
        num_solutions=num_solutions,
        min_hamming_distance=min_hamming,
        max_seconds_per_solution=float(timeout) if timeout is not None else None,
    )


def _homogeneous_policy(raw: Mapping[str, Any] | None) -> HomogeneousPolicy:
    raw = dict(raw or {})
    unknown = set(raw) - {"enabled", "keep", "rank_by"}
    if unknown:
        raise ValueError(f"unknown homogeneous policy fields: {sorted(unknown)}")
    enabled = bool(raw.get("enabled", False))
    keep_raw = raw.get("keep", "all")
    keep = -1 if keep_raw == "all" else _homogeneous_count(keep_raw)
    rank_raw = raw.get("rank_by", "objective")
    weights: dict[str, float] = {}
    if isinstance(rank_raw, Mapping):
        if set(rank_raw) != {"constraint_closeness"}:
            raise ValueError("homogeneous.rank_by mapping must contain constraint_closeness")
        closeness = rank_raw["constraint_closeness"] or {}
        if not isinstance(closeness, Mapping):
            raise TypeError("constraint_closeness must be a mapping")
        unknown_closeness = set(closeness) - {"weights"}
        if unknown_closeness:
            raise ValueError(
                f"unknown constraint_closeness fields: {sorted(unknown_closeness)}"
            )
        weights = {
            str(key): float(value)
            for key, value in dict(closeness.get("weights") or {}).items()
        }
        if any(value <= 0 for value in weights.values()):
            raise ValueError("constraint-closeness weights must be positive")
        rank_by = "constraint_closeness"
    else:
        rank_by = str(rank_raw)
    if rank_by not in {"objective", "constraint_closeness"}:
        raise ValueError("homogeneous.rank_by must be objective or constraint_closeness")
    if enabled and keep == 0:
        raise ValueError("enabled homogeneous search cannot keep zero solutions")
    return HomogeneousPolicy(enabled, keep, rank_by, tuple(sorted(weights.items())))


def _merge_mapping(
    base: Mapping[str, Any] | None, update: Mapping[str, Any] | None
) -> dict[str, Any]:
    merged = deepcopy(dict(base or {}))
    for key, value in dict(update or {}).items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _merge_mapping(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _matrix_rows(raw: Mapping[str, Any] | None) -> tuple[dict[str, Any], ...]:
    matrix = dict(raw or {})
    if not matrix:
        return ({},)
    supported = {"embedding", "depth"}
    invalid = sorted(
        key
        for key in matrix
        if key not in supported
        and not key.startswith("constraints.")
        and not key.startswith("solver.")
        and not key.startswith("homogeneous.")
    )
    if invalid:
        raise ValueError(f"unsupported MIP matrix paths: {invalid}")
    keys = tuple(matrix)
    domains = []
    for key in keys:
        values = matrix[key]
        if not isinstance(values, (list, tuple)) or not values:
            raise ValueError(f"MIP matrix field {key!r} must be a non-empty list")
        domains.append(tuple(values))
    return tuple(dict(zip(keys, values)) for values in product(*domains))


def _apply_matrix_path(config: dict[str, Any], path: str, value: Any) -> None:
    target = config
    keys = path.split(".")
    for key in keys[:-1]:
        child = target.setdefault(key, {})
        if not isinstance(child, dict):
            raise ValueError(f"MIP matrix path crosses a scalar: {path}")
        target = child
    target[keys[-1]] = deepcopy(value)


def _depth_selections(
    raw: Any,
    available_depths: tuple[int, ...],
    *,
    available_depth_counts: Mapping[str, int] | None,
    depth_granularity: str,
) -> tuple[DepthSelection, ...]:
    if not isinstance(raw, Mapping) or set(raw) == {"range"}:
        return tuple(
            DepthSelection.total_prefix(value)
            for value in _selected_values(raw, available_depths, "depth")
        )
    if "total" in raw:
        if set(raw) != {"total"}:
            raise ValueError("depth.total cannot be combined with typed depth keys")
        return tuple(
            DepthSelection.total_prefix(value)
            for value in _selected_values(raw["total"], available_depths, "depth.total")
        )
    if not raw:
        raise ValueError("typed depth selector must contain at least one sublayer kind")
    if str(depth_granularity).lower() != "subblock":
        raise ValueError("typed depth selectors require subblock depth granularity")
    available_by_kind = {
        str(kind): int(count) for kind, count in dict(available_depth_counts or {}).items()
    }
    unknown = sorted(set(str(kind) for kind in raw) - set(available_by_kind))
    if unknown:
        raise ValueError(f"unknown typed depth kinds: {unknown}")

    kinds = tuple(str(kind) for kind in raw)
    domains = [
        _selected_values(
            raw[kind],
            tuple(range(available_by_kind[kind] + 1)),
            f"depth.{kind}",
        )
        for kind in kinds
    ]
    return tuple(
        DepthSelection(tuple(sorted(zip(kinds, counts))))
        for counts in product(*domains)
    )


def normalize_mip_profiles(
    mip_cfg: Mapping[str, Any],
    *,
    available_depths: Iterable[int],
    available_embeddings: Iterable[int],
    available_depth_counts: Mapping[str, int] | None = None,
    depth_granularity: str = "subblock",
) -> tuple[MIPProfile, ...]:
    """Validate and compile public ``mip.runs`` into concrete solve specs."""

    workloads = _normalize_workloads(mip_cfg.get("workloads") or {})
    legacy = {
        key: mip_cfg.get(key)
        for key in ("profiles", "constraint_profiles", "latency_constraint_profiles")
        if mip_cfg.get(key)
    }
    if legacy:
        raise ValueError(
            "legacy MIP profile fields are no longer supported; use mip.runs: "
            + ", ".join(sorted(legacy))
        )
    raw_runs = mip_cfg.get("runs") or {}
    if not isinstance(raw_runs, Mapping):
        raise TypeError("mip.runs must be a mapping of independent run names")
    depths_available = tuple(int(value) for value in available_depths)
    embeddings_available = tuple(int(value) for value in available_embeddings)
    defaults = dict(mip_cfg.get("defaults") or {})
    unknown_defaults = set(defaults) - {"objectives", "solver", "homogeneous"}
    if unknown_defaults:
        raise ValueError(f"unknown mip.defaults fields: {sorted(unknown_defaults)}")
    profiles = []
    for run_id, run_value in raw_runs.items():
        if run_value is False:
            continue
        if not isinstance(run_value, Mapping):
            raise TypeError(f"MIP run {run_id!r} must be a mapping")
        run = dict(run_value)
        unknown_run = set(run) - {
            "constraints",
            "objectives",
            "search_space",
            "solver",
            "homogeneous",
            "variants",
        }
        if unknown_run:
            raise ValueError(f"unknown fields in MIP run {run_id!r}: {sorted(unknown_run)}")
        implicit_variant = "variants" not in run
        raw_variants = run.get("variants") or {"default": {}}
        if not isinstance(raw_variants, Mapping):
            raise TypeError(f"MIP run {run_id!r} variants must be a mapping")
        for variant_id, variant_value in raw_variants.items():
            if not isinstance(variant_value, Mapping):
                raise TypeError(f"MIP variant {run_id}.{variant_id} must be a mapping")
            variant = dict(variant_value)
            unknown_variant = set(variant) - {
                "constraints",
                "objectives",
                "search_space",
                "solver",
                "homogeneous",
                "matrix",
            }
            if unknown_variant:
                raise ValueError(
                    f"unknown fields in MIP variant {run_id}.{variant_id}: "
                    f"{sorted(unknown_variant)}"
                )
            for matrix in _matrix_rows(variant.get("matrix")):
                concrete = {
                    "constraints": _merge_mapping(
                        run.get("constraints"), variant.get("constraints")
                    ),
                    "search_space": _merge_mapping(
                        run.get("search_space"), variant.get("search_space")
                    ),
                    "solver": _merge_mapping(
                        defaults.get("solver"),
                        _merge_mapping(run.get("solver"), variant.get("solver")),
                    ),
                    "homogeneous": _merge_mapping(
                        defaults.get("homogeneous"),
                        _merge_mapping(run.get("homogeneous"), variant.get("homogeneous")),
                    ),
                }
                for path, value in matrix.items():
                    if path in {"embedding", "depth"}:
                        concrete["search_space"][path] = value
                    else:
                        _apply_matrix_path(concrete, path, value)
                search = concrete["search_space"]
                axes_default = search.get("axes_default", "all")
                if axes_default is None:
                    axes_default = "teacher"
                if axes_default not in {"all", "teacher"}:
                    raise ValueError("axes_default must be all or teacher")
                axes = search.get("axes") or {}
                if not isinstance(axes, Mapping):
                    raise TypeError("search_space.axes must be a mapping")
                depth_selections = _depth_selections(
                    search.get("depth"),
                    depths_available,
                    available_depth_counts=available_depth_counts,
                    depth_granularity=depth_granularity,
                )
                embeddings = _selected_values(
                    search.get("embedding"), embeddings_available, "embedding"
                )
                objectives = _objective_specs(
                    variant.get(
                        "objectives",
                        run.get("objectives", defaults.get("objectives")),
                    )
                )
                matrix_suffix = tuple(
                    f"{_slug(key)}-{_slug(value)}" for key, value in matrix.items()
                )
                constraints = _normalize_constraints(concrete["constraints"], workloads)
                homogeneous = _homogeneous_policy(concrete["homogeneous"])
                constraint_metrics = {constraint.metric for constraint in constraints}
                unknown_weights = set(dict(homogeneous.constraint_weights)) - constraint_metrics
                if unknown_weights:
                    raise ValueError(
                        "homogeneous constraint-closeness weights name unknown constraints: "
                        f"{sorted(unknown_weights)}"
                    )
                for objective in objectives:
                    profile_parts = [str(run_id)]
                    if not implicit_variant:
                        profile_parts.append(str(variant_id))
                    profile_parts.extend(matrix_suffix)
                    if len(objectives) > 1:
                        profile_parts.append(
                            f"objective-{_slug(objective.metric)}-{objective.direction}"
                        )
                    profiles.append(
                        MIPProfile(
                            profile_id="--".join(profile_parts),
                            run_id=str(run_id),
                            variant_id=str(variant_id),
                            objective=objective,
                            solver=_solver_options(concrete["solver"]),
                            homogeneous=homogeneous,
                            constraints=constraints,
                            workloads=deepcopy(workloads),
                            depths=tuple(selection.total for selection in depth_selections),
                            depth_selections=depth_selections,
                            embedding_widths=embeddings,
                            axes_default=str(axes_default),
                            axis_options={str(key): deepcopy(value) for key, value in axes.items()},
                        )
                    )
    ids = [profile.profile_id for profile in profiles]
    if len(ids) != len(set(ids)):
        raise ValueError("MIP run expansion produced duplicate concrete solve IDs")
    return tuple(profiles)


def _resolve_bound(
    bound: BoundValue | None,
    *,
    teacher_value: float | int | None,
    label: str,
) -> float | None:
    if bound is None:
        return None
    if not bound.relative:
        return bound.value
    if teacher_value is None:
        raise ValueError(f"cannot resolve {label}: teacher statistic is missing")
    return float(teacher_value) * bound.value


def _throughput_runtime_bound(
    bound: BoundValue | None,
    *,
    teacher_runtime_ms: float | int | None,
    workload: Mapping[str, Any],
) -> float | None:
    if bound is None:
        return None
    if bound.relative:
        if teacher_runtime_ms is None:
            raise ValueError("cannot resolve throughput percentage: teacher runtime is missing")
        return float(teacher_runtime_ms) / bound.value
    batch_size = int(workload.get("batch_size", workload.get("max_num_seqs", 1)))
    output_length = int(workload.get("generation_seq_len", 0))
    if bound.value <= 0 or output_length <= 0:
        raise ValueError("throughput constraints require positive throughput and output length")
    return 1000.0 * batch_size * output_length / bound.value


def compile_profile_constraints(
    profile: MIPProfile,
    *,
    teacher_totals: Mapping[str | None, Mapping[str, float]],
) -> dict[str, float | tuple[float | None, float | None]]:
    """Resolve percentages and produce direct additive MIP constraints."""

    compiled = {}
    for constraint in profile.constraints:
        teacher = teacher_totals.get(constraint.workload)
        if teacher is None:
            raise ValueError(
                f"teacher totals are missing for workload {constraint.workload!r}"
            )
        workload_suffix = f"@{constraint.workload}" if constraint.workload else ""
        key = f"stats.{constraint.stat_name}{workload_suffix}"
        if constraint.metric == "throughput":
            workload = profile.workloads[constraint.workload]
            teacher_runtime = teacher.get("runtime_ms")
            # Throughput bounds invert when expressed as runtime bounds.
            minimum = _throughput_runtime_bound(
                constraint.maximum,
                teacher_runtime_ms=teacher_runtime,
                workload=workload,
            )
            maximum = _throughput_runtime_bound(
                constraint.minimum,
                teacher_runtime_ms=teacher_runtime,
                workload=workload,
            )
        else:
            teacher_value = teacher.get(constraint.stat_name)
            minimum = _resolve_bound(
                constraint.minimum,
                teacher_value=teacher_value,
                label=key,
            )
            maximum = _resolve_bound(
                constraint.maximum,
                teacher_value=teacher_value,
                label=key,
            )
        if minimum is None:
            assert maximum is not None
            compiled[key] = maximum
        else:
            compiled[key] = (minimum, maximum)
    return compiled
