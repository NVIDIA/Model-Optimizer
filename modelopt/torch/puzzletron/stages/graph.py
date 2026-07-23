# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Scheduler-neutral registry for public Puzzletron pipeline stages."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable

__all__ = [
    "STAGE_REGISTRY",
    "STAGE_SPECS",
    "ArtifactChoice",
    "StageSpec",
    "distributed_stage_ids",
    "enabled_stage_ids",
    "required_stage_ids",
    "selected_parent_stage_ids",
    "stage_display_name",
    "stage_ids",
    "stage_is_enabled",
    "stage_spec",
    "topological_mapping_items",
    "topological_stage_ids",
]


@dataclass(frozen=True)
class ArtifactChoice:
    """One artifact source a stage can select from an upstream stage."""

    artifact: str
    parent: str
    when: tuple[str, bool] | None = None
    requires: tuple[tuple[str, bool], ...] = ()
    fallback: bool = False


@dataclass(frozen=True)
class StageSpec:
    """Immutable scheduler-neutral metadata for one public Puzzletron stage."""

    stage_id: str
    display_name: str
    completion_artifacts: tuple[str, ...] = ()
    granularity_label: bool = False
    required: bool = False
    default_enabled: bool = False
    enabled_when: tuple[str, bool] | None = None
    parents: tuple[str, ...] = ()
    conditional_parents: tuple[tuple[str, tuple[str, bool]], ...] = ()
    artifact_choices: tuple[ArtifactChoice, ...] = ()
    distributed: bool = False
    report_order: int = 0
    topology_order: int = 0


def topological_mapping_items(
    nodes: Mapping[str, Any],
    dependency_ids: Callable[[str, Any], Iterable[str]],
) -> tuple[tuple[str, Any], ...]:
    """Return mapping items in dependency order, independent of key serialization order."""

    known = set(nodes)
    dependencies = {
        str(node_id): tuple(dict.fromkeys(str(value) for value in dependency_ids(node_id, value)))
        for node_id, value in nodes.items()
    }
    for node_id, node_dependencies in dependencies.items():
        unknown = set(node_dependencies) - known
        if unknown:
            raise ValueError(f"node {node_id!r} references unknown dependencies {sorted(unknown)}")

    pending = dict(nodes)
    completed: set[str] = set()
    ordered: list[tuple[str, Any]] = []
    while pending:
        ready = [
            (str(node_id), value)
            for node_id, value in pending.items()
            if set(dependencies[str(node_id)]) <= completed
        ]
        if not ready:
            raise ValueError(f"dependency cycle among nodes: {sorted(pending)}")
        for node_id, value in ready:
            ordered.append((node_id, value))
            completed.add(node_id)
            del pending[node_id]
    return tuple(ordered)


_SPECS: list[StageSpec] = []


def _stage(
    stage_id: str,
    display_name: str,
    *,
    required: bool = False,
    default_enabled: bool = False,
    enabled_when: tuple[str, bool] | None = None,
    parents: tuple[str, ...] = (),
    conditional_parents: tuple[tuple[str, tuple[str, bool]], ...] = (),
    artifact_choices: tuple[ArtifactChoice, ...] = (),
    completion_artifacts: tuple[str, ...] = (),
    granularity_label: bool = False,
    distributed: bool = False,
) -> None:
    order = len(_SPECS)
    _SPECS.append(
        StageSpec(
            stage_id=stage_id,
            display_name=display_name,
            completion_artifacts=completion_artifacts,
            granularity_label=granularity_label,
            required=required,
            default_enabled=default_enabled,
            enabled_when=enabled_when,
            parents=parents,
            conditional_parents=conditional_parents,
            artifact_choices=artifact_choices,
            distributed=distributed,
            report_order=order,
            topology_order=order,
        )
    )


_stage(
    "convert",
    "Convert Checkpoint",
    required=True,
    completion_artifacts=("ckpts/teacher/config.json",),
)
_stage(
    "tokenize_data",
    "Tokenize Data",
    required=True,
    parents=("convert",),
    completion_artifacts=("dataset_cache/*.tokens",),
)
_stage(
    "vllm_stats",
    "{unit} vLLM Stats",
    parents=("convert",),
    completion_artifacts=("artifacts/vllm_stats/summary.json",),
    granularity_label=True,
)
_stage(
    "depth_importance",
    "Depth Importance Estimation",
    parents=("tokenize_data",),
    completion_artifacts=("depth/iterative/trajectory.json",),
    distributed=True,
)
_stage(
    "width_importance",
    "Width Importance Estimation",
    required=True,
    parents=("tokenize_data",),
    completion_artifacts=(
        "pruning/pruning_scores/automodel/*/activation_passes_manifest.json",
    ),
    distributed=True,
)
_stage(
    "sort",
    "Sort Checkpoint",
    required=True,
    parents=("width_importance",),
    completion_artifacts=("ckpts/sorted_teacher/config.json",),
    distributed=True,
)
_stage(
    "sort_sanity",
    "Sort Sanity Check",
    parents=("sort",),
    completion_artifacts=("artifacts/sort_sanity/summary.json",),
    distributed=True,
)
_stage(
    "width_sanity",
    "Width Sanity Check",
    parents=("sort_sanity",),
    completion_artifacts=("artifacts/width_sanity/summary.json",),
    distributed=True,
)
_stage(
    "slicing_sanity",
    "Slicing Sanity Check",
    parents=("width_sanity",),
    completion_artifacts=("artifacts/slicing_sanity/summary.json",),
)
_stage(
    "bypass_sanity",
    "Bypass Sanity Check",
    parents=("sort",),
    completion_artifacts=("artifacts/bypass_sanity/summary.json",),
    distributed=True,
)
_stage(
    "bypass",
    "{unit} Bypass",
    parents=("bypass_sanity",),
    completion_artifacts=(
        "artifacts/bypass/dp_observations.jsonl",
        "artifacts/bypass/local_kd_loss_history.json",
    ),
    granularity_label=True,
    distributed=True,
)
_stage(
    "build_library",
    "Build Block Library",
    required=True,
    parents=("bypass",),
    conditional_parents=(("vllm_stats", ("vllm_stats.enabled", True)),),
    completion_artifacts=("replacement_library.json", "candidate_library.json"),
    distributed=True,
)
_stage(
    "replacement_scoring",
    "Replace-one-{unit_lower} Scoring",
    required=True,
    parents=("build_library",),
    completion_artifacts=("artifacts/replacement_scoring/summary.json",),
    granularity_label=True,
    distributed=True,
)
_stage(
    "mip",
    "MIP Search",
    required=True,
    parents=("vllm_stats", "depth_importance", "replacement_scoring"),
    completion_artifacts=("mip/profiles/*/mip_grid.json",),
)
_stage(
    "zero_shot_evaluation",
    "Zero-shot Evaluation",
    parents=("mip",),
    completion_artifacts=("artifacts/zero_shot_evaluation",),
    distributed=True,
)
_stage(
    "aiperf",
    "AIPerf",
    parents=("mip",),
    completion_artifacts=("artifacts/aiperf/**/aiperf_results.json",),
)
_stage(
    "global_distillation_sanity",
    "Global Distillation Sanity Check",
    parents=("mip",),
    completion_artifacts=(
        "artifacts/global_distillation_sanity/**/global_distillation_sanity_summary.json",
    ),
    distributed=True,
)
_stage(
    "global_distillation",
    "Global Distillation",
    parents=("global_distillation_sanity",),
    completion_artifacts=("artifacts/global_distillation/**/global_distillation_summary.json",),
    distributed=True,
)
_stage(
    "post_distillation_evaluation",
    "Post Distillation Evaluation",
    parents=("global_distillation",),
    completion_artifacts=("artifacts/post_distillation_evaluation/**/evaluation_summary.json",),
    distributed=True,
)

STAGE_SPECS = tuple(_SPECS)
STAGE_REGISTRY = MappingProxyType({spec.stage_id: spec for spec in STAGE_SPECS})


def _registry_for(specs: Iterable[StageSpec]) -> dict[str, StageSpec]:
    registry: dict[str, StageSpec] = {}
    for spec in specs:
        if spec.stage_id in registry:
            raise ValueError(f"Duplicate Puzzletron stage ID {spec.stage_id!r}")
        registry[spec.stage_id] = spec
    for spec in registry.values():
        for parent in (
            *spec.parents,
            *(parent for parent, _ in spec.conditional_parents),
            *(choice.parent for choice in spec.artifact_choices),
        ):
            if parent not in registry:
                raise ValueError(f"unknown parent {parent!r} for stage {spec.stage_id!r}")
    return registry


def _config_value(config: Mapping[str, Any], path: str, default: Any = False) -> Any:
    value: Any = config
    for key in path.split("."):
        if not isinstance(value, Mapping) or key not in value:
            return default
        value = value[key]
    return value


def stage_ids() -> tuple[str, ...]:
    """Return public stage IDs in deterministic registry order."""

    return tuple(STAGE_REGISTRY)


def stage_spec(stage_id: str) -> StageSpec:
    """Return the immutable specification for one public stage ID."""

    try:
        return STAGE_REGISTRY[stage_id]
    except KeyError as error:
        raise ValueError(f"Unknown Puzzletron stage {stage_id!r}") from error


def stage_display_name(stage_id: str, *, granularity: str | None = None) -> str:
    """Return the public stage label for its independently configured granularity."""

    spec = stage_spec(stage_id)
    if not spec.granularity_label:
        return spec.display_name
    unit = "Subblock" if granularity == "subblock" else "Block"
    return spec.display_name.format(unit=unit, unit_lower=unit.lower())


def required_stage_ids() -> tuple[str, ...]:
    """Return required public stages in deterministic registry order."""

    return tuple(spec.stage_id for spec in STAGE_SPECS if spec.required)


def distributed_stage_ids() -> tuple[str, ...]:
    """Return stages whose existing runner launches distributed workers."""

    return tuple(spec.stage_id for spec in STAGE_SPECS if spec.distributed)


def stage_is_enabled(stage_id: str, config: Mapping[str, Any]) -> bool:
    """Return whether a stage is enabled by its required/default/conditional metadata."""

    spec = stage_spec(stage_id)
    if spec.required:
        return True
    if spec.enabled_when is not None:
        path, expected = spec.enabled_when
        return _config_value(config, path) is expected
    section = config.get(stage_id)
    if not isinstance(section, Mapping):
        return spec.default_enabled
    return bool(section.get("enabled", spec.default_enabled))


def enabled_stage_ids(config: Mapping[str, Any]) -> tuple[str, ...]:
    """Return configured stages in deterministic topological order."""

    return tuple(stage_id for stage_id in topological_stage_ids() if stage_is_enabled(stage_id, config))


def _choice_matches(choice: ArtifactChoice, config: Mapping[str, Any]) -> bool:
    conditions = (*((choice.when,) if choice.when is not None else ()), *choice.requires)
    return all(_config_value(config, path) is expected for path, expected in conditions)


def selected_parent_stage_ids(stage_id: str, config: Mapping[str, Any]) -> tuple[str, ...]:
    """Return direct parents after selecting this stage's configured artifact inputs."""

    spec = stage_spec(stage_id)
    parents = list(spec.parents)
    selected = [
        choice
        for choice in spec.artifact_choices
        if not choice.fallback and _choice_matches(choice, config)
    ]
    if not selected:
        selected = [choice for choice in spec.artifact_choices if choice.fallback]
    parents.extend(choice.parent for choice in selected)
    parents.extend(
        parent
        for parent, (path, expected) in spec.conditional_parents
        if _config_value(config, path) is expected
    )
    return tuple(dict.fromkeys(parents))


def topological_stage_ids(specs: Iterable[StageSpec] = STAGE_SPECS) -> tuple[str, ...]:
    """Return a deterministic topological order or raise for invalid graph metadata."""

    registry = _registry_for(specs)
    parents = {
        stage_id: (
            set(spec.parents)
            | {parent for parent, _ in spec.conditional_parents}
            | {choice.parent for choice in spec.artifact_choices}
        )
        for stage_id, spec in registry.items()
    }
    order = {stage_id: spec.topology_order for stage_id, spec in registry.items()}
    result: list[str] = []
    while parents:
        ready = sorted((stage_id for stage_id, values in parents.items() if not values), key=order.__getitem__)
        if not ready:
            cycle = ", ".join(sorted(parents, key=order.__getitem__))
            raise ValueError(f"Stage graph contains a cycle: {cycle}")
        stage_id = ready[0]
        result.append(stage_id)
        parents.pop(stage_id)
        for values in parents.values():
            values.discard(stage_id)
    return tuple(result)
