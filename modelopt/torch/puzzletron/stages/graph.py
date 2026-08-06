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

"""Scheduler-neutral registry for public Puzzletron pipeline stages."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Callable

__all__ = [
    "LEGACY_POST_MIP_STAGE_IDS",
    "STAGE_REGISTRY",
    "STAGE_SPECS",
    "ArtifactChoice",
    "StageSkipReason",
    "StageSpec",
    "StageStatus",
    "StageTerminalState",
    "configured_parent_stage_ids",
    "configured_stage_ids",
    "distributed_stage_ids",
    "enabled_stage_ids",
    "required_stage_ids",
    "selected_parent_stage_ids",
    "stage_display_name",
    "stage_ids",
    "stage_is_enabled",
    "stage_spec",
    "stage_terminal_state",
    "topological_mapping_items",
    "topological_stage_ids",
]

LEGACY_POST_MIP_STAGE_IDS = frozenset(
    {
        "zero_shot_evaluation",
        "aiperf",
        "global_distillation_sanity",
        "global_distillation",
        "post_distillation_evaluation",
    }
)


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
    enabled_requires: tuple[str, ...] = ()
    parents: tuple[str, ...] = ()
    conditional_parents: tuple[tuple[str, tuple[str, bool]], ...] = ()
    artifact_choices: tuple[ArtifactChoice, ...] = ()
    distributed: bool = False
    report_order: int = 0
    topology_order: int = 0


class StageStatus(str, Enum):
    """Manifest terminal states accepted by Puzzletron stage consumers."""

    SUCCESS = "success"
    IMPORTED = "imported"
    SKIPPED = "skipped"


class StageSkipReason(str, Enum):
    """Reasons that may make a skipped stage an accepted terminal state."""

    DISABLED = "disabled"
    OPTIONAL = "optional"


@dataclass(frozen=True)
class StageTerminalState:
    """Typed terminal state shared by workers, manifests, resume, and orchestration."""

    status: StageStatus
    skip_reason: StageSkipReason | None = None

    @property
    def produced_artifacts(self) -> bool:
        """Return whether completion must be backed by stage artifacts."""

        return self.status in {StageStatus.SUCCESS, StageStatus.IMPORTED}

    def allows_completion(self, stage_id: str, config: Mapping[str, Any]) -> bool:
        """Return whether graph and config semantics allow this terminal state."""

        if self.produced_artifacts:
            return True
        if self.skip_reason is StageSkipReason.DISABLED:
            return not stage_is_enabled(stage_id, config)
        return self.skip_reason is StageSkipReason.OPTIONAL and not stage_spec(stage_id).required


def stage_terminal_state(payload: Mapping[str, Any] | None) -> StageTerminalState | None:
    """Parse one manifest terminal state, returning ``None`` for invalid evidence."""

    if not isinstance(payload, Mapping):
        return None
    try:
        status = StageStatus(payload.get("status"))
    except (TypeError, ValueError):
        return None
    raw_reason = payload.get("skip_reason")
    if status is StageStatus.SKIPPED:
        try:
            reason = StageSkipReason(raw_reason)
        except (TypeError, ValueError):
            return None
    else:
        if raw_reason not in (None, ""):
            return None
        reason = None
    return StageTerminalState(status=status, skip_reason=reason)


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
    enabled_requires: tuple[str, ...] = (),
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
            enabled_requires=enabled_requires,
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
    completion_artifacts=("pruning/pruning_scores/automodel/*/activation_passes_manifest.json",),
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
    enabled_requires=("sort_sanity",),
    parents=("sort_sanity",),
    completion_artifacts=("artifacts/width_sanity/summary.json",),
    distributed=True,
)
_stage(
    "slicing_sanity",
    "Slicing Sanity Check",
    enabled_requires=("width_sanity",),
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
        unknown_requirements = set(spec.enabled_requires) - set(registry)
        if unknown_requirements:
            raise ValueError(
                f"unknown enabled requirements {sorted(unknown_requirements)} "
                f"for stage {spec.stage_id!r}"
            )
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
        enabled = True
    elif spec.enabled_when is not None:
        path, expected = spec.enabled_when
        enabled = _config_value(config, path) is expected
    else:
        section = config.get(stage_id)
        enabled = (
            spec.default_enabled
            if not isinstance(section, Mapping)
            else bool(section.get("enabled", spec.default_enabled))
        )
    return enabled and all(
        stage_is_enabled(required_stage_id, config) for required_stage_id in spec.enabled_requires
    )


def enabled_stage_ids(config: Mapping[str, Any]) -> tuple[str, ...]:
    """Return configured stages in deterministic topological order."""

    return tuple(
        stage_id for stage_id in topological_stage_ids() if stage_is_enabled(stage_id, config)
    )


def configured_stage_ids(
    config: Mapping[str, Any],
    *,
    dynamic_post_mip_stage_ids: Iterable[str] = (),
) -> tuple[str, ...]:
    """Return the exact campaign DAG after replacing the legacy post-MIP tail."""

    dynamic = tuple(dict.fromkeys(str(stage_id) for stage_id in dynamic_post_mip_stage_ids))
    enabled = enabled_stage_ids(config)
    if dynamic:
        enabled = tuple(
            stage_id for stage_id in enabled if stage_id not in LEGACY_POST_MIP_STAGE_IDS
        )
    return (*enabled, *dynamic)


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


def configured_parent_stage_ids(
    stage_id: str,
    config: Mapping[str, Any],
) -> tuple[str, ...]:
    """Return enabled parents, contracting paths through disabled optional stages."""

    enabled = set(enabled_stage_ids(config))

    def enabled_ancestors(parent_id: str) -> tuple[str, ...]:
        if parent_id in enabled:
            return (parent_id,)
        return tuple(
            ancestor
            for grandparent_id in selected_parent_stage_ids(parent_id, config)
            for ancestor in enabled_ancestors(grandparent_id)
        )

    return tuple(
        dict.fromkeys(
            ancestor
            for parent_id in selected_parent_stage_ids(stage_id, config)
            for ancestor in enabled_ancestors(parent_id)
        )
    )


def topological_stage_ids(specs: Iterable[StageSpec] = STAGE_SPECS) -> tuple[str, ...]:
    """Return a deterministic topological order or raise for invalid graph metadata."""

    registry = _registry_for(specs)
    parents = {
        stage_id: (
            set(spec.parents)
            | set(spec.enabled_requires)
            | {parent for parent, _ in spec.conditional_parents}
            | {choice.parent for choice in spec.artifact_choices}
        )
        for stage_id, spec in registry.items()
    }
    order = {stage_id: spec.topology_order for stage_id, spec in registry.items()}
    result: list[str] = []
    while parents:
        ready = sorted(
            (stage_id for stage_id, values in parents.items() if not values), key=order.__getitem__
        )
        if not ready:
            cycle = ", ".join(sorted(parents, key=order.__getitem__))
            raise ValueError(f"Stage graph contains a cycle: {cycle}")
        stage_id = ready[0]
        result.append(stage_id)
        parents.pop(stage_id)
        for values in parents.values():
            values.discard(stage_id)
    return tuple(result)
