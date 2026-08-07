# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Immutable resolved campaign configuration for Puzzletron setup v2."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from puzzletron_setup import SetupError

if TYPE_CHECKING:
    from .state import WizardState

__all__ = [
    "CompatibilityOverrides",
    "CompatibilityProjection",
    "ResolvedAxisConfig",
    "ResolvedCampaignConfig",
    "ResolvedDataConfig",
    "ResolvedField",
    "ResolvedInfrastructureConfig",
    "ResolvedModelConfig",
    "ResolvedParallelProfile",
    "ResolvedStageResource",
    "resolve_campaign_config",
]

_PARALLEL_FIELDS = (
    "tp",
    "cp",
    "pp",
    "ep",
    "dp_shard",
    "dp_replicate",
    "sequence_parallel",
)


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze(item) for item in value)
    if isinstance(value, Path):
        return str(value)
    return deepcopy(value)


def _freeze_mapping(value: Any) -> Mapping[str, Any]:
    return _freeze(value) if isinstance(value, Mapping) else MappingProxyType({})


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_plain(item) for item in value), key=repr)
    return deepcopy(value)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _mapping_path(value: Any, parts: list[str]) -> tuple[bool, Any]:
    for part in parts:
        if not isinstance(value, Mapping) or part not in value:
            return False, None
        value = value[part]
    return True, value


def _named_values(values: Any, parts: list[str]) -> dict[str, Any]:
    resolved = {}
    for name, value in _mapping(values).items():
        found, effective = _mapping_path(value, parts)
        if found:
            resolved[str(name)] = deepcopy(effective)
    return resolved


def _effective_default_value(state: WizardState, path: str, fallback: Any) -> Any:
    """Return the authored consumer value for one resolved default, if present."""
    record = state.records().get(path)
    if record is not None:
        return deepcopy(record.effective)

    root, *parts = path.split(".")
    direct_collection = state.collection(root)
    if direct_collection is not None:
        found, value = _mapping_path(direct_collection, parts)
        if found:
            return deepcopy(value)

    if path == "profiles":
        profiles = _mapping(state.collection("parallel_profiles"))
        return deepcopy(profiles) if profiles else fallback

    if root == "stages" and len(parts) == 2 and parts[1] == "instances":
        found, value = _mapping_path(state.collection("stage_resources"), parts)
        return deepcopy(value) if found else fallback

    if root == "mip" and len(parts) == 1:
        runs = _mapping(_mapping(state.collection("mip_config")).get("runs"))
        field = parts[0]
        if field == "num_solutions":
            values = _named_values(runs, ["solver", "num_solutions"])
        elif field == "objective":
            values = {
                str(name): [
                    item["metric"]
                    for item in _mapping(run).get("objectives", ())
                    if isinstance(item, Mapping) and "metric" in item
                ]
                for name, run in runs.items()
            }
        elif field == "goal_metric":
            values = {
                str(name): list(_mapping(_mapping(run).get("constraints")))
                for name, run in runs.items()
            }
        elif field == "goal_value":
            values = {
                str(name): deepcopy(_mapping(_mapping(run).get("constraints")))
                for name, run in runs.items()
            }
        else:
            values = {}
        return values or fallback

    if root == "vllm" and len(parts) == 1 and parts[0] == "enabled":
        return bool(_mapping(state.collection("vllm_measurements")))
    if root == "vllm":
        measurements = state.collection("vllm_measurements")
        if len(parts) == 1:
            workload_key = {
                "batch_size": "batch_size",
                "max_num_seqs": "max_num_seqs",
                "prefill_seq_len": "prefill_seq_len",
                "generation_seq_len": "generation_seq_len",
            }.get(parts[0])
            if workload_key:
                values = _named_values(state.collection("serving_workloads"), [workload_key])
            elif parts[0] == "granularity":
                values = _named_values(measurements, ["granularity"])
            else:
                values = {}
        elif len(parts) == 2 and parts[0] == "topology":
            values = _named_values(measurements, ["runtime_stats", "topology", parts[1]])
        else:
            values = {}
        return values or fallback

    return fallback


@dataclass(frozen=True)
class ResolvedField:
    """One resolved value and all available authoring provenance."""

    value: Any
    source: str
    requested: Any
    effective: Any

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", _freeze(self.value))
        object.__setattr__(self, "source", str(self.source))
        object.__setattr__(self, "requested", _freeze(self.requested))
        object.__setattr__(self, "effective", _freeze(self.effective))


@dataclass(frozen=True)
class ResolvedAxisConfig:
    """One inspected model-axis identity and its legal values."""

    axis_id: str
    label: str
    teacher_value: int
    values: tuple[int, ...]
    alignment: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "axis_id", str(self.axis_id))
        object.__setattr__(self, "label", str(self.label))
        object.__setattr__(self, "teacher_value", int(self.teacher_value))
        object.__setattr__(self, "values", tuple(int(value) for value in self.values))
        object.__setattr__(self, "alignment", int(self.alignment))

    def _legacy_axis(self) -> dict[str, Any]:
        return {
            "axis_id": self.axis_id,
            "label": self.label,
            "teacher_value": self.teacher_value,
            "values": list(self.values),
            "alignment": self.alignment,
        }


@dataclass(frozen=True)
class ResolvedModelConfig:
    """Stable inspected model identity used by setup and compatibility rendering."""

    source: str
    requested_revision: str | None
    resolved_revision: str | None
    is_local: bool
    config: Mapping[str, Any]
    family: str
    descriptor: str
    family_config: str
    model_type: str
    architectures: tuple[str, ...]
    multimodal: bool
    moe: bool
    num_layers: int
    num_sublayers: int
    layer_counts: Mapping[str, int]
    facts: Mapping[str, Any]
    axes: tuple[ResolvedAxisConfig, ...]
    model_extra: Mapping[str, Any]
    inventory_extra: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "source", str(self.source))
        if self.requested_revision is not None:
            object.__setattr__(self, "requested_revision", str(self.requested_revision))
        if self.resolved_revision is not None:
            object.__setattr__(self, "resolved_revision", str(self.resolved_revision))
        object.__setattr__(self, "is_local", bool(self.is_local))
        object.__setattr__(self, "config", _freeze_mapping(self.config))
        object.__setattr__(self, "family", str(self.family))
        object.__setattr__(self, "descriptor", str(self.descriptor))
        object.__setattr__(self, "family_config", str(self.family_config))
        object.__setattr__(self, "model_type", str(self.model_type))
        object.__setattr__(
            self,
            "architectures",
            tuple(str(architecture) for architecture in self.architectures),
        )
        object.__setattr__(self, "multimodal", bool(self.multimodal))
        object.__setattr__(self, "moe", bool(self.moe))
        object.__setattr__(self, "num_layers", int(self.num_layers))
        object.__setattr__(self, "num_sublayers", int(self.num_sublayers))
        object.__setattr__(self, "layer_counts", _freeze_mapping(self.layer_counts))
        object.__setattr__(self, "facts", _freeze_mapping(self.facts))
        object.__setattr__(self, "axes", tuple(self.axes))
        object.__setattr__(self, "model_extra", _freeze_mapping(self.model_extra))
        object.__setattr__(self, "inventory_extra", _freeze_mapping(self.inventory_extra))

    def _legacy_model(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "requested_revision": self.requested_revision,
            "resolved_revision": self.resolved_revision,
            "is_local": self.is_local,
            "config": _plain(self.config),
            **_plain(self.model_extra),
        }

    def _legacy_inventory(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "descriptor": self.descriptor,
            "family_config": self.family_config,
            "model_type": self.model_type,
            "architectures": list(self.architectures),
            "multimodal": self.multimodal,
            "moe": self.moe,
            "num_layers": self.num_layers,
            "num_sublayers": self.num_sublayers,
            "layer_counts": _plain(self.layer_counts),
            "facts": _plain(self.facts),
            "axes": [axis._legacy_axis() for axis in self.axes],
            **_plain(self.inventory_extra),
        }


@dataclass(frozen=True)
class ResolvedDataConfig:
    """Stable dataset semantics selected by setup."""

    source: str | None
    selected_source: str | None
    adapter: str | None
    modality: str | None
    layout: str | None
    sequence_length: int
    subsets: tuple[str, ...]
    subset_revision: str | None
    subset_weights: Mapping[str, Any]
    acquisition: Mapping[str, Any]

    def __post_init__(self) -> None:
        for name in ("source", "selected_source", "adapter", "modality", "layout"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, str(value))
        object.__setattr__(self, "sequence_length", int(self.sequence_length))
        object.__setattr__(self, "subsets", tuple(str(name) for name in self.subsets))
        if self.subset_revision is not None:
            object.__setattr__(self, "subset_revision", str(self.subset_revision))
        object.__setattr__(self, "subset_weights", _freeze_mapping(self.subset_weights))
        object.__setattr__(self, "acquisition", _freeze_mapping(self.acquisition))


@dataclass(frozen=True)
class ResolvedInfrastructureConfig:
    """Stable runner and worker-environment semantics selected by setup."""

    runner_kind: str
    slurm: Mapping[str, Any]
    execution_contract: Mapping[str, Any]
    gpus_per_node: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "runner_kind", str(self.runner_kind))
        object.__setattr__(self, "slurm", _freeze_mapping(self.slurm))
        object.__setattr__(self, "execution_contract", _freeze_mapping(self.execution_contract))
        object.__setattr__(self, "gpus_per_node", int(self.gpus_per_node))


@dataclass(frozen=True)
class ResolvedParallelProfile:
    """One immutable named model-parallel profile."""

    name: str
    source_nonempty: bool = True
    tp: int = 1
    cp: int = 1
    pp: int = 1
    dp_shard: int = 1
    dp_replicate: int = 1
    ep: int = 1
    sequence_parallel: bool = False
    consumers: tuple[str, ...] = ()
    present_fields: frozenset[str] = frozenset(_PARALLEL_FIELDS)
    extra: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", str(self.name))
        object.__setattr__(self, "source_nonempty", bool(self.source_nonempty))
        for name in ("tp", "cp", "pp", "dp_shard", "dp_replicate", "ep"):
            object.__setattr__(self, name, int(getattr(self, name)))
        object.__setattr__(self, "sequence_parallel", bool(self.sequence_parallel))
        object.__setattr__(self, "consumers", tuple(str(item) for item in self.consumers))
        object.__setattr__(
            self,
            "present_fields",
            frozenset(str(item) for item in self.present_fields),
        )
        object.__setattr__(self, "extra", _freeze_mapping(self.extra))

    def _parallel(self) -> dict[str, Any]:
        values = {
            "tp": self.tp,
            "cp": self.cp,
            "pp": self.pp,
            "ep": self.ep,
            "dp_shard": self.dp_shard,
            "dp_replicate": self.dp_replicate,
            "sequence_parallel": self.sequence_parallel,
        }
        return {key: value for key, value in values.items() if key in self.present_fields}


@dataclass(frozen=True)
class ResolvedStageResource:
    """One immutable stage allocation and optional parallel-profile binding."""

    stage_id: str
    strategy: str
    instances: int
    resource: str
    gpus_per_node: int | None
    partition: str | None
    profile_name: str | None
    parallel: Mapping[str, Any] | None
    extra: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "stage_id", str(self.stage_id))
        object.__setattr__(self, "strategy", str(self.strategy))
        object.__setattr__(self, "instances", int(self.instances))
        object.__setattr__(self, "resource", str(self.resource))
        if self.gpus_per_node is not None:
            object.__setattr__(self, "gpus_per_node", int(self.gpus_per_node))
        if self.partition is not None:
            object.__setattr__(self, "partition", str(self.partition))
        if self.profile_name is not None:
            object.__setattr__(self, "profile_name", str(self.profile_name))
        if self.parallel is not None:
            object.__setattr__(self, "parallel", _freeze_mapping(self.parallel))
        object.__setattr__(self, "extra", _freeze_mapping(self.extra))


@dataclass(frozen=True)
class CompatibilityOverrides:
    """Quarantined late overlays for today's generated member contracts."""

    experiment: Mapping[str, Any]
    runner: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "experiment", _freeze_mapping(self.experiment))
        object.__setattr__(self, "runner", _freeze_mapping(self.runner))


@dataclass(frozen=True)
class CompatibilityProjection:
    """Explicit insertion-order selections required by legacy rendering."""

    workload_id: str
    first_measurement_id: str | None
    runtime_measurement_id: str | None
    first_parallel_profile_name: str | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "workload_id", str(self.workload_id))
        for name in (
            "first_measurement_id",
            "runtime_measurement_id",
            "first_parallel_profile_name",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, str(value))


@dataclass(frozen=True)
class ResolvedCampaignConfig:
    """Immutable semantic authority consumed by setup-v2 bundle generation."""

    model: ResolvedModelConfig
    data: ResolvedDataConfig
    infrastructure: ResolvedInfrastructureConfig
    pruning: Mapping[str, Any]
    serving_workloads: Mapping[str, Mapping[str, Any]]
    vllm_measurements: Mapping[str, Mapping[str, Any]]
    mip: Mapping[str, Any]
    mip_configured: bool
    post_mip_flows: Mapping[str, Any]
    post_mip_flows_configured: bool
    parallel_profiles: Mapping[str, ResolvedParallelProfile]
    stage_resources: Mapping[str, ResolvedStageResource]
    stage_batches: Mapping[str, Any]
    result_root: str
    provenance: Mapping[str, ResolvedField]
    compatibility: CompatibilityOverrides
    compatibility_projection: CompatibilityProjection

    def __post_init__(self) -> None:
        for name in (
            "pruning",
            "serving_workloads",
            "vllm_measurements",
            "mip",
            "post_mip_flows",
            "stage_batches",
        ):
            object.__setattr__(self, name, _freeze_mapping(getattr(self, name)))
        object.__setattr__(self, "mip_configured", bool(self.mip_configured))
        object.__setattr__(
            self,
            "post_mip_flows_configured",
            bool(self.post_mip_flows_configured),
        )
        object.__setattr__(
            self,
            "parallel_profiles",
            MappingProxyType(dict(self.parallel_profiles)),
        )
        object.__setattr__(
            self,
            "stage_resources",
            MappingProxyType(dict(self.stage_resources)),
        )
        object.__setattr__(self, "result_root", str(self.result_root))
        object.__setattr__(self, "provenance", MappingProxyType(dict(self.provenance)))


def _axis(raw: Mapping[str, Any]) -> ResolvedAxisConfig:
    return ResolvedAxisConfig(
        axis_id=str(raw.get("axis_id", "")),
        label=str(raw.get("label", raw.get("axis_id", ""))),
        teacher_value=int(raw.get("teacher_value", 0)),
        values=tuple(int(value) for value in raw.get("values") or ()),
        alignment=int(raw.get("alignment", 1)),
    )


def _model(
    model: Mapping[str, Any],
    inventory: Mapping[str, Any],
    source: Any,
) -> ResolvedModelConfig:
    model_keys = {
        "source",
        "requested_revision",
        "resolved_revision",
        "is_local",
        "config",
    }
    inventory_keys = {
        "family",
        "descriptor",
        "family_config",
        "model_type",
        "architectures",
        "multimodal",
        "moe",
        "num_layers",
        "num_sublayers",
        "layer_counts",
        "facts",
        "axes",
    }
    return ResolvedModelConfig(
        source=str(model.get("source", source or "")),
        requested_revision=model.get("requested_revision"),
        resolved_revision=model.get("resolved_revision"),
        is_local=bool(model.get("is_local", False)),
        config=_mapping(model.get("config")),
        family=str(inventory.get("family", "")),
        descriptor=str(inventory.get("descriptor", "")),
        family_config=str(inventory.get("family_config", "")),
        model_type=str(inventory.get("model_type", "")),
        architectures=tuple(str(item) for item in inventory.get("architectures") or ()),
        multimodal=bool(inventory.get("multimodal", False)),
        moe=bool(inventory.get("moe", False)),
        num_layers=int(inventory.get("num_layers", 0)),
        num_sublayers=int(inventory.get("num_sublayers", 0)),
        layer_counts=_mapping(inventory.get("layer_counts")),
        facts=_mapping(inventory.get("facts")),
        axes=tuple(
            _axis(raw) for raw in inventory.get("axes") or () if isinstance(raw, Mapping)
        ),
        model_extra={key: value for key, value in model.items() if key not in model_keys},
        inventory_extra={
            key: value for key, value in inventory.items() if key not in inventory_keys
        },
    )


def _parallel_profile(name: str, raw: Mapping[str, Any]) -> ResolvedParallelProfile:
    known = {
        "name",
        "tp",
        "cp",
        "pp",
        "dp_shard",
        "dp_replicate",
        "dp",
        "ep",
        "sequence_parallel",
        "consumers",
    }
    return ResolvedParallelProfile(
        name=name,
        source_nonempty=bool(raw),
        tp=int(raw.get("tp", 1)),
        cp=int(raw.get("cp", 1)),
        pp=int(raw.get("pp", 1)),
        dp_shard=int(raw.get("dp_shard", 1)),
        dp_replicate=int(raw.get("dp_replicate", 1)),
        ep=int(raw.get("ep", 1)),
        sequence_parallel=bool(raw.get("sequence_parallel", False)),
        consumers=tuple(str(item) for item in raw.get("consumers") or ()),
        present_fields=frozenset(key for key in _PARALLEL_FIELDS if key in raw),
        extra={key: value for key, value in raw.items() if key not in known},
    )


def _stage_resource(stage_id: str, raw: Mapping[str, Any]) -> ResolvedStageResource:
    known = {
        "strategy",
        "instances",
        "resource",
        "gpus_per_node",
        "partition",
        "profile_name",
        "parallel",
    }
    parallel = raw.get("parallel")
    return ResolvedStageResource(
        stage_id=stage_id,
        strategy=str(raw.get("strategy", "single")),
        instances=int(raw.get("instances", 1)),
        resource=str(raw.get("resource", "gpu")),
        gpus_per_node=(
            int(raw["gpus_per_node"])
            if raw.get("gpus_per_node") is not None
            else None
        ),
        partition=(str(raw["partition"]) if raw.get("partition") else None),
        profile_name=(str(raw["profile_name"]) if raw.get("profile_name") else None),
        parallel=_mapping(parallel) if isinstance(parallel, Mapping) else None,
        extra={key: value for key, value in raw.items() if key not in known},
    )


def _compatibility_projection(
    serving_workloads: Mapping[str, Any],
    vllm_measurements: Mapping[str, Any],
    parallel_profiles: Mapping[str, ResolvedParallelProfile],
) -> CompatibilityProjection:
    workload_id = str(next(iter(serving_workloads), "serving-default"))
    first_measurement_id = next(iter(vllm_measurements), None)
    matching_measurement = _mapping(vllm_measurements.get(workload_id))
    runtime_measurement_id = (
        workload_id if matching_measurement else first_measurement_id
    )
    return CompatibilityProjection(
        workload_id=workload_id,
        first_measurement_id=first_measurement_id,
        runtime_measurement_id=runtime_measurement_id,
        first_parallel_profile_name=next(iter(parallel_profiles), None),
    )


def resolve_campaign_config(state: WizardState) -> ResolvedCampaignConfig:
    """Resolve one mutable wizard state into an immutable campaign snapshot."""
    payload = deepcopy(state.payload)
    collections = _mapping(payload.get("collections"))
    field_records = {
        str(path): {
            "value": deepcopy(record.value),
            "source": str(record.source),
            "requested": deepcopy(record.requested),
            "effective": deepcopy(record.effective),
        }
        for path, record in state.records().items()
    }

    def effective(path: str, default: Any = None) -> Any:
        record = field_records.get(path)
        return deepcopy(record["effective"]) if record is not None else deepcopy(default)

    model_payload = _mapping(payload.get("model"))
    inventory_payload = _mapping(payload.get("inventory"))
    model = _model(model_payload, inventory_payload, effective("model.source", ""))

    selection = _mapping(collections.get("data_subset_selection"))
    subset_records = [
        item for item in selection.get("subsets") or () if isinstance(item, Mapping)
    ]
    data_source = effective("data.source")
    data = ResolvedDataConfig(
        source=data_source,
        selected_source=effective("data.selected_source", data_source),
        adapter=effective("data.adapter", "custom"),
        modality=effective("data.modality", "text"),
        layout=effective("data.layout", "fixed"),
        sequence_length=int(effective("data.sequence_length", 4096)),
        subsets=tuple(str(record["name"]) for record in subset_records),
        subset_revision=(
            str(selection["revision"]) if selection.get("revision") is not None else None
        ),
        subset_weights={
            str(record["name"]): deepcopy(record["weight"]) for record in subset_records
        },
        acquisition=_mapping(collections.get("data_acquisition")),
    )

    infrastructure = ResolvedInfrastructureConfig(
        runner_kind=str(effective("infrastructure.runner.kind", "slurm")),
        slurm={
            "account": effective("infrastructure.runner.slurm.account", ""),
            "partition_interactive": effective(
                "infrastructure.runner.slurm.partition_interactive", "interactive"
            ),
            "partition_batch": effective(
                "infrastructure.runner.slurm.partition_batch", "batch"
            ),
            "partition_cpu": effective("infrastructure.runner.slurm.partition_cpu", None),
            "time_limit": effective("infrastructure.runner.slurm.time_limit", "4:00:00"),
            "qos": effective("infrastructure.runner.slurm.qos", None),
            "max_nodes": effective("infrastructure.runner.slurm.max_nodes", 64),
        },
        execution_contract={
            "repository": effective(
                "infrastructure.execution_contract.repository", str(Path.cwd())
            ),
            "venv": effective("infrastructure.execution_contract.venv", ".venv"),
            "container": effective("infrastructure.execution_contract.container", None),
            "container_mounts": effective(
                "infrastructure.execution_contract.container_mounts", None
            ),
            "prerun_commands": effective(
                "infrastructure.execution_contract.prerun_commands", []
            ),
            "postrun_commands": effective(
                "infrastructure.execution_contract.postrun_commands", []
            ),
        },
        gpus_per_node=int(effective("infrastructure.gpus_per_node", 8)),
    )

    profiles = {
        str(name): _parallel_profile(str(name), raw)
        for name, raw in _mapping(collections.get("parallel_profiles")).items()
        if isinstance(raw, Mapping)
    }
    resources = {
        str(stage_id): _stage_resource(str(stage_id), raw)
        for stage_id, raw in _mapping(collections.get("stage_resources")).items()
        if isinstance(raw, Mapping)
    }
    serving_workloads = _mapping(collections.get("serving_workloads"))
    vllm_measurements = _mapping(collections.get("vllm_measurements"))
    compatibility_projection = _compatibility_projection(
        serving_workloads,
        vllm_measurements,
        profiles,
    )

    provenance: dict[str, ResolvedField] = {}
    for path, raw in _mapping(collections.get("default_resolutions")).items():
        if not isinstance(raw, Mapping):
            raise SetupError(f"Default resolution {path!r} must be a mapping.")
        value = deepcopy(raw.get("value"))
        provenance[str(path)] = ResolvedField(
            value=value,
            source=str(raw.get("source", "unknown")),
            requested=deepcopy(raw.get("requested")),
            effective=_effective_default_value(
                state,
                str(path),
                deepcopy(raw.get("effective", value)),
            ),
        )
    provenance.update(
        {
            path: ResolvedField(
                value=record["value"],
                source=record["source"],
                requested=record["requested"],
                effective=record["effective"],
            )
            for path, record in field_records.items()
        }
    )

    return ResolvedCampaignConfig(
        model=model,
        data=data,
        infrastructure=infrastructure,
        pruning=_mapping(collections.get("pruning")),
        serving_workloads=serving_workloads,
        vllm_measurements=vllm_measurements,
        mip=_mapping(collections.get("mip_config")),
        mip_configured=isinstance(collections.get("mip_config"), Mapping),
        post_mip_flows=_mapping(collections.get("post_mip_flows")),
        post_mip_flows_configured=isinstance(
            collections.get("post_mip_flows"), Mapping
        ),
        parallel_profiles=profiles,
        stage_resources=resources,
        stage_batches=_mapping(collections.get("stage_batches")),
        result_root=str(effective("output.result_root", "") or ""),
        provenance=provenance,
        compatibility=CompatibilityOverrides(
            experiment=_mapping(collections.get("experiment_overrides")),
            runner=_mapping(collections.get("runner_overrides")),
        ),
        compatibility_projection=compatibility_projection,
    )
