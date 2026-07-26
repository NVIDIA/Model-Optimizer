# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Parallel mesh validation and GPU packing for campaign orchestration."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

__all__ = [
    "GpuAllocation",
    "ParallelMesh",
    "extract_stage_mesh",
    "gpus_per_instance",
    "normalize_vllm_topology",
    "pack_gpu_allocation",
    "validate_mesh",
    "vllm_topology_to_mesh",
]


@dataclass(frozen=True)
class ParallelMesh:
    """AutoModel-compatible parallel mesh for one model instance."""

    tp: int = 1
    cp: int = 1
    pp: int = 1
    ep: int = 1
    dp_shard: int = 1
    dp_replicate: int = 1

    def as_dict(self) -> dict[str, int]:
        return {
            "tp": self.tp,
            "cp": self.cp,
            "pp": self.pp,
            "ep": self.ep,
            "dp_shard": self.dp_shard,
            "dp_replicate": self.dp_replicate,
        }

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> ParallelMesh:
        def _int(name: str, default: int = 1) -> int:
            value = mapping.get(name, default)
            if value in (None, "none", "None", ""):
                return default
            return int(value)

        return cls(
            tp=_int("tp"),
            cp=_int("cp"),
            pp=_int("pp"),
            ep=_int("ep"),
            dp_shard=_int("dp_shard"),
            dp_replicate=_int("dp_replicate", _int("dp")),
        )


@dataclass(frozen=True)
class GpuAllocation:
    """Packed GPU allocation for one stage submission."""

    gpus_per_instance: int
    instances: int
    total_gpus: int
    gpus_per_node: int
    nodes: int
    exclusive: bool
    instances_per_node: int
    packed_instances: tuple[tuple[int, ...], ...]


def validate_mesh(mesh: ParallelMesh) -> None:
    """Validate AutoModel mesh constraints."""

    values = mesh.as_dict()
    invalid = {name: value for name, value in values.items() if value < 1}
    if invalid:
        raise ValueError(f"Parallel dimensions must be positive: {invalid}")
    if mesh.dp_shard % mesh.ep:
        raise ValueError(
            "dp_shard must be divisible by ep because EP overlays the FSDP shard axis; "
            f"got dp_shard={mesh.dp_shard}, ep={mesh.ep}"
        )


def gpus_per_instance(mesh: ParallelMesh) -> int:
    """Return GPU count for one coordinated model instance."""

    validate_mesh(mesh)
    return mesh.pp * mesh.dp_replicate * mesh.dp_shard * mesh.cp * mesh.tp


def pack_gpu_allocation(
    *,
    mesh: ParallelMesh,
    instances: int,
    gpus_per_node: int,
) -> GpuAllocation:
    """Pack independent instances onto nodes with cluster-safe exclusivity rules."""

    if instances < 1:
        raise ValueError(f"instances must be positive, got {instances}")
    if gpus_per_node < 1:
        raise ValueError(f"gpus_per_node must be positive, got {gpus_per_node}")

    per_instance = gpus_per_instance(mesh)
    total = per_instance * instances
    nodes = max(1, math.ceil(total / gpus_per_node))
    instances_per_node = max(1, gpus_per_node // per_instance) if per_instance else 1

    packed: list[tuple[int, ...]] = []
    remaining = instances
    while remaining > 0:
        count = min(instances_per_node, remaining)
        packed.append(tuple(range(count)))
        remaining -= count

    while len(packed) < nodes:
        packed.append(())

    used_on_last = len(packed[-1]) * per_instance if packed[-1] else 0
    exclusive = all(
        len(group) * per_instance == gpus_per_node for group in packed if group
    ) and used_on_last == gpus_per_node

    return GpuAllocation(
        gpus_per_instance=per_instance,
        instances=instances,
        total_gpus=total,
        gpus_per_node=gpus_per_node,
        nodes=nodes,
        exclusive=exclusive,
        instances_per_node=instances_per_node,
        packed_instances=tuple(packed[:nodes]),
    )


_STAGE_PARALLEL_PATHS: dict[str, tuple[str, ...]] = {
    "convert": ("convert", "automodel", "parallel"),
    "tokenize_data": ("tokenize_data", "automodel", "parallel"),
    "vllm_stats": ("vllm_stats", "runtime_stats", "topology"),
    "depth_importance": ("depth_importance", "automodel", "parallel"),
    "width_importance": ("pruning", "automodel", "parallel"),
    "sort": ("sort", "automodel", "parallel"),
    "sort_sanity": ("sort_sanity", "automodel", "parallel"),
    "width_sanity": ("width_sanity", "automodel", "parallel"),
    "slicing_sanity": ("slicing_sanity", "automodel", "parallel"),
    "bypass_sanity": ("bypass_sanity", "automodel", "parallel"),
    "bypass": ("bypass", "automodel", "parallel"),
    "build_library": ("build_library", "automodel", "parallel"),
    "replacement_scoring": ("replacement_scoring", "automodel", "parallel"),
    "zero_shot_evaluation": ("zero_shot_evaluation", "automodel", "parallel"),
    "global_distillation_sanity": ("global_distillation_sanity", "automodel", "parallel"),
    "global_distillation": ("global_distillation", "automodel", "parallel"),
    "post_distillation_evaluation": ("post_distillation_evaluation", "automodel", "parallel"),
}

_STAGE_PARALLEL_FALLBACKS: dict[str, tuple[str, ...]] = {
    "sort": ("pruning", "automodel", "parallel"),
    "bypass_sanity": ("bypass", "automodel", "parallel"),
    "build_library": ("replacement_scoring", "automodel", "parallel"),
    "post_distillation_evaluation": ("global_distillation", "automodel", "parallel"),
}


def _nested_get(config: Mapping[str, Any], path: tuple[str, ...]) -> Mapping[str, Any] | None:
    value: Any = config
    for key in path:
        if not isinstance(value, Mapping) or key not in value:
            return None
        value = value[key]
    return value if isinstance(value, Mapping) else None


def normalize_vllm_topology(topology: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize vLLM topology, including its boolean expert-parallel mode."""

    tp = int(topology.get("tensor_parallel_size", topology.get("tp", 1)) or 1)
    pp = int(topology.get("pipeline_parallel_size", topology.get("pp", 1)) or 1)
    dp = int(topology.get("data_parallel_size", topology.get("dp", 1)) or 1)
    prefill_cp = int(
        topology.get("prefill_context_parallel_size", topology.get("prefill_cp", 1)) or 1
    )
    decode_cp = int(
        topology.get("decode_context_parallel_size", topology.get("decode_cp", 1)) or 1
    )
    dimensions = {
        "tp": tp,
        "pp": pp,
        "dp": dp,
        "prefill_cp": prefill_cp,
        "decode_cp": decode_cp,
    }
    invalid = {name: value for name, value in dimensions.items() if value < 1}
    if invalid:
        raise ValueError(f"vLLM topology dimensions must be positive: {invalid}")
    if decode_cp > tp or tp % decode_cp:
        raise ValueError(f"decode context parallel size {decode_cp} must divide TP={tp}")

    explicit = topology.get("enable_expert_parallel")
    legacy = topology.get("expert_parallel_size", topology.get("ep"))
    full_ep = tp * dp
    if explicit is None:
        if legacy is None or int(legacy) == 1:
            enable_expert_parallel = False
        elif int(legacy) == full_ep:
            enable_expert_parallel = True
        else:
            raise ValueError(
                f"expert_parallel_size={legacy} is not an independent vLLM degree; "
                f"expected 1 or TP * DP={full_ep}"
            )
    else:
        enable_expert_parallel = bool(explicit)
        expected_legacy = full_ep if enable_expert_parallel else 1
        if legacy is not None and int(legacy) != expected_legacy:
            raise ValueError(
                f"expert_parallel_size={legacy} conflicts with "
                f"enable_expert_parallel={enable_expert_parallel}; "
                f"expected {expected_legacy}"
            )

    gpu_count = tp * pp * prefill_cp * dp
    configured_gpu_group = int(topology.get("gpu_group_size", gpu_count) or 1)
    if configured_gpu_group != gpu_count:
        raise ValueError(
            f"gpu_group_size={configured_gpu_group} does not match vLLM "
            f"world size={gpu_count}"
        )
    return {
        **dimensions,
        "enable_expert_parallel": enable_expert_parallel,
        "effective_ep": full_ep if enable_expert_parallel else 1,
        "gpu_count": gpu_count,
        "distributed_executor_backend": str(
            topology.get("distributed_executor_backend", "mp")
        ),
    }


def vllm_topology_to_mesh(topology: Mapping[str, Any]) -> ParallelMesh:
    """Convert vLLM topology to an allocation-only scheduler mesh."""

    canonical = normalize_vllm_topology(topology)
    mesh = ParallelMesh(
        tp=canonical["tp"],
        cp=canonical["prefill_cp"],
        pp=canonical["pp"],
        ep=1,
        dp_shard=1,
        dp_replicate=canonical["dp"],
    )
    validate_mesh(mesh)
    return mesh


def extract_stage_mesh(
    config: Mapping[str, Any],
    stage_id: str,
    override: Mapping[str, Any] | None = None,
) -> ParallelMesh:
    """Read the stage mesh from experiment config with optional override."""

    if stage_id == "vllm_stats":
        topology = _nested_get(config, ("vllm_stats", "runtime_stats", "topology"))
        mesh = vllm_topology_to_mesh(topology or {})
    elif stage_id == "aiperf":
        topology = _nested_get(config, ("aiperf", "topology"))
        mesh = vllm_topology_to_mesh(topology or {})
    else:
        path = _STAGE_PARALLEL_PATHS.get(stage_id)
        parallel = _nested_get(config, path) if path else None
        if parallel is None and stage_id in _STAGE_PARALLEL_FALLBACKS:
            parallel = _nested_get(config, _STAGE_PARALLEL_FALLBACKS[stage_id])
        mesh = ParallelMesh.from_mapping(parallel or {})

    if override:
        merged = mesh.as_dict()
        for key, value in override.items():
            if value is not None:
                merged[key] = int(value)
        mesh = ParallelMesh.from_mapping(merged)
    validate_mesh(mesh)
    return mesh
