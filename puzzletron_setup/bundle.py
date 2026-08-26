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

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render and validate portable Puzzletron setup bundles."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from puzzletron_orchestrator import vllm_topology_to_mesh
from puzzletron_orchestrator.compiler import (
    compile_campaign_plan,
    load_execution_config,
    load_runner_config,
)
from puzzletron_orchestrator.controller import dry_run_plan

from . import SetupError

__all__ = [
    "BundleResult",
    "BundleValidation",
    "build_bundles",
    "dry_run_bundle",
    "render_execution",
    "render_experiment",
    "render_runner",
    "validate_bundle",
]

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = REPOSITORY_ROOT / "examples/puzzletron/configs/base.yaml"


@dataclass(frozen=True)
class BundleValidation:
    """Dependency-light validation result for one generated bundle."""

    valid: bool
    stage_count: int = 0
    submission_count: int = 0
    error: str | None = None


@dataclass(frozen=True)
class BundleResult:
    """Locations and independent validation results for generated bundles."""

    campaign_dir: Path
    smoke: BundleValidation
    production: BundleValidation


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _deep_merge(base: Mapping[str, Any], update: Mapping[str, Any]) -> dict[str, Any]:
    merged = deepcopy(dict(base))
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _load_fragment(path: Path) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text()) or {}
    except (OSError, yaml.YAMLError) as error:
        raise SetupError(f"Cannot load canonical config fragment {path}: {error}") from error
    if not isinstance(payload, dict):
        raise SetupError(f"Canonical config fragment must be a mapping: {path}")
    payload.pop("defaults", None)
    return payload


def _atomic_yaml(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("w") as stream:
            yaml.safe_dump(_plain(value), stream, sort_keys=False, width=100)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except OSError as error:
        raise SetupError(f"Cannot write generated config {path}: {error}") from error


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("w") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except OSError as error:
        raise SetupError(f"Cannot write generated file {path}: {error}") from error


def _answers(state: Mapping[str, Any], section: str) -> dict[str, Any]:
    return _mapping(_mapping(state.get("answers")).get(section))


def _parallel(mesh: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "tp": int(mesh.get("tp", 1)),
        "cp": int(mesh.get("cp", 1)),
        "pp": int(mesh.get("pp", 1)),
        "ep": int(mesh.get("ep", 1)),
        "dp_shard": int(mesh.get("dp_shard", 1)),
        "dp_replicate": int(mesh.get("dp_replicate", 1)),
        "sequence_parallel": False,
    }


def _serving_parallel(topology: Mapping[str, Any]) -> dict[str, Any]:
    """Convert a vLLM Serving topology to the scheduler's allocation mesh."""
    try:
        mesh = vllm_topology_to_mesh(topology)
    except (TypeError, ValueError) as error:
        raise SetupError(str(error)) from error
    return {**mesh.as_dict(), "sequence_parallel": False}


def _aligned_batch_size(mesh: Mapping[str, Any], requested: int = 1) -> int:
    """Round a batch up to the mesh's minimum pipeline/DP scheduling unit."""
    unit = int(mesh.get("pp", 1)) * int(mesh.get("dp_shard", 1)) * int(mesh.get("dp_replicate", 1))
    requested = max(int(requested), 1)
    return max(unit, ((requested + unit - 1) // unit) * unit)


def _align_model_stage_batches(
    config: dict[str, Any],
    inherited_mesh: Mapping[str, Any] | None = None,
) -> None:
    """Align every model-stage batch to its inherited PP/DP scheduling unit."""
    automodel = _mapping(config.get("automodel"))
    mesh = _mapping(automodel.get("parallel")) or inherited_mesh
    if mesh:
        for key in ("micro_batch_size", "val_micro_batch_size", "local_batch_size"):
            if key in config:
                if isinstance(config[key], str) and "${" in config[key]:
                    continue
                config[key] = _aligned_batch_size(mesh, int(config[key]))
    for value in config.values():
        if isinstance(value, dict):
            _align_model_stage_batches(value, mesh)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    _align_model_stage_batches(item, mesh)


def _model_info(state: Mapping[str, Any]) -> dict[str, Any]:
    model = _mapping(state.get("model"))
    inventory = _mapping(state.get("inventory"))
    config = _mapping(model.get("config"))
    text_config = _mapping(config.get("text_config")) or config
    facts = _mapping(inventory.get("facts"))
    result = {
        "hf_repo": model.get("source"),
        "hf_revision": model.get("resolved_revision"),
        "model_type": inventory.get("model_type"),
        "architectures": list(inventory.get("architectures") or ()),
        "num_hidden_layers": int(inventory.get("num_layers", 0)),
        "layer_counts": _mapping(inventory.get("layer_counts")),
    }
    result.update(facts)
    for key in (
        "max_position_embeddings",
        "hybrid_override_pattern",
        "layers_block_type",
        "mtp_num_hidden_layers",
    ):
        if key in text_config:
            result[key] = text_config[key]
    return {key: value for key, value in result.items() if value is not None}


def _axis_config(pruning: Mapping[str, Any]) -> dict[str, Any]:
    axes = {}
    for axis_id, raw in _mapping(pruning.get("axes")).items():
        axis = _mapping(raw)
        teacher = int(axis["teacher_value"])
        values = list(dict.fromkeys(int(value) for value in axis.get("values") or ()))
        if not values:
            raise SetupError(f"Pruning axis {axis_id!r} must select at least one size.")
        axes[axis_id] = {
            "enabled": bool(axis.get("enabled", True)) and any(value < teacher for value in values),
            "teacher_value": teacher,
            "values": values,
        }
    if not axes or not any(bool(axis["enabled"]) for axis in axes.values()):
        raise SetupError(
            "Width-axis selections do not prune any dimension. "
            "Select at least one size smaller than its teacher size."
        )
    return axes


_ACTIVATION_PASS_AXES = {
    "hidden_width": ("hidden_width",),
    "ffn_iterative": ("ffn_intermediate",),
    "ffn_intermediate": ("ffn_intermediate",),
    "attention_grouped": (
        "kv_groups",
        "q_heads_per_group",
        "query_heads",
        "kv_heads",
        "qk_head_dim",
    ),
    "mla_heads": ("mla_heads",),
    "gdn_activation": (
        "gdn_key_groups",
        "gdn_value_heads_per_group",
        "gdn_key_head_dim",
        "gdn_value_head_dim",
    ),
    "moe_expert_removal": ("moe_experts",),
    "moe_experts": ("moe_experts",),
    "moe_expert_intermediate": ("moe_expert_intermediate",),
    "moe_shared_expert_intermediate": ("moe_shared_expert_intermediate",),
    "moe_latent": ("moe_latent_dim",),
    "moe_latent_dim": ("moe_latent_dim",),
    "mamba_head_and_dim": ("mamba_heads", "mamba_head_dim"),
    "ple_width": ("ple_width",),
}


def _enabled_activation_passes(
    passes: list[dict[str, Any]],
    axes: Mapping[str, Any],
) -> list[dict[str, Any]]:
    enabled_axes = {
        str(axis_id) for axis_id, raw in axes.items() if bool(_mapping(raw).get("enabled", False))
    }
    result = []
    for raw in passes:
        entry = deepcopy(raw)
        configured_axis_ids = entry.get("axis_ids")
        axis_ids = tuple(
            str(axis_id)
            for axis_id in (
                configured_axis_ids
                if configured_axis_ids is not None
                else _ACTIVATION_PASS_AXES.get(str(entry.get("name")), ())
            )
        )
        if axis_ids:
            active_axis_ids = [axis_id for axis_id in axis_ids if axis_id in enabled_axes]
            if not active_axis_ids:
                continue
            entry["axis_ids"] = active_axis_ids
            if str(entry.get("name")) == "attention_grouped":
                entry.setdefault("activation_hooks_kwargs", {})["scored_axes"] = active_axis_ids
        result.append(entry)
    return result


def _attention_heads(axes: Mapping[str, Any]) -> list[list[int]]:
    kv_axis = _mapping(axes.get("kv_groups"))
    q_axis = _mapping(axes.get("q_heads_per_group"))
    if not kv_axis or not q_axis:
        return []
    kv_values = [kv_axis["teacher_value"], *kv_axis.get("values", ())]
    q_values = [q_axis["teacher_value"], *q_axis.get("values", ())]
    pairs = {(int(kv) * int(q), int(kv)) for kv in kv_values for q in q_values}
    return [list(pair) for pair in sorted(pairs)]


def _mip_runs(state: Mapping[str, Any], *, smoke: bool) -> dict[str, Any]:
    runs = deepcopy(_mapping(_answers(state, "mip").get("runs")))
    if smoke:
        for run in runs.values():
            solver = _mapping(run.get("solver"))
            solver["num_solutions"] = min(int(solver.get("num_solutions", 3)), 64)
            solver["max_seconds_per_solution"] = min(
                int(solver.get("max_seconds_per_solution", 120)), 30
            )
            run["solver"] = solver
            homogeneous = _mapping(run.get("homogeneous"))
            if isinstance(homogeneous.get("keep"), int):
                homogeneous["keep"] = min(int(homogeneous["keep"]), 16)
            run["homogeneous"] = homogeneous
    return runs


def _post_mip_flows(
    state: Mapping[str, Any],
    *,
    smoke: bool,
    common_mesh: Mapping[str, Any],
    global_kd_mesh: Mapping[str, Any],
    default_serving_topology: Mapping[str, Any],
) -> dict[str, Any]:
    flows = deepcopy(_mapping(_answers(state, "post_mip").get("flows")))
    for flow in flows.values():
        for node in _mapping(flow.get("nodes")).values():
            node_type = node.get("type")
            config = _mapping(node.get("config"))
            if node_type in {"evaluation", "materialize"}:
                config.setdefault("automodel", {})["parallel"] = _parallel(common_mesh)
                if node_type == "evaluation":
                    config["micro_batch_size"] = _aligned_batch_size(
                        common_mesh, int(config.get("micro_batch_size", 1))
                    )
            elif node_type == "aiperf":
                config.setdefault("use_server_token_count", True)
                config.setdefault("topology", deepcopy(dict(default_serving_topology)))
                if smoke:
                    config["minimum_request_count"] = 4
                    config["requests_per_concurrency"] = 1
            elif node_type == "downstream_evaluation":
                config.setdefault("model", "vllm")
                config.setdefault("batch_size", 1)
                config.setdefault("log_samples", True)
                config.setdefault("topology", deepcopy(dict(default_serving_topology)))
                if smoke:
                    config["limit"] = min(int(config.get("limit", 8) or 8), 8)
            elif node_type == "global_kd":
                config.setdefault("automodel", {})["parallel"] = _parallel(global_kd_mesh)
                config["local_batch_size"] = _aligned_batch_size(
                    global_kd_mesh, int(config.get("local_batch_size", 1))
                )
            if smoke and node_type == "evaluation":
                config["eval_samples"] = min(int(config.get("eval_samples", 128)), 16)
            if smoke and node_type == "global_kd":
                config["max_steps"] = min(int(config.get("max_steps", 128)), 8)
            if config or "config" in node:
                node["config"] = config
    return flows


def render_experiment(state: Mapping[str, Any], budget: str) -> dict[str, Any]:
    """Render one fully composed experiment mapping for smoke or production."""
    if budget not in {"smoke", "production"}:
        raise ValueError(f"Unknown bundle budget: {budget}")
    smoke = budget == "smoke"
    inventory = _mapping(state.get("inventory"))
    profile_path = REPOSITORY_ROOT / str(inventory["family_config"])
    base = _load_fragment(BASE_CONFIG)
    family = _load_fragment(profile_path)
    config = _deep_merge(base, family)

    model = _mapping(state.get("model"))
    data = _answers(state, "data")
    data_acquisition = _mapping(data.get("acquisition"))
    pruning = _answers(state, "pruning")
    runtime = _answers(state, "runtime")
    infrastructure = _answers(state, "infrastructure")
    output = _answers(state, "output")
    meshes = _mapping(infrastructure.get("meshes"))
    common_mesh = _mapping(meshes.get("common"))
    bypass_mesh = _mapping(meshes.get("bypass"))
    global_kd_mesh = _mapping(meshes.get("global_kd"))
    axes = _axis_config(pruning)
    enabled_axis_ids = [
        axis_id for axis_id, axis in axes.items() if bool(axis.get("enabled", False))
    ]
    sequence_length = int(data["sequence_length"])
    width_samples = int(pruning["width_importance_samples"])
    depth_samples = int(pruning.get("depth_importance_samples", 128))
    sort_sanity_samples = int(pruning.get("sort_sanity_samples", 128))
    width_sanity_samples = int(pruning.get("width_sanity_samples", 128))
    replacement_samples = int(pruning["replacement_samples"])
    if smoke:
        width_samples = min(width_samples, 128)
        depth_samples = min(depth_samples, 16)
        sort_sanity_samples = min(sort_sanity_samples, 16)
        width_sanity_samples = min(width_sanity_samples, 16)
        replacement_samples = min(replacement_samples, 16)
    result_root = str(output["result_root"]).rstrip("/")
    puzzle_dir = f"{result_root}/{budget}"

    activation_passes = _enabled_activation_passes(
        list(_mapping(config.get("pruning")).get("activation_passes") or ()),
        axes,
    )
    hidden_axis = _mapping(axes.get("hidden_width"))
    if bool(hidden_axis.get("enabled", False)) and not any(
        entry.get("name") == "hidden_width" for entry in activation_passes
    ):
        activation_passes.insert(
            0,
            {
                "name": "hidden_width",
                "axis_ids": ["hidden_width"],
                "activation_hooks_kwargs": {"method": "minitron_hidden_width"},
            },
        )
    embedding_widths = (
        list(hidden_axis.get("values") or ()) if bool(hidden_axis.get("enabled", False)) else []
    )
    model_parallel = _parallel(common_mesh)
    bypass = _mapping(pruning.get("bypass"))
    bypass_samples = int(bypass.get("samples", 4096))
    if smoke:
        bypass_samples = min(bypass_samples, 64)
    scoring_batch_size = _aligned_batch_size(common_mesh)
    bypass_batch_size = _aligned_batch_size(bypass_mesh, int(bypass.get("batch_size", 8)))
    bypass_sequence_length = int(bypass.get("sequence_length", sequence_length))
    vllm_topology = {
        "tensor_parallel_size": 1,
        "pipeline_parallel_size": 1,
        "prefill_context_parallel_size": 1,
        "decode_context_parallel_size": 1,
        "distributed_executor_backend": "mp",
        "gpu_group_size": 1,
    }
    default_serving_topology = {
        **vllm_topology,
        "data_parallel_size": 1,
        "enable_expert_parallel": False,
    }

    train_cache = f"{puzzle_dir}/dataset_cache/train.tokens"
    validation_cache = f"{puzzle_dir}/dataset_cache/validation.tokens"
    overlay = {
        "defaults": ["_self_"],
        "clean_config_root": str(Path(puzzle_dir) / "resolved_config"),
        "puzzle_dir": puzzle_dir,
        "dataset_path": data["source"],
        "input_hf_model_path": model["source"],
        "model_info": _model_info(state),
        "model": {
            "source": model["source"],
            "revision": model.get("resolved_revision") or model.get("requested_revision"),
            "descriptor_override": inventory["descriptor"],
            "trust_remote_code": inventory.get("family") == "nemotron3",
            "force_hf": False,
        },
        "capability_validation": {"require_complete_pipeline": True},
        "data": {
            "modality": data["modality"],
            "layout": data["layout"],
            "max_sample_length": sequence_length,
            "sequence_length": sequence_length,
            "path": data["source"],
            "acquisition": deepcopy(data_acquisition) if data_acquisition else None,
            "packing": (
                {
                    "pack_size": sequence_length,
                    "packing_ratio": 0.9,
                    "drop_long_samples": True,
                }
                if data["layout"] == "packed_varlen"
                else None
            ),
            "calibration": {
                "path": data["source"],
                "num_samples": width_samples,
                "micro_batch_size": scoring_batch_size,
                "seq_len": sequence_length,
            },
            "replacement_scoring": {
                "num_samples": replacement_samples,
                "micro_batch_size": scoring_batch_size,
            },
        },
        "train_token_cache_path": train_cache,
        "validation_token_cache_path": validation_cache,
        "tokenize_data": {
            "enabled": data["modality"] == "text" and data["layout"] == "fixed",
            "workers": 16 if smoke else 64,
            "tokenize_batch_size": 64,
            "content_field": "messages",
            "caches": [
                {
                    "output": train_cache,
                    "split": "train",
                    "num_samples": width_samples,
                    "seq_length": sequence_length,
                    "shuffle_seed": 444,
                },
                {
                    "output": validation_cache,
                    "split": "validation",
                    "num_samples": max(
                        depth_samples,
                        sort_sanity_samples if pruning.get("sort_sanity", False) else 0,
                        width_sanity_samples if pruning.get("width_sanity", False) else 0,
                        replacement_samples,
                    ),
                    "seq_length": sequence_length,
                    "shuffle_seed": 445,
                },
            ],
        },
        "embedding_pruning": {
            "enabled": bool(embedding_widths),
            "widths": embedding_widths,
            "alignment": int(
                _mapping(_mapping(pruning.get("axes")).get("hidden_width")).get("alignment", 1)
            ),
            "cycle_widths": True,
        },
        "pruning": {
            "eval_samples": width_samples,
            "micro_batch_size": scoring_batch_size,
            "block_size": sequence_length,
            "dataset_path": data["source"],
            "activation_passes": activation_passes,
            "automodel": {"parallel": model_parallel},
            "intermediate_size_list": list(
                _mapping(axes.get("ffn_intermediate")).get("values") or ()
            ),
            "attn_heads_list": _attention_heads(axes),
        },
        "search_space": {"axes": axes},
        "sort_sanity": {
            "enabled": bool(pruning.get("sort_sanity", False)),
            "eval_samples": sort_sanity_samples,
            "micro_batch_size": scoring_batch_size,
            "block_size": sequence_length,
            "packed_token_cache_path": validation_cache,
            "automodel": {"parallel": model_parallel},
        },
        "width_sanity": {
            "enabled": bool(
                pruning.get("sort_sanity", False) and pruning.get("width_sanity", False)
            ),
            "axes": enabled_axis_ids,
            "single_load_parent_sweep": True,
            "one_case_per_axis": False,
            "target_count_per_axis": int(pruning.get("width_sanity_targets_per_axis", 2)),
            "layer_count": int(pruning.get("width_sanity_layer_count", 3)),
            "layer_selection": "spread",
            "methods": ["activation", "random", "reverse"],
            "physical_realization": bool(
                pruning.get("sort_sanity", False)
                and pruning.get("width_sanity", False)
                and pruning.get("slicing_sanity", False)
            ),
            "cleanup_physical_checkpoints": True,
            "reuse_sort_equivalence": True,
            "require_beats_random": False,
            "require_beats_reverse": False,
            "require_physical_equivalence": bool(
                pruning.get("sort_sanity", False)
                and pruning.get("width_sanity", False)
                and pruning.get("slicing_sanity", False)
            ),
            "physical_equivalence_tolerance": 0.001,
            "eval_samples": width_sanity_samples,
            "micro_batch_size": scoring_batch_size,
            "block_size": sequence_length,
            "packed_token_cache_path": validation_cache,
            "automodel": {"parallel": model_parallel},
        },
        "slicing_sanity": {
            "enabled": bool(
                pruning.get("sort_sanity", False)
                and pruning.get("width_sanity", False)
                and pruning.get("slicing_sanity", False)
            ),
            "backend": "distributed_parent_sweep",
        },
        "bypass_sanity": {
            "enabled": bool(bypass.get("sanity", False)),
            "steps": min(128, bypass_samples),
        },
        "bypass": {
            "enabled": bool(bypass.get("enabled", False)),
            "granularity": bypass.get("granularity", "subblock"),
            "automodel": {"parallel": _parallel(bypass_mesh)},
            "iter_num": 1,
            "step_num": 1,
            "data": {
                "block_size": bypass_sequence_length,
                "max_eval_samples": bypass_samples,
                "packed_token_cache_path": train_cache,
            },
            "training": {
                "training_tokens": bypass_samples * bypass_sequence_length,
                "micro_batch_size": bypass_batch_size,
                "grad_accumulation_steps": int(bypass.get("grad_accumulation_steps", 1)),
            },
        },
        "depth_importance": {
            "enabled": int(pruning.get("depth_remove", 0)) > 0,
            "granularity": pruning.get("depth_granularity", "subblock"),
            "source_checkpoint_dir": "${teacher_dir}",
            "output_dir": "${puzzle_dir}/depth/iterative",
            "expected_initial_sublayers": int(inventory.get("num_sublayers", 0)),
            "max_removals": int(pruning.get("depth_remove", 0)),
            "max_subblocks_to_remove": int(pruning.get("depth_remove", 0)),
            "metric": "lm_loss",
            "eval_samples": depth_samples,
            "micro_batch_size": scoring_batch_size,
            "block_size": sequence_length,
            "packed_token_cache_path": validation_cache,
            "automodel": {"parallel": model_parallel},
        },
        "build_library": {
            "enabled": True,
            "include_bypass": bool(bypass.get("enabled", False)),
        },
        "replacement_scoring": {
            "enabled": True,
            "granularity": pruning.get("replacement_granularity", "subblock"),
            "eval_samples": replacement_samples,
            "micro_batch_size": scoring_batch_size,
            "block_size": sequence_length,
            "dataset_path": data["source"],
            "packed_token_cache_path": validation_cache,
            "automodel": {"parallel": model_parallel},
        },
        "vllm_stats": {
            "enabled": bool(runtime.get("vllm_enabled", False)),
            "model_hidden_sizes": list(_mapping(axes.get("hidden_width")).get("values") or ()),
            "batch_sizes": [int(runtime.get("concurrency", 1))],
            "prefill_seq_len": int(runtime.get("isl", sequence_length)),
            "generation_seq_len": int(runtime.get("osl", 1024)),
            "runtime_stats": {
                "enabled": bool(runtime.get("vllm_enabled", False)),
                "granularity": runtime.get("granularity", "subblock"),
                "max_num_seqs": int(runtime.get("concurrency", 1)),
                "fixed_overhead_relative_tolerance": 0.6,
                "topology": deepcopy(vllm_topology),
            },
        },
        "mip": {
            "enabled": True,
            "score_granularity": pruning.get("replacement_granularity", "subblock"),
            "workloads": {
                runtime.get("workload_id", "serving-default"): {
                    "isl": int(runtime.get("isl", sequence_length)),
                    "osl": int(runtime.get("osl", 1024)),
                    "batch_size": int(runtime.get("concurrency", 1)),
                    "concurrency": int(runtime.get("concurrency", 1)),
                }
            },
            "runs": _mip_runs(state, smoke=smoke),
        },
        "post_mip": {
            "flows": _post_mip_flows(
                state,
                smoke=smoke,
                common_mesh=common_mesh,
                global_kd_mesh=global_kd_mesh,
                default_serving_topology=default_serving_topology,
            )
        },
        "zero_shot_evaluation": {"enabled": False},
        "aiperf": {"enabled": False},
        "global_distillation_sanity": {"enabled": False},
        "global_distillation": {
            "enabled": False,
            "automodel": {"parallel": _parallel(global_kd_mesh)},
            "local_batch_size": _aligned_batch_size(global_kd_mesh),
        },
        "post_distillation_evaluation": {"enabled": False},
    }
    if data.get("subsets"):
        overlay["data"].update(
            {
                "subsets": list(data["subsets"]),
                "subset_revision": data.get("subset_revision"),
                "subset_weights": deepcopy(_mapping(data.get("subset_weights"))),
            }
        )
    if not data_acquisition:
        overlay["data"].pop("acquisition", None)
    if data["modality"] == "multimodal" or data["layout"] != "fixed":
        overlay["tokenize_data"]["caches"] = []
    rendered = _deep_merge(config, overlay)
    if data["modality"] == "multimodal" or data["layout"] != "fixed":
        for section in (
            "sort_sanity",
            "width_sanity",
            "depth_importance",
            "replacement_scoring",
        ):
            rendered[section].pop("packed_token_cache_path", None)
        rendered["bypass"]["data"].pop("packed_token_cache_path", None)
    _align_model_stage_batches(rendered)
    return rendered


def render_runner(state: Mapping[str, Any], budget: str) -> dict[str, Any]:
    """Render one existing-schema Slurm or SSH bare-metal runner mapping."""
    del budget
    infrastructure = _answers(state, "infrastructure")
    runner_answers = _mapping(infrastructure.get("runner"))
    runner = {
        "kind": runner_answers["kind"],
        "execution_contract": deepcopy(infrastructure["execution_contract"]),
    }
    if runner["kind"] == "slurm":
        slurm = deepcopy(_mapping(runner_answers.get("slurm")))
        slurm["partition"] = slurm.get("partition_batch", "batch")
        runner["slurm"] = slurm
    elif runner["kind"] == "baremetal":
        runner["inventory"] = deepcopy(_mapping(runner_answers.get("inventory")))
    else:
        raise SetupError(f"Unsupported generated runner kind: {runner['kind']}")
    return {"runner": runner}


def _post_mip_candidate_limits(experiment: Mapping[str, Any]) -> dict[str, int | None]:
    """Return known upper bounds on candidates entering each post-MIP stage."""
    limits: dict[str, int | None] = {}
    flows = _mapping(_mapping(experiment.get("post_mip")).get("flows"))
    for flow_id, flow in flows.items():
        nodes = _mapping(flow.get("nodes"))
        pending = dict(nodes)
        output_limits: dict[str, int | None] = {"source": None}
        while pending:
            progressed = False
            for node_id, raw_node in tuple(pending.items()):
                node = _mapping(raw_node)
                input_id = str(node.get("input", "source"))
                if input_id not in output_limits:
                    continue
                output_limit = output_limits[input_id]
                if node.get("type") == "filter" and node.get("mode") in {
                    "top_k",
                    "aggregate_rank",
                }:
                    top_k = node.get("top_k", 1)
                    filter_limit = (
                        sum(int(value) for value in top_k.values())
                        if isinstance(top_k, Mapping)
                        else int(top_k)
                    )
                    if node.get("best_selection_mode") == "best_per_concurrency":
                        input_node = _mapping(nodes.get(input_id))
                        if input_node.get("type") == "aiperf":
                            raw_concurrency = _mapping(input_node.get("config")).get(
                                "concurrency", [1]
                            )
                            concurrency_count = (
                                1
                                if isinstance(raw_concurrency, (int, str))
                                else len({int(value) for value in raw_concurrency})
                            )
                            filter_limit *= concurrency_count
                    output_limit = (
                        filter_limit if output_limit is None else min(output_limit, filter_limit)
                    )
                output_limits[str(node_id)] = output_limit
                limits[f"post.{flow_id}.{node_id}"] = output_limit
                pending.pop(node_id)
                progressed = True
            if not progressed:
                unresolved = ", ".join(sorted(str(node_id) for node_id in pending))
                raise SetupError(
                    f"Cannot resolve post-MIP candidate limits for {flow_id}: {unresolved}"
                )
    return limits


def _dynamic_stage_entries(
    experiment: Mapping[str, Any],
    workers: Mapping[str, Any],
    gpus_per_node: int,
    common: Mapping[str, Any],
    single_gpu: Mapping[str, Any],
    cpu_partition: str | None,
    *,
    pool_source_evaluations: bool,
) -> dict[str, Any]:
    entries = {}
    candidate_limits = _post_mip_candidate_limits(experiment)
    for flow_id, flow in _mapping(_mapping(experiment.get("post_mip")).get("flows")).items():
        for node_id, node in _mapping(flow.get("nodes")).items():
            node_type = str(node.get("type"))
            selector = node_type in {"filter", "manual_filter"}
            cpu_stage = selector or node_type == "materialize"
            pooled_evaluation = (
                pool_source_evaluations
                and node_type == "evaluation"
                and str(node.get("input", "source")) == "source"
            )
            worker_limit = int(workers.get("pool" if pooled_evaluation else "sharded", 1))
            candidate_limit = candidate_limits[f"post.{flow_id}.{node_id}"]
            instances = (
                worker_limit if candidate_limit is None else min(worker_limit, candidate_limit)
            )
            entry = {
                "strategy": (
                    "single" if selector else "persistent_pool" if pooled_evaluation else "sharded"
                ),
                "instances": 1 if cpu_stage else max(1, instances),
                "gpus_per_node": gpus_per_node,
            }
            if cpu_stage and cpu_partition:
                entry.update(resource="cpu", partition=cpu_partition)
            if node_type == "evaluation":
                entry["parallel"] = dict(common)
            elif node_type in {"aiperf", "downstream_evaluation"}:
                config = _mapping(node.get("config"))
                entry["parallel"] = _serving_parallel(_mapping(config.get("topology")))
            elif node_type == "materialize":
                entry["parallel"] = dict(single_gpu)
            entries[f"post.{flow_id}.{node_id}"] = entry
    return entries


def render_execution(
    state: Mapping[str, Any], experiment: Mapping[str, Any], budget: str
) -> dict[str, Any]:
    """Render scheduler-neutral execution semantics and dynamic node resources."""
    del budget
    infrastructure = _answers(state, "infrastructure")
    workers = _mapping(infrastructure.get("workers"))
    meshes = _mapping(infrastructure.get("meshes"))
    common = _parallel(_mapping(meshes.get("common")))
    bypass = _parallel(_mapping(meshes.get("bypass")))
    global_kd = _parallel(_mapping(meshes.get("global_kd")))
    single_gpu = _parallel({})
    vllm_topology = _mapping(
        _mapping(_mapping(experiment.get("vllm_stats")).get("runtime_stats")).get("topology")
    )
    vllm_parallel = _serving_parallel(vllm_topology)
    gpus_per_node = int(infrastructure.get("gpus_per_node", 8))
    runner = _mapping(infrastructure.get("runner"))
    cpu_partition = str(_mapping(runner.get("slurm")).get("partition_cpu") or "").strip() or None
    pool_workers = int(workers.get("pool", 1))
    sharded_workers = int(workers.get("sharded", 1))
    embedding_widths = list(_mapping(experiment.get("embedding_pruning")).get("widths") or ())
    stages: dict[str, dict[str, Any]] = {
        "convert": {"strategy": "single", "instances": 1, "parallel": single_gpu},
        "tokenize_data": {"strategy": "single", "instances": 1},
        "vllm_stats": {
            "strategy": "sharded",
            "instances": sharded_workers,
            "parallel": vllm_parallel,
        },
        "depth_importance": {
            "strategy": "persistent_pool",
            "instances": pool_workers,
            "parallel": common,
        },
        "width_importance": {"strategy": "single", "instances": 1, "parallel": common},
        "sort": {"strategy": "single", "instances": 1, "parallel": single_gpu},
        "sort_sanity": {"strategy": "single", "instances": 1, "parallel": common},
        "width_sanity": {"strategy": "single", "instances": 1, "parallel": common},
        "slicing_sanity": {"strategy": "single", "instances": 1},
        "bypass_sanity": {"strategy": "single", "instances": 1, "parallel": bypass},
        "bypass": {
            "strategy": "single",
            "instances": 1,
            "parallel": bypass,
        },
        "build_library": {
            "strategy": "single",
            "instances": 1,
            "parallel": single_gpu,
        },
        "replacement_scoring": {
            "strategy": "sharded" if len(embedding_widths) > 1 else "persistent_pool",
            "instances": sharded_workers if len(embedding_widths) > 1 else pool_workers,
            "parallel": common,
        },
        "mip": {"strategy": "single", "instances": 1},
    }
    stages.update(
        _dynamic_stage_entries(
            experiment,
            workers,
            gpus_per_node,
            common,
            single_gpu,
            cpu_partition,
            pool_source_evaluations=not bool(state.get("detailed", False)),
        )
    )
    cpu_stage_ids = {"convert", "tokenize_data", "build_library", "mip"}
    if cpu_partition:
        for stage_id in cpu_stage_ids:
            stages[stage_id].update(resource="cpu", partition=cpu_partition)
    for stage_id, stage in stages.items():
        stage.setdefault("gpus_per_node", gpus_per_node)
        if "short_kd" in stage_id or "global_kd" in stage_id:
            stage["parallel"] = global_kd
    return {
        "execution": {
            "defaults": {
                "failure_policy": "strict",
                "halt_policy": "drain",
                "gpus_per_node": gpus_per_node,
            },
            "stages": stages,
        }
    }


def _bundle_paths(bundle_dir: Path) -> tuple[Path, Path, Path]:
    return (
        bundle_dir / "experiment.yaml",
        bundle_dir / "runner.yaml",
        bundle_dir / "execution.yaml",
    )


def _compile_bundle(bundle_dir: Path):
    experiment_path, runner_path, execution_path = _bundle_paths(bundle_dir)
    runner = load_runner_config(runner_path)
    execution = load_execution_config(execution_path)
    return compile_campaign_plan(
        experiment_config_path=experiment_path,
        runner=runner,
        execution=execution,
        stage_filter="full",
    )


def _submission_payload(submission) -> dict[str, Any]:
    return {
        "stage_id": submission.stage_id,
        "work_id": submission.work_id,
        "attempt_id": submission.attempt_id,
        "nodes": submission.nodes,
        "gpus": submission.gpus,
        "gpus_per_node": submission.gpus_per_node,
        "task_count": submission.task_count,
        "gpus_per_task": submission.gpus_per_task,
        "tasks_per_group": submission.tasks_per_group,
        "group_count": submission.group_count,
        "task_capacity": submission.task_capacity,
        "unused_gpus": submission.unused_gpus,
        "launcher": submission.launcher,
        "exclusive": submission.exclusive,
        "argv": list(submission.argv),
    }


def dry_run_bundle(bundle_dir: Path) -> str:
    """Compile a bundle and serialize its no-submission execution plan."""
    plan = _compile_bundle(bundle_dir)
    submissions = dry_run_plan(plan)
    lines = [
        "Puzzletron orchestration dry run — no jobs submitted",
        f"Stages: {len(plan.stages)}",
        f"Submissions: {len(submissions)}",
        "",
    ]
    for stage in plan.stages:
        count = sum(item.stage_id == stage.stage_id for item in submissions)
        lines.append(
            f"{stage.stage_id}: {count} submission(s), strategy={stage.strategy.value}, "
            f"gpus_per_instance={stage.gpus_per_instance}, nodes={stage.nodes}"
        )
    lines.extend(["", json.dumps([_submission_payload(item) for item in submissions], indent=2)])
    return "\n".join(lines) + "\n"


def validate_bundle(bundle_dir: Path) -> BundleValidation:
    """Validate and dry-run one bundle without constructing an executor."""
    try:
        plan = _compile_bundle(bundle_dir)
        submissions = dry_run_plan(plan)
        return BundleValidation(
            valid=True,
            stage_count=len(plan.stages),
            submission_count=len(submissions),
        )
    except Exception as error:
        return BundleValidation(valid=False, error=f"{type(error).__name__}: {error}")


def _readme(
    campaign_dir: Path,
    state: Mapping[str, Any],
    validations: Mapping[str, BundleValidation],
) -> str:
    model = _mapping(state.get("model"))
    inventory = _mapping(state.get("inventory"))
    data = _answers(state, "data")
    mip = _answers(state, "mip")
    infrastructure = _answers(state, "infrastructure")
    repository = str(_mapping(infrastructure.get("execution_contract")).get("repository", "."))
    orchestrator = Path(repository) / "examples/puzzletron/orchestrate.py"
    rows = infrastructure.get("resource_rows") or ()
    lines = [
        "# Generated Puzzletron campaign",
        "",
        f"- Model: `{model.get('source')}`",
        f"- Resolved revision: `{model.get('resolved_revision') or 'local'}`",
        f"- Profile: `{inventory.get('family')} / {inventory.get('descriptor')}`",
        f"- Dataset modality: `{data.get('modality')}` ({data.get('modality_evidence')})",
        f"- MIP runs: `{', '.join(_mapping(mip.get('runs')))}`",
        "",
        "Run the commands below from this campaign directory.",
        "Smoke and production are independent. Running smoke is recommended but never gates production.",
        "The setup wizard did not submit any jobs.",
        "",
        "## Resource summary",
        "",
        "| Stage group | Instances | GPUs/instance | Nodes |",
        "| --- | ---: | ---: | ---: |",
    ]
    lines.extend(
        (f"| {row['stage']} | {row['instances']} | {row['gpus_per_instance']} | {row['nodes']} |")
        for row in rows
    )
    for budget in ("smoke", "production"):
        validation = validations[budget]
        status = (
            f"valid ({validation.stage_count} stages, {validation.submission_count} submissions)"
            if validation.valid
            else f"invalid: {validation.error}"
        )
        bundle = Path(budget)
        lines.extend(
            [
                "",
                f"## {budget.title()}",
                "",
                f"Validation: **{status}**",
                "",
                "```bash",
                f"python {orchestrator} \\",
                f"  --experiment {bundle / 'experiment.yaml'} \\",
                f"  --runner {bundle / 'runner.yaml'} \\",
                f"  --execution {bundle / 'execution.yaml'} --stage full --dry-run",
                "",
                f"python {orchestrator} \\",
                f"  --experiment {bundle / 'experiment.yaml'} \\",
                f"  --runner {bundle / 'runner.yaml'} \\",
                f"  --execution {bundle / 'execution.yaml'} --stage full",
                "```",
            ]
        )
    lines.extend(
        [
            "",
            "## Resume setup",
            "",
            "```bash",
            f"python {Path(repository) / 'examples/puzzletron/puzzletron_setup.py'} --resume .",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def build_bundles(campaign_dir: Path, state: Mapping[str, Any]) -> BundleResult:
    """Generate, independently validate, and dry-run smoke and production bundles."""
    campaign_dir = campaign_dir.expanduser().resolve()
    for budget in ("smoke", "production"):
        bundle_dir = campaign_dir / budget
        experiment = render_experiment(state, budget)
        runner = render_runner(state, budget)
        execution = render_execution(state, experiment, budget)
        experiment_path, runner_path, execution_path = _bundle_paths(bundle_dir)
        _atomic_yaml(experiment_path, experiment)
        _atomic_yaml(runner_path, runner)
        _atomic_yaml(execution_path, execution)

    validations = {
        budget: validate_bundle(campaign_dir / budget) for budget in ("smoke", "production")
    }
    for budget, validation in validations.items():
        plan_path = campaign_dir / budget / "dry-run-plan.txt"
        if validation.valid:
            _atomic_text(plan_path, dry_run_bundle(campaign_dir / budget))
            print(
                f"{budget.title()} bundle valid: {validation.stage_count} stages, "
                f"{validation.submission_count} submissions."
            )
        else:
            _atomic_text(
                plan_path,
                "Puzzletron orchestration dry run failed — no jobs submitted\n"
                f"{validation.error}\n",
            )
            print(f"{budget.title()} bundle validation failed: {validation.error}")
    _atomic_text(campaign_dir / "README.md", _readme(campaign_dir, state, validations))
    return BundleResult(
        campaign_dir=campaign_dir,
        smoke=validations["smoke"],
        production=validations["production"],
    )
