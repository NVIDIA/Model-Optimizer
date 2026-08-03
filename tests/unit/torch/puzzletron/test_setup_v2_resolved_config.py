# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolved-config tests with deliberately copied, frozen legacy render oracles."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

import puzzletron_setup.v2.bundle as bundle_module
from puzzletron_setup.bundle import render_execution, render_experiment, render_runner
from puzzletron_setup.v2.bundle import (
    _legacy_state_from_resolved,
    _render_experiment_v2,
    build_bundles_v2,
    render_execution_v2,
    render_experiment_v2,
    render_runner_v2,
)
from puzzletron_setup.v2.resolved import CompatibilityProjection, resolve_campaign_config
from puzzletron_setup.v2.state import WizardState
from puzzletron_setup.v2.validation import validate_state

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]


def _ordered(value, *, reverse: bool):
    items = list(value.items())
    return dict(reversed(items)) if reverse else dict(items)


def _state(tmp_path: Path, *, reverse: bool = False) -> WizardState:
    state = WizardState.start(tmp_path / ("reverse" if reverse else "campaign"), defaults_path=None)
    inventory = {
        "family": "qwen3_5",
        "descriptor": "qwen3_5",
        "family_config": "examples/puzzletron/configs/families/qwen3_5/family.yaml",
        "model_type": "qwen3_5",
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "multimodal": True,
        "moe": False,
        "num_layers": 4,
        "num_sublayers": 8,
        "layer_counts": {"full_attention": 4},
        "facts": {
            "hidden_size": 1024,
            "num_attention_heads": 16,
            "num_key_value_heads": 4,
            "intermediate_size": 4096,
        },
        "axes": [
            {
                "axis_id": "hidden_width",
                "label": "Hidden width",
                "teacher_value": 1024,
                "values": [1024, 768],
                "alignment": 256,
            }
        ],
    }
    model = {
        "source": "Qwen/Qwen3.5-Test",
        "requested_revision": "main",
        "resolved_revision": "0123456789abcdef",
        "is_local": False,
        "config": {
            "model_type": "qwen3_5",
            "text_config": {
                "num_hidden_layers": 4,
                "hidden_size": 1024,
                "layer_types": ["full_attention"] * 4,
            },
        },
        "inventory": inventory,
    }
    state.set_model(_ordered(model, reverse=reverse), _ordered(inventory, reverse=reverse))

    fields = [
        ("model.source", "Qwen/Qwen3.5-Test", "user", None, None),
        ("data.source", "/datasets/text", "user", None, None),
        ("data.selected_source", "org/text", "user", None, None),
        ("data.adapter", "custom", "builtin", None, None),
        ("data.modality", "text", "inferred", None, None),
        ("data.layout", "fixed", "builtin", None, None),
        ("data.sequence_length", 2048, "defaults_file", None, None),
        (
            "infrastructure.execution_contract.repository",
            str(REPOSITORY_ROOT),
            "user",
            None,
            None,
        ),
        ("infrastructure.execution_contract.venv", ".venv", "builtin", None, None),
        ("infrastructure.execution_contract.container", None, "builtin", None, None),
        (
            "infrastructure.execution_contract.container_mounts",
            None,
            "builtin",
            None,
            None,
        ),
        (
            "infrastructure.execution_contract.prerun_commands",
            ["module load cuda"],
            "user",
            None,
            None,
        ),
        (
            "infrastructure.execution_contract.postrun_commands",
            [],
            "builtin",
            None,
            None,
        ),
        ("infrastructure.runner.kind", "slurm", "builtin", None, None),
        ("infrastructure.runner.slurm.account", "account", "user", None, None),
        (
            "infrastructure.runner.slurm.partition_interactive",
            "interactive",
            "builtin",
            None,
            None,
        ),
        (
            "infrastructure.runner.slurm.partition_batch",
            "batch",
            "builtin",
            None,
            None,
        ),
        ("infrastructure.runner.slurm.partition_cpu", "cpu", "user", None, None),
        ("infrastructure.runner.slurm.time_limit", "4:00:00", "builtin", None, None),
        ("infrastructure.runner.slurm.qos", None, "builtin", None, None),
        ("infrastructure.runner.slurm.max_nodes", 64, "builtin", None, None),
        ("infrastructure.gpus_per_node", 8, "builtin", None, None),
        ("stages.width_importance.batch", 4, "preset", 3, 4),
        ("output.result_root", "/results", "user", None, None),
    ]
    for path, value, source, requested, effective in reversed(fields) if reverse else fields:
        state.set_field(
            path,
            value,
            source=source,
            requested=requested,
            effective=effective,
        )

    pruning = {
        "width_importance_samples": 256,
        "replacement_samples": 32,
        "depth_remove": 0,
        "axes": {
            "hidden_width": {
                "enabled": True,
                "teacher_value": 1024,
                "values": [1024, 768],
                "alignment": 256,
            }
        },
        "bypass": {"enabled": False},
    }
    workloads = {
        "latency-first": {
            "prefill_seq_len": 2048,
            "generation_seq_len": 256,
            "batch_size": 2,
            "max_num_seqs": 2,
        },
        "throughput-second": {
            "prefill_seq_len": 4096,
            "generation_seq_len": 512,
            "batch_size": 4,
            "max_num_seqs": 4,
        },
    }
    measurements = {
        "latency-first": {
            "prefill_seq_len": 2048,
            "generation_seq_len": 256,
            "batch_size": 2,
            "max_num_seqs": 2,
            "granularity": "block",
            "runtime_stats": {"max_num_seqs": 2},
        },
        "throughput-second": {
            "prefill_seq_len": 4096,
            "generation_seq_len": 512,
            "batch_size": 4,
            "max_num_seqs": 4,
            "granularity": "subblock",
            "runtime_stats": {"max_num_seqs": 4},
        },
    }
    profiles = {
        "model": {
            "name": "model",
            "tp": 2,
            "cp": 1,
            "pp": 1,
            "dp_shard": 1,
            "dp_replicate": 1,
            "ep": 1,
            "sequence_parallel": True,
            "consumers": ["width_importance"],
        },
        "single": {
            "name": "single",
            "tp": 1,
            "cp": 1,
            "pp": 1,
            "dp_shard": 1,
            "dp_replicate": 1,
            "ep": 1,
            "sequence_parallel": False,
            "consumers": [],
        },
    }
    collections = {
        "data_acquisition": {},
        "data_subset_selection": {},
        "pruning": _ordered(pruning, reverse=reverse),
        "serving_workloads": workloads,
        "vllm_measurements": measurements,
        "mip_config": {"runs": {}, "marker": "named"},
        "post_mip_flows": {"selection": {"source": {"run": "default"}, "nodes": {}}},
        "parallel_profiles": profiles,
        "stage_resources": {
            "width_importance": {
                "strategy": "single",
                "instances": 1,
                "resource": "gpu",
                "gpus_per_node": 8,
                "profile_name": "model",
            },
            "custom": {
                "strategy": "sharded",
                "instances": 3,
                "resource": "gpu",
                "gpus_per_node": 8,
                "parallel": {"tp": 1, "cp": 1, "pp": 1},
            },
        },
        "stage_batches": {"pruning.micro_batch_size": 7},
        "experiment_overrides": {
            "compatibility_marker": "kept",
            "vllm_stats": {"prefill_seq_len": 999},
            "mip": {"marker": "compatibility"},
            "post_mip": {"marker": "compatibility"},
            "pruning": {"micro_batch_size": 3},
            "width_importance": {"automodel": {"parallel": {"tp": 99}}},
        },
        "runner_overrides": {"runner": {"slurm": {"partition": "late-override"}}},
        "default_resolutions": _ordered(
            {
                "pruning.depth_remove": {"value": 0, "source": "preset"},
                "mip.num_solutions": {"value": 8, "source": "defaults_file"},
            },
            reverse=reverse,
        ),
    }
    for name, value in collections.items():
        state.set_collection(name, value)
    return state


def _mapping(value):
    return dict(value) if isinstance(value, Mapping) else {}


def _deep_merge(base, update):
    merged = deepcopy(dict(base))
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _set_dotted(config, dotted, value):
    current = config
    parts = dotted.split(".")
    for part in parts[:-1]:
        current = current.setdefault(part, {})
    current[parts[-1]] = deepcopy(value)


def _legacy_projection(state: WizardState) -> dict:
    workloads = _mapping(state.collection("serving_workloads"))
    measurements = _mapping(state.collection("vllm_measurements"))
    workload_id, workload = next(iter(workloads.items()), ("serving-default", {}))
    measurement = _mapping(measurements.get(workload_id))
    if not measurement and measurements:
        measurement = _mapping(next(iter(measurements.values())))
    profiles = _mapping(state.collection("parallel_profiles"))
    mesh = {key: 1 for key in ("tp", "cp", "pp", "dp_shard", "dp_replicate", "ep")}
    if profiles:
        first = next(iter(profiles.values()))
        mesh = {key: first.get(key, 1) for key in mesh}
    infrastructure = {
        "runner": {
            "kind": state.get_field("infrastructure.runner.kind", "slurm"),
            "slurm": {
                "account": state.get_field("infrastructure.runner.slurm.account", ""),
                "partition_interactive": state.get_field(
                    "infrastructure.runner.slurm.partition_interactive", "interactive"
                ),
                "partition_batch": state.get_field(
                    "infrastructure.runner.slurm.partition_batch", "batch"
                ),
                "partition_cpu": state.get_field("infrastructure.runner.slurm.partition_cpu"),
                "time_limit": state.get_field(
                    "infrastructure.runner.slurm.time_limit", "4:00:00"
                ),
                "qos": state.get_field("infrastructure.runner.slurm.qos"),
                "max_nodes": state.get_field("infrastructure.runner.slurm.max_nodes", 64),
            },
        },
        "execution_contract": {
            key: state.get_field(f"infrastructure.execution_contract.{key}", default)
            for key, default in {
                "repository": str(Path.cwd()),
                "venv": ".venv",
                "container": None,
                "container_mounts": None,
                "prerun_commands": [],
                "postrun_commands": [],
            }.items()
        },
        "gpus_per_node": state.get_field("infrastructure.gpus_per_node", 8),
        "meshes": {
            "common": deepcopy(mesh),
            "bypass": deepcopy(mesh),
            "global_kd": deepcopy(mesh),
        },
        "workers": {
            "pool": state.get_field("infrastructure.gpus_per_node", 8),
            "sharded": state.get_field("infrastructure.gpus_per_node", 8),
        },
    }
    selection = _mapping(state.collection("data_subset_selection"))
    subset_records = [_mapping(item) for item in selection.get("subsets") or ()]
    return {
        "schema_version": 1,
        "wizard_version": "1",
        "detailed": True,
        "model": deepcopy(state.payload["model"]),
        "inventory": deepcopy(state.payload["inventory"]),
        "answers": {
            "data": {
                "source": state.get_field("data.source"),
                "selected_source": state.get_field(
                    "data.selected_source", state.get_field("data.source")
                ),
                "adapter": state.get_field("data.adapter", "custom"),
                "modality": state.get_field("data.modality", "text"),
                "layout": state.get_field("data.layout", "fixed"),
                "sequence_length": state.get_field("data.sequence_length", 4096),
                "subsets": [record["name"] for record in subset_records],
                "subset_revision": selection.get("revision"),
                "subset_weights": {
                    record["name"]: record["weight"] for record in subset_records
                },
                "acquisition": deepcopy(state.collection("data_acquisition") or {}),
            },
            "pruning": deepcopy(state.collection("pruning") or {}),
            "runtime": {
                "vllm_enabled": bool(measurements),
                "granularity": measurement.get("granularity", "subblock"),
                "workload_id": workload_id,
                "isl": int(
                    workload.get("prefill_seq_len", state.get_field("data.sequence_length", 4096))
                ),
                "osl": int(workload.get("generation_seq_len", 1024)),
                "concurrency": int(workload.get("max_num_seqs", 1)),
            },
            "mip": deepcopy(state.collection("mip_config") or {"runs": {}}),
            "post_mip": {"flows": deepcopy(state.collection("post_mip_flows") or {})},
            "infrastructure": infrastructure,
            "output": {"result_root": state.get_field("output.result_root")},
        },
    }


def _old_render_experiment_v2(state: WizardState, budget: str) -> dict:
    rendered = render_experiment(_legacy_projection(state), budget)
    rendered = _deep_merge(rendered, _mapping(state.collection("experiment_overrides")))
    measurements = _mapping(state.collection("vllm_measurements"))
    if measurements:
        rendered.setdefault("vllm_stats", {})["measurements"] = deepcopy(measurements)
        rendered["vllm_stats"]["enabled"] = True
        first = _mapping(next(iter(measurements.values())))
        rendered["vllm_stats"].update(
            {
                "prefill_seq_len": int(first.get("prefill_seq_len", 4096)),
                "generation_seq_len": int(first.get("generation_seq_len", 1024)),
                "batch_sizes": [int(first.get("batch_size", 1))],
            }
        )
        rendered["vllm_stats"].setdefault("runtime_stats", {}).update(
            deepcopy(_mapping(first.get("runtime_stats")))
        )
    mip = state.collection("mip_config")
    if isinstance(mip, Mapping):
        rendered["mip"] = _deep_merge(rendered.get("mip") or {}, mip)
        rendered["mip"]["enabled"] = True
    flows = state.collection("post_mip_flows")
    if isinstance(flows, Mapping):
        rendered["post_mip"] = {"flows": deepcopy(dict(flows))}
    for dotted, value in _mapping(state.collection("stage_batches")).items():
        _set_dotted(rendered, str(dotted), value)
    resources = _mapping(state.collection("stage_resources"))
    profiles = _mapping(state.collection("parallel_profiles"))
    parallel_paths = {
        "depth_importance": "depth_importance.automodel.parallel",
        "width_importance": "pruning.automodel.parallel",
        "sort_sanity": "sort_sanity.automodel.parallel",
        "width_sanity": "width_sanity.automodel.parallel",
        "bypass": "bypass.automodel.parallel",
        "replacement_scoring": "replacement_scoring.automodel.parallel",
    }
    for stage_id, dotted in parallel_paths.items():
        profile_name = _mapping(resources.get(stage_id)).get("profile_name")
        profile = _mapping(profiles.get(str(profile_name)))
        if profile:
            _set_dotted(
                rendered,
                dotted,
                {
                    key: profile[key]
                    for key in (
                        "tp",
                        "cp",
                        "pp",
                        "ep",
                        "dp_shard",
                        "dp_replicate",
                        "sequence_parallel",
                    )
                    if key in profile
                },
            )
    return rendered


def _old_render_runner_v2(state: WizardState, budget: str) -> dict:
    return _deep_merge(
        render_runner(_legacy_projection(state), budget),
        _mapping(state.collection("runner_overrides")),
    )


def _old_render_execution_v2(state: WizardState, budget: str) -> dict:
    experiment = _old_render_experiment_v2(state, budget)
    rendered = render_execution(_legacy_projection(state), experiment, budget)
    stages = rendered["execution"]["stages"]
    resources = _mapping(state.collection("stage_resources"))
    profiles = _mapping(state.collection("parallel_profiles"))
    default_gpus = state.get_field("infrastructure.gpus_per_node", 8)
    for stage_id, raw in resources.items():
        resource = _mapping(raw)
        entry = {
            "strategy": str(resource.get("strategy", "single")),
            "instances": int(resource.get("instances", 1)),
            "resource": str(resource.get("resource", "gpu")),
            "gpus_per_node": int(resource.get("gpus_per_node", default_gpus)),
        }
        if resource.get("partition"):
            entry["partition"] = str(resource["partition"])
        profile_name = resource.get("profile_name")
        if profile_name:
            profile = _mapping(profiles.get(str(profile_name)))
            entry["parallel"] = {
                key: profile[key]
                for key in (
                    "tp",
                    "cp",
                    "pp",
                    "ep",
                    "dp_shard",
                    "dp_replicate",
                    "sequence_parallel",
                )
                if key in profile
            }
        elif isinstance(resource.get("parallel"), Mapping):
            entry["parallel"] = deepcopy(dict(resource["parallel"]))
        stages[str(stage_id)] = entry
    return rendered


def test_resolved_identity_is_order_and_yaml_format_stable(tmp_path: Path) -> None:
    state = _state(tmp_path)
    reverse = _state(tmp_path, reverse=True)
    config = resolve_campaign_config(state)
    reverse_config = resolve_campaign_config(reverse)
    reformatted_state = _state(tmp_path / "reformatted")
    reformatted_payload = yaml.safe_load(reformatted_state.path.read_text())
    reformatted_state.path.write_text(
        yaml.safe_dump(reformatted_payload, default_flow_style=True, sort_keys=True)
    )
    reformatted_config = resolve_campaign_config(WizardState.resume(reformatted_state.path))

    assert config.semantic_digest == reverse_config.semantic_digest
    assert config.semantic_digest == reformatted_config.semantic_digest
    assert config.provenance_digest == reverse_config.provenance_digest
    assert config.provenance_digest == reformatted_config.provenance_digest
    assert config.model.facts_digest == reverse_config.model.facts_digest
    assert len(config.semantic_digest) == 64
    assert config.compatibility_projection == CompatibilityProjection(
        workload_id="latency-first",
        first_measurement_id="latency-first",
        runtime_measurement_id="latency-first",
        first_parallel_profile_name="model",
    )
    assert reverse_config.compatibility_projection == config.compatibility_projection


def test_resolved_provenance_and_qwen_identity_are_complete(tmp_path: Path) -> None:
    config = resolve_campaign_config(_state(tmp_path))

    assert config.provenance["stages.width_importance.batch"].requested == 3
    assert config.provenance["stages.width_importance.batch"].effective == 4
    assert config.provenance["mip.num_solutions"].source == "defaults_file"
    assert config.model.requested_revision == "main"
    assert config.model.resolved_revision == "0123456789abcdef"
    assert config.model.facts["hidden_size"] == 1024
    assert config.model.descriptor == "qwen3_5"
    assert config.model.model_type == "qwen3_5"
    assert config.model.multimodal is True
    assert config.data.modality == "text"


def test_resolved_snapshot_is_deeply_immutable(tmp_path: Path) -> None:
    state = _state(tmp_path)
    config = resolve_campaign_config(state)
    semantic_digest = config.semantic_digest
    provenance_digest = config.provenance_digest
    state.payload["model"]["config"]["text_config"]["hidden_size"] = 4096
    pruning = deepcopy(state.collection("pruning"))
    pruning["axes"]["hidden_width"]["values"].append(512)
    state.set_collection("pruning", pruning)
    state.set_field("data.sequence_length", 8192, source="user")

    assert config.semantic_digest == semantic_digest
    assert config.provenance_digest == provenance_digest
    assert config.model.config["text_config"]["hidden_size"] == 1024
    assert config.pruning["axes"]["hidden_width"]["values"] == (1024, 768)
    with pytest.raises(TypeError):
        config.pruning["depth_remove"] = 2
    with pytest.raises(TypeError):
        config.model.config["text_config"]["hidden_size"] = 2048
    with pytest.raises(FrozenInstanceError):
        config.compatibility_projection.workload_id = "throughput-second"


def test_semantic_digest_tracks_resolved_values_and_compatibility(tmp_path: Path) -> None:
    config = resolve_campaign_config(_state(tmp_path))
    semantic_digest = config.semantic_digest
    override_state = _state(tmp_path / "override")
    override_state.set_collection("experiment_overrides", {"changed": True})
    assert resolve_campaign_config(override_state).semantic_digest != semantic_digest
    value_state = _state(tmp_path / "value")
    value_state.set_field("data.sequence_length", 4096, source="user")
    assert resolve_campaign_config(value_state).semantic_digest != semantic_digest


def test_result_root_is_location_only_semantics(tmp_path: Path) -> None:
    config = resolve_campaign_config(_state(tmp_path))
    state = _state(tmp_path / "different")
    state.set_field("output.result_root", "/different/results", source="user")
    path_config = resolve_campaign_config(state)

    assert path_config.semantic_digest == config.semantic_digest
    assert path_config.provenance_digest != config.provenance_digest


@pytest.mark.parametrize(
    ("path", "value"),
    [
        ("infrastructure.execution_contract.repository", "/different/repository"),
        ("infrastructure.execution_contract.venv", "/different/venv"),
        (
            "infrastructure.execution_contract.container_mounts",
            "/different:/workspace",
        ),
    ],
)
def test_execution_contract_fields_independently_change_semantics(
    tmp_path: Path,
    path: str,
    value: str,
) -> None:
    config = resolve_campaign_config(_state(tmp_path))
    state = _state(tmp_path / "different")
    state.set_field(path, value, source="user")

    changed = resolve_campaign_config(state)

    assert changed.semantic_digest != config.semantic_digest


def test_model_identity_separates_requested_alias_from_resolved_facts(tmp_path: Path) -> None:
    config = resolve_campaign_config(_state(tmp_path))

    alias_state = _state(tmp_path / "alias")
    alias_state.payload["model"]["requested_revision"] = "latest"
    alias_state.save()
    alias_config = resolve_campaign_config(alias_state)
    assert alias_config.semantic_digest == config.semantic_digest
    assert alias_config.model.facts_digest == config.model.facts_digest
    assert alias_config.provenance_digest != config.provenance_digest

    resolved_state = _state(tmp_path / "resolved")
    resolved_state.payload["model"]["resolved_revision"] = "fedcba9876543210"
    resolved_state.save()
    resolved_config = resolve_campaign_config(resolved_state)
    assert resolved_config.semantic_digest != config.semantic_digest
    assert resolved_config.model.facts_digest != config.model.facts_digest

    facts_state = _state(tmp_path / "facts")
    facts_state.payload["inventory"]["facts"]["hidden_size"] = 2048
    facts_state.save()
    facts_config = resolve_campaign_config(facts_state)
    assert facts_config.semantic_digest != config.semantic_digest
    assert facts_config.model.facts_digest != config.model.facts_digest


def test_compatibility_projection_preserves_reordered_mapping_rendering(
    tmp_path: Path,
) -> None:
    config = resolve_campaign_config(_state(tmp_path))
    base_render = _render_experiment_v2(config, "production")
    reordered = replace(
        config,
        serving_workloads=dict(reversed(list(config.serving_workloads.items()))),
        vllm_measurements=dict(reversed(list(config.vllm_measurements.items()))),
        parallel_profiles=dict(reversed(list(config.parallel_profiles.items()))),
    )

    assert reordered.compatibility_projection == config.compatibility_projection
    assert reordered.semantic_digest == config.semantic_digest
    assert _render_experiment_v2(reordered, "production") == base_render


def test_projection_workload_id_independently_controls_rendering(tmp_path: Path) -> None:
    config = resolve_campaign_config(_state(tmp_path))
    changed = replace(
        config,
        compatibility_projection=replace(
            config.compatibility_projection,
            workload_id="throughput-second",
        ),
    )
    base_render = _render_experiment_v2(config, "production")
    changed_render = _render_experiment_v2(changed, "production")

    assert changed.semantic_digest != config.semantic_digest
    assert tuple(base_render["mip"]["workloads"]) == ("latency-first",)
    assert tuple(changed_render["mip"]["workloads"]) == ("throughput-second",)


def test_projection_first_measurement_independently_controls_rendering(
    tmp_path: Path,
) -> None:
    config = resolve_campaign_config(_state(tmp_path))
    changed = replace(
        config,
        compatibility_projection=replace(
            config.compatibility_projection,
            first_measurement_id="throughput-second",
        ),
    )
    base_render = _render_experiment_v2(config, "production")
    changed_render = _render_experiment_v2(changed, "production")

    assert changed.semantic_digest != config.semantic_digest
    assert base_render["vllm_stats"]["prefill_seq_len"] == 2048
    assert changed_render["vllm_stats"]["prefill_seq_len"] == 4096


def test_projection_runtime_measurement_independently_controls_rendering(
    tmp_path: Path,
) -> None:
    config = resolve_campaign_config(_state(tmp_path))
    changed = replace(
        config,
        compatibility_projection=replace(
            config.compatibility_projection,
            runtime_measurement_id="throughput-second",
        ),
    )
    base_render = _render_experiment_v2(config, "production")
    changed_render = _render_experiment_v2(changed, "production")

    assert changed.semantic_digest != config.semantic_digest
    assert base_render["vllm_stats"]["runtime_stats"]["granularity"] == "block"
    assert changed_render["vllm_stats"]["runtime_stats"]["granularity"] == "subblock"


def test_projection_first_profile_independently_controls_rendering(tmp_path: Path) -> None:
    config = resolve_campaign_config(_state(tmp_path))
    changed = replace(
        config,
        compatibility_projection=replace(
            config.compatibility_projection,
            first_parallel_profile_name="single",
        ),
    )
    base_render = _render_experiment_v2(config, "production")
    changed_render = _render_experiment_v2(changed, "production")

    assert changed.semantic_digest != config.semantic_digest
    assert base_render["depth_importance"]["automodel"]["parallel"]["tp"] == 2
    assert changed_render["depth_importance"]["automodel"]["parallel"]["tp"] == 1


@pytest.mark.parametrize("budget", ["smoke", "production"])
def test_current_member_payloads_and_precedence_are_preserved(
    tmp_path: Path,
    budget: str,
) -> None:
    state = _state(tmp_path)
    state.set_collection("legacy_state", {"stale": True})

    assert _legacy_state_from_resolved(resolve_campaign_config(state)) == _legacy_projection(state)
    experiment = render_experiment_v2(state, budget)
    assert experiment == _old_render_experiment_v2(state, budget)
    assert render_runner_v2(state, budget) == _old_render_runner_v2(state, budget)
    assert render_execution_v2(state, budget) == _old_render_execution_v2(state, budget)
    assert experiment["compatibility_marker"] == "kept"
    assert experiment["vllm_stats"]["prefill_seq_len"] == 2048
    assert experiment["mip"]["marker"] == "named"
    assert "marker" not in experiment["post_mip"]
    assert experiment["pruning"]["micro_batch_size"] == 7
    assert experiment["pruning"]["automodel"]["parallel"]["tp"] == 2
    assert render_runner_v2(state, budget)["runner"]["slurm"]["partition"] == "late-override"
    assert render_execution_v2(state, budget)["execution"]["stages"]["width_importance"] == {
        "strategy": "single",
        "instances": 1,
        "resource": "gpu",
        "gpus_per_node": 8,
        "parallel": {
            "tp": 2,
            "cp": 1,
            "pp": 1,
            "ep": 1,
            "dp_shard": 1,
            "dp_replicate": 1,
            "sequence_parallel": True,
        },
    }


def test_budget_render_order_is_stable(tmp_path: Path) -> None:
    state = _state(tmp_path)
    smoke_first = (
        render_experiment_v2(state, "smoke"),
        render_runner_v2(state, "smoke"),
        render_execution_v2(state, "smoke"),
    )
    production_second = (
        render_experiment_v2(state, "production"),
        render_runner_v2(state, "production"),
        render_execution_v2(state, "production"),
    )
    production_first = (
        render_experiment_v2(state, "production"),
        render_runner_v2(state, "production"),
        render_execution_v2(state, "production"),
    )
    smoke_second = (
        render_experiment_v2(state, "smoke"),
        render_runner_v2(state, "smoke"),
        render_execution_v2(state, "smoke"),
    )

    assert smoke_first == smoke_second
    assert production_first == production_second


@pytest.mark.parametrize(
    ("path", "value"),
    [
        ("data.modality", None),
        ("data.modality", ""),
        ("data.layout", None),
        ("data.layout", ""),
    ],
)
def test_nullable_data_strings_preserve_legacy_rendering(
    tmp_path: Path,
    path: str,
    value: str | None,
) -> None:
    state = _state(tmp_path)
    state.set_field(path, value, source="user")
    resolved_legacy = _legacy_state_from_resolved(resolve_campaign_config(state))
    data_key = path.removeprefix("data.")

    assert resolved_legacy == _legacy_projection(state)
    assert resolved_legacy["answers"]["data"][data_key] == value
    assert render_experiment_v2(state, "production") == _old_render_experiment_v2(
        state, "production"
    )


def test_zero_stage_gpus_preserve_legacy_execution_rendering(tmp_path: Path) -> None:
    state = _state(tmp_path)
    resources = deepcopy(state.collection("stage_resources"))
    resources["custom"]["gpus_per_node"] = 0
    state.set_collection("stage_resources", resources)

    execution = render_execution_v2(state, "production")

    assert execution == _old_render_execution_v2(state, "production")
    assert execution["execution"]["stages"]["custom"]["gpus_per_node"] == 0


@pytest.mark.parametrize("budget", ["smoke", "production"])
def test_explicit_empty_mip_and_post_mip_mappings_preserve_rendering(
    tmp_path: Path,
    budget: str,
) -> None:
    state = _state(tmp_path)
    state.set_collection("mip_config", {})
    state.set_collection("post_mip_flows", {})

    assert render_experiment_v2(state, budget) == _old_render_experiment_v2(state, budget)


def test_absent_semantic_collections_preserve_compatibility_overrides(tmp_path: Path) -> None:
    state = _state(tmp_path)
    state.payload["collections"].pop("mip_config")
    state.payload["collections"].pop("post_mip_flows")
    state.save()

    experiment = render_experiment_v2(state, "production")

    assert experiment == _old_render_experiment_v2(state, "production")
    assert experiment["mip"]["marker"] == "compatibility"
    assert experiment["post_mip"]["marker"] == "compatibility"


def test_sparse_parallel_profile_preserves_emitted_key_presence(tmp_path: Path) -> None:
    state = _state(tmp_path)
    state.set_collection("parallel_profiles", {"sparse": {"name": "sparse", "tp": 2}})
    resources = deepcopy(state.collection("stage_resources"))
    resources["width_importance"]["profile_name"] = "sparse"
    state.set_collection("stage_resources", resources)

    assert render_experiment_v2(state, "production") == _old_render_experiment_v2(
        state, "production"
    )
    assert render_execution_v2(state, "production") == _old_render_execution_v2(
        state, "production"
    )


def test_empty_parallel_profile_preserves_accepted_legacy_rendering(tmp_path: Path) -> None:
    state = _state(tmp_path)
    state.set_collection("parallel_profiles", {"empty": {}})
    resources = deepcopy(state.collection("stage_resources"))
    resources["width_importance"]["profile_name"] = "empty"
    state.set_collection("stage_resources", resources)

    assert validate_state(state) == ()
    config = resolve_campaign_config(state)
    assert config.parallel_profiles["empty"].source_nonempty is False
    configured_profile = replace(config.parallel_profiles["empty"], source_nonempty=True)
    configured = replace(config, parallel_profiles={"empty": configured_profile})
    assert configured.semantic_digest != config.semantic_digest

    experiment = render_experiment_v2(state, "production")
    execution = render_execution_v2(state, "production")

    assert experiment == _old_render_experiment_v2(state, "production")
    assert execution == _old_render_execution_v2(state, "production")
    assert experiment["pruning"]["automodel"]["parallel"] == {
        "tp": 1,
        "cp": 1,
        "pp": 1,
        "ep": 1,
        "dp_shard": 1,
        "dp_replicate": 1,
        "sequence_parallel": False,
    }
    assert execution["execution"]["stages"]["width_importance"]["parallel"] == {}


def test_build_resolves_once_and_replaces_stale_legacy_projection(
    tmp_path: Path,
    monkeypatch,
) -> None:
    state = _state(tmp_path)
    expected_legacy = _legacy_projection(state)
    state.set_collection("legacy_state", {"stale": True})
    calls = 0
    resolver = resolve_campaign_config

    def counted_resolve(current_state):
        nonlocal calls
        calls += 1
        return resolver(current_state)

    monkeypatch.setattr(bundle_module, "resolve_campaign_config", counted_resolve)
    monkeypatch.setattr(
        bundle_module,
        "validate_bundle",
        lambda path: SimpleNamespace(valid=True, error=None),
    )
    monkeypatch.setattr(bundle_module, "dry_run_bundle", lambda path: "dry run\n")

    build_bundles_v2(state.campaign_dir, state)

    assert calls == 1
    assert state.collection("legacy_state") == expected_legacy
    provenance = yaml.safe_load((state.campaign_dir / "resolved_defaults.yaml").read_text())
    assert provenance["stages.width_importance.batch"] == {
        "value": 4,
        "requested": 3,
        "effective": 4,
        "source": "preset",
    }
