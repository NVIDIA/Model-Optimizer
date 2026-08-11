# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Behavioral tests for setup-v2 resolved snapshots and bundle rendering."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

import puzzletron_setup.v2.bundle as bundle_module
from puzzletron_setup.v2.bundle import (
    build_bundles_v2,
    render_execution_v2,
    render_experiment_v2,
    render_runner_v2,
)
from puzzletron_setup.v2.resolved import resolve_campaign_config
from puzzletron_setup.v2.state import WizardState

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]


def _campaign_state(tmp_path: Path) -> WizardState:
    """Create the smallest valid state that exercises resolved rendering boundaries."""
    state = WizardState.start(tmp_path / "campaign", defaults_path=None)
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
    }
    model = {
        "source": "Qwen/Qwen3.5-Test",
        "requested_revision": "main",
        "resolved_revision": "0123456789abcdef",
        "is_local": False,
        "config": {
            "model_type": "qwen3_5",
            "text_config": {"hidden_size": 1024},
        },
    }
    state.set_model(model, inventory)

    fields = {
        "model.source": "Qwen/Qwen3.5-Test",
        "data.source": "/datasets/text",
        "data.selected_source": "org/text",
        "data.adapter": "custom",
        "data.modality": "text",
        "data.layout": "fixed",
        "data.sequence_length": 2048,
        "infrastructure.execution_contract.repository": str(REPOSITORY_ROOT),
        "infrastructure.execution_contract.venv": ".venv",
        "infrastructure.runner.kind": "slurm",
        "infrastructure.runner.slurm.account": "account",
        "infrastructure.runner.slurm.partition_batch": "batch",
        "infrastructure.runner.slurm.partition_cpu": "cpu",
        "infrastructure.gpus_per_node": 8,
        "output.result_root": "/results",
    }
    for path, value in fields.items():
        state.set_field(path, value, source="test")
    state.set_field(
        "stages.width_importance.batch",
        4,
        source="preset",
        requested=3,
        effective=4,
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
    }
    collections = {
        "data_acquisition": {},
        "data_subset_selection": {},
        "pruning": pruning,
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
        },
        "stage_batches": {"pruning.micro_batch_size": 7},
        "experiment_overrides": {
            "compatibility_marker": "kept",
            "vllm_stats": {"prefill_seq_len": 999},
            "mip": {"marker": "compatibility"},
            "post_mip": {"marker": "compatibility"},
            "pruning": {
                "micro_batch_size": 3,
                "automodel": {"parallel": {"tp": 99}},
            },
        },
        "runner_overrides": {"runner": {"slurm": {"partition": "late-override"}}},
        "default_resolutions": {
            "pruning.depth_remove": {"value": 0, "source": "preset"},
            "mip.num_solutions": {"value": 8, "source": "defaults_file"},
            "stages.width_importance.batch": {
                "value": 100,
                "requested": 90,
                "effective": 100,
                "source": "defaults_file",
            },
        },
    }
    for name, value in collections.items():
        state.set_collection(name, value)
    return state


def test_snapshot_captures_effective_values_and_authoring_provenance(tmp_path: Path) -> None:
    state = _campaign_state(tmp_path)

    snapshot = resolve_campaign_config(state)

    batch = snapshot.provenance["stages.width_importance.batch"]
    assert (batch.requested, batch.effective, batch.source) == (3, 4, "preset")
    assert snapshot.provenance["mip.num_solutions"].source == "defaults_file"
    assert snapshot.model.resolved_revision == "0123456789abcdef"
    assert snapshot.model.facts["hidden_size"] == 1024
    assert snapshot.model.descriptor == "qwen3_5"
    assert snapshot.data.modality == "text"


def test_snapshot_is_isolated_from_later_wizard_state_mutations(tmp_path: Path) -> None:
    state = _campaign_state(tmp_path)
    snapshot = resolve_campaign_config(state)

    state.payload["model"]["config"]["text_config"]["hidden_size"] = 4096
    pruning = deepcopy(state.collection("pruning"))
    pruning["axes"]["hidden_width"]["values"].append(512)
    state.set_collection("pruning", pruning)
    state.set_field("data.sequence_length", 8192, source="user")

    assert snapshot.model.config["text_config"]["hidden_size"] == 1024
    assert snapshot.pruning["axes"]["hidden_width"]["values"] == (1024, 768)
    assert snapshot.data.sequence_length == 2048


def test_snapshot_rejects_direct_mutation(tmp_path: Path) -> None:
    snapshot = resolve_campaign_config(_campaign_state(tmp_path))

    with pytest.raises(TypeError):
        snapshot.pruning["depth_remove"] = 2
    with pytest.raises(FrozenInstanceError):
        snapshot.compatibility_projection.workload_id = "throughput-second"


def test_renderer_uses_requested_revision_until_model_revision_is_pinned(tmp_path: Path) -> None:
    state = _campaign_state(tmp_path)
    state.payload["model"]["requested_revision"] = "candidate"
    state.payload["model"]["resolved_revision"] = None
    state.save()

    experiment = render_experiment_v2(state, "production")

    assert experiment["model"]["revision"] == "candidate"


def test_resolved_sections_take_precedence_over_compatibility_overrides(tmp_path: Path) -> None:
    state = _campaign_state(tmp_path)

    experiment = render_experiment_v2(state, "production")

    assert experiment["compatibility_marker"] == "kept"
    assert experiment["vllm_stats"]["prefill_seq_len"] == 2048
    assert tuple(experiment["vllm_stats"]["measurements"]) == (
        "latency-first",
        "throughput-second",
    )
    assert experiment["mip"]["marker"] == "named"
    assert "marker" not in experiment["post_mip"]
    assert experiment["pruning"]["micro_batch_size"] == 7
    assert experiment["pruning"]["automodel"]["parallel"]["tp"] == 2


def test_runner_compatibility_override_is_applied_to_resolved_runner(tmp_path: Path) -> None:
    state = _campaign_state(tmp_path)

    runner = render_runner_v2(state, "production")

    assert runner["runner"]["slurm"]["partition"] == "late-override"
    assert runner["runner"]["slurm"]["account"] == "account"


def test_execution_uses_resolved_stage_resource_and_parallel_profile(tmp_path: Path) -> None:
    state = _campaign_state(tmp_path)

    execution = render_execution_v2(state, "production")

    assert execution["execution"]["stages"]["width_importance"] == {
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


def test_absent_semantic_collections_preserve_compatibility_overrides(tmp_path: Path) -> None:
    state = _campaign_state(tmp_path)
    state.payload["collections"].pop("mip_config")
    state.payload["collections"].pop("post_mip_flows")
    state.save()

    experiment = render_experiment_v2(state, "production")

    assert experiment["mip"]["marker"] == "compatibility"
    assert experiment["post_mip"]["marker"] == "compatibility"


def test_build_freezes_one_snapshot_before_rendering_both_budgets(
    tmp_path: Path,
    monkeypatch,
) -> None:
    state = _campaign_state(tmp_path)

    def mutate_state_after_smoke(path: Path) -> SimpleNamespace:
        if path.name == "smoke":
            state.set_field("stages.width_importance.batch", 99, source="user")
            state.set_collection("stage_batches", {"pruning.micro_batch_size": 99})
            state.set_collection("experiment_overrides", {"mutated": True})
            state.set_collection(
                "runner_overrides",
                {"runner": {"slurm": {"partition": "mutated"}}},
            )
        return SimpleNamespace(valid=True, error=None)

    monkeypatch.setattr(
        bundle_module,
        "validate_bundle",
        mutate_state_after_smoke,
    )
    monkeypatch.setattr(bundle_module, "dry_run_bundle", lambda path: "dry run\n")

    build_bundles_v2(state.campaign_dir, state)

    assert state.get_field("stages.width_importance.batch") == 99
    assert state.collection("stage_batches")["pruning.micro_batch_size"] == 99
    assert state.collection("legacy_state") is None
    for budget in ("smoke", "production"):
        experiment = yaml.safe_load((state.campaign_dir / budget / "experiment.yaml").read_text())
        runner = yaml.safe_load((state.campaign_dir / budget / "runner.yaml").read_text())
        assert experiment["pruning"]["micro_batch_size"] == 7
        assert "mutated" not in experiment
        assert runner["runner"]["slurm"]["partition"] == "late-override"
    provenance = yaml.safe_load((state.campaign_dir / "resolved_defaults.yaml").read_text())
    assert provenance["stages.width_importance.batch"] == {
        "value": 4,
        "requested": 3,
        "effective": 4,
        "source": "preset",
    }
