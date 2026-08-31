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

"""Guided profiles, defaults, navigation, and bundle output for Puzzletron setup v2."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import yaml

import puzzletron_setup.v2.cli as cli_module
import puzzletron_setup.v2.wizard as wizard_module
from puzzletron_orchestrator.compiler import (
    compile_campaign_plan,
    load_execution_config,
    load_runner_config,
)
from puzzletron_setup import SetupError
from puzzletron_setup.inspection import InspectedModel
from puzzletron_setup.profiles import AxisInventory, ModelInventory
from puzzletron_setup.v2.defaults import DefaultsResolver
from puzzletron_setup.v2.presets import get_setup_preset
from puzzletron_setup.v2.prompts import NonInteractiveBackend, PromptChoice, ScriptedBackend
from puzzletron_setup.v2.session import WizardSession
from puzzletron_setup.v2.state import WizardState
from puzzletron_setup.v2.wizard import (
    _CUSTOM_DATA_SOURCE,
    _acquisition_sample_requirements,
    data_section,
    infrastructure_section,
)

_QWEN_FAMILY_CONFIG = "examples/puzzletron/configs/families/qwen3_5/family.yaml"
_NEMOTRON_FAMILY_CONFIG = "examples/puzzletron/configs/families/nemotron3/family.yaml"


# Public facade and guided-profile defaults


def test_guided_profile_defaults_are_selected_by_model_family(tmp_path):
    families = {}
    for family, num_solutions in (("first", 2), ("second", 7)):
        family_dir = tmp_path / family
        family_dir.mkdir()
        family_config = family_dir / "family.yaml"
        family_config.write_text("family: {}\n")
        (family_dir / "setup_v2_defaults.yaml").write_text(
            yaml.safe_dump(
                {
                    "schema_version": 1,
                    "profiles": {"smoke": {"mip": {"num_solutions": num_solutions}}},
                }
            )
        )
        families[family] = family_config

    preset = get_setup_preset("smoke")

    assert preset.resolved_defaults(families["first"])["mip"]["num_solutions"] == 2
    assert preset.resolved_defaults(families["second"])["mip"]["num_solutions"] == 7


def test_ambiguous_model_specific_defaults_fail_closed(tmp_path):
    family_config = tmp_path / "family.yaml"
    family_config.write_text("family: {}\n")
    (tmp_path / "setup_v2_defaults.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": 2,
                "profiles": {"smoke": {"mip": {"num_solutions": 2}}},
                "model_overrides": {
                    "first": {
                        "match": {"num_layers": 24},
                        "profiles": {"smoke": {"mip": {"num_solutions": 3}}},
                    },
                    "second": {
                        "match": {"facts": {"hidden_size": 1024}},
                        "profiles": {"smoke": {"mip": {"num_solutions": 4}}},
                    },
                },
            }
        )
    )
    inventory = SimpleNamespace(num_layers=24, facts={"hidden_size": 1024})

    with pytest.raises(SetupError, match="matches multiple guided setup overrides"):
        get_setup_preset("smoke").resolved_defaults(family_config, inventory)


@pytest.mark.parametrize("family_config", [_QWEN_FAMILY_CONFIG, _NEMOTRON_FAMILY_CONFIG])
@pytest.mark.parametrize("preset_name", ["smoke", "balanced", "high-confidence"])
def test_each_model_family_defines_every_guided_profile(family_config, preset_name):
    defaults = get_setup_preset(preset_name).resolved_defaults(family_config)

    assert defaults["pruning"]
    assert defaults["mip"]


# Non-interactive CLI behavior


def test_non_interactive_backend_uses_semantic_defaults() -> None:
    backend = NonInteractiveBackend()
    choices = [PromptChoice("First", "first"), PromptChoice("Second", "second")]

    assert backend.text("Path:", "/resolved/path") == "/resolved/path"
    assert backend.text("Optional commands:", "") == ""
    assert backend.select("Choice:", choices, "second") == "second"
    assert backend.checkbox("Choices:", choices, ["first"]) == ["first"]
    with pytest.raises(SetupError, match="requires a default"):
        backend.text("Path:", None)


@pytest.mark.parametrize(
    "argv",
    [
        ["--resume", "existing", "--campaign-dir", "new"],
        ["--non-interactive", "--defaults", "defaults.yaml"],
        ["--non-interactive", "--campaign-dir", "campaign"],
    ],
)
def test_cli_rejects_invalid_automation_argument_combinations(argv):
    with pytest.raises(SystemExit) as error:
        cli_module.main(argv)

    assert error.value.code == 2


def test_cli_incomplete_noninteractive_defaults_fail_fast(tmp_path, capsys):
    defaults = tmp_path / "defaults.yaml"
    defaults.write_text("schema_version: 1\n")

    assert (
        cli_module.main(
            [
                "--non-interactive",
                "--defaults",
                str(defaults),
                "--campaign-dir",
                str(tmp_path / "campaign"),
            ]
        )
        == 2
    )
    assert "Setup stopped: Enter a model path or Hugging Face URL." in capsys.readouterr().out


def test_cli_invalid_noninteractive_vllm_topology_fails_fast(tmp_path, monkeypatch, capsys):
    campaign = tmp_path / "campaign"
    model_path = tmp_path / "model"
    dataset = tmp_path / "dataset"
    model_path.mkdir()
    dataset.mkdir()
    inspected = _qwen_inspected_model(model_path)
    monkeypatch.setattr(wizard_module, "inspect_model", lambda source: inspected)
    monkeypatch.setattr(
        wizard_module,
        "infer_dataset_modality",
        lambda source: SimpleNamespace(modality="text", evidence="local fixture"),
    )
    defaults = tmp_path / "defaults.yaml"
    defaults.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "model": {"source": str(model_path)},
                "data": {"source": str(dataset), "modality": "text"},
                "infrastructure": {
                    "gpus_per_node": 1,
                    "execution_contract": {
                        "repository": "/worker/modelopt",
                        "venv": "/worker/venv",
                    },
                },
                "vllm": {
                    "enabled": True,
                    "topology": {"tensor_parallel_size": 3},
                },
            },
            sort_keys=False,
        )
    )

    assert (
        cli_module.main(
            [
                "--non-interactive",
                "--defaults",
                str(defaults),
                "--campaign-dir",
                str(campaign),
                "--profile",
                "smoke",
            ]
        )
        == 2
    )
    output = capsys.readouterr().out
    assert "Setup stopped: Non-interactive vLLM topology is incompatible" in output
    assert "TP=3 is incompatible" in output
    assert "valid choices [1, 2, 4, 8]" in output


# Resume and navigation behavior


def test_resume_full_promotes_guided_state_and_preserves_profile_baseline(
    tmp_path,
    monkeypatch,
):
    state = WizardState.start(
        tmp_path / "campaign",
        defaults_path=None,
        setup_mode="quick",
        preset="balanced",
    )
    state.payload["model"] = {"source": "saved-model"}
    state.save()
    resolved_baseline = {}
    inspected = SimpleNamespace(
        inventory=SimpleNamespace(family_config=_QWEN_FAMILY_CONFIG),
    )

    def capture_baseline(session, resolver, context):
        del session, context
        resolved = resolver.resolve_default("pruning.depth_remove")
        resolved_baseline.update(value=resolved.value, source=resolved.source)
        return True

    monkeypatch.setattr(wizard_module, "SECTION_BUILDERS", (capture_baseline,))
    monkeypatch.setattr(wizard_module, "SECTION_NAMES", ("model",))
    monkeypatch.setattr(wizard_module, "inspect_model", lambda source: inspected)
    monkeypatch.setattr(wizard_module, "build_bundles_v2", lambda campaign, state: None)

    wizard_module.run_wizard_v2(
        resume=state.campaign_dir,
        defaults_path=None,
        backend=ScriptedBackend([]),
        full=True,
    )

    resumed = WizardState.resume(state.path)
    assert resumed.setup_mode == "full"
    assert resumed.preset == "balanced"
    assert resolved_baseline == {"value": 4, "source": "preset"}


@pytest.mark.parametrize(
    ("contents", "message"),
    [
        (None, "Defaults file does not exist"),
        ("schema_version: 2\n", "Unsupported defaults schema 2; expected 1"),
    ],
)
def test_invalid_resume_replacement_defaults_file_is_not_persisted(
    tmp_path,
    monkeypatch,
    contents,
    message,
):
    state = WizardState.start(
        tmp_path / "campaign",
        defaults_path=None,
        setup_mode="quick",
        preset="balanced",
    )
    replacement = tmp_path / "invalid.yaml"
    if contents is not None:
        replacement.write_text(contents)
    monkeypatch.setattr(wizard_module, "SECTION_BUILDERS", ())
    monkeypatch.setattr(wizard_module, "SECTION_NAMES", ())

    with pytest.raises(SetupError, match=message):
        wizard_module.run_wizard_v2(
            resume=state.campaign_dir,
            defaults_path=replacement,
            backend=ScriptedBackend([]),
        )

    assert WizardState.resume(state.path).defaults_path is None


# Guided defaults and data validation


def test_legacy_state_without_setup_metadata_resumes_in_full_mode(tmp_path):
    state = WizardState.start(tmp_path / "campaign", defaults_path=None)
    state.payload.pop("setup")
    state.save()

    resumed = WizardState.resume(state.path)

    assert resumed.setup_mode == "full"
    assert resumed.preset is None


def test_guided_data_rejects_an_explicit_modality_incompatible_with_the_model(
    tmp_path,
    monkeypatch,
):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    state = WizardState.start(
        tmp_path / "campaign",
        defaults_path=None,
        setup_mode="quick",
        preset="balanced",
    )
    monkeypatch.setattr(
        "puzzletron_setup.v2.wizard.infer_dataset_modality",
        lambda source: SimpleNamespace(modality="text", evidence="local fixture"),
    )

    with pytest.raises(SetupError, match="multimodal.*incompatible"):
        data_section(
            WizardSession(
                state,
                ScriptedBackend([_CUSTOM_DATA_SOURCE, str(dataset)]),
                guided=True,
            ),
            DefaultsResolver(file_defaults={"data": {"modality": "multimodal"}}),
            _context(),
        )


@pytest.mark.parametrize("limit", ["not-a-number", -1, 0, True])
def test_quality_comparison_limit_must_be_a_positive_integer(limit):
    resolver = DefaultsResolver(
        file_defaults={
            "post_mip": {
                "quality_comparison": {
                    "enabled": True,
                    "reference_checkpoint": "/teacher",
                    "limit": limit,
                }
            }
        }
    )

    with pytest.raises(
        SetupError,
        match=r"post_mip\.quality_comparison\.limit must be a positive integer",
    ):
        wizard_module._quality_comparison_defaults(resolver)


# Bundle generation and review output


def test_width_sanity_samples_contribute_without_sort_sanity(tmp_path):
    state = WizardState.start(tmp_path / "campaign", defaults_path=None)
    state.set_collection(
        "pruning",
        {
            "width_importance_samples": 2,
            "replacement_samples": 3,
            "sort_sanity": False,
            "width_sanity": True,
            "width_sanity_samples": 11,
        },
    )

    assert _acquisition_sample_requirements(state) == (2, 11)


def test_guided_wizard_runs_real_sections_and_generates_valid_bundles(
    tmp_path,
    monkeypatch,
):
    campaign = tmp_path / "campaign"
    model_path = tmp_path / "model"
    dataset = tmp_path / "dataset"
    model_path.mkdir()
    dataset.mkdir()
    inspected = _qwen_inspected_model(model_path)
    monkeypatch.setattr(wizard_module, "inspect_model", lambda source: inspected)
    monkeypatch.setattr(
        wizard_module,
        "infer_dataset_modality",
        lambda source: SimpleNamespace(modality="text", evidence="local fixture"),
    )
    defaults = tmp_path / "defaults.yaml"
    defaults.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "model": {"source": str(model_path)},
                "data": {
                    "source": str(dataset),
                    "modality": "text",
                    "layout": "fixed",
                    "sequence_length": 32,
                },
                "infrastructure": {
                    "gpus_per_node": 1,
                    "runner": {
                        "slurm": {"job_name_prefix": "acct-puzzletron"},
                    },
                    "execution_contract": {
                        "repository": "/worker/modelopt",
                        "venv": "/worker/venv",
                    },
                },
            },
            sort_keys=False,
        )
    )

    result = wizard_module.run_wizard_v2(
        resume=None,
        defaults_path=defaults,
        backend=NonInteractiveBackend(),
        campaign_dir=campaign,
        setup_profile="smoke",
    )

    assert result == campaign.resolve()
    assert (campaign / "smoke" / "experiment.yaml").is_file()
    assert (campaign / "production" / "experiment.yaml").is_file()
    assert (campaign / "resolved_defaults.yaml").is_file()
    generated = WizardState.resume(campaign)
    assert generated.collection("pruning")["depth_remove"] == 0
    assert generated.collection("pruning")["width_importance_samples"] == 8
    assert generated.collection("pruning")["replacement_samples"] == 4
    assert generated.collection("default_resolutions")["pruning.depth_remove"] == {
        "value": 0,
        "source": "model_profile",
    }
    assert generated.collection("default_resolutions")["pruning.width_importance_samples"] == {
        "value": 8,
        "source": "model_profile",
    }
    assert (
        generated.collection("default_resolutions")["post_mip.quality_comparison"]["source"]
        == "model_profile"
    )
    smoke = yaml.safe_load((campaign / "smoke" / "experiment.yaml").read_text())
    smoke_flow = next(iter(smoke["post_mip"]["flows"].values()))
    smoke_comparison = smoke_flow["nodes"]["quality_benchmarks"]
    assert smoke_comparison["config"]["limit"] == 8
    assert "recorded_observation" not in smoke_comparison["config"]
    smoke_runner = yaml.safe_load((campaign / "smoke" / "runner.yaml").read_text())
    assert smoke_runner["runner"]["slurm"]["job_name_prefix"] == "acct-puzzletron"
    production = yaml.safe_load((campaign / "production" / "experiment.yaml").read_text())
    flow = next(iter(production["post_mip"]["flows"].values()))
    comparison = flow["nodes"]["quality_benchmarks"]
    assert comparison["type"] == "downstream_evaluation"
    assert comparison["input"] == "best"
    assert comparison["failure_policy"] == "strict"
    assert comparison["config"]["tasks"] == ["ifeval", "gsm8k"]
    assert comparison["config"]["limit"] == 100
    assert "recorded_observation" not in comparison["config"]
    resolved_defaults = yaml.safe_load((campaign / "resolved_defaults.yaml").read_text())
    assert resolved_defaults["pruning.depth_remove"] == {
        "value": 0,
        "requested": None,
        "effective": 0,
        "source": "model_profile",
    }
    pruning = generated.collection("pruning")
    pruning["depth_remove"] = 1
    generated.set_collection("pruning", pruning)
    profiles = generated.collection("parallel_profiles")
    first_profile = next(iter(profiles.values()))
    profiles["secondary"] = {**first_profile, "tp": 2}
    generated.set_collection("parallel_profiles", profiles)

    wizard_module.build_bundles_v2(campaign, generated)

    resolved_defaults = yaml.safe_load((campaign / "resolved_defaults.yaml").read_text())
    assert resolved_defaults["pruning.depth_remove"] == {
        "value": 0,
        "requested": None,
        "effective": 1,
        "source": "model_profile",
    }
    assert resolved_defaults["profiles"]["effective"] == profiles


def test_guided_wizard_generates_the_complete_qwen_vlm_flow(tmp_path, monkeypatch):
    campaign = tmp_path / "campaign"
    model_path = tmp_path / "model"
    dataset = tmp_path / "dataset"
    model_path.mkdir()
    dataset.mkdir()
    inspected = _qwen_inspected_model(model_path, multimodal=True)
    monkeypatch.setattr(wizard_module, "inspect_model", lambda source: inspected)
    monkeypatch.setattr(
        wizard_module,
        "infer_dataset_modality",
        lambda source: SimpleNamespace(modality="multimodal", evidence="local fixture"),
    )
    defaults = tmp_path / "defaults.yaml"
    defaults.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "model": {"source": str(model_path)},
                "data": {
                    "source": str(dataset),
                    "modality": "multimodal",
                    "layout": "padded_varlen",
                    "sequence_length": 512,
                },
                "infrastructure": {
                    "gpus_per_node": 1,
                    "execution_contract": {
                        "repository": "/worker/modelopt",
                        "venv": "/worker/venv",
                    },
                },
            },
            sort_keys=False,
        )
    )

    result = wizard_module.run_wizard_v2(
        resume=None,
        defaults_path=defaults,
        backend=NonInteractiveBackend(),
        campaign_dir=campaign,
        setup_profile="smoke",
    )

    assert result == campaign.resolve()
    smoke = yaml.safe_load((campaign / "smoke" / "experiment.yaml").read_text())
    smoke_flow = next(iter(smoke["post_mip"]["flows"].values()))
    smoke_quality = smoke_flow["nodes"]["quality_benchmarks"]["config"]
    assert smoke_quality["profile"] == "qwen35_vlm_e2e_full_eval"
    assert smoke_quality["limit"] == 8
    assert "recorded_observation" not in smoke_quality
    production = yaml.safe_load((campaign / "production" / "experiment.yaml").read_text())
    flow_id, flow = next(iter(production["post_mip"]["flows"].items()))
    assert flow_id == "params-90"
    nodes = flow["nodes"]
    assert tuple(nodes) == (
        "image_eval",
        "best_vlm_loss",
        "materialized",
        "vlm_serving",
        "fastest_vlm",
        "short_kd",
        "final_image_eval",
        "best",
        "quality_benchmarks",
    )
    assert nodes["vlm_serving"]["config"]["image_batch_sizes"] == [1, 6, 12]
    assert nodes["fastest_vlm"]["metric"] == (
        "vlm_serving.images_12.concurrency_1.image_throughput"
    )
    quality = nodes["quality_benchmarks"]
    assert quality["input"] == "best"
    assert quality["config"]["profile"] == "qwen35_vlm_e2e_full_eval"
    assert "model" not in quality["config"]
    assert "log_samples" not in quality["config"]
    assert "recorded_observation" not in quality["config"]
    assert production["global_distillation"]["domain"] == "vlm"
    assert production["global_distillation"]["freeze_policy"] == "train_all"
    assert production["tokenize_data"]["enabled"] is False
    assert production["tokenize_data"]["caches"] == []
    assert production["depth_importance"]["enabled"] is False
    assert production["bypass"]["enabled"] is False
    axes = production["search_space"]["axes"]
    assert {axis_id for axis_id, axis in axes.items() if axis["enabled"]} == {"ffn_intermediate"}
    plan = compile_campaign_plan(
        experiment_config_path=campaign / "production" / "experiment.yaml",
        runner=load_runner_config(campaign / "production" / "runner.yaml"),
        execution=load_execution_config(campaign / "production" / "execution.yaml"),
        stage_filter="full",
    )
    stage_ids = tuple(stage.stage_id for stage in plan.stages)
    assert f"post.{flow_id}.vlm_serving" in stage_ids
    assert f"post.{flow_id}.short_kd" in stage_ids
    assert stage_ids[-1] == f"post.{flow_id}.quality_benchmarks"
    stages = {stage.stage_id: stage for stage in plan.stages}
    assert all(stage.nodes == 1 for stage in plan.stages)
    assert all(stage.total_gpus <= 1 for stage in plan.stages)
    assert stages[f"post.{flow_id}.vlm_serving"].total_gpus == 1
    assert stages[f"post.{flow_id}.short_kd"].total_gpus == 1
    assert stages[f"post.{flow_id}.quality_benchmarks"].total_gpus == 1


# Interactive prompt navigation


def test_guided_infrastructure_records_cpu_partition_default(tmp_path):
    state = WizardState.start(tmp_path / "campaign", defaults_path=None)
    resolver = DefaultsResolver(
        file_defaults={
            "infrastructure": {
                "execution_contract": {
                    "repository": "/worker/modelopt",
                    "venv": "/worker/venv",
                },
                "runner": {
                    "kind": "slurm",
                    "slurm": {
                        "account": "acct",
                        "partition": "gpu",
                        "partition_cpu": "cpu",
                    },
                },
            }
        }
    )

    assert infrastructure_section(
        WizardSession(state, ScriptedBackend(["defaults"]), guided=True),
        resolver,
        {},
    )

    assert state.get_field("infrastructure.runner.slurm.partition") == "gpu"
    assert state.get_field("infrastructure.runner.slurm.partition_cpu") == "cpu"


def test_customize_partition_prompt_renders_list_default_as_comma_separated(tmp_path):
    state = WizardState.start(tmp_path / "campaign", defaults_path=None)
    resolver = DefaultsResolver(
        file_defaults={
            "infrastructure": {
                "runner": {
                    "slurm": {
                        "partition": ["gpu-a", "gpu-b"],
                        "partition_cpu": ["cpu-a", "cpu-b"],
                    }
                },
            }
        }
    )

    class PartitionDefaultBackend(ScriptedBackend):
        partition_default = None
        cpu_partition_default = None

        def text(self, message: str, default: str) -> str:
            if message.startswith("Eligible Slurm partitions"):
                self.partition_default = default
                return default
            if message.startswith("Eligible CPU-only Slurm partitions"):
                self.cpu_partition_default = default
                return default
            return super().text(message, default)

    backend = PartitionDefaultBackend(
        [
            "customize",
            "/worker/modelopt",
            "/worker/venv",
            "",
            "",
            "acct",
            "pt",
            "4:00:00",
            "8",
            "",
        ]
    )

    assert infrastructure_section(
        WizardSession(state, backend),
        resolver,
        {},
    )

    assert backend.partition_default == "gpu-a,gpu-b"
    assert backend.cpu_partition_default == "cpu-a,cpu-b"
    assert state.get_field("infrastructure.runner.slurm.partition") == "gpu-a,gpu-b"
    assert state.get_field("infrastructure.runner.slurm.partition_cpu") == "cpu-a,cpu-b"
    assert backend.remaining == 0


def test_resume_preserves_legacy_partition_fields(tmp_path):
    state = WizardState.start(tmp_path / "campaign", defaults_path=None)
    legacy = {
        "partition_batch": "batch",
        "partition_interactive": "interactive",
        "partition_cpu": "cpu",
        "interactive_max_nodes": 2,
    }
    for field, value in legacy.items():
        state.set_field(f"infrastructure.runner.slurm.{field}", value, source="user")

    resumed = WizardState.resume(state.path)

    for field, value in legacy.items():
        assert resumed.get_field(f"infrastructure.runner.slurm.{field}") == value


def _qwen_inspected_model(model_path, *, multimodal: bool = False) -> InspectedModel:
    inventory = ModelInventory(
        family="qwen3_5",
        descriptor="qwen3_5" if multimodal else "qwen3_5_text",
        family_config=_QWEN_FAMILY_CONFIG,
        model_type="qwen3_5" if multimodal else "qwen3_5_text",
        architectures=(
            ("Qwen3_5ForConditionalGeneration",) if multimodal else ("Qwen3_5ForCausalLM",)
        ),
        multimodal=multimodal,
        moe=False,
        num_layers=24,
        num_sublayers=48,
        layer_counts={"full_attention": 6, "linear_attention": 18},
        facts={
            "hidden_size": 1024,
            "num_attention_heads": 8,
            "num_key_value_heads": 2,
            "intermediate_size": 3584,
        },
        axes=(
            AxisInventory(
                axis_id="hidden_width",
                label="Hidden width",
                teacher_value=1024,
                values=(1024, 768),
                alignment=256,
            ),
            AxisInventory(
                axis_id="kv_groups",
                label="KV groups",
                teacher_value=2,
                values=(2, 1),
                alignment=1,
            ),
            AxisInventory(
                axis_id="q_heads_per_group",
                label="Query heads per KV group",
                teacher_value=4,
                values=(4, 2),
                alignment=1,
            ),
            AxisInventory(
                axis_id="ffn_intermediate",
                label="FFN intermediate width",
                teacher_value=3584,
                values=(3584, 3072, 2048),
                alignment=256,
            ),
            AxisInventory(
                axis_id="gdn_key_groups",
                label="Gated-delta key groups",
                teacher_value=16,
                values=(16, 12, 8),
                alignment=1,
            ),
            AxisInventory(
                axis_id="gdn_value_heads_per_group",
                label="Gated-delta value heads per group",
                teacher_value=1,
                values=(1,),
                alignment=1,
            ),
            AxisInventory(
                axis_id="gdn_key_head_dim",
                label="Gated-delta key head dimension",
                teacher_value=128,
                values=(128, 96),
                alignment=32,
            ),
            AxisInventory(
                axis_id="gdn_value_head_dim",
                label="Gated-delta value head dimension",
                teacher_value=128,
                values=(128, 96),
                alignment=32,
            ),
        ),
    )
    return InspectedModel(
        source=str(model_path),
        requested_revision=None,
        resolved_revision=None,
        is_local=True,
        config={
            "model_type": "qwen3_5" if multimodal else "qwen3_5_text",
            "text_config": {
                "num_hidden_layers": 24,
                "layer_types": ["linear_attention"] * 18 + ["full_attention"] * 6,
            },
            **({"vision_config": {"model_type": "qwen3_5_vision_encoder"}} if multimodal else {}),
        },
        inventory=inventory,
    )


def _context():
    return {
        "model": SimpleNamespace(
            inventory=SimpleNamespace(multimodal=False),
        )
    }
