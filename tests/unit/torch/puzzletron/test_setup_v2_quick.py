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

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

import puzzletron_setup.v2.bundle as bundle_module
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
from puzzletron_setup.v2.defaults import DefaultsResolver, validate_defaults
from puzzletron_setup.v2.presets import get_setup_preset
from puzzletron_setup.v2.prompts import NonInteractiveBackend, ScriptedBackend
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


def test_only_cpu_slurm_integer_defaults_accept_null() -> None:
    defaults = validate_defaults(
        {
            "schema_version": 1,
            "infrastructure": {
                "runner": {"slurm": {"cpu_cpus_per_task": None, "cpu_memory_mb": None}}
            },
        }
    )

    assert defaults["infrastructure"]["runner"]["slurm"] == {
        "cpu_cpus_per_task": None,
        "cpu_memory_mb": None,
    }
    with pytest.raises(SetupError, match=r"data\.sequence_length must be an integer"):
        validate_defaults({"schema_version": 1, "data": {"sequence_length": None}})


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


def test_cli_incomplete_noninteractive_defaults_fail_fast(tmp_path):
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

    with pytest.raises(SetupError, match=r"multimodal.*incompatible"):
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
        wizard_module._quality_comparison_defaults(resolver, modality="text")


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
                    "gpus_per_node": 8,
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
        setup_profile="balanced",
    )

    assert result == campaign.resolve()
    smoke = yaml.safe_load((campaign / "smoke" / "experiment.yaml").read_text())
    smoke_execution = yaml.safe_load((campaign / "smoke" / "execution.yaml").read_text())
    smoke_flow = next(iter(smoke["post_mip"]["flows"].values()))
    smoke_quality = smoke_flow["nodes"]["quality_benchmarks"]["config"]
    assert smoke_quality["profile"] == "qwen35_vlm_realworldqa64_mmmu120_mvbench160_frozen_rows_v3"
    assert "limit" not in smoke_quality
    assert smoke_quality["limit_mm_per_prompt"] == {"image": 32}
    assert smoke_quality["max_model_len"] == 32768
    assert "recorded_observation" not in smoke_quality
    assert smoke_execution["execution"]["stages"]["post.params-90.short_kd"]["instances"] == 1
    production = yaml.safe_load((campaign / "production" / "experiment.yaml").read_text())
    production_execution = yaml.safe_load((campaign / "production" / "execution.yaml").read_text())
    flow_id, flow = next(iter(production["post_mip"]["flows"].items()))
    assert flow_id == "params-90"
    nodes = flow["nodes"]
    quality = nodes["quality_benchmarks"]
    assert "model" not in quality["config"]
    assert "recorded_observation" not in quality["config"]
    assert production_execution["execution"]["stages"]["replacement_scoring"]["instances"] == 8
    assert production_execution["execution"]["stages"]["post.params-90.short_kd"]["instances"] == 8
    assert production["global_distillation"]["domain"] == "vlm"
    assert production["global_distillation"]["freeze_policy"] == "train_all"
    assert production["tokenize_data"]["enabled"] is False
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
    mip = next(stage for stage in plan.stages if stage.stage_id == "mip")
    assert mip.resource == "cpu"
    assert mip.total_gpus == 0
    snapshots = {
        budget: (campaign / budget / "dry-run-plan.txt").read_text()
        for budget in ("smoke", "production")
    }
    wizard_module.build_bundles_v2(campaign, WizardState.resume(campaign))
    assert snapshots == {
        budget: (campaign / budget / "dry-run-plan.txt").read_text()
        for budget in ("smoke", "production")
    }

    changed_state = WizardState.resume(campaign)
    changed_state.set_field(
        "infrastructure.runner.slurm.partition",
        "replacement-gpu-partition",
        source="user",
    )
    readme_path = campaign / "README.md"
    readme_path.write_text(f"{readme_path.read_text()}\nrollback sentinel\n")

    def campaign_files():
        return {
            path.relative_to(campaign): path.read_bytes()
            for path in campaign.rglob("*")
            if path.is_file()
        }

    published_files = campaign_files()
    assert b"replacement-gpu-partition" not in published_files[Path("smoke/runner.yaml")]

    render_plan = bundle_module.dry_run_bundle

    def fail_production_plan(bundle):
        if bundle.parent == campaign and bundle.name == "production":
            raise RuntimeError("dry-run rendering failed")
        return render_plan(bundle)

    monkeypatch.setattr(bundle_module, "dry_run_bundle", fail_production_plan)
    with pytest.raises(RuntimeError, match="dry-run rendering failed"):
        wizard_module.build_bundles_v2(campaign, changed_state)
    assert campaign_files() == published_files

    monkeypatch.setattr(bundle_module, "dry_run_bundle", render_plan)
    replace = bundle_module.os.replace

    def fail_readme_publish(source, target):
        if target == campaign / "README.md" and source.name == "README.md":
            raise RuntimeError("README publication failed")
        return replace(source, target)

    monkeypatch.setattr(bundle_module.os, "replace", fail_readme_publish)
    with pytest.raises(RuntimeError, match="README publication failed"):
        wizard_module.build_bundles_v2(campaign, changed_state)
    assert campaign_files() == published_files


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
