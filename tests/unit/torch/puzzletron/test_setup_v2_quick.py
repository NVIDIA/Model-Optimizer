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

import sys
from types import ModuleType, SimpleNamespace

import pytest
import yaml

import puzzletron_setup.v2.bundle as bundle_module
import puzzletron_setup.v2.cli as cli_module
import puzzletron_setup.v2.wizard as wizard_module
import puzzletron_setup.v2.wizard_common as wizard_common_module
from puzzletron_setup import SetupError
from puzzletron_setup.inspection import InspectedModel
from puzzletron_setup.profiles import AxisInventory, ModelInventory
from puzzletron_setup.v2.cli import _parser
from puzzletron_setup.v2.defaults import DefaultsResolver
from puzzletron_setup.v2.presets import QUICK_SETUP_PRESETS, get_setup_preset
from puzzletron_setup.v2.prompts import (
    BACK,
    InteractiveBackend,
    NonInteractiveBackend,
    PromptChoice,
    ScriptedBackend,
    _bind_escape_back,
)
from puzzletron_setup.v2.session import WizardSession
from puzzletron_setup.v2.state import WizardState
from puzzletron_setup.v2.wizard import (
    _CUSTOM_DATA_SOURCE,
    _PUZZLE_KD_DATA_SOURCE,
    _acquisition_sample_requirements,
    _fresh_state,
    _section_action,
    data_section,
    depth_section,
    infrastructure_section,
    output_review_section,
)

_QWEN_FAMILY_CONFIG = "examples/puzzletron/configs/families/qwen3_5/family.yaml"
_NEMOTRON_FAMILY_CONFIG = "examples/puzzletron/configs/families/nemotron3/family.yaml"


def test_common_helpers_remain_available_from_wizard_facade():
    names = (
        "BUILTINS",
        "CANONICAL_STAGE_STRATEGIES",
        "STATIC_MODEL_BATCH_PATHS",
        "STATIC_MODEL_STAGES",
        "_default_axis_values",
        "_depth_granularity_choices",
        "_guided_integer_default",
        "_integer_field",
        "_mapping_copy",
        "_nested_records",
        "_plain_review_value",
        "_print_default_decisions",
        "_record_default",
        "_replacement_granularity_choices",
        "_resolved",
        "_resolver",
        "_section_action",
        "_text_field",
        "_vllm_granularity_choices",
    )

    for name in names:
        assert getattr(wizard_module, name) is getattr(wizard_common_module, name)


def _qwen_inventory(
    *,
    num_layers,
    hidden_size,
    intermediate_size,
    num_attention_heads,
    num_key_value_heads,
):
    return SimpleNamespace(
        num_layers=num_layers,
        facts={
            "hidden_size": hidden_size,
            "intermediate_size": intermediate_size,
            "num_attention_heads": num_attention_heads,
            "num_key_value_heads": num_key_value_heads,
        },
    )


def _context():
    return {
        "model": SimpleNamespace(
            inventory=SimpleNamespace(multimodal=False),
        )
    }


def test_guided_profiles_explain_cost_and_load_family_defaults():
    assert [preset.name for preset in QUICK_SETUP_PRESETS] == [
        "smoke",
        "balanced",
        "high-confidence",
    ]
    assert "recommended" in get_setup_preset("balanced").choice_title.lower()

    resolver = DefaultsResolver(
        builtins={"pruning": {"bypass": {"enabled": True}}},
        preset_defaults=get_setup_preset("smoke").resolved_defaults(_QWEN_FAMILY_CONFIG),
    )

    resolved = resolver.resolve_default("pruning.bypass.enabled")
    assert resolved.value is False
    assert resolved.source == "preset"


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


def test_same_family_profile_uses_smaller_sample_counts_for_qwen_0p8b():
    qwen_0p8b = _qwen_inventory(
        num_layers=24,
        hidden_size=1024,
        intermediate_size=3584,
        num_attention_heads=8,
        num_key_value_heads=2,
    )
    qwen_9b = _qwen_inventory(
        num_layers=32,
        hidden_size=4096,
        intermediate_size=12288,
        num_attention_heads=16,
        num_key_value_heads=4,
    )

    smoke = get_setup_preset("smoke")
    small_defaults = smoke.resolved_defaults(_QWEN_FAMILY_CONFIG, qwen_0p8b)
    large_defaults = smoke.resolved_defaults(_QWEN_FAMILY_CONFIG, qwen_9b)

    assert small_defaults["pruning"]["width_importance_samples"] == 8
    assert small_defaults["pruning"]["replacement_samples"] == 4
    assert large_defaults["pruning"]["width_importance_samples"] == 512
    assert large_defaults["pruning"]["replacement_samples"] == 32
    assert small_defaults["pruning"]["depth_remove"] == 1
    assert large_defaults["pruning"]["depth_remove"] == 1


def test_model_override_changes_only_its_historical_qwen_9b_values():
    qwen_9b = _qwen_inventory(
        num_layers=32,
        hidden_size=4096,
        intermediate_size=12288,
        num_attention_heads=16,
        num_key_value_heads=4,
    )

    defaults = get_setup_preset("high-confidence").resolved_defaults(
        _QWEN_FAMILY_CONFIG,
        qwen_9b,
    )

    assert defaults["pruning"]["depth_importance_samples"] == 128
    assert defaults["pruning"]["replacement_samples"] == 128
    assert defaults["pruning"]["bypass"]["enabled"] is False
    assert defaults["mip"]["goal_value"] == "75%"
    assert defaults["pruning"]["width_importance_samples"] == 65536
    assert defaults["mip"]["num_solutions"] == 16


def test_qwen_27b_balanced_profile_uses_its_historical_campaign_budgets():
    qwen_27b = _qwen_inventory(
        num_layers=64,
        hidden_size=5120,
        intermediate_size=17408,
        num_attention_heads=24,
        num_key_value_heads=4,
    )

    defaults = get_setup_preset("balanced").resolved_defaults(
        _QWEN_FAMILY_CONFIG,
        qwen_27b,
    )

    assert defaults["pruning"]["width_importance_samples"] == 16384
    assert defaults["pruning"]["replacement_samples"] == 16
    assert defaults["mip"]["goal_value"] == "85%"


def test_nemotron_nano_profiles_use_model_specific_smoke_and_search_budgets():
    nano = SimpleNamespace(
        num_layers=52,
        facts={
            "hidden_size": 2688,
            "intermediate_size": 1856,
            "num_attention_heads": 32,
            "num_key_value_heads": 2,
            "num_experts": 128,
        },
    )

    smoke = get_setup_preset("smoke").resolved_defaults(_NEMOTRON_FAMILY_CONFIG, nano)
    high_confidence = get_setup_preset("high-confidence").resolved_defaults(
        _NEMOTRON_FAMILY_CONFIG,
        nano,
    )

    assert smoke["pruning"]["width_importance_samples"] == 2
    assert smoke["pruning"]["bypass"]["enabled"] is True
    assert smoke["mip"]["num_solutions"] == 1
    assert high_confidence["pruning"]["depth_remove"] == 5
    assert high_confidence["pruning"]["width_importance_samples"] == 8192
    assert high_confidence["mip"]["num_solutions"] == 5


def test_explicit_defaults_still_override_model_specific_profile():
    qwen_0p8b = _qwen_inventory(
        num_layers=24,
        hidden_size=1024,
        intermediate_size=3584,
        num_attention_heads=8,
        num_key_value_heads=2,
    )
    family_defaults, model_profile_defaults = get_setup_preset("smoke").resolved_default_layers(
        _QWEN_FAMILY_CONFIG, qwen_0p8b
    )
    resolver = DefaultsResolver(
        preset_defaults=family_defaults,
        model_profile_defaults=model_profile_defaults,
        file_defaults={"pruning": {"width_importance_samples": 64}},
    )

    resolved = resolver.resolve_default("pruning.width_importance_samples")

    assert resolved.value == 64
    assert resolved.source == "defaults_file"


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


def test_guided_profile_defaults_fail_closed_when_family_profile_is_missing(tmp_path):
    family_config = tmp_path / "family.yaml"
    family_config.write_text("family: {}\n")
    (tmp_path / "setup_v2_defaults.yaml").write_text(
        yaml.safe_dump({"schema_version": 1, "profiles": {}})
    )

    with pytest.raises(SetupError, match="profile 'balanced' is not configured"):
        get_setup_preset("balanced").resolved_defaults(family_config)


def test_guided_profile_defaults_reject_unknown_family_profile(tmp_path):
    family_config = tmp_path / "family.yaml"
    family_config.write_text("family: {}\n")
    (tmp_path / "setup_v2_defaults.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "profiles": {
                    "smoke": {"mip": {"num_solutions": 1}},
                    "smkoe": {"mip": {"num_solutions": 2}},
                },
            }
        )
    )

    with pytest.raises(SetupError, match="Unknown guided setup profiles.*smkoe"):
        get_setup_preset("smoke").resolved_defaults(family_config)


@pytest.mark.parametrize("family_config", [_QWEN_FAMILY_CONFIG, _NEMOTRON_FAMILY_CONFIG])
@pytest.mark.parametrize("preset_name", ["smoke", "balanced", "high-confidence"])
def test_each_model_family_defines_every_guided_profile(family_config, preset_name):
    defaults = get_setup_preset(preset_name).resolved_defaults(family_config)

    assert defaults["pruning"]
    assert defaults["mip"]


def test_explicit_defaults_override_guided_profile():
    resolver = DefaultsResolver(
        preset_defaults={"mip": {"num_solutions": 2}},
        file_defaults={"mip": {"num_solutions": 5}},
    )

    resolved = resolver.resolve_default("mip.num_solutions")
    assert resolved.value == 5
    assert resolved.source == "defaults_file"
    assert resolver.resolutions()["mip.num_solutions"] == resolved


def test_fresh_guided_state_records_profile_and_cli_full_is_explicit(tmp_path):
    campaign = tmp_path / "campaign"
    state = _fresh_state(
        ScriptedBackend(["balanced", str(campaign)]),
        None,
        full=False,
    )

    assert state.setup_mode == "quick"
    assert state.preset == "balanced"
    assert _parser().parse_args([]).full is False
    assert _parser().parse_args(["--full"]).full is True


def test_non_interactive_backend_uses_semantic_defaults() -> None:
    backend = NonInteractiveBackend()
    choices = [PromptChoice("First", "first"), PromptChoice("Second", "second")]

    assert backend.text("Path:", "/resolved/path") == "/resolved/path"
    assert backend.text("Optional commands:", "") == ""
    assert backend.select("Choice:", choices, "second") == "second"
    assert backend.checkbox("Choices:", choices, ["first"]) == ["first"]
    with pytest.raises(SetupError, match="requires a default"):
        backend.text("Path:", None)


def test_cli_forwards_full_to_the_wizard(monkeypatch, tmp_path):
    captured = {}

    def run_wizard_v2(**kwargs):
        captured.update(kwargs)
        return tmp_path / "campaign"

    monkeypatch.setattr(wizard_module, "run_wizard_v2", run_wizard_v2)

    assert cli_module.main(["--full"]) == 0
    assert captured["full"] is True


def test_cli_forwards_non_interactive_campaign_contract(monkeypatch, tmp_path):
    captured = {}
    defaults = tmp_path / "defaults.yaml"
    defaults.write_text("schema_version: 1\n")
    campaign = tmp_path / "campaign"

    def run_wizard_v2(**kwargs):
        captured.update(kwargs)
        return campaign

    monkeypatch.setattr(wizard_module, "run_wizard_v2", run_wizard_v2)

    assert (
        cli_module.main(
            [
                "--defaults",
                str(defaults),
                "--campaign-dir",
                str(campaign),
                "--profile",
                "smoke",
                "--non-interactive",
            ]
        )
        == 0
    )
    assert captured["campaign_dir"] == campaign
    assert captured["setup_profile"] == "smoke"
    assert isinstance(captured["backend"], NonInteractiveBackend)


@pytest.mark.parametrize("full", [False, True])
def test_fresh_state_reprompts_for_empty_campaign_directory(tmp_path, full, capsys):
    campaign = tmp_path / ("full" if full else "guided")
    answers = ["", str(campaign)] if full else ["balanced", "", str(campaign)]

    state = _fresh_state(ScriptedBackend(answers), None, full=full)

    assert state.campaign_dir == campaign.resolve()
    assert "Enter a campaign directory path." in capsys.readouterr().out


def test_back_from_first_guided_section_can_change_profile(
    tmp_path,
    monkeypatch,
):
    state = WizardState.start(
        tmp_path / "campaign",
        defaults_path=None,
        setup_mode="quick",
        preset="balanced",
    )
    attempts = 0

    def model_builder(session, resolver, context):
        nonlocal attempts
        del resolver, context
        attempts += 1
        session.begin("model")
        if attempts == 1:
            return (
                session.select(
                    "model.source",
                    "Model:",
                    [PromptChoice("Custom", "custom"), PromptChoice("Known", "known")],
                )
                is not BACK
            )
        return True

    monkeypatch.setattr(wizard_module, "SECTION_BUILDERS", (model_builder,))
    monkeypatch.setattr(wizard_module, "SECTION_NAMES", ("model",))
    monkeypatch.setattr(wizard_module, "build_bundles_v2", lambda campaign, state: None)

    wizard_module.run_wizard_v2(
        resume=state.campaign_dir,
        defaults_path=None,
        backend=ScriptedBackend([BACK, "smoke"]),
    )

    assert WizardState.resume(state.path).preset == "smoke"
    assert attempts == 2


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


def test_resume_replacement_defaults_file_is_persisted(
    tmp_path,
    monkeypatch,
):
    state = WizardState.start(
        tmp_path / "campaign",
        defaults_path=None,
        setup_mode="quick",
        preset="balanced",
    )
    replacement = tmp_path / "replacement.yaml"
    replacement.write_text(yaml.safe_dump({"schema_version": 1}))
    monkeypatch.setattr(wizard_module, "SECTION_BUILDERS", ())
    monkeypatch.setattr(wizard_module, "SECTION_NAMES", ())
    monkeypatch.setattr(wizard_module, "build_bundles_v2", lambda campaign, state: None)

    wizard_module.run_wizard_v2(
        resume=state.campaign_dir,
        defaults_path=replacement,
        backend=ScriptedBackend([]),
    )

    assert WizardState.resume(state.path).defaults_path == replacement.resolve()


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


@pytest.mark.parametrize(
    ("defaults", "message"),
    [
        ({"pruning": {"depth_remove": "invalid"}}, "pruning.depth_remove must be an integer"),
        ({"pruning": {"depth_remove": -1}}, "pruning.depth_remove must be at least 0"),
        (
            {"pruning": {"depth_remove": {"unexpected": 1}}},
            "pruning.depth_remove must be an integer",
        ),
        ({"pruning": {"bypass": {"enabled": "false"}}}, "bypass.enabled must be a boolean"),
        ({"profiles": {"bad": {"tp": "invalid"}}}, "profiles.bad.tp must be an integer"),
        (
            {"profiles": {"bad": {"tp": {"unexpected": 1}}}},
            "profiles.bad.tp must be an integer",
        ),
        (
            {"profiles": {"bad": {"sequence_parallel": "false"}}},
            "profiles.bad.sequence_parallel must be a boolean",
        ),
        (
            {"profiles": {"bad": {"consumers": None}}},
            "profiles.bad.consumers must be a sequence of strings",
        ),
        (
            {"profiles": {"bad": {"consumers": "stage"}}},
            "profiles.bad.consumers must be a sequence of strings",
        ),
        (
            {"profiles": {"bad": {"consumers": [123]}}},
            "profiles.bad.consumers must be a sequence of strings",
        ),
        (
            {"stages": {"depth_importance": {"batch": "invalid"}}},
            "stages.depth_importance.batch must be an integer",
        ),
        (
            {"stages": {"depth_importance": {"batch": {"unexpected": 1}}}},
            "stages.depth_importance.batch must be an integer",
        ),
        (
            {"pruning": {"axes": {"hidden_width": {"values": ["invalid"]}}}},
            "pruning.axes.hidden_width.values must be a sequence of integers",
        ),
        ({"profiles": {"bad": None}}, "profiles.bad must be a mapping"),
        (
            {"infrastructure": {"execution_contract": {"prerun_commands": "echo bad"}}},
            "prerun_commands must be a sequence of strings",
        ),
        (
            {"infrastructure": {"execution_contract": {"postrun_commands": [123]}}},
            "postrun_commands must be a sequence of strings",
        ),
        ({"data": {"subsets": 123}}, "data.subsets must be a string or a sequence of strings"),
        (
            {"data": {"acquisition": {"subsets": [123]}}},
            "data.acquisition.subsets must be a string or a sequence of strings",
        ),
    ],
)
def test_guided_defaults_reject_invalid_leaf_values(tmp_path, defaults, message):
    family_config = tmp_path / "family.yaml"
    family_config.write_text("family: {}\n")
    (tmp_path / "setup_v2_defaults.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "profiles": {"balanced": defaults},
            }
        )
    )

    with pytest.raises(SetupError, match=message):
        get_setup_preset("balanced").resolved_defaults(family_config)


def test_build_bundles_rejects_non_mapping_default_resolution(tmp_path, monkeypatch):
    state = WizardState.start(tmp_path / "campaign", defaults_path=None)
    state.set_collection("default_resolutions", {"broken.path": "invalid"})
    monkeypatch.setattr(bundle_module, "validate_state", lambda state: ())
    monkeypatch.setattr(bundle_module, "render_experiment_v2", lambda state, budget: {})
    monkeypatch.setattr(bundle_module, "render_runner_v2", lambda state, budget: {})
    monkeypatch.setattr(bundle_module, "render_execution_v2", lambda state, budget: {})
    monkeypatch.setattr(
        bundle_module,
        "validate_bundle",
        lambda bundle: SimpleNamespace(valid=True, error=None),
    )
    monkeypatch.setattr(bundle_module, "dry_run_bundle", lambda bundle: "")

    with pytest.raises(SetupError, match="Default resolution 'broken.path' must be a mapping"):
        bundle_module.build_bundles_v2(state.campaign_dir, state)


def test_legacy_state_without_setup_metadata_resumes_in_full_mode(tmp_path):
    state = WizardState.start(tmp_path / "campaign", defaults_path=None)
    state.payload.pop("setup")
    state.save()

    resumed = WizardState.resume(state.path)

    assert resumed.setup_mode == "full"
    assert resumed.preset is None


def test_setup_transitions_repair_non_mapping_setup_metadata(tmp_path):
    state = WizardState.start(tmp_path / "campaign", defaults_path=None)
    state.payload["setup"] = None

    state.set_setup_mode("full")
    state.payload["setup"] = "invalid"
    state.set_preset("balanced")

    resumed = WizardState.resume(state.path)
    assert resumed.setup_mode == "full"
    assert resumed.preset == "balanced"


def test_guided_data_asks_only_for_source_and_uses_nested_defaults(
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
    backend = ScriptedBackend([_CUSTOM_DATA_SOURCE, str(dataset)])
    monkeypatch.setattr(
        "puzzletron_setup.v2.wizard.infer_dataset_modality",
        lambda source: SimpleNamespace(modality="text", evidence="local fixture"),
    )

    assert data_section(
        WizardSession(state, backend, guided=True),
        DefaultsResolver(
            preset_defaults=get_setup_preset("balanced").resolved_defaults(_QWEN_FAMILY_CONFIG),
            file_defaults={
                "data": {
                    "modality": "text",
                    "layout": "padded",
                    "sequence_length": 2048,
                }
            },
        ),
        _context(),
    )

    assert backend.remaining == 0
    assert state.get_field("data.source") == str(dataset.resolve())
    assert state.get_field("data.modality") == "text"
    assert state.get_field("data.layout") == "padded_varlen"
    assert state.get_field("data.sequence_length") == 2048
    assert state.field("data.modality").source == "defaults_file"
    assert state.field("data.layout").source == "defaults_file"


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


@pytest.mark.parametrize(
    ("answers", "file_defaults", "error_path"),
    [
        (
            [_PUZZLE_KD_DATA_SOURCE],
            {"data": {"acquisition": {"seed": "not-an-integer"}}},
            "data.acquisition.seed",
        ),
        (
            [_CUSTOM_DATA_SOURCE, "{dataset}"],
            {"data": {"sequence_length": "not-an-integer"}},
            "data.sequence_length",
        ),
    ],
)
def test_guided_data_rejects_non_integer_defaults(
    tmp_path,
    monkeypatch,
    answers,
    file_defaults,
    error_path,
):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    rendered_answers = [str(dataset) if answer == "{dataset}" else answer for answer in answers]
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

    with pytest.raises(SetupError, match=rf"{error_path} must be an integer"):
        data_section(
            WizardSession(state, ScriptedBackend(rendered_answers), guided=True),
            DefaultsResolver(file_defaults=file_defaults),
            _context(),
        )


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


def test_guided_review_renders_the_actual_non_parameter_mip_constraint(
    tmp_path,
    capsys,
):
    state = WizardState.start(
        tmp_path / "campaign",
        defaults_path=None,
        setup_mode="quick",
        preset="balanced",
    )
    state.set_collection(
        "mip_config",
        {
            "runs": {
                "memory-search": {
                    "constraints": {"memory": {"at": {"serving-default": {"max": "24GiB"}}}},
                    "solver": {"num_solutions": 4},
                }
            }
        },
    )
    state.set_field("model.source", ("model",))

    assert output_review_section(
        WizardSession(state, ScriptedBackend([True]), guided=True),
        DefaultsResolver(),
        {},
    )

    output = capsys.readouterr().out
    assert "constraints:" in output
    assert "memory:" in output
    assert "24GiB" in output
    assert "- model" in output
    assert "parameter_target" not in output


def test_guided_wizard_runs_real_sections_and_generates_valid_bundles(
    tmp_path,
    monkeypatch,
):
    campaign = tmp_path / "campaign"
    model_path = tmp_path / "model"
    dataset = tmp_path / "dataset"
    model_path.mkdir()
    dataset.mkdir()
    inventory = ModelInventory(
        family="qwen3_5",
        descriptor="qwen3_5_text",
        family_config="examples/puzzletron/configs/families/qwen3_5/family.yaml",
        model_type="qwen3_5_text",
        architectures=("Qwen3_5ForCausalLM",),
        multimodal=False,
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
        ),
    )
    inspected = InspectedModel(
        source=str(model_path),
        requested_revision=None,
        resolved_revision=None,
        is_local=True,
        config={
            "model_type": "qwen3_5_text",
            "text_config": {
                "num_hidden_layers": 24,
                "layer_types": ["linear_attention"] * 18 + ["full_attention"] * 6,
            },
        },
        inventory=inventory,
    )
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
                    "execution_contract": {
                        "repository": "/worker/modelopt",
                        "venv": "/worker/venv",
                    }
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
    assert generated.collection("pruning")["depth_remove"] == 1
    assert generated.collection("pruning")["width_importance_samples"] == 8
    assert generated.collection("pruning")["replacement_samples"] == 4
    assert generated.collection("default_resolutions")["pruning.depth_remove"] == {
        "value": 1,
        "source": "preset",
    }
    assert generated.collection("default_resolutions")["pruning.width_importance_samples"] == {
        "value": 8,
        "source": "model_profile",
    }
    resolved_defaults = yaml.safe_load((campaign / "resolved_defaults.yaml").read_text())
    assert resolved_defaults["pruning.depth_remove"] == {
        "value": 1,
        "requested": None,
        "effective": 1,
        "source": "preset",
    }
    pruning = generated.collection("pruning")
    pruning["depth_remove"] = 0
    generated.set_collection("pruning", pruning)
    profiles = generated.collection("parallel_profiles")
    first_profile = next(iter(profiles.values()))
    profiles["secondary"] = {**first_profile, "tp": 2}
    generated.set_collection("parallel_profiles", profiles)

    wizard_module.build_bundles_v2(campaign, generated)

    resolved_defaults = yaml.safe_load((campaign / "resolved_defaults.yaml").read_text())
    assert resolved_defaults["pruning.depth_remove"] == {
        "value": 1,
        "requested": None,
        "effective": 0,
        "source": "preset",
    }
    assert resolved_defaults["profiles"]["effective"] == profiles


def test_guided_section_uses_profile_without_action_prompt(tmp_path):
    state = WizardState.start(
        tmp_path / "campaign",
        defaults_path=None,
        setup_mode="quick",
        preset="balanced",
    )
    backend = ScriptedBackend([])
    session = WizardSession(state, backend, guided=True)

    action = _section_action(
        session,
        "mip",
        "Configure the search.",
        {"num_solutions": 8},
    )

    assert action == "defaults"
    assert backend.remaining == 0


def test_guided_infrastructure_prompts_for_unresolved_worker_paths(tmp_path):
    state = WizardState.start(
        tmp_path / "campaign",
        defaults_path=None,
        setup_mode="quick",
        preset="balanced",
    )
    backend = ScriptedBackend(["defaults", "/worker/modelopt", "/worker/venv"])

    assert infrastructure_section(
        WizardSession(state, backend, guided=True),
        DefaultsResolver(),
        {},
    )

    assert state.get_field("infrastructure.execution_contract.repository") == "/worker/modelopt"
    assert state.get_field("infrastructure.execution_contract.venv") == "/worker/venv"
    assert state.get_field("infrastructure.runner.kind") == "slurm"
    assert backend.remaining == 0


def test_full_section_keeps_the_existing_customize_prompt(tmp_path):
    state = WizardState.start(tmp_path / "campaign", defaults_path=None)
    backend = ScriptedBackend(["customize"])

    action = _section_action(
        WizardSession(state, backend),
        "mip",
        "Configure the search.",
        {"num_solutions": 8},
    )

    assert action == "customize"
    assert backend.remaining == 0


def test_back_reasks_the_previous_prompt_with_replay_intact(tmp_path):
    state = WizardState.start(tmp_path / "campaign", defaults_path=None)
    backend = ScriptedBackend(["one", "two", BACK, "revised"])
    session = WizardSession(state, backend)
    session.begin("data")

    assert session.text("data.one", "One:") == "one"
    assert session.text("data.two", "Two:") == "two"
    assert session.text("data.three", "Three:") is BACK
    target = session.consume_back_target()
    assert target is not None
    assert target.prompt_id == "data.two"

    session.begin("data")
    assert session.text("data.one", "One:") == "one"
    assert session.text("data.two", "Two:") == "revised"
    assert backend.remaining == 0


def test_depth_back_replays_conditional_path_and_zero_removes_resources(tmp_path):
    state = WizardState.start(tmp_path / "campaign", defaults_path=None)
    state.set_collection("stage_resources", {"depth_importance": {"instances": 8}})
    state.set_collection(
        "stage_batches",
        {"depth_importance.micro_batch_size": 8},
    )
    backend = ScriptedBackend(["customize", "subblock", 2, BACK, 0])
    session = WizardSession(state, backend)
    context = {
        "model": SimpleNamespace(
            inventory=SimpleNamespace(num_sublayers=8, num_layers=4),
        )
    }

    assert not depth_section(session, DefaultsResolver(), context)
    target = session.consume_back_target()
    assert target is not None
    assert target.prompt_id == "pruning.depth_remove"

    assert depth_section(session, DefaultsResolver(), context)
    assert state.collection("pruning")["depth_remove"] == 0
    assert "depth_importance" not in state.collection("stage_resources")
    assert "depth_importance.micro_batch_size" not in state.collection("stage_batches")
    assert backend.remaining == 0


def test_escape_returns_back_for_text_select_and_checkbox(monkeypatch):
    questions = []

    class _Bindings:
        def __init__(self):
            self.handlers = {}

        def add(self, key, eager=False):
            assert eager

            def register(handler):
                self.handlers[key] = handler
                return handler

            return register

    class _Application:
        def __init__(self):
            self.key_bindings = _Bindings()
            self.result = None

        def exit(self, *, result):
            self.result = result

    class _Question:
        def __init__(self):
            self.application = _Application()
            questions.append(self)

        def ask(self):
            event = SimpleNamespace(app=self.application)
            self.application.key_bindings.handlers["escape"](event)
            return self.application.result

    class _Questionary:
        @staticmethod
        def Style(value):  # noqa: N802 - mirrors questionary's public constructor
            return value

        @staticmethod
        def Choice(**kwargs):  # noqa: N802 - mirrors questionary's public constructor
            return kwargs

        @staticmethod
        def Separator(value):  # noqa: N802 - mirrors questionary's public constructor
            return value

        @staticmethod
        def text(*args, **kwargs):
            return _Question()

        @staticmethod
        def select(*args, **kwargs):
            return _Question()

        @staticmethod
        def checkbox(*args, **kwargs):
            return _Question()

    monkeypatch.setattr(
        "puzzletron_setup.v2.prompts._questionary",
        lambda: _Questionary(),
    )
    backend = InteractiveBackend()

    assert backend.text("Text:", "") is BACK
    assert backend.select("Select:", [PromptChoice("One", 1)], 1) is BACK
    assert backend.checkbox("Checkbox:", [PromptChoice("One", 1)], [1]) is BACK
    assert len(questions) == 3


def test_escape_binding_supports_a_merged_binding_adapter(monkeypatch):
    registered = {}
    existing_bindings = object()

    class _Bindings:
        def add(self, key, eager=False):
            assert eager

            def register(handler):
                registered[key] = handler
                return handler

            return register

    escape_bindings = _Bindings()
    merged_bindings = object()

    key_binding_module = ModuleType("prompt_toolkit.key_binding")
    key_binding_module.KeyBindings = lambda: escape_bindings

    def merge_key_bindings(bindings):
        assert bindings == [existing_bindings, escape_bindings]
        return merged_bindings

    key_binding_module.merge_key_bindings = merge_key_bindings
    prompt_toolkit_module = ModuleType("prompt_toolkit")
    prompt_toolkit_module.key_binding = key_binding_module
    monkeypatch.setitem(sys.modules, "prompt_toolkit", prompt_toolkit_module)
    monkeypatch.setitem(sys.modules, "prompt_toolkit.key_binding", key_binding_module)

    application = SimpleNamespace(key_bindings=existing_bindings, result=None)
    application.exit = lambda *, result: setattr(application, "result", result)
    question = SimpleNamespace(application=application)

    _bind_escape_back(question)

    assert question.application.key_bindings is merged_bindings
    registered["escape"](SimpleNamespace(app=application))
    assert application.result is BACK
