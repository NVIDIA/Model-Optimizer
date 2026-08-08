# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest
import yaml

import puzzletron_setup.v2.wizard as wizard_module
from puzzletron_setup import SetupError
from puzzletron_setup.inspection import InspectedModel
from puzzletron_setup.profiles import AxisInventory, ModelInventory
from puzzletron_setup.v2.cli import _parser
from puzzletron_setup.v2.defaults import DefaultsResolver
from puzzletron_setup.v2.presets import QUICK_SETUP_PRESETS, get_setup_preset
from puzzletron_setup.v2.prompts import (
    BACK,
    InteractiveBackend,
    PromptChoice,
    ScriptedBackend,
    _bind_escape_back,
)
from puzzletron_setup.v2.session import WizardSession
from puzzletron_setup.v2.state import WizardState
from puzzletron_setup.v2.wizard import (
    _CUSTOM_DATA_SOURCE,
    _CUSTOM_MODEL_SOURCE,
    _fresh_state,
    _section_action,
    data_section,
    depth_section,
    infrastructure_section,
    output_review_section,
)

_QWEN_FAMILY_CONFIG = "examples/puzzletron/configs/families/qwen3_5/family.yaml"
_NEMOTRON_FAMILY_CONFIG = "examples/puzzletron/configs/families/nemotron3/family.yaml"


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
    resolver = DefaultsResolver(
        preset_defaults=get_setup_preset("smoke").resolved_defaults(
            _QWEN_FAMILY_CONFIG,
            qwen_0p8b,
        ),
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
    monkeypatch.setattr(wizard_module, "_refresh_legacy_state", lambda state: None)
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
    monkeypatch.setattr(wizard_module, "_refresh_legacy_state", lambda state: None)
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
    monkeypatch.setattr(wizard_module, "_refresh_legacy_state", lambda state: None)
    monkeypatch.setattr(wizard_module, "build_bundles_v2", lambda campaign, state: None)

    wizard_module.run_wizard_v2(
        resume=state.campaign_dir,
        defaults_path=replacement,
        backend=ScriptedBackend([]),
    )

    assert WizardState.resume(state.path).defaults_path == replacement.resolve()


def test_legacy_state_without_setup_metadata_resumes_in_full_mode(tmp_path):
    state = WizardState.start(tmp_path / "campaign", defaults_path=None)
    state.payload.pop("setup")
    state.save()

    resumed = WizardState.resume(state.path)

    assert resumed.setup_mode == "full"
    assert resumed.preset is None


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

    assert output_review_section(
        WizardSession(state, ScriptedBackend([True]), guided=True),
        DefaultsResolver(),
        {},
    )

    output = capsys.readouterr().out
    assert "constraints:" in output
    assert "memory:" in output
    assert "24GiB" in output
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
    backend = ScriptedBackend(
        [
            "smoke",
            str(campaign),
            _CUSTOM_MODEL_SOURCE,
            str(model_path),
            _CUSTOM_DATA_SOURCE,
            str(dataset),
            "defaults",
            "/worker/modelopt",
            "/worker/venv",
            True,
        ]
    )

    result = wizard_module.run_wizard_v2(
        resume=None,
        defaults_path=None,
        backend=backend,
    )

    assert result == campaign.resolve()
    assert backend.remaining == 0
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
