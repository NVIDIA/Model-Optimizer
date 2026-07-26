from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from modelopt.torch.puzzletron.campaigns.preflight import (
    ModelMetadata,
    _find_mtp_fields,
    run_preflight,
)
from modelopt.torch.puzzletron.campaigns.schema import (
    CampaignStageIdentity,
    DatasetKind,
    ModelKind,
    ParallelTopology,
    default_cross_model_campaign,
    load_campaign,
)

EXPECTED_MODEL_IDS = {
    "qwen35_dense",
    "llama31_8b",
    "gpt_oss_20b",
    "qwen36_35b_a3b",
    "nemotron3_nano_30b_a3b",
}


def test_default_campaign_has_all_models_once() -> None:
    campaign = default_cross_model_campaign()

    assert len(campaign.models) == len(EXPECTED_MODEL_IDS)
    assert {model.model_id for model in campaign.models} == EXPECTED_MODEL_IDS
    assert len({model.hf_id for model in campaign.models}) == len(EXPECTED_MODEL_IDS)


def test_default_campaign_uses_the_approved_model_gate_order() -> None:
    campaign = default_cross_model_campaign()

    assert [model.model_id for model in campaign.models] == [
        "qwen35_dense",
        "nemotron3_nano_30b_a3b",
        "gpt_oss_20b",
        "qwen36_35b_a3b",
        "llama31_8b",
    ]


def test_default_campaign_uses_tiny_axis_complete_smoke_budget() -> None:
    campaign = default_cross_model_campaign()

    # Qwen multimodal samples need room for visual tokens; the canonical
    # acceptance smoke keeps the requested 2K pack even when sample/step
    # counts are tiny.
    assert campaign.sequence_length == 2048
    assert campaign.activation_samples == 16
    assert campaign.kd_steps == 8


def test_force_hf_is_disabled_for_every_model() -> None:
    campaign = default_cross_model_campaign()

    assert all(model.force_hf is False for model in campaign.models)


def test_dense_and_moe_topologies_use_requested_ranks_or_documented_native_exception() -> None:
    campaign = default_cross_model_campaign()
    exception_topologies = {
        "llama31_8b": ParallelTopology(tp=2, cp=1, pp=2, fsdp=2, ep=1),
        "gpt_oss_20b": ParallelTopology(tp=1, cp=1, pp=2, fsdp=1, ep=2),
        "qwen36_35b_a3b": ParallelTopology(tp=1, cp=1, pp=2, fsdp=1, ep=2),
        "nemotron3_nano_30b_a3b": ParallelTopology(
            tp=1, cp=1, pp=2, fsdp=1, ep=2
        ),
    }
    assert {
        model.model_id for model in campaign.models if model.topology_exception is not None
    } == set(exception_topologies)
    for model in campaign.models:
        if model.topology_exception is not None:
            assert model.topology == exception_topologies[model.model_id]
            continue
        assert model.topology.world_size == 16
        if model.model_kind is ModelKind.MOE:
            assert model.topology == ParallelTopology(tp=2, cp=2, pp=2, fsdp=1, ep=2)
        else:
            assert model.topology == ParallelTopology(tp=2, cp=2, pp=2, fsdp=2, ep=1)


def test_modality_selects_the_canonical_dataset() -> None:
    campaign = default_cross_model_campaign()

    for model in campaign.models:
        if model.is_multimodal:
            assert model.dataset is DatasetKind.PINNED_INTERSYN
        else:
            assert model.dataset is DatasetKind.PUZZLE_KD_TEXT


def test_campaign_fingerprint_is_stable_and_sensitive() -> None:
    campaign = default_cross_model_campaign()
    same = default_cross_model_campaign()

    assert campaign.fingerprint == same.fingerprint
    changed_model = dataclasses.replace(campaign.models[0], hf_revision="different")
    changed = dataclasses.replace(campaign, models=(changed_model, *campaign.models[1:]))
    assert changed.fingerprint != campaign.fingerprint


def test_force_hf_true_is_rejected() -> None:
    campaign = default_cross_model_campaign()
    invalid = dataclasses.replace(campaign.models[0], force_hf=True)

    with pytest.raises(ValueError, match="force_hf=False"):
        invalid.validate()


def test_moe_cannot_use_fsdp_two() -> None:
    campaign = default_cross_model_campaign()
    moe = next(model for model in campaign.models if model.model_kind is ModelKind.MOE)
    invalid = dataclasses.replace(moe, topology=ParallelTopology(tp=2, cp=2, pp=2, fsdp=2, ep=1))

    with pytest.raises(ValueError, match="EP=2.*FSDP=1"):
        invalid.validate()


def test_text_model_cannot_use_multimodal_dataset() -> None:
    campaign = default_cross_model_campaign()
    text = next(model for model in campaign.models if not model.is_multimodal)
    invalid = dataclasses.replace(text, dataset=DatasetKind.PINNED_INTERSYN)

    with pytest.raises(ValueError, match="text-only"):
        invalid.validate()


def test_duplicate_ids_are_rejected() -> None:
    campaign = default_cross_model_campaign()
    duplicate = dataclasses.replace(campaign.models[1], model_id=campaign.models[0].model_id)
    invalid = dataclasses.replace(campaign, models=(campaign.models[0], duplicate))

    with pytest.raises(ValueError, match="model_id.*unique"):
        invalid.validate()


def test_stage_identity_includes_model_stage_campaign_and_upstream() -> None:
    campaign = default_cross_model_campaign()
    identity = CampaignStageIdentity.create(
        campaign,
        model_id="llama31_8b",
        stage="activation",
        upstream_identities=("conversion-a",),
    )
    changed = CampaignStageIdentity.create(
        campaign,
        model_id="llama31_8b",
        stage="activation",
        upstream_identities=("conversion-b",),
    )

    assert identity.model_id == "llama31_8b"
    assert identity.stage == "activation"
    assert identity.campaign_fingerprint == campaign.fingerprint
    assert identity.fingerprint != changed.fingerprint


def test_load_campaign_validates_yaml(tmp_path: Path) -> None:
    config = tmp_path / "campaign.yaml"
    config.write_text(
        """
sequence_length: 2048
activation_samples: 16
kd_steps: 16
models:
  - model_id: llama
    hf_id: meta-llama/Llama-3.1-8B-Instruct
    hf_revision: pinned-revision
    model_kind: dense
    is_multimodal: false
    dataset: puzzle_kd_text
    force_hf: false
    expect_native_automodel: true
    mtp_policy: if_present
    topology: {tp: 2, cp: 2, pp: 2, fsdp: 2, ep: 1}
""".strip()
    )

    campaign = load_campaign(config)

    assert len(campaign.models) == 1
    assert campaign.models[0].hf_revision == "pinned-revision"


def test_preflight_records_resolution_without_loading_weights() -> None:
    campaign = default_cross_model_campaign()

    def load_metadata(model):
        return ModelMetadata(
            immutable_revision=f"sha-{model.model_id}",
            architectures=("ExampleForCausalLM",),
            model_type="example",
            selected_model_class="NativeExample" if model.expect_native_automodel else "HFExample",
            native_automodel=model.expect_native_automodel,
            descriptor_name="generic_decoder",
            tokenizer_available=True,
            processor_available=model.is_multimodal,
            parallel_support={"tp": True, "cp": True, "pp": True, "ep": True},
            mtp_fields=("mtp_num_hidden_layers",) if model.model_id == "qwen36_35b_a3b" else (),
        )

    result = run_preflight(campaign, load_metadata)

    assert result.success
    assert len(result.models) == len(EXPECTED_MODEL_IDS)
    assert all(model.immutable_revision.startswith("sha-") for model in result.models)
    assert all(model.native_automodel for model in result.models)


def test_preflight_fails_closed_on_missing_descriptor() -> None:
    campaign = dataclasses.replace(default_cross_model_campaign(), models=(default_cross_model_campaign().models[0],))

    def load_metadata(_model):
        return ModelMetadata(
            immutable_revision="sha",
            architectures=("Example",),
            model_type="example",
            selected_model_class="Example",
            native_automodel=True,
            descriptor_name=None,
            tokenizer_available=True,
            processor_available=True,
        )

    result = run_preflight(campaign, load_metadata)

    assert not result.success
    assert "descriptor" in result.models[0].errors[0]


def test_preflight_rejects_requested_parallel_dimension_unsupported_by_model() -> None:
    model = default_cross_model_campaign().models[0]
    campaign = dataclasses.replace(default_cross_model_campaign(), models=(model,))

    def load_metadata(_model):
        return ModelMetadata(
            immutable_revision="sha",
            architectures=("Example",),
            model_type="example",
            selected_model_class="NativeExample",
            native_automodel=True,
            descriptor_name="example",
            tokenizer_available=True,
            processor_available=True,
            parallel_support={"tp": True, "cp": True, "pp": False, "ep": False},
        )

    result = run_preflight(campaign, load_metadata)

    assert not result.success
    assert "PP=2" in " ".join(result.models[0].errors)


def test_preflight_preserves_probe_stage_errors_and_partial_metadata() -> None:
    campaign = dataclasses.replace(default_cross_model_campaign(), models=(default_cross_model_campaign().models[0],))

    def load_metadata(_model):
        return ModelMetadata(
            immutable_revision="sha",
            architectures=("Example",),
            model_type="example",
            selected_model_class="NativeExample",
            native_automodel=True,
            descriptor_name=None,
            tokenizer_available=True,
            processor_available=True,
            probe_errors=("descriptor: unsupported example",),
        )

    result = run_preflight(campaign, load_metadata)

    assert result.models[0].immutable_revision == "sha"
    assert "descriptor: unsupported example" in result.models[0].errors


def test_mtp_field_scan_accepts_integer_config_keys() -> None:
    config = {"layers": {0: {"mtp_num_hidden_layers": 1}, 1: {"width": 128}}}

    assert _find_mtp_fields(config) == ("layers.0.mtp_num_hidden_layers",)
