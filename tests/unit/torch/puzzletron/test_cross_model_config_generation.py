from __future__ import annotations

import dataclasses
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from omegaconf import OmegaConf

from modelopt.torch.puzzletron.campaigns import config_generation
from modelopt.torch.puzzletron.anymodel.models.gpt_oss.gpt_oss_model_descriptor import (
    GptOssModelDescriptor,
)
from modelopt.torch.puzzletron.anymodel.models.qwen3_5.qwen3_5_model_descriptor import (
    Qwen3P5MoeVLModelDescriptor,
    Qwen3P5TextModelDescriptor,
)
from modelopt.torch.puzzletron.anymodel.models.llama.llama_model_descriptor import (
    LlamaModelDescriptor,
)
from modelopt.torch.puzzletron.anymodel.models.nemotron_h.nemotron_h_model_descriptor import (
    NemotronHModelDescriptor,
)
from modelopt.torch.puzzletron.campaigns.activation_passes import compile_activation_passes
from modelopt.torch.puzzletron.campaigns.config_generation import (
    _axis_inventory,
    _deferred_sort_axes,
    _embedding_alignment,
    _stage_parallel,
    generate_campaign_configs,
)
from modelopt.torch.puzzletron.anymodel.capabilities import AxisCapabilities
from modelopt.torch.puzzletron.campaigns.preflight import CampaignPreflight, ModelPreflight
from modelopt.torch.puzzletron.campaigns.schema import default_cross_model_campaign
from modelopt.torch.puzzletron.pipeline_config import pipeline_config_from_path
from modelopt.torch.puzzletron.stages import pipeline


BASE_CONFIG = Path("examples/puzzletron/configs/base.yaml")
NEMOTRON3_FAMILY_CONFIG = Path(
    "examples/puzzletron/configs/families/nemotron3/family.yaml"
)
NANO_PRODUCTION_CONFIG = Path(
    "examples/puzzletron/configs/families/nemotron3/"
    "nano_30b_a3b_bf16/runs/default.yaml"
)
QWEN_FULL_PIPELINE_CONFIG = Path(
    "examples/puzzletron/configs/families/qwen3_5/"
    "qwen3p5_9b/runs/default.yaml"
)
_NANO_ONE_NODE_MESH = {
    "tp": 1,
    "cp": 1,
    "pp": 2,
    "ep": 4,
    "dp_shard": 4,
    "dp_replicate": 1,
}


def test_public_puzzletron_sources_do_not_reference_removed_clean_config_tree() -> None:
    paths = (
        Path("modelopt/torch/puzzletron/campaigns/config_generation.py"),
        Path("examples/puzzletron/generate_cross_model_configs.py"),
        Path("examples/puzzletron/verify_cross_model_configs.py"),
        Path("examples/puzzletron/launch_cross_model_stage_matrix.sh"),
        Path("examples/puzzletron/tools/pretokenize_dataset.py"),
    )

    stale = [str(path) for path in paths if "configs/clean" in path.read_text()]

    assert stale == []


@pytest.mark.parametrize(
    ("path", "expected_by_stage"),
    [
        (
            NANO_PRODUCTION_CONFIG,
            {
                stage: _NANO_ONE_NODE_MESH
                for stage in (
                    "pruning",
                    "bypass",
                    "replacement_scoring",
                    "realize_model",
                )
            },
        ),
        (
            QWEN_FULL_PIPELINE_CONFIG,
            {
                "pruning": {
                    "tp": 1,
                    "cp": 4,
                    "pp": 2,
                    "ep": 1,
                    "dp_shard": 1,
                    "dp_replicate": 1,
                },
                "bypass": {
                    "tp": 1,
                    "cp": 1,
                    "pp": 8,
                    "ep": 1,
                    "dp_shard": 1,
                    "dp_replicate": 1,
                },
                "replacement_scoring": {
                    "tp": 1,
                    "cp": 4,
                    "pp": 2,
                    "ep": 1,
                    "dp_shard": 1,
                    "dp_replicate": 1,
                },
                "realize_model": {
                    "tp": 1,
                    "cp": 4,
                    "pp": 2,
                    "ep": 1,
                    "dp_shard": 1,
                    "dp_replicate": 1,
                },
            },
        ),
    ],
)
def test_production_configs_own_automodel_parallelism_per_stage(
    path, expected_by_stage
) -> None:
    config = pipeline_config_from_path(path)

    assert "parallel" not in config
    assert "recipe_path" not in config
    for section, expected in expected_by_stage.items():
        parallel = config[section]["automodel"]["parallel"]
        assert {key: parallel[key] for key in expected} == expected


def test_sharded_runtime_execution_is_deferred_from_library_stage() -> None:
    assert pipeline._runtime_stats_are_sharded(
        OmegaConf.create(
            {"calc_subblock_stats": {"runtime_stats": {"execution": "sharded"}}}
        )
    )


def test_scenario_utilities_infer_descriptors_from_checkpoints() -> None:
    for script in (
        "examples/puzzletron/prepare_sparse_runtime_stats.py",
        "examples/puzzletron/prepare_sparse_replacement_scoring.py",
        "examples/puzzletron/run_width_depth_mips.py",
    ):
        source = Path(script).read_text()
        assert 'cfg["descriptor"]' not in source
        assert "resolve_descriptor_from_pretrained" in source


def _preflight(campaign) -> CampaignPreflight:
    records = []
    for model in campaign.models:
        records.append(
            ModelPreflight(
                model_id=model.model_id,
                hf_id=model.hf_id,
                immutable_revision=f"sha-{model.model_id}",
                architectures=("Example",),
                model_type="example",
                selected_model_class="Example",
                native_automodel=model.expect_native_automodel,
                descriptor_name="llama",
                tokenizer_available=True,
                processor_available=model.is_multimodal,
                nested_text_config=None,
                mtp_fields=("mtp_num_hidden_layers",) if model.model_id == "qwen35_dense" else (),
                parallel_support=None,
                axis_score_methods={"hidden_width": "minitron_hidden_width"},
                topology=dataclasses.asdict(model.topology),
                errors=(),
            )
        )
    return CampaignPreflight(campaign_fingerprint=campaign.fingerprint, models=tuple(records))


def _config_loader(_model, _record):
    return SimpleNamespace(
        hidden_size=1024,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=128,
        intermediate_size=4096,
        tie_word_embeddings=False,
    )


def test_canonical_base_config_does_not_require_a_descriptor_override() -> None:
    config = yaml.safe_load(BASE_CONFIG.read_text())

    assert "descriptor_override" not in config["model"]
    assert "descriptor" not in config["pruning"]
    assert "descriptor" not in config["replacement_scoring"]
    assert "descriptor" not in config["realize_model"]


def test_nemotron3_family_scores_enabled_hidden_width_axis() -> None:
    config = yaml.safe_load(NEMOTRON3_FAMILY_CONFIG.read_text())
    passes = config["pruning"]["activation_passes"]

    hidden_passes = [
        activation_pass
        for activation_pass in passes
        if activation_pass["activation_hooks_kwargs"]["method"]
        == "minitron_hidden_width"
    ]

    assert [activation_pass["name"] for activation_pass in hidden_passes] == ["hidden_width"]


def test_stage_parallel_uses_descriptor_sequence_parallel_capability() -> None:
    campaign = default_cross_model_campaign()
    models = {model.model_id: model for model in campaign.models}
    config = _config_loader(None, None)

    qwen_capabilities = Qwen3P5TextModelDescriptor.puzzletron_capabilities(config)
    llama_capabilities = LlamaModelDescriptor.puzzletron_capabilities(config)

    assert _stage_parallel(
        models["qwen35_dense"], capabilities=qwen_capabilities
    )["sequence_parallel"] is False
    assert _stage_parallel(
        models["llama31_8b"], capabilities=llama_capabilities
    )["sequence_parallel"] is True


def test_descriptors_default_dynamic_shape_diagnosis_to_eager_execution() -> None:
    assert LlamaModelDescriptor.stage_execution_policy() == {
        "torch_compile_disabled_stages": ("activation_diagnostic",)
    }
    assert Qwen3P5TextModelDescriptor.stage_execution_policy() == {
        "torch_compile_disabled_stages": ("activation_diagnostic",)
    }


def test_embedding_alignment_is_compatible_with_teacher_and_reduced_widths() -> None:
    assert _embedding_alignment({"teacher_value": 1024, "values": [512]}) == 128
    assert _embedding_alignment({"teacher_value": 2880, "values": [1440]}) == 32


def test_equivalence_tolerances_are_descriptor_owned_for_sensitive_storage_formats() -> None:
    ordinary = config_generation._equivalence_tolerances(LlamaModelDescriptor)
    quantized = config_generation._equivalence_tolerances(GptOssModelDescriptor)
    fused_moe = config_generation._equivalence_tolerances(Qwen3P5MoeVLModelDescriptor)

    assert ordinary["max_abs_lm_loss_delta"] == 1.0e-3
    assert ordinary["max_kl_div"] == 1.0e-2
    assert quantized["max_abs_lm_loss_delta"] == 2.5e-2
    assert quantized["max_kl_div"] == 5.0e-2
    assert fused_moe["max_abs_lm_loss_delta"] == 1.5e-2
    assert fused_moe["max_kl_div"] == 1.0e-2
    assert fused_moe["min_top_1_logit_agreement"] == 0.95


def test_unscored_sortable_axes_are_deferred_from_checkpoint_permutation() -> None:
    capabilities = SimpleNamespace(
        axes={
            "scored": AxisCapabilities(
                axis_id="scored", subblock_kind="model", field="width", sortable=True
            ),
            "latent_unscored": AxisCapabilities(
                axis_id="latent_unscored", subblock_kind="moe", field="rank", sortable=True
            ),
            "runtime_variant": AxisCapabilities(
                axis_id="runtime_variant",
                subblock_kind="attention",
                field="window",
                sortable=True,
                variant_only=True,
            ),
        }
    )

    assert _deferred_sort_axes(
        capabilities, [{"axis_id": "scored", "method": "minitron"}]
    ) == ["latent_unscored"]


def test_stage_parallel_separates_fsdp_sharding_and_replication() -> None:
    campaign = default_cross_model_campaign()
    model = campaign.models[0]
    config = _config_loader(None, None)
    capabilities = Qwen3P5TextModelDescriptor.puzzletron_capabilities(config)

    parallel = _stage_parallel(model, capabilities=capabilities)

    assert parallel["dp_shard"] == model.topology.ep
    assert parallel["dp_replicate"] == model.topology.fsdp


def test_moe_top_k_is_a_generic_variant_axis_for_gpt_oss_and_nano() -> None:
    gpt_config = SimpleNamespace(
        hidden_size=2880,
        num_attention_heads=64,
        num_key_value_heads=8,
        head_dim=64,
        intermediate_size=2880,
        num_local_experts=32,
        experts_per_token=4,
        sliding_window=128,
    )
    nano_config = SimpleNamespace(
        hidden_size=2688,
        num_attention_heads=32,
        num_key_value_heads=2,
        head_dim=64,
        intermediate_size=1856,
        num_experts=128,
        num_experts_per_tok=6,
        shared_expert_intermediate_size=3712,
        mamba_num_heads=64,
        mamba_head_dim=64,
    )

    gpt_axes, _ = _axis_inventory(GptOssModelDescriptor, gpt_config)
    nano_axes, _ = _axis_inventory(NemotronHModelDescriptor, nano_config)

    assert gpt_axes["moe_top_k"] == {
        "enabled": True,
        "teacher_value": 4,
        "values": [2],
    }
    assert nano_axes["moe_top_k"] == {
        "enabled": True,
        "teacher_value": 6,
        "values": [3],
    }
    assert gpt_axes["sliding_window_size"]["values"] == [64, 128, "full"]


def test_qwen_dense_inventory_excludes_moe_axes_and_prefers_iterative_ffn() -> None:
    config = SimpleNamespace(
        model_type="qwen3_5_text",
        hidden_size=1024,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=128,
        intermediate_size=3584,
        linear_num_key_heads=16,
        linear_num_value_heads=16,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
    )

    search_axes, activation_axes = _axis_inventory(Qwen3P5TextModelDescriptor, config)

    assert not any(axis_id.startswith("moe_") for axis_id in search_axes)
    methods = {entry["axis_id"]: entry["method"] for entry in activation_axes}
    assert methods["ffn_intermediate"] == "iterative"


def test_every_generated_search_axis_declares_vllm_control() -> None:
    config = SimpleNamespace(
        model_type="qwen3_5_text",
        hidden_size=1024,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=128,
        intermediate_size=3584,
        linear_num_key_heads=16,
        linear_num_value_heads=16,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
    )

    search_axes, _ = _axis_inventory(Qwen3P5TextModelDescriptor, config)
    capabilities = Qwen3P5TextModelDescriptor.puzzletron_capabilities(config)

    assert search_axes
    assert {
        axis_id
        for axis_id in search_axes
        if not capabilities.axes[axis_id].vllm_export
    } == set()


def test_compiler_groups_shared_scorers_and_serializes_generic_targets() -> None:
    config = SimpleNamespace(
        hidden_size=1024,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=128,
        intermediate_size=4096,
    )
    _, axes = _axis_inventory(LlamaModelDescriptor, config)

    passes = compile_activation_passes(LlamaModelDescriptor, config, axes)

    assert [item["name"] for item in passes] == [
        "hidden_width",
        "ffn_intermediate",
        "attention_grouped",
    ]
    assert passes[1]["axis_ids"] == ["ffn_intermediate"]
    assert passes[1]["activation_hooks_kwargs"]["method"] == "iterative"
    assert passes[1]["pruning_mixin"]["layer_descriptor"]["down_proj_name"] == (
        "mlp.down_proj"
    )
    assert passes[2]["axis_ids"] == ["kv_groups", "q_heads_per_group"]
    assert passes[2]["activation_hooks_kwargs"]["method"] == (
        "grouped_attention_contribution"
    )


def test_qwen_moe_compiler_has_attention_gdn_and_grouped_expert_targets() -> None:
    text_config = SimpleNamespace(
        model_type="qwen3_5_moe_text",
        hidden_size=2048,
        num_attention_heads=16,
        num_key_value_heads=2,
        head_dim=128,
        num_experts=64,
        num_experts_per_tok=8,
        moe_intermediate_size=1024,
        shared_expert_intermediate_size=1024,
        linear_num_key_heads=16,
        linear_num_value_heads=16,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
    )
    config = SimpleNamespace(model_type="qwen3_5_moe", text_config=text_config)
    _, axes = _axis_inventory(Qwen3P5MoeVLModelDescriptor, config)

    passes = compile_activation_passes(Qwen3P5MoeVLModelDescriptor, config, axes)
    by_name = {item["name"]: item for item in passes}

    assert by_name["attention_grouped"]["pruning_mixin"]["layer_descriptor"][
        "o_proj_name"
    ] == "self_attn.o_proj"
    assert by_name["gdn_activation"]["pruning_mixin"]["layer_descriptor"][
        "target_name"
    ] == "linear_attn"
    assert by_name["moe_experts"]["pruning_mixin"]["layer_descriptor"][
        "require_attrs"
    ] == ["gate", "experts"]


def test_nemotron_compiler_has_dense_ffn_target_for_hybrid_layers() -> None:
    passes = compile_activation_passes(
        NemotronHModelDescriptor,
        SimpleNamespace(),
        [{"axis_id": "ffn_intermediate", "method": "iterative"}],
    )

    descriptor = passes[0]["pruning_mixin"]["layer_descriptor"]
    assert descriptor["down_proj_name"] == "mixer.down_proj"
    assert descriptor["ffn_prefix_name"] == "model.layers.{layer_idx}.mixer"


def test_nemotron_inventory_generates_aligned_reduced_hidden_width() -> None:
    config = SimpleNamespace(
        hidden_size=2688,
        num_hidden_layers=52,
        num_attention_heads=32,
        num_key_value_heads=2,
        head_dim=128,
        n_routed_experts=128,
        moe_intermediate_size=1856,
        moe_shared_expert_intermediate_size=3712,
        num_experts_per_tok=6,
        mamba_num_heads=64,
        mamba_head_dim=64,
        tie_word_embeddings=False,
    )

    axes, activation = _axis_inventory(NemotronHModelDescriptor, config)

    assert axes["hidden_width"] == {
        "enabled": True,
        "teacher_value": 2688,
        "values": [1344],
    }
    assert _embedding_alignment(axes["hidden_width"]) == 64
    assert {entry["axis_id"] for entry in activation} >= {"hidden_width"}
