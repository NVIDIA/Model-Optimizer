from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import pytest
import torch

from modelopt.torch.puzzletron.anymodel.capabilities import (
    AxisCapabilities,
    CapabilityValidationError,
    default_capabilities,
    validate_capabilities,
)
from modelopt.torch.puzzletron.anymodel.models.gpt_oss.gpt_oss_converter import GptOssConverter
from modelopt.torch.puzzletron.anymodel.models.gpt_oss.gpt_oss_model_descriptor import (
    GptOssModelDescriptor,
)
from modelopt.torch.puzzletron.anymodel.models.llama.llama_model_descriptor import (
    LlamaModelDescriptor,
)
from modelopt.torch.puzzletron.anymodel.models.nemotron_h.nemotron_h_model_descriptor import (
    NemotronHModelDescriptor,
)
from modelopt.torch.puzzletron.anymodel.models.qwen3_5.qwen3_5_model_descriptor import (
    Qwen3P5TextModelDescriptor,
    Qwen3P5VLModelDescriptor,
)
from modelopt.torch.puzzletron.anymodel.registry import infer_descriptor_name, resolve_descriptor
from modelopt.torch.puzzletron.block_config import (
    AttentionConfig,
    BlockConfig,
    FFNConfig,
    MambaConfig,
    MoEConfig,
)
from modelopt.torch.puzzletron.stage_runner import (
    StageResult,
    _preflight,
    _resolve_capabilities,
    run_stage,
)


@pytest.mark.parametrize(
    ("model_type", "architecture", "expected", "expected_descriptor"),
    [
        (
            "qwen3_5_text",
            "Qwen3_5ForConditionalGeneration",
            "qwen3_5",
            Qwen3P5VLModelDescriptor,
        ),
        (
            "qwen3_5",
            "Qwen3_5ForCausalLM",
            "qwen3_5_text",
            Qwen3P5TextModelDescriptor,
        ),
        (
            "qwen3_6_text",
            "Qwen3_6ForConditionalGeneration",
            "qwen3_6",
            Qwen3P5VLModelDescriptor,
        ),
        (
            "qwen3_6",
            "Qwen3_6ForCausalLM",
            "qwen3_6_text",
            Qwen3P5TextModelDescriptor,
        ),
    ],
    ids=["qwen3.5-vlm", "qwen3.5-text", "qwen3.6-vlm", "qwen3.6-text"],
)
def test_qwen_dense_architecture_wins_over_conflicting_model_type(
    model_type, architecture, expected, expected_descriptor
) -> None:
    config = SimpleNamespace(model_type=model_type, architectures=[architecture])

    assert infer_descriptor_name(config) == (
        expected,
        f"architectures contains {architecture}",
    )
    resolution = resolve_descriptor(config)
    assert resolution.name == expected
    assert resolution.descriptor is expected_descriptor


@pytest.mark.parametrize(
    ("model_type", "architecture", "expected"),
    [
        ("qwen3_5_moe", "Qwen3_5MoeForConditionalGeneration", "qwen3_5_moe"),
        ("qwen3_5_moe_text", "Qwen3_5MoeForCausalLM", "qwen3_5_moe_text"),
    ],
)
def test_cross_model_registry_resolution(model_type, architecture, expected) -> None:
    config = SimpleNamespace(model_type=model_type, architectures=[architecture])

    assert infer_descriptor_name(config)[0] == expected
    assert resolve_descriptor(config).name == expected


def _text_config(**overrides):
    values = {
        "model_type": "fixture",
        "hidden_size": 2048,
        "num_hidden_layers": 40,
        "num_attention_heads": 16,
        "num_key_value_heads": 2,
        "head_dim": 128,
        "intermediate_size": 8192,
        "num_experts": 128,
        "moe_intermediate_size": 512,
        "shared_expert_intermediate_size": 512,
        "tie_word_embeddings": False,
        "enable_moe_block": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_qwen_moe_contract_combines_moe_gdn_attention_vlm_and_mtp() -> None:
    text = _text_config(
        model_type="qwen3_5_moe_text",
        layer_types=["linear_attention", "full_attention"],
        linear_num_key_heads=16,
        linear_num_value_heads=32,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
    )
    config = SimpleNamespace(
        model_type="qwen3_5_moe",
        architectures=["Qwen3_5MoeForConditionalGeneration"],
        text_config=text,
    )
    resolution = resolve_descriptor(config)
    contract = resolution.descriptor.generic_decoder_contract(config)
    capabilities = resolution.descriptor.puzzletron_capabilities(config)

    assert contract.routed_moe is not None
    assert contract.vision is not None
    assert contract.mtp is not None
    assert {
        "gdn_key_groups",
        "gdn_value_heads_per_group",
        "gdn_key_head_dim",
        "gdn_value_head_dim",
        "moe_experts",
        "moe_expert_intermediate",
        "moe_shared_expert_intermediate",
    } <= set(capabilities.axes)
    assert resolution.descriptor.sorted_teacher_layout_kwargs(text) == {
        "q_gate_row_group": 1,
        "mamba_module": "linear_attn",
        "gated_delta_net": True,
        "moe_fused_expert_subnames": (
            "experts.gate_up_proj",
            "experts.down_proj",
        ),
        "moe_fused_gate_up_subnames": ("experts.gate_up_proj",),
        "moe_fused_down_subnames": ("experts.down_proj",),
        "moe_shared_expert_subname": "shared_expert",
        "moe_shared_gate_subname": "gate_proj",
    }


def test_qwen_moe_embedding_spec_covers_mtp_attention_gdn_and_experts() -> None:
    config = _text_config(
        model_type="qwen3_5_moe_text",
        hidden_size=8,
        architectures=["Qwen3_5MoeForCausalLM"],
    )
    descriptor = resolve_descriptor(config).descriptor
    spec = descriptor.embedding_pruning_spec(config, widths=(8, 4), alignment=1)
    state = {
        "mtp.fc.weight": torch.zeros(8, 16),
        "mtp.pre_fc_norm_embedding.weight": torch.zeros(8),
        "mtp.pre_fc_norm_hidden.weight": torch.zeros(8),
        "mtp.norm.weight": torch.zeros(8),
        "mtp.layers.0.input_layernorm.weight": torch.zeros(8),
        "mtp.layers.0.post_attention_layernorm.weight": torch.zeros(8),
        "mtp.layers.0.self_attn.q_proj.weight": torch.zeros(16, 8),
        "mtp.layers.0.self_attn.o_proj.weight": torch.zeros(8, 16),
        "mtp.layers.0.linear_attn.in_proj_qkv.weight": torch.zeros(24, 8),
        "mtp.layers.0.linear_attn.out_proj.weight": torch.zeros(8, 16),
        "mtp.layers.0.mlp.gate.weight": torch.zeros(4, 8),
        "mtp.layers.0.mlp.experts.gate_up_proj": torch.zeros(4, 12, 8),
        "mtp.layers.0.mlp.experts.down_proj": torch.zeros(4, 8, 6),
        "mtp.layers.0.mlp.shared_expert.gate_proj.weight": torch.zeros(6, 8),
        "mtp.layers.0.mlp.shared_expert.down_proj.weight": torch.zeros(8, 6),
        "mtp.layers.0.mlp.shared_expert_gate.weight": torch.zeros(1, 8),
    }

    audit = spec.audit_state_dict(state)

    assert set(audit["handled"]) == set(state)


@pytest.mark.parametrize(
    ("layer_idx", "window", "expected_types", "expected_window"),
    [
        (0, "full", ["full_attention", "full_attention"], 128),
        (1, 64, ["sliding_attention", "sliding_attention"], 64),
    ],
)
def test_gpt_oss_native_layer_config_applies_window_and_attention_type(
    layer_idx, window, expected_types, expected_window
) -> None:
    config = SimpleNamespace(
        layer_types=["sliding_attention", "full_attention"],
        sliding_window=128,
    )
    block = BlockConfig(
        subblock_configs=(
            AttentionConfig(
                num_query_heads=4,
                num_kv_heads=2,
                sliding_window_size=window,
            ),
        )
    )

    GptOssModelDescriptor.patch_layer_config(config, block, layer_idx)

    assert config.layer_types == expected_types
    assert config.sliding_window == expected_window


def test_gpt_oss_converter_preserves_each_layers_attention_window() -> None:
    config = SimpleNamespace(
        num_hidden_layers=4,
        num_local_experts=32,
        experts_per_token=4,
        intermediate_size=2880,
        num_key_value_heads=8,
        num_attention_heads=64,
        sliding_window=128,
        layer_types=[
            "sliding_attention",
            "full_attention",
            "sliding_attention",
            "full_attention",
        ],
    )

    blocks = [
        BlockConfig(**block)
        for block in GptOssConverter.create_block_configs_from_main_config(config)
    ]

    assert [
        block.require_subblock("attention").sliding_window_size for block in blocks
    ] == [128, "full", 128, "full"]


def test_gpt_oss_hf_and_native_share_contract_field_mapping() -> None:
    block = BlockConfig(
        subblock_configs=(
            AttentionConfig(num_query_heads=8, num_kv_heads=2),
            MoEConfig(num_experts=16, expert_intermediate_size=1440, top_k=2),
        )
    )

    assert GptOssModelDescriptor.block_config_to_layer_overrides(block) == {
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "num_local_experts": 16,
        "intermediate_size": 1440,
        "num_experts_per_tok": 2,
    }


def test_gpt_oss_weight_groups_accept_native_automodel_moe_names() -> None:
    native_names = [
        "model.layers.0.post_attention_layernorm.weight",
        "model.layers.0.mlp.gate.weight",
        "model.layers.0.mlp.gate.bias",
        "model.layers.0.mlp.experts.gate_and_up_projs",
        "model.layers.0.mlp.experts.down_projs",
    ]

    groups = GptOssModelDescriptor.get_weight_groups(native_names, 1)

    assert groups["block_0_ffn"] == native_names


def test_gpt_oss_native_automodel_uses_flex_attention_without_te() -> None:
    assert GptOssModelDescriptor.automodel_model_kwargs(
        object(), distributed={"tp_size": 2, "cp_size": 2, "ep_size": 2}
    ) == {"backend": {"attn": "flex"}}


def test_nemotron_weight_groups_accept_native_automodel_moe_names() -> None:
    native_block_names = [
        "model.layers.0.norm.weight",
        "model.layers.0.mixer.gate.weight",
        "model.layers.0.mixer.gate.e_score_correction_bias",
        "model.layers.0.mixer.experts.gate_and_up_projs",
        "model.layers.0.mixer.experts.down_projs",
    ]
    native_output_names = ["model.norm.weight", "lm_head.weight"]

    groups = NemotronHModelDescriptor.get_weight_groups(
        native_block_names + native_output_names, 1
    )

    assert groups["block_0_ffn"] == native_block_names
    assert groups["lm_head"] == native_output_names


def test_nemotron_hidden_width_spec_covers_hf_and_native_hybrid_tensors() -> None:
    config = SimpleNamespace(
        hidden_size=2688,
        tie_word_embeddings=False,
        moe_latent_size=640,
    )
    latent_size = 640
    spec = NemotronHModelDescriptor.embedding_pruning_spec(
        config, widths=(2688, 1344), alignment=64
    )
    state = {
        "backbone.embeddings.weight": torch.zeros(8, 2688),
        "backbone.layers.0.norm.weight": torch.zeros(2688),
        "backbone.layers.0.mixer.in_proj.weight": torch.zeros(16, 2688),
        "backbone.layers.0.mixer.out_proj.weight": torch.zeros(2688, 16),
        "backbone.layers.1.mixer.q_proj.weight": torch.zeros(8, 2688),
        "backbone.layers.1.mixer.o_proj.weight": torch.zeros(2688, 8),
        "backbone.layers.2.mixer.gate.weight": torch.zeros(4, 2688),
        "backbone.layers.2.mixer.fc1_latent_proj.weight": torch.zeros(latent_size, 2688),
        "backbone.layers.2.mixer.fc2_latent_proj.weight": torch.zeros(2688, latent_size),
        "backbone.layers.2.mixer.experts.0.up_proj.weight": torch.zeros(6, latent_size),
        "backbone.layers.2.mixer.experts.0.down_proj.weight": torch.zeros(latent_size, 6),
        "backbone.layers.2.mixer.shared_experts.up_proj.weight": torch.zeros(12, 2688),
        "backbone.layers.2.mixer.shared_experts.down_proj.weight": torch.zeros(2688, 12),
        "backbone.norm_f.weight": torch.zeros(2688),
        "lm_head.weight": torch.zeros(8, 2688),
        "model.layers.3.mixer.fc1_latent_proj.weight": torch.zeros(latent_size, 2688),
        "model.layers.3.mixer.fc2_latent_proj.weight": torch.zeros(2688, latent_size),
        "model.layers.3.mixer.experts.gate_and_up_projs": torch.zeros(4, latent_size, 6),
        "model.layers.3.mixer.experts.down_projs": torch.zeros(4, 6, latent_size),
        "mtp.layers.0.eh_proj.weight": torch.zeros(2688, 2 * 2688),
        "mtp.layers.0.enorm.weight": torch.zeros(2688),
        "mtp.layers.0.hnorm.weight": torch.zeros(2688),
        "mtp.layers.0.norm.weight": torch.zeros(2688),
        "mtp.layers.0.mixer.q_proj.weight": torch.zeros(2688, 2688),
        "mtp.layers.0.mixer.k_proj.weight": torch.zeros(4, 2688),
        "mtp.layers.0.mixer.v_proj.weight": torch.zeros(4, 2688),
        "mtp.layers.0.mixer.o_proj.weight": torch.zeros(2688, 2688),
        "mtp.layers.1.norm.weight": torch.zeros(2688),
        "mtp.layers.1.final_layernorm.weight": torch.zeros(2688),
        "mtp.layers.1.mixer.gate.weight": torch.zeros(4, 2688),
        "mtp.layers.1.mixer.fc1_latent_proj.weight": torch.zeros(latent_size, 2688),
        "mtp.layers.1.mixer.fc2_latent_proj.weight": torch.zeros(2688, latent_size),
        "mtp.layers.1.mixer.experts.0.up_proj.weight": torch.zeros(6, latent_size),
        "mtp.layers.1.mixer.experts.0.down_proj.weight": torch.zeros(latent_size, 6),
        "mtp.layers.1.mixer.shared_experts.up_proj.weight": torch.zeros(12, 2688),
        "mtp.layers.1.mixer.shared_experts.down_proj.weight": torch.zeros(2688, 12),
    }

    audit = spec.audit_state_dict(state)
    sliced = spec.slice_state_dict(state, 1344)

    assert set(state) - set(audit["handled"]) == {
        "backbone.layers.2.mixer.experts.0.up_proj.weight",
        "backbone.layers.2.mixer.experts.0.down_proj.weight",
        "model.layers.3.mixer.experts.gate_and_up_projs",
        "model.layers.3.mixer.experts.down_projs",
        "mtp.layers.1.mixer.experts.0.up_proj.weight",
        "mtp.layers.1.mixer.experts.0.down_proj.weight",
    }
    assert sliced["backbone.layers.0.mixer.in_proj.weight"].shape == (16, 1344)
    assert sliced["backbone.layers.0.mixer.out_proj.weight"].shape == (1344, 16)
    assert sliced["backbone.layers.2.mixer.fc1_latent_proj.weight"].shape == (
        latent_size,
        1344,
    )
    assert sliced["backbone.layers.2.mixer.fc2_latent_proj.weight"].shape == (
        1344,
        latent_size,
    )
    assert sliced["model.layers.3.mixer.experts.gate_and_up_projs"].shape == (
        4,
        latent_size,
        6,
    )
    assert sliced["model.layers.3.mixer.experts.down_projs"].shape == (
        4,
        6,
        latent_size,
    )
    assert sliced["mtp.layers.0.eh_proj.weight"].shape == (1344, 2 * 1344)
    assert sliced["mtp.layers.0.mixer.q_proj.weight"].shape == (2688, 1344)
    assert sliced["mtp.layers.0.mixer.o_proj.weight"].shape == (1344, 2688)
    assert sliced["mtp.layers.1.mixer.fc1_latent_proj.weight"].shape == (
        latent_size,
        1344,
    )
    assert sliced["mtp.layers.1.mixer.fc2_latent_proj.weight"].shape == (
        1344,
        latent_size,
    )
    assert spec.validate_width(1344, tp_size=2) == 1344


def test_nemotron_nano_hidden_width_spec_slices_non_latent_experts() -> None:
    config = SimpleNamespace(
        hidden_size=8,
        tie_word_embeddings=False,
        moe_latent_size=None,
    )
    spec = NemotronHModelDescriptor.embedding_pruning_spec(
        config, widths=(8, 4), alignment=1
    )
    state = {
        "backbone.layers.2.mixer.experts.0.up_proj.weight": torch.zeros(6, 8),
        "backbone.layers.2.mixer.experts.0.down_proj.weight": torch.zeros(8, 6),
        "model.layers.3.mixer.experts.gate_and_up_projs": torch.zeros(4, 8, 6),
        "model.layers.3.mixer.experts.down_projs": torch.zeros(4, 6, 8),
        "mtp.layers.1.mixer.experts.0.up_proj.weight": torch.zeros(6, 8),
        "mtp.layers.1.mixer.experts.0.down_proj.weight": torch.zeros(8, 6),
    }

    audit = spec.audit_state_dict(state)
    sliced = spec.slice_state_dict(state, 4)

    assert set(audit["handled"]) == set(state)
    assert sliced["backbone.layers.2.mixer.experts.0.up_proj.weight"].shape == (6, 4)
    assert sliced["backbone.layers.2.mixer.experts.0.down_proj.weight"].shape == (4, 6)
    assert sliced["model.layers.3.mixer.experts.gate_and_up_projs"].shape == (4, 4, 6)
    assert sliced["model.layers.3.mixer.experts.down_projs"].shape == (4, 6, 4)
    assert sliced["mtp.layers.1.mixer.experts.0.up_proj.weight"].shape == (6, 4)
    assert sliced["mtp.layers.1.mixer.experts.0.down_proj.weight"].shape == (4, 6)


def test_nemotron_capabilities_include_hidden_width_pipeline() -> None:
    capability = NemotronHModelDescriptor.puzzletron_capabilities(
        SimpleNamespace(hidden_size=2688)
    ).axes["hidden_width"]

    assert capability.score_hooks == ("minitron_hidden_width",)
    assert capability.sort_impl == "sorted_teacher.embedding"
    assert capability.materialize_impl == "materialize.hidden_width"
    assert capability.runtime_slice_impl == "runtime_hidden_width"
    assert capability.vllm_export is True


def test_nemotron_moe_variant_axes_cover_scoring_and_materialization() -> None:
    axes = NemotronHModelDescriptor.puzzletron_capabilities(
        SimpleNamespace(hidden_size=2688)
    ).axes

    assert axes["moe_experts"].runtime_slice_impl == "solution_recipe.moe_expert_reroute"
    assert axes["moe_top_k"].materialize_impl == "materialize.config_only_moe_top_k"


def test_nemotron_nano_capabilities_exclude_absent_latent_axis() -> None:
    nano_axes = NemotronHModelDescriptor.puzzletron_capabilities(
        SimpleNamespace(hidden_size=2688, moe_latent_size=None)
    ).axes
    super_axes = NemotronHModelDescriptor.puzzletron_capabilities(
        SimpleNamespace(hidden_size=2688, moe_latent_size=640)
    ).axes

    assert "moe_latent_dim" not in nano_axes
    assert "moe_latent_dim" in super_axes


def test_nemotron_equivalence_tolerance_accounts_for_bf16_hidden_basis_permutation() -> None:
    assert NemotronHModelDescriptor.checkpoint_equivalence_tolerances() == {
        "max_abs_lm_loss_delta": 5.0e-3,
        "max_kl_div": 1.0e-2,
        "min_top_1_logit_agreement": 0.9,
    }


def test_gpt_oss_pipeline_patch_restores_native_inner_forward() -> None:
    class Inner:
        def forward(self):
            return "native"

        def __call__(self):
            return self.forward()

    inner = Inner()
    inner.forward = lambda: "generic-pp"
    model_part = SimpleNamespace(model=inner)

    assert GptOssModelDescriptor.patch_pipeline_model_part(model_part) is True
    assert inner() == "native"


def test_gpt_oss_embedding_spec_preserves_mxfp4_input_channel_blocks() -> None:
    config = SimpleNamespace(hidden_size=64, tie_word_embeddings=False)
    spec = GptOssModelDescriptor.embedding_pruning_spec(
        config, widths=(64, 32), alignment=32
    )
    state = {
        "model.layers.0.mlp.experts.gate_up_proj_blocks": torch.zeros(2, 128, 2, 16),
        "model.layers.0.mlp.experts.gate_up_proj_scales": torch.zeros(2, 128, 2),
        "model.layers.0.mlp.experts.down_proj_blocks": torch.zeros(2, 64, 2, 16),
        "model.layers.0.mlp.experts.down_proj_scales": torch.zeros(2, 64, 2),
        "model.layers.0.mlp.experts.down_proj_bias": torch.zeros(2, 64),
    }

    audit = spec.audit_state_dict(state)
    order = spec.order_from_scores(torch.cat((torch.zeros(32), torch.ones(32))))
    permuted = spec.permute_state_dict(state, order)
    sliced = spec.slice_state_dict(permuted, 32)

    assert len(audit["handled"]) == len(state)
    assert torch.equal(order, torch.cat((torch.arange(32, 64), torch.arange(32))))
    assert sliced["model.layers.0.mlp.experts.gate_up_proj_blocks"].shape == (2, 128, 1, 16)
    assert sliced["model.layers.0.mlp.experts.down_proj_blocks"].shape == (2, 32, 2, 16)


def test_gpt_oss_embedding_spec_slices_post_bypass_fused_experts() -> None:
    config = SimpleNamespace(hidden_size=64, tie_word_embeddings=False)
    spec = GptOssModelDescriptor.embedding_pruning_spec(
        config, widths=(64, 32), alignment=32
    )
    state = {
        "model.layers.0.mlp.experts.gate_up_proj": torch.zeros(2, 64, 128),
        "model.layers.0.mlp.experts.down_proj": torch.zeros(2, 128, 64),
    }

    audit = spec.audit_state_dict(state)
    sliced = spec.slice_state_dict(state, 32)

    assert len(audit["handled"]) == 2
    assert sliced["model.layers.0.mlp.experts.gate_up_proj"].shape == (2, 32, 128)
    assert sliced["model.layers.0.mlp.experts.down_proj"].shape == (2, 128, 32)


def test_stage_runtime_receives_inferred_descriptor_without_persisting_override(
    monkeypatch, tmp_path
) -> None:
    resolution = resolve_descriptor(
        SimpleNamespace(model_type="llama", architectures=["LlamaForCausalLM"])
    )
    monkeypatch.setattr(
        "modelopt.torch.puzzletron.stage_runner._resolve_capabilities",
        lambda _config: resolution,
    )
    observed = {}

    def handler(config, manifest):
        observed["runtime_descriptor"] = config["_runtime"]["descriptor"]
        observed["manifest_config"] = manifest.config
        observed["resolution"] = manifest.inputs["descriptor_resolution"]
        return StageResult(
            stage="width_importance",
            status="success",
            manifest_path=tmp_path / "width_importance.json",
            message="ok",
        )

    input_config = {
        "experiment": {"dir": str(tmp_path)},
        "model": {"source": "/checkpoint", "force_hf": False},
        "bypass": {"backend": "automodel"},
        "search_space": {"axes": {}},
    }

    run_stage(
        input_config,
        "width_importance",
        handlers={"width_importance": handler},
    )

    assert observed["runtime_descriptor"] == "llama"
    assert "_runtime" not in observed["manifest_config"]
    assert "descriptor_override" not in observed["manifest_config"]["model"]
    assert observed["resolution"]["name"] == "llama"


def test_gpt_oss_declares_attention_sink_as_query_head_state() -> None:
    assert GptOssModelDescriptor.sorted_teacher_layout_kwargs(SimpleNamespace()) == {
        "attention_q_head_subnames": ("sinks",),
        "moe_router_subname": "router",
        "moe_router_aux_subnames": ("router.bias",),
        "moe_fused_expert_subnames": (
            "experts.gate_up_proj_blocks",
            "experts.gate_up_proj_scales",
            "experts.gate_up_proj_bias",
            "experts.down_proj_blocks",
            "experts.down_proj_scales",
            "experts.down_proj_bias",
        ),
        "moe_fused_gate_up_subnames": (
            "experts.gate_up_proj_blocks",
            "experts.gate_up_proj_scales",
            "experts.gate_up_proj_bias",
        ),
        "moe_fused_down_subnames": (
            "experts.down_proj_blocks",
            "experts.down_proj_scales",
        ),
        "moe_expert_intermediate_group_size": 32,
        "moe_expert_order_mode": "metadata_only",
        "moe_fused_gate_layout": "interleaved",
    }


def test_generic_window_capability_is_discovered_for_non_gpt_attention() -> None:
    config = _text_config(sliding_window=512)

    axis = LlamaModelDescriptor.puzzletron_capabilities(config).axes[
        "sliding_window_size"
    ]

    assert axis.variant_only
    assert axis.vllm_export
    assert axis.values == (256, 512, "full")


def test_stage_resolution_uses_converted_teacher_for_dynamic_capabilities(
    tmp_path, monkeypatch
) -> None:
    teacher = tmp_path / "teacher"
    teacher.mkdir()
    sentinel = object()
    calls = []
    monkeypatch.setattr(
        "modelopt.torch.puzzletron.stage_runner.resolve_descriptor_from_pretrained",
        lambda path, **kwargs: calls.append(path) or sentinel,
    )

    resolution = _resolve_capabilities(
        {
            "teacher_dir": str(teacher),
            "model": {
                "source": "org/remote-model",
                "descriptor_override": "llama",
                "trust_remote_code": True,
            },
        }
    )

    assert resolution is sentinel
    assert calls == [str(teacher)]


def test_stage_preflight_requires_vllm_control_for_runtime_stats(monkeypatch) -> None:
    captured = {}
    monkeypatch.setattr(
        "modelopt.torch.puzzletron.stage_runner.validate_capabilities",
        lambda capabilities, **kwargs: captured.update(kwargs),
    )

    _preflight(
        {
            "model": {"force_hf": False},
            "parallel": {"ep": 1},
            "search_space": {
                "axes": {"mla_q_lora_rank": {"enabled": True, "values": [384]}}
            },
            "calc_subblock_stats": {
                "runtime_stats": {"enabled": True, "backend": "vllm"}
            },
        },
        SimpleNamespace(capabilities=object()),
        stage="library",
    )

    assert captured["require_vllm"] is True


def test_complete_pipeline_preflight_names_every_missing_axis_consumer() -> None:
    base = default_capabilities(descriptor_name="fixture")
    broken = AxisCapabilities(
        axis_id="broken_width",
        subblock_kind="ffn",
        field="intermediate_size",
    )
    capabilities = dataclasses.replace(base, axes={"broken_width": broken})

    with pytest.raises(CapabilityValidationError) as error:
        validate_capabilities(
            capabilities,
            enabled_axes=("broken_width",),
            force_hf=False,
            require_vllm=True,
            require_complete_pipeline=True,
        )

    message = str(error.value)
    assert "activation scorer" in message
    assert "sort_impl" in message
    assert "materialize_impl" in message
    assert "runtime_slice_impl" in message
    assert "vLLM export" in message


def test_variant_axis_is_complete_without_activation_or_sorting() -> None:
    base = default_capabilities(descriptor_name="fixture")
    variant = AxisCapabilities(
        axis_id="window",
        subblock_kind="attention",
        field="sliding_window_size",
        sortable=False,
        variant_only=True,
        materialize_impl="block_config.variant",
        runtime_slice_impl="block_config.variant",
        vllm_export=True,
    )
    capabilities = dataclasses.replace(base, axes={"window": variant})

    validate_capabilities(
        capabilities,
        enabled_axes=("window",),
        force_hf=False,
        require_vllm=True,
        require_complete_pipeline=True,
    )


def test_qwen_dense_smoke_axes_have_complete_pipeline_consumers() -> None:
    config = _text_config(
        model_type="qwen3_5_text",
        linear_num_key_heads=16,
        linear_num_value_heads=16,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
    )
    capabilities = Qwen3P5TextModelDescriptor.puzzletron_capabilities(config)

    validate_capabilities(
        capabilities,
        enabled_axes=(
            "hidden_width",
            "ffn_intermediate",
            "kv_groups",
            "q_heads_per_group",
            "gdn_key_groups",
            "gdn_key_head_dim",
            "gdn_value_head_dim",
        ),
        force_hf=False,
        require_vllm=True,
        require_complete_pipeline=True,
    )


def test_nemotron_updates_remote_read_only_layer_types_through_hybrid_pattern() -> None:
    class RemoteNemotronConfig:
        def __init__(self):
            self.num_hidden_layers = 2
            self.hybrid_override_pattern = "M-"

        @property
        def layers_block_type(self):
            mapping = {"M": "mamba", "-": "mlp", "*": "attention", "E": "moe"}
            return [mapping[value] for value in self.hybrid_override_pattern]

    config = RemoteNemotronConfig()
    blocks = [
        BlockConfig(subblock_configs=(MambaConfig(num_heads=2, head_dim=8),)),
        BlockConfig(subblock_configs=(FFNConfig(intermediate_size=16),)),
    ]

    NemotronHModelDescriptor.set_block_configs(config, blocks)

    assert config.hybrid_override_pattern == "M-"
    assert config.layers_block_type == ["mamba", "mlp"]


def test_nemotron_runtime_proxy_supplies_legacy_hybrid_pattern(monkeypatch) -> None:
    captured = {}

    class _CapturedConfig(Exception):
        def __init__(self, hybrid_override_pattern=None, **kwargs):
            captured.update(kwargs)
            captured["hybrid_override_pattern"] = hybrid_override_pattern
            raise self

    runtime = SimpleNamespace(
        vocab_size=32,
        hidden_size=16,
        num_attention_heads=4,
        num_key_value_heads=2,
        prefill_seq_len=8,
        generation_seq_len=2,
        model_config_value=lambda _name, default=None: default,
    )
    blocks = [
        BlockConfig(
            subblock_configs=(
                AttentionConfig(num_query_heads=4, num_kv_heads=2),
                FFNConfig(no_op=True),
            )
        )
    ]
    monkeypatch.setattr(
        NemotronHModelDescriptor, "_runtime_config_cls", lambda: _CapturedConfig
    )
    monkeypatch.setattr(
        NemotronHModelDescriptor, "_runtime_model_cls", lambda: object
    )

    with pytest.raises(_CapturedConfig):
        NemotronHModelDescriptor.create_runtime_benchmark_model(runtime, blocks)

    assert "layers_block_type" not in captured
    assert captured["hybrid_override_pattern"] == "*"
    assert captured["num_hidden_layers"] == 1


def test_nemotron_runtime_proxy_uses_native_layer_types_for_new_config() -> None:
    class _NewConfig:
        def __init__(self, layers_block_type=None, **kwargs):
            pass

    assert NemotronHModelDescriptor._runtime_layer_config_kwargs(
        _NewConfig, ["attention", "moe"]
    ) == {"layers_block_type": ["attention", "moe"]}


def test_nemotron_runtime_proxy_uses_active_mamba_shape_for_global_initialization(
    monkeypatch,
) -> None:
    captured = {}

    class _CapturedConfig(Exception):
        def __init__(self, **kwargs):
            captured.update(kwargs)
            raise self

    values = {"mamba_num_heads": 64, "mamba_head_dim": 64, "ssm_state_size": 128}
    runtime = SimpleNamespace(
        vocab_size=32,
        hidden_size=16,
        num_attention_heads=4,
        num_key_value_heads=2,
        prefill_seq_len=8,
        generation_seq_len=2,
        model_config_value=lambda name, default=None: values.get(name, default),
    )
    blocks = [
        BlockConfig(subblock_configs=(AttentionConfig(num_query_heads=4, num_kv_heads=2),)),
        BlockConfig(
            subblock_configs=(MambaConfig(num_heads=48, head_dim=56, state_dim=96),)
        ),
    ]
    monkeypatch.setattr(NemotronHModelDescriptor, "_runtime_config_cls", lambda: _CapturedConfig)
    monkeypatch.setattr(NemotronHModelDescriptor, "_runtime_model_cls", lambda: object)

    with pytest.raises(_CapturedConfig):
        NemotronHModelDescriptor.create_runtime_benchmark_model(runtime, blocks)

    assert captured["mamba_num_heads"] == 48
    assert captured["mamba_head_dim"] == 56
    assert captured["ssm_state_size"] == 96


def test_llama_requests_scaffold_only_for_cacheless_candidate() -> None:
    attention = BlockConfig(
        subblock_configs=(AttentionConfig(no_op=False), FFNConfig(no_op=True))
    )
    ffn = BlockConfig(
        subblock_configs=(AttentionConfig(no_op=True), FFNConfig(no_op=False))
    )

    assert LlamaModelDescriptor.runtime_benchmark_scaffold_policy(attention) == "none"
    assert (
        LlamaModelDescriptor.runtime_benchmark_scaffold_policy(ffn)
        == "attention_scaffold_per_pp_stage"
    )


def test_nemotron_requests_attention_scaffold_for_cacheless_candidates() -> None:
    attention = BlockConfig(
        subblock_configs=(AttentionConfig(no_op=False, num_query_heads=32, num_kv_heads=2),)
    )
    mamba = BlockConfig(
        subblock_configs=(MambaConfig(no_op=False, num_heads=48, head_dim=64, state_dim=128),)
    )
    moe = BlockConfig(
        subblock_configs=(
            MoEConfig(no_op=False, num_experts=16, expert_intermediate_size=512),
        )
    )

    assert NemotronHModelDescriptor.runtime_benchmark_scaffold_policy(attention) == "none"
    assert (
        NemotronHModelDescriptor.runtime_benchmark_scaffold_policy(mamba)
        == "attention_scaffold_per_pp_stage"
    )
    assert (
        NemotronHModelDescriptor.runtime_benchmark_scaffold_policy(moe)
        == "attention_scaffold_per_pp_stage"
    )


def test_nemotron_runtime_proxy_normalizes_legacy_tied_weight_lists() -> None:
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 2))
    model._tied_weights_keys = ["lm_head.weight"]
    model[0]._tied_weights_keys = ("weight",)

    NemotronHModelDescriptor._normalize_runtime_tied_weight_metadata(model)

    assert model._tied_weights_keys == {"lm_head.weight": "lm_head.weight"}
    assert model[0]._tied_weights_keys == {"weight": "weight"}


def test_nemotron_runtime_bounding_preserves_native_moe_and_mamba_families() -> None:
    values = {
        "runtime_proxy_enabled": True,
        "n_routed_experts": 512,
        "moe_intermediate_size": 2688,
        "moe_shared_expert_intermediate_size": 5376,
        "moe_latent_size": 1024,
        "num_experts_per_tok": 22,
        "runtime_proxy_max_experts": 16,
        "runtime_proxy_max_expert_intermediate": 512,
        "runtime_proxy_max_shared_expert_intermediate": 512,
        "runtime_proxy_max_latent": 256,
        "runtime_proxy_max_top_k": 4,
    }
    runtime = SimpleNamespace(
        num_key_value_heads=8,
        model_config_value=lambda name, default=None: values.get(name, default),
    )
    moe_block = BlockConfig(
        subblock_configs=(
            AttentionConfig(no_op=True),
            MoEConfig(
                num_experts=256,
                expert_intermediate_size=1344,
                shared_expert_intermediate_size=2688,
                top_k=11,
                latent_dim=512,
            ),
        )
    )
    mamba_block = BlockConfig(
        subblock_configs=(MambaConfig(num_heads=64, head_dim=32), FFNConfig(no_op=True))
    )

    bounded_moe = NemotronHModelDescriptor._runtime_proxy_block_config(runtime, moe_block)
    bounded_mamba = NemotronHModelDescriptor._runtime_proxy_block_config(runtime, mamba_block)

    moe = bounded_moe.require_subblock("moe")
    assert (moe.num_experts, moe.expert_intermediate_size, moe.latent_dim, moe.top_k) == (
        8,
        256,
        128,
        2,
    )
    assert bounded_moe.require_subblock("attention").no_op is True
    assert bounded_mamba.require_subblock("mamba") == mamba_block.require_subblock("mamba")


def test_nemotron_runtime_uses_exact_moe_dimensions_by_default() -> None:
    values = {
        "n_routed_experts": 128,
        "moe_intermediate_size": 1856,
        "moe_shared_expert_intermediate_size": 3712,
        "moe_latent_size": None,
        "num_experts_per_tok": 6,
        "runtime_proxy_max_experts": 16,
        "runtime_proxy_max_expert_intermediate": 512,
        "runtime_proxy_max_shared_expert_intermediate": 512,
        "runtime_proxy_max_latent": 256,
        "runtime_proxy_max_top_k": 4,
    }
    runtime = SimpleNamespace(
        model_config_value=lambda name, default=None: values.get(name, default),
    )
    block = BlockConfig(
        subblock_configs=(
            MoEConfig(
                num_experts=96,
                expert_intermediate_size=1344,
                shared_expert_intermediate_size=2560,
                top_k=4,
                latent_dim=None,
            ),
        )
    )

    assert NemotronHModelDescriptor._runtime_proxy_block_config(runtime, block) == block
