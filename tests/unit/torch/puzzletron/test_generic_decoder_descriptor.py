from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from modelopt.torch.puzzletron.anymodel.models.generic_decoder import (
    DecoderLayout,
    GatedDenseFFNContract,
    GenericDecoderContract,
    MTPContract,
    RoutedMoEContract,
    StandardGQAAttentionContract,
    VisionLanguageContract,
)
from modelopt.torch.puzzletron.anymodel.registry import infer_descriptor_name
from modelopt.torch.puzzletron.pruning.embedding_pruning import TensorAxisRule


def _dense_contract(*, vlm: bool = False) -> GenericDecoderContract:
    prefix = "model.language_model" if vlm else "model"
    config_path = ("text_config",) if vlm else ()
    return GenericDecoderContract(
        descriptor_name="fixture_dense_vlm" if vlm else "fixture_dense",
        model_family="fixture",
        layout=DecoderLayout(
            language_config_path=config_path,
            language_prefix=prefix,
            layer_template=f"{prefix}.layers.{{layer_idx}}",
            input_embedding=f"{prefix}.embed_tokens",
            output_embedding="lm_head",
            final_norm=f"{prefix}.norm",
            layer_norm_names=("input_layernorm", "post_attention_layernorm"),
        ),
        attention=StandardGQAAttentionContract(),
        dense_ffn=GatedDenseFFNContract(),
        vision=(
            VisionLanguageContract(
                module_names=("model.visual",),
                projector_output_config_paths=(("vision_config", "out_hidden_size"),),
                projector_rules=(
                    TensorAxisRule(
                        r"^model\.visual\.projector\.(?:weight|bias)$",
                        (0,),
                        "projector language-width output",
                    ),
                ),
            )
            if vlm
            else None
        ),
        mtp=MTPContract(
            tensor_rules=(
                TensorAxisRule(
                    r"^mtp\.fc\.weight$",
                    (0,),
                    "MTP residual fusion",
                    chunked_axes=((1, 2),),
                ),
            ),
        ),
    )


def _moe_contract(*, vlm: bool = False) -> GenericDecoderContract:
    dense = _dense_contract(vlm=vlm)
    return GenericDecoderContract(
        descriptor_name="fixture_moe_vlm" if vlm else "fixture_moe",
        model_family="fixture_moe",
        layout=dense.layout,
        attention=dense.attention,
        routed_moe=RoutedMoEContract(
            module_name="mlp",
            experts_name="experts",
            router_name="gate",
            shared_expert_name="shared_expert",
        ),
        vision=dense.vision,
        mtp=dense.mtp,
        native_automodel_supported=True,
        ep_supported=True,
    )


def _config(*, vlm: bool = False, moe: bool = False):
    text = SimpleNamespace(
        hidden_size=8,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=2,
        intermediate_size=16,
        num_experts=8,
        moe_intermediate_size=12,
        shared_expert_intermediate_size=16,
        tie_word_embeddings=True,
    )
    return (
        SimpleNamespace(
            text_config=text,
            hidden_size=8,
            vision_config=SimpleNamespace(out_hidden_size=8),
        )
        if vlm
        else text
    )


def _dense_state(*, vlm: bool = False) -> dict[str, torch.Tensor]:
    prefix = "model.language_model" if vlm else "model"
    state = {
        f"{prefix}.embed_tokens.weight": torch.randn(32, 8),
        "lm_head.weight": torch.randn(32, 8),
        f"{prefix}.norm.weight": torch.randn(8),
        f"{prefix}.layers.0.input_layernorm.weight": torch.randn(8),
        f"{prefix}.layers.0.post_attention_layernorm.weight": torch.randn(8),
        f"{prefix}.layers.0.self_attn.q_proj.weight": torch.randn(8, 8),
        f"{prefix}.layers.0.self_attn.k_proj.weight": torch.randn(4, 8),
        f"{prefix}.layers.0.self_attn.v_proj.weight": torch.randn(4, 8),
        f"{prefix}.layers.0.self_attn.o_proj.weight": torch.randn(8, 8),
        f"{prefix}.layers.0.mlp.gate_proj.weight": torch.randn(16, 8),
        f"{prefix}.layers.0.mlp.up_proj.weight": torch.randn(16, 8),
        f"{prefix}.layers.0.mlp.down_proj.weight": torch.randn(8, 16),
        "mtp.fc.weight": torch.randn(8, 16),
    }
    if vlm:
        state["model.visual.projector.weight"] = torch.randn(8, 6)
        state["model.visual.encoder.weight"] = torch.randn(8, 8)
    return state


def _moe_state(*, vlm: bool = False) -> dict[str, torch.Tensor]:
    state = _dense_state(vlm=vlm)
    prefix = "model.language_model" if vlm else "model"
    for key in list(state):
        if f"{prefix}.layers.0.mlp." in key:
            del state[key]
    state.update(
        {
            f"{prefix}.layers.0.mlp.gate.weight": torch.randn(8, 8),
            f"{prefix}.layers.0.mlp.experts.0.gate_proj.weight": torch.randn(12, 8),
            f"{prefix}.layers.0.mlp.experts.0.up_proj.weight": torch.randn(12, 8),
            f"{prefix}.layers.0.mlp.experts.0.down_proj.weight": torch.randn(8, 12),
            f"{prefix}.layers.0.mlp.shared_expert.gate_proj.weight": torch.randn(16, 8),
            f"{prefix}.layers.0.mlp.shared_expert.up_proj.weight": torch.randn(16, 8),
            f"{prefix}.layers.0.mlp.shared_expert.down_proj.weight": torch.randn(8, 16),
        }
    )
    return state


def test_dense_text_and_vlm_contracts_cover_hidden_sensitive_tensors() -> None:
    for vlm in (False, True):
        contract = _dense_contract(vlm=vlm)
        spec = contract.embedding_pruning_spec(_config(vlm=vlm), widths=(8, 4), alignment=2)

        audit = spec.audit_state_dict(_dense_state(vlm=vlm))

        assert audit["handled"]
        if vlm:
            assert "model.visual.encoder.weight" in audit["exempt"]
            child = spec.update_config_object(_config(vlm=True), 4)
            assert child.text_config.hidden_size == 4
            assert child.vision_config.out_hidden_size == 4


def test_moe_text_and_vlm_contracts_cover_router_experts_and_shared_expert() -> None:
    for vlm in (False, True):
        contract = _moe_contract(vlm=vlm)
        spec = contract.embedding_pruning_spec(_config(vlm=vlm, moe=True), widths=(8, 4), alignment=2)

        audit = spec.audit_state_dict(_moe_state(vlm=vlm))

        assert any("experts.0" in key for key in audit["handled"])
        assert any("shared_expert" in key for key in audit["handled"])
        assert any(key.endswith("gate.weight") for key in audit["handled"])


def test_contract_derives_one_aligned_reduced_value_per_supported_axis() -> None:
    dense = _dense_contract().reduced_axis_values(_config(), alignment=2)
    moe = _moe_contract().reduced_axis_values(_config(moe=True), alignment=2)

    assert dense == {
        "hidden_width": 4,
        "ffn_intermediate": 8,
        "kv_groups": 1,
        "q_heads_per_group": 1,
    }
    assert moe == {
        "hidden_width": 4,
        "kv_groups": 1,
        "q_heads_per_group": 1,
        "moe_experts": 4,
        "moe_expert_intermediate": 6,
        "moe_shared_expert_intermediate": 8,
    }


class _Attention(nn.Module):
    pass


class _MLP(nn.Module):
    pass


class _Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _Attention()
        self.mlp = _MLP()


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([_Layer(), _Layer()])


def test_structural_discovery_returns_only_declared_layer_submodules() -> None:
    targets = _dense_contract().discover_prunable_modules(_Model())

    assert set(targets) == {
        "model.layers.0.attention",
        "model.layers.0.ffn",
        "model.layers.1.attention",
        "model.layers.1.ffn",
    }
    assert isinstance(targets["model.layers.0.attention"], _Attention)


def test_contract_capabilities_are_component_owned() -> None:
    dense = _dense_contract().capabilities()
    moe = _moe_contract().capabilities()

    assert set(dense.axes) == {
        "hidden_width",
        "ffn_intermediate",
        "kv_groups",
        "q_heads_per_group",
    }
    assert set(moe.axes) == {
        "hidden_width",
        "kv_groups",
        "q_heads_per_group",
        "moe_experts",
        "moe_expert_intermediate",
        "moe_shared_expert_intermediate",
        "moe_top_k",
    }
    assert not dense.parallelism.ep
    assert moe.parallelism.ep


def test_registry_accepts_preflight_alias_for_unknown_architecture() -> None:
    config = SimpleNamespace(model_type="brand_new", architectures=["BrandNewForCausalLM"])

    name, reason = infer_descriptor_name(
        config,
        descriptor_aliases={"BrandNewForCausalLM": "generic_fixture"},
    )

    assert name == "generic_fixture"
    assert "preflight alias" in reason


def test_registry_exact_mapping_wins_over_preflight_alias() -> None:
    config = SimpleNamespace(model_type="llama", architectures=["LlamaForCausalLM"])

    name, reason = infer_descriptor_name(
        config,
        descriptor_aliases={"LlamaForCausalLM": "wrong"},
    )

    assert name == "llama"
    assert "architectures" in reason
